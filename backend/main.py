"""
Main FastAPI application for the search taxonomy ML backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .model_service import TaxonomyModelService
from .store import DatabaseStore, GlobalLearningStore, UserEventStore
from .taxonomy_data import (
    CATEGORY_LIST,
    classify_intent_probabilities,
    expand_query_to_tags,
    lexical_category_probabilities,
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
DB_FILE = ROOT / "runtime" / "events.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_FILE}")
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET")

COMMERCE_ORIENTED_CATEGORIES = {
    "/Shopping",
    "/Finance",
    "/Real Estate",
    "/Autos & Vehicles",
    "/Travel",
    "/Business & Industrial",
}


def _env_float(name: str, default: float) -> float:
    """
    Parse an environment variable as float.
    :param name: Environment variable name.
    :param default: Fallback default value.
    :return: Parsed float or default.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


TRANSACTIONAL_INTENT_THRESHOLD = min(
    1.0,
    max(0.0, _env_float("TRANSACTIONAL_INTENT_THRESHOLD", 0.30)),
)

model_service = TaxonomyModelService(artifact_dir=ARTIFACT_DIR)
database_store = DatabaseStore(database_url=DATABASE_URL, db_path=DB_FILE)
user_store = UserEventStore(db=database_store)
global_store = GlobalLearningStore(db=database_store)
auth_scheme = HTTPBearer(auto_error=False)

app = FastAPI(title="Search Taxonomy ML Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    """
    Request model for search queries.
    """
    query: str = Field(min_length=2, max_length=500)


class FeedbackRequest(BaseModel):
    """
    Request model for feedback on search queries.
    """
    query: str = Field(min_length=2, max_length=500)
    category: str
    confidence: float = Field(default=1.0, ge=0.1, le=2.0)


class ConversionClickRequest(BaseModel):
    """
    Request model for simulated conversion clicks from tag interactions.
    """
    tag: str = Field(min_length=1, max_length=200)
    category: str
    intensity: float = Field(default=1.0, ge=0.1, le=3.0)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> str:
    """
    Resolve stable user_id from JWT claims or X-User-Id header.
    """
    if credentials is not None:
        if credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unsupported authorization scheme",
            )

        token = credentials.credentials
        try:
            if AUTH_JWT_SECRET:
                claims = jwt.decode(token, AUTH_JWT_SECRET, algorithms=["HS256"])
            else:
                claims = jwt.decode(token, options={"verify_signature": False})
        except jwt.InvalidTokenError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid auth token",
            ) from error

        for key in ("sub", "oid", "user_id"):
            value = claims.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if x_user_id and x_user_id.strip():
        return x_user_id.strip()

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing user identity. Provide Bearer token or X-User-Id header.",
    )


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint.
    :return: A simple status message indicating the service is running.
    """
    return {"status": "ok", "model": "sgd-log-loss-hashing+user-embedding"}


@app.get("/taxonomy")
def taxonomy() -> Dict[str, List[str]]:
    """
    Retrieve the list of available categories in the taxonomy.
    :return: A dictionary containing the list of categories.
    """
    return {
        "categories": CATEGORY_LIST,
    }


@app.get("/history")
def history(
    limit: int = Query(default=20, ge=1, le=200),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, object]:
    """
    Retrieve the recent search history.
    :param limit: The maximum number of recent searches to return.
    :return: A dictionary containing the recent search items.
    """
    return {
        "items": user_store.get_recent_searches(user_id=user_id, limit=limit),
    }


@app.get("/embedding")
def embedding(
    limit: int = Query(default=6, ge=1, le=20),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, object]:
    """
    Inspect the persisted user embedding and top similarity categories.
    :param limit: Number of top categories to return.
    :return: Embedding diagnostics for drift monitoring.
    """
    vector = user_store.get_user_embedding(
        user_id=user_id,
        dimensions=model_service.embedding_dimensions,
    )
    probabilities = model_service.category_probabilities_from_user_embedding(
        vector)
    ranked = sorted(probabilities.items(),
                    key=lambda item: item[1], reverse=True)
    norm = sum(value * value for value in vector) ** 0.5

    return {
        "dimensions": model_service.embedding_dimensions,
        "vector_norm": float(norm),
        "top_categories": [
            {"category": category, "probability": float(probability)}
            for category, probability in ranked[:limit]
        ],
    }


@app.post("/history/clear")
def clear_history(user_id: str = Depends(get_current_user_id)) -> Dict[str, str]:
    """
    Clear all search and feedback history from the database.
    :return: A status message indicating the history has been cleared.
    """
    user_store.clear_all(user_id=user_id)
    return {"status": "cleared"}


@app.post("/conversion/click")
def conversion_click(
    payload: ConversionClickRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, object]:
    """
    Track a simulated conversion click event from cloud-tag interaction.
    :param payload: The click payload with tag, category, and intensity.
    :return: Status and updated conversion affinity snapshot for the category.
    """
    if payload.category not in CATEGORY_LIST:
        raise HTTPException(status_code=400, detail="Unknown category")

    user_store.increment_conversion_affinity(
        user_id=user_id,
        category=payload.category,
        amount=payload.intensity,
    )
    global_store.increment_conversion_affinity(payload.category, payload.intensity)
    affinities = user_store.get_conversion_affinity(user_id=user_id, categories=CATEGORY_LIST)
    return {
        "status": "tracked",
        "tag": payload.tag,
        "category": payload.category,
        "conversion_affinity": affinities.get(payload.category, 0.0),
    }


@app.post("/search")
def search(
    payload: SearchRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, object]:
    """
    Perform a search query and return the predicted category along with probabilities.
    :param payload: The search request payload containing the query.
    :return: A dictionary containing the search results and related information.
    """
    model_probabilities = model_service.predict_proba(payload.query)
    lexical_probabilities = lexical_category_probabilities(payload.query)
    intent_probabilities = classify_intent_probabilities(payload.query)
    intent_top = max(intent_probabilities.items(), key=lambda item: item[1])[0]
    transactional_intent_score = intent_probabilities.get("transactional", 0.0)
    user_embedding = user_store.get_user_embedding(
        user_id=user_id,
        dimensions=model_service.embedding_dimensions,
    )
    updated_user_embedding = model_service.update_user_embedding(
        user_embedding,
        payload.query,
        decay=0.95,
        learning_rate=0.05,
    )
    user_store.set_user_embedding(user_id=user_id, vector=updated_user_embedding)
    embedding_probabilities = model_service.category_probabilities_from_user_embedding(
        updated_user_embedding)

    lexical_peak = max(lexical_probabilities.values())
    lexical_weight = min(0.68, 0.34 + lexical_peak * 0.65)
    model_weight = 1.0 - lexical_weight

    blended = {
        category: model_probabilities[category] * model_weight +
        lexical_probabilities[category] * lexical_weight
        for category in CATEGORY_LIST
    }
    embedding_peak = max(embedding_probabilities.values())
    embedding_weight = min(0.60, 0.26 + embedding_peak * 0.45)
    baseline_weight = 1.0 - embedding_weight
    fused = {
        category: blended[category] * baseline_weight +
        embedding_probabilities[category] * embedding_weight
        for category in CATEGORY_LIST
    }

    if transactional_intent_score >= TRANSACTIONAL_INTENT_THRESHOLD:
        intent_boost = 1.0 + min(0.45, transactional_intent_score * 0.50)
        for category in COMMERCE_ORIENTED_CATEGORIES:
            if category in fused:
                fused[category] *= intent_boost

    total = sum(fused.values()) or 1.0
    click_probabilities = {category: value /
                           total for category, value in fused.items()}

    user_conversion_affinity = user_store.get_conversion_affinity(
        user_id=user_id,
        categories=CATEGORY_LIST,
    )
    global_conversion_affinity = global_store.get_conversion_affinity(CATEGORY_LIST)
    smoothed_conversion = {
        category: user_conversion_affinity.get(category, 0.0) + global_conversion_affinity.get(category, 0.0) + 1.0
        for category in CATEGORY_LIST
    }
    conversion_total = sum(smoothed_conversion.values()) or 1.0
    conversion_probabilities = {
        category: value / conversion_total
        for category, value in smoothed_conversion.items()
    }

    reranked = {
        category: click_probabilities[category] *
        conversion_probabilities[category]
        for category in CATEGORY_LIST
    }
    reranked_total = sum(reranked.values()) or 1.0
    probabilities = {
        category: value / reranked_total
        for category, value in reranked.items()
    }

    predicted_category = max(probabilities.items(),
                             key=lambda item: item[1])[0]
    predicted_score = probabilities[predicted_category]

    tag_updates = expand_query_to_tags(payload.query, probabilities, top_k=40)
    user_store.record_search(
        user_id=user_id,
        query=payload.query,
        predicted_category=predicted_category,
        probabilities=probabilities,
        intent_probabilities=intent_probabilities,
        tag_updates=tag_updates,
    )

    model_top = max(model_probabilities.items(), key=lambda item: item[1])[0]
    lexical_top = max(lexical_probabilities.items(),
                      key=lambda item: item[1])[0]
    embedding_top = max(embedding_probabilities.items(),
                        key=lambda item: item[1])[0]

    online_sample_weight = 0.35
    if transactional_intent_score >= TRANSACTIONAL_INTENT_THRESHOLD:
        online_sample_weight = min(
            0.9, online_sample_weight + 0.25 + transactional_intent_score * 0.2)

    if predicted_score >= 0.58 and (model_top == lexical_top or lexical_peak < 0.28 or embedding_top == predicted_category):
        model_service.online_update(
            payload.query,
            predicted_category,
            sample_weight=online_sample_weight,
        )

    top_segments = user_store.get_top_segments(user_id=user_id, limit=6)
    cloud_words = user_store.get_cloud_words(user_id=user_id, limit=180)

    return {
        "query": payload.query,
        "predicted_category": predicted_category,
        "probabilities": probabilities,
        "top_segments": top_segments,
        "cloud_words": cloud_words,
        "model_source": "python-backend",
        "model_top": model_top,
        "lexical_top": lexical_top,
        "embedding_top": embedding_top,
        "embedding_weight": embedding_weight,
        "intent_top": intent_top,
        "intent_probabilities": intent_probabilities,
        "online_sample_weight": online_sample_weight,
    }


@app.post("/feedback")
def feedback(
    payload: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, object]:
    """
    Submit feedback for a search query.
    :param payload: The feedback request payload containing the query, category, and confidence.
    :return: A dictionary containing the status and the updated query information.
    """
    if payload.category not in CATEGORY_LIST:
        raise HTTPException(status_code=400, detail="Unknown category")

    user_store.record_feedback(
        user_id=user_id,
        query=payload.query,
        true_category=payload.category,
        confidence=payload.confidence,
    )
    global_store.record_feedback(
        payload.query,
        payload.category,
        payload.confidence,
    )

    feedback_intents = classify_intent_probabilities(payload.query)
    if (
        payload.confidence >= 1.0
        and feedback_intents.get("transactional", 0.0) >= TRANSACTIONAL_INTENT_THRESHOLD
    ):
        increment = min(1.5, payload.confidence * 0.55)
        user_store.increment_conversion_affinity(
            user_id=user_id,
            category=payload.category,
            amount=increment,
        )
        global_store.increment_conversion_affinity(
            payload.category,
            amount=increment,
        )

    model_service.online_update(
        payload.query, payload.category, sample_weight=payload.confidence)

    return {
        "status": "updated",
        "query": payload.query,
        "category": payload.category,
    }


@app.post("/retrain")
def retrain() -> Dict[str, object]:
    """
    Retrain the model using the initial training data and feedback examples.
    :return: A dictionary containing the status and the number of categories.
    """
    model_service.train_initial_model()
    for query, category, confidence in global_store.get_feedback_examples():
        model_service.online_update(query, category, sample_weight=confidence)

    return {
        "status": "retrained",
        "categories": len(CATEGORY_LIST),
    }
