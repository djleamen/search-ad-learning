"""
Main FastAPI application for the search taxonomy ML backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model_service import TaxonomyModelService
from .store import EventStore
from .taxonomy_data import CATEGORY_LIST, expand_query_to_tags, lexical_category_probabilities

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
DB_FILE = ROOT / "runtime" / "events.db"

model_service = TaxonomyModelService(artifact_dir=ARTIFACT_DIR)
event_store = EventStore(db_path=DB_FILE)

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


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint.
    :return: A simple status message indicating the service is running.
    """
    return {"status": "ok", "model": "sgd-log-loss-hashing"}


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
def history(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, object]:
    """
    Retrieve the recent search history.
    :param limit: The maximum number of recent searches to return.
    :return: A dictionary containing the recent search items.
    """
    return {
        "items": event_store.get_recent_searches(limit=limit),
    }


@app.post("/history/clear")
def clear_history() -> Dict[str, str]:
    """
    Clear all search and feedback history from the database.
    :return: A status message indicating the history has been cleared.
    """
    event_store.clear_all()
    return {"status": "cleared"}


@app.post("/search")
def search(payload: SearchRequest) -> Dict[str, object]:
    """
    Perform a search query and return the predicted category along with probabilities.
    :param payload: The search request payload containing the query.
    :return: A dictionary containing the search results and related information.
    """
    model_probabilities = model_service.predict_proba(payload.query)
    lexical_probabilities = lexical_category_probabilities(payload.query)

    lexical_peak = max(lexical_probabilities.values())
    lexical_weight = min(0.68, 0.34 + lexical_peak * 0.65)
    model_weight = 1.0 - lexical_weight

    blended = {
        category: model_probabilities[category] * model_weight + lexical_probabilities[category] * lexical_weight
        for category in CATEGORY_LIST
    }
    total = sum(blended.values()) or 1.0
    probabilities = {category: value / total for category, value in blended.items()}

    predicted_category = max(probabilities.items(), key=lambda item: item[1])[0]
    predicted_score = probabilities[predicted_category]

    tag_updates = expand_query_to_tags(payload.query, probabilities, top_k=40)
    event_store.record_search(payload.query, predicted_category, probabilities, tag_updates)

    model_top = max(model_probabilities.items(), key=lambda item: item[1])[0]
    lexical_top = max(lexical_probabilities.items(), key=lambda item: item[1])[0]

    if predicted_score >= 0.62 and (model_top == lexical_top or lexical_peak < 0.28):
        model_service.online_update(payload.query, predicted_category, sample_weight=0.35)

    top_segments = event_store.get_top_segments(limit=6)
    cloud_words = event_store.get_cloud_words(limit=180)

    return {
        "query": payload.query,
        "predicted_category": predicted_category,
        "probabilities": probabilities,
        "top_segments": top_segments,
        "cloud_words": cloud_words,
        "model_source": "python-backend",
        "model_top": model_top,
        "lexical_top": lexical_top,
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> Dict[str, object]:
    """
    Submit feedback for a search query.
    :param payload: The feedback request payload containing the query, category, and confidence.
    :return: A dictionary containing the status and the updated query information.
    """
    if payload.category not in CATEGORY_LIST:
        raise HTTPException(status_code=400, detail="Unknown category")

    event_store.record_feedback(payload.query, payload.category, payload.confidence)
    model_service.online_update(payload.query, payload.category, sample_weight=payload.confidence)

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
    for query, category, confidence in event_store.get_feedback_examples():
        model_service.online_update(query, category, sample_weight=confidence)

    return {
        "status": "retrained",
        "categories": len(CATEGORY_LIST),
    }
