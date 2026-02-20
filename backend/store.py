"""
Database stores for global learning state and per-user personalization state.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Sequence, Tuple
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DatabaseStore:
    """
    Shared database store and schema initialization.
    """

    def __init__(self, database_url: str, db_path: Path | None = None) -> None:
        if not database_url and db_path is not None:
            database_url = f"sqlite:///{db_path}"
        if not database_url:
            raise ValueError("DATABASE_URL is required")

        if database_url.startswith("sqlite:///") and db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine: Engine = create_engine(database_url, future=True)
        self.write_lock = Lock()
        self.segment_decay_lambda = self._load_segment_decay_lambda()
        self._init_db()

    def _load_segment_decay_lambda(self) -> float:
        raw_value = os.getenv("SEGMENT_DECAY_LAMBDA", "0.08")
        try:
            value = float(raw_value)
            return max(0.0, value)
        except (TypeError, ValueError):
            return 0.08

    def _init_db(self) -> None:
        schema_statements = [
            """
            CREATE TABLE IF NOT EXISTS global_feedback_events (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                true_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS global_conversion_affinity (
                category TEXT PRIMARY KEY,
                score REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS global_model_registry (
                model_version TEXT PRIMARY KEY,
                artifact_uri TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_search_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                probabilities_json TEXT NOT NULL,
                intent_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_feedback_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                true_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_category_totals (
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, category)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_tag_totals (
                user_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                category TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, tag, category)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_embedding (
                user_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_conversion_affinity (
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, category)
            )
            """,
        ]

        with self.engine.begin() as connection:
            for statement in schema_statements:
                connection.execute(text(statement))


class UserEventStore:
    """
    Store for private, per-user history and personalization state.
    """

    def __init__(self, db: DatabaseStore) -> None:
        self.db = db

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_search(
        self,
        user_id: str,
        query: str,
        predicted_category: str,
        probabilities: Dict[str, float],
        intent_probabilities: Dict[str, float],
        tag_updates: Sequence[Tuple[str, str, float]],
    ) -> None:
        timestamp = self._now()

        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO user_search_events (
                            id, user_id, query, predicted_category, probabilities_json, intent_json, created_at
                        )
                        VALUES (
                            :id, :user_id, :query, :predicted_category, :probabilities_json, :intent_json, :created_at
                        )
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "user_id": user_id,
                        "query": query,
                        "predicted_category": predicted_category,
                        "probabilities_json": json.dumps(probabilities),
                        "intent_json": json.dumps(intent_probabilities),
                        "created_at": timestamp,
                    },
                )

                for category, score in probabilities.items():
                    connection.execute(
                        text(
                            """
                            INSERT INTO user_category_totals (user_id, category, score, updated_at)
                            VALUES (:user_id, :category, :score, :updated_at)
                            ON CONFLICT(user_id, category) DO UPDATE SET
                                score = user_category_totals.score + excluded.score,
                                updated_at = excluded.updated_at
                            """
                        ),
                        {
                            "user_id": user_id,
                            "category": category,
                            "score": float(score),
                            "updated_at": timestamp,
                        },
                    )

                for tag, category, weight in tag_updates:
                    connection.execute(
                        text(
                            """
                            INSERT INTO user_tag_totals (user_id, tag, category, score)
                            VALUES (:user_id, :tag, :category, :score)
                            ON CONFLICT(user_id, tag, category) DO UPDATE SET
                                score = user_tag_totals.score + excluded.score
                            """
                        ),
                        {
                            "user_id": user_id,
                            "tag": tag,
                            "category": category,
                            "score": float(weight),
                        },
                    )

    def record_feedback(self, user_id: str, query: str, true_category: str, confidence: float) -> None:
        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO user_feedback_events (id, user_id, query, true_category, confidence, created_at)
                        VALUES (:id, :user_id, :query, :true_category, :confidence, :created_at)
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "user_id": user_id,
                        "query": query,
                        "true_category": true_category,
                        "confidence": float(confidence),
                        "created_at": self._now(),
                    },
                )

    def clear_all(self, user_id: str) -> None:
        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(text("DELETE FROM user_feedback_events WHERE user_id = :user_id"), {"user_id": user_id})
                connection.execute(text("DELETE FROM user_search_events WHERE user_id = :user_id"), {"user_id": user_id})
                connection.execute(text("DELETE FROM user_category_totals WHERE user_id = :user_id"), {"user_id": user_id})
                connection.execute(text("DELETE FROM user_tag_totals WHERE user_id = :user_id"), {"user_id": user_id})
                connection.execute(text("DELETE FROM user_embedding WHERE user_id = :user_id"), {"user_id": user_id})
                connection.execute(text("DELETE FROM user_conversion_affinity WHERE user_id = :user_id"), {"user_id": user_id})

    def increment_conversion_affinity(self, user_id: str, category: str, amount: float) -> None:
        increment = max(0.0, float(amount))
        if increment <= 0.0:
            return

        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO user_conversion_affinity (user_id, category, score, updated_at)
                        VALUES (:user_id, :category, :score, :updated_at)
                        ON CONFLICT(user_id, category) DO UPDATE SET
                            score = user_conversion_affinity.score + excluded.score,
                            updated_at = excluded.updated_at
                        """
                    ),
                    {
                        "user_id": user_id,
                        "category": category,
                        "score": increment,
                        "updated_at": self._now(),
                    },
                )

    def get_conversion_affinity(self, user_id: str, categories: Sequence[str]) -> Dict[str, float]:
        requested = [str(category) for category in categories]
        if not requested:
            return {}

        with self.db.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT category, score
                    FROM user_conversion_affinity
                    WHERE user_id = :user_id
                    """
                ),
                {"user_id": user_id},
            ).mappings().all()

        loaded = {str(row["category"]): max(0.0, float(row["score"])) for row in rows}
        return {category: loaded.get(category, 0.0) for category in requested}

    def get_user_embedding(self, user_id: str, dimensions: int) -> List[float]:
        safe_dimensions = max(1, int(dimensions))

        with self.db.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT vector_json
                    FROM user_embedding
                    WHERE user_id = :user_id
                    """
                ),
                {"user_id": user_id},
            ).mappings().first()

        if row is None:
            return [0.0] * safe_dimensions

        try:
            vector = json.loads(row["vector_json"])
            if not isinstance(vector, list):
                return [0.0] * safe_dimensions
            normalized = [float(value) for value in vector[:safe_dimensions]]
            if len(normalized) < safe_dimensions:
                normalized.extend([0.0] * (safe_dimensions - len(normalized)))
            return normalized
        except (TypeError, ValueError, json.JSONDecodeError):
            return [0.0] * safe_dimensions

    def set_user_embedding(self, user_id: str, vector: Sequence[float]) -> None:
        serialized = json.dumps([float(value) for value in vector])

        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO user_embedding (user_id, vector_json)
                        VALUES (:user_id, :vector_json)
                        ON CONFLICT(user_id) DO UPDATE SET vector_json = excluded.vector_json
                        """
                    ),
                    {
                        "user_id": user_id,
                        "vector_json": serialized,
                    },
                )

    def get_top_segments(self, user_id: str, limit: int = 6) -> List[Dict[str, float]]:
        with self.db.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT category, score, updated_at
                    FROM user_category_totals
                    WHERE user_id = :user_id
                    ORDER BY score DESC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": limit},
            ).mappings().all()

        now = datetime.now(timezone.utc)
        decay_lambda = self.db.segment_decay_lambda
        decayed_rows = []
        for row in rows:
            updated_raw = row["updated_at"]
            try:
                updated_at = datetime.fromisoformat(updated_raw)
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (now - updated_at).total_seconds() / 86400.0)
            except (TypeError, ValueError):
                age_days = 0.0

            adjusted_score = float(row["score"]) * math.exp(-decay_lambda * age_days)
            decayed_rows.append(
                {
                    "category": row["category"],
                    "adjusted_score": adjusted_score,
                }
            )

        total = sum(item["adjusted_score"] for item in decayed_rows) or 1.0
        return [
            {
                "category": item["category"],
                "score": float(item["adjusted_score"]),
                "probability": float(item["adjusted_score"] / total),
            }
            for item in decayed_rows
        ]

    def get_cloud_words(self, user_id: str, limit: int = 180) -> List[Dict[str, float]]:
        with self.db.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT tag, category, score
                    FROM user_tag_totals
                    WHERE user_id = :user_id
                    ORDER BY score DESC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": limit},
            ).mappings().all()

        return [
            {
                "tag": row["tag"],
                "category": row["category"],
                "weight": float(row["score"]),
            }
            for row in rows
        ]

    def get_recent_searches(self, user_id: str, limit: int = 20) -> List[Dict[str, object]]:
        safe_limit = max(1, min(limit, 200))

        with self.db.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        se.id,
                        se.query,
                        se.predicted_category,
                        se.probabilities_json,
                        se.intent_json,
                        se.created_at,
                        fe.true_category AS feedback_category,
                        fe.confidence AS feedback_confidence,
                        fe.created_at AS feedback_created_at
                    FROM user_search_events se
                    LEFT JOIN user_feedback_events fe
                        ON fe.id = (
                            SELECT f.id
                            FROM user_feedback_events f
                            WHERE f.user_id = se.user_id
                              AND f.query = se.query
                            ORDER BY f.created_at DESC
                            LIMIT 1
                        )
                    WHERE se.user_id = :user_id
                    ORDER BY se.created_at DESC
                    LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": safe_limit},
            ).mappings().all()

        output: List[Dict[str, object]] = []
        for row in rows:
            probabilities = json.loads(row["probabilities_json"])
            intent_probabilities = json.loads(row["intent_json"] or "{}")
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            ranked_intents = sorted(
                intent_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            output.append(
                {
                    "id": str(row["id"]),
                    "query": row["query"],
                    "predicted_category": row["predicted_category"],
                    "created_at": row["created_at"],
                    "top_intents": [
                        {"intent": intent, "probability": float(probability)}
                        for intent, probability in ranked_intents[:3]
                    ],
                    "top_categories": [
                        {"category": category, "probability": float(probability)}
                        for category, probability in ranked[:3]
                    ],
                    "feedback": {
                        "category": row["feedback_category"],
                        "confidence": float(row["feedback_confidence"]) if row["feedback_confidence"] is not None else None,
                        "created_at": row["feedback_created_at"],
                    }
                    if row["feedback_category"]
                    else None,
                }
            )

        return output


class GlobalLearningStore:
    """
    Store for hidden global learning state shared across all users.
    """

    def __init__(self, db: DatabaseStore) -> None:
        self.db = db

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_feedback(self, query: str, true_category: str, confidence: float) -> None:
        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO global_feedback_events (id, query, true_category, confidence, created_at)
                        VALUES (:id, :query, :true_category, :confidence, :created_at)
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "query": query,
                        "true_category": true_category,
                        "confidence": float(confidence),
                        "created_at": self._now(),
                    },
                )

    def increment_conversion_affinity(self, category: str, amount: float) -> None:
        increment = max(0.0, float(amount))
        if increment <= 0.0:
            return

        with self.db.write_lock:
            with self.db.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO global_conversion_affinity (category, score, updated_at)
                        VALUES (:category, :score, :updated_at)
                        ON CONFLICT(category) DO UPDATE SET
                            score = global_conversion_affinity.score + excluded.score,
                            updated_at = excluded.updated_at
                        """
                    ),
                    {
                        "category": category,
                        "score": increment,
                        "updated_at": self._now(),
                    },
                )

    def get_conversion_affinity(self, categories: Sequence[str]) -> Dict[str, float]:
        requested = [str(category) for category in categories]
        if not requested:
            return {}

        with self.db.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT category, score
                    FROM global_conversion_affinity
                    """
                )
            ).mappings().all()

        loaded = {str(row["category"]): max(0.0, float(row["score"])) for row in rows}
        return {category: loaded.get(category, 0.0) for category in requested}

    def get_feedback_examples(self) -> List[Tuple[str, str, float]]:
        with self.db.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT query, true_category, confidence
                    FROM global_feedback_events
                    ORDER BY created_at ASC
                    """
                )
            ).mappings().all()

        return [(row["query"], row["true_category"], float(row["confidence"])) for row in rows]
