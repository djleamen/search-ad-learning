"""
Event Store for recording search events, feedback, and maintaining category/tag totals.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Sequence, Tuple


class EventStore:
    """
    Event Store for recording search events, feedback, and maintaining category/tag totals.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the EventStore.
        :param db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.segment_decay_lambda = self._load_segment_decay_lambda()
        self._init_db()

    def _load_segment_decay_lambda(self) -> float:
        """
        Load segment decay lambda from environment.
        :return: Non-negative decay lambda.
        """
        raw_value = os.getenv("SEGMENT_DECAY_LAMBDA", "0.08")
        try:
            value = float(raw_value)
            return max(0.0, value)
        except (TypeError, ValueError):
            return 0.08

    def _connect(self) -> sqlite3.Connection:
        """
        Create a new connection to the SQLite database.
        :return: A SQLite connection object.
        """
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        """
        Initialize the SQLite database by creating the necessary tables if they do not exist.
        """
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    true_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS category_totals (
                    category TEXT PRIMARY KEY,
                    score REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tag_totals (
                    tag TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS user_embedding (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    vector_json TEXT NOT NULL
                );
                """
            )
            self._ensure_category_totals_schema(connection)
            connection.commit()

    def _ensure_category_totals_schema(self, connection: sqlite3.Connection) -> None:
        """
        Ensure category_totals has the updated_at column for recency decay support.
        :param connection: Active SQLite connection.
        """
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(category_totals)").fetchall()
        }
        if "updated_at" in columns:
            return

        connection.execute(
            """
            ALTER TABLE category_totals
            ADD COLUMN updated_at TEXT
            """
        )
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            UPDATE category_totals
            SET updated_at = ?
            WHERE updated_at IS NULL OR updated_at = ''
            """,
            (now,),
        )

    def record_search(
        self,
        query: str,
        predicted_category: str,
        probabilities: Dict[str, float],
        tag_updates: Sequence[Tuple[str, str, float]],
    ) -> None:
        """
        Record a search event in the database, along with category totals and tag updates.
        :param query: The search query.
        :param predicted_category: The category predicted by the model.
        :param probabilities: The predicted probabilities for each category.
        :param tag_updates: A sequence of tuples containing tag updates in the form (tag, category, weight).
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO search_events (query, predicted_category, probabilities_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query, predicted_category, json.dumps(
                        probabilities), timestamp),
                )

                for category, score in probabilities.items():
                    connection.execute(
                        """
                        INSERT INTO category_totals (category, score, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(category) DO UPDATE SET
                            score = category_totals.score + excluded.score,
                            updated_at = excluded.updated_at
                        """,
                        (category, score, timestamp),
                    )

                for tag, category, weight in tag_updates:
                    connection.execute(
                        """
                        INSERT INTO tag_totals (tag, category, score)
                        VALUES (?, ?, ?)
                        ON CONFLICT(tag) DO UPDATE SET
                            score = tag_totals.score + excluded.score,
                            category = excluded.category
                        """,
                        (tag, category, weight),
                    )

                connection.commit()

    def record_feedback(self, query: str, true_category: str, confidence: float) -> None:
        """
        Record a feedback event in the database.
        :param query: The search query.
        :param true_category: The true category of the query.
        :param confidence: The confidence score of the feedback.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback_events (query, true_category, confidence, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query, true_category, confidence, timestamp),
                )
                connection.commit()

    def clear_all(self) -> None:
        """
        Clear all data from the database, including feedback events, search events, category totals, and tag totals.
        """
        with self._lock:
            with self._connect() as connection:
                connection.executescript(
                    """
                    DELETE FROM feedback_events;
                    DELETE FROM search_events;
                    DELETE FROM category_totals;
                    DELETE FROM tag_totals;
                    DELETE FROM user_embedding;
                    """
                )
                connection.commit()

    def get_user_embedding(self, dimensions: int) -> List[float]:
        """
        Retrieve the persisted user embedding vector.
        :param dimensions: Expected embedding dimensionality.
        :return: A dense embedding vector with the requested dimensionality.
        """
        safe_dimensions = max(1, int(dimensions))

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT vector_json
                FROM user_embedding
                WHERE id = 1
                """
            ).fetchone()

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

    def set_user_embedding(self, vector: Sequence[float]) -> None:
        """
        Persist the user embedding vector.
        :param vector: Dense embedding vector values.
        """
        serialized = json.dumps([float(value) for value in vector])

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO user_embedding (id, vector_json)
                    VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET vector_json = excluded.vector_json
                    """,
                    (serialized,),
                )
                connection.commit()

    def get_top_segments(self, limit: int = 6) -> List[Dict[str, float]]:
        """
        Get the top categories based on their scores.
        :param limit: The maximum number of top categories to return.
        :return: A list of dictionaries containing category, score, and probability.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT category, score, updated_at
                FROM category_totals
                ORDER BY score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        now = datetime.now(timezone.utc)
        decay_lambda = self.segment_decay_lambda
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

    def get_cloud_words(self, limit: int = 180) -> List[Dict[str, float]]:
        """
        Get the top tags based on their scores.
        :param limit: The maximum number of top tags to return.
        :return: A list of dictionaries containing tag, category, and weight.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT tag, category, score
                FROM tag_totals
                ORDER BY score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "tag": row["tag"],
                "category": row["category"],
                "weight": float(row["score"]),
            }
            for row in rows
        ]

    def get_feedback_examples(self) -> List[Tuple[str, str, float]]:
        """
        Get all feedback examples from the database.
        :return: A list of tuples containing query, true_category, and confidence.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT query, true_category, confidence
                FROM feedback_events
                ORDER BY id ASC
                """
            ).fetchall()

        return [(row["query"], row["true_category"], float(row["confidence"])) for row in rows]

    def get_recent_searches(self, limit: int = 20) -> List[Dict[str, object]]:
        """
        Get the most recent search events from the database, along with their feedback if available.
        :param limit: The maximum number of recent searches to return.
        :return: A list of dictionaries containing search event details and feedback.
        """
        safe_limit = max(1, min(limit, 200))

        with self._connect() as connection:
            rows = connection.execute(
                """
                                SELECT
                                        se.id,
                                        se.query,
                                        se.predicted_category,
                                        se.probabilities_json,
                                        se.created_at,
                                        fe.true_category AS feedback_category,
                                        fe.confidence AS feedback_confidence,
                                        fe.created_at AS feedback_created_at
                                FROM search_events se
                                LEFT JOIN feedback_events fe
                                    ON fe.id = (
                                        SELECT id
                                        FROM feedback_events
                                        WHERE query = se.query
                                        ORDER BY id DESC
                                        LIMIT 1
                                    )
                                ORDER BY se.id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        output: List[Dict[str, object]] = []
        for row in rows:
            probabilities = json.loads(row["probabilities_json"])
            ranked = sorted(probabilities.items(),
                            key=lambda item: item[1], reverse=True)
            output.append(
                {
                    "id": int(row["id"]),
                    "query": row["query"],
                    "predicted_category": row["predicted_category"],
                    "created_at": row["created_at"],
                    "top_categories": [
                        {"category": category,
                            "probability": float(probability)}
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
