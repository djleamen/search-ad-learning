"""
Event Store for recording search events, feedback, and maintaining category/tag totals.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Sequence, Tuple


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
        self._init_db()

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
                    score REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS tag_totals (
                    tag TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0
                );
                """
            )
            connection.commit()

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
                    (query, predicted_category, json.dumps(probabilities), timestamp),
                )

                for category, score in probabilities.items():
                    connection.execute(
                        """
                        INSERT INTO category_totals (category, score)
                        VALUES (?, ?)
                        ON CONFLICT(category) DO UPDATE SET score = category_totals.score + excluded.score
                        """,
                        (category, score),
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
                    """
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
                SELECT category, score
                FROM category_totals
                ORDER BY score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        total = sum(row["score"] for row in rows) or 1.0
        return [
            {
                "category": row["category"],
                "score": float(row["score"]),
                "probability": float(row["score"] / total),
            }
            for row in rows
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
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            output.append(
                {
                    "id": int(row["id"]),
                    "query": row["query"],
                    "predicted_category": row["predicted_category"],
                    "created_at": row["created_at"],
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
