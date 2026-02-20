"""
Taxonomy Model Service for managing the machine learning model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from .taxonomy_data import CATEGORY_LIST, TAXONOMY, build_seed_corpus


class TaxonomyModelService:
    """
    Taxonomy Model Service for managing the machine learning model.
    Uses a stochastic gradient descent classifier with log loss and 
    a hashing vectorizer for text features. 
    """
    def __init__(self, artifact_dir: Path) -> None:
        """
        Initialize the TaxonomyModelService.
        :param artifact_dir: The directory where the model artifacts will be stored.
        """
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_file = self.artifact_dir / "taxonomy_model.joblib"

        self.categories: List[str] = list(CATEGORY_LIST)
        self.vectorizer = HashingVectorizer(
            n_features=2**18,
            alternate_sign=False,
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
        )
        self.embedding_dimensions = 64
        self.embedding_vectorizer = HashingVectorizer(
            n_features=self.embedding_dimensions,
            alternate_sign=False,
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
        )
        self.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-6,
            penalty="l2",
            random_state=42,
            max_iter=12,
            tol=1e-3,
        )
        self.category_prototypes = self._build_category_prototypes()
        self.is_trained = False

        self._load_or_train()

    def _load_or_train(self) -> None:
        """
        Load the model from the artifact file if it exists, otherwise train the initial model.
        """
        if self.artifact_file.exists():
            payload = joblib.load(self.artifact_file)
            self.classifier = payload["classifier"]
            self.categories = payload["categories"]
            self.is_trained = True
            return

        self.train_initial_model()

    def save(self) -> None:
        """
        Save the current model to the artifact file.
        """
        payload = {
            "classifier": self.classifier,
            "categories": self.categories,
        }
        joblib.dump(payload, self.artifact_file)

    def train_initial_model(self) -> None:
        """
        Train the initial model using the seed corpus.
        """
        samples = build_seed_corpus()
        texts = [text for text, _ in samples]
        labels = np.array([label for _, label in samples])

        x_matrix = self.vectorizer.transform(texts)
        classes = np.array(self.categories)

        for _ in range(4):
            self.classifier.partial_fit(x_matrix, labels, classes=classes)

        self.is_trained = True
        self.save()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a dense vector to unit length.
        :param vector: Dense vector.
        :return: Unit-normalized vector.
        """
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return np.zeros_like(vector, dtype=np.float64)
        return vector.astype(np.float64) / norm

    def _encode_query_embedding(self, query: str) -> np.ndarray:
        """
        Encode a query into a normalized dense embedding vector.
        :param query: Query text.
        :return: Dense normalized embedding.
        """
        sparse = self.embedding_vectorizer.transform([query])
        dense = np.asarray(sparse.todense(), dtype=np.float64)[0]
        return self._normalize_vector(dense)

    def _build_category_prototypes(self) -> Dict[str, np.ndarray]:
        """
        Build normalized prototype vectors for each category.
        :return: Mapping of category to prototype vector.
        """
        grouped_texts: Dict[str, List[str]] = {category: [] for category in self.categories}

        for category, payload in TAXONOMY.items():
            grouped_texts.setdefault(category, [])
            grouped_texts[category].extend(payload.get("training_queries", []))
            grouped_texts[category].extend(payload.get("tags", []))

        for text, label in build_seed_corpus():
            grouped_texts.setdefault(label, [])
            grouped_texts[label].append(text)

        prototypes: Dict[str, np.ndarray] = {}
        for category in self.categories:
            texts = grouped_texts.get(category, [])
            if not texts:
                prototypes[category] = np.zeros(self.embedding_dimensions, dtype=np.float64)
                continue

            sparse = self.embedding_vectorizer.transform(texts)
            dense = np.asarray(sparse.todense(), dtype=np.float64)
            mean_vector = dense.mean(axis=0)
            prototypes[category] = self._normalize_vector(mean_vector)

        return prototypes

    def predict_proba(self, query: str) -> Dict[str, float]:
        """
        Predict the category probabilities for a given query.
        :param query: The search query to classify.
        :return: A dictionary mapping categories to their predicted probabilities.
        """
        if not self.is_trained:
            self.train_initial_model()

        x_matrix = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(x_matrix)[0]

        paired = sorted(zip(self.categories, probabilities), key=lambda item: item[1], reverse=True)
        total = sum(value for _, value in paired) or 1.0
        return {category: float(value / total) for category, value in paired}

    def online_update(self, query: str, category: str, sample_weight: float = 0.5) -> None:
        """
        Update the model online with a new query and its corresponding category.
        :param query: The search query to update the model with.
        :param category: The category of the query.
        :param sample_weight: The weight of the sample for online learning.
        """
        if category not in self.categories:
            return

        x_matrix = self.vectorizer.transform([query])
        y = np.array([category])
        classes = np.array(self.categories)
        weights = np.array([sample_weight])

        self.classifier.partial_fit(x_matrix, y, classes=classes, sample_weight=weights)
        self.save()

    def update_user_embedding(
        self,
        user_vector: Sequence[float],
        query: str,
        decay: float = 0.95,
        learning_rate: float = 0.05,
    ) -> List[float]:
        """
        Apply an embedding update from a query to the persistent user vector.
        :param user_vector: Existing user embedding vector.
        :param query: Query text to encode as an embedding delta.
        :param decay: Exponential decay factor for historical behavior.
        :param learning_rate: Update strength for the new query vector.
        :return: Updated normalized user embedding vector.
        """
        clipped_decay = min(max(decay, 0.0), 1.0)
        clipped_learning = min(max(learning_rate, 0.0), 1.0)

        base = np.zeros(self.embedding_dimensions, dtype=np.float64)
        source = np.array(list(user_vector), dtype=np.float64)
        usable = min(source.shape[0], self.embedding_dimensions)
        if usable > 0:
            base[:usable] = source[:usable]

        base = self._normalize_vector(base)
        query_embedding = self._encode_query_embedding(query)

        updated = clipped_decay * base + clipped_learning * query_embedding
        return self._normalize_vector(updated).tolist()

    def category_probabilities_from_user_embedding(self, user_vector: Sequence[float]) -> Dict[str, float]:
        """
        Convert a user embedding into category probabilities via cosine similarity.
        :param user_vector: Current user embedding vector.
        :return: Category probability map.
        """
        base = np.zeros(self.embedding_dimensions, dtype=np.float64)
        source = np.array(list(user_vector), dtype=np.float64)
        usable = min(source.shape[0], self.embedding_dimensions)
        if usable > 0:
            base[:usable] = source[:usable]

        normalized_user = self._normalize_vector(base)
        if float(np.linalg.norm(normalized_user)) <= 1e-8:
            uniform = 1.0 / max(len(self.categories), 1)
            return {category: uniform for category in self.categories}

        raw_scores: Dict[str, float] = {}
        for category in self.categories:
            prototype = self.category_prototypes.get(category)
            if prototype is None:
                raw_scores[category] = 0.0
                continue
            similarity = float(np.dot(normalized_user, prototype))
            raw_scores[category] = max(similarity, 0.0)

        total = sum(raw_scores.values())
        if total <= 1e-12:
            uniform = 1.0 / max(len(self.categories), 1)
            return {category: uniform for category in self.categories}

        return {category: (score / total) for category, score in raw_scores.items()}
