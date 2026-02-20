"""
Taxonomy Model Service for managing the machine learning model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier

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
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dimensions = int(
            self.embedding_model.get_sentence_embedding_dimension())

        self.classifier = MLPClassifier(
            hidden_layer_sizes=(192, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            random_state=42,
            max_iter=350,
            early_stopping=True,
            n_iter_no_change=10,
            tol=1e-4,
        )
        self.category_prototypes = self._build_category_prototypes()
        self.is_trained = False

        self._load_or_train()

    def _load_or_train(self) -> None:
        """
        Load the model from the artifact file if it exists, otherwise train the initial model.
        """
        if self.artifact_file.exists():
            try:
                payload = joblib.load(self.artifact_file)
                loaded_classifier = payload.get("classifier")
                loaded_categories = payload.get("categories")
                loaded_embedding_model_name = payload.get(
                    "embedding_model_name")

                if (
                    isinstance(loaded_classifier, MLPClassifier)
                    and isinstance(loaded_categories, list)
                    and loaded_embedding_model_name == self.embedding_model_name
                ):
                    self.classifier = loaded_classifier
                    self.categories = [str(category)
                                       for category in loaded_categories]
                    self.category_prototypes = self._build_category_prototypes()
                    self.is_trained = True
                    return
            except (OSError, ValueError, TypeError, KeyError):
                pass

        self.train_initial_model()

    def save(self) -> None:
        """
        Save the current model to the artifact file.
        """
        payload = {
            "classifier": self.classifier,
            "categories": self.categories,
            "embedding_model_name": self.embedding_model_name,
        }
        joblib.dump(payload, self.artifact_file)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode text samples using the sentence embedding model.
        :param texts: Input text samples.
        :return: 2D embedding matrix.
        """
        if not texts:
            return np.zeros((0, self.embedding_dimensions), dtype=np.float64)

        vectors = self.embedding_model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)

    def train_initial_model(self) -> None:
        """
        Train the initial model using the seed corpus.
        """
        samples = build_seed_corpus()
        texts = [text for text, _ in samples]
        labels = np.array([label for _, label in samples])

        x_matrix = self._encode_texts(texts)
        self.classifier.fit(x_matrix, labels)

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
        dense = self._encode_texts([query])[0]
        return self._normalize_vector(dense)

    def _build_category_prototypes(self) -> Dict[str, np.ndarray]:
        """
        Build normalized prototype vectors for each category.
        :return: Mapping of category to prototype vector.
        """
        grouped_texts: Dict[str, List[str]] = {
            category: [] for category in self.categories}

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
                prototypes[category] = np.zeros(
                    self.embedding_dimensions, dtype=np.float64)
                continue

            dense = self._encode_texts(texts)
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

        x_matrix = self._encode_texts([query])
        probabilities = self.classifier.predict_proba(x_matrix)[0]

        paired = sorted(zip(self.categories, probabilities),
                        key=lambda item: item[1], reverse=True)
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

        if not self.is_trained:
            self.train_initial_model()

        x_base = self._encode_texts([query])
        repeat_count = max(1, int(round(max(0.1, sample_weight) * 4)))
        x_matrix = np.repeat(x_base, repeat_count, axis=0)
        y = np.array([category] * repeat_count)

        if hasattr(self.classifier, "classes_"):
            self.classifier.partial_fit(x_matrix, y)
        else:
            self.classifier.partial_fit(
                x_matrix, y, classes=np.array(self.categories))

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
