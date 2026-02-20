"""
Taxonomy Model Service for managing the machine learning model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from .taxonomy_data import CATEGORY_LIST, build_seed_corpus


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
        self.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-6,
            penalty="l2",
            random_state=42,
            max_iter=12,
            tol=1e-3,
        )
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
