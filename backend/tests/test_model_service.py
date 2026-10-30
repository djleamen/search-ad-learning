"""Model service robustness: corrupt artifacts and concurrent online updates.

sentence-transformers (and torch) are stubbed with a tiny deterministic
embedder so this runs in CI in seconds.
"""
import hashlib
import sys
import threading
import types

import joblib
import numpy as np


class _FakeEmbedder:
    def __init__(self, _name):
        pass

    @staticmethod
    def get_sentence_embedding_dimension():
        return 16

    @staticmethod
    def encode(texts, **_kwargs):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()[:16]
            rows.append(np.frombuffer(digest, dtype=np.uint8).astype(np.float64) / 255.0)
        return np.vstack(rows)


sys.modules.setdefault(
    "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_FakeEmbedder)
)

import backend.model_service as model_service_module  # noqa: E402
from backend.model_service import TaxonomyModelService  # noqa: E402
from backend.taxonomy_data import CATEGORY_LIST, build_seed_corpus  # noqa: E402


def _small_seed_corpus(per_category=3):
    """A few samples per category: same classes, a fraction of the fit time."""
    kept, seen = [], {}
    for text, label in build_seed_corpus():
        if seen.get(label, 0) < per_category:
            kept.append((text, label))
            seen[label] = seen.get(label, 0) + 1
    return kept


model_service_module.build_seed_corpus = _small_seed_corpus


def test_corrupt_artifact_retrains_instead_of_crashing(tmp_path):
    (tmp_path / "taxonomy_model.joblib").write_bytes(b"\x80\x04truncated")
    service = TaxonomyModelService(artifact_dir=tmp_path)
    assert service.is_trained
    assert "classifier" in joblib.load(tmp_path / "taxonomy_model.joblib")


def test_concurrent_updates_and_predictions_stay_consistent(tmp_path):
    service = TaxonomyModelService(artifact_dir=tmp_path)
    errors = []

    def worker(n):
        try:
            for i in range(10):
                category = CATEGORY_LIST[(n + i) % len(CATEGORY_LIST)]
                service.online_update(f"query {n} {i}", category, sample_weight=0.5)
                probs = service.predict_proba(f"query {n} {i}")
                assert abs(sum(probs.values()) - 1.0) < 1e-6
        except Exception as error:  # pylint: disable=broad-except
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert "classifier" in joblib.load(tmp_path / "taxonomy_model.joblib")
    assert not list(tmp_path.glob("*.tmp"))
