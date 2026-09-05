"""Checks for the pure lexical/intent helpers (no model or DB needed)."""
import math

from backend.taxonomy_data import (
    CATEGORY_LIST,
    classify_intent_probabilities,
    expand_query_to_tags,
    lexical_category_probabilities,
)


def test_lexical_probabilities_are_a_distribution_over_all_categories():
    probs = lexical_category_probabilities("cheap flights and hotel booking")
    assert set(probs) == set(CATEGORY_LIST)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert max(probs, key=probs.get) == "/Travel"


def test_intent_probabilities_flag_transactional_queries():
    buy = classify_intent_probabilities("buy cheap gaming monitor")
    ask = classify_intent_probabilities("how does a gaming monitor work")
    assert math.isclose(sum(buy.values()), 1.0, rel_tol=1e-9)
    assert buy["transactional"] > ask["transactional"]
    assert ask["informational"] > buy["informational"]


def test_expand_query_to_tags_respects_top_k_and_skips_low_probability():
    probs = {category: 0.0 for category in CATEGORY_LIST}
    probs["/Travel"] = 1.0
    tags = expand_query_to_tags("weekend getaways", probs, top_k=5)
    assert len(tags) == 5
    assert {category for _, category, _ in tags} == {"/Travel"}
    assert tags == sorted(tags, key=lambda item: item[2], reverse=True)
