from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np


ItemId: TypeAlias = int | str
RecommendedItems: TypeAlias = Sequence[ItemId]
RelevantItems: TypeAlias = set[ItemId]
RecommendationByUser: TypeAlias = Mapping[int | str, RecommendedItems]
RelevantByUser: TypeAlias = Mapping[int | str, RelevantItems]
ItemVectorById: TypeAlias = Mapping[ItemId, Sequence[float] | np.ndarray]


def _validate_k(k: int) -> None:
    """Validate a top-k cutoff."""

    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")


def _deduplicate_ranked_items(recommended: RecommendedItems) -> list[ItemId]:
    """Remove duplicate ranked items while preserving first-seen order."""

    seen: set[ItemId] = set()
    deduplicated: list[ItemId] = []

    for item_id in recommended:
        if item_id in seen:
            continue
        seen.add(item_id)
        deduplicated.append(item_id)

    return deduplicated


def _top_k_items(recommended: RecommendedItems, k: int) -> list[ItemId]:
    """Return the first k unique recommended items in rank order."""

    _validate_k(k)
    return _deduplicate_ranked_items(recommended)[:k]


def recall_at_k(recommended: RecommendedItems, relevant: RelevantItems, k: int) -> float:
    """Calculate Recall@K for one ranked recommendation list."""

    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended, k)
    hits = len(set(top_k).intersection(relevant))
    return hits / len(relevant)


def precision_at_k(recommended: RecommendedItems, relevant: RelevantItems, k: int) -> float:
    """Calculate Precision@K using K as the denominator."""

    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended, k)
    hits = len(set(top_k).intersection(relevant))
    return hits / k


def hit_rate_at_k(recommended: RecommendedItems, relevant: RelevantItems, k: int) -> float:
    """Calculate binary HitRate@K for one ranked recommendation list."""

    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended, k)
    return float(any(item_id in relevant for item_id in top_k))


def average_precision_at_k(
    recommended: RecommendedItems,
    relevant: RelevantItems,
    k: int,
) -> float:
    """Calculate Average Precision@K for one ranked recommendation list."""

    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended, k)
    hits = 0
    precision_sum = 0.0

    for rank, item_id in enumerate(top_k, start=1):
        if item_id not in relevant:
            continue

        hits += 1
        precision_sum += hits / rank

    if hits == 0:
        return 0.0

    return precision_sum / min(len(relevant), k)


def map_at_k(
    all_recommendations: RecommendationByUser,
    all_relevant: RelevantByUser,
    k: int,
) -> float:
    """Calculate macro-averaged MAP@K across users with non-empty relevance sets."""

    _validate_k(k)
    if not all_relevant:
        return 0.0

    user_scores = [
        average_precision_at_k(all_recommendations.get(user_id, []), relevant, k)
        for user_id, relevant in all_relevant.items()
        if relevant
    ]

    if not user_scores:
        return 0.0

    return float(np.mean(user_scores))


def ndcg_at_k(
    recommended: RecommendedItems,
    relevance_scores: Mapping[ItemId, float | int],
    k: int,
) -> float:
    """Calculate graded NDCG@K for one ranked recommendation list."""

    if not relevance_scores:
        return 0.0

    top_k = _top_k_items(recommended, k)
    gains = [_rating_to_gain(relevance_scores.get(item_id, 0.0)) for item_id in top_k]
    dcg = _discounted_cumulative_gain(gains)

    ideal_gains = sorted(
        (_rating_to_gain(score) for score in relevance_scores.values()),
        reverse=True,
    )[:k]
    idcg = _discounted_cumulative_gain(ideal_gains)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def catalog_coverage(
    all_recommendations: RecommendationByUser,
    catalog_items: set[ItemId],
) -> float:
    """Calculate catalog coverage from unique recommended items."""

    if not catalog_items:
        raise ValueError("catalog_items must not be empty.")

    recommended_items = {
        item_id
        for recommended in all_recommendations.values()
        for item_id in _deduplicate_ranked_items(recommended)
    }
    valid_recommended_items = recommended_items.intersection(catalog_items)
    return len(valid_recommended_items) / len(catalog_items)


def diversity_at_k(
    recommended: RecommendedItems,
    item_features: ItemVectorById,
    k: int,
) -> float:
    """Calculate mean pairwise cosine dissimilarity for top-k recommendations."""

    top_k = _top_k_items(recommended, k)
    vectors = _valid_vectors_for_items(top_k, item_features)

    if len(vectors) < 2:
        return 0.0

    dissimilarity_sum = 0.0
    pair_count = 0

    for first_index in range(len(vectors)):
        for second_index in range(first_index + 1, len(vectors)):
            similarity = float(np.dot(vectors[first_index], vectors[second_index]))
            dissimilarity_sum += 1.0 - similarity
            pair_count += 1

    if pair_count == 0:
        return 0.0

    return dissimilarity_sum / pair_count


def _rating_to_gain(rating: float | int) -> float:
    rating_value = float(rating)
    if rating_value >= 10:
        return 4.0
    if rating_value >= 9:
        return 3.0
    if rating_value >= 8:
        return 2.0
    if rating_value >= 7:
        return 1.0
    return 0.0


def _discounted_cumulative_gain(gains: Sequence[float]) -> float:
    return sum(
        ((2.0**gain) - 1.0) / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )


def _valid_vector(vector: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None

    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return None

    norm = np.linalg.norm(array)
    if norm == 0.0 or not np.isfinite(norm):
        return None

    return array / norm


def _valid_vectors_for_items(
    item_ids: Sequence[ItemId],
    item_features: ItemVectorById,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    expected_size: int | None = None

    for item_id in item_ids:
        vector = _valid_vector(item_features.get(item_id))
        if vector is None:
            continue

        if expected_size is None:
            expected_size = vector.size
        if vector.size != expected_size:
            continue

        vectors.append(vector)

    return vectors
