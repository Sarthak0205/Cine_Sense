from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from evaluation.datasets import EvalUser, ITEM_ID_COL, SCORE_COL, USER_ID_COL
from evaluation.metrics import (
    ItemId,
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


DEFAULT_MODEL_NAME = "popularity_baseline"
DEFAULT_DATASET_VERSION = "anime_user_watches_v1"
DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_RECOMMENDATION_COUNT = 20


class Recommender(Protocol):
    """Minimal recommender interface required by the benchmark."""

    model_name: str

    def recommend(self, user_id: int, train_items: set[int], k: int) -> Sequence[ItemId]:
        """Return ranked item recommendations for one user."""


@dataclass
class PopularityBaseline:
    """Global popularity recommender fitted from training interactions only."""

    model_name: str = DEFAULT_MODEL_NAME
    item_ranking: list[int] = field(default_factory=list)

    def fit(self, train_interactions: pd.DataFrame) -> "PopularityBaseline":
        """Compute a deterministic global popularity ranking from train interactions."""

        item_counts = train_interactions[ITEM_ID_COL].value_counts(sort=False)
        ranking_frame = item_counts.rename("interaction_count").reset_index()
        ranking_frame.columns = [ITEM_ID_COL, "interaction_count"]
        ranking_frame.sort_values(
            ["interaction_count", ITEM_ID_COL],
            ascending=[False, True],
            inplace=True,
            kind="mergesort",
        )
        self.item_ranking = ranking_frame[ITEM_ID_COL].astype(int).tolist()
        return self

    def recommend(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Return globally popular items, excluding items already in the user's train set."""

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        excluded_items = set(train_items)
        recommendations: list[int] = []

        for item_id in self.item_ranking:
            if item_id in excluded_items:
                continue

            recommendations.append(item_id)
            if len(recommendations) == k:
                break

        return recommendations


def evaluate_user(
    user: EvalUser,
    recommender: Recommender,
    *,
    use_validation: bool = False,
    relevance_scores: Mapping[ItemId, float | int] | None = None,
    recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
) -> dict[str, float | int]:
    """Evaluate one user against validation or test targets."""

    target_items = user.validation_items if use_validation else user.test_items
    if not target_items:
        return _empty_user_result(user.user_id)

    if hasattr(recommender, "recommend_for_user"):
        raw_recommendations = recommender.recommend_for_user(user.user_id, user.train_items, recommendation_count)
    else:
        raw_recommendations = recommender.recommend(user.user_id, user.train_items, recommendation_count)
    recommendations = _filter_train_items(raw_recommendations, user.train_items)
    target_scores = _target_relevance_scores(target_items, relevance_scores)

    return {
        "user_id": user.user_id,
        "recall@10": recall_at_k(recommendations, target_items, 10),
        "recall@20": recall_at_k(recommendations, target_items, 20),
        "ndcg@10": ndcg_at_k(recommendations, target_scores, 10),
        "hit_rate@10": hit_rate_at_k(recommendations, target_items, 10),
        "map@10": average_precision_at_k(recommendations, target_items, 10),
        "precision@10": precision_at_k(recommendations, target_items, 10),
    }


def evaluate_model(
    recommender: Recommender,
    eval_users: Sequence[EvalUser],
    *,
    use_validation: bool = False,
    relevance_scores_by_user: Mapping[int, Mapping[ItemId, float | int]] | None = None,
    max_users: int | None = None,
    recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
) -> dict[str, object]:
    """Evaluate a recommender over users and return per-user and aggregate metrics."""

    selected_users = eval_users[:max_users] if max_users is not None else eval_users
    per_user_results: list[dict[str, float | int]] = []

    for user in selected_users:
        target_items = user.validation_items if use_validation else user.test_items
        if not target_items:
            continue

        user_relevance_scores = (
            relevance_scores_by_user.get(user.user_id)
            if relevance_scores_by_user is not None
            else None
        )
        per_user_results.append(
            evaluate_user(
                user,
                recommender,
                use_validation=use_validation,
                relevance_scores=user_relevance_scores,
                recommendation_count=recommendation_count,
            )
        )

    return {
        "model_name": recommender.model_name,
        "evaluation_target": "validation" if use_validation else "test",
        "evaluated_users": len(per_user_results),
        "metrics": aggregate_metrics(per_user_results),
        "per_user": per_user_results,
    }


def aggregate_metrics(per_user_results: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    """Macro-average benchmark metrics across evaluated users."""

    metric_names = ["recall@10", "recall@20", "ndcg@10", "hit_rate@10", "map@10", "precision@10"]
    if not per_user_results:
        return {metric_name: 0.0 for metric_name in metric_names}

    return {
        metric_name: float(np.mean([float(result[metric_name]) for result in per_user_results]))
        for metric_name in metric_names
    }


def save_results(
    results: Mapping[str, object],
    output_path: str | Path,
    *,
    model_name: str | None = None,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    split_seed: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
) -> None:
    """Save benchmark results as deterministic, human-readable JSON."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": model_name or results.get("model_name"),
        "dataset_version": dataset_version,
        "split_seed": split_seed,
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "evaluation_target": results.get("evaluation_target"),
        "evaluated_users": results.get("evaluated_users"),
        "metrics": results.get("metrics", {}),
    }

    output_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def relevance_scores_from_interactions(interactions: pd.DataFrame) -> dict[int, dict[int, int]]:
    """Build user-item score mappings for graded NDCG from validation or test interactions."""

    relevance_scores: dict[int, dict[int, int]] = {}
    for user_id, user_rows in interactions.groupby(USER_ID_COL, sort=False):
        relevance_scores[int(user_id)] = dict(
            zip(
                user_rows[ITEM_ID_COL].astype(int),
                user_rows[SCORE_COL].astype(int),
                strict=False,
            )
        )
    return relevance_scores


def _filter_train_items(recommendations: Sequence[ItemId], train_items: set[int]) -> list[ItemId]:
    excluded_items = set(train_items)
    return [item_id for item_id in recommendations if item_id not in excluded_items]


def _target_relevance_scores(
    target_items: set[int],
    relevance_scores: Mapping[ItemId, float | int] | None,
) -> dict[ItemId, float | int]:
    if relevance_scores is None:
        return {item_id: 7 for item_id in target_items}

    return {
        item_id: relevance_scores[item_id]
        for item_id in target_items
        if item_id in relevance_scores
    }


def _empty_user_result(user_id: int) -> dict[str, float | int]:
    return {
        "user_id": user_id,
        "recall@10": 0.0,
        "recall@20": 0.0,
        "ndcg@10": 0.0,
        "hit_rate@10": 0.0,
        "map@10": 0.0,
        "precision@10": 0.0,
    }
