from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from evaluation.benchmark import evaluate_model, relevance_scores_from_interactions, save_results
from cinesense.recommenders.base import CineSenseHybridLiteWeighted
from evaluation.datasets import (
    ITEM_ID_COL,
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)


RESULTS_DIR = Path("evaluation/results")
SCHEMES = {
    "Weighted A": "raw_score",
    "Weighted B": "normalized",
    "Weighted C": "strong",
}


def main() -> None:
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog[ITEM_ID_COL].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)
    eval_users = build_eval_users(split, use_validation=True)
    eval_users_1000 = eval_users[:1000]
    eval_user_ids = [user.user_id for user in eval_users_1000]
    relevance_scores_by_user = relevance_scores_from_interactions(split.validation)

    results_by_scheme = {}
    model = CineSenseHybridLiteWeighted(
        semantic_weight=0.85,
        popularity_weight=0.15,
        rating_weight_scheme="raw_score",
    ).fit(catalog, split.train, user_ids=eval_user_ids)

    for label, scheme in SCHEMES.items():
        model.model_name = f"cinesense_hybrid_lite_weighted_{scheme}"
        model.rating_weight_scheme = scheme
        model.user_item_weights = model._build_user_item_weights(split.train, eval_user_ids)

        results = evaluate_model(
            model,
            eval_users_1000,
            use_validation=True,
            relevance_scores_by_user=relevance_scores_by_user,
        )
        result_path = RESULTS_DIR / f"cinesense_hybrid_lite_weighted_{scheme}_validation.json"
        save_results(
            results,
            result_path,
            split_seed=split.seed,
            train_ratio=split.train_ratio,
            val_ratio=split.val_ratio,
            test_ratio=split.test_ratio,
        )
        results_by_scheme[label] = results["metrics"]

    print(json.dumps(results_by_scheme, indent=2, sort_keys=True))
    print_comparison_table(results_by_scheme)


def print_comparison_table(results_by_scheme: dict[str, dict[str, float]]) -> None:
    rows = [
        ("Hybrid C", 0.0598, 0.1058, 0.0665, 0.5170),
    ]
    for label in ["Weighted A", "Weighted B", "Weighted C"]:
        metrics = results_by_scheme[label]
        rows.append(
            (
                label,
                metrics["recall@10"],
                metrics["recall@20"],
                metrics["ndcg@10"],
                metrics["hit_rate@10"],
            )
        )

    print()
    print(f"{'Model':<18} {'Recall@10':>10} {'Recall@20':>10} {'NDCG@10':>10} {'HitRate@10':>11}")
    for model_name, recall_10, recall_20, ndcg_10, hit_rate_10 in rows:
        print(
            f"{model_name:<18} "
            f"{recall_10:>10.4f} "
            f"{recall_20:>10.4f} "
            f"{ndcg_10:>10.4f} "
            f"{hit_rate_10:>11.4f}"
        )


if __name__ == "__main__":
    main()
