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
from cinesense.recommenders.two_stage import CineSenseTwoStage
from evaluation.datasets import (
    ITEM_ID_COL,
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)


RESULT_PATH = Path("evaluation/results/cinesense_two_stage_validation.json")


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

    model = CineSenseTwoStage().fit(catalog, split.train, user_ids=eval_user_ids)
    results = evaluate_model(
        model,
        eval_users_1000,
        use_validation=True,
        relevance_scores_by_user=relevance_scores_from_interactions(split.validation),
    )
    save_results(
        results,
        RESULT_PATH,
        split_seed=split.seed,
        train_ratio=split.train_ratio,
        val_ratio=split.val_ratio,
        test_ratio=split.test_ratio,
    )

    print(json.dumps(results["metrics"], indent=2, sort_keys=True))
    print_comparison_table(results["metrics"])
    print(f"saved results: {RESULT_PATH}")


def print_comparison_table(two_stage_metrics: dict[str, float]) -> None:
    rows = [
        ("Popularity", 0.0481, 0.0803, 0.0745, 0.4520),
        ("Hybrid C", 0.0598, 0.1058, 0.0665, 0.5170),
        ("Weighted B", 0.0547, 0.0795, 0.0907, 0.4920),
        (
            "TwoStage",
            two_stage_metrics["recall@10"],
            two_stage_metrics["recall@20"],
            two_stage_metrics["ndcg@10"],
            two_stage_metrics["hit_rate@10"],
        ),
    ]

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
