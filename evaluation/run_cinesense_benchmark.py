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
from cinesense.recommenders.base import CineSenseV1Recommender
from evaluation.datasets import (
    ITEM_ID_COL,
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)


RESULT_PATH = Path("evaluation/results/cinesense_v1_validation.json")


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

    model = CineSenseV1Recommender().fit(catalog)
    results = evaluate_model(
        model,
        eval_users,
        use_validation=True,
        relevance_scores_by_user=relevance_scores_from_interactions(split.validation),
        max_users=1000,
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
    print(f"saved results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
