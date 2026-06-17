from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from evaluation.benchmark import (
    PopularityBaseline,
    evaluate_model,
    relevance_scores_from_interactions,
)
from evaluation.datasets import (
    ITEM_ID_COL,
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)


def main() -> None:
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog[ITEM_ID_COL].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)
    eval_users = build_eval_users(split)

    model = PopularityBaseline().fit(split.train)
    results = evaluate_model(
        model,
        eval_users,
        relevance_scores_by_user=relevance_scores_from_interactions(split.test),
        max_users=1000,
    )

    print(json.dumps(results["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
