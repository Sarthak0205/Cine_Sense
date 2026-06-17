import os
import random
import tempfile
import numpy as np
import pandas as pd


from cinesense.recommenders.two_stage import CineSenseTwoStage
from cinesense.utils.model_storage import save_model, load_model
from cinesense.utils.text import ITEM_ID_COL
from evaluation.datasets import (
    build_eval_users,
    build_positive_interactions,
    filter_users,
    load_anime_catalog,
    load_user_watches,
    split_user_interactions,
)
from evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    average_precision_at_k,
    precision_at_k,
)


def load_dataset_split():
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog[ITEM_ID_COL].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)
    return catalog, split


def test_parquet_catalog_dtypes():
    """Verify that catalog serialization via parquet preserves pandas dtypes."""
    catalog, split = load_dataset_split()
    model = CineSenseTwoStage().fit(catalog, split.train)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_model(
            model=model,
            catalog_df=model.catalog,
            dir_path=temp_dir,
            model_version="test_v1",
            catalog_version="test_cat",
            embedding_version="test_emb",
        )

        loaded_model, loaded_catalog_df, metadata = load_model(temp_dir)

        # Check shape equivalence
        assert loaded_catalog_df.shape[0] == model.catalog.shape[0]

        # Verify dtypes are exactly identical
        for col in model.catalog.columns:
            assert col in loaded_catalog_df.columns
            # Parquet should preserve nullable/string types accurately
            assert loaded_catalog_df[col].dtype == model.catalog[col].dtype

        print("\n[PASSED] test_parquet_catalog_dtypes: Catalog data types preserved exactly.")


def test_model_export_import_roundtrip():
    """Verify recommendations before export match recommendations after import exactly."""
    catalog, split = load_dataset_split()

    # Fit a small model/subset to speed up or fit on the users
    eval_users = build_eval_users(split, use_validation=True)
    user_samples = eval_users[:5]
    user_ids = [u.user_id for u in user_samples]

    model = CineSenseTwoStage().fit(catalog, split.train, user_ids=user_ids)

    # Collect recommendations before export
    recs_before = {}
    for user in user_samples:
        train_items = user.train_items
        user_rows = split.train[split.train["user_id"] == user.user_id]
        ratings = dict(zip(user_rows["anime_id"], user_rows["score"]))
        recs_before[user.user_id] = model.recommend(
            anime_ids=list(train_items),
            ratings=ratings,
            top_k=10,
        )

    # Roundtrip save & load
    with tempfile.TemporaryDirectory() as temp_dir:
        save_model(
            model=model,
            catalog_df=model.catalog,
            dir_path=temp_dir,
            model_version="roundtrip_v1",
            catalog_version="rt_cat",
            embedding_version="rt_emb",
        )

        loaded_model, _, _ = load_model(temp_dir)

        # Collect recommendations after import
        for user in user_samples:
            train_items = user.train_items
            user_rows = split.train[split.train["user_id"] == user.user_id]
            ratings = dict(zip(user_rows["anime_id"], user_rows["score"]))
            recs_after = loaded_model.recommend(
                anime_ids=list(train_items),
                ratings=ratings,
                top_k=10,
            )
            # Verify exact match
            assert recs_before[user.user_id] == recs_after
            assert len(recs_after) == 10

    print("[PASSED] test_model_export_import_roundtrip: Recommendations before export and after import are identical.")


def test_1000_users_equivalence_and_metrics():
    """Verify 100% equivalence in lists and metrics over 1000 random users."""
    catalog, split = load_dataset_split()

    eval_users = build_eval_users(split, use_validation=True)
    print(f"\nTotal eligible eval users: {len(eval_users)}")

    # Sample 1000 users deterministically
    random.seed(42)
    sampled_users = random.sample(eval_users, 1000)
    sampled_user_ids = [u.user_id for u in sampled_users]

    print("Fitting model...")
    model = CineSenseTwoStage().fit(catalog, split.train, user_ids=sampled_user_ids)

    # Dictionary mappings for relevance score lookup
    relevance_scores_by_user = {}
    for user_id, user_rows in split.validation.groupby("user_id"):
        if int(user_id) in sampled_user_ids:
            relevance_scores_by_user[int(user_id)] = dict(
                zip(user_rows["anime_id"].astype(int), user_rows["score"].astype(int))
            )

    eval_metrics = []
    prod_metrics = []

    print("Running equivalence validation on 1000 users...")
    for idx, user in enumerate(sampled_users):
        user_id = user.user_id
        train_items = user.train_items
        target_items = user.validation_items

        # 1. Stateful recommendation
        eval_recs = model.recommend_for_user(user_id, train_items, k=20)

        # 2. Stateless recommendation (with ratings dictionary)
        user_rows = split.train[split.train["user_id"] == user_id]
        ratings = dict(zip(user_rows["anime_id"].astype(int), user_rows["score"].astype(int)))
        prod_recs = model.recommend(list(train_items), ratings=ratings, top_k=20)

        # Verify exact match (IDs and order)
        assert eval_recs == prod_recs, f"Mismatched recommendations for user {user_id}"

        # Get relevance scores
        user_rel_scores = relevance_scores_by_user.get(user_id, {})
        target_scores = {
            item_id: user_rel_scores.get(item_id, 7) for item_id in target_items
        }

        # Calculate metrics for eval route
        eval_metrics.append({
            "recall@10": recall_at_k(eval_recs, target_items, 10),
            "recall@20": recall_at_k(eval_recs, target_items, 20),
            "ndcg@10": ndcg_at_k(eval_recs, target_scores, 10),
            "hit_rate@10": hit_rate_at_k(eval_recs, target_items, 10),
            "map@10": average_precision_at_k(eval_recs, target_items, 10),
            "precision@10": precision_at_k(eval_recs, target_items, 10),
        })

        # Calculate metrics for prod route
        prod_metrics.append({
            "recall@10": recall_at_k(prod_recs, target_items, 10),
            "recall@20": recall_at_k(prod_recs, target_items, 20),
            "ndcg@10": ndcg_at_k(prod_recs, target_scores, 10),
            "hit_rate@10": hit_rate_at_k(prod_recs, target_items, 10),
            "map@10": average_precision_at_k(prod_recs, target_items, 10),
            "precision@10": precision_at_k(prod_recs, target_items, 10),
        })

    # Assert that all metric lists are exactly identical
    assert eval_metrics == prod_metrics, "Per-user computed metrics differed between eval and prod pathways!"

    # Compute macro-average aggregates
    metric_names = ["recall@10", "recall@20", "ndcg@10", "hit_rate@10", "map@10", "precision@10"]
    avg_eval = {m: float(np.mean([x[m] for x in eval_metrics])) for m in metric_names}
    avg_prod = {m: float(np.mean([x[m] for x in prod_metrics])) for m in metric_names}

    print("\nBenchmark Equivalence Validation Results (1000 Users):")
    print(f"{'Metric':<15} | {'Evaluation Mode':<17} | {'Production Mode':<17} | {'Difference':<10}")
    print("-" * 70)
    for m in metric_names:
        diff = abs(avg_eval[m] - avg_prod[m])
        print(f"{m:<15} | {avg_eval[m]:<17.6f} | {avg_prod[m]:<17.6f} | {diff:<10.6f}")
        assert diff == 0.0, f"Benchmark discrepancy detected on metric {m}: diff={diff}"

    print("\n[PASSED] test_1000_users_equivalence_and_metrics: 100% equivalence verified.")


if __name__ == "__main__":
    # If run directly, run all test functions
    test_parquet_catalog_dtypes()
    test_model_export_import_roundtrip()
    test_1000_users_equivalence_and_metrics()
