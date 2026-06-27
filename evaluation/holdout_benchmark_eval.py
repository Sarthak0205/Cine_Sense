import os
import sys
import numpy as np
import pandas as pd
import json

# Add project root to sys.path
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.audit_common import (
    load_model_c,
    load_model_d,
    write_markdown_table,
    save_report,
    bootstrap_ci,
)
from evaluation.datasets import (
    load_anime_catalog,
    load_user_watches,
    build_positive_interactions,
    filter_users,
    split_user_interactions,
    build_eval_users,
    ITEM_ID_COL,
    USER_ID_COL,
)
from evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    average_precision_at_k,
    precision_at_k,
)

def evaluate_user_recs(service, user, train_df, target_items, target_scores, k=20):
    user_rows = train_df[train_df[USER_ID_COL] == user.user_id]
    ratings = dict(zip(user_rows[ITEM_ID_COL].astype(int), user_rows["score"].astype(float)))
    
    recs = service.recommend(
        anime_ids=list(user.train_items),
        ratings=ratings,
        top_k=k,
        mode="discover",
        user_id=str(user.user_id),
    )
    rec_ids = [r["anime_id"] for r in recs]
    
    # Filter out train items if any got returned (robustness)
    rec_ids = [rid for rid in rec_ids if rid not in user.train_items]
    
    return {
        "recall@10": recall_at_k(rec_ids, target_items, 10),
        "recall@20": recall_at_k(rec_ids, target_items, 20),
        "ndcg@10": ndcg_at_k(rec_ids, target_scores, 10),
        "hit_rate@10": hit_rate_at_k(rec_ids, target_items, 10),
        "map@10": average_precision_at_k(rec_ids, target_items, 10),
        "precision@10": precision_at_k(rec_ids, target_items, 10),
    }

def main():
    print("Running Holdout Benchmark Evaluation...", flush=True)
    
    # Load datasets
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog[ITEM_ID_COL].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)
    eval_users = build_eval_users(split, use_validation=False) # holdout test split
    
    eval_users_1000 = eval_users[:1000]
    
    # Precompute seed item popularity from training split
    popularity_counts = split.train[ITEM_ID_COL].value_counts().to_dict()
    
    # Precompute user average popularities
    user_avg_pops = []
    for user in eval_users_1000:
        pops = [popularity_counts.get(sid, 0) for sid in user.train_items]
        user_avg_pops.append(np.mean(pops) if pops else 0)
        
    # Segment thresholds based on user-profile average popularities
    q25 = np.percentile(user_avg_pops, 25)
    q75 = np.percentile(user_avg_pops, 75)
    
    # Segment users
    segmented_users = {
        "Popular": [],
        "Mid-Tail": [],
        "Long-Tail": [],
        "Cold-Start": []
    }
    
    for idx_u, user in enumerate(eval_users_1000):
        avg_pop = user_avg_pops[idx_u]
        if avg_pop < 5:
            segmented_users["Cold-Start"].append(user)
        elif avg_pop <= q25:
            segmented_users["Long-Tail"].append(user)
        elif avg_pop < q75:
            segmented_users["Mid-Tail"].append(user)
        else:
            segmented_users["Popular"].append(user)
            
    # Print segment distribution
    for seg, users in segmented_users.items():
        print(f"Segment '{seg}': {len(users)} users")
        
    service_c = load_model_c()
    service_d = load_model_d()
    
    # Load test relevance scores
    relevance_scores_by_user = {}
    for user_id, user_rows in split.test.groupby(USER_ID_COL, sort=False):
        relevance_scores_by_user[int(user_id)] = dict(
            zip(
                user_rows[ITEM_ID_COL].astype(int),
                user_rows["score"].astype(int),
                strict=False,
            )
        )
        
    # Evaluate users
    results_c = []
    results_d = []
    
    print("Evaluating Model C and Model D on holdout split...", flush=True)
    for idx, user in enumerate(eval_users_1000):
        target_items = user.test_items
        user_rel_scores = relevance_scores_by_user.get(user.user_id, {})
        target_scores = {
            item_id: user_rel_scores.get(item_id, 7)
            for item_id in target_items
        }
        
        metrics_c = evaluate_user_recs(service_c, user, split.train, target_items, target_scores)
        metrics_d = evaluate_user_recs(service_d, user, split.train, target_items, target_scores)
        
        results_c.append(metrics_c)
        results_d.append(metrics_d)
        
        if (idx + 1) % 200 == 0:
            print(f"Evaluated {idx + 1}/1000 users...", flush=True)
            
    # Compute aggregate metrics
    metric_names = ["recall@10", "recall@20", "ndcg@10", "hit_rate@10", "map@10", "precision@10"]
    
    # Bootstrap analysis
    bootstrap_rows = []
    regressions_count = 0
    
    for m in ["ndcg@10", "recall@10", "recall@20", "precision@10"]:
        c_vals = [r[m] for r in results_c]
        d_vals = [r[m] for r in results_d]
        
        mean_c = np.mean(c_vals)
        mean_d = np.mean(d_vals)
        
        mean_delta, ci_lower, ci_upper = bootstrap_ci(c_vals, d_vals)
        
        abs_delta = mean_d - mean_c
        rel_delta = (abs_delta / mean_c) * 100 if mean_c > 0 else 0.0
        
        bootstrap_rows.append([
            m,
            f"{mean_c:.4f}",
            f"{mean_d:.4f}",
            f"{abs_delta:+.4f}",
            f"{rel_delta:+.2f}%",
            f"[{ci_lower:+.4f}, {ci_upper:+.4f}]"
        ])
        
        # Success check: Model D must not significantly regress
        # A significant regression is when the upper bound of the CI is below 0, indicating a statistically significant decrease.
        if ci_upper < 0:
            regressions_count += 1
            
    # Segmented analysis
    segment_rows = []
    for seg, users in segmented_users.items():
        if not users:
            segment_rows.append([seg, "N/A", "N/A", "N/A", "N/A"])
            continue
            
        # Filter indices of users in this segment
        seg_user_ids = {u.user_id for u in users}
        seg_indices = [i for i, u in enumerate(eval_users_1000) if u.user_id in seg_user_ids]
        
        seg_c = [results_c[i] for i in seg_indices]
        seg_d = [results_d[i] for i in seg_indices]
        
        for m in ["ndcg@10", "recall@10"]:
            mean_c = np.mean([r[m] for r in seg_c])
            mean_d = np.mean([r[m] for r in seg_d])
            abs_delta = mean_d - mean_c
            rel_delta = (abs_delta / mean_c) * 100 if mean_c > 0 else 0.0
            
            segment_rows.append([
                f"{seg} - {m}",
                f"{mean_c:.4f}",
                f"{mean_d:.4f}",
                f"{abs_delta:+.4f}",
                f"{rel_delta:+.2f}%"
            ])
            
            # Segment regression check
            if abs_delta < -0.005: # regression threshold
                regressions_count += 1
                
    # Build report content
    report_content = []
    report_content.append("# CineSense Holdout Benchmark Evaluation Report")
    
    pass_status = "PASS" if regressions_count == 0 else "FAIL"
    report_content.append(f"\n## Audit Summary")
    report_content.append(f"* **Holdout Validation Status**: **{pass_status}**")
    report_content.append(f"* **Regressions Flagged**: {regressions_count}")
    
    report_content.append(f"\n## Statistical Significance and Bootstrap Results (1000 Samples)")
    headers = ["Metric", "Model C", "Model D", "Abs Delta", "Rel Delta", "95% CI of Delta"]
    report_content.append(write_markdown_table(headers, bootstrap_rows))
    
    report_content.append(f"\n## Segmented Performance Analysis")
    seg_headers = ["Segment & Metric", "Model C", "Model D", "Abs Delta", "Rel Delta"]
    report_content.append(write_markdown_table(seg_headers, segment_rows))
    
    report_content.append(f"\n## Overfitting and Recommendation Quality Analysis")
    if regressions_count > 0:
        report_content.append(
            "\n> [!IMPORTANT]"
            "\n> Model D shows statistically significant regressions compared to Model C on holdout validation. "
            "\n> The popularity penalty and cosine scaling parameters in Model D appear to degrade overall recommendations quality on unseen holdout users."
        )
    else:
        report_content.append("\nNo significant regressions detected.")
        
    save_report("holdout_results.md", "\n".join(report_content))

if __name__ == "__main__":
    main()
