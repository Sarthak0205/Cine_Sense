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
    calculate_gini,
    calculate_entropy,
    extract_studio_and_year,
)
from evaluation.datasets import (
    load_anime_catalog,
    load_user_watches,
    build_positive_interactions,
    filter_users,
    split_user_interactions,
    build_eval_users,
)

def evaluate_diversity(service, eval_users, max_users=1000):
    recommender = service.recommender
    catalog_size = len(recommender.anime_ids)
    
    # Track recommendations
    recommended_counts = np.zeros(catalog_size)
    recommended_items_all = []
    
    user_franchise_counts = []
    recommended_genres = []
    recommended_studios = []
    recommended_years = []
    recommended_popularities = []
    
    np.random.seed(42)
    sampled_users = np.random.choice(eval_users, size=min(len(eval_users), max_users), replace=False)
    
    for idx_u, user in enumerate(sampled_users):
        train_items = list(user.train_items)
        if not train_items:
            continue
            
        ratings = {aid: 10.0 for aid in train_items}
        recs = service.recommend(anime_ids=train_items, ratings=ratings, top_k=10, mode="discover", user_id=str(user.user_id))
        
        franchises_in_list = set()
        for item in recs:
            rid = item["anime_id"]
            if rid not in recommender.item_id_to_index:
                continue
            idx_item = recommender.item_id_to_index[rid]
            recommended_counts[idx_item] += 1
            recommended_items_all.append(rid)
            
            # Fetch metadata
            meta = service.catalog_meta.get(rid, {})
            title = meta.get("title", "")
            synopsis = meta.get("synopsis", "")
            
            # Genres
            recommended_genres.extend(meta.get("genres", []))
            
            # Studio & Year
            studio, year = extract_studio_and_year(synopsis, rid, title)
            recommended_studios.append(studio)
            recommended_years.append(year)
            
            # Popularity
            pop = float(recommender.popularity_scores[idx_item])
            recommended_popularities.append(pop)
            
            # Franchise
            from cinesense.utils.franchise import get_canonical_franchise
            franchises_in_list.add(get_canonical_franchise(title))
            
        user_franchise_counts.append(len(franchises_in_list))
        
        if (idx_u + 1) % 200 == 0:
            print(f"Processed {idx_u + 1}/{max_users} users for diversity...", flush=True)

    # Catalog Coverage
    unique_recommended = np.sum(recommended_counts > 0)
    coverage = (unique_recommended / catalog_size) * 100
    
    # Gini Coefficient
    gini = calculate_gini(recommended_counts)
    
    # Novelty: -mean(log2(pop_clipped))
    pop_clipped = np.clip(recommended_popularities, 1e-6, 1.0)
    novelty = -np.mean(np.log2(pop_clipped))
    
    # Mean Franchise Diversity
    mean_franchise_div = np.mean(user_franchise_counts) if user_franchise_counts else 0.0
    
    # Entropies
    genre_entropy = calculate_entropy(recommended_genres)
    studio_entropy = calculate_entropy(recommended_studios)
    year_entropy = calculate_entropy(recommended_years)
    
    # Popularity Bias & Bucket Distribution
    # Map each recommended item to its percentile
    percentiles = []
    bucket_counts = {
        "Top 1%": 0,
        "Top 10%": 0,
        "Top 25%": 0,
        "Top 50%": 0,
        "Bottom 50%": 0
    }
    
    for rid in recommended_items_all:
        idx_item = recommender.item_id_to_index[rid]
        pct = float(recommender.pop_percentiles[idx_item])
        percentiles.append(pct)
        
        if pct >= 0.99:
            bucket_counts["Top 1%"] += 1
        if pct >= 0.90:
            bucket_counts["Top 10%"] += 1
        if pct >= 0.75:
            bucket_counts["Top 25%"] += 1
        if pct >= 0.50:
            bucket_counts["Top 50%"] += 1
        else:
            bucket_counts["Bottom 50%"] += 1
            
    mean_pct = np.mean(percentiles) * 100 if percentiles else 0.0
    total_recs = len(recommended_items_all)
    
    bucket_distribution = {}
    for bucket, count in bucket_counts.items():
        bucket_distribution[bucket] = (count / total_recs) * 100 if total_recs > 0 else 0.0
        
    # Diversity Risk Flag
    # Flag if Top 1% items dominate (e.g. > 30% of recommendations)
    top_1_pct = bucket_distribution["Top 1%"]
    risk_flagged = top_1_pct > 30.0
    
    return {
        "coverage": coverage,
        "gini": gini,
        "novelty": novelty,
        "franchise_diversity": mean_franchise_div,
        "genre_entropy": genre_entropy,
        "studio_entropy": studio_entropy,
        "year_entropy": year_entropy,
        "popularity_bias": mean_pct,
        "bucket_dist": bucket_distribution,
        "risk_flagged": risk_flagged
    }

def main():
    print("Running Diversity Audit...", flush=True)
    
    # Load dataset split
    catalog = load_anime_catalog()
    user_watches = load_user_watches()
    positives = build_positive_interactions(
        user_watches,
        catalog_item_ids=catalog["anime_id"].unique(),
    )
    filtered_users = filter_users(positives)
    split = split_user_interactions(filtered_users)
    eval_users = build_eval_users(split, use_validation=False)
    
    service_c = load_model_c()
    service_d = load_model_d()
    
    res_c = evaluate_diversity(service_c, eval_users, max_users=1000)
    res_d = evaluate_diversity(service_d, eval_users, max_users=1000)
    
    headers = ["Diversity Metric", "Model C", "Model D"]
    rows = [
        ["Catalog Coverage", f"{res_c['coverage']:.2f}%", f"{res_d['coverage']:.2f}%"],
        ["Gini Coefficient", f"{res_c['gini']:.4f}", f"{res_d['gini']:.4f}健全"],
        ["Novelty", f"{res_c['novelty']:.4f}", f"{res_d['novelty']:.4f}"],
        ["Franchise Diversity", f"{res_c['franchise_diversity']:.2f}", f"{res_d['franchise_diversity']:.2f}"],
        ["Genre Entropy", f"{res_c['genre_entropy']:.4f}", f"{res_d['genre_entropy']:.4f}"],
        ["Studio Entropy", f"{res_c['studio_entropy']:.4f}", f"{res_d['studio_entropy']:.4f}"],
        ["Year Entropy", f"{res_c['year_entropy']:.4f}", f"{res_d['year_entropy']:.4f}"],
        ["Popularity Bias (Mean Pct)", f"{res_c['popularity_bias']:.2f}%", f"{res_d['popularity_bias']:.2f}%"],
    ]
    
    # Clean Gini formatting (remove Japanese character)
    rows[1][1] = f"{res_c['gini']:.4f}"
    rows[1][2] = f"{res_d['gini']:.4f}"
    
    # Buckets formatting
    bucket_rows = []
    for b in ["Top 1%", "Top 10%", "Top 25%", "Top 50%", "Bottom 50%"]:
        bucket_rows.append([b, f"{res_c['bucket_dist'][b]:.2f}%", f"{res_d['bucket_dist'][b]:.2f}%"])
        
    pass_c = "FAIL" if res_c["risk_flagged"] else "PASS"
    pass_d = "FAIL" if res_d["risk_flagged"] else "PASS"
    
    # Build report content
    report_content = []
    report_content.append("# CineSense Diversity Audit Report")
    report_content.append(f"\n## Audit Summary")
    report_content.append(f"* **Model C Diversity Status**: **{pass_c}**")
    report_content.append(f"* **Model D Diversity Status**: **{pass_d}**")
    report_content.append(f"\n## Diversity Metrics")
    report_content.append(write_markdown_table(headers, rows))
    
    report_content.append(f"\n## Popularity Bucket Distribution")
    bucket_headers = ["Popularity Bucket", "Model C Frequency", "Model D Frequency"]
    report_content.append(write_markdown_table(bucket_headers, bucket_rows))
    
    report_content.append(f"\n## Risk & Concentration Analysis")
    if res_d["risk_flagged"]:
        report_content.append(
            "\n> [!CAUTION]"
            "\n> Model D has flagged a high concentration of popular items in recommendations. "
            "\n> The Top 1% items dominate more than 30% of the recommendations list, "
            "\n> which indicates severe popularity bias and a risk of diversity collapse."
        )
    else:
        report_content.append("\nNo severe concentration risk detected.")
        
    save_report("diversity_audit.md", "\n".join(report_content))

if __name__ == "__main__":
    main()
