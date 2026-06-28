import os
import sys
import numpy as np
import pandas as pd
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.audit_common import (
    load_model_c,
    load_model_d,
    write_markdown_table,
    save_report,
)
from evaluation.datasets import (
    load_anime_catalog,
    load_user_watches,
    build_positive_interactions,
    filter_users,
    split_user_interactions,
    build_eval_users,
)

def evaluate_explanation_truthfulness(service, eval_users, max_users=100):
    total_claims = 0
    valid_claims = 0
    
    total_recs = 0
    attribution_correct = 0
    sum_shares_consistent = 0
    
    coverage_1 = 0
    coverage_2 = 0
    coverage_3 = 0
    generic_count = 0
    
    # Configure model params based on service config
    cosine_power = service.rerank_config.cosine_power
    popularity_penalty = service.rerank_config.popularity_penalty
    jaccard_weight = service.rerank_config.jaccard_weight
    distance_weight = service.rerank_config.distance_weight
    
    recommender = service.recommender
    
    # We sample max_users
    np.random.seed(42)
    sampled_users = np.random.choice(eval_users, size=min(len(eval_users), max_users), replace=False)
    
    for user in sampled_users:
        train_items = [aid for aid in user.train_items if aid in recommender.item_id_to_index]
        if not train_items:
            continue
            
        # Get recommendations
        # We need to construct ratings
        ratings = {aid: 10.0 for aid in train_items} # default rating
        recs = service.recommend(anime_ids=train_items, ratings=ratings, top_k=10, mode="discover", user_id=str(user.user_id))
        
        for item in recs:
            total_recs += 1
            rec_id = item["anime_id"]
            explanation = item.get("explanation", {})
            if not explanation:
                continue
                
            # Check reasons coverage
            reasons = explanation.get("reasons", [])
            n_reasons = len(reasons)
            if n_reasons >= 1:
                coverage_1 += 1
            if n_reasons >= 2:
                coverage_2 += 1
            if n_reasons >= 3:
                coverage_3 += 1
                
            # Check generic vs specific
            # A reason is generic if it only claims semantic similarity
            primary_reason = explanation.get("reason", "")
            is_generic = "semantic similarity" in primary_reason.lower() and not any(
                x in primary_reason.lower() for x in ["watched by", "fans", "themes", "elements", "patterns"]
            )
            if is_generic:
                generic_count += 1
                
            # Verify Claims Truthfulness
            # Each explanation claims some reasons. Let's validate each reason in `reasons`.
            for r_text in reasons:
                r_low = r_text.lower()
                
                # Claims validation
                is_valid = True
                
                # 1. Semantic Claims
                if "semantic similarity" in r_low:
                    # Look up best seed and check cosine similarity
                    matched_seed_id = explanation.get("matched_seed", {}).get("anime_id")
                    if matched_seed_id:
                        idx_rec = recommender.item_id_to_index[rec_id]
                        idx_seed = recommender.item_id_to_index[matched_seed_id]
                        cosine_sim = float(np.dot(recommender.catalog_embeddings[idx_rec], recommender.catalog_embeddings[idx_seed]))
                        
                        total_claims += 1
                        if "strong semantic similarity" in r_low:
                            if cosine_sim < 0.75:
                                is_valid = False
                        elif "high semantic similarity" in r_low:
                            if cosine_sim < 0.60:
                                is_valid = False
                        # otherwise normal semantic similarity has no lower bound
                        if is_valid:
                            valid_claims += 1
                            
                # 2. Collaborative Claims
                elif "frequently watched by" in r_low and "fans" in r_low:
                    # Find which seed it references
                    # Parse seed title from the reason text, or check Jaccard similarity for all seeds
                    # Verify if Jaccard similarity to any seed mentioned in the reasons is >= 0.05
                    total_claims += 1
                    has_matching_jaccard = False
                    for s_id in train_items:
                        jac = service._lookup_jaccard(s_id, rec_id)
                        if jac >= 0.05:
                            s_title = service.catalog_meta.get(s_id, {}).get("title", "").lower()
                            if s_title and s_title in r_low:
                                has_matching_jaccard = True
                                break
                    if has_matching_jaccard:
                        valid_claims += 1
                    else:
                        is_valid = False
                        
                # 3. Similar User / Graph Claims
                elif "collaborative relevance" in r_low or "enjoy" in r_low or "patterns" in r_low:
                    total_claims += 1
                    has_matching_distance = False
                    
                    if "both" in r_low:
                        # Multi-seed reason. Both seeds must have distance <= 2
                        valid_seeds = []
                        for s_id in train_items:
                            dist = service._lookup_distance(s_id, rec_id)
                            if dist <= 2:
                                s_title = service.catalog_meta.get(s_id, {}).get("title", "").lower()
                                if s_title and s_title in r_low:
                                    valid_seeds.append(s_id)
                        if len(valid_seeds) >= 2:
                            has_matching_distance = True
                    else:
                        for s_id in train_items:
                            dist = service._lookup_distance(s_id, rec_id)
                            s_title = service.catalog_meta.get(s_id, {}).get("title", "").lower()
                            if s_title and s_title in r_low:
                                if "high collaborative relevance" in r_low and dist == 1:
                                    has_matching_distance = True
                                    break
                                elif "watched by users" in r_low and dist == 2:
                                    has_matching_distance = True
                                    break
                                elif dist <= 2:
                                    has_matching_distance = True
                                    break
                                    
                    if has_matching_distance:
                        valid_claims += 1
                    else:
                        is_valid = False
                        
                # 4. Genre Claims
                elif "themes" in r_low or "elements" in r_low:
                    total_claims += 1
                    # Extract genres mentioned (e.g. Action and Thriller)
                    cand_genres = service.catalog_meta.get(rec_id, {}).get("genres", [])
                    cand_genres_low = [g.lower() for g in cand_genres]
                    
                    # Verify at least one mentioned genre matches candidate genres
                    has_matching_genre = False
                    for g in cand_genres_low:
                        if g in r_low:
                            has_matching_genre = True
                            break
                    if has_matching_genre:
                        valid_claims += 1
                    else:
                        is_valid = False
            
            # Attribution Accuracy
            # Winning seed according to model contribution vs matched_seed
            best_seed_id = None
            max_contrib = -float('inf')
            
            rec_idx = recommender.item_id_to_index[rec_id]
            emb_rec = recommender.catalog_embeddings[rec_idx]
            
            for s_id in train_items:
                s_idx = recommender.item_id_to_index[s_id]
                emb_seed = recommender.catalog_embeddings[s_idx]
                sim = float(np.dot(emb_rec, emb_seed))
                
                jac = service._lookup_jaccard(s_id, rec_id)
                dist = service._lookup_distance(s_id, rec_id)
                dist_score = 0.5 if dist == 1 else (1.0 / 3.0) if dist == 2 else 0.0
                
                # Model contrib formula
                contrib = sim * recommender.semantic_weight + jaccard_weight * jac * (sim ** cosine_power) + distance_weight * dist_score
                if contrib > max_contrib:
                    max_contrib = contrib
                    best_seed_id = s_id
                    
            matched_seed_id = explanation.get("matched_seed", {}).get("anime_id")
            if matched_seed_id == best_seed_id:
                attribution_correct += 1
                
            # Contribution Consistency
            # Sum of seed shares must be 100% +- 1%
            shares = explanation.get("seed_shares", {})
            sum_shares = sum(float(v.get("share", 0.0)) for v in shares.values())
            if abs(sum_shares - 1.0) <= 0.01:
                sum_shares_consistent += 1
                
    precision = (valid_claims / total_claims) * 100 if total_claims > 0 else 100.0
    attrib_acc = (attribution_correct / total_recs) * 100 if total_recs > 0 else 100.0
    consistency = (sum_shares_consistent / total_recs) * 100 if total_recs > 0 else 100.0
    generic_rate = (generic_count / total_recs) * 100 if total_recs > 0 else 0.0
    
    cov_1 = (coverage_1 / total_recs) * 100 if total_recs > 0 else 0.0
    cov_2 = (coverage_2 / total_recs) * 100 if total_recs > 0 else 0.0
    cov_3 = (coverage_3 / total_recs) * 100 if total_recs > 0 else 0.0
    
    return {
        "precision": precision,
        "attribution_accuracy": attrib_acc,
        "consistency": consistency,
        "generic_rate": generic_rate,
        "coverage_1": cov_1,
        "coverage_2": cov_2,
        "coverage_3": cov_3,
    }

def main():
    print("Running Explanation Truthfulness Audit...", flush=True)
    
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
    
    res_c = evaluate_explanation_truthfulness(service_c, eval_users, max_users=100)
    res_d = evaluate_explanation_truthfulness(service_d, eval_users, max_users=100)
    
    # Table headers and rows
    headers = ["Metric", "Model C", "Model D"]
    rows = [
        ["Explanation Precision", f"{res_c['precision']:.2f}%", f"{res_d['precision']:.2f}%"],
        ["Attribution Accuracy", f"{res_c['attribution_accuracy']:.2f}%", f"{res_d['attribution_accuracy']:.2f}%"],
        ["Consistency (Shares Sum)", f"{res_c['consistency']:.2f}%", f"{res_d['consistency']:.2f}%"],
        ["Generic Explanation Rate", f"{res_c['generic_rate']:.2f}%", f"{res_d['generic_rate']:.2f}%"],
        ["Explanation Coverage >= 1", f"{res_c['coverage_1']:.2f}%", f"{res_d['coverage_1']:.2f}%"],
        ["Explanation Coverage >= 2", f"{res_c['coverage_2']:.2f}%", f"{res_d['coverage_2']:.2f}%"],
        ["Explanation Coverage >= 3", f"{res_c['coverage_3']:.2f}%", f"{res_d['coverage_3']:.2f}%"],
    ]
    
    # Check gates
    # Gate 2: Explanation Precision >= 95%
    # Gate 3: Generic Explanation Rate < 30%
    pass_c = "PASS" if res_c["precision"] >= 95.0 and res_c["generic_rate"] < 30.0 else "FAIL"
    pass_d = "PASS" if res_d["precision"] >= 95.0 and res_d["generic_rate"] < 30.0 else "FAIL"
    
    # Build report content
    report_content = []
    report_content.append("# CineSense Explanation Truthfulness Audit Report")
    report_content.append(f"\n## Audit Summary")
    report_content.append(f"* **Model C Status**: **{pass_c}**")
    report_content.append(f"* **Model D Status**: **{pass_d}**")
    report_content.append(f"\n## Performance Metrics")
    report_content.append(write_markdown_table(headers, rows))
    
    report_content.append(f"\n## Key Findings")
    
    # Explanation attribution accuracy analysis
    report_content.append(f"\n### 1. Attribution Accuracy Regression")
    report_content.append(f"* Model C Attribution Accuracy: {res_c['attribution_accuracy']:.2f}%")
    report_content.append(f"* Model D Attribution Accuracy: {res_d['attribution_accuracy']:.2f}%")
    
    if res_d["attribution_accuracy"] < res_c["attribution_accuracy"]:
        report_content.append(
            "\n> [!WARNING]"
            "\n> Model D shows a regression in Attribution Accuracy. "
            "\n> This is due to a mismatch between Model D's scoring formula (which weights Jaccard by `cosine_sim ** 2` and subtracts a popularity penalty) "
            "\n> and the hardcoded explanation relevance formula: `relevance = sim + 1.0 * jac + 0.3 * dist_score`. "
            "\n> As a result, the explanation selected a `matched_seed` that was not the seed that actually drove the recommendation score."
        )
    else:
        report_content.append("\nNo regression detected.")
        
    save_report("explanation_audit.md", "\n".join(report_content))

if __name__ == "__main__":
    main()
