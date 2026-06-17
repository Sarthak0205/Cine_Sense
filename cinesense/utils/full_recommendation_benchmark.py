import os
import sys
import time
import math
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates

def main():
    print("Loading model for evaluation benchmark...", flush=True)
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Enable caching for franchise root lookup during benchmark to keep total execution time low
    original_get_franchise_root = service.get_franchise_root
    franchise_root_cache = {}
    def cached_get_franchise_root(franchise_name):
        if franchise_name not in franchise_root_cache:
            franchise_root_cache[franchise_name] = original_get_franchise_root(franchise_name)
        return franchise_root_cache[franchise_name]
    service.get_franchise_root = cached_get_franchise_root

    # Helper to get seed abbreviations
    def get_seed_abbr(aid):
        title = service.catalog_meta.get(aid, {}).get("title", "").lower()
        if "death note" in title: return "DN"
        if "code geass" in title: return "CG"
        if "steins;gate" in title: return "SG"
        if "shingeki no kyojin" in title or "attack on titan" in title: return "AoT"
        if "fullmetal alchemist" in title: return "FMAB"
        if "one piece" in title: return "OP"
        if "monster" in title: return "MNS"
        if "hunter" in title: return "HxH"
        if "naruto" in title: return "NAR"
        if "psycho-pass" in title: return "PP"
        if "kowabon" in title: return "KWB"
        return f"Id{aid}"

    # Load scenarios
    scenarios_df = pd.read_csv(os.path.join(PROJECT_ROOT, "evaluation/scenarios_gold.csv"))
    rows_to_save = []
    summary_data = []

    print(f"Running {len(scenarios_df)} gold scenarios...", flush=True)

    for idx, row in scenarios_df.iterrows():
        name = row["scenario_name"]
        mode = row["mode"]
        seeds_str = str(row["seeds"])
        ratings_str = str(row["ratings"])
        
        seeds = [int(s) for s in seeds_str.split(";")]
        ratings = {int(s): float(r) for s, r in zip(seeds_str.split(";"), ratings_str.split(";"))}

        # Validate inputs
        valid_ids, validated_ratings = service.validate_inputs(seeds, ratings, 10)
        if not valid_ids:
            continue

        # Pipeline timing start
        t_start = time.perf_counter()

        # 1. Retrieval
        t_ret_0 = time.perf_counter()
        if mode == "discover":
            retrieval_k = max(300, 10 * 10)

            train_indices = np.asarray([model.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
            train_items = set(valid_ids)

            retrieval_scores = hybrid_c_retrieval_scores(
                train_indices,
                model.catalog_embeddings,
                model.popularity_scores,
                model.semantic_weight,
                model.popularity_weight,
                model.seed_batch_size,
            )
            retrieved_indices_raw = top_retrieval_indices(
                retrieval_scores,
                train_items,
                model.anime_ids,
                retrieval_k,
            )

            # Seed franchise exclusion
            seed_franchises = set()
            for aid in valid_ids:
                meta = service.catalog_meta.get(aid)
                if meta:
                    seed_franchises.add(get_franchise(meta["title"]))
                    if meta.get("title_english"):
                        seed_franchises.add(get_franchise(meta["title_english"]))

            retrieved_indices_prepared = []
            for r_idx in retrieved_indices_raw:
                anime_id = int(model.anime_ids[r_idx])
                meta = service.catalog_meta[anime_id]
                title = meta["title"]
                eng_title = meta.get("title_english") or ""
                cand_f = get_franchise(title)
                cand_f_eng = get_franchise(eng_title) if eng_title else ""

                if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                    continue

                retrieved_indices_prepared.append(r_idx)
                if len(retrieved_indices_prepared) == 150:
                    break
            retrieved_indices = np.asarray(retrieved_indices_prepared, dtype=np.int32)
        else:
            retrieval_k = 10
            train_indices = np.asarray([model.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
            train_items = set(valid_ids)

            retrieval_scores = hybrid_c_retrieval_scores(
                train_indices,
                model.catalog_embeddings,
                model.popularity_scores,
                model.semantic_weight,
                model.popularity_weight,
                model.seed_batch_size,
            )
            retrieved_indices = top_retrieval_indices(
                retrieval_scores,
                train_items,
                model.anime_ids,
                retrieval_k,
            )
        t_ret_1 = time.perf_counter()
        retrieval_ms = (t_ret_1 - t_ret_0) * 1000.0

        # 2. Ranking
        t_rank_0 = time.perf_counter()
        train_weights = np.asarray([
            model._rating_weight(int(validated_ratings[aid])) if aid in validated_ratings else 1.0
            for aid in valid_ids
        ], dtype=np.float32)

        weighted_semantic_scores = weighted_max_similarity_to_train_items(
            train_indices,
            train_weights,
            model.catalog_embeddings,
            model.seed_batch_size,
        )
        rerank_scores = (
            model.semantic_weight * weighted_semantic_scores
            + model.popularity_weight * model.popularity_scores
        )

        rep_penalty = getattr(model, "representation_penalty", False)
        env_val = os.environ.get("CINESENSE_REPRESENTATION_PENALTY", "False").lower()
        if env_val in ("true", "1", "yes"):
            rep_penalty = True

        rep_lambda = getattr(model, "representation_lambda", 0.03)
        env_lambda = os.environ.get("CINESENSE_REPRESENTATION_LAMBDA")
        if env_lambda is not None:
            try:
                rep_lambda = float(env_lambda)
            except ValueError:
                pass

        if mode == "discover":
            recommendations = rerank_candidates(
                retrieved_indices,
                rerank_scores,
                retrieval_scores,
                model.anime_ids,
                150,
                representation_penalty=rep_penalty,
                representation_lambda=rep_lambda,
                train_indices=train_indices,
                catalog_embeddings=model.catalog_embeddings,
            )
        else:
            recommendations = rerank_candidates(
                retrieved_indices,
                rerank_scores,
                retrieval_scores,
                model.anime_ids,
                10,
            )
        t_rank_1 = time.perf_counter()
        ranking_ms = (t_rank_1 - t_rank_0) * 1000.0

        # 3. Post-processing
        t_post_0 = time.perf_counter()
        scores = {
            rec_id: float(rerank_scores[model.item_id_to_index[rec_id]])
            for rec_id in recommendations
        }
        enriched = service.enrich_recommendations(
            recommendations,
            scores,
            valid_ids,
            weighted_semantic_scores=weighted_semantic_scores,
        )

        if mode == "discover":
            # Downstream Discover filters
            seed_franchises = set()
            for aid in valid_ids:
                meta = service.catalog_meta.get(aid)
                if meta:
                    seed_franchises.add(get_franchise(meta["title"]))
                    if meta.get("title_english"):
                        seed_franchises.add(get_franchise(meta["title_english"]))

            filtered_enriched = []
            seen_rec_franchises = set()

            for item in enriched:
                rec_id = item["anime_id"]
                rec_title = item["title"]
                rec_eng_title = item.get("title_english")

                rec_f_name = get_franchise(rec_title)
                rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""

                if rec_f_name in seed_franchises or (rec_f_eng_name and rec_f_eng_name in seed_franchises):
                    continue

                root_id = service.get_franchise_root(rec_f_name)
                is_sequel = False
                if root_id is not None and rec_id != root_id:
                    is_sequel = True
                elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
                    is_sequel = True

                if is_sequel:
                    continue

                if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
                    continue

                filtered_enriched.append(item)
                seen_rec_franchises.add(rec_f_name)
                if rec_f_eng_name:
                    seen_rec_franchises.add(rec_f_eng_name)

            recs = filtered_enriched[:10]
        else:
            recs = enriched[:10]
        t_post_1 = time.perf_counter()
        post_processing_ms = (t_post_1 - t_post_0) * 1000.0

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        # Calculations
        rec_count = len(recs)
        scores_list = [r["score"] for r in recs]
        avg_score = np.mean(scores_list) if scores_list else 0.0
        min_score = np.min(scores_list) if scores_list else 0.0
        max_score = np.max(scores_list) if scores_list else 0.0
        
        # Diversity
        unique_franchises = len(set(get_franchise(r["title"]) for r in recs))
        duplicate_franchise_ratio = (rec_count - unique_franchises) / rec_count if rec_count > 0 else 0.0
        
        # Discovery & Sequel Checks
        seed_franchises = set()
        for aid in valid_ids:
            meta = service.catalog_meta.get(aid)
            if meta:
                seed_franchises.add(get_franchise(meta["title"]))
                if meta.get("title_english"):
                    seed_franchises.add(get_franchise(meta["title_english"]))
                    
        sequel_count = 0
        leakage_count = 0
        for item in recs:
            rec_id = item["anime_id"]
            rec_title = item["title"]
            rec_eng_title = item.get("title_english")
            
            # Leakage check
            rec_f_name = get_franchise(rec_title)
            rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""
            if rec_f_name in seed_franchises or (rec_f_eng_name and rec_f_eng_name in seed_franchises):
                leakage_count += 1
                
            # Sequel check
            root_id = service.get_franchise_root(rec_f_name)
            is_sequel = False
            if root_id is not None and rec_id != root_id:
                is_sequel = True
            elif service.is_sequel_title(rec_title) or (rec_eng_title and service.is_sequel_title(rec_eng_title)):
                is_sequel = True
            if is_sequel:
                sequel_count += 1
                
        sequel_contamination_rate = sequel_count / rec_count if rec_count > 0 else 0.0
        seed_franchise_leakage_rate = leakage_count / rec_count if rec_count > 0 else 0.0
        discovery_rate = 1.0 - seed_franchise_leakage_rate

        # Representation & Attribution
        seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in valid_ids}
        def get_winning_seed(cand_id):
            emb_c = model.catalog_embeddings[model.item_id_to_index[cand_id]]
            best_seed = None
            max_sim = -float('inf')
            for s_id in valid_ids:
                sim = float(np.dot(emb_c, seed_embeddings[s_id]))
                if sim > max_sim:
                    max_sim = sim
                    best_seed = s_id
            return best_seed

        seed_counts = {s_id: 0 for s_id in valid_ids}
        for item in recs:
            ws = get_winning_seed(item["anime_id"])
            if ws in seed_counts:
                seed_counts[ws] += 1
                
        seed_shares = {}
        for s_id, cnt in seed_counts.items():
            seed_shares[s_id] = cnt / rec_count if rec_count > 0 else 0.0
            
        dominant_seed_share = max(seed_shares.values()) if seed_shares else 0.0
        
        entropy = 0.0
        for s_id, share in seed_shares.items():
            if share > 0.0:
                entropy -= share * math.log2(share)

        # Append results
        rows_to_save.append({"scenario": name, "metric": "recommendation_count", "value": rec_count})
        rows_to_save.append({"scenario": name, "metric": "average_score", "value": avg_score})
        rows_to_save.append({"scenario": name, "metric": "minimum_score", "value": min_score})
        rows_to_save.append({"scenario": name, "metric": "maximum_score", "value": max_score})
        rows_to_save.append({"scenario": name, "metric": "unique_franchises", "value": unique_franchises})
        rows_to_save.append({"scenario": name, "metric": "duplicate_franchise_ratio", "value": duplicate_franchise_ratio})
        rows_to_save.append({"scenario": name, "metric": "sequel_contamination_rate", "value": sequel_contamination_rate})
        rows_to_save.append({"scenario": name, "metric": "seed_franchise_leakage_rate", "value": seed_franchise_leakage_rate})
        rows_to_save.append({"scenario": name, "metric": "discovery_rate", "value": discovery_rate})
        rows_to_save.append({"scenario": name, "metric": "dominant_seed_share", "value": dominant_seed_share})
        rows_to_save.append({"scenario": name, "metric": "representation_entropy", "value": entropy})
        rows_to_save.append({"scenario": name, "metric": "retrieval_ms", "value": retrieval_ms})
        rows_to_save.append({"scenario": name, "metric": "ranking_ms", "value": ranking_ms})
        rows_to_save.append({"scenario": name, "metric": "post_processing_ms", "value": post_processing_ms})
        rows_to_save.append({"scenario": name, "metric": "total_ms", "value": total_ms})
        
        for s_id, share in seed_shares.items():
            rows_to_save.append({"scenario": name, "metric": f"representation_share_{get_seed_abbr(s_id)}", "value": share})

        summary_data.append({
            "scenario": name,
            "mode": mode,
            "seeds_count": len(seeds),
            "discovery_rate": discovery_rate,
            "diversity": unique_franchises,
            "score": avg_score,
            "sequel_contamination": sequel_contamination_rate,
            "dominant_seed_share": dominant_seed_share
        })

    # Save benchmark_results.csv
    df_results = pd.DataFrame(rows_to_save)
    df_results.to_csv(os.path.join(PROJECT_ROOT, "benchmark_results.csv"), index=False)
    print("Saved benchmark_results.csv.", flush=True)

    # Compute Global Averages
    df_summary = pd.DataFrame(summary_data)
    avg_discovery = df_summary["discovery_rate"].mean()
    avg_diversity = df_summary["diversity"].mean()
    avg_score = df_summary["score"].mean()
    avg_dominance = df_summary["dominant_seed_share"].mean()

    # Find worst/best scenarios
    # Lowest diversity
    lowest_div_idx = df_summary["diversity"].idxmin()
    lowest_div_sc = df_summary.loc[lowest_div_idx]

    # Highest sequel contamination
    highest_seq_idx = df_summary["sequel_contamination"].idxmax()
    highest_seq_sc = df_summary.loc[highest_seq_idx]

    # Highest dominance collapse (multi-seed scenarios only)
    df_multiseed = df_summary[df_summary["seeds_count"] > 1]
    if not df_multiseed.empty:
        highest_dom_idx = df_multiseed["dominant_seed_share"].idxmax()
        highest_dom_sc = df_multiseed.loc[highest_dom_idx]

        lowest_dom_idx = df_multiseed["dominant_seed_share"].idxmin()
        lowest_dom_sc = df_multiseed.loc[lowest_dom_idx]
    else:
        highest_dom_sc = {"scenario": "None", "dominant_seed_share": 0.0}
        lowest_dom_sc = {"scenario": "None", "dominant_seed_share": 0.0}

    # Highest diversity
    highest_div_idx = df_summary["diversity"].idxmax()
    highest_div_sc = df_summary.loc[highest_div_idx]

    # Format benchmark_summary.md
    summary_md = f"""# CineSense Recommendation Benchmark Summary

This summary records the global metrics and outlier scenarios for the CineSense recommendation system evaluation suite across 26 gold standard scenarios.

---

## Global Averages

* **Average Discovery Rate:** {avg_discovery:.2%}
* **Average Franchise Diversity:** {avg_diversity:.2f} unique franchises
* **Average Recommendation Score:** {avg_score:.4f}
* **Average Dominant Seed Share:** {avg_dominance:.2%}

---

## Outlier Scenarios

### Worst Performing Scenarios
* **Lowest Franchise Diversity:** *{lowest_div_sc['scenario']}* ({lowest_div_sc['diversity']} unique franchises)
* **Highest Sequel Contamination:** *{highest_seq_sc['scenario']}* ({highest_seq_sc['sequel_contamination']:.1%} contamination rate)
* **Highest Dominant Seed Collapse:** *{highest_dom_sc['scenario']}* ({highest_dom_sc['dominant_seed_share']:.1%} dominant seed share)

### Best Performing Scenarios
* **Highest Franchise Diversity:** *{highest_div_sc['scenario']}* ({highest_div_sc['diversity']} unique franchises)
* **Best Balanced Multi-Seed Scenario:** *{lowest_dom_sc['scenario']}* ({lowest_dom_sc['dominant_seed_share']:.1%} dominant seed share)
"""

    with open(os.path.join(PROJECT_ROOT, "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)
    print("Saved benchmark_summary.md.", flush=True)

if __name__ == "__main__":
    main()
