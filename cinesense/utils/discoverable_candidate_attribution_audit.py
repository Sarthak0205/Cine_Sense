import os
import sys
import re
import numpy as np
import pandas as pd

# Set PYTHONPATH
PROJECT_ROOT = "/Users/sdc/Projects/CineSense-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.utils.model_storage import load_model
from cinesense.services.recommendation import RecommendationService, get_franchise
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices

def main():
    # Load model
    model, catalog_df, metadata = load_model(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1"))
    service = RecommendationService(model, catalog_df)

    # Seeds and ratings (Death Note, Code Geass, Steins;Gate)
    seeds = [1535, 1575, 9253]
    ratings = {1535: 10.0, 1575: 9.0, 9253: 8.0}
    seed_names = {1535: "Death Note", 1575: "Code Geass", 9253: "Steins;Gate"}

    # Run Stage 1 retrieval for Top 150 pool
    train_indices = np.asarray([model.item_id_to_index[aid] for aid in seeds], dtype=np.int32)
    train_weights = np.asarray([
        model._rating_weight(int(ratings[aid]))
        for aid in seeds
    ], dtype=np.float32)

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
        set(seeds),
        model.anime_ids,
        150,
    )

    # Determine seed franchises
    seed_franchises = set()
    for aid in seeds:
        meta = service.catalog_meta.get(aid)
        if meta:
            seed_franchises.add(get_franchise(meta["title"]))
            if meta.get("title_english"):
                seed_franchises.add(get_franchise(meta["title_english"]))

    # Precompute seed embeddings mapping for winning seed attribution
    seed_embeddings = {s_id: model.catalog_embeddings[model.item_id_to_index[s_id]] for s_id in seeds}

    # Extract candidates details
    candidates = []
    for rank_idx, idx in enumerate(retrieved_indices):
        anime_id = int(model.anime_ids[idx])
        meta = service.catalog_meta[anime_id]
        title = meta["title"]
        eng_title = meta.get("title_english") or ""
        ret_score = float(retrieval_scores[idx])
        
        # Compute winning seed attribution
        emb_c = model.catalog_embeddings[idx]
        best_seed = None
        max_sim = -float('inf')
        for s_id, emb_s in seed_embeddings.items():
            sim = float(np.dot(emb_c, emb_s))
            if sim > max_sim:
                max_sim = sim
                best_seed = s_id
                
        candidates.append({
            "anime_id": anime_id,
            "title": title,
            "title_english": eng_title,
            "winning_seed": best_seed,
            "retrieval_rank": rank_idx + 1,
            "retrieval_score": ret_score,
            "idx": idx,
        })

    # Discover Filter Classification Keywords
    keywords = ["movie", "film", "ova", "ona", "special", "specials", "recap", "pilot"]
    def matches_keywords(t):
        if not t:
            return False
        t_low = t.lower()
        return any(kw in t_low for kw in keywords)

    # Classify candidates sequentially
    seen_rec_franchises = set()
    for cand in candidates:
        anime_id = cand["anime_id"]
        title = cand["title"]
        eng_title = cand["title_english"]
        
        # A. Same franchise
        cand_f = get_franchise(title)
        cand_f_eng = get_franchise(eng_title) if eng_title else ""
        is_same_f = (cand_f in seed_franchises) or (cand_f_eng and cand_f_eng in seed_franchises)
        if is_same_f:
            cand["category"] = "Same Franchise"
            continue
            
        # B. Sequel
        root_id = service.get_franchise_root(cand_f)
        is_sequel = False
        if root_id is not None and anime_id != root_id:
            is_sequel = True
        elif service.is_sequel_title(title) or (eng_title and service.is_sequel_title(eng_title)):
            is_sequel = True
            
        if is_sequel:
            cand["category"] = "Sequel"
            continue
            
        # C. Movie/OVA/Special
        if matches_keywords(title) or matches_keywords(eng_title):
            cand["category"] = "Movie/OVA/Special"
            continue
            
        # D/E. Duplicate Franchise / Discover Eligible
        if cand_f in seen_rec_franchises or (cand_f_eng and cand_f_eng in seen_rec_franchises):
            cand["category"] = "Duplicate Franchise"
        else:
            cand["category"] = "Discover Eligible"
            seen_rec_franchises.add(cand_f)
            if cand_f_eng:
                seen_rec_franchises.add(cand_f_eng)

    # Run actual recommendation to get baseline final top 10 counts
    model.representation_penalty = False
    if "CINESENSE_REPRESENTATION_PENALTY" in os.environ:
        del os.environ["CINESENSE_REPRESENTATION_PENALTY"]
    
    recs = service.recommend(seeds, ratings=ratings, top_k=10, mode="discover")
    final_top_10_counts = {s_id: 0 for s_id in seeds}
    for item in recs:
        ws = item["explanation"].get("matched_seed", {}).get("anime_id")
        if ws in final_top_10_counts:
            final_top_10_counts[ws] += 1

    print("\n" + "="*80)
    print("CINESENSE DISCOVERABLE CANDIDATE ATTRIBUTION AUDIT")
    print("="*80 + "\n")

    # Step 3: Table A - Survival Rates
    print("### Table A — Seed Survival Rates")
    print(f"| {'Seed':<15} | {'Raw Candidates':<14} | {'Survive Filters':<15} | {'Survival %':<10} |")
    print(f"| {'-'*15} | {'-'*14} | {'-'*15} | {'-'*10} |")
    
    surv_counts = {}
    for s_id in seeds:
        raw = sum(1 for c in candidates if c["winning_seed"] == s_id)
        surv = sum(1 for c in candidates if c["winning_seed"] == s_id and c["category"] == "Discover Eligible")
        pct = (surv / raw) if raw > 0 else 0.0
        surv_counts[s_id] = {"raw": raw, "surv": surv, "pct": pct}
        print(f"| {seed_names[s_id]:<15} | {raw:<14} | {surv:<15} | {pct:<10.1%} |")
    print()

    # Step 4: Table B - Filter Breakdown
    print("### Table B — Filter Breakdown")
    
    def print_table_b_for_pool(limit):
        print(f"\n#### Candidates Pool: Top {limit}")
        print(f"| {'Seed':<15} | {'Same Franchise':<14} | {'Sequel':<6} | {'Movie/OVA':<9} | {'Duplicate':<9} | {'Discover Eligible':<17} |")
        print(f"| {'-'*15} | {'-'*14} | {'-'*6} | {'-'*9} | {'-'*9} | {'-'*17} |")
        for s_id in seeds:
            sub = [c for c in candidates[:limit] if c["winning_seed"] == s_id]
            f_same = sum(1 for c in sub if c["category"] == "Same Franchise")
            seq = sum(1 for c in sub if c["category"] == "Sequel")
            mov = sum(1 for c in sub if c["category"] == "Movie/OVA/Special")
            dup = sum(1 for c in sub if c["category"] == "Duplicate Franchise")
            elg = sum(1 for c in sub if c["category"] == "Discover Eligible")
            print(f"| {seed_names[s_id]:<15} | {f_same:<14} | {seq:<6} | {mov:<9} | {dup:<9} | {elg:<17} |")

    print_table_b_for_pool(50)
    print_table_b_for_pool(100)
    print_table_b_for_pool(150)
    print()

    # Step 5: Table C - Pipeline Drop-Off Analysis
    print("### Table C — Pipeline Drop-Off Analysis")
    print(f"| {'Seed':<15} | {'Retrieved':<9} | {'After Discover Filters':<22} | {'Final Top 10':<12} |")
    print(f"| {'-'*15} | {'-'*9} | {'-'*22} | {'-'*12} |")
    for s_id in seeds:
        ret = surv_counts[s_id]["raw"]
        aft = surv_counts[s_id]["surv"]
        top10 = final_top_10_counts[s_id]
        print(f"| {seed_names[s_id]:<15} | {ret:<9} | {aft:<22} | {top10:<12} |")
    print()

    # Step 6: Table D - Discoverable Attribution Simulation
    total_surviving = sum(surv_counts[s_id]["surv"] for s_id in seeds)
    print("### Table D — Discoverable Attribution Simulation")
    print(f"| {'Seed':<15} | {'Current Attribution %':<22} | {'Discoverable Attribution %':<27} |")
    print(f"| {'-'*15} | {'-'*22} | {'-'*27} |")
    for s_id in seeds:
        curr_pct = surv_counts[s_id]["raw"] / len(candidates)
        disc_pct = (surv_counts[s_id]["surv"] / total_surviving) if total_surviving > 0 else 0.0
        print(f"| {seed_names[s_id]:<15} | {curr_pct:<22.1%} | {disc_pct:<27.1%} |")
    print()

    # Diagnose Root Cause
    print("### Final Diagnosis")
    
    cg_id = 1575
    cg_raw = surv_counts[cg_id]["raw"]
    cg_surv = surv_counts[cg_id]["surv"]
    cg_top10 = final_top_10_counts[cg_id]
    
    reasons = []
    
    # Analyze if retrieval is the issue
    if cg_raw < 10:
        diagnosis = "ROOT CAUSE: RETRIEVAL"
        reasons.append(f"Retrieval provided only {cg_raw} Code Geass candidates out of 150.")
    # Analyze if discover filtering removes most
    elif (cg_raw - cg_surv) / cg_raw > 0.80 and cg_surv < 5:
        diagnosis = "ROOT CAUSE: DISCOVER FILTERING"
        reasons.append(f"Discover filtering eliminated {(cg_raw - cg_surv) / cg_raw:.1%} of Code Geass candidates, leaving only {cg_surv} discover eligible ones.")
    # Analyze if attribution is dominated by filtered items
    elif cg_raw >= 10 and cg_surv < 5:
        diagnosis = "ROOT CAUSE: ATTRIBUTION"
        reasons.append(f"Attribution is dominated by filtered candidates. While {cg_raw} candidates were attributed, only {cg_surv} survived discover filters.")
    # Analyze if ranking is suppressing discoverable items
    elif cg_surv >= 5 and cg_top10 == 0:
        diagnosis = "ROOT CAUSE: RANKING"
        reasons.append(f"Ranking suppressed discoverable candidates. {cg_surv} candidates survived discover filters, but 0 reached the final Top 10 recommendations.")
    else:
        diagnosis = "ROOT CAUSE: COMBINED EFFECT"
        reasons.append(f"Combined effect of filtering and ranking. {cg_raw} retrieved -> {cg_surv} surviving -> {cg_top10} final Top 10.")
        
    print(diagnosis)
    print("\nEvidence:")
    for r in reasons:
        print(f" - {r}")
    print()

if __name__ == "__main__":
    main()
