import os
import sys
import json
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cinesense.services.recommendation import get_franchise

# Genre rules
THEME_RULES = {
    'psychological': {'psychological', 'mind', 'conscience', 'trauma', 'identity', 'memory', 'mystery', 'detective', 'murder', 'killer', 'criminal', 'crime', 'pursuit', 'monster', 'genius', 'strategy', 'manipulation', 'suspense'},
    'crime': {'crime', 'criminal', 'detective', 'murder', 'killer', 'police', 'investigation', 'case', 'justice', 'pursuit', 'evidence', 'victim', 'law', 'trial', 'conspiracy'},
    'war': {'war', 'military', 'army', 'soldier', 'battle', 'battlefield', 'oppression', 'survival', 'walls', 'invasion', 'rebellion', 'resistance', 'humanity', 'weapon', 'corps'},
    'romance': {'romance', 'romantic', 'love', 'fate', 'heart', 'relationship', 'couple', 'feelings', 'emotional', 'dream', 'body', 'swap', 'supernatural', 'confession', 'date'},
    'sci-fi': {'scientific', 'science', 'future', 'technology', 'experiment', 'lab', 'time', 'travel', 'space', 'robot', 'mecha', 'dimension', 'microwave', 'hacker', 'machine', 'alien'},
    'horror': {'horror', 'ghoul', 'gore', 'blood', 'flesh', 'monster', 'demon', 'vampire', 'curse', 'fear', 'creature', 'supernatural', 'dark', 'terror', 'ghost'},
    'action': {'action', 'battle', 'fight', 'fighting', 'combat', 'power', 'warrior', 'soldier', 'army', 'military', 'weapon', 'attack', 'mission', 'survival'},
    'adventure': {'adventure', 'journey', 'travel', 'crew', 'pirate', 'treasure', 'island', 'quest', 'explore', 'world', 'ship', 'sea', 'grand', 'kingdom'},
    'fantasy': {'fantasy', 'magic', 'magical', 'demon', 'dragon', 'kingdom', 'curse', 'spirit', 'supernatural', 'wizard', 'sword', 'myth', 'god', 'beast'},
    'historical': {'historical', 'history', 'era', 'samurai', 'warrior', 'viking', 'medieval', 'ancient', 'kingdom', 'empire', 'japan', 'england', 'denmark', 'thorfinn', 'revenge'},
    'thriller': {'thriller', 'suspense', 'mystery', 'death', 'chase', 'danger', 'conspiracy', 'secret', 'murder', 'killer', 'survival', 'terror', 'mind', 'game'},
    'slice of life': {'slice of life', 'slice', 'everyday', 'school life', 'peaceful', 'comfy', 'relaxing'}
}

def clean_title(t):
    return re.sub(r'[^a-z0-9]', '', str(t).lower())

def is_sequel_title(title):
    if not title: return False
    t_low = title.lower().strip()
    if re.search(r"\b(?:season|part|vol|volume|movie|ova|ona)\s*(?:[2-9]|\d{2,}|ii|iii|iv|v|vi|vii|viii|ix|x)\b", t_low):
        return True
    if re.search(r"\b(?:2nd|3rd|[4-9]th|\d{2,}th)\s*(?:season|part|movie|ova|ona|series)\b", t_low):
        return True
    if re.search(r"\b(?:ii|iii|iv|v|vi|vii|viii|ix|x)\b", t_low):
        return True
    if re.search(r"\bx[2-9]\b", t_low):
        return True
    if re.search(r"\br[2-9]\b", t_low):
        return True
    if re.search(r"\bs[2-9]\b", t_low):
        return True
    if re.search(r"[\s:!\?\-]+(?:[2-9]|\d{2,})$", t_low):
        return True
    return False

def compute_metrics(relevance_grades, global_idcg=None):
    p10 = sum(1 for r in relevance_grades[:10] if r > 0) / 10.0
    dcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevance_grades[:10]))
    
    if global_idcg is not None:
        ndcg10 = dcg10 / global_idcg if global_idcg > 0 else 0.0
    else:
        ideal_grades = sorted(relevance_grades, reverse=True)[:10]
        idcg10 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades))
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
        
    mrr = 0.0
    for i, r in enumerate(relevance_grades[:10]):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
    return ndcg10, mrr, p10

def main():
    print("Loading catalog and datasets...", flush=True)
    animes_df = pd.read_csv(os.path.join(PROJECT_ROOT, "archive-2/animes.csv"))
    clean_to_id = {}
    id_to_title = {}
    for _, row in animes_df.iterrows():
        aid = int(row["anime_id"])
        title = str(row["title"])
        eng_title = str(row["title_english"]) if pd.notna(row["title_english"]) else ""
        clean_to_id[clean_title(title)] = aid
        id_to_title[aid] = title
        if eng_title:
            clean_to_id[clean_title(eng_title)] = aid

    MANUAL_OVERRIDES = {
        clean_title("Re:Zero - Starting Life in Another World"): 31240,
        clean_title("Salaryman Kintaro"): 1608,
        clean_title("The Garden of Sinners"): 2593,
        clean_title("Devilman Crybaby"): 35120,
        clean_title("Yu Yu Hakusho"): 392
    }

    def resolve_title_to_id(title_str):
        c_title = clean_title(title_str)
        if c_title in MANUAL_OVERRIDES:
            return MANUAL_OVERRIDES[c_title]
        return clean_to_id.get(c_title)

    # Load baseline assets
    baseline_assets = np.load(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1/model_assets.npz"))
    anime_ids_v1 = baseline_assets["anime_ids"].astype(np.int32)
    id_to_v1_idx = {int(aid): idx for idx, aid in enumerate(anime_ids_v1.tolist())}
    popularity_scores = baseline_assets["popularity_scores"].astype(np.float32)
    catalog_embeddings = baseline_assets["catalog_embeddings"].astype(np.float32)

    # Calculate popularity percentiles
    sorted_pop_indices = np.argsort(popularity_scores)
    pop_percentiles = np.zeros_like(popularity_scores)
    for rank, idx in enumerate(sorted_pop_indices):
        pop_percentiles[idx] = rank / len(popularity_scores)

    # Load precomputed graph assets to ensure identical lookups as production
    print("Loading precomputed graph assets...", flush=True)
    graph_assets = np.load(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1/graph_assets.npz"))
    neighbor_ids = graph_assets["neighbor_ids"]
    neighbor_jaccards = graph_assets["neighbor_jaccards"]
    distance_lookup = graph_assets["distance_lookup"]
    supported_anime_ids = graph_assets["supported_anime_ids"]
    col_sums = graph_assets["col_sums"]
    anime_to_idx = {int(aid): idx for idx, aid in enumerate(supported_anime_ids.tolist())}

    def get_jaccard(aid1, aid2):
        idx_s = anime_to_idx.get(aid1)
        idx_c = anime_to_idx.get(aid2)
        if idx_s is None or idx_c is None:
            return 0.0
        row_ids = neighbor_ids[idx_s]
        pos = np.searchsorted(row_ids, aid2)
        if pos < len(row_ids) and row_ids[pos] == aid2:
            return float(neighbor_jaccards[idx_s][pos])
        return 0.0

    def get_distance(aid1, aid2):
        idx_s = anime_to_idx.get(aid1)
        idx_c = anime_to_idx.get(aid2)
        if idx_s is None or idx_c is None:
            return 10
        dist = distance_lookup[idx_s, idx_c]
        return int(dist) if dist > 0 else 10

    # Load gold standard
    with open(os.path.join(PROJECT_ROOT, "evaluation/gold_standard_v2.json")) as f:
        gold_dataset = json.load(f)

    catalog_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "cinesense/models/twostage_v1/catalog.parquet"))
    catalog_meta = {}
    for _, row in catalog_df.iterrows():
        aid = int(row["anime_id"])
        catalog_meta[aid] = {
            "title": str(row.get("title", "")),
            "title_english": str(row.get("title_english", "")) if pd.notna(row.get("title_english")) else None
        }

    # Map seeds and calculate global IDCG
    gold_seeds_mapped = []
    for entry in gold_dataset:
        seed_id = entry.get("anime_id")
        if not seed_id:
            seed_id = resolve_title_to_id(entry["seed"])
        if not seed_id or int(seed_id) not in id_to_v1_idx:
            continue
        seed_id = int(seed_id)
        
        good_recs = set()
        acc_recs = set()
        for r_title in entry["good_recommendations"]:
            rid = resolve_title_to_id(r_title)
            if rid: good_recs.add(int(rid))
        for r_title in entry["acceptable_recommendations"]:
            rid = resolve_title_to_id(r_title)
            if rid: acc_recs.add(int(rid))
            
        # Global IDCG
        ideal_grades = [2] * len(good_recs) + [1] * len(acc_recs)
        ideal_grades.sort(reverse=True)
        ideal_grades = ideal_grades[:10]
        while len(ideal_grades) < 10:
            ideal_grades.append(0)
        global_idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_grades))
            
        gold_seeds_mapped.append({
            "seed_id": seed_id,
            "seed_title": entry["seed"],
            "good": good_recs,
            "acceptable": acc_recs,
            "all_relevant": good_recs.union(acc_recs),
            "global_idcg": global_idcg
        })

    # Retrieve top 100 candidates for all seeds
    print("Generating candidate pools...", flush=True)
    seed_candidates = {}
    for s_data in gold_seeds_mapped:
        seed_id = s_data["seed_id"]
        
        seed_meta = catalog_meta.get(seed_id, {})
        seed_f = get_franchise(seed_meta.get("title", ""))
        seed_f_eng = get_franchise(seed_meta.get("title_english", "")) if seed_meta.get("title_english") else ""
        seed_franchises = {seed_f, seed_f_eng} - {""}
        
        candidates = []
        for aid, idx in id_to_v1_idx.items():
            if aid == seed_id:
                continue
            meta = catalog_meta[aid]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            f_title = get_franchise(title)
            f_eng = get_franchise(eng_title) if eng_title else ""
            if f_title in seed_franchises or (f_eng and f_eng in seed_franchises):
                continue
            
            # Discover score
            score = 0.85 * float(np.dot(catalog_embeddings[id_to_v1_idx[seed_id]], catalog_embeddings[idx])) + 0.15 * popularity_scores[idx]
            candidates.append((aid, score, idx))
            
        candidates.sort(key=lambda x: -x[1])
        
        enriched_100 = []
        seen_franchises = set()
        for aid, score, idx in candidates:
            if is_sequel_title(catalog_meta[aid]["title"]) or is_sequel_title(catalog_meta[aid]["title_english"]):
                continue
            if get_franchise(catalog_meta[aid]["title"]) in seen_franchises:
                continue
            seen_franchises.add(get_franchise(catalog_meta[aid]["title"]))
            enriched_100.append((aid, score, idx))
            if len(enriched_100) == 100:
                break
                
        seed_candidates[seed_id] = enriched_100

    # Models to evaluate
    models = [
        {"name": "Model A (Semantic Only)", "alpha": 0.0, "beta": 0.0, "power": 0, "penalty": 0.0},
        {"name": "Model B (Semantic + Jaccard)", "alpha": 1.0, "beta": 0.0, "power": 0, "penalty": 0.0},
        {"name": "Model C (Semantic + Jac + Dist)", "alpha": 1.0, "beta": 0.05, "power": 0, "penalty": 0.0},
        {"name": "Model D (Full Production)", "alpha": 1.0, "beta": 0.05, "power": 2, "penalty": 0.05},
    ]

    print("\n" + "="*80)
    print("RUNNING BENCHMARK EVALUATIONS ON 35 SEEDS")
    print("="*80)

    model_results = {}
    
    for m in models:
        ndcgs, mrrs, precs = [], [], []
        discovery_rates = []
        franchise_diversities = []
        
        seed_scores = {}
        
        for s_data in gold_seeds_mapped:
            seed_id = s_data["seed_id"]
            candidates = seed_candidates[seed_id]
            good_set = s_data["good"]
            acc_set = s_data["acceptable"]
            global_idcg = s_data["global_idcg"]
            
            # Seed franchise details for checking Discovery Rate
            seed_meta = catalog_meta.get(seed_id, {})
            seed_franchises = {get_franchise(seed_meta.get("title", ""))}
            if seed_meta.get("title_english"):
                seed_franchises.add(get_franchise(seed_meta["title_english"]))
            seed_franchises.discard("")
            
            scored_candidates = []
            
            for aid, sem_score, idx in candidates:
                cosine = float(np.dot(catalog_embeddings[id_to_v1_idx[seed_id]], catalog_embeddings[idx]))
                jac = get_jaccard(seed_id, aid)
                dist = get_distance(seed_id, aid)
                
                # Distance score
                if dist == 1:
                    dist_score = 0.5
                elif dist == 2:
                    dist_score = 1.0 / 3.0
                else:
                    dist_score = 0.0
                    
                # Popularity penalty
                pop_pct = pop_percentiles[idx]
                pop_pen = m["penalty"] * max(0.0, pop_pct - 0.95)
                
                # Formula
                score = (
                    sem_score
                    + m["alpha"] * jac * (cosine ** m["power"])
                    + m["beta"] * dist_score
                    - pop_pen
                )
                scored_candidates.append((aid, score, idx))
                
            # Rerank and keep top 10
            scored_candidates.sort(key=lambda x: -x[1])
            top_10 = scored_candidates[:10]
            top_10_ids = [x[0] for x in top_10]
            
            # Metrics
            grades = [2 if a in good_set else (1 if a in acc_set else 0) for a in top_10_ids]
            while len(grades) < 10:
                grades.append(0)
            ndcg, mrr, p10 = compute_metrics(grades, global_idcg=global_idcg)
            
            ndcgs.append(ndcg)
            mrrs.append(mrr)
            precs.append(p10)
            seed_scores[s_data["seed_title"]] = ndcg
            
            # Discovery Rate & Franchise Diversity
            new_franchise_count = 0
            rec_franchises = []
            for a_id in top_10_ids:
                meta = catalog_meta[a_id]
                rec_f = get_franchise(meta["title"])
                rec_f_eng = get_franchise(meta.get("title_english") or "")
                
                is_same = (rec_f in seed_franchises) or (rec_f_eng and rec_f_eng in seed_franchises)
                if not is_same:
                    new_franchise_count += 1
                rec_franchises.append(rec_f)
                
            dr = (new_franchise_count / len(top_10)) * 100.0 if top_10 else 0.0
            fd = len(set(rec_franchises))
            
            discovery_rates.append(dr)
            franchise_diversities.append(fd)
            
        model_results[m["name"]] = {
            "ndcg": np.mean(ndcgs),
            "mrr": np.mean(mrrs),
            "p10": np.mean(precs),
            "dr": np.mean(discovery_rates),
            "fd": np.mean(franchise_diversities),
            "seed_scores": seed_scores
        }
        
    # Print Comparison Table
    print(f"\n| {'Model':<30} | {'NDCG@10':<8} | {'MRR':<8} | {'P@10':<8} | {'Disc. Rate':<10} | {'Diversity':<9} |")
    print(f"| {'-'*30} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*9} |")
    for name, res in model_results.items():
        print(f"| {name:<30} | {res['ndcg']:<8.4f} | {res['mrr']:<8.4f} | {res['p10']:<8.2%} | {res['dr']:<9.1f}% | {res['fd']:<9.2f} |")

    # Evaluate promotion gates on Model D
    print("\n" + "="*80)
    print("PROMOTION GATES ALIGNMENT AUDIT (Model D)")
    print("="*80)
    d_res = model_results["Model D (Full Production)"]
    gates = [
        ("NDCG@10", d_res["ndcg"], 0.22, "NDCG@10 >= 0.22"),
        ("MRR", d_res["mrr"], 0.50, "MRR >= 0.50"),
        ("Precision@10", d_res["p10"], 0.18, "Precision@10 >= 18%"),
        ("Discovery Rate", d_res["dr"]/100.0, 0.95, "Discovery Rate >= 95%"),
        ("Franchise Diversity", d_res["fd"], 7.0, "Franchise Diversity >= 7"),
    ]
    
    all_passed = True
    for metric, val, threshold, desc in gates:
        status = "PASSED" if val >= threshold else "FAILED"
        if val < threshold: all_passed = False
        val_fmt = f"{val:.2%}" if "%" in desc else f"{val:.4f}"
        print(f"  - {metric:<20} | Status: {status:<8} | Value: {val_fmt:<8} | Gate: {desc}")
        
    print(f"\n  VERDICT: {'ALL PROMOTION GATES MET!' if all_passed else 'SOME GATES FAILED'}")

    # Seed level gains and regressions (Model D vs Model A)
    print("\n" + "="*80)
    print("SEED-LEVEL ANALYSIS (Model D vs Model A)")
    print("="*80)
    gains, regressions = [], []
    scores_a = model_results["Model A (Semantic Only)"]["seed_scores"]
    scores_d = model_results["Model D (Full Production)"]["seed_scores"]
    
    for title in scores_a:
        delta = scores_d[title] - scores_a[title]
        if delta > 0.001:
            gains.append((title, delta))
        elif delta < -0.001:
            regressions.append((title, delta))
            
    print(f"Total Seeds Improved: {len(gains)}")
    for title, delta in sorted(gains, key=lambda x: -x[1])[:10]:
        print(f"  - {title:<35} | {delta:+.4f}")
        
    print(f"Total Seeds Degraded: {len(regressions)}")
    for title, delta in sorted(regressions, key=lambda x: x[1])[:10]:
        print(f"  - {title:<35} | {delta:+.4f}")

if __name__ == "__main__":
    main()
