import json
import re
import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Theme-rules mapping for dynamic genre extraction
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

def get_franchise(title: str) -> str:
    title = str(title).lower().strip()
    overrides = {"attack on skytree": "attack on titan", "shingeki no kyotou": "shingeki no kyojin"}
    if title in overrides:
        return overrides[title]
    title = re.split(r'[:\-\(!]', title)[0].strip()
    if title in overrides:
        return overrides[title]
    title = re.sub(
        r'\b(season|movie|film|ova|ona|tv|specials|special|recap|pilot|part|chapter|edition|remaster|remake|the animation|rewrite|relight|summary|3d|ii|iii|iv|v|vi|vii|viii|ix|x|\d+st|\d+nd|\d+rd|\d+th|\d+)\b',
        '',
        title
    )
    return re.sub(r'\s+', ' ', title).strip()

def clean_title(t):
    return re.sub(r'[^a-z0-9]', '', str(t).lower())

def main():
    print("Step 1: Loading twostage_v1 model assets...", flush=True)
    baseline_assets = np.load("cinesense/models/twostage_v1/model_assets.npz")
    anime_ids_v1 = baseline_assets["anime_ids"].astype(np.int32)
    id_to_v1_idx = {int(aid): idx for idx, aid in enumerate(anime_ids_v1.tolist())}
    popularity_scores = baseline_assets["popularity_scores"].astype(np.float32)
    catalog_embeddings = baseline_assets["catalog_embeddings"].astype(np.float32)
    
    print("Step 2: Loading anime catalog metadata...", flush=True)
    animes_df = pd.read_csv("archive-2/animes.csv")
    anime_metadata = {}
    clean_to_id = {}
    
    for _, row in animes_df.iterrows():
        aid = int(row["anime_id"])
        title = str(row["title"])
        eng_title = str(row["title_english"]) if pd.notna(row["title_english"]) else ""
        synopsis = str(row["synopsis"]) if pd.notna(row["synopsis"]) else ""
        
        f_title = get_franchise(title)
        f_eng = get_franchise(eng_title) if eng_title else ""
        
        anime_metadata[aid] = {
            "title": title,
            "title_english": eng_title,
            "franchise": (f_title, f_eng),
            "synopsis": synopsis.lower(),
            "genres": []
        }
        
        clean_to_id[clean_title(title)] = aid
        if eng_title:
            clean_to_id[clean_title(eng_title)] = aid

    # Populate genres based on THEME_RULES
    for aid, meta in anime_metadata.items():
        text = f"{meta['title']} {meta['title_english']} {meta['synopsis']}".lower()
        meta["genres"] = [theme for theme, words in THEME_RULES.items() if any(w in text for w in words)]

    print("Step 3: Loading Gold Standard and resolving Quarantine sets...", flush=True)
    with open("evaluation/gold_standard_v2.json") as f:
        gold_data = json.load(f)
        
    seed_ids = set()
    rec_ids = set()
    
    # Manual overrides dictionary for unresolved titles
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
        
    for entry in gold_data:
        seed_id = entry.get("anime_id")
        if seed_id:
            seed_ids.add(int(seed_id))
        else:
            sid = resolve_title_to_id(entry["seed"])
            if sid:
                seed_ids.add(sid)
                
        for r_title in entry["good_recommendations"] + entry["acceptable_recommendations"]:
            rid = resolve_title_to_id(r_title)
            if rid:
                rec_ids.add(rid)
            else:
                print(f"Warning: Could not resolve title '{r_title}'")

    seed_franchises = set()
    for sid in seed_ids:
        meta = anime_metadata.get(sid)
        if meta:
            seed_franchises.add(meta["franchise"][0])
            if meta["franchise"][1]:
                seed_franchises.add(meta["franchise"][1])
                
    # Strategy D Filter: Quarantine all seed franchises + direct recommendations
    def is_quarantined(aid):
        if aid in seed_ids:
            return True
        meta = anime_metadata.get(aid)
        if not meta:
            return False
        f_title, f_eng = meta["franchise"]
        if (f_title in seed_franchises) or (f_eng and f_eng in seed_franchises):
            return True
        if aid in rec_ids:
            return True
        return False

    print("Step 4: Streaming and filtering user watch history...", flush=True)
    chunks = []
    v1_set = set(id_to_v1_idx.keys())
    for chunk in pd.read_csv("archive-2/user_watches.csv", usecols=["user_id", "anime_id", "score"], chunksize=5_000_000):
        pos_chunk = chunk[(chunk["score"] >= 7) & chunk["anime_id"].isin(v1_set)]
        chunks.append(pos_chunk)
    df_pos = pd.concat(chunks, ignore_index=True)
    
    # Deduplicate to prevent double-counting
    df_pos = df_pos.drop_duplicates(subset=["user_id", "anime_id"])
    
    # Apply support filtering (counts >= 100)
    anime_counts = df_pos["anime_id"].value_counts()
    supported_animes = set(anime_counts[anime_counts >= 100].index)
    median_watches = anime_counts[anime_counts >= 100].median()
    print(f"Supported nodes: {len(supported_animes)}, Median watch count: {median_watches}", flush=True)
    
    df_pos_filtered = df_pos[df_pos["anime_id"].isin(supported_animes)].copy()
    
    user_unique = df_pos_filtered["user_id"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_unique)}
    supported_animes_list = list(supported_animes)
    anime_to_idx = {aid: idx for idx, aid in enumerate(supported_animes_list)}
    
    df_pos_filtered["user_idx"] = df_pos_filtered["user_id"].map(user_to_idx)
    df_pos_filtered["anime_idx"] = df_pos_filtered["anime_id"].map(anime_to_idx)
    
    print("Step 5: Computing Jaccard overlaps using sparse matrices...", flush=True)
    X = csr_matrix((np.ones(len(df_pos_filtered), dtype=np.float32), 
                    (df_pos_filtered["user_idx"].values, df_pos_filtered["anime_idx"].values)), 
                   shape=(len(user_unique), len(supported_animes_list)))
    
    col_sums = np.array(X.sum(axis=0)).flatten()
    I = X.T.dot(X)
    
    # Prune low co-watches early (co-watches < 15) to prevent memory issues
    I.data[I.data < 15] = 0
    I.eliminate_zeros()
    
    # Convert to COO to build overlaps
    I_coo = I.tocoo()
    mask = I_coo.row < I_coo.col
    r_idx, c_idx, intersect = I_coo.row[mask], I_coo.col[mask], I_coo.data[mask]
    
    unions = col_sums[r_idx] + col_sums[c_idx] - intersect
    jaccard_scores = intersect / unions
    
    # Dynamic threshold logic
    niche_mask = np.zeros(len(supported_animes_list), dtype=bool)
    for aid, idx in anime_to_idx.items():
        if anime_counts.get(aid, 0) < median_watches:
            niche_mask[idx] = True
            
    is_niche_pair = niche_mask[r_idx] | niche_mask[c_idx]
    pair_thresholds = np.where(is_niche_pair, 0.14, 0.18)
    
    threshold_mask = jaccard_scores >= pair_thresholds
    matching_rows = r_idx[threshold_mask]
    matching_cols = c_idx[threshold_mask]
    matching_jaccards = jaccard_scores[threshold_mask]
    
    # Filter quarantined nodes from active positive pairs
    pairs_to_filter = []
    for r, c, j in zip(matching_rows, matching_cols, matching_jaccards):
        a1 = supported_animes_list[r]
        a2 = supported_animes_list[c]
        if not is_quarantined(a1) and not is_quarantined(a2):
            pairs_to_filter.append((a1, a2, j))

    print(f"Generated {len(pairs_to_filter)} quarantined-free positive pairs.", flush=True)
    
    # Set seed for reproducibility
    random.seed(42)
    
    # 10% Node holdout split
    active_anime_ids = set()
    for a1, a2, _ in pairs_to_filter:
        active_anime_ids.add(a1)
        active_anime_ids.add(a2)
        
    active_nodes = list(active_anime_ids)
    random.shuffle(active_nodes)
    num_val_nodes = int(len(active_nodes) * 0.10)
    val_nodes = set(active_nodes[:num_val_nodes])
    val_nodes_idxs = {id_to_v1_idx[aid] for aid in val_nodes}
    
    print(f"Total active nodes: {len(active_nodes)}, Held-out validation nodes: {len(val_nodes)}", flush=True)
    
    # Role-swapping safeguards for training and validation splits
    train_pairs_filtered = []
    train_weights = []
    val_pairs_filtered = []
    val_weights = []
    
    for a1, a2, j in pairs_to_filter:
        idx1 = id_to_v1_idx[a1]
        idx2 = id_to_v1_idx[a2]
        
        if (a1 in val_nodes) or (a2 in val_nodes):
            # Validation: force validation node to be anchor (index 0)
            if a1 in val_nodes:
                val_pairs_filtered.append((idx1, idx2))
            else:
                val_pairs_filtered.append((idx2, idx1))
            val_weights.append(j)
        else:
            # Training: randomly swap to remove index order bias
            if random.random() < 0.5:
                idx1, idx2 = idx2, idx1
            train_pairs_filtered.append((idx1, idx2))
            train_weights.append(j)
            
    print(f"Train pairs: {len(train_pairs_filtered)}, Validation pairs: {len(val_pairs_filtered)}", flush=True)

    # Pre-compile CSR pointers into a nested row dictionary for fast co-watch querying
    I_rows = {}
    for idx in range(len(supported_animes_list)):
        start = I.indptr[idx]
        end = I.indptr[idx + 1]
        I_rows[idx] = dict(zip(I.indices[start:end], I.data[start:end]))

    def get_jaccard(idx1, idx2):
        aid1 = anime_ids_v1[idx1]
        aid2 = anime_ids_v1[idx2]
        r = anime_to_idx.get(aid1)
        c = anime_to_idx.get(aid2)
        if r is None or c is None:
            return 0.0
        co_watches = I_rows.get(r, {}).get(c, 0.0)
        if co_watches == 0.0:
            return 0.0
        return co_watches / (col_sums[r] + col_sums[c] - co_watches)

    # Genre pre-grouping for same-genre negatives search optimization
    genre_to_animes = {theme: [] for theme in THEME_RULES}
    for aid, meta in anime_metadata.items():
        idx = id_to_v1_idx.get(aid)
        if idx is not None:
            for genre in meta["genres"]:
                genre_to_animes[genre].append(idx)

    # Precompute negative mining lists
    quarantined_indices = {id_to_v1_idx[aid] for aid in id_to_v1_idx if is_quarantined(aid)}
    forbidden_train_negs = val_nodes_idxs.union(quarantined_indices)
    popular_100_indices = np.argsort(-popularity_scores)[:100].tolist()

    print("Step 6: Performing optimized negative mining...", flush=True)
    
    def mine_negatives(anchor_idx, forbidden_set):
        negs = []
        anchor_aid = anime_ids_v1[anchor_idx]
        anchor_genres = anime_metadata.get(anchor_aid, {}).get("genres", [])
        
        # 1. 4x Popular Unrelated (Jaccard < 0.01)
        for p_idx in popular_100_indices:
            if p_idx == anchor_idx or p_idx in forbidden_set:
                continue
            if get_jaccard(anchor_idx, p_idx) < 0.01:
                negs.append(p_idx)
                if len(negs) == 4:
                    break
                    
        # 2. 4x Same-Genre Unrelated (Jaccard < 0.005)
        candidate_pool = set().union(*(genre_to_animes[g] for g in anchor_genres))
        for c_idx in candidate_pool:
            if c_idx == anchor_idx or c_idx in negs or c_idx in forbidden_set:
                continue
            if get_jaccard(anchor_idx, c_idx) < 0.005:
                negs.append(c_idx)
                if len(negs) == 8:
                    break
                    
        # 3. 2x Baseline False-Positives (Highly ranked, Jaccard < 0.01)
        emb_anchor = catalog_embeddings[anchor_idx]
        sims = catalog_embeddings @ emb_anchor
        scores = 0.85 * sims + 0.15 * popularity_scores
        top_recs = np.argsort(-scores)
        
        for c_idx in top_recs:
            if c_idx == anchor_idx or c_idx in negs or c_idx in forbidden_set:
                continue
            if get_jaccard(anchor_idx, c_idx) < 0.01:
                negs.append(int(c_idx))
                if len(negs) == 10:
                    break
                    
        # Fallback cascade to guarantee 10 negatives
        if len(negs) < 10:
            for p_idx in popular_100_indices:
                if p_idx == anchor_idx or p_idx in negs or p_idx in forbidden_set:
                    continue
                negs.append(p_idx)
                if len(negs) == 10:
                    break
        while len(negs) < 10:
            rand_idx = random.randint(0, len(popularity_scores) - 1)
            if rand_idx != anchor_idx and rand_idx not in negs and rand_idx not in forbidden_set:
                negs.append(rand_idx)
                
        return negs

    # Anchor-level caching
    train_neg_cache = {}
    val_neg_cache = {}
    
    train_negatives = []
    for anchor, pos in train_pairs_filtered:
        if anchor not in train_neg_cache:
            train_neg_cache[anchor] = mine_negatives(anchor, forbidden_train_negs)
        train_negatives.append(train_neg_cache[anchor])
        
    val_negatives = []
    for anchor, pos in val_pairs_filtered:
        if anchor not in val_neg_cache:
            val_neg_cache[anchor] = mine_negatives(anchor, quarantined_indices)
        val_negatives.append(val_neg_cache[anchor])

    # Assertions / Integrity Checks
    print("Step 7: Running validation assertions on pairs...", flush=True)
    # Check no quarantined IDs in splits
    for idx1, idx2 in train_pairs_filtered:
        assert idx1 not in quarantined_indices, "Training pair leaks quarantined node!"
        assert idx2 not in quarantined_indices, "Training pair leaks quarantined node!"
    for idx1, idx2 in val_pairs_filtered:
        assert idx1 not in quarantined_indices, "Validation pair leaks quarantined node!"
        assert idx2 not in quarantined_indices, "Validation pair leaks quarantined node!"
        
    # Check graph holdout integrity
    train_nodes = set(np.array(train_pairs_filtered).flatten())
    assert len(train_nodes.intersection(val_nodes_idxs)) == 0, "Graph contamination: Validation nodes present in training pairs!"
    
    # Check validation anchors are strictly held out
    val_anchors = {p[0] for p in val_pairs_filtered}
    assert val_anchors.issubset(val_nodes_idxs), "Validation anchor leakage: Anchor is not a held-out validation node!"
    
    # Check no duplicate pairs
    assert len(set(train_pairs_filtered)) == len(train_pairs_filtered), "Found duplicate pairs in training set!"
    assert len(set(val_pairs_filtered)) == len(val_pairs_filtered), "Found duplicate pairs in validation set!"
    
    # Volume check
    assert len(train_pairs_filtered) >= 40000, f"Insufficient positive pairs: {len(train_pairs_filtered)}"

    print("Step 8: Exporting training and validation pairs to models/twostage_v2...", flush=True)
    os.makedirs("cinesense/models/twostage_v2", exist_ok=True)
    np.savez_compressed(
        "cinesense/models/twostage_v2/train_pairs.npz",
        train_pairs=np.array(train_pairs_filtered, dtype=np.int32),
        train_weights=np.array(train_weights, dtype=np.float32),
        train_negatives=np.array(train_negatives, dtype=np.int32),
        val_pairs=np.array(val_pairs_filtered, dtype=np.int32),
        val_weights=np.array(val_weights, dtype=np.float32),
        val_negatives=np.array(val_negatives, dtype=np.int32)
    )
    print("Pairs generated and verified successfully.", flush=True)

if __name__ == "__main__":
    main()
