import numpy as np
import inspect
import re
import os


def weighted_max_similarity_to_train_items(
    train_indices: np.ndarray,
    train_weights: np.ndarray,
    catalog_embeddings: np.ndarray,
    seed_batch_size: int = 128,
) -> np.ndarray:
    """Computes similarity to training items using a hybrid of weighted average and max raw similarity."""
    if catalog_embeddings is None or catalog_embeddings.size == 0:
        raise RuntimeError("Catalog embeddings are not initialized.")

    total_weights = np.sum(train_weights)
    if total_weights <= 0:
        return np.zeros(len(catalog_embeddings), dtype=np.float32)

    weighted_sim_sum = np.zeros(len(catalog_embeddings), dtype=np.float32)
    max_sim = np.full(len(catalog_embeddings), -np.inf, dtype=np.float32)
    batch_size = max(1, seed_batch_size)

    for start in range(0, train_indices.size, batch_size):
        batch_indices = train_indices[start : start + batch_size]
        batch_weights = train_weights[start : start + batch_size]
        train_embeddings = catalog_embeddings[batch_indices]
        
        # Raw similarity
        batch_scores = catalog_embeddings @ train_embeddings.T
        
        # Running max of raw similarity (not scaled by weight)
        max_sim = np.maximum(max_sim, batch_scores.max(axis=1))
        
        # Running sum of weighted similarities
        weighted_batch_scores = batch_scores * batch_weights.reshape(1, -1)
        weighted_sim_sum += weighted_batch_scores.sum(axis=1)

    weighted_avg = weighted_sim_sum / total_weights
    
    rebalance_weight_str = os.environ.get("CINESENSE_REBALANCE_WEIGHT", "0.70")
    try:
        weight_avg = float(rebalance_weight_str)
    except ValueError:
        weight_avg = 0.70
    weight_max = 1.0 - weight_avg
    
    hybrid_score = weight_avg * weighted_avg + weight_max * max_sim
    return hybrid_score


def rerank_candidates(
    retrieved_indices: np.ndarray,
    rerank_scores: np.ndarray,
    retrieval_scores: np.ndarray,
    anime_ids: np.ndarray,
    top_k: int,
    representation_penalty: bool = False,
    representation_lambda: float = 0.03,
    train_indices: np.ndarray = None,
    catalog_embeddings: np.ndarray = None,
) -> list[int]:
    """Sorts and selects top candidates based on rerank scores, retrieval scores, and anime IDs, optionally applying representation penalty."""
    
    # 1. Look up caller stack frame to retrieve recommender context or variables if not explicitly provided
    frame = inspect.currentframe()
    recommender_obj = None
    caller_train_indices = None
    
    try:
        f = frame.f_back
        while f:
            locals_dict = f.f_locals
            if 'self' in locals_dict:
                self_obj = locals_dict['self']
                cls_name = self_obj.__class__.__name__
                if cls_name in ('CineSenseTwoStage', 'CineSenseHybridLiteWeighted'):
                    recommender_obj = self_obj
            if 'train_indices' in locals_dict and caller_train_indices is None:
                caller_train_indices = locals_dict['train_indices']
            f = f.f_back
    finally:
        del frame

    # Resolve dependencies
    if train_indices is None:
        train_indices = caller_train_indices
    if catalog_embeddings is None and recommender_obj is not None:
        catalog_embeddings = getattr(recommender_obj, 'catalog_embeddings', None)

    # Resolve flags (from arguments, recommender properties, or environment variables)
    if not representation_penalty:
        if recommender_obj is not None:
            representation_penalty = getattr(recommender_obj, 'representation_penalty', False)
        env_val = os.environ.get("CINESENSE_REPRESENTATION_PENALTY", "False").lower()
        if env_val in ("true", "1", "yes"):
            representation_penalty = True

    if representation_penalty:
        if recommender_obj is not None:
            representation_lambda = getattr(recommender_obj, 'representation_lambda', 0.03)
        env_lambda = os.environ.get("CINESENSE_REPRESENTATION_LAMBDA")
        if env_lambda is not None:
            try:
                representation_lambda = float(env_lambda)
            except ValueError:
                pass

    # 2. Return standard baseline sorting if penalty is not active or required data is missing
    if not representation_penalty or train_indices is None or catalog_embeddings is None:
        reranked_indices = sorted(
            retrieved_indices.tolist(),
            key=lambda index: (
                -float(rerank_scores[index]),
                -float(retrieval_scores[index]),
                int(anime_ids[index]),
            ),
        )
        return [int(anime_ids[index]) for index in reranked_indices[:top_k]]

    # 3. Greedy Selection with Penalty
    seed_ids = [int(anime_ids[idx]) for idx in train_indices]
    selected_count_by_seed = {s_id: 0 for s_id in seed_ids}
    
    # Precompute seed embeddings mapping once
    item_id_to_index = {int(aid): idx for idx, aid in enumerate(anime_ids.tolist())}
    seed_embeddings = {}
    for s_id in seed_ids:
        if s_id in item_id_to_index:
            seed_embeddings[s_id] = catalog_embeddings[item_id_to_index[s_id]]

    # Precompute winning seed attribution once for every retrieved candidate
    winning_seeds = {}
    for idx in retrieved_indices:
        emb_c = catalog_embeddings[idx]
        best_seed = None
        max_sim = -float('inf')
        for s_id, emb_s in seed_embeddings.items():
            sim = float(np.dot(emb_c, emb_s))
            if sim > max_sim:
                max_sim = sim
                best_seed = s_id
        winning_seeds[idx] = best_seed

    pool_indices = retrieved_indices.tolist()
    selected_indices = []
    audit_dict = {}

    # Greedy Selection Loop
    while pool_indices:
        best_idx = None
        best_adjusted_score = -float('inf')
        best_retrieval_score = -float('inf')
        best_anime_id = float('inf')
        best_winning_seed = None
        best_base_score = None
        
        for idx in pool_indices:
            base_score = float(rerank_scores[idx])
            winning_seed = winning_seeds.get(idx)
            
            count_val = selected_count_by_seed.get(winning_seed, 0) if winning_seed is not None else 0
            adjusted_score = base_score - representation_lambda * count_val
            ret_score = float(retrieval_scores[idx])
            aid = int(anime_ids[idx])
            
            # Tie breakers matching standard sort key
            if (adjusted_score > best_adjusted_score) or \
               (abs(adjusted_score - best_adjusted_score) < 1e-9 and ret_score > best_retrieval_score) or \
               (abs(adjusted_score - best_adjusted_score) < 1e-9 and abs(ret_score - best_retrieval_score) < 1e-9 and aid < best_anime_id):
                best_adjusted_score = adjusted_score
                best_retrieval_score = ret_score
                best_anime_id = aid
                best_idx = idx
                best_winning_seed = winning_seed
                best_base_score = base_score
                
        if best_idx is None:
            break
            
        selected_indices.append(best_idx)
        pool_indices.remove(best_idx)
        
        best_id = int(anime_ids[best_idx])
        
        # Audit metadata
        audit_dict[best_id] = {
            "winning_seed": int(best_winning_seed) if best_winning_seed is not None else None,
            "base_score": float(best_base_score),
            "adjusted_score": float(best_adjusted_score),
            "selected_count_before": int(selected_count_by_seed.get(best_winning_seed, 0)) if best_winning_seed is not None else 0,
        }

        # After selecting an item, increment counts
        if best_winning_seed is not None:
            selected_count_by_seed[best_winning_seed] += 1

    if recommender_obj is not None:
        recommender_obj._representation_penalty_audit = audit_dict

    return [int(anime_ids[idx]) for idx in selected_indices[:top_k]]
