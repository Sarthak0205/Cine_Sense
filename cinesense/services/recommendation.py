from __future__ import annotations

import re
import os
import numpy as np
import pandas as pd
from cinesense.recommenders.two_stage import CineSenseTwoStage
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices


def get_franchise(title: str) -> str:
    """Heuristic to extract the base franchise name from an anime title."""
    title = str(title).lower().strip()
    
    # Custom manual overrides for known crossover/special titles
    overrides = {
        "attack on skytree": "attack on titan",
        "shingeki no kyotou": "shingeki no kyojin",
    }
    if title in overrides:
        return overrides[title]
    
    # Split on major delimiters like colons, dashes, parenthesis, exclamation marks
    title = re.split(r'[:\-\(!]', title)[0].strip()
    
    if title in overrides:
        return overrides[title]
    
    # Strip standard franchise suffixes/keywords
    title = re.sub(
        r'\b(season|movie|film|ova|ona|tv|specials|special|recap|pilot|part|chapter|edition|remaster|remake|the animation|rewrite|relight|summary|3d|ii|iii|iv|v|vi|vii|viii|ix|x|\d+st|\d+nd|\d+rd|\d+th|\d+)\b',
        '',
        title
    )
    # Remove double spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def is_sequel_title(title: str) -> bool:
    """Determines if a title indicates a sequel, second season, movie sequel, etc."""
    if not title:
        return False
    t_low = title.lower().strip()
    
    # 1. Explicit season/part numbers, Roman numerals II to X
    if re.search(r'\b(?:season|part|vol|volume|movie|ova|ona)\s*(?:[2-9]|\d{2,}|ii|iii|iv|v|vi|vii|viii|ix|x)\b', t_low):
        return True
        
    # Ordinal numbers: "2nd season", "3rd part", etc.
    if re.search(r'\b(?:2nd|3rd|[4-9]th|\d{2,}th)\s*(?:season|part|movie|ova|ona|series)\b', t_low):
        return True
        
    # Standalone Roman numerals (II to X)
    if re.search(r'\b(?:ii|iii|iv|v|vi|vii|viii|ix|x)\b', t_low):
        return True
        
    # x2, x3, etc. (like Durarara!!x2)
    if re.search(r'\bx[2-9]\b', t_low):
        return True
        
    # r2, r3, etc. (like Code Geass R2)
    if re.search(r'\br[2-9]\b', t_low):
        return True
        
    # s2, s3, etc. (like Zero no Tsukaima S2)
    if re.search(r'\bs[2-9]\b', t_low):
        return True
        
    # 2. Trailing numbers preceded by spaces or delimiters (e.g. "Title 2", "Title 3")
    if re.search(r'[\s:!\?\-]+(?:[2-9]|\d{2,})$', t_low):
        return True
        
    return False


class RecommendationService:
    """Production service coordinating recommendation generation, input validation, and explanations."""

    def __init__(self, recommender: CineSenseTwoStage, catalog_df: pd.DataFrame, rerank_config=None, telemetry=None):
        self.recommender = recommender
        self.catalog_df = catalog_df
        from cinesense.config.graph_rerank import GraphRerankConfig
        self.rerank_config = rerank_config or GraphRerankConfig.from_env()
        self.telemetry = telemetry

        # Build an in-memory O(1) dictionary mapping for catalog metadata lookup
        self.catalog_meta = {}
        for _, row in self.catalog_df.iterrows():
            anime_id = int(row["anime_id"])
            self.catalog_meta[anime_id] = {
                "title": str(row.get("title", "")),
                "title_english": str(row.get("title_english", "")) if pd.notna(row.get("title_english")) else None,
                "synopsis": str(row.get("synopsis", "")) if pd.notna(row.get("synopsis")) else None,
            }

    def _increment_counter(self, name: str) -> None:
        if self.telemetry is not None:
            val = getattr(self.telemetry, name, 0)
            setattr(self.telemetry, name, val + 1)

    def _log_warning_rate_limited(self, category: str, message: str) -> None:
        if not hasattr(self, "_warning_counts"):
            self._warning_counts = {}
        count = self._warning_counts.get(category, 0)
        self._warning_counts[category] = count + 1
        if count == 0:
            import sys
            print(f"WARNING: [Telemetry Category: {category}] {message}", file=sys.stderr)
        elif count % 100 == 0:
            import sys
            print(f"WARNING: [Telemetry Category: {category}] {message} (suppressed {count - 1} times)", file=sys.stderr)

    def search_anime(self, query: str, limit: int = 20) -> list[dict]:
        """Case-insensitive, partial substring search on title and title_english."""
        if not query:
            return []

        query_lower = query.lower()
        results = []
        for aid, meta in self.catalog_meta.items():
            title_lower = meta["title"].lower()
            title_eng_lower = (meta["title_english"] or "").lower()

            if query_lower in title_lower or query_lower in title_eng_lower:
                results.append({
                    "anime_id": aid,
                    "title": meta["title"],
                    "title_english": meta["title_english"],
                })
                if len(results) == limit:
                    break
        return results

    def get_anime_details(self, anime_id: int) -> dict | None:
        """Fetch details for a specific anime ID."""
        meta = self.catalog_meta.get(anime_id)
        if meta is None:
            return None
        return {
            "anime_id": anime_id,
            "title": meta["title"],
            "title_english": meta["title_english"],
            "synopsis": meta["synopsis"],
        }

    def validate_inputs(
        self,
        anime_ids: list[int],
        ratings: dict[int, float] | None = None,
        top_k: int = 10,
    ) -> tuple[list[int], dict[int, float]]:
        """Deduplicates input IDs, filters out unknown items, and validates that ratings are correct types and bounds."""
        if not isinstance(anime_ids, list):
            raise TypeError("anime_ids must be a list of integers.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # Deduplicate and cast keys to standard Python integers
        seen = set()
        deduped_ids = []
        for aid in anime_ids:
            if not isinstance(aid, (int, np.integer)) or isinstance(aid, bool):
                raise TypeError(f"Anime ID '{aid}' must be an integer.")
            aid_int = int(aid)
            if aid_int not in seen:
                seen.add(aid_int)
                deduped_ids.append(aid_int)

        # Filter out IDs not present in the catalog / model index
        valid_ids = [aid for aid in deduped_ids if aid in self.recommender.item_id_to_index]

        validated_ratings = {}
        if ratings is not None:
            if not isinstance(ratings, dict):
                raise TypeError("ratings must be a dictionary or None.")
            for aid, score in ratings.items():
                if not isinstance(aid, (int, np.integer)) or isinstance(aid, bool):
                    raise TypeError(f"Rating key '{aid}' must be an integer.")
                aid_int = int(aid)

                # Skip ratings for items that are not in the validated seed list
                if aid_int not in seen:
                    continue

                if not isinstance(score, (int, float, np.number)) or isinstance(score, bool):
                    raise TypeError(f"Rating for anime {aid_int} must be a number.")

                score_val = float(score)
                if not (1.0 <= score_val <= 10.0):
                    raise ValueError(f"Rating for anime {aid_int} must be between 1.0 and 10.0.")

                validated_ratings[aid_int] = score_val

        return valid_ids, validated_ratings

    def generate_explanations(self, recommended_id: int, seed_ids: list[int]) -> dict:
        """Identifies the strongest matching seed anime based on cosine similarity and constructs an explanation."""
        if not seed_ids:
            return {}

        rec_idx = self.recommender.item_id_to_index[recommended_id]
        emb_rec = self.recommender.catalog_embeddings[rec_idx]

        best_seed = None
        max_sim = -float("inf")

        for seed_id in seed_ids:
            seed_idx = self.recommender.item_id_to_index[seed_id]
            emb_seed = self.recommender.catalog_embeddings[seed_idx]
            # Since embeddings are pre-normalized, dot product equals cosine similarity
            sim = float(np.dot(emb_rec, emb_seed))
            if sim > max_sim:
                max_sim = sim
                best_seed = seed_id

        pop_score = float(self.recommender.popularity_scores[rec_idx])
        best_seed_meta = self.catalog_meta.get(best_seed, {})
        best_seed_title = best_seed_meta.get("title", f"Anime {best_seed}")

        reason = f"Recommended because it is highly similar to '{best_seed_title}'."

        return {
            "matched_seed": {
                "anime_id": int(best_seed),
                "title": best_seed_title,
            },
            "similarity": max_sim,
            "popularity": pop_score,
            "reason": reason,
        }

    def is_sequel_title(self, title: str) -> bool:
        """Determines if a title indicates a sequel, second season, movie sequel, etc. (for backward compatibility)"""
        return is_sequel_title(title)

    def get_franchise_root(self, franchise_name: str) -> int | None:
        """Finds the root anime ID (usually first season) for a given franchise name."""
        recommender = self.recommender
        if hasattr(recommender, "franchise_name_to_root"):
            return recommender.franchise_name_to_root.get(franchise_name)

        if not hasattr(self, "_franchise_root_cache"):
            self._franchise_root_cache = {}
        if franchise_name in self._franchise_root_cache:
            return self._franchise_root_cache[franchise_name]

        candidates = []
        for aid, meta in self.catalog_meta.items():
            f_title = get_franchise(meta["title"])
            f_eng = get_franchise(meta["title_english"]) if meta["title_english"] else ""
            if f_title == franchise_name or f_eng == franchise_name:
                candidates.append((aid, meta["title"]))
                
        if not candidates:
            self._franchise_root_cache[franchise_name] = None
            return None
            
        # Filter candidates to find those that are NOT sequels
        non_sequels = []
        for aid, title in candidates:
            if not is_sequel_title(title):
                non_sequels.append((aid, title))
                
        if non_sequels:
            # Sort by title length first, then by popularity score
            non_sequels.sort(key=lambda x: (len(x[1]), -self.recommender.popularity_scores[self.recommender.item_id_to_index[x[0]]]))
            res = non_sequels[0][0]
        else:
            # If all are detected as sequels, pick the one with the shortest title
            candidates.sort(key=lambda x: (len(x[1]), -self.recommender.popularity_scores[self.recommender.item_id_to_index[x[0]]]))
            res = candidates[0][0]
            
        self._franchise_root_cache[franchise_name] = res
        return res

    def enrich_recommendations(
        self,
        recommendations: list[int],
        scores: dict[int, float],
        seed_ids: list[int],
        weighted_semantic_scores: np.ndarray | None = None,
    ) -> list[dict]:
        """Maps recommended IDs to their catalog details and attaches explanation reasoning and internal audit instrumentation."""
        enriched = []
        for rec_id in recommendations:
            meta = self.catalog_meta.get(rec_id, {})
            score = scores.get(rec_id, 0.0)
            explanation = self.generate_explanations(rec_id, seed_ids)
            
            # Map standard reason for backward compatibility
            reason_str = explanation.get("reason", "")
            explanation["summary"] = reason_str
            explanation["reasons"] = ["High semantic similarity"]

            item = {
                "anime_id": rec_id,
                "title": meta.get("title", f"Anime {rec_id}"),
                "title_english": meta.get("title_english"),
                "score": score,
                "explanation": explanation,
            }
            
            # Internal audit instrumentation (not exposed in Pydantic REST response)
            if weighted_semantic_scores is not None and rec_id in self.recommender.item_id_to_index:
                rec_idx = self.recommender.item_id_to_index[rec_id]
                semantic = float(self.recommender.semantic_weight * weighted_semantic_scores[rec_idx])
                popularity = float(self.recommender.popularity_weight * self.recommender.popularity_scores[rec_idx])
                item["_audit"] = {
                    "semantic_component": semantic,
                    "popularity_component": popularity,
                    "final_score": score,
                }
                
            enriched.append(item)
        return enriched

    def _lookup_jaccard(self, s_id: int, c_id: int) -> float:
        """Looks up the precomputed Jaccard similarity between a seed and candidate using binary search."""
        recommender = self.recommender
        if not getattr(recommender, "graph_available", False):
            return 0.0
        
        idx_s = recommender.anime_to_idx.get(s_id)
        idx_c = recommender.anime_to_idx.get(c_id)
        if idx_s is None or idx_c is None:
            self._increment_counter("graph_lookup_failures")
            self._log_warning_rate_limited("graph_lookup_failures", f"Seed {s_id} or candidate {c_id} not found in Jaccard neighbor graph.")
            return 0.0
            
        row_ids = recommender.neighbor_ids[idx_s]
        pos = np.searchsorted(row_ids, c_id)
        if pos < len(row_ids) and row_ids[pos] == c_id:
            return float(recommender.neighbor_jaccards[idx_s][pos])
        return 0.0

    def _lookup_distance(self, s_id: int, c_id: int) -> int:
        """Looks up the precomputed collaborative distance (1, 2, or disconnected=10) between a seed and candidate."""
        recommender = self.recommender
        if not getattr(recommender, "graph_available", False):
            return 10
            
        idx_s = recommender.anime_to_idx.get(s_id)
        idx_c = recommender.anime_to_idx.get(c_id)
        if idx_s is None or idx_c is None:
            self._increment_counter("graph_lookup_failures")
            self._log_warning_rate_limited("graph_lookup_failures", f"Seed {s_id} or candidate {c_id} not found in distance lookup graph.")
            return 10
            
        dist = recommender.distance_lookup[idx_s, idx_c]
        if dist > 0:
            return int(dist)
        return 10

    def recommend(
        self,
        anime_ids: list[int],
        ratings: dict[int, float] | None = None,
        top_k: int = 10,
        mode: str = "similar",
        user_id: str | None = None,
    ) -> list[dict]:
        """Runs the entire recommendation workflow: validation, inference, score calculation, and enrichment."""
        # 1. Validate inputs
        valid_ids, validated_ratings = self.validate_inputs(anime_ids, ratings, top_k)
        if not valid_ids:
            return []

        # Determine routing path for A/B testing
        rerank_enabled = self.rerank_config.rerank_enabled
        is_treatment = False
        if mode == "discover" and rerank_enabled and getattr(self.recommender, "graph_available", False) and user_id is not None:
            import zlib
            bucket = zlib.crc32(user_id.encode('utf-8')) % 100
            if bucket < self.rerank_config.traffic_percent:
                is_treatment = True

        if is_treatment:
            self._increment_counter("ab_treatment_requests")
        else:
            self._increment_counter("ab_control_requests")

        if mode == "discover":
            # Step 1: Increase retrieval pool
            retrieval_k = max(300, top_k * 10)

            # Stage 1 retrieval
            train_indices = np.asarray([self.recommender.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
            train_items = set(valid_ids)

            retrieval_scores = hybrid_c_retrieval_scores(
                train_indices,
                self.recommender.catalog_embeddings,
                self.recommender.popularity_scores,
                self.recommender.semantic_weight,
                self.recommender.popularity_weight,
                self.recommender.seed_batch_size,
            )
            retrieved_indices_raw = top_retrieval_indices(
                retrieval_scores,
                train_items,
                self.recommender.anime_ids,
                retrieval_k,
            )

            # Step 2: Build seed franchise set
            seed_franchises = set()
            for aid in valid_ids:
                meta = self.catalog_meta.get(aid)
                if meta:
                    seed_franchises.add(get_franchise(meta["title"]))
                    if meta.get("title_english"):
                        seed_franchises.add(get_franchise(meta["title_english"]))

            # Step 3: Apply retrieval-stage exclusion
            retrieved_indices_prepared = []
            excluded_count = 0
            for idx in retrieved_indices_raw:
                anime_id = int(self.recommender.anime_ids[idx])
                meta = self.catalog_meta[anime_id]
                title = meta["title"]
                eng_title = meta.get("title_english") or ""
                cand_f = get_franchise(title)
                cand_f_eng = get_franchise(eng_title) if eng_title else ""

                if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                    excluded_count += 1
                    continue

                retrieved_indices_prepared.append(idx)
                if len(retrieved_indices_prepared) == 150:
                    break

            # Step 5: Add audit logging structure
            self.candidate_audit = {
                "retrieved": len(retrieved_indices_raw),
                "excluded_seed_franchise": excluded_count,
                "remaining": len(retrieved_indices_prepared),
            }

            retrieved_indices = np.asarray(retrieved_indices_prepared, dtype=np.int32)

            # Stage 2 ranking
            train_weights = np.asarray([
                self.recommender._rating_weight(int(validated_ratings[aid])) if aid in validated_ratings else 1.0
                for aid in valid_ids
            ], dtype=np.float32)

            weighted_semantic_scores = weighted_max_similarity_to_train_items(
                train_indices,
                train_weights,
                self.recommender.catalog_embeddings,
                self.recommender.seed_batch_size,
            )
            rerank_scores = (
                self.recommender.semantic_weight * weighted_semantic_scores
                + self.recommender.popularity_weight * self.recommender.popularity_scores
            )

            # Rerank candidates
            rep_penalty = self.rerank_config.representation_penalty
            rep_lambda = self.rerank_config.representation_lambda

            recommendations = rerank_candidates(
                retrieved_indices,
                rerank_scores,
                retrieval_scores,
                self.recommender.anime_ids,
                150,  # Keep up to 150 ranked items for subsequent filters
                representation_penalty=rep_penalty,
                representation_lambda=rep_lambda,
                train_indices=train_indices,
                catalog_embeddings=self.recommender.catalog_embeddings,
            )
        else:
            # Similar mode (baseline logic)
            retrieval_k = top_k
            recommendations = self.recommender.recommend(
                anime_ids=valid_ids,
                ratings=validated_ratings,
                top_k=retrieval_k,
            )

            train_indices = np.asarray([self.recommender.item_id_to_index[aid] for aid in valid_ids], dtype=np.int32)
            train_weights = np.asarray([
                self.recommender._rating_weight(int(validated_ratings[aid])) if aid in validated_ratings else 1.0
                for aid in valid_ids
            ], dtype=np.float32)

            weighted_semantic_scores = weighted_max_similarity_to_train_items(
                train_indices,
                train_weights,
                self.recommender.catalog_embeddings,
                self.recommender.seed_batch_size,
            )
            rerank_scores = (
                self.recommender.semantic_weight * weighted_semantic_scores
                + self.recommender.popularity_weight * self.recommender.popularity_scores
            )

        if not recommendations:
            return []

        scores = {
            rec_id: float(rerank_scores[self.recommender.item_id_to_index[rec_id]])
            for rec_id in recommendations
        }

        # 4. Enrich recommendations with metadata, explanations, and audit stats
        enriched = self.enrich_recommendations(
            recommendations,
            scores,
            valid_ids,
            weighted_semantic_scores=weighted_semantic_scores,
        )

        # 5. Apply discovery filtering if mode == "discover"
        if mode == "discover":
            seed_franchises = set()
            for aid in valid_ids:
                meta = self.catalog_meta.get(aid)
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

                # A. Filter out seed franchises (already mostly excluded, but keep for robustness)
                rec_f_name = get_franchise(rec_title)
                rec_f_eng_name = get_franchise(rec_eng_title) if rec_eng_title else ""

                if rec_f_name in seed_franchises or (rec_f_eng_name and rec_f_eng_name in seed_franchises):
                    continue

                # B. Sequel Filtering (discard if sequel)
                root_id = self.get_franchise_root(rec_f_name)
                is_sequel = False
                if root_id is not None and rec_id != root_id:
                    is_sequel = True
                elif self.is_sequel_title(rec_title) or (rec_eng_title and self.is_sequel_title(rec_eng_title)):
                    is_sequel = True

                if is_sequel:
                    continue

                # C. Franchise Deduplication (Max-One-Per-Franchise)
                if rec_f_name in seen_rec_franchises or (rec_f_eng_name and rec_f_eng_name in seen_rec_franchises):
                    continue

                filtered_enriched.append(item)
                seen_rec_franchises.add(rec_f_name)
                if rec_f_eng_name:
                    seen_rec_franchises.add(rec_f_eng_name)

            # Check if Jaccard Reranking is enabled and available from centralized config
            jaccard_weight = self.rerank_config.jaccard_weight
            distance_weight = self.rerank_config.distance_weight
            cosine_power = self.rerank_config.cosine_power
            popularity_penalty = self.rerank_config.popularity_penalty

            if is_treatment:
                candidates_pool = filtered_enriched[:100]
                reranked_pool = []
                
                # Precompute seed embeddings
                seed_embeddings = []
                for s_id in valid_ids:
                    s_idx = self.recommender.item_id_to_index[s_id]
                    seed_embeddings.append(self.recommender.catalog_embeddings[s_idx])
                
                for item in candidates_pool:
                    rec_id = item["anime_id"]
                    rec_idx = self.recommender.item_id_to_index[rec_id]
                    emb_rec = self.recommender.catalog_embeddings[rec_idx]
                    
                    # Compute max Cosine similarity
                    cosine_sim = -1.0
                    for emb_s in seed_embeddings:
                        sim = float(np.dot(emb_rec, emb_s))
                        if sim > cosine_sim:
                            cosine_sim = sim
                            
                    # Compute max Jaccard similarity
                    max_jaccard = 0.0
                    for s_id in valid_ids:
                        jac = self._lookup_jaccard(s_id, rec_id)
                        if jac > max_jaccard:
                            max_jaccard = jac
                            
                    # Compute min Distance and Distance Score
                    min_dist = 10
                    for s_id in valid_ids:
                        dist = self._lookup_distance(s_id, rec_id)
                        if dist < min_dist:
                            min_dist = dist
                    
                    if min_dist == 1:
                        distance_score = 0.5
                    elif min_dist == 2:
                        distance_score = 1.0 / 3.0
                    else:
                        distance_score = 0.0
                        
                    # Compute Popularity Percentile
                    pop_pct = float(self.recommender.pop_percentiles[rec_idx])
                    
                    # Popularity Penalty
                    pop_pen = popularity_penalty * max(0.0, pop_pct - 0.95)
                    
                    # Semantic Score (item score)
                    semantic_score = item["score"]
                    
                    # Phase 4 Runtime Safety Guards
                    # Jaccard bounds check: 0.0 <= jaccard <= 1.0
                    jaccard_valid = (0.0 <= max_jaccard <= 1.0)
                    # Cosine bounds check: -1.0 <= cosine <= 1.0
                    cosine_valid = (-1.0 <= cosine_sim <= 1.0)
                    # Distance check: distance in {1, 2, None}
                    dist_to_check = None if min_dist == 10 else min_dist
                    distance_valid = (dist_to_check in (1, 2, None))
                    
                    if not (jaccard_valid and cosine_valid and distance_valid):
                        # Graceful fallback: disable graph contribution for this candidate
                        rerank_score = semantic_score
                        max_jaccard = 0.0
                        distance_score = 0.0
                        cosine_sim = 0.0
                        pop_pen = 0.0
                    else:
                        # Apply reranking formula
                        rerank_score = (
                            semantic_score
                            + jaccard_weight * max_jaccard * (cosine_sim ** cosine_power)
                            + distance_weight * distance_score
                            - pop_pen
                        )
                    
                    # Phase 5 structured reasons
                    reasons = []
                    if cosine_sim >= 0.60:
                        reasons.append("High semantic similarity")
                    if max_jaccard >= 0.10:
                        reasons.append("Strong co-watch overlap")
                    if min_dist in (1, 2):
                        reasons.append("Frequently watched by similar users")
                        
                    if not reasons:
                        reasons.append("High semantic similarity")
                        
                    # Generate human-readable summary combining reasons
                    summary_parts = []
                    if "High semantic similarity" in reasons:
                        summary_parts.append("semantically similar")
                    if "Strong co-watch overlap" in reasons:
                        summary_parts.append("frequently co-watched")
                    if "Frequently watched by similar users" in reasons:
                        summary_parts.append("watched by similar users")
                        
                    if len(summary_parts) == 3:
                        summary = f"Recommended because it is {summary_parts[0]}, {summary_parts[1]}, and {summary_parts[2]}."
                    elif len(summary_parts) == 2:
                        summary = f"Recommended because it is {summary_parts[0]} and {summary_parts[1]}."
                    else:
                        summary = f"Recommended because it is {summary_parts[0]}."
                    
                    # Attach reranked score and copy items
                    new_item = item.copy()
                    new_item["score"] = rerank_score
                    
                    # Copy and update explanation dict
                    old_exp = item.get("explanation", {})
                    new_exp = old_exp.copy()
                    new_exp["summary"] = summary
                    new_exp["reasons"] = reasons
                    new_item["explanation"] = new_exp
                    
                    if "_audit" in new_item:
                        new_item["_audit"] = new_item["_audit"].copy()
                        new_item["_audit"].update({
                            "semantic_score": semantic_score,
                            "cosine_similarity": cosine_sim,
                            "jaccard_similarity": max_jaccard,
                            "distance_score": distance_score,
                            "popularity_percentile": pop_pct,
                            "popularity_penalty": pop_pen,
                            "graph_rerank_score": rerank_score,
                        })
                    reranked_pool.append(new_item)
                    
                # Check representation penalty config
                rep_penalty = self.rerank_config.representation_penalty
                rep_lambda = self.rerank_config.representation_lambda
                        
                if rep_penalty and len(valid_ids) > 1:
                    selected_count_by_seed = {s_id: 0 for s_id in valid_ids}
                    pool = list(reranked_pool)
                    selected_recs = []
                    
                    while pool and len(selected_recs) < top_k:
                        best_item = None
                        best_adjusted_score = -float('inf')
                        
                        for item in pool:
                            base_score = item["score"]
                            winning_seed = item.get("explanation", {}).get("matched_seed", {}).get("anime_id")
                            
                            count_val = selected_count_by_seed.get(winning_seed, 0) if winning_seed is not None else 0
                            adjusted_score = base_score - rep_lambda * count_val
                            
                            if adjusted_score > best_adjusted_score:
                                best_adjusted_score = adjusted_score
                                best_item = item
                                
                        if best_item is None:
                            break
                            
                        selected_recs.append(best_item)
                        pool.remove(best_item)
                        
                        winning_seed = best_item.get("explanation", {}).get("matched_seed", {}).get("anime_id")
                        if winning_seed is not None:
                            selected_count_by_seed[winning_seed] += 1
                            
                    enriched = selected_recs
                else:
                    # Sort pool by the new rerank_score descending
                    reranked_pool.sort(key=lambda x: -x["score"])
                    enriched = reranked_pool[:top_k]
            else:
                enriched = filtered_enriched[:top_k]

        return enriched

