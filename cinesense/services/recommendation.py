from __future__ import annotations

import re
import os
import json
import numpy as np
import pandas as pd
from cinesense.recommenders.two_stage import CineSenseTwoStage
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices
from cinesense.utils.franchise import load_franchise_aliases, get_canonical_franchise
from cinesense.utils.text import normalize_synopsis
from cinesense.utils.match_quality import get_match_quality


def get_franchise(title: str) -> str:
    """Heuristic to extract the base franchise name from an anime title."""
    return get_canonical_franchise(title)


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
        self.franchise_aliases = load_franchise_aliases()
        
        # Load theme rules from configuration
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        theme_rules_path = os.path.join(dir_path, "config", "theme_rules.json")
        if os.path.exists(theme_rules_path):
            with open(theme_rules_path, "r", encoding="utf-8") as f:
                self.theme_rules = json.load(f)
        else:
            self.theme_rules = {}

        # Build an in-memory O(1) dictionary mapping for catalog metadata lookup
        self.catalog_meta = {}
        for _, row in self.catalog_df.iterrows():
            anime_id = int(row["anime_id"])
            title = str(row.get("title", ""))
            title_english = str(row.get("title_english", "")) if pd.notna(row.get("title_english")) else ""
            synopsis = str(row.get("synopsis", "")) if pd.notna(row.get("synopsis")) else ""
            
            # Extract genres dynamically using theme rules mapping
            text = f"{title} {title_english} {synopsis}".lower()
            genres = [theme for theme, words in self.theme_rules.items() if any(w in text for w in words)]
            
            self.catalog_meta[anime_id] = {
                "title": title,
                "title_english": title_english if title_english else None,
                "synopsis": normalize_synopsis(synopsis),
                "genres": genres,
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
            "synopsis": normalize_synopsis(meta["synopsis"]),
            "genres": meta.get("genres", []),
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
        """Identifies seed contributions, computes multi-source relevance signals, and constructs explanations."""
        if not seed_ids:
            return {}

        rec_idx = self.recommender.item_id_to_index[recommended_id]
        emb_rec = self.recommender.catalog_embeddings[rec_idx]
        rec_meta = self.catalog_meta.get(recommended_id, {})
        rec_genres = rec_meta.get("genres", [])

        # 1. Compute combined relevance for each seed to determine the dominant matched seed
        best_seed = None
        max_relevance = -float("inf")
        seed_similarities = {}
        
        jaccard_vals = {}
        distance_vals = {}

        for seed_id in seed_ids:
            seed_idx = self.recommender.item_id_to_index[seed_id]
            emb_seed = self.recommender.catalog_embeddings[seed_idx]
            sim = float(np.dot(emb_rec, emb_seed))
            sim = max(-1.0, min(1.0, sim))
            seed_similarities[seed_id] = sim
            
            jac = self._lookup_jaccard(seed_id, recommended_id)
            dist = self._lookup_distance(seed_id, recommended_id)
            jaccard_vals[seed_id] = jac
            distance_vals[seed_id] = dist
            
            # Collaborative Proximity score contribution
            dist_score = 0.5 if dist == 1 else (1.0 / 3.0) if dist == 2 else 0.0
            
            # Combined relevance score for matched seed selection
            relevance = sim + 1.0 * jac + 0.3 * dist_score
            
            if relevance > max_relevance:
                max_relevance = relevance
                best_seed = seed_id

        pop_score = float(self.recommender.popularity_scores[rec_idx])
        best_seed_meta = self.catalog_meta.get(best_seed, {})
        best_seed_title = best_seed_meta.get("title", f"Anime {best_seed}")

        # 2. Compute shares
        pos_sims = {s_id: max(0.01, sim) for s_id, sim in seed_similarities.items()}
        sum_sims = sum(pos_sims.values())
        seed_shares = {}
        for s_id, val in pos_sims.items():
            seed_shares[s_id] = (val / sum_sims) if sum_sims > 0 else (1.0 / len(seed_ids))

        # 3. Gather signals for explanation generation with priority
        reasons_with_priority = []

        # Signal 1: Multi-seed co-watch overlap (highest priority if it exists)
        if len(seed_ids) >= 2:
            sorted_seeds = sorted(seed_ids, key=lambda s: seed_shares[s], reverse=True)
            s1, s2 = sorted_seeds[0], sorted_seeds[1]
            s1_title = self.catalog_meta.get(s1, {}).get("title", f"Anime {s1}")
            s2_title = self.catalog_meta.get(s2, {}).get("title", f"Anime {s2}")
            
            if distance_vals[s1] <= 2 and distance_vals[s2] <= 2:
                reasons_with_priority.append((1, f"Appears in the viewing patterns of users who enjoy both {s1_title} and {s2_title}"))
                reasons_with_priority.append((2, f"Frequently watched by fans of both {s1_title} and {s2_title}"))

        # Signal 2: Co-watch / Jaccard / distance signals
        for s_id in seed_ids:
            s_title = self.catalog_meta.get(s_id, {}).get("title", f"Anime {s_id}")
            jac = jaccard_vals[s_id]
            dist = distance_vals[s_id]
            
            if jac >= 0.05:
                reasons_with_priority.append((3, f"Frequently watched by {s_title} fans"))
            if dist == 1:
                reasons_with_priority.append((4, f"High collaborative relevance to {s_title}"))
            elif dist == 2:
                reasons_with_priority.append((5, f"Watched by users who enjoy {s_title}"))

        # Signal 3: Genre themes
        best_seed_genres = best_seed_meta.get("genres", [])
        shared_genres = list(set(rec_genres).intersection(best_seed_genres))
        if shared_genres:
            genre_str = " and ".join(shared_genres[:2])
            reasons_with_priority.append((6, f"Strong {genre_str} themes"))
            reasons_with_priority.append((7, f"Shares {genre_str} elements with {best_seed_title}"))

        # Signal 4: Semantic similarity
        max_sim = seed_similarities[best_seed]
        if max_sim >= 0.75:
            reasons_with_priority.append((8, f"Strong semantic similarity to {best_seed_title}"))
        elif max_sim >= 0.60:
            reasons_with_priority.append((8, f"High semantic similarity to {best_seed_title}"))
        else:
            reasons_with_priority.append((9, f"Semantic similarity to {best_seed_title}"))

        # Sort by priority ascending and deduplicate to get at most 3
        unique_reasons = []
        for priority, text in sorted(reasons_with_priority, key=lambda x: x[0]):
            if text not in unique_reasons:
                unique_reasons.append(text)
                if len(unique_reasons) == 3:
                    break

        if not unique_reasons:
            unique_reasons.append(f"Semantic similarity to {best_seed_title}")

        reason_str = unique_reasons[0]

        formatted_shares = {
            int(s_id): {
                "title": self.catalog_meta.get(s_id, {}).get("title", f"Anime {s_id}"),
                "share": float(share)
            }
            for s_id, share in seed_shares.items()
        }

        return {
            "matched_seed": {
                "anime_id": int(best_seed),
                "title": best_seed_title,
            },
            "similarity": max_sim,
            "popularity": pop_score,
            "reason": reason_str,
            "summary": reason_str,
            "reasons": unique_reasons,
            "seed_shares": formatted_shares
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
            
            # Calculate match_score and match_badge dynamically from raw score (Phase 4)
            scaled = 6.0 + (score - 0.2) * (3.5 / 0.6)
            match_score = round(max(1.0, min(10.0, scaled)), 1)
            
            match_badge = get_match_quality(match_score)

            item = {
                "anime_id": rec_id,
                "title": meta.get("title", f"Anime {rec_id}"),
                "title_english": meta.get("title_english"),
                "score": score,
                "match_score": match_score,
                "match_badge": match_badge,
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
        mode: str = "discover",
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
        if mode == "discover" and rerank_enabled and getattr(self.recommender, "graph_available", False):
            if user_id is not None:
                import zlib
                bucket = zlib.crc32(user_id.encode('utf-8')) % 100
                if bucket < self.rerank_config.traffic_percent:
                    is_treatment = True

        if is_treatment:
            self._increment_counter("ab_treatment_requests")
        else:
            self._increment_counter("ab_control_requests")

        if mode != "discover":
            # Similar mode (baseline logic)
            retrieval_k = top_k
            recommendations = self.recommender.recommend(
                anime_ids=valid_ids,
                ratings=validated_ratings,
                top_k=retrieval_k,
            )

            if not recommendations:
                return []

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
            scores = {
                rec_id: float(rerank_scores[self.recommender.item_id_to_index[rec_id]])
                for rec_id in recommendations
            }
            return self.enrich_recommendations(
                recommendations,
                scores,
                valid_ids,
                weighted_semantic_scores=weighted_semantic_scores,
            )

        # Stage 1: Retrieval (Unified for discover mode to support filtering)
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
        
        # Increase retrieval pool to ensure we have enough items after filtering
        retrieval_k = max(300, top_k * 10)
        retrieved_indices_raw = top_retrieval_indices(
            retrieval_scores,
            train_items,
            self.recommender.anime_ids,
            retrieval_k,
        )

        # Build seed franchise set using canonical mappings (Phase 1)
        seed_franchises = set()
        for aid in valid_ids:
            meta = self.catalog_meta.get(aid)
            if meta:
                seed_franchises.add(get_canonical_franchise(meta["title"]))
                if meta.get("title_english"):
                    seed_franchises.add(get_canonical_franchise(meta["title_english"]))

        # Apply retrieval-stage exclusion BEFORE final ranking in discover mode (Phase 1)
        retrieved_indices_prepared = []
        excluded_count = 0
        for idx in retrieved_indices_raw:
            anime_id = int(self.recommender.anime_ids[idx])
            meta = self.catalog_meta[anime_id]
            title = meta["title"]
            eng_title = meta.get("title_english") or ""
            cand_f = get_canonical_franchise(title)
            cand_f_eng = get_canonical_franchise(eng_title) if eng_title else ""

            if cand_f in seed_franchises or (cand_f_eng and cand_f_eng in seed_franchises):
                excluded_count += 1
                continue

            retrieved_indices_prepared.append(idx)
            if len(retrieved_indices_prepared) == 150:
                break

        self.candidate_audit = {
            "retrieved": len(retrieved_indices_raw),
            "excluded_seed_franchise": excluded_count,
            "remaining": len(retrieved_indices_prepared),
        }

        retrieved_indices = np.asarray(retrieved_indices_prepared, dtype=np.int32)
        if retrieved_indices.size == 0:
            return []

        # Stage 2: Ranking
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

        # 5. Apply discovery filtering
        if True:
            filtered_enriched = []
            seen_rec_franchises = set()

            for item in enriched:
                rec_id = item["anime_id"]
                rec_title = item["title"]
                rec_eng_title = item.get("title_english")

                # A. Filter out seed franchises (already mostly excluded, but keep for robustness)
                rec_f_name = get_canonical_franchise(rec_title)
                rec_f_eng_name = get_canonical_franchise(rec_eng_title) if rec_eng_title else ""

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
                    pop_pen = popularity_penalty * max(0.0, pop_pct - 0.95)
                    semantic_score = item["score"]
                    
                    # Phase 4 Runtime Safety Guards
                    jaccard_valid = (0.0 <= max_jaccard <= 1.0)
                    cosine_valid = (-1.0 <= cosine_sim <= 1.0)
                    dist_to_check = None if min_dist == 10 else min_dist
                    distance_valid = (dist_to_check in (1, 2, None))
                    
                    if not (jaccard_valid and cosine_valid and distance_valid):
                        rerank_score = semantic_score
                        max_jaccard = 0.0
                        distance_score = 0.0
                        cosine_sim = 0.0
                        pop_pen = 0.0
                    else:
                        rerank_score = (
                            semantic_score
                            + jaccard_weight * max_jaccard * (cosine_sim ** cosine_power)
                            + distance_weight * distance_score
                            - pop_pen
                        )
                    
                    # Recompute match score and badge using new reranked score (Phase 4)
                    scaled = 6.0 + (rerank_score - 0.2) * (3.5 / 0.6)
                    match_score = round(max(1.0, min(10.0, scaled)), 1)
                    match_badge = get_match_quality(match_score)

                    new_item = item.copy()
                    new_item["score"] = rerank_score
                    new_item["match_score"] = match_score
                    new_item["match_badge"] = match_badge
                    
                    # Update explanation reasons using updated reranked score parameters
                    explanation = self.generate_explanations(rec_id, valid_ids)
                    new_item["explanation"] = explanation
                    
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
                    reranked_pool.sort(key=lambda x: -x["score"])
                    enriched = reranked_pool[:top_k]
            else:
                enriched = filtered_enriched[:top_k]

        if not is_treatment:
            for item in enriched:
                old_exp = item.get("explanation", {})
                new_exp = old_exp.copy()
                new_exp["summary"] = old_exp.get("reason", "High semantic similarity")
                new_exp["reasons"] = ["High semantic similarity"]
                item["explanation"] = new_exp

        return enriched

