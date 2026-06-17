from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd

from cinesense.recommenders.base import CineSenseHybridLiteWeighted
from cinesense.retrieval.hybrid_c import hybrid_c_retrieval_scores, top_retrieval_indices
from cinesense.ranking.weighted_b import weighted_max_similarity_to_train_items, rerank_candidates

DEFAULT_MODEL_NAME = "cinesense_two_stage"
DEFAULT_RETRIEVAL_CANDIDATES = 100


@dataclass
class CineSenseTwoStage(CineSenseHybridLiteWeighted):
    """Hybrid C retrieval followed by Weighted B reranking over retrieved candidates."""

    model_name: str = DEFAULT_MODEL_NAME
    semantic_weight: float = 0.85
    popularity_weight: float = 0.15
    rating_weight_scheme: str = "normalized"
    retrieval_candidate_count: int = DEFAULT_RETRIEVAL_CANDIDATES

    def fit(
        self,
        anime_catalog: pd.DataFrame,
        train_interactions: pd.DataFrame,
        user_ids: Iterable[int] | None = None,
    ) -> CineSenseTwoStage:
        """Fit shared embeddings, popularity scores, and normalized rating weights."""
        super().fit(anime_catalog, train_interactions, user_ids=user_ids)
        return self

    def recommend(
        self,
        anime_ids: list[int],
        ratings: dict[int, float] | None = None,
        top_k: int = 10,
    ) -> list[int]:
        """Retrieve with Hybrid C, then rerank retrieved candidates with Weighted B scoring.
        
        Stateless production API.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseTwoStage must be fitted before recommend().")
        if self.popularity_scores.size != len(self.anime_ids):
            raise RuntimeError("Popularity scores are not initialized correctly.")

        # Filter and map inputs to indices
        train_indices_list = []
        train_weights_list = []
        train_items = set()

        for item_id in anime_ids:
            if item_id not in self.item_id_to_index:
                continue
            train_indices_list.append(self.item_id_to_index[item_id])
            train_items.add(item_id)

            if ratings is not None and item_id in ratings:
                weight = self._rating_weight(int(ratings[item_id]))
            else:
                weight = 1.0
            train_weights_list.append(weight)

        train_indices = np.asarray(train_indices_list, dtype=np.int32)
        train_weights = np.asarray(train_weights_list, dtype=np.float32)

        if train_indices.size == 0:
            return []

        # Call Stage 1 retrieval
        retrieval_scores = hybrid_c_retrieval_scores(
            train_indices,
            self.catalog_embeddings,
            self.popularity_scores,
            self.semantic_weight,
            self.popularity_weight,
            self.seed_batch_size,
        )
        retrieved_indices = top_retrieval_indices(
            retrieval_scores,
            train_items,
            self.anime_ids,
            self.retrieval_candidate_count,
        )
        if retrieved_indices.size == 0:
            return []

        # Call Stage 2 ranking
        weighted_semantic_scores = weighted_max_similarity_to_train_items(
            train_indices,
            train_weights,
            self.catalog_embeddings,
            self.seed_batch_size,
        )
        rerank_scores = (
            self.semantic_weight * weighted_semantic_scores
            + self.popularity_weight * self.popularity_scores
        )

        return rerank_candidates(
            retrieved_indices,
            rerank_scores,
            retrieval_scores,
            self.anime_ids,
            top_k,
        )

    def recommend_for_user(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Retrieve with Hybrid C, then rerank retrieved candidates with Weighted B scoring.
        
        Evaluation compatibility API.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseTwoStage must be fitted before recommend().")
        if self.popularity_scores.size != len(self.anime_ids):
            raise RuntimeError("Popularity scores are not initialized correctly.")

        train_indices = np.asarray(
            [
                self.item_id_to_index[item_id]
                for item_id in train_items
                if item_id in self.item_id_to_index
            ],
            dtype=np.int32,
        )
        if train_indices.size == 0:
            return []

        # Call Stage 1 retrieval
        retrieval_scores = hybrid_c_retrieval_scores(
            train_indices,
            self.catalog_embeddings,
            self.popularity_scores,
            self.semantic_weight,
            self.popularity_weight,
            self.seed_batch_size,
        )
        retrieved_indices = top_retrieval_indices(
            retrieval_scores,
            train_items,
            self.anime_ids,
            self.retrieval_candidate_count,
        )
        if retrieved_indices.size == 0:
            return []

        # Call Stage 2 ranking
        train_indices, train_weights = self._weighted_train_indices(user_id, train_items)
        if train_indices.size == 0:
            return []

        weighted_semantic_scores = weighted_max_similarity_to_train_items(
            train_indices,
            train_weights,
            self.catalog_embeddings,
            self.seed_batch_size,
        )
        rerank_scores = (
            self.semantic_weight * weighted_semantic_scores
            + self.popularity_weight * self.popularity_scores
        )

        return rerank_candidates(
            retrieved_indices,
            rerank_scores,
            retrieval_scores,
            self.anime_ids,
            k,
        )
