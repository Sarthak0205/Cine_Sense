from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.cinesense_hybrid_lite_weighted import CineSenseHybridLiteWeighted


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
    ) -> "CineSenseTwoStage":
        """Fit shared embeddings, popularity scores, and normalized rating weights."""

        super().fit(anime_catalog, train_interactions, user_ids=user_ids)
        return self

    def recommend(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Retrieve with Hybrid C, then rerank retrieved candidates with Weighted B scoring."""

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

        retrieval_scores = self._hybrid_c_retrieval_scores(train_indices)
        retrieved_indices = self._top_retrieval_indices(retrieval_scores, train_items)
        if retrieved_indices.size == 0:
            return []

        train_indices, train_weights = self._weighted_train_indices(user_id, train_items)
        if train_indices.size == 0:
            return []

        weighted_semantic_scores = self._weighted_max_similarity_to_train_items(
            train_indices,
            train_weights,
        )
        rerank_scores = (
            self.semantic_weight * weighted_semantic_scores
            + self.popularity_weight * self.popularity_scores
        )

        reranked_indices = sorted(
            retrieved_indices.tolist(),
            key=lambda index: (-float(rerank_scores[index]), -float(retrieval_scores[index]), int(self.anime_ids[index])),
        )
        return [int(self.anime_ids[index]) for index in reranked_indices[:k]]

    def _hybrid_c_retrieval_scores(self, train_indices: np.ndarray) -> np.ndarray:
        semantic_scores = self._max_similarity_to_train_items(train_indices)
        return (
            self.semantic_weight * semantic_scores
            + self.popularity_weight * self.popularity_scores
        )

    def _top_retrieval_indices(self, retrieval_scores: np.ndarray, train_items: set[int]) -> np.ndarray:
        ranked_indices = np.argsort(-retrieval_scores, kind="mergesort")
        excluded_items = set(train_items)
        retrieved_indices = []

        for index in ranked_indices:
            anime_id = int(self.anime_ids[index])
            if anime_id in excluded_items:
                continue

            retrieved_indices.append(int(index))
            if len(retrieved_indices) == self.retrieval_candidate_count:
                break

        return np.asarray(retrieved_indices, dtype=np.int32)
