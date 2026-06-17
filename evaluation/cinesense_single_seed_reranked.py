from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from evaluation.cinesense_single_seed_max import CineSenseSingleSeedMaxRecommender


DEFAULT_MODEL_NAME = "cinesense_single_seed_max_reranked"
DEFAULT_CANDIDATE_POOL_SIZE = 500

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "when",
    "where",
    "who",
    "with",
    "after",
    "before",
    "into",
    "over",
    "about",
    "while",
    "which",
    "being",
    "been",
    "will",
    "can",
    "one",
    "two",
    "new",
    "now",
    "young",
    "life",
    "world",
    "time",
    "day",
    "find",
    "must",
    "however",
    "become",
    "anime",
    "series",
    "movie",
    "season",
    "special",
    "episode",
    "ova",
    "ona",
    "some",
    "also",
    "because",
    "could",
    "would",
    "should",
    "might",
    "many",
    "every",
    "each",
    "even",
    "only",
    "both",
    "such",
    "through",
    "together",
    "during",
    "more",
    "most",
    "once",
    "upon",
    "known",
    "what",
    "against",
    "these",
    "those",
    "alongside",
    "another",
    "around",
}

THEME_RULES = {
    "psychological": {
        "psychological",
        "mind",
        "conscience",
        "trauma",
        "identity",
        "memory",
        "mystery",
        "detective",
        "murder",
        "killer",
        "criminal",
        "crime",
        "pursuit",
        "monster",
        "genius",
        "strategy",
        "manipulation",
        "suspense",
    },
    "crime": {
        "crime",
        "criminal",
        "detective",
        "murder",
        "killer",
        "police",
        "investigation",
        "case",
        "justice",
        "pursuit",
        "evidence",
        "victim",
        "law",
        "trial",
        "conspiracy",
    },
    "war": {
        "war",
        "military",
        "army",
        "soldier",
        "battle",
        "battlefield",
        "oppression",
        "survival",
        "walls",
        "invasion",
        "rebellion",
        "resistance",
        "humanity",
        "weapon",
        "corps",
    },
    "romance": {
        "romance",
        "romantic",
        "love",
        "fate",
        "heart",
        "relationship",
        "couple",
        "feelings",
        "emotional",
        "dream",
        "body",
        "swap",
        "supernatural",
        "confession",
        "date",
    },
    "sci-fi": {
        "scientific",
        "science",
        "future",
        "technology",
        "experiment",
        "lab",
        "time",
        "travel",
        "space",
        "robot",
        "mecha",
        "dimension",
        "microwave",
        "hacker",
        "machine",
        "alien",
    },
    "horror": {
        "horror",
        "ghoul",
        "gore",
        "blood",
        "flesh",
        "monster",
        "demon",
        "vampire",
        "curse",
        "fear",
        "creature",
        "supernatural",
        "dark",
        "terror",
        "ghost",
    },
    "action": {
        "action",
        "battle",
        "fight",
        "fighting",
        "combat",
        "power",
        "warrior",
        "soldier",
        "army",
        "military",
        "weapon",
        "attack",
        "mission",
        "survival",
    },
    "adventure": {
        "adventure",
        "journey",
        "travel",
        "crew",
        "pirate",
        "treasure",
        "island",
        "quest",
        "explore",
        "world",
        "ship",
        "sea",
        "grand",
        "kingdom",
    },
    "fantasy": {
        "fantasy",
        "magic",
        "magical",
        "demon",
        "dragon",
        "kingdom",
        "curse",
        "spirit",
        "supernatural",
        "wizard",
        "sword",
        "myth",
        "god",
        "beast",
    },
    "historical": {
        "historical",
        "history",
        "era",
        "samurai",
        "warrior",
        "viking",
        "medieval",
        "ancient",
        "kingdom",
        "empire",
        "japan",
        "england",
        "denmark",
        "thorfinn",
        "revenge",
    },
    "thriller": {
        "thriller",
        "suspense",
        "mystery",
        "death",
        "chase",
        "danger",
        "conspiracy",
        "secret",
        "murder",
        "killer",
        "survival",
        "terror",
        "mind",
        "game",
    },
}

THEME_PRIORITY = {
    "psychological": 3,
    "crime": 3,
    "war": 3,
    "adventure": 3,
    "historical": 3,
    "thriller": 2,
    "action": 2,
    "sci-fi": 2,
    "fantasy": 1,
    "horror": 1,
    "romance": 2,
}


@dataclass
class CineSenseSingleSeedMaxReranked(CineSenseSingleSeedMaxRecommender):
    """Single-seed max retrieval with notebook-style keyword, bigram, and theme reranking."""

    model_name: str = DEFAULT_MODEL_NAME
    semantic_weight: float = 0.60
    keyword_weight: float = 0.25
    theme_weight: float = 0.15
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE
    item_features: list[dict[str, set[str]]] = field(default_factory=list, init=False, repr=False)

    def fit(self, *args, **kwargs) -> "CineSenseSingleSeedMaxReranked":
        """Fit embeddings and precompute text features used by the reranker."""

        super().fit(*args, **kwargs)
        self.item_features = [_extract_features(tags) for tags in self.catalog["tags"].tolist()]
        return self

    def recommend(self, user_id: int, train_items: set[int], k: int) -> list[int]:
        """Recommend ranked anime IDs after reranking top semantic candidates."""

        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseSingleSeedMaxReranked must be fitted before recommend().")

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

        semantic_scores, best_seed_indices = self._semantic_scores_and_best_seeds(train_indices)
        candidate_indices = self._top_candidate_indices(semantic_scores, train_items)
        if candidate_indices.size == 0:
            return []

        reranked = []
        for candidate_index in candidate_indices:
            seed_index = int(best_seed_indices[candidate_index])
            keyword_score = self._keyword_overlap_score(seed_index, int(candidate_index))
            theme_score = self._theme_score(seed_index, int(candidate_index))
            semantic_score = float(semantic_scores[candidate_index])
            final_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_score
                + self.theme_weight * theme_score
            )
            reranked.append((final_score, semantic_score, int(self.anime_ids[candidate_index])))

        reranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [anime_id for _, _, anime_id in reranked[:k]]

    def _semantic_scores_and_best_seeds(
        self,
        train_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.catalog_embeddings is None:
            raise RuntimeError("CineSenseSingleSeedMaxReranked must be fitted before scoring.")

        max_scores = np.full(len(self.anime_ids), -np.inf, dtype=np.float32)
        best_seed_indices = np.full(len(self.anime_ids), -1, dtype=np.int32)
        batch_size = max(1, self.seed_batch_size)

        for start in range(0, train_indices.size, batch_size):
            batch_indices = train_indices[start : start + batch_size]
            train_embeddings = self.catalog_embeddings[batch_indices]
            batch_scores = self.catalog_embeddings @ train_embeddings.T
            batch_best_positions = np.argmax(batch_scores, axis=1)
            batch_max_scores = batch_scores[
                np.arange(batch_scores.shape[0]),
                batch_best_positions,
            ]
            update_mask = batch_max_scores > max_scores
            max_scores[update_mask] = batch_max_scores[update_mask]
            best_seed_indices[update_mask] = batch_indices[batch_best_positions[update_mask]]

        return max_scores, best_seed_indices

    def _top_candidate_indices(self, semantic_scores: np.ndarray, train_items: set[int]) -> np.ndarray:
        ranked_indices = np.argsort(-semantic_scores, kind="mergesort")
        excluded_items = set(train_items)
        candidate_indices = []

        for index in ranked_indices:
            anime_id = int(self.anime_ids[index])
            if anime_id in excluded_items:
                continue

            candidate_indices.append(int(index))
            if len(candidate_indices) == self.candidate_pool_size:
                break

        return np.asarray(candidate_indices, dtype=np.int32)

    def _keyword_overlap_score(self, seed_index: int, candidate_index: int) -> float:
        seed_features = self.item_features[seed_index]
        candidate_features = self.item_features[candidate_index]

        keyword_score = _overlap_ratio(seed_features["keywords"], candidate_features["keywords"])
        bigram_score = _overlap_ratio(seed_features["bigrams"], candidate_features["bigrams"])
        return (0.65 * keyword_score) + (0.35 * bigram_score)

    def _theme_score(self, seed_index: int, candidate_index: int) -> float:
        seed_themes = self.item_features[seed_index]["themes"]
        candidate_themes = self.item_features[candidate_index]["themes"]
        return _overlap_ratio(seed_themes, candidate_themes)


def _extract_features(text: str) -> dict[str, set[str]]:
    keywords = _extract_keywords(text)
    bigrams = _extract_bigrams(text)
    themes = _extract_themes(keywords, bigrams)
    return {
        "keywords": keywords,
        "bigrams": bigrams,
        "themes": themes,
    }


def _extract_keywords(text: str) -> set[str]:
    return {
        word
        for word in _tokenize(text)
        if len(word) >= 4 and word not in STOP_WORDS
    }


def _extract_bigrams(text: str) -> set[str]:
    words = [word for word in _tokenize(text) if len(word) > 2 and word not in STOP_WORDS]
    return {
        f"{first_word} {second_word}"
        for first_word, second_word in zip(words, words[1:], strict=False)
        if first_word != second_word
    }


def _extract_themes(keywords: set[str], bigrams: set[str]) -> set[str]:
    terms = keywords.union(bigrams)
    theme_scores = []

    for theme, theme_words in THEME_RULES.items():
        score = 0.0
        for term in terms:
            words = set(term.split())
            overlap = words.intersection(theme_words)
            if overlap:
                score += len(overlap)

        if score > 0:
            theme_scores.append((score * THEME_PRIORITY.get(theme, 1), theme))

    theme_scores.sort(key=lambda item: (-item[0], item[1]))
    return {theme for _, theme in theme_scores[:3]}


def _overlap_ratio(seed_terms: set[str], candidate_terms: set[str]) -> float:
    if not seed_terms:
        return 0.0

    return len(seed_terms.intersection(candidate_terms)) / len(seed_terms)


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", normalized).strip().split()
