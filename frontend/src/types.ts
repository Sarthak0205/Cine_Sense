export interface Anime {
  anime_id: number;
  title: string;
  title_english: string | null;
  synopsis?: string;
  genres?: string[];
}

export interface MatchedSeed {
  anime_id: number;
  title: string;
}

export interface SeedShare {
  title: string;
  share: number;
}

export interface Explanation {
  matched_seed: MatchedSeed | null;
  similarity: number | null;
  popularity: number | null;
  summary: string;
  reasons: string[];
  reason: string;
  seed_shares?: Record<number, SeedShare> | null;
}

export interface Recommendation {
  anime_id: number;
  title: string;
  title_english: string | null;
  score: number;
  match_score: number;
  match_badge: string;
  explanation: Explanation;
}
