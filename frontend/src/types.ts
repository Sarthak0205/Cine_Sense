export interface Anime {
  anime_id: number;
  title: string;
  title_english: string | null;
  synopsis?: string;
}

export interface MatchedSeed {
  anime_id: number;
  title: string;
}

export interface Explanation {
  matched_seed: MatchedSeed;
  similarity: number;
  popularity: number;
  reason: string;
}

export interface Recommendation {
  anime_id: number;
  title: string;
  title_english: string | null;
  score: number;
  explanation: Explanation;
}
