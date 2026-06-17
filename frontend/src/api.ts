import type { Anime, Recommendation } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function searchAnime(query: string, limit: number = 20): Promise<Anime[]> {
  const url = `${API_BASE_URL}/anime/search?q=${encodeURIComponent(query)}&limit=${limit}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Search request failed: ${response.statusText}`);
  }
  const data = await response.json();
  return data.results || [];
}

export async function getAnimeDetails(animeId: number): Promise<Anime> {
  const url = `${API_BASE_URL}/anime/${animeId}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch anime details: ${response.statusText}`);
  }
  return await response.json();
}

export async function getRecommendations(
  animeIds: number[],
  ratings?: Record<number, number>,
  topK: number = 10,
  mode: "discover" | "similar" = "discover"
): Promise<Recommendation[]> {
  const url = `${API_BASE_URL}/recommend`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      anime_ids: animeIds,
      ratings: ratings && Object.keys(ratings).length > 0 ? ratings : null,
      top_k: topK,
      mode: mode,
    }),
  });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `Failed to fetch recommendations: ${response.statusText}`);
  }
  const data = await response.json();
  return data.recommendations || [];
}
