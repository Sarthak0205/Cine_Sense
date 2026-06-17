import React, { useState, useEffect } from "react";
import type { Anime, Recommendation } from "./types";
import { searchAnime, getRecommendations } from "./api";
import SearchBox from "./components/SearchBox";
import SearchResultsDropdown from "./components/SearchResultsDropdown";
import SelectedAnimeCard from "./components/SelectedAnimeCard";
import ExampleSeeds from "./components/ExampleSeeds";
import RecommendationGrid from "./components/RecommendationGrid";
import RecommendationDetailsModal from "./components/RecommendationDetailsModal";

export const App: React.FC = () => {
  // Selections and Ratings
  const [selectedAnime, setSelectedAnime] = useState<Anime[]>([]);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  // Search States
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Anime[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);

  // Modal States
  const [selectedRec, setSelectedRec] = useState<Recommendation | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Status States
  const [isRecommending, setIsRecommending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Debounced search query
  useEffect(() => {
    if (!query.trim()) {
      setSearchResults([]);
      setIsSearching(false);
      setShowDropdown(false);
      return;
    }

    setIsSearching(true);
    const delayDebounce = setTimeout(async () => {
      try {
        const results = await searchAnime(query.trim());
        setSearchResults(results);
        setShowDropdown(true);
        setError(null);
      } catch (err: any) {
        setError(err.message || "Failed to search anime.");
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(delayDebounce);
  }, [query]);

  // Reset highlightedIndex when results or dropdown visibility changes
  useEffect(() => {
    setHighlightedIndex(-1);
  }, [searchResults, showDropdown]);

  // Clear recommendations if seeds or ratings change to prevent stale recommendations
  useEffect(() => {
    setRecommendations([]);
  }, [selectedAnime, ratings]);

  // Add anime to selection
  const handleSelectAnime = (anime: Anime) => {
    // Avoid duplicates
    if (!selectedAnime.some((item) => item.anime_id === anime.anime_id)) {
      setSelectedAnime([...selectedAnime, anime]);
    }
    setQuery("");
    setShowDropdown(false);
  };

  // Remove anime from selection
  const handleRemoveAnime = (animeId: number) => {
    setSelectedAnime(selectedAnime.filter((item) => item.anime_id !== animeId));
    const newRatings = { ...ratings };
    delete newRatings[animeId];
    setRatings(newRatings);
  };

  // Update rating chip selection
  const handleRateAnime = (animeId: number, ratingVal: number) => {
    setRatings((prev) => {
      // Toggle rating if clicked again
      if (prev[animeId] === ratingVal) {
        const next = { ...prev };
        delete next[animeId];
        return next;
      }
      return { ...prev, [animeId]: ratingVal };
    });
  };

  // Fetch recommendations
  const handleGetRecommendations = async () => {
    if (selectedAnime.length === 0) return;

    setIsRecommending(true);
    setError(null);
    try {
      const animeIds = selectedAnime.map((item) => item.anime_id);
      const recs = await getRecommendations(animeIds, ratings);
      setRecommendations(recs);
    } catch (err: any) {
      setError(err.message || "Failed to generate recommendations. Please try again.");
    } finally {
      setIsRecommending(false);
    }
  };

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown || searchResults.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev + 1) % searchResults.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev - 1 + searchResults.length) % searchResults.length);
    } else if (e.key === "Enter") {
      if (highlightedIndex >= 0 && highlightedIndex < searchResults.length) {
        e.preventDefault();
        handleSelectAnime(searchResults[highlightedIndex]);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      setShowDropdown(false);
      (e.target as HTMLInputElement).blur();
    }
  };

  const handleSelectRecommendation = (rec: Recommendation) => {
    setSelectedRec(rec);
    setIsModalOpen(true);
  };

  const selectedIds = selectedAnime.map((item) => item.anime_id);

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>CineSense</h1>
          <span className="version-badge">v1.0.0 (twostage)</span>
        </div>
      </header>

      <main className="app-main">
        {/* Search section */}
        <section className="search-section">
          <div className="search-box-wrapper">
            <SearchBox
              query={query}
              onChange={setQuery}
              isSearching={isSearching}
              onClear={() => {
                setQuery("");
                setShowDropdown(false);
              }}
              onKeyDown={handleSearchKeyDown}
            />
            {showDropdown && (
              <SearchResultsDropdown
                results={searchResults}
                onSelect={handleSelectAnime}
                onClose={() => setShowDropdown(false)}
                highlightedIndex={highlightedIndex}
              />
            )}
          </div>
          <ExampleSeeds
            onSelectExample={handleSelectAnime}
            selectedIds={selectedIds}
          />
        </section>

        {/* Global Error Banner */}
        {error && (
          <div className="error-banner">
            <span className="error-icon">&#x26A0;</span>
            <span className="error-message">{error}</span>
            <button className="close-error-btn" onClick={() => setError(null)}>
              &#x2715;
            </button>
          </div>
        )}

        {/* Workspace Layout */}
        <div className="workspace-layout">
          {/* Left panel: Selections */}
          <div className="workspace-panel selection-panel">
            <div className="panel-header">
              <h2>My Selection</h2>
              <span className="badge">{selectedAnime.length} items</span>
            </div>

            {selectedAnime.length === 0 ? (
              <div className="empty-state">
                <p>Select anime you enjoyed to generate recommendations.</p>
                <p className="hint">Use the search box above or click on the example seeds to get started.</p>
              </div>
            ) : (
              <div className="selections-list">
                {selectedAnime.map((anime) => (
                  <SelectedAnimeCard
                    key={anime.anime_id}
                    anime={anime}
                    rating={ratings[anime.anime_id]}
                    onRemove={() => handleRemoveAnime(anime.anime_id)}
                    onRate={(ratingVal) => handleRateAnime(anime.anime_id, ratingVal)}
                  />
                ))}

                <button
                  type="button"
                  className="recommend-cta-btn"
                  onClick={handleGetRecommendations}
                  disabled={isRecommending}
                >
                  {isRecommending ? "Generating recommendations..." : "Get Recommendations"}
                </button>
              </div>
            )}
          </div>

          {/* Right panel: Recommendations */}
          <div className="workspace-panel recommendations-panel">
            <div className="panel-header">
              <h2>Recommendations</h2>
            </div>

            {isRecommending ? (
              <div className="recommendations-container">
                <div className="loading-text-badge">
                  <span className="shimmer-pulse">Generating recommendations using CineSenseTwoStage model...</span>
                </div>
                <div className="recommendation-grid">
                  {Array.from({ length: 10 }).map((_, index) => (
                    <div className="recommendation-card skeleton-card" key={index}>
                      <div className="card-header">
                        <div className="card-titles">
                          <div className="skeleton-box skeleton-title"></div>
                          <div className="skeleton-box skeleton-subtitle"></div>
                        </div>
                        <div className="skeleton-box skeleton-score"></div>
                      </div>
                      <div className="card-body">
                        <div className="skeleton-box skeleton-badge"></div>
                        <div className="skeleton-box skeleton-reason-row row-long"></div>
                        <div className="skeleton-box skeleton-reason-row row-medium"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : selectedAnime.length === 0 ? (
              /* State A: Onboarding empty state */
              <div className="empty-state onboarding-empty">
                <div className="empty-state-icon">✨</div>
                <h3>Find your next favorite anime</h3>
                <p className="empty-state-body">
                  Select anime you already enjoy and CineSense will discover new recommendations tailored to your taste.
                </p>
                <div className="empty-state-guide">
                  <div className="connector-path">
                    <span className="connector-arrow">↑</span>
                    <span className="connector-text">Try one of these quick examples:</span>
                  </div>
                  <div className="quick-example-chips-row">
                    <button
                      type="button"
                      className="empty-state-chip"
                      onClick={() => handleSelectAnime({ anime_id: 1535, title: "death note", title_english: "Death Note" })}
                    >
                      Death Note
                    </button>
                    <button
                      type="button"
                      className="empty-state-chip"
                      onClick={() => handleSelectAnime({ anime_id: 16498, title: "shingeki no kyojin", title_english: "Attack on Titan" })}
                    >
                      Attack on Titan
                    </button>
                    <button
                      type="button"
                      className="empty-state-chip"
                      onClick={() => handleSelectAnime({ anime_id: 21, title: "one piece", title_english: "One Piece" })}
                    >
                      One Piece
                    </button>
                  </div>
                </div>
              </div>
            ) : recommendations.length === 0 ? (
              /* State B: Seeds Selected but no recommendations generated */
              <div className="empty-state ready-empty">
                <div className="empty-state-icon pulse-gold">🎯</div>
                <h3>Ready to discover</h3>
                <p className="empty-state-body">
                  You've selected anime as input. Click 'Get Recommendations' to generate personalized recommendations.
                </p>
                <div className="pulse-arrow-pointer">
                  <span>← Click the button under your selection to begin</span>
                </div>
              </div>
            ) : (
              <div className="recommendations-container">
                <RecommendationGrid
                  recommendations={recommendations}
                  onSelectRecommendation={handleSelectRecommendation}
                />
              </div>
            )}
          </div>
        </div>
      </main>

      <RecommendationDetailsModal
        isOpen={isModalOpen}
        recommendation={selectedRec}
        onClose={() => {
          setIsModalOpen(false);
          setSelectedRec(null);
        }}
      />
    </div>
  );
};

export default App;
