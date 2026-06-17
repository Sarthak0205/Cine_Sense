import React from "react";
import type { Anime } from "../types";

interface SelectedAnimeCardProps {
  anime: Anime;
  rating: number | undefined;
  onRemove: () => void;
  onRate: (rating: number) => void;
}

export const SelectedAnimeCard: React.FC<SelectedAnimeCardProps> = ({
  anime,
  rating,
  onRemove,
  onRate,
}) => {
  const ratingOptions = [7, 8, 9, 10];

  return (
    <div className="selected-anime-card">
      <div className="card-header">
        <div className="card-titles">
          <span className="card-title">{anime.title}</span>
          {anime.title_english && (
            <span className="card-subtitle">{anime.title_english}</span>
          )}
        </div>
        <button
          type="button"
          className="remove-anime-btn"
          onClick={onRemove}
          title="Remove from selections"
        >
          &#x2715;
        </button>
      </div>

      <div className="card-rating-container">
        <span className="rating-label">Rating:</span>
        <div className="rating-chips">
          {ratingOptions.map((opt) => {
            const isActive = rating === opt;
            return (
              <button
                key={opt}
                type="button"
                className={`rating-chip ${isActive ? "active" : ""}`}
                onClick={() => onRate(opt)}
              >
                {opt}
              </button>
            );
          })}
          <span className="rating-status">
            {rating ? `(${rating}/10)` : "(Unrated)"}
          </span>
        </div>
      </div>
    </div>
  );
};
export default SelectedAnimeCard;
