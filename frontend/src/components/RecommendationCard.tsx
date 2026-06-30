import React from "react";
import type { Recommendation } from "../types";
import { getMatchQuality } from "../utils/matchQuality";

interface RecommendationCardProps {
  rec: Recommendation;
  onClick: () => void;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({ rec, onClick }) => {
  const matchedSeed = rec.explanation.matched_seed ? rec.explanation.matched_seed.title : null;
  const quality = getMatchQuality(rec.match_score);
  const badgeClass = `badge-${quality.toLowerCase().replace(" ", "-")}`;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <div
      className="recommendation-card"
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      aria-label={`View recommendations details for ${rec.title}`}
    >
      <div className="card-header">
        <div className="card-titles">
          <span className="card-title" title={rec.title}>{rec.title}</span>
          {rec.title_english && (
            <span className="card-subtitle" title={rec.title_english}>{rec.title_english}</span>
          )}
        </div>
        <div className={`card-score ${badgeClass}`} title={quality}>{rec.match_score} / 10</div>
      </div>

      <div className="card-body">
        {matchedSeed && (
          <div className="matched-seed-badge">
            Matched with: <span className="seed-name">{matchedSeed}</span>
          </div>
        )}
        <div className="why-like-section">
          <span className="why-like-label">Why you may like it:</span>
          <p className="explanation-text">{rec.explanation.reason}</p>
        </div>
      </div>
    </div>
  );
};
export default RecommendationCard;
