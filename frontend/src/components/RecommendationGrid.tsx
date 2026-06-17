import React from "react";
import type { Recommendation } from "../types";
import RecommendationCard from "./RecommendationCard";

interface RecommendationGridProps {
  recommendations: Recommendation[];
  onSelectRecommendation: (rec: Recommendation) => void;
}

export const RecommendationGrid: React.FC<RecommendationGridProps> = ({
  recommendations,
  onSelectRecommendation,
}) => {
  return (
    <div className="recommendation-grid">
      {recommendations.map((rec) => (
        <RecommendationCard
          key={rec.anime_id}
          rec={rec}
          onClick={() => onSelectRecommendation(rec)}
        />
      ))}
    </div>
  );
};
export default RecommendationGrid;
