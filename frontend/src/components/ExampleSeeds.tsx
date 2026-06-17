import React from "react";
import type { Anime } from "../types";

interface ExampleSeedsProps {
  onSelectExample: (anime: Anime) => void;
  selectedIds: number[];
}

export const ExampleSeeds: React.FC<ExampleSeedsProps> = ({
  onSelectExample,
  selectedIds,
}) => {
  const examples: Anime[] = [
    {
      anime_id: 1535,
      title: "death note",
      title_english: "Death Note",
    },
    {
      anime_id: 16498,
      title: "shingeki no kyojin",
      title_english: "Attack on Titan",
    },
    {
      anime_id: 21,
      title: "one piece",
      title_english: "One Piece",
    },
    {
      anime_id: 9253,
      title: "steins;gate",
      title_english: "Steins;Gate",
    },
  ];

  return (
    <div className="example-seeds-container">
      <span className="example-seeds-label">Try:</span>
      <div className="example-seeds-list">
        {examples.map((anime) => {
          const isSelected = selectedIds.includes(anime.anime_id);
          return (
            <button
              key={anime.anime_id}
              type="button"
              className={`example-seed-btn ${isSelected ? "selected" : ""}`}
              onClick={() => !isSelected && onSelectExample(anime)}
              disabled={isSelected}
            >
              {anime.title_english || anime.title}
            </button>
          );
        })}
      </div>
    </div>
  );
};
export default ExampleSeeds;
