import React, { useEffect, useRef } from "react";
import type { Anime } from "../types";

interface SearchResultsDropdownProps {
  results: Anime[];
  onSelect: (anime: Anime) => void;
  onClose: () => void;
  highlightedIndex: number;
}

export const SearchResultsDropdown: React.FC<SearchResultsDropdownProps> = ({
  results,
  onSelect,
  onClose,
  highlightedIndex,
}) => {
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on click outside (but not on Escape, which is now input-scoped)
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [onClose]);

  // Scroll highlighted item into view automatically
  useEffect(() => {
    if (highlightedIndex >= 0 && dropdownRef.current) {
      const items = dropdownRef.current.querySelectorAll(".search-dropdown-item");
      const activeItem = items[highlightedIndex] as HTMLElement;
      if (activeItem) {
        activeItem.scrollIntoView({
          block: "nearest",
          behavior: "auto",
        });
      }
    }
  }, [highlightedIndex]);

  if (results.length === 0) {
    return (
      <div className="search-dropdown empty-dropdown" ref={dropdownRef}>
        No matching anime found. Try another search query.
      </div>
    );
  }

  return (
    <div className="search-dropdown" ref={dropdownRef}>
      {results.map((anime, index) => (
        <div
          key={anime.anime_id}
          className={`search-dropdown-item ${index === highlightedIndex ? "highlighted" : ""}`}
          onClick={() => onSelect(anime)}
        >
          <div className="item-title">{anime.title}</div>
          {anime.title_english && (
            <div className="item-subtitle">{anime.title_english}</div>
          )}
        </div>
      ))}
    </div>
  );
};
export default SearchResultsDropdown;
