import React from "react";

interface SearchBoxProps {
  query: string;
  onChange: (value: string) => void;
  isSearching: boolean;
  onClear: () => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
}

export const SearchBox: React.FC<SearchBoxProps> = ({
  query,
  onChange,
  isSearching,
  onClear,
  onKeyDown,
}) => {
  return (
    <div className="search-box-container">
      <div className="search-input-wrapper">
        <input
          type="text"
          className="search-input"
          placeholder="Search anime by title..."
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
        />
        {query && (
          <button type="button" className="clear-search-btn" onClick={onClear}>
            &#x2715;
          </button>
        )}
        {isSearching && <div className="search-spinner"></div>}
      </div>
    </div>
  );
};
export default SearchBox;
