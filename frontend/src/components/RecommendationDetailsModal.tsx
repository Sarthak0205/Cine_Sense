import React, { useEffect, useRef, useState } from "react";
import type { Anime, Recommendation } from "../types";
import { getAnimeDetails } from "../api";

interface RecommendationDetailsModalProps {
  isOpen: boolean;
  recommendation: Recommendation | null;
  onClose: () => void;
}

export const RecommendationDetailsModal: React.FC<RecommendationDetailsModalProps> = ({
  isOpen,
  recommendation,
  onClose,
}) => {
  const [details, setDetails] = useState<Anime | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [, setError] = useState<string | null>(null);

  const modalRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const previousActiveElement = useRef<HTMLElement | null>(null);

  // Background loading + race condition protection
  useEffect(() => {
    if (!isOpen || !recommendation) {
      setDetails(null);
      setIsLoading(false);
      return;
    }

    let active = true;
    setIsLoading(true);
    setDetails(null);
    setError(null);

    getAnimeDetails(recommendation.anime_id)
      .then((data) => {
        if (active) {
          setDetails(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (active) {
          setError(err.message || "Failed to load details");
          setIsLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [isOpen, recommendation]);

  // Focus management & Trap Focus
  useEffect(() => {
    if (isOpen) {
      // Record currently focused element to restore it later
      previousActiveElement.current = document.activeElement as HTMLElement;

      // Focus close button automatically on next tick
      const timer = setTimeout(() => {
        closeBtnRef.current?.focus();
      }, 50);

      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === "Escape") {
          onClose();
          return;
        }

        if (e.key === "Tab" && modalRef.current) {
          const focusable = modalRef.current.querySelectorAll<HTMLElement>(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
          );
          if (focusable.length === 0) return;

          const first = focusable[0];
          const last = focusable[focusable.length - 1];

          if (e.shiftKey) {
            // Shift + Tab
            if (document.activeElement === first) {
              e.preventDefault();
              last.focus();
            }
          } else {
            // Tab
            if (document.activeElement === last) {
              e.preventDefault();
              first.focus();
            }
          }
        }
      };

      document.addEventListener("keydown", handleKeyDown);
      return () => {
        document.removeEventListener("keydown", handleKeyDown);
        clearTimeout(timer);
        // Restore focus when modal closes/unmounts
        previousActiveElement.current?.focus();
      };
    }
  }, [isOpen, onClose]);

  if (!isOpen || !recommendation) return null;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick} role="dialog" aria-modal="true">
      <div className="modal-container" ref={modalRef}>
        <div className="modal-header">
          <div className="modal-titles">
            <h2 className="modal-title">{recommendation.title}</h2>
            {recommendation.title_english && (
              <span className="modal-subtitle">{recommendation.title_english}</span>
            )}
          </div>
          <button
            type="button"
            className="modal-close-btn"
            ref={closeBtnRef}
            onClick={onClose}
            aria-label="Close modal"
          >
            &#x2715;
          </button>
        </div>

        <div className="modal-body">
          {/* Top segment: Score and seed matching */}
          <div className="modal-meta-grid">
            <div className="modal-meta-item">
              <span className="meta-label">Match Score</span>
              <span className="meta-value score-highlight">
                {(recommendation.score * 100).toFixed(1)}%
              </span>
            </div>
            {recommendation.explanation.matched_seed && (
              <div className="modal-meta-item">
                <span className="meta-label">Matched With</span>
                <span className="meta-value seed-name">
                  {recommendation.explanation.matched_seed.title}
                </span>
              </div>
            )}
          </div>

          {/* Explanation section */}
          <div className="modal-section explanation-section">
            <h3>Why you may like it:</h3>
            <p className="explanation-text">{recommendation.explanation.reason}</p>
          </div>

          {/* Synopsis Section with Shimmer Loading */}
          <div className="modal-section synopsis-section">
            <h3>Synopsis</h3>
            {isLoading ? (
              <div className="synopsis-skeleton-container" aria-label="Loading synopsis...">
                <div className="skeleton-box skeleton-text-row row-long"></div>
                <div className="skeleton-box skeleton-text-row row-medium"></div>
                <div className="skeleton-box skeleton-text-row row-long"></div>
                <div className="skeleton-box skeleton-text-row row-short"></div>
              </div>
            ) : details?.synopsis ? (
              <p className="synopsis-text">{details.synopsis}</p>
            ) : (
              <p className="synopsis-text empty-synopsis">No synopsis available for this title.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationDetailsModal;
