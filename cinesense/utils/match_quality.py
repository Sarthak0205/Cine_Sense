def get_match_quality(score: float) -> str:
    """Returns the qualitative match quality label."""

    if score >= 8.5:
        return "Excellent Match"

    if score >= 7.0:
        return "Good Match"

    if score >= 5.0:
        return "Similar Match"

    return "Discovery Pick"
