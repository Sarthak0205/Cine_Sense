export function getMatchQuality(score: number): string {
  if (score >= 8.5) {
    return "Excellent Match";
  }

  if (score >= 7.0) {
    return "Good Match";
  }

  if (score >= 5.0) {
    return "Similar Match";
  }

  return "Discovery Pick";
}
