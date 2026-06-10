// Colour for a predicted rating (1–5): dim slate for low, bright amber for high — so the grid reads
// like a heatmap of how much each user is predicted to like each item.
const LOW: [number, number, number] = [51, 65, 85]; // slate-700
const HIGH: [number, number, number] = [251, 191, 36]; // amber-400

export function ratingColor(rating: number): string {
    const t = Math.max(0, Math.min(1, (rating - 1) / 4));
    const r = Math.round(LOW[0] + (HIGH[0] - LOW[0]) * t);
    const g = Math.round(LOW[1] + (HIGH[1] - LOW[1]) * t);
    const b = Math.round(LOW[2] + (HIGH[2] - LOW[2]) * t);
    return `rgb(${r}, ${g}, ${b})`;
}

/** Dark text on bright (high-rating) cells, light text on dim ones — keeps the number readable. */
export function ratingTextColor(rating: number): string {
    return rating >= 3.5 ? '#1f2937' : '#e2e8f0';
}
