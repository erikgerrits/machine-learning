// Colorblind-safe two-class palette: sky-blue (class 0) ↔ amber (class 1).
const C0: [number, number, number] = [56, 189, 248]; // sky-400
const C1: [number, number, number] = [251, 146, 60]; // orange-400

export const CLASS_COLORS = ['#38bdf8', '#fb923c'];

export function classColor(label: number): string {
    return CLASS_COLORS[label] ?? '#e2e8f0';
}

/**
 * Maps a model output in [0, 1] to an RGBA color for the decision-boundary heatmap.
 * Hue interpolates class-0 → class-1; alpha encodes *certainty* (distance from 0.5), so the
 * decision boundary itself shows up as a faint seam and confident regions glow solidly.
 */
export function boundaryRGBA(value: number): [number, number, number, number] {
    const t = Math.min(1, Math.max(0, value));
    const r = Math.round(C0[0] + (C1[0] - C0[0]) * t);
    const g = Math.round(C0[1] + (C1[1] - C0[1]) * t);
    const b = Math.round(C0[2] + (C1[2] - C0[2]) * t);
    const certainty = Math.abs(t - 0.5) * 2; // 0 at the boundary → 1 at the extremes
    const a = Math.round(40 + 150 * certainty);
    return [r, g, b, a];
}
