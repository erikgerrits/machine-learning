/**
 * mulberry32 — a tiny, seedable pseudo-random number generator returning floats in [0, 1).
 *
 * Calling it with the same seed always yields the same sequence, which is what makes the
 * library's seeded models reproducible (e.g. {@link Matrix.rand}'s weight initialisation and
 * k-means' centroid placement). It is a few lines of integer math, so the library stays
 * dependency-free — no external RNG package required.
 *
 * @example
 * const random = mulberry32(42);
 * random(); // 0.6011037519201636 — and always the same for seed 42
 */
export default function mulberry32 (seed: number): () => number {
    let state = seed >>> 0;

    return function () {
        state = (state + 0x6d2b79f5) | 0;
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
