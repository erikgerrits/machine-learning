// A tiny, seedable PRNG so every dataset is reproducible — the same seed always yields the
// exact same points. (We keep this in the app rather than the library because it's only used
// to fabricate demo data.)

/** mulberry32 — a fast, decent-quality 32-bit seeded generator returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Draws one standard-normal sample (mean 0, variance 1) via the Box–Muller transform. */
export function gaussian(rand: () => number): number {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
