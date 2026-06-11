// Datasets for the Bayesian-regression playground. Each is a hidden curve plus a layout of noisy
// observations that deliberately leaves some of the domain *unobserved* — an empty edge, a gap in the
// middle, or just sparse coverage — so you can watch the model's error bars stay wide exactly where it
// has no evidence. The playground reveals the points one at a time and refits after each.

export interface BayesianScenario {
    id: string;
    label: string;
    blurb: string;
    domain: [number, number];
    yRange: [number, number];
    truth: (x: number) => number;
    /** A fixed layout of observations (already noisy), revealed incrementally by the playground. */
    generate: (seed: number) => { x: number; y: number }[];
}

function noisyPoints(seed: number, xs: number[], truth: (x: number) => number, noise: number) {
    let s = seed;
    const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
    return xs.map(x => ({ x, y: truth(x) + (rand() - 0.5) * 2 * noise }));
}

function spread(seed: number, count: number, lo: number, hi: number): number[] {
    let s = seed * 7 + 1;
    const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
    return Array.from({ length: count }, () => lo + rand() * (hi - lo)).sort((a, b) => a - b);
}

const WAVE = (x: number) => Math.sin(2 * Math.PI * x) * 0.6;

export const BAYESIAN_SCENARIOS: BayesianScenario[] = [
    {
        id: 'unseen-edges',
        label: 'Unseen edges',
        blurb: 'All the data sits in the middle; the model has never seen the edges. Watch the error band collapse onto the points but flare out as soon as it has to extrapolate past them.',
        domain: [0, 1],
        yRange: [-1.4, 1.4],
        truth: WAVE,
        generate: (seed) => noisyPoints(seed, spread(seed, 28, 0.22, 0.72), WAVE, 0.08),
    },
    {
        id: 'the-gap',
        label: 'The gap',
        blurb: 'Two clusters of data with a hole between them. The band stays tight on each cluster and balloons across the gap — honest that it is guessing where it has no points, even surrounded by data.',
        domain: [0, 1],
        yRange: [-1.4, 1.4],
        truth: WAVE,
        generate: (seed) => noisyPoints(seed, [...spread(seed, 13, 0.05, 0.32), ...spread(seed + 1, 13, 0.68, 0.95)], WAVE, 0.08),
    },
    {
        id: 'sparse',
        label: 'Sparse evidence',
        blurb: 'Only a handful of points across the whole range. The band pinches in at each one and bulges between them — more data anywhere will tighten it there.',
        domain: [0, 1],
        yRange: [-1.2, 1.2],
        truth: (x) => 0.75 * Math.exp(-((x - 0.45) ** 2) / 0.05) - 0.35,
        generate: (seed) => noisyPoints(seed, spread(seed, 9, 0.05, 0.95), (x) => 0.75 * Math.exp(-((x - 0.45) ** 2) / 0.05) - 0.35, 0.06),
    },
];
