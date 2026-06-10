import { gaussian, mulberry32 } from './rng';

/** The visible region of feature space a dataset lives in (used to scale plots + grids). */
export interface Domain {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
}

/** A generated dataset: 2 input features per row, a single 0/1 target per row. */
export interface DataSet {
    inputs: number[][];
    targets: number[][];
}

export interface DatasetSpec {
    id: string;
    label: string;
    /** Short note on what makes this dataset interesting. */
    blurb: string;
    domain: Domain;
    /** A learning rate that converges nicely here — applied as the default when selected. */
    recommendedLr: number;
    /** A hidden-layer shape that solves this dataset — applied as the default when selected. */
    recommendedHidden: number[];
    /** Deterministic given the seed; `n` is ignored by the fixed (XNOR/XOR) sets. */
    generate: (seed: number, n: number) => DataSet;
}

// --- Fixed logic-gate datasets (the canonical non-linear toy problems) ---

const LOGIC_DOMAIN: Domain = { xMin: -0.4, xMax: 1.4, yMin: -0.4, yMax: 1.4 };
const LOGIC_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]];

function logicGate(targets: number[][]): () => DataSet {
    return () => ({ inputs: LOGIC_INPUTS.map(row => [...row]), targets: targets.map(row => [...row]) });
}

// --- Synthetic clouds ---

function moons(seed: number, n: number): DataSet {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const half = Math.floor(n / 2);

    for (let i = 0; i < n; i++) {
        const upper = i < half;
        const span = upper ? half : n - half;
        const t = (Math.PI * (upper ? i : i - half)) / span;
        const x = (upper ? Math.cos(t) : 1 - Math.cos(t)) + gaussian(rand) * 0.1;
        const y = (upper ? Math.sin(t) : 0.5 - Math.sin(t)) + gaussian(rand) * 0.1;
        inputs.push([x, y]);
        targets.push([upper ? 0 : 1]);
    }
    return { inputs, targets };
}

function circles(seed: number, n: number): DataSet {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const half = Math.floor(n / 2);

    for (let i = 0; i < n; i++) {
        const inner = i < half;
        const angle = rand() * 2 * Math.PI;
        const radius = (inner ? 0.4 : 1.0) + gaussian(rand) * 0.08;
        inputs.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
        targets.push([inner ? 1 : 0]);
    }
    return { inputs, targets };
}

function spiral(seed: number, n: number): DataSet {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const perArm = Math.floor(n / 2);

    for (let arm = 0; arm < 2; arm++) {
        for (let i = 0; i < perArm; i++) {
            const t = i / perArm;
            const radius = t;
            const theta = arm * Math.PI + t * 2.2 * Math.PI + gaussian(rand) * 0.1;
            inputs.push([radius * Math.sin(theta), radius * Math.cos(theta)]);
            targets.push([arm]);
        }
    }
    return { inputs, targets };
}

function blobs(seed: number, n: number): DataSet {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const half = Math.floor(n / 2);

    for (let i = 0; i < n; i++) {
        const first = i < half;
        const cx = first ? -0.5 : 0.5;
        const cy = first ? -0.5 : 0.5;
        inputs.push([cx + gaussian(rand) * 0.22, cy + gaussian(rand) * 0.22]);
        targets.push([first ? 0 : 1]);
    }
    return { inputs, targets };
}

export const DATASETS: DatasetSpec[] = [
    {
        id: 'xnor',
        label: 'XNOR',
        blurb: 'The matching-dials pairing: delight when both cues agree (bold-bold or mild-mild), a flop when they clash. The "both-or-neither" shape the perceptron just failed on.',
        domain: LOGIC_DOMAIN,
        recommendedLr: 1,
        recommendedHidden: [8],
        generate: logicGate([[1], [0], [0], [1]]),
    },
    {
        id: 'xor',
        label: 'XOR',
        blurb: 'The mirror image: a pairing works only when the two cues differ. Same four corners, opposite calls.',
        domain: LOGIC_DOMAIN,
        recommendedLr: 1,
        recommendedHidden: [8],
        generate: logicGate([[0], [1], [1], [0]]),
    },
    {
        id: 'moons',
        label: 'Two moons',
        blurb: 'Two tastes that interleave like crescents — the boundary curves beautifully as it learns.',
        domain: { xMin: -1.5, xMax: 2.5, yMin: -1.2, yMax: 1.6 },
        recommendedLr: 3,
        recommendedHidden: [12],
        generate: moons,
    },
    {
        id: 'circles',
        label: 'Circles',
        blurb: 'A core preference ringed by its opposite. The network has to learn a closed region.',
        domain: { xMin: -1.4, xMax: 1.4, yMin: -1.4, yMax: 1.4 },
        recommendedLr: 1,
        recommendedHidden: [8],
        generate: circles,
    },
    {
        id: 'spiral',
        label: 'Spiral',
        blurb: 'The showstopper: two tastes wound tightly together. A deeper network untangles the arms — give it a moment.',
        domain: { xMin: -1.2, xMax: 1.2, yMin: -1.2, yMax: 1.2 },
        recommendedLr: 3,
        recommendedHidden: [16, 16],
        generate: spiral,
    },
    {
        id: 'blobs',
        label: 'Blobs',
        blurb: 'Two clearly separate crowds — even a tiny network nails it fast.',
        domain: { xMin: -1.5, xMax: 1.5, yMin: -1.5, yMax: 1.5 },
        recommendedLr: 1,
        recommendedHidden: [8],
        generate: blobs,
    },
];
