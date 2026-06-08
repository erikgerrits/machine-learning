import type { DataSet, Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// Binary dough-batch datasets for Chapter 4. Each day is a batch plotted by two features —
// proving warmth (x) and yeast freshness (y) — labelled rose (1) or flopped (0). The shapes are
// the classic blobs/moons, re-themed: a clean split, an overlapping one, and a curved one a
// straight line can't follow.
export interface BinaryDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    generate: (seed: number, n: number) => DataSet;
}

const WIDE: Domain = { xMin: -1.6, xMax: 1.6, yMin: -1.6, yMax: 1.6 };

// Two gaussian clouds: flops in the cold-&-stale corner, rises in the warm-&-fresh corner.
function blobs(offset: number, spread: number) {
    return (seed: number, n: number): DataSet => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        const targets: number[][] = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < n; i++) {
            const rose = i >= half;
            const cx = rose ? offset : -offset;
            const cy = rose ? offset : -offset;
            inputs.push([cx + gaussian(rand) * spread, cy + gaussian(rand) * spread]);
            targets.push([rose ? 1 : 0]);
        }
        return { inputs, targets };
    };
}

// Two interleaving crescents — two proofing methods that curl around each other.
function moons(seed: number, n: number): DataSet {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
        const upper = i < half;
        const span = upper ? half : n - half;
        const t = (Math.PI * (upper ? i : i - half)) / span;
        const x = (upper ? Math.cos(t) : 1 - Math.cos(t)) + gaussian(rand) * 0.12;
        const y = (upper ? Math.sin(t) : 0.5 - Math.sin(t)) + gaussian(rand) * 0.12;
        inputs.push([x, y]);
        targets.push([upper ? 0 : 1]);
    }
    return { inputs, targets };
}

export const DOUGH_DATASETS: BinaryDataset[] = [
    {
        id: 'clean',
        label: 'Tried & true',
        blurb: 'Cold & stale flops; warm & fresh rises. A clean diagonal split — the line nails it.',
        domain: WIDE,
        generate: blobs(0.6, 0.28),
    },
    {
        id: 'borderline',
        label: 'Borderline batches',
        blurb: 'The two clouds overlap. The line finds the best split it can — but batches near it are genuine coin-flips.',
        domain: WIDE,
        generate: blobs(0.42, 0.5),
    },
    {
        id: 'two-methods',
        label: 'Two methods',
        blurb: 'Two proofing methods curl around each other. A straight line catches most — but cannot follow the curve. (Hold that thought.)',
        domain: { xMin: -1.5, xMax: 2.5, yMin: -1.2, yMax: 1.6 },
        generate: moons,
    },
];
