import type { DataSet, Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// Curved binary datasets for Chapter 6: two pastry types (croissants vs. danishes) plotted by two
// measurements, arranged so the types interleave, ring, or spiral — shapes no straight line can
// separate, but that k-NN traces effortlessly by looking at what's nearby.
export interface PastryDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    generate: (seed: number, n: number) => DataSet;
}

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

export const PASTRY_DATASETS: PastryDataset[] = [
    {
        id: 'two-recipes',
        label: 'Two recipes',
        blurb: 'Croissants and danishes interleave in two crescents — no straight cut separates them, but their neighbours give them away.',
        domain: { xMin: -1.5, xMax: 2.5, yMin: -1.2, yMax: 1.6 },
        generate: moons,
    },
    {
        id: 'ring',
        label: 'A ring',
        blurb: 'One type clusters in the middle, the other rings around it. A straight line cannot wrap a ring; k-NN just follows it.',
        domain: { xMin: -1.4, xMax: 1.4, yMin: -1.4, yMax: 1.4 },
        generate: circles,
    },
    {
        id: 'swirl',
        label: 'The swirl',
        blurb: 'The showstopper: two types coil around each other. k-NN traces the spiral arm for arm.',
        domain: { xMin: -1.2, xMax: 1.2, yMin: -1.2, yMax: 1.2 },
        generate: spiral,
    },
    {
        id: 'easy',
        label: 'Easy day',
        blurb: 'Two clearly separated piles — k-NN nails these as easily as any straight line would.',
        domain: { xMin: -1.5, xMax: 1.5, yMin: -1.5, yMax: 1.5 },
        generate: blobs,
    },
];
