import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

/** A 3-class 2D dataset with one-hot targets (N×3). */
export interface MulticlassDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    recommendedLr: number;
    classes: number;
    generate: (seed: number, n: number) => { inputs: number[][]; targets: number[][] };
}

const DOMAIN: Domain = { xMin: -1.4, xMax: 1.4, yMin: -1.4, yMax: 1.4 };

function oneHot(classIndex: number, classes = 3): number[] {
    const row = new Array(classes).fill(0);
    row[classIndex] = 1;
    return row;
}

/** Three Gaussian clusters at the vertices of a triangle — cleanly separable. */
function blobs3(seed: number, n: number) {
    const rand = mulberry32(seed);
    const centers = [
        [0, 0.75],
        [-0.7, -0.45],
        [0.7, -0.45],
    ];
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < n; i++) {
        const c = i % 3;
        inputs.push([centers[c][0] + gaussian(rand) * 0.22, centers[c][1] + gaussian(rand) * 0.22]);
        targets.push(oneHot(c));
    }
    return { inputs, targets };
}

/** Three horizontal bands — separable by horizontal lines. */
function stripes3(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < n; i++) {
        const c = i % 3;
        const x = rand() * 2.4 - 1.2;
        const y = (c - 1) * 0.8 + gaussian(rand) * 0.12; // bands at y ≈ -0.8, 0, 0.8
        inputs.push([x, y]);
        targets.push(oneHot(c));
    }
    return { inputs, targets };
}

/** Three rotating wedges meeting at the center — boundaries are roughly linear from the origin. */
function pinwheel3(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < n; i++) {
        const c = i % 3;
        const radius = Math.sqrt(rand()) * 1.1;
        const angle = (c * 2 * Math.PI) / 3 + rand() * ((2 * Math.PI) / 3) + gaussian(rand) * 0.08;
        inputs.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
        targets.push(oneHot(c));
    }
    return { inputs, targets };
}

export const MULTICLASS_DATASETS: MulticlassDataset[] = [
    {
        id: 'blobs3',
        label: 'Three blobs',
        blurb: 'Three separated clusters — one-vs-rest carves clean wedges between them.',
        domain: DOMAIN,
        recommendedLr: 1,
        classes: 3,
        generate: blobs3,
    },
    {
        id: 'stripes3',
        label: 'Three stripes',
        blurb: 'Horizontal bands — each class peels off the rest with a straight line.',
        domain: DOMAIN,
        recommendedLr: 1,
        classes: 3,
        generate: stripes3,
    },
    {
        id: 'pinwheel3',
        label: 'Pinwheel',
        blurb: 'Three wedges around the center — the straight boundaries only roughly fit.',
        domain: DOMAIN,
        recommendedLr: 1,
        classes: 3,
        generate: pinwheel3,
    },
];
