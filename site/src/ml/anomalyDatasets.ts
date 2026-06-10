import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// 2-D transaction clouds for the anomaly-detection playground (Chapter 16). The bulk of each set is
// "normal" — a Gaussian the detector fits — with a sprinkle of anomalies scattered wide (the fraud /
// glitch / spoiled-batch points). The shapes vary what "normal" looks like: a round blob, a tilted
// correlated one (where Mahalanobis earns its keep), and two clusters (where one Gaussian is a poor fit).
export interface AnomalyDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    threshold: number;
    generate: (seed: number, n: number) => { inputs: number[][] };
}

const DOMAIN: Domain = { xMin: -3.2, xMax: 3.2, yMin: -3.2, yMax: 3.2 };

/** Scatter `count` wide-ranging anomalies across the plane (kept clear of the dense centre). */
function scatterAnomalies(rand: () => number, count: number, inputs: number[][]) {
    let placed = 0;
    while (placed < count) {
        const x = (rand() * 2 - 1) * 2.9;
        const y = (rand() * 2 - 1) * 2.9;
        if (Math.hypot(x, y) > 1.6) {
            inputs.push([x, y]);
            placed++;
        }
    }
}

/** One round cloud of everyday transactions, plus a few wild outliers. */
function everyday(seed: number, n: number) {
    const rand = mulberry32(seed);
    const anomalies = Math.floor(n * 0.08);
    const inputs: number[][] = [];
    for (let i = 0; i < n - anomalies; i++) {
        inputs.push([gaussian(rand) * 0.6, gaussian(rand) * 0.6]);
    }
    scatterAnomalies(rand, anomalies, inputs);
    return { inputs };
}

/** A strongly correlated cloud (spend and frequency rise together), plus outliers. */
function tilted(seed: number, n: number) {
    const rand = mulberry32(seed);
    const anomalies = Math.floor(n * 0.08);
    const angle = (35 * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const inputs: number[][] = [];
    for (let i = 0; i < n - anomalies; i++) {
        const along = gaussian(rand) * 1.4;
        const across = gaussian(rand) * 0.32;
        inputs.push([along * cos - across * sin, along * sin + across * cos]);
    }
    scatterAnomalies(rand, anomalies, inputs);
    return { inputs };
}

/** Two normal crowds — a single Gaussian can't model both at once (a teachable miss). */
function twoHabits(seed: number, n: number) {
    const rand = mulberry32(seed);
    const anomalies = Math.floor(n * 0.06);
    const normal = n - anomalies;
    const half = Math.floor(normal / 2);
    const inputs: number[][] = [];
    for (let i = 0; i < normal; i++) {
        const right = i >= half;
        const cx = right ? 1.4 : -1.4;
        inputs.push([cx + gaussian(rand) * 0.45, gaussian(rand) * 0.45]);
    }
    scatterAnomalies(rand, anomalies, inputs);
    return { inputs };
}

export const ANOMALY_DATASETS: AnomalyDataset[] = [
    {
        id: 'everyday',
        label: 'Everyday transactions',
        blurb: 'A dense crowd of ordinary transactions with a few wild ones scattered around it. The detector learns the blob and flags whatever sits far outside it.',
        domain: DOMAIN,
        threshold: 3,
        generate: everyday,
    },
    {
        id: 'tilted',
        label: 'Tilted normal',
        blurb: 'Spend and frequency rise together, so "normal" is a tilted ellipse. Watch a point off the diagonal get flagged while one just as far along it stays normal — that\'s Mahalanobis distance, not plain distance.',
        domain: DOMAIN,
        threshold: 3,
        generate: tilted,
    },
    {
        id: 'two-habits',
        label: 'Two habits',
        blurb: 'Two separate crowds of regulars. One Gaussian has to straddle both, so its centre lands in the empty gap — and the model misjudges what\'s normal. A single bell curve is the wrong shape here.',
        domain: DOMAIN,
        threshold: 2.5,
        generate: twoHabits,
    },
];
