import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// Unlabelled customer clouds for the DBSCAN playground (Chapter 14). The shapes are chosen to show
// what density-based clustering can do that k-means and hierarchical can't: find arbitrarily shaped
// dense groups and, crucially, leave the stragglers *out* — flagged as noise rather than forced
// into a group. `epsilon` / `minPoints` are sensible starting values for each set.
export interface DbscanDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    epsilon: number;
    minPoints: number;
    generate: (seed: number, n: number) => { inputs: number[][] };
}

const DOMAIN: Domain = { xMin: -1.4, xMax: 1.4, yMin: -1.4, yMax: 1.4 };

/** Two tight crowds plus a sparse sprinkle of one-off / oddball customers scattered everywhere. */
function blobsWithNoise(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const noiseCount = Math.floor(n * 0.18);
    const blobCount = n - noiseCount;
    const half = Math.floor(blobCount / 2);

    for (let i = 0; i < blobCount; i++) {
        const right = i >= half;
        const cx = right ? 0.7 : -0.7;
        const cy = right ? 0.5 : -0.5;
        inputs.push([cx + gaussian(rand) * 0.16, cy + gaussian(rand) * 0.16]);
    }
    for (let i = 0; i < noiseCount; i++) {
        inputs.push([(rand() * 2 - 1) * 1.3, (rand() * 2 - 1) * 1.3]);
    }
    return { inputs };
}

/** Two interleaving crescents — DBSCAN traces each curve as one dense, connected group. */
function moons(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
        const upper = i < half;
        const span = upper ? half : n - half;
        const t = (Math.PI * (upper ? i : i - half)) / span;
        const x = (upper ? Math.cos(t) : 1 - Math.cos(t)) + gaussian(rand) * 0.07;
        const y = (upper ? Math.sin(t) : 0.5 - Math.sin(t)) + gaussian(rand) * 0.07;
        inputs.push([x, y]);
    }
    return { inputs };
}

/** A tight core crowd encircled by a ring of regulars — two groups no centroid could separate. */
function ring(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const core = Math.floor(n * 0.4);
    const ringCount = n - core;
    for (let i = 0; i < n; i++) {
        if (i < core) {
            inputs.push([gaussian(rand) * 0.12, gaussian(rand) * 0.12]);
        } else {
            // Spread the ring points roughly evenly (with a little jitter) so the loop stays a single
            // connected band — fully random angles leave gaps that fragment it.
            const angle = ((i - core + (rand() - 0.5) * 0.7) / ringCount) * 2 * Math.PI;
            const radius = 0.9 + gaussian(rand) * 0.05;
            inputs.push([Math.cos(angle) * radius, Math.sin(angle) * radius]);
        }
    }
    return { inputs };
}

export const DBSCAN_DATASETS: DbscanDataset[] = [
    {
        id: 'regulars-noise',
        label: 'Regulars & oddballs',
        blurb: 'Two solid crowds of regulars, dusted with one-off visitors. DBSCAN keeps the crowds and marks the strays as noise — the groups k-means was forced to swallow.',
        domain: DOMAIN,
        epsilon: 0.2,
        minPoints: 4,
        generate: blobsWithNoise,
    },
    {
        id: 'moons',
        label: 'Blended habits',
        blurb: 'Two habits curling around each other. Density follows the curve, so each crescent comes out as one clean group — no linkage tuning, no k.',
        domain: { xMin: -1.5, xMax: 2.5, yMin: -1.2, yMax: 1.6 },
        epsilon: 0.22,
        minPoints: 4,
        generate: moons,
    },
    {
        id: 'ring',
        label: 'A ring of regulars',
        blurb: 'A tight core crowd inside a ring of others. They share a centre, so k-means is hopeless — but they are two separate dense regions, which is all DBSCAN cares about.',
        domain: DOMAIN,
        epsilon: 0.18,
        minPoints: 4,
        generate: ring,
    },
];
