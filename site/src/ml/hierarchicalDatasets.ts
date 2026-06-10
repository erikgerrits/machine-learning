import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// Unlabelled customer clouds for the hierarchical-clustering playground (Chapter 13). Same idea as
// the k-means data — each point is a customer placed by two measured habits — but the shapes are
// chosen to show what *linkage* changes: a clean nested tree, a curve single-linkage can follow,
// and a bridge that single-linkage chains across while complete-linkage refuses to.
export interface HierarchicalDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    recommendedClusters: number;
    generate: (seed: number, n: number) => { inputs: number[][] };
}

const DOMAIN: Domain = { xMin: -1.4, xMax: 1.4, yMin: -1.4, yMax: 1.4 };

/** `count` Gaussian blobs whose centres sit evenly around a circle. */
function ringOfBlobs(count: number, radius = 0.9, spread = 0.16) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const centers = Array.from({ length: count }, (_, c) => {
            const angle = (c / count) * 2 * Math.PI - Math.PI / 2;
            return [radius * Math.cos(angle), radius * Math.sin(angle)];
        });

        const inputs: number[][] = [];
        for (let i = 0; i < n; i++) {
            const center = centers[i % count];
            inputs.push([center[0] + gaussian(rand) * spread, center[1] + gaussian(rand) * spread]);
        }
        return { inputs };
    };
}

/** Two interleaving crescents — single-linkage can trace the curve; complete-linkage chops it. */
function moons(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
        const upper = i < half;
        const span = upper ? half : n - half;
        const t = (Math.PI * (upper ? i : i - half)) / span;
        const x = (upper ? Math.cos(t) : 1 - Math.cos(t)) + gaussian(rand) * 0.08;
        const y = (upper ? Math.sin(t) : 0.5 - Math.sin(t)) + gaussian(rand) * 0.08;
        inputs.push([x, y]);
    }
    return { inputs };
}

/** Two crowds joined by a thin trickle of in-between customers — the classic single-linkage trap. */
function bridge(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const bridgeCount = Math.floor(n * 0.16);
    const blobCount = n - bridgeCount;
    const half = Math.floor(blobCount / 2);

    for (let i = 0; i < blobCount; i++) {
        const right = i >= half;
        const cx = right ? 0.85 : -0.85;
        inputs.push([cx + gaussian(rand) * 0.17, gaussian(rand) * 0.17]);
    }
    for (let i = 0; i < bridgeCount; i++) {
        inputs.push([(rand() * 2 - 1) * 0.85, gaussian(rand) * 0.05]);
    }
    return { inputs };
}

export const HIERARCHICAL_DATASETS: HierarchicalDataset[] = [
    {
        id: 'blobs3',
        label: 'Three types',
        blurb: 'Three clearly separate kinds of customer. The dendrogram shows a big gap before the final joins — slide k to 3 and the cut lands right in it.',
        domain: DOMAIN,
        recommendedClusters: 3,
        generate: ringOfBlobs(3, 0.85, 0.17),
    },
    {
        id: 'moons',
        label: 'Blended habits',
        blurb: 'Two habits curling around each other. Complete-linkage chops them straight across; switch to single-linkage and it follows the curve like a string of beads.',
        domain: { xMin: -1.5, xMax: 2.5, yMin: -1.2, yMax: 1.6 },
        recommendedClusters: 2,
        generate: moons,
    },
    {
        id: 'bridge',
        label: 'A slow drift',
        blurb: 'Two crowds linked by a thin trickle of in-between regulars. Single-linkage chains right across the bridge into one blob; complete-linkage keeps the two crowds apart.',
        domain: DOMAIN,
        recommendedClusters: 2,
        generate: bridge,
    },
];
