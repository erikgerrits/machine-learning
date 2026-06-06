import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

/**
 * Unlabelled 2-D point clouds for the k-means playground. Unlike the supervised datasets these
 * carry **no targets** — clustering is unsupervised, so all the model ever sees is `inputs`.
 * `recommendedClusters` is the value of k that matches the data's natural structure.
 */
export interface ClusteringDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    recommendedClusters: number;
    generate: (seed: number, n: number) => { inputs: number[][] };
}

const DOMAIN: Domain = { xMin: -1.4, xMax: 1.4, yMin: -1.4, yMax: 1.4 };

/** `count` Gaussian blobs whose centres sit evenly around a circle. */
function ringOfBlobs(count: number, radius = 0.95, spread = 0.16) {
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

/** Uniform noise — no real clusters, so k-means just tiles the plane into Voronoi cells. */
function scatter(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    for (let i = 0; i < n; i++) {
        inputs.push([rand() * 2.4 - 1.2, rand() * 2.4 - 1.2]);
    }
    return { inputs };
}

/** Two interleaving crescents — round-cluster k-means can't follow the curve (a teachable miss). */
function moons(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
        const upper = i < half;
        const span = upper ? half : n - half;
        const t = (Math.PI * (upper ? i : i - half)) / span;
        const x = (upper ? Math.cos(t) : 1 - Math.cos(t)) + gaussian(rand) * 0.1;
        const y = (upper ? Math.sin(t) : 0.5 - Math.sin(t)) + gaussian(rand) * 0.1;
        inputs.push([x, y]);
    }
    return { inputs };
}

export const CLUSTERING_DATASETS: ClusteringDataset[] = [
    {
        id: 'blobs5',
        label: 'Five blobs',
        blurb: 'Five tidy Gaussian clusters — k-means nails these when k matches.',
        domain: DOMAIN,
        recommendedClusters: 5,
        generate: ringOfBlobs(5),
    },
    {
        id: 'blobs3',
        label: 'Three blobs',
        blurb: 'Three well-separated clusters around a triangle. Try k = 3.',
        domain: DOMAIN,
        recommendedClusters: 3,
        generate: ringOfBlobs(3, 0.85, 0.18),
    },
    {
        id: 'scatter',
        label: 'Uniform scatter',
        blurb: 'No real structure — k-means still partitions the plane into k Voronoi cells.',
        domain: { xMin: -1.3, xMax: 1.3, yMin: -1.3, yMax: 1.3 },
        recommendedClusters: 4,
        generate: scatter,
    },
    {
        id: 'moons',
        label: 'Two moons',
        blurb: 'Curved clusters. k-means assumes round blobs, so it splits the moons oddly.',
        domain: { xMin: -1.5, xMax: 2.5, yMin: -1.2, yMax: 1.6 },
        recommendedClusters: 2,
        generate: moons,
    },
];
