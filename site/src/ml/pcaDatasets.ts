import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// 2-D customer clouds for the PCA playground (Chapter 15). Each point is a customer scored on two
// survey questions; PCA finds the directions of greatest variance. Two questions is the most you
// can *see* — but the same maths is what collapses a thirty-question survey to two axes. The shapes
// vary how much one axis really matters: a tight diagonal (one axis is enough), two crowds along a
// diagonal (reduction that keeps the split), and a near-round blob (nothing to reduce).
export interface PcaDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    generate: (seed: number, n: number) => { inputs: number[][] };
}

const DOMAIN: Domain = { xMin: -3, xMax: 3, yMin: -3, yMax: 3 };

/** A Gaussian cloud stretched `s1` along a direction rotated `angleDeg`, and `s2` across it. */
function correlatedCloud(angleDeg: number, s1: number, s2: number) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const angle = (angleDeg * Math.PI) / 180;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const inputs: number[][] = [];
        for (let i = 0; i < n; i++) {
            const along = gaussian(rand) * s1;
            const across = gaussian(rand) * s2;
            inputs.push([along * cos - across * sin, along * sin + across * cos]);
        }
        return { inputs };
    };
}

/** Two crowds offset along the diagonal — PC1 lines up with the gap between them. */
function twoCrowds(seed: number, n: number) {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
        const top = i < half;
        const cx = top ? -1.1 : 1.1;
        const cy = top ? -1.1 : 1.1;
        inputs.push([cx + gaussian(rand) * 0.45, cy + gaussian(rand) * 0.45]);
    }
    return { inputs };
}

export const PCA_DATASETS: PcaDataset[] = [
    {
        id: 'one-axis',
        label: 'Two tastes, one axis',
        blurb: 'Two survey questions that move together — answer one and you basically know the other. PCA finds the single axis they share; projecting onto it barely loses a thing.',
        domain: DOMAIN,
        generate: correlatedCloud(35, 1.7, 0.28),
    },
    {
        id: 'two-crowds',
        label: 'Two crowds',
        blurb: 'Two clusters strung along a diagonal. PC1 lines up with the gap, so flattening onto that one axis keeps the two crowds clearly apart — reduction that loses the noise, not the signal.',
        domain: DOMAIN,
        generate: twoCrowds,
    },
    {
        id: 'round',
        label: 'No clear axis',
        blurb: 'A roughly round blob — the two questions are unrelated and vary about equally. Neither axis dominates, so squashing to one really would throw away half the picture. PCA is honest about that.',
        domain: DOMAIN,
        generate: correlatedCloud(20, 1.05, 0.95),
    },
];
