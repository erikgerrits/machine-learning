import type { Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

/** A 1-D regression dataset: one input feature per row, one continuous target per row. */
export interface RegressionDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    /** A learning rate that converges quickly here — applied as the default when selected. */
    recommendedLr: number;
    generate: (seed: number, n: number) => { inputs: number[][]; targets: number[][] };
}

// Inputs are kept in [-1, 1] so plain gradient descent converges quickly without feature scaling.
const DOMAIN: Domain = { xMin: -1.2, xMax: 1.2, yMin: -2, yMax: 2 };

function line(slope: number, intercept: number, noise: number) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        const targets: number[][] = [];
        for (let i = 0; i < n; i++) {
            const x = rand() * 2 - 1;
            inputs.push([x]);
            targets.push([slope * x + intercept + gaussian(rand) * noise]);
        }
        return { inputs, targets };
    };
}

export const REGRESSION_DATASETS: RegressionDataset[] = [
    {
        id: 'trend',
        label: 'Upward trend',
        blurb: 'A gentle positive slope with a little noise.',
        domain: DOMAIN,
        recommendedLr: 0.5,
        generate: line(0.8, 0.1, 0.12),
    },
    {
        id: 'steep',
        label: 'Steep',
        blurb: 'A strong slope — the line has to rotate a long way to fit.',
        domain: DOMAIN,
        recommendedLr: 0.4,
        generate: line(1.6, -0.1, 0.15),
    },
    {
        id: 'negative',
        label: 'Downward',
        blurb: 'A negative slope — the line tilts the other way.',
        domain: DOMAIN,
        recommendedLr: 0.5,
        generate: line(-0.9, 0.2, 0.12),
    },
    {
        id: 'noisy',
        label: 'Noisy',
        blurb: 'A weak trend buried in noise — the best fit is far from perfect.',
        domain: DOMAIN,
        recommendedLr: 0.5,
        generate: line(0.5, 0.0, 0.4),
    },
];
