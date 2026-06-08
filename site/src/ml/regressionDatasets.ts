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

// Café demand scenarios for Chapter 1. The numbers stay in [-1, 1] (x = temperature, y = items
// sold, both relative to a mild/normal day) so plain gradient descent converges without feature
// scaling — the story supplies the meaning, the axes show the direction.
export const REGRESSION_DATASETS: RegressionDataset[] = [
    {
        id: 'weekday',
        label: 'A normal week',
        blurb: 'Croissant sales drift up on warmer days — a gentle, reliable trend.',
        domain: DOMAIN,
        recommendedLr: 0.5,
        generate: line(0.8, 0.1, 0.12),
    },
    {
        id: 'summer-surge',
        label: 'Summer surge',
        blurb: 'As it heats up, demand climbs fast — the line has to rotate a long way to keep up.',
        domain: DOMAIN,
        recommendedLr: 0.4,
        generate: line(1.6, -0.1, 0.15),
    },
    {
        id: 'cocoa',
        label: 'Hot cocoa',
        blurb: 'Cocoa sells in the cold: the warmer it gets, the less goes out — the line tilts down.',
        domain: DOMAIN,
        recommendedLr: 0.5,
        generate: line(-0.9, 0.2, 0.12),
    },
    {
        id: 'chaotic',
        label: 'Chaotic week',
        blurb: 'A street fair throws everything off — weather barely predicts sales, and no line fits well.',
        domain: DOMAIN,
        recommendedLr: 0.5,
        generate: line(0.5, 0.0, 0.4),
    },
];
