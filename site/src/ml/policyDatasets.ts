import type { DataSet, Domain } from './datasets';
import { mulberry32 } from './rng';

// Binary "should we comp this order?" datasets for Chapter 8. Two features per case — how long
// the customer waited (x) and how badly the order went wrong (y), both relative to normal — and a
// yes/no label set by a hidden policy. A little label noise so deeper trees visibly overfit.
export interface PolicyDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    generate: (seed: number, n: number) => DataSet;
}

const DOMAIN: Domain = { xMin: -1.2, xMax: 1.2, yMin: -1.2, yMax: 1.2 };

function fromRule(rule: (x: number, y: number) => boolean, noise = 0.06) {
    return (seed: number, n: number): DataSet => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        const targets: number[][] = [];
        for (let i = 0; i < n; i++) {
            const x = rand() * 2.4 - 1.2;
            const y = rand() * 2.4 - 1.2;
            let label = rule(x, y) ? 1 : 0;
            if (rand() < noise) label = 1 - label; // a few mislabelled cases — the noise to resist
            inputs.push([x, y]);
            targets.push([label]);
        }
        return { inputs, targets };
    };
}

export const POLICY_DATASETS: PolicyDataset[] = [
    {
        id: 'two-strikes',
        label: 'Two strikes',
        blurb: 'Comp only when the wait was long AND the order went badly wrong — one corner of the map. Two cuts and the tree has it.',
        domain: DOMAIN,
        generate: fromRule((x, y) => x > 0.2 && y > 0.2),
    },
    {
        id: 'either-or',
        label: 'Either-or',
        blurb: 'Comp if either the wait or the trouble was bad — an L-shaped region the tree carves with a couple of splits.',
        domain: DOMAIN,
        generate: fromRule((x, y) => x > 0.3 || y > 0.3),
    },
    {
        id: 'quadrants',
        label: 'Quadrants',
        blurb: 'A tricky policy: comp when exactly one thing went wrong. No straight line splits it — but two cuts do.',
        domain: DOMAIN,
        generate: fromRule((x, y) => x > 0 !== y > 0),
    },
];
