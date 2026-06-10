import type { DataSet, Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// Logic-gate clouds for the perceptron interlude. Four little clusters at the corners of the unit
// square, labelled by a gate. AND and OR are linearly separable — one straight line splits them, so
// the perceptron converges. XOR is not — no single line works, and the perceptron never settles.
export interface GateDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    generate: (seed: number, n: number) => DataSet;
}

const DOMAIN: Domain = { xMin: -0.4, xMax: 1.4, yMin: -0.4, yMax: 1.4 };
const CORNERS = [[0, 0], [0, 1], [1, 0], [1, 1]];

/** Four corner clusters, each labelled by `labels[corner]`. */
function gate(labels: number[]) {
    return (seed: number, n: number): DataSet => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        const targets: number[][] = [];
        const perCorner = Math.max(1, Math.floor(n / 4));
        for (let c = 0; c < 4; c++) {
            for (let j = 0; j < perCorner; j++) {
                inputs.push([CORNERS[c][0] + gaussian(rand) * 0.11, CORNERS[c][1] + gaussian(rand) * 0.11]);
                targets.push([labels[c]]);
            }
        }
        return { inputs, targets };
    };
}

export const GATE_DATASETS: GateDataset[] = [
    {
        id: 'and',
        label: 'AND',
        blurb: 'Fire only when both inputs are high — one corner positive. A single line separates it, so the perceptron locks on and the errors drop to zero.',
        domain: DOMAIN,
        generate: gate([0, 0, 0, 1]),
    },
    {
        id: 'or',
        label: 'OR',
        blurb: 'Fire when either input is high — three corners positive. Still one line away, so the perceptron converges just fine.',
        domain: DOMAIN,
        generate: gate([0, 1, 1, 1]),
    },
    {
        id: 'xor',
        label: 'XOR',
        blurb: 'Fire when exactly one input is high — opposite corners share a class. No straight line can split it, so the boundary lurches forever and the error never reaches zero. This is the wall.',
        domain: DOMAIN,
        generate: gate([0, 1, 1, 0]),
    },
];
