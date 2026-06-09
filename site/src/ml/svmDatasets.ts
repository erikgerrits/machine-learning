import type { DataSet, Domain } from './datasets';
import { gaussian, mulberry32 } from './rng';

// Binary "should we comp this order?" datasets for Chapter 11 (support vector machines). The same
// two features as the trees chapters — how long the customer waited (x) and how badly the order
// went wrong (y), relative to normal — labelled comp (1) / no comp (0). The shapes are chosen to
// show off the margin: a wide clean gap, a narrow noisy one, and a curved case only a kernel can
// carve.
export interface SvmDataset {
    id: string;
    label: string;
    blurb: string;
    domain: Domain;
    /** Whether this set is meant to be split by a straight line ('linear') or needs a kernel ('rbf'). */
    suggestedKernel: 'linear' | 'rbf';
    generate: (seed: number, n: number) => DataSet;
}

const WIDE: Domain = { xMin: -1.6, xMax: 1.6, yMin: -1.6, yMax: 1.6 };

// Two gaussian clouds on a diagonal: fine orders in one corner, disasters in the other.
function blobs(offset: number, spread: number) {
    return (seed: number, n: number): DataSet => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        const targets: number[][] = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < n; i++) {
            const comp = i >= half;
            const cx = comp ? offset : -offset;
            const cy = comp ? offset : -offset;
            inputs.push([cx + gaussian(rand) * spread, cy + gaussian(rand) * spread]);
            targets.push([comp ? 1 : 0]);
        }
        return { inputs, targets };
    };
}

// A pocket of always-comp orders (one specific bad combination) ringed by orders that never get
// comped — a closed curve no straight line can trace.
function ring(seed: number, n: number): DataSet {
    const rand = mulberry32(seed);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < n; i++) {
        const comp = i < half;
        const radius = comp ? 0.45 * Math.sqrt(rand()) : 0.95 + rand() * 0.45;
        const angle = rand() * 2 * Math.PI;
        inputs.push([
            Math.cos(angle) * radius + gaussian(rand) * 0.06,
            Math.sin(angle) * radius + gaussian(rand) * 0.06,
        ]);
        targets.push([comp ? 1 : 0]);
    }
    return { inputs, targets };
}

export const SVM_DATASETS: SvmDataset[] = [
    {
        id: 'clear-cut',
        label: 'Clear-cut',
        blurb: 'Fine orders and outright disasters sit in opposite corners with a wide empty gap between. Many lines separate them — the SVM finds the one dead-centre.',
        domain: WIDE,
        suggestedKernel: 'linear',
        generate: blobs(0.7, 0.22),
    },
    {
        id: 'borderline',
        label: 'Borderline',
        blurb: 'The two clouds crowd together and a few cases land on the wrong side. Lower C to let the margin breathe through the noise; raise it to insist on every point.',
        domain: WIDE,
        suggestedKernel: 'linear',
        generate: blobs(0.42, 0.5),
    },
    {
        id: 'surrounded',
        label: 'Surrounded',
        blurb: 'A pocket of always-comp orders, ringed by ones that never are. A straight line is hopeless — switch the kernel to RBF and watch it wrap the pocket.',
        domain: WIDE,
        suggestedKernel: 'rbf',
        generate: ring,
    },
];
