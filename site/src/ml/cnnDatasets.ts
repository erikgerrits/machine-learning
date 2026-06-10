import { gaussian, mulberry32 } from './rng';

// Tiny grayscale "latte-art stroke" images for the CNN playground (Chapter 21). Each image is a
// flattened `size × size` intensity grid (row-major, 0–1) holding one stroke — horizontal, vertical,
// or diagonal — drawn at a random position with a little noise. Because the stroke can sit anywhere,
// a flat dense net would see every position as a different input; a CNN's shared filters spot the
// orientation wherever it lands. Targets are one-hot over the three directions.
export interface CnnDataset {
    id: string;
    label: string;
    blurb: string;
    size: number;
    classNames: string[];
    recommendedFilters: number;
    recommendedLr: number;
    generate: (seed: number, n: number) => { inputs: number[][]; targets: number[][] };
}

const CLASS_NAMES = ['— horizontal', '| vertical', '/ diagonal'];

function strokeImage(rand: () => number, klass: number, size: number, noise: number): number[] {
    const image = new Array<number>(size * size).fill(0);
    const set = (i: number, j: number, v: number) => {
        if (i >= 0 && i < size && j >= 0 && j < size) {
            image[i * size + j] = Math.max(image[i * size + j], v);
        }
    };

    if (klass === 0) {
        const r = 2 + Math.floor(rand() * (size - 4));
        for (let j = 0; j < size; j++) { set(r, j, 1); set(r - 1, j, 0.4); set(r + 1, j, 0.4); }
    } else if (klass === 1) {
        const c = 2 + Math.floor(rand() * (size - 4));
        for (let i = 0; i < size; i++) { set(i, c, 1); set(i, c - 1, 0.4); set(i, c + 1, 0.4); }
    } else {
        const offset = -3 + Math.floor(rand() * 7);
        for (let i = 0; i < size; i++) { const j = i + offset; set(i, j, 1); set(i, j - 1, 0.4); set(i, j + 1, 0.4); }
    }

    for (let p = 0; p < image.length; p++) {
        image[p] = Math.min(1, Math.max(0, image[p] + gaussian(rand) * noise));
    }
    return image;
}

function strokes(size: number, noise: number) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        const targets: number[][] = [];
        for (let i = 0; i < n; i++) {
            const klass = i % 3;
            inputs.push(strokeImage(rand, klass, size, noise));
            targets.push([klass === 0 ? 1 : 0, klass === 1 ? 1 : 0, klass === 2 ? 1 : 0]);
        }
        return { inputs, targets };
    };
}

export const CNN_DATASETS: CnnDataset[] = [
    {
        id: 'strokes',
        label: 'Latte-art strokes',
        blurb: 'Classify the pour direction — horizontal, vertical, or diagonal — wherever the stroke lands. Watch the filters turn into little edge detectors.',
        size: 12,
        classNames: CLASS_NAMES,
        recommendedFilters: 6,
        recommendedLr: 0.25,
        generate: strokes(12, 0.06),
    },
    {
        id: 'messy',
        label: 'Messy pours',
        blurb: 'The same three directions, but grainier — more noise on every pixel. The CNN still finds the orientation; it just needs a few more passes.',
        size: 12,
        classNames: CLASS_NAMES,
        recommendedFilters: 8,
        recommendedLr: 0.2,
        generate: strokes(12, 0.16),
    },
];
