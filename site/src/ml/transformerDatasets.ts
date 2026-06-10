import { mulberry32 } from './rng';

// Fixed-length café reviews for the transformer playground (Chapter 23). Position 0 is a [CLS] slot;
// the rest are filler words with exactly one sentiment word dropped in at a random spot. The class
// is that word's polarity — so the model has to find the one word that matters, wherever it sits,
// and the [CLS] attention row shows it doing exactly that.
export interface TransformerDataset {
    id: string;
    label: string;
    blurb: string;
    vocab: string[];
    classNames: string[];
    positiveTokens: number[];
    negativeTokens: number[];
    sequenceLength: number;
    generate: (seed: number, n: number) => { inputs: number[][]; targets: number[][] };
}

const VOCAB = [
    '[CLS]', 'the', 'was', 'place', 'coffee', 'service', 'staff', // 0–6: CLS + fillers
    'great', 'lovely', 'perfect', // 7–9 positive
    'terrible', 'slow', 'rude', // 10–12 negative
];
const FILLERS = [1, 2, 3, 4, 5, 6];
const POSITIVE = [7, 8, 9];
const NEGATIVE = [10, 11, 12];

function generator(length: number) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const pick = (arr: number[]) => arr[Math.floor(rand() * arr.length)];
        const inputs: number[][] = [];
        const targets: number[][] = [];

        for (let i = 0; i < n; i++) {
            const positive = i % 2 === 0;
            const seq = [0]; // [CLS]
            for (let p = 1; p < length; p++) seq.push(pick(FILLERS));
            const slot = 1 + Math.floor(rand() * (length - 1));
            seq[slot] = positive ? pick(POSITIVE) : pick(NEGATIVE);

            inputs.push(seq);
            targets.push(positive ? [1, 0] : [0, 1]);
        }
        return { inputs, targets };
    };
}

export const TRANSFORMER_DATASETS: TransformerDataset[] = [
    {
        id: 'reviews',
        label: 'Café reviews',
        blurb: 'One sentiment word ("lovely", "rude") hidden among fillers, at a random spot. The model must find it — watch the [CLS] attention bar lock onto it.',
        vocab: VOCAB,
        classNames: ['positive', 'negative'],
        positiveTokens: POSITIVE,
        negativeTokens: NEGATIVE,
        sequenceLength: 6,
        generate: generator(6),
    },
    {
        id: 'long',
        label: 'Longer reviews',
        blurb: 'More filler to look past before finding the one word that decides the sentiment. Attention still picks it out — that\'s the point of looking everywhere at once.',
        vocab: VOCAB,
        classNames: ['positive', 'negative'],
        positiveTokens: POSITIVE,
        negativeTokens: NEGATIVE,
        sequenceLength: 9,
        generate: generator(9),
    },
];
