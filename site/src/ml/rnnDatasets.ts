import { mulberry32 } from './rng';

// Tiny café-review data for the RNN playground (Chapter 22). Each review is a sequence of token ids
// (0 = <pad>), right-padded to a common length; the label is its sentiment. Reviews are built from
// templates so the sentiment rides on the adjectives — which is what makes the learned word
// embeddings organise: positive words drift one way, negative the other, fillers stay in the middle.
export interface RnnDataset {
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

// index 0 is reserved for <pad>.
const VOCAB = [
    '<pad>', 'the', 'was', 'and', 'really', // 0–4 fillers
    'coffee', 'latte', 'service', 'place', 'staff', 'cake', // 5–10 nouns
    'great', 'amazing', 'lovely', 'friendly', 'perfect', 'cozy', // 11–16 positive
    'terrible', 'cold', 'slow', 'rude', 'stale', 'awful', // 17–22 negative
];
const NOUNS = [5, 6, 7, 8, 9, 10];
const POSITIVE = [11, 12, 13, 14, 15, 16];
const NEGATIVE = [17, 18, 19, 20, 21, 22];

function pad(tokens: number[], length: number) {
    const row = tokens.slice(0, length);
    while (row.length < length) row.push(0);
    return row;
}

function generator(length: number, withLongTemplates: boolean) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const pick = (arr: number[]) => arr[Math.floor(rand() * arr.length)];
        const inputs: number[][] = [];
        const targets: number[][] = [];

        for (let i = 0; i < n; i++) {
            const positive = i % 2 === 0;
            const adj = () => (positive ? pick(POSITIVE) : pick(NEGATIVE));
            const noun = () => pick(NOUNS);

            const templates: number[][] = [
                [1, noun(), 2, adj()],              // the <noun> was <adj>
                [adj(), noun()],                    // <adj> <noun>
                [1, noun(), 2, 4, adj()],           // the <noun> was really <adj>
            ];
            if (withLongTemplates) {
                templates.push([noun(), 2, adj(), 3, adj()]);          // <noun> was <adj> and <adj>
                templates.push([1, noun(), 2, adj(), 3, 1, noun(), 2, adj()]); // two clauses
            }
            const tokens = templates[Math.floor(rand() * templates.length)];

            inputs.push(pad(tokens, length));
            targets.push(positive ? [1, 0] : [0, 1]);
        }
        return { inputs, targets };
    };
}

export const RNN_DATASETS: RnnDataset[] = [
    {
        id: 'reviews',
        label: 'Café reviews',
        blurb: 'Short reviews — "the latte was lovely", "rude staff". The net reads each one word by word and learns which words swing the sentiment.',
        vocab: VOCAB,
        classNames: ['positive', 'negative'],
        positiveTokens: POSITIVE,
        negativeTokens: NEGATIVE,
        sequenceLength: 6,
        generate: generator(6, false),
    },
    {
        id: 'long',
        label: 'Longer reviews',
        blurb: 'Two-clause reviews that run longer, so the hidden state has to carry meaning further before the verdict. Same vocabulary, more to remember.',
        vocab: VOCAB,
        classNames: ['positive', 'negative'],
        positiveTokens: POSITIVE,
        negativeTokens: NEGATIVE,
        sequenceLength: 10,
        generate: generator(10, true),
    },
];
