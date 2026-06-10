import { gaussian, mulberry32 } from './rng';

// Ratings data for the recommender playground (Chapter 18). A users × items matrix of 1–5 ratings
// with 0 meaning "hasn't tried it" — the blanks matrix factorization fills in. Ratings are built
// from a hidden 2-factor taste (drink strength × sweetness), so there's real structure for the model
// to recover, then a chunk is hidden so there's something to recommend.
export interface RecommenderDataset {
    id: string;
    label: string;
    blurb: string;
    users: string[];
    items: string[];
    generate: (seed: number) => { inputs: number[][] };
}

const USERS = ['Alvarez', 'Bea', 'Tomás', 'Priya', 'Sam', 'Iris', 'Owen', 'Mei', 'Dan'];
const ITEMS = ['Espresso', 'Latte', 'Tea', 'Cocoa', 'Croissant', 'Muffin', 'Bagel', 'Cookie'];

// Each item's hidden profile: [strength, sweetness], roughly in [-1, 1].
const ITEM_PROFILES: number[][] = [
    [1.0, -0.6], // Espresso
    [0.6, 0.1], // Latte
    [0.3, -0.2], // Tea
    [-0.6, 0.9], // Cocoa
    [-0.2, 0.5], // Croissant
    [-0.3, 0.9], // Muffin
    [-0.2, -0.7], // Bagel
    [-0.4, 1.0], // Cookie
];

// A few taste archetypes users are drawn from: [strength preference, sweetness preference].
const ARCHETYPES: number[][] = [
    [1.0, -0.8], // strong & savoury
    [-0.6, 1.0], // sweet tooth
    [0.2, 0.2], // balanced
    [0.3, -0.4], // light & savoury
];

function clampRating(value: number) {
    return Math.max(1, Math.min(5, Math.round(value)));
}

function ratingsGenerator(hideProbability: number, tasteNoise: number) {
    return (seed: number) => {
        const rand = mulberry32(seed);
        const inputs: number[][] = [];
        for (let u = 0; u < USERS.length; u++) {
            const base = ARCHETYPES[u % ARCHETYPES.length];
            const taste = [base[0] + gaussian(rand) * tasteNoise, base[1] + gaussian(rand) * tasteNoise];
            const row: number[] = [];
            for (let i = 0; i < ITEMS.length; i++) {
                const profile = ITEM_PROFILES[i];
                const rating = clampRating(3 + 2 * (taste[0] * profile[0] + taste[1] * profile[1]));
                row.push(rand() < hideProbability ? 0 : rating);
            }
            inputs.push(row);
        }
        return { inputs };
    };
}

export const RECOMMENDER_DATASETS: RecommenderDataset[] = [
    {
        id: 'regulars',
        label: 'The regulars',
        blurb: 'Nine regulars, each with a clear taste — strong-and-savoury, sweet tooth, and so on. Half their menu is untried; the model learns each taste and fills the blanks.',
        users: USERS,
        items: ITEMS,
        generate: ratingsGenerator(0.5, 0.18),
    },
    {
        id: 'sparse',
        label: 'Barely any data',
        blurb: 'Most of the grid is blank and tastes are fuzzier. With so few ratings the predictions get shakier — collaborative filtering still leans on the crowd, but it has less to lean on.',
        users: USERS,
        items: ITEMS,
        generate: ratingsGenerator(0.66, 0.3),
    },
];
