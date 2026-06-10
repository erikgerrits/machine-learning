import { mulberry32 } from './rng';

// "Basket" data for the association-rules playground (Chapter 17). Each row is one receipt; each
// column is a menu item (1 = on the receipt). Baskets are built from planted combos plus a little
// noise, so there are real "buys X also buys Y" patterns to discover — and enough noise that the
// support/confidence thresholds actually matter.
export interface AssociationDataset {
    id: string;
    label: string;
    blurb: string;
    items: string[];
    minSupport: number;
    minConfidence: number;
    generate: (seed: number, n: number) => { inputs: number[][] };
}

const MENU = ['Espresso', 'Latte', 'Croissant', 'Muffin', 'Bagel', 'OJ', 'Cookie', 'Tea'];

/** Build baskets from a set of planted combos (each added with `comboProbability`) plus item noise. */
function basketGenerator(combos: number[][], comboProbability: number, noiseProbability: number) {
    return (seed: number, n: number) => {
        const rand = mulberry32(seed);
        const itemCount = MENU.length;
        const inputs: number[][] = [];

        for (let t = 0; t < n; t++) {
            const basket = new Set<number>();
            for (const combo of combos) {
                if (rand() < comboProbability) {
                    for (const item of combo) {
                        basket.add(item);
                    }
                }
            }
            for (let i = 0; i < itemCount; i++) {
                if (rand() < noiseProbability) {
                    basket.add(i);
                }
            }
            // Avoid empty receipts — give a stray basket one random item.
            if (basket.size === 0) {
                basket.add(Math.floor(rand() * itemCount));
            }

            const row = new Array<number>(itemCount).fill(0);
            for (const item of basket) {
                row[item] = 1;
            }
            inputs.push(row);
        }
        return { inputs };
    };
}

export const ASSOCIATION_DATASETS: AssociationDataset[] = [
    {
        id: 'cafe',
        label: 'Café receipts',
        blurb: 'Clean combos: espresso with a croissant, latte with a muffin, tea with a cookie, bagel with juice. The strong rules jump straight out.',
        items: MENU,
        minSupport: 0.12,
        minConfidence: 0.5,
        // [Espresso, Croissant], [Latte, Muffin], [Tea, Cookie], [Bagel, OJ]
        generate: basketGenerator([[0, 2], [1, 3], [7, 6], [4, 5]], 0.32, 0.04),
    },
    {
        id: 'busy',
        label: 'Busy menu',
        blurb: 'Weaker, overlapping habits and more noise. Crank the support and confidence bars up to burn off the coincidences and leave only the rules worth trusting.',
        items: MENU,
        minSupport: 0.1,
        minConfidence: 0.5,
        // overlapping combos (croissant shows up with both espresso and latte) + a 3-item habit
        generate: basketGenerator([[0, 2], [1, 2], [1, 3], [7, 6], [0, 6], [4, 5, 3]], 0.22, 0.1),
    },
];
