// Scenarios for the contextual-bandit playground. The twist over Chapter 24: there's no single best
// offer — the winner depends on WHO is at the counter. Each scenario lists the offers (arms), the
// customer types (the context), and a hidden table of sell-rates: rates[arm][type] = how often that
// offer sells to that type. The bandit never sees this table; it must learn a per-customer policy by
// trying offers and watching. We encode each customer type as a one-hot context vector, so an arm's
// learned weight for a type lines up directly with its true rate for that type.

export interface ContextualScenario {
    id: string;
    label: string;
    blurb: string;
    arms: string[];      // the offers to choose between
    types: string[];     // the customer contexts that walk in
    rates: number[][];   // hidden rates[armIndex][typeIndex] — P(offer sells | this customer)
}

export const CONTEXTUAL_SCENARIOS: ContextualScenario[] = [
    {
        id: 'rush-hours',
        label: 'Rush hours',
        blurb: 'A different crowd each part of the day, and a different offer wins each one. The bandit has to learn three policies at once — one per column.',
        arms: ['Cinnamon roll', 'Iced latte', 'Espresso tonic'],
        types: ['Morning regular', 'Midday office', 'Evening tourist'],
        rates: [
            [0.80, 0.40, 0.30], // cinnamon roll — a morning thing
            [0.35, 0.75, 0.45], // iced latte — midday favourite
            [0.25, 0.40, 0.78], // espresso tonic — an evening treat
        ],
    },
    {
        id: 'two-crowds',
        label: 'Two crowds',
        blurb: 'Students want cheap, the office crowd wants fancy — and the oat flat white never quite wins anyone, a distractor the bandit must learn to skip.',
        arms: ['Budget drip', 'Oat flat white', 'Pour-over'],
        types: ['Students', 'Office crowd'],
        rates: [
            [0.70, 0.30], // budget drip — students
            [0.45, 0.55], // oat flat white — never the best
            [0.25, 0.72], // pour-over — office crowd
        ],
    },
    {
        id: 'one-size',
        label: 'One size fits all',
        blurb: 'Here context is a red herring: the cookie wins for everyone. A contextual bandit should discover that the customer type doesn’t matter and behave like a plain bandit.',
        arms: ['Cookie', 'Brownie'],
        types: ['Regulars', 'Newcomers'],
        rates: [
            [0.70, 0.68], // cookie — wins for both
            [0.40, 0.42], // brownie — loses to both
        ],
    },
];

/** One-hot context vector for customer type `typeIndex` in a scenario with `types.length` types. */
export function contextOf(scenario: ContextualScenario, typeIndex: number): number[] {
    return scenario.types.map((_, i) => (i === typeIndex ? 1 : 0));
}

/** The best achievable sell-rate for a given customer type — what a perfect, all-knowing chooser earns. */
export function bestRateForType(scenario: ContextualScenario, typeIndex: number): number {
    return Math.max(...scenario.arms.map((_, arm) => scenario.rates[arm][typeIndex]));
}
