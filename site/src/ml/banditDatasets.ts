// Menus for the multi-armed bandit playground. Each "special" is an arm with a *hidden* true
// sell-rate — the probability it sells when featured. The bandit never sees these numbers; it has
// to discover them by featuring specials and watching what happens. The three menus dial up the
// difficulty of the explore/exploit problem: an obvious winner, a near-tie that punishes settling
// too early, and one gem hidden among forgettable specials.

export interface BanditArm {
    name: string;
    rate: number; // hidden P(sells) when featured, in [0, 1]
}

export interface BanditMenu {
    id: string;
    label: string;
    blurb: string;
    arms: BanditArm[];
}

export const BANDIT_MENUS: BanditMenu[] = [
    {
        id: 'clear-winner',
        label: 'Clear winner',
        blurb: 'One special outsells the rest by a mile. Even a little exploring finds it fast — then exploit and never look back.',
        arms: [
            { name: 'Espresso tonic', rate: 0.25 },
            { name: 'Oat-milk latte', rate: 0.45 },
            { name: 'Cinnamon roll', rate: 0.82 },
        ],
    },
    {
        id: 'photo-finish',
        label: 'Photo finish',
        blurb: 'Two specials are almost tied at the top. Stop exploring too soon and you may crown the wrong one — this is where the strategy earns its keep.',
        arms: [
            { name: 'Almond croissant', rate: 0.62 },
            { name: 'Pain au chocolat', rate: 0.58 },
            { name: 'Berry muffin', rate: 0.48 },
            { name: 'Banana bread', rate: 0.4 },
        ],
    },
    {
        id: 'dark-horse',
        label: 'Dark horse',
        blurb: 'Three forgettable specials and one quiet star. The bandit must keep sampling the unremarkable ones long enough to notice the gem.',
        arms: [
            { name: 'Plain scone', rate: 0.32 },
            { name: 'Lemon tart', rate: 0.38 },
            { name: 'Matcha cookie', rate: 0.3 },
            { name: 'Pistachio babka', rate: 0.72 },
        ],
    },
];

/** The best achievable sell-rate on a menu — what a bandit with perfect knowledge would earn per day. */
export function bestRate(menu: BanditMenu): number {
    return Math.max(...menu.arms.map(a => a.rate));
}
