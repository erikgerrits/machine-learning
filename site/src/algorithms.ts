/**
 * The catalog of algorithms the site covers, in **course order** — each is a chapter of the
 * story-driven course (Nadia & The Drifting Leaf; see docs/course-plan.md). The sidebar nav and
 * landing cards derive from this list; routes are declared (and lazy-loaded) in App.tsx. Chapters
 * carry their course `chapter` number and `part`; gaps in the numbering are roadmap chapters not
 * yet built. Bring one live by flipping `status` and adding its tutorial route.
 */
export interface AlgorithmEntry {
    id: string;
    path: string;
    title: string;
    tagline: string;
    status: 'live' | 'soon';
    /** Course chapter number (gaps = roadmap chapters still to come). */
    chapter?: number;
    /** Course part this chapter belongs to (used to group the sidebar). */
    part?: string;
}

const PART_0 = 'Part 0 · Foundations';
const PART_1 = 'Part 1 · Predicting & deciding';
const PART_2 = 'Part 2 · A wider toolbox';
const PART_3 = 'Part 3 · Understanding customers';
const PART_4 = 'Part 4 · Deep learning';

export const ALGORITHMS: AlgorithmEntry[] = [
    {
        id: 'the-ledger',
        path: '/the-ledger',
        title: 'The Ledger',
        tagline: 'A failing café, a shoebox of receipts, and a way of seeing.',
        status: 'live',
        part: PART_0,
    },
    {
        id: 'linear-regression',
        path: '/linear-regression',
        title: 'Linear Regression',
        tagline: 'How many croissants to bake before sunrise?',
        status: 'live',
        chapter: 1,
        part: PART_1,
    },
    {
        id: 'many-features',
        path: '/many-features',
        title: 'Many Features',
        tagline: 'One clue is not enough — predict from temperature, day, and crowd at once.',
        status: 'live',
        chapter: 2,
        part: PART_1,
    },
    {
        id: 'overfitting',
        path: '/overfitting',
        title: 'Overfitting',
        tagline: 'Too many clues, and the model memorises instead of learning.',
        status: 'live',
        chapter: 3,
        part: PART_1,
    },
    {
        id: 'logistic-regression',
        path: '/logistic-regression',
        title: 'Logistic Regression',
        tagline: 'Will this batch of dough rise? A yes/no call.',
        status: 'live',
        chapter: 4,
        part: PART_1,
    },
    {
        id: 'multiclass-logistic-regression',
        path: '/multiclass-logistic-regression',
        title: 'Multiclass Logistic',
        tagline: 'Under-proofed, just right, or over? Sorting into three.',
        status: 'live',
        chapter: 5,
        part: PART_1,
    },
    {
        id: 'nearest-neighbors',
        path: '/nearest-neighbors',
        title: 'Nearest Neighbors',
        tagline: 'Sort lookalike pastries by the company they keep.',
        status: 'live',
        chapter: 6,
        part: PART_2,
    },
    {
        id: 'naive-bayes',
        path: '/naive-bayes',
        title: 'Naive Bayes',
        tagline: 'Triage the inbox: reservation, complaint, or spam?',
        status: 'live',
        chapter: 7,
        part: PART_2,
    },
    {
        id: 'decision-trees',
        path: '/decision-trees',
        title: 'Decision Trees',
        tagline: 'A "comp this order?" policy you can read and defend.',
        status: 'live',
        chapter: 8,
        part: PART_2,
    },
    {
        id: 'random-forests',
        path: '/random-forests',
        title: 'Random Forests',
        tagline: 'Ask a crowd of trees, average the vote — steadier than one.',
        status: 'live',
        chapter: 9,
        part: PART_2,
    },
    {
        id: 'gradient-boosting',
        path: '/gradient-boosting',
        title: 'Gradient Boosting',
        tagline: 'Build trees in sequence, each fixing the last’s mistakes.',
        status: 'live',
        chapter: 10,
        part: PART_2,
    },
    {
        id: 'k-means',
        path: '/k-means',
        title: 'k-Means',
        tagline: 'Finds clusters in unlabelled data all on its own.',
        status: 'live',
        chapter: 12,
        part: PART_3,
    },
    {
        id: 'neural-network',
        path: '/neural-network',
        title: 'Neural Network',
        tagline: 'Bends curved boundaries no straight line could draw.',
        status: 'live',
        chapter: 20,
        part: PART_4,
    },
];
