import Matrix from "../../math/linear-algebra/Matrix";

/** A frequent itemset: a set of item indices that co-occur in at least `minSupport` of baskets. */
export interface FrequentItemset {
    items: number[];
    support: number;
}

/**
 * An association rule `antecedent → consequent` with its three classic strength measures:
 * - **support** — fraction of all baskets containing the whole rule (how common it is),
 * - **confidence** — of baskets with the antecedent, the fraction that also have the consequent
 *   (how reliable it is), and
 * - **lift** — confidence divided by the consequent's own support (how much more often they appear
 *   together than chance alone would predict; `> 1` means a real positive association).
 */
export interface AssociationRule {
    antecedent: number[];
    consequent: number[];
    support: number;
    confidence: number;
    lift: number;
}

/**
 * **Association rule mining** by the **Apriori** algorithm — the "market basket" model. Where every
 * other method here treats a row as a point in space, this one treats it as a **basket**: a set of
 * items that were bought (or events that happened) together. It hunts for the patterns *"customers
 * who get X tend to also get Y"* — the coffee-and-croissant combos hiding in the receipts.
 *
 * Apriori works in two stages. First it finds the **frequent itemsets** — sets of items that appear
 * together in at least `minSupport` of baskets — growing them one item at a time and leaning on the
 * key shortcut that *a set can only be frequent if all of its subsets are* (so it never wastes time
 * counting a pair whose halves are already rare). Then it turns each frequent itemset into candidate
 * **rules** and keeps the ones whose `confidence` clears the bar, reporting support, confidence, and
 * lift for each.
 *
 * Input is a binary **basket matrix**: one row per basket, one column per item, `1` if the basket
 * holds that item. It takes no targets. `predict` completes a basket — for each row it scores the
 * items *not* present by the confidence of the best rule that fires, i.e. "what to suggest next".
 *
 * @example
 * const rules = new AssociationRules().setMinSupport(0.4).setMinConfidence(0.6);
 * rules.train(new Matrix([[1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]]));
 * rules.getRules(); // e.g. [{ antecedent: [1], consequent: [0], confidence: 1, lift: 1.33, … }]
 */
export default class AssociationRules {

    private minSupport = 0.1;
    private minConfidence = 0.5;
    private maxItemsetSize = 3;

    private itemCount = 0;
    private frequentItemsets: FrequentItemset[] = [];
    private rules: AssociationRule[] = [];
    private supportByKey = new Map<string, number>();

    public constructor () {}

    public train (inputs: Matrix) {
        const rows = inputs.toArray();
        const n = rows.length;
        this.itemCount = n > 0 ? rows[0].length : 0;

        // Each basket as a set of the item indices it contains.
        const baskets = rows.map(row => {
            const set = new Set<number>();
            for (let i = 0; i < row.length; i++) {
                if (row[i] >= 0.5) {
                    set.add(i);
                }
            }
            return set;
        });

        const minCount = this.minSupport * n;
        this.frequentItemsets = [];
        this.supportByKey = new Map();

        // Stage 1, level 1: the frequent single items.
        const singleCounts = new Array<number>(this.itemCount).fill(0);
        for (const basket of baskets) {
            for (const item of basket) {
                singleCounts[item]++;
            }
        }
        const frequentSingles: number[] = [];
        let current: number[][] = [];
        for (let i = 0; i < this.itemCount; i++) {
            if (n > 0 && singleCounts[i] >= minCount) {
                frequentSingles.push(i);
                current.push([i]);
                this.record([i], singleCounts[i] / n);
            }
        }

        // Stage 1, levels 2..maxItemsetSize: grow frequent sets one item at a time.
        for (let k = 2; k <= this.maxItemsetSize && current.length > 0; k++) {
            const candidates = generateCandidates(current, frequentSingles);
            const next: number[][] = [];
            for (const candidate of candidates) {
                let count = 0;
                for (const basket of baskets) {
                    if (candidate.every(item => basket.has(item))) {
                        count++;
                    }
                }
                if (count >= minCount) {
                    next.push(candidate);
                    this.record(candidate, count / n);
                }
            }
            current = next;
        }

        // Stage 2: turn frequent itemsets into rules that clear the confidence bar.
        this.rules = [];
        for (const itemset of this.frequentItemsets) {
            if (itemset.items.length < 2) {
                continue;
            }
            for (const antecedent of nonEmptyProperSubsets(itemset.items)) {
                const consequent = itemset.items.filter(item => !antecedent.includes(item));
                const confidence = itemset.support / this.support(antecedent);
                if (confidence >= this.minConfidence) {
                    const lift = confidence / this.support(consequent);
                    this.rules.push({ antecedent, consequent, support: itemset.support, confidence, lift });
                }
            }
        }
        this.rules.sort((a, b) => b.lift - a.lift || b.confidence - a.confidence);

        return this;
    }

    /**
     * Basket completion: for each input basket, a score per item it does *not* already contain —
     * the confidence of the strongest rule whose antecedent the basket satisfies and whose
     * consequent is that item (0 if no rule fires, or the item is already present).
     */
    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => {
            const present = new Set<number>();
            for (let i = 0; i < row.length; i++) {
                if (row[i] >= 0.5) {
                    present.add(i);
                }
            }

            const scores = new Array<number>(this.itemCount).fill(0);
            for (const rule of this.rules) {
                if (rule.consequent.length !== 1) {
                    continue;
                }
                const item = rule.consequent[0];
                if (present.has(item)) {
                    continue;
                }
                if (rule.antecedent.every(a => present.has(a)) && rule.confidence > scores[item]) {
                    scores[item] = rule.confidence;
                }
            }
            return scores;
        }));
    }

    /* Parameter setters */

    public setMinSupport (minSupport: number) {
        this.minSupport = minSupport;
        return this;
    }

    public setMinConfidence (minConfidence: number) {
        this.minConfidence = minConfidence;
        return this;
    }

    public setMaxItemsetSize (maxItemsetSize: number) {
        this.maxItemsetSize = maxItemsetSize;
        return this;
    }

    /* Parameter getters */

    public getMinSupport () {
        return this.minSupport;
    }

    public getMinConfidence () {
        return this.minConfidence;
    }

    public getMaxItemsetSize () {
        return this.maxItemsetSize;
    }

    /** Every itemset that cleared `minSupport`, with its support. */
    public getFrequentItemsets (): FrequentItemset[] {
        return this.frequentItemsets.map(itemset => ({ items: itemset.items.slice(), support: itemset.support }));
    }

    /** Every rule that cleared `minConfidence`, sorted strongest (highest lift) first. */
    public getRules (): AssociationRule[] {
        return this.rules.map(rule => ({ ...rule, antecedent: rule.antecedent.slice(), consequent: rule.consequent.slice() }));
    }

    /* Private methods */

    private record (items: number[], support: number) {
        this.frequentItemsets.push({ items, support });
        this.supportByKey.set(items.join(","), support);
    }

    private support (items: number[]) {
        return this.supportByKey.get(items.join(",")) ?? 0;
    }
}

/** Grow size-k candidates by adding each frequent single item to a frequent (k-1)-itemset. */
function generateCandidates (previous: number[][], frequentSingles: number[]) {
    const seen = new Set<string>();
    const candidates: number[][] = [];
    for (const itemset of previous) {
        for (const item of frequentSingles) {
            if (itemset.includes(item)) {
                continue;
            }
            const candidate = [...itemset, item].sort((a, b) => a - b);
            const key = candidate.join(",");
            if (!seen.has(key)) {
                seen.add(key);
                candidates.push(candidate);
            }
        }
    }
    return candidates;
}

/** All subsets of a sorted itemset except the empty set and the whole set (order preserved). */
function nonEmptyProperSubsets (items: number[]) {
    const subsets: number[][] = [];
    const total = 1 << items.length;
    for (let mask = 1; mask < total - 1; mask++) {
        const subset: number[] = [];
        for (let i = 0; i < items.length; i++) {
            if (mask & (1 << i)) {
                subset.push(items[i]);
            }
        }
        subsets.push(subset);
    }
    return subsets;
}
