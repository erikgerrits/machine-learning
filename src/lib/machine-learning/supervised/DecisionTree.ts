import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/** A node in a trained tree: either a split (internal) or a class distribution (leaf). */
export interface DecisionTreeNode {
    /** Internal node: send a row left if `feature[featureIndex] < threshold`, else right. */
    featureIndex?: number;
    threshold?: number;
    left?: DecisionTreeNode;
    right?: DecisionTreeNode;
    /** Leaf node: the class probability distribution of the training rows that reached it. */
    distribution?: number[];
}

/**
 * A CART-style classification tree — a flowchart of yes/no questions, each of the form "is
 * feature j below some threshold?", chosen greedily to split the classes as cleanly as possible
 * (the split that most reduces **Gini impurity**). Targets are one-hot (one column per class),
 * like {@link MulticlassLogisticRegression}; `predict` returns each leaf's class distribution, so
 * the predicted class is the argmax across the columns.
 *
 * There's no loss curve and no gradient descent: the whole tree is built in a single recursive
 * pass. Its great virtue is legibility — the trained tree (`getRoot`) reads back as plain rules.
 * Depth is the main dial: a shallow tree underfits, a deep one can carve a tiny region around
 * every noisy point (overfitting), so `setMaxDepth` doubles as the regularizer.
 */
export default class DecisionTree {

    private maxDepth = 5;
    private minSamplesSplit = 2;
    private maxFeatures = 0; // 0 = consider every feature at each split (a plain, deterministic tree)
    private seed: number | undefined = undefined;

    private root: DecisionTreeNode;
    private classCount = 0;
    private random: () => number = () => 0; // seeded per train(), only used when maxFeatures > 0

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const rows = inputs.toArray();
        const labels = targets.toArray().map(argmax);
        this.classCount = targets.getColumnCount();
        this.random = mulberry32(this.seed !== undefined ? this.seed : Math.floor(Math.random() * 0x100000000));
        this.root = this.buildNode(rows, labels, rows.map((_, i) => i), 0);
    }

    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => this.distributionFor(row)));
    }

    /* Parameter setters */

    public setMaxDepth (maxDepth: number) {
        this.maxDepth = maxDepth;
        return this;
    }

    public setMinSamplesSplit (minSamplesSplit: number) {
        this.minSamplesSplit = minSamplesSplit;
        return this;
    }

    /** Number of features to consider at each split (0 = all). Random forests set this < total. */
    public setMaxFeatures (maxFeatures: number) {
        this.maxFeatures = maxFeatures;
        return this;
    }

    /** Seed for the per-split feature sampling (only matters when maxFeatures > 0). */
    public setSeed (seed: number) {
        this.seed = seed;
        return this;
    }

    /* Parameter getters */

    public getMaxDepth () {
        return this.maxDepth;
    }

    public getMinSamplesSplit () {
        return this.minSamplesSplit;
    }

    public getMaxFeatures () {
        return this.maxFeatures;
    }

    public getSeed () {
        return this.seed;
    }

    public getRoot () {
        return this.root;
    }

    /* Private methods */

    private buildNode (rows: number[][], labels: number[], indices: number[], depth: number): DecisionTreeNode {
        const distribution = this.distribution(labels, indices);

        // Stop: max depth reached, too few samples to split, or the node is already pure.
        if (depth >= this.maxDepth || indices.length < this.minSamplesSplit || distribution.some(p => p === 1)) {
            return { distribution };
        }

        const split = this.bestSplit(rows, labels, indices);
        if (split === undefined) {
            return { distribution };
        }

        const left: number[] = [];
        const right: number[] = [];
        for (const i of indices) {
            (rows[i][split.featureIndex] < split.threshold ? left : right).push(i);
        }

        return {
            featureIndex: split.featureIndex,
            threshold: split.threshold,
            left: this.buildNode(rows, labels, left, depth + 1),
            right: this.buildNode(rows, labels, right, depth + 1),
        };
    }

    private bestSplit (rows: number[][], labels: number[], indices: number[]) {
        const parentImpurity = gini(this.distribution(labels, indices));
        const featureCount = rows[0].length;
        let best: { featureIndex: number; threshold: number; gain: number } | undefined;

        for (const feature of this.featureSubset(featureCount)) {
            const values = Array.from(new Set(indices.map(i => rows[i][feature]))).sort((a, b) => a - b);

            for (let v = 0; v < values.length - 1; v++) {
                const threshold = (values[v] + values[v + 1]) / 2;
                const left = indices.filter(i => rows[i][feature] < threshold);
                const right = indices.filter(i => rows[i][feature] >= threshold);
                if (left.length === 0 || right.length === 0) continue;

                const weightedImpurity =
                    (left.length / indices.length) * gini(this.distribution(labels, left)) +
                    (right.length / indices.length) * gini(this.distribution(labels, right));
                const gain = parentImpurity - weightedImpurity;

                if (gain > 0 && (best === undefined || gain > best.gain)) {
                    best = { featureIndex: feature, threshold, gain };
                }
            }
        }

        return best;
    }

    /** All feature indices, or a fresh random subset of `maxFeatures` of them (the forest's trick). */
    private featureSubset (featureCount: number) {
        const indices = Array.from({ length: featureCount }, (_, i) => i);
        if (this.maxFeatures <= 0 || this.maxFeatures >= featureCount) {
            return indices;
        }
        // Fisher–Yates partial shuffle: draw `maxFeatures` features at random for this split.
        for (let i = 0; i < this.maxFeatures; i++) {
            const j = i + Math.floor(this.random() * (featureCount - i));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        return indices.slice(0, this.maxFeatures);
    }

    private distribution (labels: number[], indices: number[]) {
        const counts = new Array(this.classCount).fill(0);
        for (const i of indices) counts[labels[i]]++;
        return counts.map(count => count / indices.length);
    }

    private distributionFor (row: number[]): number[] {
        let node = this.root;
        // A node without a distribution is always an internal split, so its fields are defined.
        while (node.distribution === undefined) {
            node = row[node.featureIndex!] < node.threshold! ? node.left! : node.right!;
        }
        return node.distribution;
    }
}

function argmax (values: number[]) {
    let bestIndex = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[bestIndex]) {
            bestIndex = i;
        }
    }
    return bestIndex;
}

/** Gini impurity of a class distribution: 0 when pure, highest when the classes are even. */
function gini (distribution: number[]) {
    return 1 - distribution.reduce((sum, probability) => sum + probability * probability, 0);
}
