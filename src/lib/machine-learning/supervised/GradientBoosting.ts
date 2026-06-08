import Matrix from "../../math/linear-algebra/Matrix";

/** A node of an internal regression tree: a split (internal) or a mean residual (leaf). */
interface RegressionNode {
    featureIndex?: number;
    threshold?: number;
    left?: RegressionNode;
    right?: RegressionNode;
    value?: number;
}

/**
 * Gradient boosting for binary classification — the tabular-data workhorse. Where a random forest
 * grows many trees in parallel and votes, boosting builds them **in sequence**: it starts from a
 * constant guess (the base rate) and, each round, fits a small regression tree to the part it's
 * still getting wrong — the **residual** between the true label and the current predicted
 * probability — then adds that tree in, shrunk by a `learningRate`. Round after round it chips away
 * at the leftover error.
 *
 * Targets are one-hot (the positive class is the last column), like the other classifiers; the
 * running model lives in log-odds space and `predict` returns `[P(class 0), P(class 1)]` via a
 * sigmoid, so the predicted class is the argmax. There's no randomness here — boosting is fully
 * deterministic given its hyperparameters.
 */
export default class GradientBoosting {

    private numberOfTrees = 50;
    private learningRate = 0.3;
    private maxDepth = 3;
    private minSamplesSplit = 4;

    private trees: RegressionNode[] = [];
    private initialScore = 0; // base log-odds, before any tree

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const rows = inputs.toArray();
        const labels = targets.toArray().map(row => row[row.length - 1]); // positive class = last column
        const exampleCount = rows.length;

        const baseRate = clamp(labels.reduce((sum, value) => sum + value, 0) / exampleCount);
        this.initialScore = Math.log(baseRate / (1 - baseRate));
        this.trees = [];

        const scores = new Array(exampleCount).fill(this.initialScore);

        for (let round = 0; round < this.numberOfTrees; round++) {
            // Negative gradient of the logistic loss = (true label − current probability) = residual.
            const residuals = scores.map((score, i) => labels[i] - sigmoid(score));

            const tree = this.buildTree(rows, residuals, rows.map((_, i) => i), 0);
            this.trees.push(tree);

            for (let i = 0; i < exampleCount; i++) {
                scores[i] += this.learningRate * predictTree(tree, rows[i]);
            }
        }
    }

    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => {
            let score = this.initialScore;
            for (const tree of this.trees) {
                score += this.learningRate * predictTree(tree, row);
            }
            const probability = sigmoid(score);
            return [1 - probability, probability];
        }));
    }

    /* Parameter setters */

    public setNumberOfTrees (numberOfTrees: number) {
        this.numberOfTrees = numberOfTrees;
        return this;
    }

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setMaxDepth (maxDepth: number) {
        this.maxDepth = maxDepth;
        return this;
    }

    public setMinSamplesSplit (minSamplesSplit: number) {
        this.minSamplesSplit = minSamplesSplit;
        return this;
    }

    /* Parameter getters */

    public getNumberOfTrees () {
        return this.numberOfTrees;
    }

    public getLearningRate () {
        return this.learningRate;
    }

    public getMaxDepth () {
        return this.maxDepth;
    }

    public getMinSamplesSplit () {
        return this.minSamplesSplit;
    }

    /* Private: a least-squares regression tree fit to the residuals (leaves hold the mean residual). */

    private buildTree (rows: number[][], residuals: number[], indices: number[], depth: number): RegressionNode {
        const value = mean(residuals, indices);

        if (depth >= this.maxDepth || indices.length < this.minSamplesSplit) {
            return { value };
        }

        const split = bestSplit(rows, residuals, indices);
        if (split === undefined) {
            return { value };
        }

        const left: number[] = [];
        const right: number[] = [];
        for (const i of indices) {
            (rows[i][split.featureIndex] < split.threshold ? left : right).push(i);
        }

        return {
            featureIndex: split.featureIndex,
            threshold: split.threshold,
            left: this.buildTree(rows, residuals, left, depth + 1),
            right: this.buildTree(rows, residuals, right, depth + 1),
        };
    }
}

function sigmoid (z: number) {
    return 1 / (1 + Math.exp(-z));
}

function clamp (probability: number) {
    return Math.min(1 - 1e-6, Math.max(1e-6, probability));
}

function mean (values: number[], indices: number[]) {
    let sum = 0;
    for (const i of indices) sum += values[i];
    return sum / indices.length;
}

/** Sum of squared deviations from the mean — the regression tree's impurity. */
function sse (values: number[], indices: number[]) {
    const m = mean(values, indices);
    let total = 0;
    for (const i of indices) {
        const deviation = values[i] - m;
        total += deviation * deviation;
    }
    return total;
}

function bestSplit (rows: number[][], residuals: number[], indices: number[]) {
    const parentSse = sse(residuals, indices);
    const featureCount = rows[0].length;
    let best: { featureIndex: number; threshold: number; gain: number } | undefined;

    for (let feature = 0; feature < featureCount; feature++) {
        const values = Array.from(new Set(indices.map(i => rows[i][feature]))).sort((a, b) => a - b);

        for (let v = 0; v < values.length - 1; v++) {
            const threshold = (values[v] + values[v + 1]) / 2;
            const left = indices.filter(i => rows[i][feature] < threshold);
            const right = indices.filter(i => rows[i][feature] >= threshold);
            if (left.length === 0 || right.length === 0) continue;

            const gain = parentSse - (sse(residuals, left) + sse(residuals, right));
            if (gain > 0 && (best === undefined || gain > best.gain)) {
                best = { featureIndex: feature, threshold, gain };
            }
        }
    }

    return best;
}

function predictTree (node: RegressionNode, row: number[]): number {
    while (node.value === undefined) {
        node = row[node.featureIndex!] < node.threshold! ? node.left! : node.right!;
    }
    return node.value;
}
