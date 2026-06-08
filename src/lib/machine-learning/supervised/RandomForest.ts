import Matrix from "../../math/linear-algebra/Matrix";
import DecisionTree from "./DecisionTree";
import mulberry32 from "../../math/random/mulberry32";

/**
 * A random forest — a committee of {@link DecisionTree}s whose votes are averaged. Each tree is
 * grown on a different **bootstrap sample** of the data (rows drawn with replacement) and, when
 * `maxFeatures` is set, considers only a random subset of features at each split. Those two dabs
 * of randomness **decorrelate** the trees, so their individual quirks and over-fitting cancel out
 * in the average while the signal they agree on survives.
 *
 * Targets are one-hot, like {@link MulticlassLogisticRegression} and {@link DecisionTree};
 * `predict` returns the averaged class distribution across all trees (argmax = predicted class).
 * The trade compared with a single tree: far steadier predictions, at the cost of the one thing a
 * lone tree gave you — a rulebook you could read.
 */
export default class RandomForest {

    private numberOfTrees = 50;
    private maxDepth = 8;
    private minSamplesSplit = 2;
    private maxFeatures = 0;
    private seed = 0;

    private trees: DecisionTree[] = [];
    private classCount = 0;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const rows = inputs.toArray();
        const labels = targets.toArray();
        const exampleCount = rows.length;
        this.classCount = targets.getColumnCount();
        this.trees = [];

        for (let t = 0; t < this.numberOfTrees; t++) {
            const random = mulberry32(this.seed + t);

            // Bootstrap: draw `exampleCount` rows with replacement — each tree sees a different slice.
            const sampledRows: number[][] = [];
            const sampledTargets: number[][] = [];
            for (let i = 0; i < exampleCount; i++) {
                const index = Math.floor(random() * exampleCount);
                sampledRows.push(rows[index]);
                sampledTargets.push(labels[index]);
            }

            const tree = new DecisionTree()
                .setMaxDepth(this.maxDepth)
                .setMinSamplesSplit(this.minSamplesSplit)
                .setMaxFeatures(this.maxFeatures)
                .setSeed(this.seed + 1000 + t); // a distinct seed so each tree's feature picks differ
            tree.train(new Matrix(sampledRows), new Matrix(sampledTargets));
            this.trees.push(tree);
        }
    }

    public predict (inputs: Matrix) {
        const rowCount = inputs.getRowCount();
        const totals = Array.from({ length: rowCount }, () => new Array(this.classCount).fill(0));

        for (const tree of this.trees) {
            const treePredictions = tree.predict(inputs).toArray();
            for (let i = 0; i < rowCount; i++) {
                for (let c = 0; c < this.classCount; c++) {
                    totals[i][c] += treePredictions[i][c];
                }
            }
        }

        return new Matrix(totals.map(row => row.map(value => value / this.trees.length)));
    }

    /* Parameter setters */

    public setNumberOfTrees (numberOfTrees: number) {
        this.numberOfTrees = numberOfTrees;
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

    public setMaxFeatures (maxFeatures: number) {
        this.maxFeatures = maxFeatures;
        return this;
    }

    public setSeed (seed: number) {
        this.seed = seed;
        return this;
    }

    /* Parameter getters */

    public getNumberOfTrees () {
        return this.numberOfTrees;
    }

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

    public getTrees () {
        return this.trees;
    }
}
