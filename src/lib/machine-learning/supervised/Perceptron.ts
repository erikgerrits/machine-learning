import Matrix from "../../math/linear-algebra/Matrix";

/**
 * The **perceptron** — a single artificial neuron, and the historical seed of every neural network.
 * It computes one weighted sum of its inputs plus a bias, then fires (`1`) or doesn't (`0`) through
 * a hard **step**: `output = (w · x + b ≥ 0) ? 1 : 0`. Its decision boundary is therefore a single
 * straight line (or plane).
 *
 * It learns by the **perceptron rule**, not gradient descent: walk the examples, and whenever it
 * gets one wrong, nudge the weights and bias *toward* that example's correct answer by
 * `learningRate · (target − prediction) · x`. Rosenblatt's convergence theorem promises that if the
 * two classes can be separated by a straight line, this finds such a line in a finite number of
 * passes. The flip side — famously — is that if they *can't* (the XOR problem), it never settles: the
 * line keeps lurching, forever. That limitation is exactly what stacking neurons into layers
 * (a {@link FeedforwardNeuralNetwork}) overcomes.
 *
 * Targets are a single `0`/`1` column. Weights start at zero, so training is deterministic; like the
 * other iterative models, repeated `train()` calls continue the same run (handy for animating the
 * boundary as it learns).
 *
 * @example
 * const perceptron = new Perceptron().setLearningRate(0.1).setNumberOfEpochs(20);
 * perceptron.train(new Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]), new Matrix([[0], [0], [0], [1]]));
 * perceptron.predict(new Matrix([[1, 1]])).toArray(); // [[1]] — it learned AND
 */
export default class Perceptron {

    private learningRate = 0.1;
    private numberOfEpochs = 10;

    private weights: number[];
    private bias = 0;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const rows = inputs.toArray();
        const labels = targets.toArray().map(row => (row[0] >= 0.5 ? 1 : 0));
        const featureCount = rows.length > 0 ? rows[0].length : 0;

        if (this.weights === undefined || this.weights.length !== featureCount) {
            this.weights = new Array<number>(featureCount).fill(0);
            this.bias = 0;
        }

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            for (let i = 0; i < rows.length; i++) {
                const prediction = this.activate(rows[i]);
                const error = labels[i] - prediction; // -1, 0, or +1

                if (error !== 0) {
                    for (let f = 0; f < featureCount; f++) {
                        this.weights[f] += this.learningRate * error * rows[i][f];
                    }
                    this.bias += this.learningRate * error;
                }
            }
        }

        return this;
    }

    /** The neuron's `0`/`1` output for each input row (step of the weighted sum). */
    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => [this.activate(row)]));
    }

    public reset () {
        this.weights = undefined;
        this.bias = 0;
        return this;
    }

    /* Parameter setters */

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setNumberOfEpochs (numberOfEpochs: number) {
        this.numberOfEpochs = numberOfEpochs;
        return this;
    }

    /* Parameter getters */

    public getLearningRate () {
        return this.learningRate;
    }

    public getNumberOfEpochs () {
        return this.numberOfEpochs;
    }

    /** The learned weight per feature (a `1 × d` row), defining the orientation of the boundary line. */
    public getWeights () {
        return new Matrix([this.weights ? this.weights.slice() : []]);
    }

    public getBias () {
        return this.bias;
    }

    /* Private methods */

    private activate (row: number[]) {
        if (this.weights === undefined) {
            return 0;
        }
        let sum = this.bias;
        for (let f = 0; f < this.weights.length; f++) {
            sum += this.weights[f] * row[f];
        }
        return sum >= 0 ? 1 : 0;
    }
}
