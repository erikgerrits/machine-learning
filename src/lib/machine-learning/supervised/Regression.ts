import Matrix from "../../math/linear-algebra/Matrix";

/**
 * Shared gradient-descent engine for the regression-style models. Subclasses only differ in how
 * they turn the linear combination `inputs · hypothesis` into a prediction (`predictFromEnriched-
 * Inputs`): linear regression returns it directly, logistic regression squashes it with a sigmoid.
 *
 * `train` repeatedly predicts, measures the error against the targets, and steps the hypothesis
 * (the weight vector) downhill. Inputs are "enriched" with a leading column of 1s so the bias
 * term rides along as just another weight. Batch size selects batch / mini-batch / stochastic
 * gradient descent, and an optional L2 regularization factor discourages large weights.
 */
abstract class Regression {

    private hypothesis: Matrix;

    private numberOfEpochs = 1000;
    private batchSize = 0;
    private learningRate = 0.001;
    private regularizationFactor = 0;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const exampleCount = inputs.getRowCount();

        inputs = this.enrichedWithBiases(inputs);

        if (this.hypothesis === undefined) {
            this.hypothesis = Matrix.zeros(inputs.getColumnCount(), 1);
        }

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const batchSize = this.batchSize !== 0 ? this.batchSize : Number.POSITIVE_INFINITY;

            for (let batchStartIndex = 0; batchStartIndex < exampleCount; batchStartIndex += batchSize) {
                const batchEndIndex = Math.min(batchStartIndex + batchSize - 1, inputs.getRowCount() - 1);
                this.trainBatch(inputs.getRows(batchStartIndex, batchEndIndex), targets.getRows(batchStartIndex, batchEndIndex));
            }
        }
    }

    public predict (inputs: Matrix) {
        return this.predictFromEnrichedInputs(this.enrichedWithBiases(inputs));
    }


    /* Parameter setters */

    /**
     * Set batch size to
     * - 0 for batch gradient descent
     * - 1 for stochastic gradient descent
     * - >1 for mini-batch gradient descent
     *
     * @param batchSize
     */
    public setBatchSize (batchSize = 0) {
        this.batchSize = batchSize;
        return this;
    }

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setNumberOfEpochs (numberOfEpochs: number) {
        this.numberOfEpochs = numberOfEpochs;
        return this;
    }

    public setRegularizationFactor (regularizationFactor: number) {
        this.regularizationFactor = regularizationFactor;
        return this;
    }

    public setHypothesis (hypothesis: Matrix) {
        this.hypothesis = hypothesis;
        return this;
    }

    public resetHypothesis () {
        this.hypothesis = undefined;
        return this;
    }

    /* Parameter getters */

    public getLearningRate () {
        return this.learningRate;
    }

    public getNumberOfEpochs () {
        return this.numberOfEpochs;
    }

    public getRegularizationFactor () {
        return this.regularizationFactor;
    }

    public getHypothesis () {
        return this.hypothesis;
    }

    /* Protected methods */

    protected abstract predictFromEnrichedInputs (inputs: Matrix): Matrix;

    /* Private methods */

    private enrichedWithBiases (inputs: Matrix) {
        return Matrix.appendLeft(inputs, Matrix.ones(inputs.getRowCount(), 1));
    }

    private trainBatch (inputs: Matrix, targets: Matrix) {
        const exampleCount = inputs.getRowCount();

        const predictions = this.predictFromEnrichedInputs(inputs);

        const errors = predictions.subtract(targets).transpose();

        const gradient = errors.multiply(inputs).transpose();

        const newHypothesis = Matrix.subtract(this.hypothesis, gradient.multiply(this.learningRate / exampleCount));

        if (this.regularizationFactor > 0) {
            const regularizationVector = this.hypothesis.multiply(this.learningRate * this.regularizationFactor / exampleCount).setElement(0, 0, 0);
            newHypothesis.subtract(regularizationVector);
        }

        this.hypothesis = newHypothesis;
    }
}

export default Regression;