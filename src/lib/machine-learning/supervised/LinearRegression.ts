import Matrix from "../../math/linear-algebra/Matrix";

export default class LinearRegression {

    private hypothesis: Matrix;

    private learningRate = 1;
    private maximumIterations = 1000;
    private regularizationFactor = 0;
    private errorStop = 0;

    public constructor (private inputs: Matrix, private outputs: Matrix) {
        this.inputs = this.enrichedWithBiases(inputs);
        this.resetHypothesis();
    }

    public train () {
        const exampleCount = this.inputs.getRowCount();

        let totalError = Number.POSITIVE_INFINITY;

        let i = 0;
        while (totalError > this.errorStop && i++ < this.maximumIterations) {
            const predictions = Matrix.multiply(this.inputs, this.hypothesis);

            const errors = predictions.subtract(this.outputs);
            totalError = Matrix.transform(errors, (error) => Math.abs(error)).getSum();

            const errorsRowVector = Matrix.transpose(errors);
            const gradient = Matrix.multiply(errorsRowVector, this.inputs).transpose();

            const newHypothesis = Matrix.subtract(this.hypothesis, gradient.multiply(this.learningRate / exampleCount));

            if (this.regularizationFactor > 0) {
                const regularizationVector = this.hypothesis.multiply(this.learningRate * this.regularizationFactor / exampleCount).setElement(0, 0, 0);
                newHypothesis.subtract(regularizationVector);
            }

            this.hypothesis = newHypothesis;
        }

        return this.hypothesis;
    }

    public predict (inputs: Matrix) {
        return Matrix.multiply(this.enrichedWithBiases(inputs), this.hypothesis);
    }


    /* Parameter setters */

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setMaximumIterations (maximumIterations: number) {
        this.maximumIterations = maximumIterations;
        return this;
    }

    public setRegularizationFactor (regularizationFactor: number) {
        this.regularizationFactor = regularizationFactor;
        return this;
    }

    public setHypothesis (hypothesis: Matrix) {
        this.hypothesis = hypothesis;
    }

    public resetHypothesis () {
        this.hypothesis = Matrix.zeros(this.inputs.getColumnCount(), 1);
    }

    /* Parameter getters */

    public getLearningRate () {
        return this.learningRate;
    }

    public getMaximumIterations () {
        return this.maximumIterations;
    }

    public getRegularizationFactor () {
        return this.regularizationFactor;
    }

    public getHypothesis () {
        return this.hypothesis;
    }

    /* Private methods */

    private enrichedWithBiases (inputs: Matrix) {
        return Matrix.appendLeft(inputs, Matrix.ones(inputs.getRowCount(), 1));
    }
}