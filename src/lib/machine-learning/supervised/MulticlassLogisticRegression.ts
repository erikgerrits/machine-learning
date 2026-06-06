import LogisticRegression from "./LogisticRegression";
import Matrix from "../../math/linear-algebra/Matrix";

/**
 * Extends binary {@link LogisticRegression} to many classes with the **one-vs-rest** strategy:
 * train one logistic classifier per class (each learning "this class vs. everything else"), then
 * predict by taking the class whose classifier is most confident. Targets are one-hot encoded
 * (one column per class); the predicted class is the argmax across the output columns.
 */
export default class MulticlassLogisticRegression {

    private numberOfEpochs = 1000;
    private batchSize = 0;
    private learningRate = 0.001;
    private regularizationFactor = 0;

    private logisticRegressions: LogisticRegression[];

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        if (this.logisticRegressions === undefined) {
            this.logisticRegressions = [];

            for (let i = 0; i < targets.getColumnCount(); i++) {
                const logisticRegression = new LogisticRegression();
                logisticRegression.setNumberOfEpochs(this.numberOfEpochs);
                logisticRegression.setBatchSize(this.batchSize);
                logisticRegression.setLearningRate(this.learningRate);
                logisticRegression.setRegularizationFactor(this.regularizationFactor);
                this.logisticRegressions.push(logisticRegression);
            }
        }

        this.logisticRegressions.forEach((logisticRegression, i) => logisticRegression.train(inputs, targets.getColumn(i)));
    }

    public predict (inputs: Matrix) {
        return this.logisticRegressions.reduce((accumulatedPredictions: Matrix, logisticRegression) => accumulatedPredictions.appendRight(logisticRegression.predict(inputs)), new Matrix([]));
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

    public setHypothesis (hypothesesPerClass: Matrix[]) {
        this.logisticRegressions.forEach((logisticRegression, i) => logisticRegression.setHypothesis(hypothesesPerClass[i]));
        return this;
    }

    public resetHypothesis () {
        this.logisticRegressions = undefined;
        return this;
    }

    /* Parameter getters */

    public getBatchSize () {
        return this.batchSize;
    }

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
        return this.logisticRegressions.map(logisticRegression => logisticRegression.getHypothesis());
    }
}