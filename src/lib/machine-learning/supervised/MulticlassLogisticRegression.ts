import LogisticRegression from "./LogisticRegression";
import Matrix from "../../math/linear-algebra/Matrix";

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
    }

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
    }

    public setNumberOfEpochs (numberOfEpochs: number) {
        this.numberOfEpochs = numberOfEpochs;
    }

    public setRegularizationFactor (regularizationFactor: number) {
        this.regularizationFactor = regularizationFactor;
    }

    public setHypothesis (hypothesesPerClass: Matrix[]) {
        this.logisticRegressions.forEach((logisticRegression, i) => logisticRegression.setHypothesis(hypothesesPerClass[i]));
    }

    public resetHypothesis () {
        this.logisticRegressions = undefined;
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