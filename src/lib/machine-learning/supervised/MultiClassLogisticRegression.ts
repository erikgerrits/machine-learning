import LogisticRegression from "./LogisticRegression";
import Matrix from "../../math/linear-algebra/Matrix";

export default class MulticlassLogisticRegression {

    private logisticRegressions: LogisticRegression[];

    public constructor (private inputs: Matrix, private outputs: Matrix) {
        this.resetHypothesis();
    }

    public train () {
        this.logisticRegressions.forEach(logisticRegression => logisticRegression.train());
    }

    public predict (inputs: Matrix) {
        return this.logisticRegressions.reduce((accumulatedPredictions: Matrix, logisticRegression) => accumulatedPredictions.appendRight(logisticRegression.predict(inputs)), new Matrix([]));
    }


    /* Parameter setters */

    public setLearningRate (learningRate: number) {
        this.logisticRegressions.forEach(logisticRegression => logisticRegression.setLearningRate(learningRate));
        return this;
    }

    public setMaximumIterations (maximumIterations: number) {
        this.logisticRegressions.forEach(logisticRegression => logisticRegression.setMaximumIterations(maximumIterations));
        return this;
    }

    public setRegularizationFactor (regularizationFactor: number) {
        this.logisticRegressions.forEach(logisticRegression => logisticRegression.setRegularizationFactor(regularizationFactor));
        return this;
    }

    public setHypothesis (hypothesesPerClass: Matrix[]) {
        this.logisticRegressions.forEach((logisticRegression, i) => logisticRegression.setHypothesis(hypothesesPerClass[i]));
    }

    public resetHypothesis () {
        this.logisticRegressions = [];

        for (let i = 0; i < this.outputs.getColumnCount(); i++) {
            this.logisticRegressions.push(new LogisticRegression(this.inputs, this.outputs.getColumn(i)));
        }
    }

    /* Parameter getters */

    public getLearningRate () {
        return this.logisticRegressions[0].getLearningRate();
    }

    public getMaximumIterations () {
        return this.logisticRegressions[0].getMaximumIterations();
    }

    public getRegularizationFactor () {
        return this.logisticRegressions[0].getRegularizationFactor();
    }

    public getHypothesis () {
        return this.logisticRegressions.map(logisticRegression => logisticRegression.getHypothesis());
    }
}