import Matrix from "../../math/linear-algebra/Matrix";
import Regression from "./Regression";

/**
 * Predicts a continuous value as a straight-line (linear) combination of the inputs:
 * `prediction = inputs · hypothesis`. Trained by gradient descent on squared error — the classic
 * "fit a line/plane through the data" model. See {@link Regression} for the shared training loop.
 *
 * @example
 * const model = new LinearRegression().setLearningRate(0.02).setNumberOfEpochs(10000);
 * model.train(new Matrix([[5], [7], [9]]), new Matrix([[2000], [2400], [2800]]));
 * model.predict(new Matrix([[11]])); // ≈ [[3200]]
 */
export default class LinearRegression extends Regression {

    protected predictFromEnrichedInputs (inputs: Matrix) {
        return Matrix.multiply(inputs, this.getHypothesis());
    }
}