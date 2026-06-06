import Matrix from "../../math/linear-algebra/Matrix";
import Regression from "./Regression";

/**
 * Binary classifier: squashes the linear combination `inputs · hypothesis` through a sigmoid to
 * produce a probability in (0, 1), then you threshold at 0.5. Its decision boundary is a straight
 * line/plane. See {@link Regression} for the shared gradient-descent training loop.
 */
export default class LogisticRegression extends Regression {

    protected predictFromEnrichedInputs (inputs: Matrix) {
        return Matrix.multiply(inputs, this.getHypothesis()).transform(element => 1 / (1 + Math.exp(-element)));
    }
}