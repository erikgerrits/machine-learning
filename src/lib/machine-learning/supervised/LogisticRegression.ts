import Matrix from "../../math/linear-algebra/Matrix";
import Regression from "./Regression";

export default class LogisticRegression extends Regression {

    protected predictFromEnrichedInputs (inputs: Matrix) {
        return Matrix.multiply(inputs, this.getHypothesis()).transform(element => 1 / (1 + Math.exp(-element)));
    }
}