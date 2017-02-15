import Matrix from "../../math/linear-algebra/Matrix";
import Regression from "./Regression";

export default class LinearRegression extends Regression {

    protected predictFromEnrichedInputs (inputs: Matrix) {
        return Matrix.multiply(inputs, this.getHypothesis());
    }
}