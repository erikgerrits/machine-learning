import Matrix from "../../math/linear-algebra/Matrix";

export default class FeedForwardNeuralNetwork {

    private weigthMatrices: Matrix[] = [];

    public constructor (nodeCounts: number[]) {
        for (let weightLayer = 0; weightLayer < nodeCounts.length - 1; weightLayer++) {
            const incomingNodeCount = nodeCounts[weightLayer];
            const outgoingNodeCount = nodeCounts[weightLayer + 1];

            // Good epsilon "strategy" according to excercise 4 in week 5 of Stanford Machine Learning course on Coursera by Andrew Ng
            const epsilon = Math.sqrt(6) / Math.sqrt(incomingNodeCount + outgoingNodeCount);

            this.weigthMatrices.push(Matrix.rand(incomingNodeCount + 1, outgoingNodeCount, epsilon, 0));
        }
    }

    public execute (X: Matrix, y?: Matrix, iterations = 1, learningRate = 1) {
        let activations: Matrix[];

        for (let iteration = 0; iteration < iterations; iteration++) {
            activations = [X.getClone()];
            const incomingActivations: Matrix[] = [];

            for (let weightLayer = 0; weightLayer < this.weigthMatrices.length; weightLayer++) {
                // Add bias nodes to the activations
                activations[weightLayer].appendLeft(Matrix.ones(activations[weightLayer].getRowCount(), 1));

                // Multiply the activations with the weights towards the next layer to compute the input to each node in the next layer
                incomingActivations[weightLayer + 1] = Matrix.multiply(activations[weightLayer], this.weigthMatrices[weightLayer]);

                // Apply the sigmoid function to activations to get the output signal of the nodes in the next layer
                activations[weightLayer + 1] = Matrix.transform(incomingActivations[weightLayer + 1], (element) => 1 / (1 + Math.exp(-element)));
            }

            if (y !== undefined) {
                const errors: Matrix[] = [];
                const gradients: Matrix[] = [];

                // Calculate the errors in the output nodes by subracting the expected desired activations from the actual activations
                errors[this.weigthMatrices.length] = Matrix.subtract(activations[this.weigthMatrices.length], y);

                for (let weightLayerIndex = this.weigthMatrices.length - 1; weightLayerIndex > 0; weightLayerIndex--) {
                    // Transpose the weight matrix  of the incoming node layer and remove the weights from the bias nodes since we don't need to compute an error for the bias node.
                    const weightMatrix = Matrix.transpose(this.weigthMatrices[weightLayerIndex]).getColumns(1);

                    // Backpropagate the errors from the outgoing node layer to the incoming node layer
                    errors[weightLayerIndex] = Matrix.multiply(errors[weightLayerIndex + 1], weightMatrix).multiplyElementWise(Matrix.transform(incomingActivations[weightLayerIndex], (value) => this.sigmoidGradient(value)));
                }

                for (let weightLayer = 0; weightLayer < this.weigthMatrices.length; weightLayer++) {
                    // Compute the gradient of the previous weightLayer based on the computed errors
                    gradients[weightLayer] = Matrix.transpose(activations[weightLayer]).multiply(errors[weightLayer + 1]).multiply(1 / X.getRowCount());

                    this.weigthMatrices[weightLayer].subtract(gradients[weightLayer].multiply(learningRate));
                }
            }
        }

        return activations[this.weigthMatrices.length];
    }

    private sigmoid (value: number) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    private sigmoidGradient (value: number) {
        return this.sigmoid(value) * (1 - this.sigmoid(value));
    }
}