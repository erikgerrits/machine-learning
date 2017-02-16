import Matrix from "../../math/linear-algebra/Matrix";

export default class FeedforwardNeuralNetwork {

    private weightMatrices: Matrix[] = [];

    private learningRate = 0.001;
    private maximumIterations = 1000;

    // Use the sigmoid function as default activation function
    private activationFunction = (value: number) => 1.0 / (1.0 + Math.exp(-value));
    private activationGradientFunction = (value: number) => this.activationFunction(value) * (1 - this.activationFunction(value));

    public constructor (private inputs: Matrix, private outputs: Matrix, numberOfNodesPerHiddenLayer: number[], randomSeed?: number) {
        const nodeCounts = numberOfNodesPerHiddenLayer.slice();
        nodeCounts.unshift(inputs.getColumnCount());
        nodeCounts.push(outputs.getColumnCount());

        for (let weightLayer = 0; weightLayer < nodeCounts.length - 1; weightLayer++) {
            const incomingNodeCount = nodeCounts[weightLayer];
            const outgoingNodeCount = nodeCounts[weightLayer + 1];

            // Good epsilon "strategy" according to excercise 4 in week 5 of Stanford Machine Learning course on Coursera by Andrew Ng
            const epsilon = Math.sqrt(6) / Math.sqrt(incomingNodeCount + outgoingNodeCount);

            this.weightMatrices.push(Matrix.rand(incomingNodeCount + 1, outgoingNodeCount, epsilon, randomSeed));
        }
    }

    public train () {
        for (let iteration = 0; iteration < this.maximumIterations; iteration++) {
            const [activations, incomingActivations] = this.forwardPropagate(this.inputs);
            this.backPropagate(activations, incomingActivations);
        }
    }

    public predict (inputs: Matrix) {
        const [activations] = this.forwardPropagate(inputs);

        return activations[this.weightMatrices.length];
    }

    /* Parameter setters */

    public setActivationFunction (activationFunction: (value: number) => number) {
        this.activationFunction = activationFunction;
    }

    public setActivationGradientFunction (activationGradientFunction: (value: number) => number) {
        this.activationGradientFunction = activationGradientFunction;
    }

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setMaximumIterations (maximumIterations: number) {
        this.maximumIterations = maximumIterations;
        return this;
    }

    /* Parameter getters */

    public getActivationFunction () {
        return this.activationFunction;
    }

    public getActivationGradientFunction () {
        return this.activationGradientFunction;
    }

    public getLearningRate () {
        return this.learningRate;
    }

    public getMaximumIterations () {
        return this.maximumIterations;
    }

    /* Private methods */

    private forwardPropagate (inputs: Matrix) {
        const activations = [this.inputs.getClone()];
        const incomingActivations: Matrix[] = [];

        for (let weightLayer = 0; weightLayer < this.weightMatrices.length; weightLayer++) {
            // Add bias nodes to the activations
            activations[weightLayer].appendLeft(Matrix.ones(activations[weightLayer].getRowCount(), 1));

            // Multiply the activations with the weights towards the next layer to compute the input to each node in the next layer
            incomingActivations[weightLayer + 1] = Matrix.multiply(activations[weightLayer], this.weightMatrices[weightLayer]);

            // Apply the sigmoid function to activations to get the output signal of the nodes in the next layer
            activations[weightLayer + 1] = Matrix.transform(incomingActivations[weightLayer + 1], element => this.activationFunction(element));
        }

        return [activations, incomingActivations];
    }

    private backPropagate (activations: Matrix[], incomingActivations: Matrix[]) {
        const errors: Matrix[] = [];

        // Calculate the errors in the output nodes by subtracting the expected desired activations from the actual activations
        errors[this.weightMatrices.length] = Matrix.subtract(activations[this.weightMatrices.length], this.outputs);

        for (let weightLayerIndex = this.weightMatrices.length - 1; weightLayerIndex > 0; weightLayerIndex--) {
            // Transpose the weight matrix  of the incoming node layer and remove the weights from the bias nodes since we don't need to compute an error for the bias node.
            const weightMatrix = Matrix.transpose(this.weightMatrices[weightLayerIndex]).getColumns(1);

            // Backpropagate the errors from the outgoing node layer to the incoming node layer
            errors[weightLayerIndex] = Matrix.multiply(errors[weightLayerIndex + 1], weightMatrix).multiplyElementWise(Matrix.transform(incomingActivations[weightLayerIndex], value => this.activationGradientFunction(value)));
        }

        for (let weightLayer = 0; weightLayer < this.weightMatrices.length; weightLayer++) {
            // Compute the gradient of the previous weightLayer based on the computed errors
            const gradients = Matrix.transpose(activations[weightLayer]).multiply(errors[weightLayer + 1]).multiply(1 / this.inputs.getRowCount());

            this.weightMatrices[weightLayer].subtract(gradients.multiply(this.learningRate));
        }
    }
}