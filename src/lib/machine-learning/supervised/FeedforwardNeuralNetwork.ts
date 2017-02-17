import Matrix from "../../math/linear-algebra/Matrix";

export default class FeedforwardNeuralNetwork {

    private weightMatrices: Matrix[] = [];

    private numberOfEpochs = 1000;
    private batchSize = 0;
    private learningRate = 0.001;

    // Use the sigmoid function as default activation function
    private activationFunction = (value: number) => 1.0 / (1.0 + Math.exp(-value));
    private activationGradientFunction = (value: number) => this.activationFunction(value) * (1 - this.activationFunction(value));

    public constructor (numberOfNodesPerLayer: number[], randomSeed?: number) {
        for (let weightLayer = 0; weightLayer < numberOfNodesPerLayer.length - 1; weightLayer++) {
            const incomingNodeCount = numberOfNodesPerLayer[weightLayer];
            const outgoingNodeCount = numberOfNodesPerLayer[weightLayer + 1];

            // Good epsilon "strategy" according to exercise 4 in week 5 of Stanford Machine Learning course on Coursera by Andrew Ng
            const epsilon = Math.sqrt(6) / Math.sqrt(incomingNodeCount + outgoingNodeCount);

            this.weightMatrices.push(Matrix.rand(incomingNodeCount + 1, outgoingNodeCount, epsilon, randomSeed));
        }
    }

    public train (inputs: Matrix, targets: Matrix) {
        const exampleCount = inputs.getRowCount();

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const batchSize = this.batchSize !== 0 ? this.batchSize : Number.POSITIVE_INFINITY;

            for (let batchStartIndex = 0; batchStartIndex < exampleCount; batchStartIndex += batchSize) {
                const batchEndIndex = Math.min(batchStartIndex + batchSize - 1, inputs.getRowCount() - 1);
                this.trainBatch(inputs.getRows(batchStartIndex, batchEndIndex), targets.getRows(batchStartIndex, batchEndIndex));
            }
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

    public setLearningRate (learningRate = 0.001) {
        this.learningRate = learningRate;
    }

    /**
     * The number of iterations over the full training set.
     *
     * @param numberOfEpochs
     */
    public setNumberOfEpochs (numberOfEpochs = 1000) {
        this.numberOfEpochs = numberOfEpochs;
    }

    /* Parameter getters */

    public getActivationFunction () {
        return this.activationFunction;
    }

    public getActivationGradientFunction () {
        return this.activationGradientFunction;
    }

    public getBatchSize () {
        return this.batchSize;
    }

    public getLearningRate () {
        return this.learningRate;
    }

    public getNumberOfEpochs () {
        return this.numberOfEpochs;
    }

    /* Private methods */

    private trainBatch (inputs: Matrix, targets: Matrix) {
        const [activations, incomingActivations] = this.forwardPropagate(inputs);
        this.backPropagate(activations, incomingActivations, targets);
    }

    private forwardPropagate (inputs: Matrix) {
        const activations = [inputs.getClone()];
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

    private backPropagate (activations: Matrix[], incomingActivations: Matrix[], targets: Matrix) {
        const errors: Matrix[] = [];

        // Calculate the errors in the output nodes by subtracting the expected desired activations from the actual activations
        errors[this.weightMatrices.length] = Matrix.subtract(activations[this.weightMatrices.length], targets);

        for (let weightLayerIndex = this.weightMatrices.length - 1; weightLayerIndex > 0; weightLayerIndex--) {
            // Transpose the weight matrix  of the incoming node layer and remove the weights from the bias nodes since we don't need to compute an error for the bias node.
            const weightMatrix = Matrix.transpose(this.weightMatrices[weightLayerIndex]).getColumns(1);

            // Backpropagate the errors from the outgoing node layer to the incoming node layer
            errors[weightLayerIndex] = Matrix.multiply(errors[weightLayerIndex + 1], weightMatrix).multiplyElementWise(Matrix.transform(incomingActivations[weightLayerIndex], value => this.activationGradientFunction(value)));
        }

        for (let weightLayer = 0; weightLayer < this.weightMatrices.length; weightLayer++) {
            // Compute the gradient of the previous weightLayer based on the computed errors
            const gradients = Matrix.transpose(activations[weightLayer]).multiply(errors[weightLayer + 1]).multiply(1 / targets.getRowCount());

            this.weightMatrices[weightLayer].subtract(gradients.multiply(this.learningRate));
        }
    }
}