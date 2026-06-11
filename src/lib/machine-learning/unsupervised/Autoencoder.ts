import Matrix from "../../math/linear-algebra/Matrix";

/**
 * An **autoencoder** — a neural network trained to copy its input to its output through a narrow
 * **bottleneck**, and the first model in Part 6. It's unsupervised: the target *is* the input, so it
 * needs no labels. The trick is the squeeze in the middle — an **encoder** compresses each input down
 * to a small **code** (far fewer numbers than the input), and a **decoder** rebuilds the input from
 * that code alone. To reconstruct well through such a tight channel, the network is forced to discover
 * the few factors that actually vary in the data and throw the rest away.
 *
 * That single idea pays off three ways:
 * - **Compression / representation** — the code is a learned, low-dimensional summary (a non-linear
 *   cousin of {@link PCA}, since the layers are non-linear).
 * - **Denoising** — noise doesn't fit through the bottleneck, so feeding a corrupted input yields a
 *   *cleaned* reconstruction, snapped back onto the manifold of normal data.
 * - **Anomaly detection** — train on normal data and anything unlike it reconstructs poorly, so a high
 *   {@link reconstructionError} flags the odd one out.
 *
 * The architecture is symmetric: hidden layers step the input down to the code and a mirror-image
 * decoder steps back up (e.g. inputSize → 16 → 2 → 16 → inputSize). Sigmoid activations throughout, so
 * inputs should be scaled to `[0, 1]`; training minimises mean-squared reconstruction error by
 * gradient descent. The backprop is finite-difference checked by {@link checkGradients}.
 *
 * @example
 * const ae = new Autoencoder().setHiddenSizes([16]).setCodeSize(2).setNumberOfEpochs(300);
 * ae.train(images);                 // images: rows of pixel intensities in [0, 1]
 * const codes = ae.encode(images);  // each row compressed to 2 numbers
 * ae.reconstruct(noisyImages);      // noise squeezed out by the bottleneck
 */
export default class Autoencoder {

    private inputSize = 0;          // inferred from the data on first train if left at 0
    private hiddenSizes = [16];      // encoder hidden layers (decoder mirrors them)
    private codeSize = 2;
    private learningRate = 0.5;
    private numberOfEpochs = 300;
    private seed = 0;

    private weightMatrices: Matrix[] = [];
    private encoderWeightCount = 0; // how many weight matrices belong to the encoder

    public constructor () {}

    /** Train to reconstruct `inputs` (rows of features scaled to [0, 1]) through the bottleneck. */
    public train (inputs: Matrix) {
        this.ensureBuilt(inputs.getColumnCount());
        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const gradients = this.computeGradients(inputs);
            for (let l = 0; l < this.weightMatrices.length; l++) {
                this.weightMatrices[l] = Matrix.subtract(this.weightMatrices[l], Matrix.multiply(gradients[l], this.learningRate));
            }
        }
        return this;
    }

    /** Compress each input row to its low-dimensional code. */
    public encode (inputs: Matrix) {
        this.ensureBuilt(inputs.getColumnCount());
        let activation = inputs;
        for (let l = 0; l < this.encoderWeightCount; l++) {
            activation = this.layerForward(activation, l);
        }
        return activation;
    }

    /** Rebuild inputs from codes (rows of length codeSize). */
    public decode (codes: Matrix) {
        let activation = codes;
        for (let l = this.encoderWeightCount; l < this.weightMatrices.length; l++) {
            activation = this.layerForward(activation, l);
        }
        return activation;
    }

    /** Encode then decode — the network's reconstruction of its input. */
    public reconstruct (inputs: Matrix) {
        this.ensureBuilt(inputs.getColumnCount());
        return this.forward(inputs).activations[this.weightMatrices.length];
    }

    /** Per-row mean-squared reconstruction error — small for typical inputs, large for anomalies. */
    public reconstructionError (inputs: Matrix): number[] {
        const reconstructed = this.reconstruct(inputs);
        const original = inputs.toArray();
        const output = reconstructed.toArray();
        return original.map((row, i) => {
            let sum = 0;
            for (let j = 0; j < row.length; j++) sum += (output[i][j] - row[j]) ** 2;
            return sum / row.length;
        });
    }

    /** The training objective: ½·mean-over-rows of the summed squared reconstruction error. */
    public computeLoss (inputs: Matrix): number {
        return this.cost(this.reconstruct(inputs), inputs);
    }

    /**
     * Verifies the analytic gradients against numerical (finite-difference) ones on a small random
     * problem. Returns true if every weight's gradient matches to a tight tolerance — the safety net
     * proving the backprop is correct.
     */
    public checkGradients (): boolean {
        const probe = new Autoencoder().setHiddenSizes([5]).setCodeSize(2).setSeed(1);
        const inputs = Matrix.rand(4, 6, 0.5, 7).transform(v => v + 0.5); // 4 rows of 6 features in (0, 1)
        probe.ensureBuilt(6);

        const analytic = probe.computeGradients(inputs);
        const epsilon = 1e-4;
        let maxError = 0;

        for (let l = 0; l < probe.weightMatrices.length; l++) {
            const rows = probe.weightMatrices[l].getRowCount();
            const cols = probe.weightMatrices[l].getColumnCount();
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const original = probe.weightMatrices[l].getElement(r, c);
                    probe.weightMatrices[l].setElement(r, c, original + epsilon);
                    const lossPlus = probe.cost(probe.reconstruct(inputs), inputs);
                    probe.weightMatrices[l].setElement(r, c, original - epsilon);
                    const lossMinus = probe.cost(probe.reconstruct(inputs), inputs);
                    probe.weightMatrices[l].setElement(r, c, original);

                    const numeric = (lossPlus - lossMinus) / (2 * epsilon);
                    const analyticValue = analytic[l].getElement(r, c);
                    const denominator = Math.max(1e-8, Math.abs(numeric) + Math.abs(analyticValue));
                    maxError = Math.max(maxError, Math.abs(numeric - analyticValue) / denominator);
                }
            }
        }
        return maxError < 1e-4;
    }

    /* Parameter setters */

    public setInputSize (inputSize: number) { this.inputSize = inputSize; return this; }
    public setHiddenSizes (hiddenSizes: number[]) { this.hiddenSizes = hiddenSizes; return this; }
    public setCodeSize (codeSize: number) { this.codeSize = codeSize; return this; }
    public setLearningRate (learningRate: number) { this.learningRate = learningRate; return this; }
    public setNumberOfEpochs (numberOfEpochs: number) { this.numberOfEpochs = numberOfEpochs; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getInputSize () { return this.inputSize; }
    public getHiddenSizes () { return this.hiddenSizes.slice(); }
    public getCodeSize () { return this.codeSize; }
    public getLearningRate () { return this.learningRate; }
    public getNumberOfEpochs () { return this.numberOfEpochs; }
    public getSeed () { return this.seed; }
    public getWeightMatrices () { return this.weightMatrices.map(w => w.getClone()); }

    /* Private methods */

    private ensureBuilt (inputSize: number) {
        if (this.inputSize !== inputSize) this.inputSize = inputSize;
        const encoderSizes = [this.inputSize, ...this.hiddenSizes, this.codeSize];
        const layerSizes = [...encoderSizes, ...this.hiddenSizes.slice().reverse(), this.inputSize];
        this.encoderWeightCount = this.hiddenSizes.length + 1;

        const matches = this.weightMatrices.length === layerSizes.length - 1
            && this.weightMatrices.every((w, l) => w.getRowCount() === layerSizes[l] + 1 && w.getColumnCount() === layerSizes[l + 1]);
        if (matches) return;

        this.weightMatrices = [];
        for (let l = 0; l < layerSizes.length - 1; l++) {
            const epsilon = Math.sqrt(6) / Math.sqrt(layerSizes[l] + layerSizes[l + 1]); // Xavier-style range
            this.weightMatrices.push(Matrix.rand(layerSizes[l] + 1, layerSizes[l + 1], epsilon, this.seed + l));
        }
    }

    /** Forward one layer: prepend a bias column, multiply by the layer weights, apply sigmoid. */
    private layerForward (activation: Matrix, layer: number) {
        const withBias = Matrix.appendLeft(activation, Matrix.ones(activation.getRowCount(), 1));
        return Matrix.transform(Matrix.multiply(withBias, this.weightMatrices[layer]), sigmoid);
    }

    /** Full forward pass, keeping every layer's activation (with bias) and pre-activation. */
    private forward (inputs: Matrix) {
        const activations: Matrix[] = [inputs];
        const preActivations: Matrix[] = [inputs];
        for (let l = 0; l < this.weightMatrices.length; l++) {
            const withBias = Matrix.appendLeft(activations[l], Matrix.ones(activations[l].getRowCount(), 1));
            const z = Matrix.multiply(withBias, this.weightMatrices[l]);
            preActivations.push(z);
            activations.push(Matrix.transform(z, sigmoid));
        }
        return { activations, preActivations };
    }

    private computeGradients (inputs: Matrix): Matrix[] {
        const { activations, preActivations } = this.forward(inputs);
        const last = this.weightMatrices.length;
        const n = inputs.getRowCount();
        const gradients: Matrix[] = new Array(this.weightMatrices.length);

        // Output error for MSE: (output − input) ⊙ sigmoid'(z).
        let delta = Matrix.subtract(activations[last], inputs).multiplyElementWise(Matrix.transform(preActivations[last], sigmoidPrime));

        for (let l = last - 1; l >= 0; l--) {
            const withBias = Matrix.appendLeft(activations[l], Matrix.ones(activations[l].getRowCount(), 1));
            gradients[l] = Matrix.multiply(Matrix.transpose(withBias), delta).multiply(1 / n);

            if (l > 0) {
                const weightsNoBias = this.weightMatrices[l].getRows(1); // drop the bias row
                delta = Matrix.multiply(delta, Matrix.transpose(weightsNoBias))
                    .multiplyElementWise(Matrix.transform(preActivations[l], sigmoidPrime));
            }
        }
        return gradients;
    }

    /** ½ · mean-over-rows of the summed squared error between reconstruction and target. */
    private cost (reconstructed: Matrix, target: Matrix): number {
        const a = reconstructed.toArray();
        const t = target.toArray();
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < a[i].length; j++) sum += (a[i][j] - t[i][j]) ** 2;
        }
        return sum / (2 * a.length);
    }
}

function sigmoid (value: number) {
    return 1 / (1 + Math.exp(-value));
}

function sigmoidPrime (value: number) {
    const s = sigmoid(value);
    return s * (1 - s);
}
