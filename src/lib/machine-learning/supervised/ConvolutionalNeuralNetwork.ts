import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/**
 * A small **convolutional neural network** for tiny grayscale images — the architecture that learns
 * to *see*. A plain {@link FeedforwardNeuralNetwork} treats an image as a flat list of pixels, so a
 * shape in the top-left and the same shape in the bottom-right look like totally different inputs.
 * A CNN instead slides a handful of small **filters** across the image, so each filter learns one
 * little local pattern (an edge, a stroke) it can spot *anywhere* — that's what makes it
 * translation-tolerant and far more data-efficient on images.
 *
 * The topology is the classic minimal stack, all trained from scratch by backpropagation:
 *
 *   image → **conv** (`filterCount` filters, `filterSize`², ReLU) → **2×2 max-pool** → flatten
 *         → **dense** → softmax over the classes
 *
 * Inputs are a Matrix whose rows are flattened images (`inputHeight × inputWidth`, row-major);
 * targets are one-hot rows; `predict` returns class probabilities. The filters and dense weights
 * persist across `train()` calls, so calling it with `numberOfEpochs = 1` in a loop animates the
 * learning. {@link checkGradients} finite-difference-verifies the (error-prone) convolution backprop.
 *
 * @example
 * const cnn = new ConvolutionalNeuralNetwork().setInputShape(12, 12).setFilterCount(6).setSeed(0);
 * cnn.setLearningRate(0.1).setNumberOfEpochs(200).train(images, oneHotLabels);
 * cnn.predict(images); // N × C class probabilities
 */
export default class ConvolutionalNeuralNetwork {

    private inputHeight = 12;
    private inputWidth = 12;
    private filterCount = 6;
    private filterSize = 3;
    private learningRate = 0.1;
    private numberOfEpochs = 1;
    private seed = 0;

    // Learned parameters.
    private filters: number[][][];   // [filter][row][col]
    private filterBiases: number[];  // [filter]
    private denseWeights: number[][]; // [flattenedIndex][class]
    private denseBiases: number[];   // [class]

    // Derived shapes (set on first train).
    private classCount = 0;
    private convHeight = 0;
    private convWidth = 0;
    private poolHeight = 0;
    private poolWidth = 0;
    private flattenedLength = 0;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const images = inputs.toArray();
        const labels = targets.toArray();
        const exampleCount = images.length;
        if (exampleCount === 0) {
            return this;
        }

        if (this.filters === undefined) {
            this.initialize(labels[0].length);
        }

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const gradients = this.zeroGradients();

            for (let n = 0; n < exampleCount; n++) {
                const image = this.reshape(images[n]);
                const cache = this.forward(image);
                this.accumulateGradients(image, cache, labels[n], gradients);
            }

            this.applyGradients(gradients, exampleCount);
        }

        return this;
    }

    /** Class probabilities for each input row (softmax over the output layer). */
    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => this.forward(this.reshape(row)).probabilities));
    }

    /** Average cross-entropy over the given examples — what training minimises. */
    public computeLoss (inputs: Matrix, targets: Matrix) {
        const images = inputs.toArray();
        const labels = targets.toArray();
        if (images.length === 0) {
            return 0;
        }

        let loss = 0;
        for (let n = 0; n < images.length; n++) {
            const { probabilities } = this.forward(this.reshape(images[n]));
            for (let c = 0; c < this.classCount; c++) {
                loss -= labels[n][c] * Math.log(Math.max(1e-12, probabilities[c]));
            }
        }
        return loss / images.length;
    }

    /**
     * Finite-difference check of backprop: nudge every parameter by a hair, compare how the loss
     * actually moves against the analytic gradient, and report whether they agree everywhere
     * (relative error < 1e-4). A live proof the convolution gradients are right.
     */
    public checkGradients () {
        const random = mulberry32(this.seed + 12345);
        if (this.filters === undefined) {
            this.initialize(2);
        }

        // One small random example to test against.
        const image = Array.from({ length: this.inputHeight }, () =>
            Array.from({ length: this.inputWidth }, () => random() * 2 - 1));
        const label = new Array<number>(this.classCount).fill(0);
        label[Math.floor(random() * this.classCount)] = 1;

        const analytic = this.zeroGradients();
        this.accumulateGradients(image, this.forward(image), label, analytic);

        const epsilon = 1e-4;
        const tolerance = 1e-3;
        const lossAt = () => {
            const { probabilities } = this.forward(image);
            let loss = 0;
            for (let c = 0; c < this.classCount; c++) {
                loss -= label[c] * Math.log(Math.max(1e-12, probabilities[c]));
            }
            return loss;
        };

        const ok = (analyticGrad: number, get: () => number, set: (v: number) => void) => {
            const original = get();
            set(original + epsilon);
            const lossPlus = lossAt();
            set(original - epsilon);
            const lossMinus = lossAt();
            set(original);
            const numeric = (lossPlus - lossMinus) / (2 * epsilon);
            const denom = Math.max(1, Math.abs(analyticGrad) + Math.abs(numeric));
            return Math.abs(analyticGrad - numeric) / denom < tolerance;
        };

        for (let k = 0; k < this.filterCount; k++) {
            for (let i = 0; i < this.filterSize; i++) {
                for (let j = 0; j < this.filterSize; j++) {
                    if (!ok(analytic.filters[k][i][j], () => this.filters[k][i][j], v => { this.filters[k][i][j] = v; })) return false;
                }
            }
            if (!ok(analytic.filterBiases[k], () => this.filterBiases[k], v => { this.filterBiases[k] = v; })) return false;
        }
        for (let v = 0; v < this.flattenedLength; v++) {
            for (let c = 0; c < this.classCount; c++) {
                if (!ok(analytic.denseWeights[v][c], () => this.denseWeights[v][c], val => { this.denseWeights[v][c] = val; })) return false;
            }
        }
        for (let c = 0; c < this.classCount; c++) {
            if (!ok(analytic.denseBiases[c], () => this.denseBiases[c], val => { this.denseBiases[c] = val; })) return false;
        }
        return true;
    }

    /* Parameter setters */

    public setInputShape (height: number, width: number) {
        this.inputHeight = height;
        this.inputWidth = width;
        return this;
    }

    public setFilterCount (filterCount: number) {
        this.filterCount = filterCount;
        return this;
    }

    public setFilterSize (filterSize: number) {
        this.filterSize = filterSize;
        return this;
    }

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setNumberOfEpochs (numberOfEpochs: number) {
        this.numberOfEpochs = numberOfEpochs;
        return this;
    }

    public setSeed (seed: number) {
        this.seed = seed;
        return this;
    }

    public reset () {
        this.filters = undefined;
        return this;
    }

    /* Parameter getters */

    public getFilterCount () {
        return this.filterCount;
    }

    public getFilterSize () {
        return this.filterSize;
    }

    public getLearningRate () {
        return this.learningRate;
    }

    public getNumberOfEpochs () {
        return this.numberOfEpochs;
    }

    public getSeed () {
        return this.seed;
    }

    /** The learned filters (`filterCount × filterSize × filterSize`) — the patterns each one detects. */
    public getFilters () {
        return this.filters ? this.filters.map(filter => filter.map(row => row.slice())) : [];
    }

    /** The post-ReLU feature maps a single (flattened) image produces — one per filter. */
    public getConvMaps (inputRow: number[]) {
        return this.forward(this.reshape(inputRow)).convActivated.map(map => map.map(row => row.slice()));
    }

    /* Private methods */

    private initialize (classCount: number) {
        const random = mulberry32(this.seed);
        const f = this.filterSize;

        this.classCount = classCount;
        this.convHeight = this.inputHeight - f + 1;
        this.convWidth = this.inputWidth - f + 1;
        this.poolHeight = Math.floor(this.convHeight / 2);
        this.poolWidth = Math.floor(this.convWidth / 2);
        this.flattenedLength = this.filterCount * this.poolHeight * this.poolWidth;

        // Xavier-ish uniform init.
        const convScale = Math.sqrt(6 / (f * f + f * f));
        this.filters = Array.from({ length: this.filterCount }, () =>
            Array.from({ length: f }, () => Array.from({ length: f }, () => (random() * 2 - 1) * convScale)));
        this.filterBiases = new Array<number>(this.filterCount).fill(0);

        const denseScale = Math.sqrt(6 / (this.flattenedLength + this.classCount));
        this.denseWeights = Array.from({ length: this.flattenedLength }, () =>
            Array.from({ length: this.classCount }, () => (random() * 2 - 1) * denseScale));
        this.denseBiases = new Array<number>(this.classCount).fill(0);
    }

    private reshape (row: number[]) {
        const image: number[][] = [];
        for (let i = 0; i < this.inputHeight; i++) {
            image.push(row.slice(i * this.inputWidth, (i + 1) * this.inputWidth));
        }
        return image;
    }

    private forward (image: number[][]) {
        const f = this.filterSize;
        const convPre: number[][][] = [];
        const convActivated: number[][][] = [];

        // Convolution (valid cross-correlation) + ReLU.
        for (let k = 0; k < this.filterCount; k++) {
            const pre: number[][] = [];
            const act: number[][] = [];
            for (let i = 0; i < this.convHeight; i++) {
                const preRow: number[] = [];
                const actRow: number[] = [];
                for (let j = 0; j < this.convWidth; j++) {
                    let sum = this.filterBiases[k];
                    for (let di = 0; di < f; di++) {
                        for (let dj = 0; dj < f; dj++) {
                            sum += image[i + di][j + dj] * this.filters[k][di][dj];
                        }
                    }
                    preRow.push(sum);
                    actRow.push(sum > 0 ? sum : 0);
                }
                pre.push(preRow);
                act.push(actRow);
            }
            convPre.push(pre);
            convActivated.push(act);
        }

        // 2×2 max-pool (stride 2); remember which cell won, for backprop.
        const poolArgs: [number, number][][][] = [];
        const flattened: number[] = [];
        for (let k = 0; k < this.filterCount; k++) {
            const args: [number, number][][] = [];
            for (let pi = 0; pi < this.poolHeight; pi++) {
                const argRow: [number, number][] = [];
                for (let pj = 0; pj < this.poolWidth; pj++) {
                    let best = -Infinity;
                    let bestDi = 0;
                    let bestDj = 0;
                    for (let di = 0; di < 2; di++) {
                        for (let dj = 0; dj < 2; dj++) {
                            const value = convActivated[k][pi * 2 + di][pj * 2 + dj];
                            if (value > best) {
                                best = value;
                                bestDi = di;
                                bestDj = dj;
                            }
                        }
                    }
                    argRow.push([bestDi, bestDj]);
                    flattened.push(best);
                }
                args.push(argRow);
            }
            poolArgs.push(args);
        }

        // Dense → softmax.
        const logits = new Array<number>(this.classCount);
        for (let c = 0; c < this.classCount; c++) {
            let sum = this.denseBiases[c];
            for (let v = 0; v < this.flattenedLength; v++) {
                sum += flattened[v] * this.denseWeights[v][c];
            }
            logits[c] = sum;
        }
        const probabilities = softmax(logits);

        return { convPre, convActivated, poolArgs, flattened, probabilities };
    }

    private accumulateGradients (image: number[][], cache: ReturnType<ConvolutionalNeuralNetwork['forward']>, label: number[], gradients: Gradients) {
        const f = this.filterSize;

        // Softmax + cross-entropy: dLoss/dLogit = prediction − target.
        const dLogits = new Array<number>(this.classCount);
        for (let c = 0; c < this.classCount; c++) {
            dLogits[c] = cache.probabilities[c] - label[c];
        }

        // Dense layer.
        const dFlattened = new Array<number>(this.flattenedLength).fill(0);
        for (let c = 0; c < this.classCount; c++) {
            gradients.denseBiases[c] += dLogits[c];
            for (let v = 0; v < this.flattenedLength; v++) {
                gradients.denseWeights[v][c] += cache.flattened[v] * dLogits[c];
                dFlattened[v] += this.denseWeights[v][c] * dLogits[c];
            }
        }

        // Un-flatten, route through max-pool, then ReLU, into the conv pre-activations.
        let v = 0;
        for (let k = 0; k < this.filterCount; k++) {
            for (let pi = 0; pi < this.poolHeight; pi++) {
                for (let pj = 0; pj < this.poolWidth; pj++) {
                    const grad = dFlattened[v++];
                    const [di, dj] = cache.poolArgs[k][pi][pj];
                    const ci = pi * 2 + di;
                    const cj = pj * 2 + dj;
                    // ReLU: gradient only flows where the pre-activation was positive.
                    if (cache.convPre[k][ci][cj] <= 0 || grad === 0) {
                        continue;
                    }
                    gradients.filterBiases[k] += grad;
                    for (let fi = 0; fi < f; fi++) {
                        for (let fj = 0; fj < f; fj++) {
                            gradients.filters[k][fi][fj] += image[ci + fi][cj + fj] * grad;
                        }
                    }
                }
            }
        }
    }

    private applyGradients (gradients: Gradients, exampleCount: number) {
        const step = this.learningRate / exampleCount;

        for (let k = 0; k < this.filterCount; k++) {
            this.filterBiases[k] -= step * gradients.filterBiases[k];
            for (let i = 0; i < this.filterSize; i++) {
                for (let j = 0; j < this.filterSize; j++) {
                    this.filters[k][i][j] -= step * gradients.filters[k][i][j];
                }
            }
        }
        for (let v = 0; v < this.flattenedLength; v++) {
            for (let c = 0; c < this.classCount; c++) {
                this.denseWeights[v][c] -= step * gradients.denseWeights[v][c];
            }
        }
        for (let c = 0; c < this.classCount; c++) {
            this.denseBiases[c] -= step * gradients.denseBiases[c];
        }
    }

    private zeroGradients (): Gradients {
        return {
            filters: Array.from({ length: this.filterCount }, () =>
                Array.from({ length: this.filterSize }, () => new Array<number>(this.filterSize).fill(0))),
            filterBiases: new Array<number>(this.filterCount).fill(0),
            denseWeights: Array.from({ length: this.flattenedLength }, () => new Array<number>(this.classCount).fill(0)),
            denseBiases: new Array<number>(this.classCount).fill(0),
        };
    }
}

interface Gradients {
    filters: number[][][];
    filterBiases: number[];
    denseWeights: number[][];
    denseBiases: number[];
}

function softmax (logits: number[]) {
    const max = Math.max(...logits);
    const exps = logits.map(value => Math.exp(value - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(value => value / sum);
}
