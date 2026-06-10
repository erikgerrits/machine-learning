import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/**
 * A small **recurrent neural network** (Elman RNN) with a learned **embedding** layer — the
 * architecture that learns to *read*. A feedforward net (or a CNN) takes a fixed-size input all at
 * once and has no notion of order; a sequence — the words of a review, a run of visits — arrives one
 * step at a time and the meaning lives in the order. An RNN walks the sequence carrying a **hidden
 * state** that it updates at each step, so what it read earlier colours how it reads what comes next.
 *
 * The pipeline, trained end-to-end by **backpropagation through time**:
 *
 *   token ids → **embedding** (each token → a learned vector) → **recurrent step**
 *             `hₜ = tanh(Wxh·xₜ + Whh·hₜ₋₁ + b)` → **dense** on the final state → softmax over classes
 *
 * The embedding is the quiet star: training nudges tokens that play similar roles toward similar
 * vectors, so "similar sits close" falls out for free (the same trick recommenders use for items).
 *
 * Inputs are a Matrix whose rows are token-id sequences (integers; **0 is reserved for padding** and
 * skipped), right-padded to a common length; targets are one-hot rows; `predict` returns class
 * probabilities. Weights persist across `train()` calls (animate by looping with `numberOfEpochs = 1`).
 * {@link checkGradients} finite-difference-verifies the through-time gradients.
 *
 * @example
 * const rnn = new RecurrentNeuralNetwork().setVocabSize(20).setEmbeddingDim(2).setHiddenSize(8).setSeed(0);
 * rnn.setLearningRate(0.1).setNumberOfEpochs(300).train(reviews, sentimentOneHot);
 * rnn.predict(reviews);     // N × C probabilities
 * rnn.getEmbeddings();      // V × D — the learned word vectors
 */
export default class RecurrentNeuralNetwork {

    private vocabSize = 16;
    private embeddingDim = 8;
    private hiddenSize = 12;
    private learningRate = 0.1;
    private numberOfEpochs = 1;
    private seed = 0;

    // Learned parameters.
    private embeddings: number[][]; // [token][dim]
    private weightsInputHidden: number[][];  // [dim][hidden]   (Wxh)
    private weightsHiddenHidden: number[][]; // [hidden][hidden] (Whh)
    private hiddenBias: number[];            // [hidden]
    private weightsHiddenOutput: number[][]; // [hidden][class]  (Why)
    private outputBias: number[];            // [class]
    private classCount = 0;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const sequences = inputs.toArray();
        const labels = targets.toArray();
        if (sequences.length === 0) {
            return this;
        }

        if (this.embeddings === undefined) {
            this.initialize(labels[0].length);
        }

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const gradients = this.zeroGradients();
            for (let n = 0; n < sequences.length; n++) {
                const tokens = toTokens(sequences[n]);
                const cache = this.forward(tokens);
                this.accumulateGradients(cache, labels[n], gradients);
            }
            this.applyGradients(gradients, sequences.length);
        }

        return this;
    }

    /** Class probabilities for each input sequence (softmax over the output of the final state). */
    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => this.forward(toTokens(row)).probabilities));
    }

    /** Average cross-entropy over the given sequences. */
    public computeLoss (inputs: Matrix, targets: Matrix) {
        const sequences = inputs.toArray();
        const labels = targets.toArray();
        if (sequences.length === 0) {
            return 0;
        }
        let loss = 0;
        for (let n = 0; n < sequences.length; n++) {
            const { probabilities } = this.forward(toTokens(sequences[n]));
            for (let c = 0; c < this.classCount; c++) {
                loss -= labels[n][c] * Math.log(Math.max(1e-12, probabilities[c]));
            }
        }
        return loss / sequences.length;
    }

    /**
     * Finite-difference check of backprop-through-time: nudge every parameter, compare the actual
     * loss change to the analytic gradient, report whether they agree everywhere (relative error
     * < 1e-3). A live proof the through-time gradients are right — easy to get subtly wrong.
     */
    public checkGradients () {
        const random = mulberry32(this.seed + 7919);
        if (this.embeddings === undefined) {
            this.initialize(2);
        }

        const length = 4;
        const tokens: number[] = [];
        for (let i = 0; i < length; i++) {
            tokens.push(1 + Math.floor(random() * (this.vocabSize - 1))); // avoid the padding token 0
        }
        const label = new Array<number>(this.classCount).fill(0);
        label[Math.floor(random() * this.classCount)] = 1;

        const analytic = this.zeroGradients();
        this.accumulateGradients(this.forward(tokens), label, analytic);

        const epsilon = 1e-4;
        const tolerance = 1e-3;
        const lossAt = () => {
            const { probabilities } = this.forward(tokens);
            let loss = 0;
            for (let c = 0; c < this.classCount; c++) {
                loss -= label[c] * Math.log(Math.max(1e-12, probabilities[c]));
            }
            return loss;
        };
        const ok = (analyticGrad: number, get: () => number, set: (v: number) => void) => {
            const original = get();
            set(original + epsilon);
            const plus = lossAt();
            set(original - epsilon);
            const minus = lossAt();
            set(original);
            const numeric = (plus - minus) / (2 * epsilon);
            return Math.abs(analyticGrad - numeric) / Math.max(1, Math.abs(analyticGrad) + Math.abs(numeric)) < tolerance;
        };

        // Only the embeddings actually used by the test sequence get a gradient.
        for (const t of new Set(tokens)) {
            for (let d = 0; d < this.embeddingDim; d++) {
                if (!ok(analytic.embeddings[t][d], () => this.embeddings[t][d], v => { this.embeddings[t][d] = v; })) return false;
            }
        }
        for (let d = 0; d < this.embeddingDim; d++) {
            for (let h = 0; h < this.hiddenSize; h++) {
                if (!ok(analytic.weightsInputHidden[d][h], () => this.weightsInputHidden[d][h], v => { this.weightsInputHidden[d][h] = v; })) return false;
            }
        }
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let h = 0; h < this.hiddenSize; h++) {
                if (!ok(analytic.weightsHiddenHidden[i][h], () => this.weightsHiddenHidden[i][h], v => { this.weightsHiddenHidden[i][h] = v; })) return false;
            }
            if (!ok(analytic.hiddenBias[i], () => this.hiddenBias[i], v => { this.hiddenBias[i] = v; })) return false;
        }
        for (let h = 0; h < this.hiddenSize; h++) {
            for (let c = 0; c < this.classCount; c++) {
                if (!ok(analytic.weightsHiddenOutput[h][c], () => this.weightsHiddenOutput[h][c], v => { this.weightsHiddenOutput[h][c] = v; })) return false;
            }
        }
        for (let c = 0; c < this.classCount; c++) {
            if (!ok(analytic.outputBias[c], () => this.outputBias[c], v => { this.outputBias[c] = v; })) return false;
        }
        return true;
    }

    /* Parameter setters */

    public setVocabSize (vocabSize: number) { this.vocabSize = vocabSize; return this; }
    public setEmbeddingDim (embeddingDim: number) { this.embeddingDim = embeddingDim; return this; }
    public setHiddenSize (hiddenSize: number) { this.hiddenSize = hiddenSize; return this; }
    public setLearningRate (learningRate: number) { this.learningRate = learningRate; return this; }
    public setNumberOfEpochs (numberOfEpochs: number) { this.numberOfEpochs = numberOfEpochs; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }
    public reset () { this.embeddings = undefined; return this; }

    /* Parameter getters */

    public getVocabSize () { return this.vocabSize; }
    public getEmbeddingDim () { return this.embeddingDim; }
    public getHiddenSize () { return this.hiddenSize; }
    public getLearningRate () { return this.learningRate; }
    public getNumberOfEpochs () { return this.numberOfEpochs; }
    public getSeed () { return this.seed; }

    /** The learned embedding vector for each token (`vocabSize × embeddingDim`). */
    public getEmbeddings () {
        return this.embeddings ? this.embeddings.map(row => row.slice()) : [];
    }

    /* Private methods */

    private initialize (classCount: number) {
        const random = mulberry32(this.seed);
        const rand = (scale: number) => (random() * 2 - 1) * scale;
        this.classCount = classCount;

        const D = this.embeddingDim;
        const H = this.hiddenSize;
        this.embeddings = Array.from({ length: this.vocabSize }, () => Array.from({ length: D }, () => rand(0.5)));
        this.weightsInputHidden = Array.from({ length: D }, () => Array.from({ length: H }, () => rand(Math.sqrt(6 / (D + H)))));
        this.weightsHiddenHidden = Array.from({ length: H }, () => Array.from({ length: H }, () => rand(Math.sqrt(6 / (H + H)))));
        this.hiddenBias = new Array<number>(H).fill(0);
        this.weightsHiddenOutput = Array.from({ length: H }, () => Array.from({ length: classCount }, () => rand(Math.sqrt(6 / (H + classCount)))));
        this.outputBias = new Array<number>(classCount).fill(0);
    }

    private forward (tokens: number[]) {
        const H = this.hiddenSize;
        const realTokens: number[] = [];
        const hiddenStates: number[][] = [new Array<number>(H).fill(0)]; // h_0 = zeros

        for (const token of tokens) {
            if (token <= 0) {
                continue; // padding: carry the hidden state unchanged
            }
            const previous = hiddenStates[hiddenStates.length - 1];
            const embedding = this.embeddings[token];
            const next = new Array<number>(H);
            for (let h = 0; h < H; h++) {
                let sum = this.hiddenBias[h];
                for (let d = 0; d < this.embeddingDim; d++) sum += embedding[d] * this.weightsInputHidden[d][h];
                for (let i = 0; i < H; i++) sum += previous[i] * this.weightsHiddenHidden[i][h];
                next[h] = Math.tanh(sum);
            }
            realTokens.push(token);
            hiddenStates.push(next);
        }

        const finalState = hiddenStates[hiddenStates.length - 1];
        const logits = new Array<number>(this.classCount);
        for (let c = 0; c < this.classCount; c++) {
            let sum = this.outputBias[c];
            for (let h = 0; h < H; h++) sum += finalState[h] * this.weightsHiddenOutput[h][c];
            logits[c] = sum;
        }

        return { realTokens, hiddenStates, probabilities: softmax(logits) };
    }

    private accumulateGradients (cache: ReturnType<RecurrentNeuralNetwork['forward']>, label: number[], gradients: Gradients) {
        const H = this.hiddenSize;
        const { realTokens, hiddenStates, probabilities } = cache;
        const steps = realTokens.length;
        const finalState = hiddenStates[steps];

        // Output layer: softmax + cross-entropy gives prediction − target.
        const dLogits = new Array<number>(this.classCount);
        for (let c = 0; c < this.classCount; c++) dLogits[c] = probabilities[c] - label[c];

        let dh = new Array<number>(H).fill(0);
        for (let c = 0; c < this.classCount; c++) {
            gradients.outputBias[c] += dLogits[c];
            for (let h = 0; h < H; h++) {
                gradients.weightsHiddenOutput[h][c] += finalState[h] * dLogits[c];
                dh[h] += this.weightsHiddenOutput[h][c] * dLogits[c];
            }
        }

        // Backprop through time, latest step first.
        for (let t = steps - 1; t >= 0; t--) {
            const token = realTokens[t];
            const embedding = this.embeddings[token];
            const current = hiddenStates[t + 1];
            const previous = hiddenStates[t];

            const dRaw = new Array<number>(H);
            for (let h = 0; h < H; h++) {
                dRaw[h] = dh[h] * (1 - current[h] * current[h]); // tanh'
            }

            const dhPrev = new Array<number>(H).fill(0);
            for (let h = 0; h < H; h++) {
                const g = dRaw[h];
                gradients.hiddenBias[h] += g;
                for (let d = 0; d < this.embeddingDim; d++) {
                    gradients.weightsInputHidden[d][h] += embedding[d] * g;
                    gradients.embeddings[token][d] += this.weightsInputHidden[d][h] * g;
                }
                for (let i = 0; i < H; i++) {
                    gradients.weightsHiddenHidden[i][h] += previous[i] * g;
                    dhPrev[i] += this.weightsHiddenHidden[i][h] * g;
                }
            }
            dh = dhPrev;
        }
    }

    private applyGradients (g: Gradients, count: number) {
        const step = this.learningRate / count;
        const update = (param: number[][], grad: number[][]) => {
            for (let i = 0; i < param.length; i++) {
                for (let j = 0; j < param[i].length; j++) param[i][j] -= step * grad[i][j];
            }
        };
        update(this.embeddings, g.embeddings);
        update(this.weightsInputHidden, g.weightsInputHidden);
        update(this.weightsHiddenHidden, g.weightsHiddenHidden);
        update(this.weightsHiddenOutput, g.weightsHiddenOutput);
        for (let h = 0; h < this.hiddenSize; h++) this.hiddenBias[h] -= step * g.hiddenBias[h];
        for (let c = 0; c < this.classCount; c++) this.outputBias[c] -= step * g.outputBias[c];
    }

    private zeroGradients (): Gradients {
        const zeros = (rows: number, cols: number) => Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
        return {
            embeddings: zeros(this.vocabSize, this.embeddingDim),
            weightsInputHidden: zeros(this.embeddingDim, this.hiddenSize),
            weightsHiddenHidden: zeros(this.hiddenSize, this.hiddenSize),
            hiddenBias: new Array<number>(this.hiddenSize).fill(0),
            weightsHiddenOutput: zeros(this.hiddenSize, this.classCount),
            outputBias: new Array<number>(this.classCount).fill(0),
        };
    }
}

interface Gradients {
    embeddings: number[][];
    weightsInputHidden: number[][];
    weightsHiddenHidden: number[][];
    hiddenBias: number[];
    weightsHiddenOutput: number[][];
    outputBias: number[];
}

function toTokens (row: number[]) {
    return row.map(value => Math.round(value));
}

function softmax (logits: number[]) {
    const max = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}
