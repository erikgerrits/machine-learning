import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/**
 * A minimal **transformer** — a single self-attention block with a classification head, the core of
 * the architecture behind modern language models (including the one that wrote this library). Where
 * an RNN ferries the whole past through one hidden state read left-to-right, **attention** lets every
 * position look **directly at every other position** at once and decide, per pair, how much to care.
 *
 * The block, trained end-to-end by backprop through the attention:
 *
 *   tokens → token + positional **embeddings** (X)
 *          → **Q, K, V** = X·Wq, X·Wk, X·Wv
 *          → scores = QKᵀ/√d → **softmax** per row → attention weights A
 *          → context = A·V
 *          → classify from the first position's context (a "[CLS]" slot) → softmax
 *
 * The first token of every sequence is treated as a classification slot: its context vector — a
 * learned, attention-weighted blend of the whole sequence — feeds the output layer. Reading off that
 * slot's attention row shows *which words the model decided mattered*.
 *
 * Sequences are fixed-length token-id rows (no padding/masking — keep them all the same length, with
 * the CLS token at position 0). This is one head and one layer — the essential mechanism; real
 * transformers stack many multi-head blocks with feed-forward layers, residuals, and normalisation.
 * {@link checkGradients} finite-difference-verifies the attention gradients.
 *
 * @example
 * const t = new Transformer().setVocabSize(20).setModelDim(8).setMaxLength(6).setSeed(0);
 * t.setLearningRate(0.05).setNumberOfEpochs(300).train(sequences, oneHot);
 * t.getAttention(sequences.toArray()[0]); // L×L attention; row 0 = what [CLS] looked at
 */
export default class Transformer {

    private vocabSize = 16;
    private modelDim = 8;
    private maxLength = 8;
    private learningRate = 0.05;
    private numberOfEpochs = 1;
    private seed = 0;

    // Learned parameters.
    private tokenEmbeddings: number[][];    // [token][dim]
    private positionEmbeddings: number[][]; // [position][dim]
    private wq: number[][]; private wk: number[][]; private wv: number[][]; // [dim][dim]
    private wOut: number[][];               // [dim][class]
    private outBias: number[];              // [class]
    private classCount = 0;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const sequences = inputs.toArray();
        const labels = targets.toArray();
        if (sequences.length === 0) {
            return this;
        }
        if (this.tokenEmbeddings === undefined) {
            this.initialize(labels[0].length);
        }

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const gradients = this.zeroGradients();
            for (let n = 0; n < sequences.length; n++) {
                const tokens = sequences[n].map(v => Math.round(v));
                this.accumulateGradients(tokens, this.forward(tokens), labels[n], gradients);
            }
            this.applyGradients(gradients, sequences.length);
        }
        return this;
    }

    /** Class probabilities for each input sequence. */
    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => this.forward(row.map(v => Math.round(v))).probabilities));
    }

    /** Average cross-entropy over the given sequences. */
    public computeLoss (inputs: Matrix, targets: Matrix) {
        const sequences = inputs.toArray();
        const labels = targets.toArray();
        if (sequences.length === 0) return 0;
        let loss = 0;
        for (let n = 0; n < sequences.length; n++) {
            const { probabilities } = this.forward(sequences[n].map(v => Math.round(v)));
            for (let c = 0; c < this.classCount; c++) loss -= labels[n][c] * Math.log(Math.max(1e-12, probabilities[c]));
        }
        return loss / sequences.length;
    }

    /** The attention matrix for one sequence: `A[i][j]` = how much position i attends to position j. */
    public getAttention (sequence: number[]) {
        return this.forward(sequence.map(v => Math.round(v))).attention.map(row => row.slice());
    }

    /**
     * Finite-difference check of backprop through the self-attention (the trickiest gradient here —
     * softmax-of-dot-products feeding a weighted sum). Reports whether analytic and numeric agree.
     */
    public checkGradients () {
        const random = mulberry32(this.seed + 4242);
        if (this.tokenEmbeddings === undefined) this.initialize(2);

        const length = Math.min(4, this.maxLength);
        const tokens = Array.from({ length }, () => Math.floor(random() * this.vocabSize));
        const label = new Array<number>(this.classCount).fill(0);
        label[Math.floor(random() * this.classCount)] = 1;

        const analytic = this.zeroGradients();
        this.accumulateGradients(tokens, this.forward(tokens), label, analytic);

        const epsilon = 1e-4;
        const tolerance = 1e-3;
        const lossAt = () => {
            const { probabilities } = this.forward(tokens);
            let loss = 0;
            for (let c = 0; c < this.classCount; c++) loss -= label[c] * Math.log(Math.max(1e-12, probabilities[c]));
            return loss;
        };
        const ok = (g: number, get: () => number, set: (v: number) => void) => {
            const o = get();
            set(o + epsilon); const plus = lossAt();
            set(o - epsilon); const minus = lossAt();
            set(o);
            const numeric = (plus - minus) / (2 * epsilon);
            return Math.abs(g - numeric) / Math.max(1, Math.abs(g) + Math.abs(numeric)) < tolerance;
        };
        const checkMatrix = (g: number[][], p: number[][]) => {
            for (let i = 0; i < p.length; i++) for (let j = 0; j < p[i].length; j++) {
                if (!ok(g[i][j], () => p[i][j], v => { p[i][j] = v; })) return false;
            }
            return true;
        };

        if (!checkMatrix(analytic.wq, this.wq)) return false;
        if (!checkMatrix(analytic.wk, this.wk)) return false;
        if (!checkMatrix(analytic.wv, this.wv)) return false;
        if (!checkMatrix(analytic.wOut, this.wOut)) return false;
        for (let c = 0; c < this.classCount; c++) if (!ok(analytic.outBias[c], () => this.outBias[c], v => { this.outBias[c] = v; })) return false;
        for (const t of new Set(tokens)) {
            for (let d = 0; d < this.modelDim; d++) if (!ok(analytic.tokenEmbeddings[t][d], () => this.tokenEmbeddings[t][d], v => { this.tokenEmbeddings[t][d] = v; })) return false;
        }
        for (let p = 0; p < length; p++) {
            for (let d = 0; d < this.modelDim; d++) if (!ok(analytic.positionEmbeddings[p][d], () => this.positionEmbeddings[p][d], v => { this.positionEmbeddings[p][d] = v; })) return false;
        }
        return true;
    }

    /* Setters */
    public setVocabSize (v: number) { this.vocabSize = v; return this; }
    public setModelDim (d: number) { this.modelDim = d; return this; }
    public setMaxLength (l: number) { this.maxLength = l; return this; }
    public setLearningRate (lr: number) { this.learningRate = lr; return this; }
    public setNumberOfEpochs (e: number) { this.numberOfEpochs = e; return this; }
    public setSeed (s: number) { this.seed = s; return this; }
    public reset () { this.tokenEmbeddings = undefined; return this; }

    /* Getters */
    public getVocabSize () { return this.vocabSize; }
    public getModelDim () { return this.modelDim; }
    public getMaxLength () { return this.maxLength; }
    public getLearningRate () { return this.learningRate; }
    public getNumberOfEpochs () { return this.numberOfEpochs; }
    public getSeed () { return this.seed; }

    /* Private */

    private initialize (classCount: number) {
        const random = mulberry32(this.seed);
        const D = this.modelDim;
        const rand = (scale: number) => (random() * 2 - 1) * scale;
        const square = () => Array.from({ length: D }, () => Array.from({ length: D }, () => rand(1 / Math.sqrt(D))));

        this.classCount = classCount;
        this.tokenEmbeddings = Array.from({ length: this.vocabSize }, () => Array.from({ length: D }, () => rand(0.4)));
        this.positionEmbeddings = Array.from({ length: this.maxLength }, () => Array.from({ length: D }, () => rand(0.4)));
        this.wq = square(); this.wk = square(); this.wv = square();
        this.wOut = Array.from({ length: D }, () => Array.from({ length: classCount }, () => rand(Math.sqrt(6 / (D + classCount)))));
        this.outBias = new Array<number>(classCount).fill(0);
    }

    private forward (tokens: number[]) {
        const D = this.modelDim;
        const L = tokens.length;
        const scaling = 1 / Math.sqrt(D);

        // Embedded input X = token + positional.
        const x: number[][] = [];
        for (let t = 0; t < L; t++) {
            const row = new Array<number>(D);
            for (let d = 0; d < D; d++) row[d] = this.tokenEmbeddings[tokens[t]][d] + this.positionEmbeddings[t][d];
            x.push(row);
        }

        const q = matMul(x, this.wq);
        const k = matMul(x, this.wk);
        const v = matMul(x, this.wv);

        // Attention scores → row softmax → context.
        const attention: number[][] = [];
        const context: number[][] = [];
        for (let i = 0; i < L; i++) {
            const scores = new Array<number>(L);
            for (let j = 0; j < L; j++) {
                let dot = 0;
                for (let d = 0; d < D; d++) dot += q[i][d] * k[j][d];
                scores[j] = dot * scaling;
            }
            const a = softmax(scores);
            attention.push(a);
            const ctx = new Array<number>(D).fill(0);
            for (let j = 0; j < L; j++) for (let d = 0; d < D; d++) ctx[d] += a[j] * v[j][d];
            context.push(ctx);
        }

        // Classify from the [CLS] position (index 0).
        const cls = context[0];
        const logits = new Array<number>(this.classCount);
        for (let c = 0; c < this.classCount; c++) {
            let sum = this.outBias[c];
            for (let d = 0; d < D; d++) sum += cls[d] * this.wOut[d][c];
            logits[c] = sum;
        }

        return { x, q, k, v, attention, context, probabilities: softmax(logits) };
    }

    private accumulateGradients (tokens: number[], cache: ReturnType<Transformer['forward']>, label: number[], g: Gradients) {
        const D = this.modelDim;
        const L = tokens.length;
        const scaling = 1 / Math.sqrt(D);
        const { x, q, k, v, attention, context, probabilities } = cache;

        // Output layer (from CLS context).
        const dLogits = new Array<number>(this.classCount);
        for (let c = 0; c < this.classCount; c++) dLogits[c] = probabilities[c] - label[c];

        const dContext = Array.from({ length: L }, () => new Array<number>(D).fill(0));
        for (let c = 0; c < this.classCount; c++) {
            g.outBias[c] += dLogits[c];
            for (let d = 0; d < D; d++) {
                g.wOut[d][c] += context[0][d] * dLogits[c];
                dContext[0][d] += this.wOut[d][c] * dLogits[c]; // only CLS feeds the head
            }
        }

        const dq = Array.from({ length: L }, () => new Array<number>(D).fill(0));
        const dk = Array.from({ length: L }, () => new Array<number>(D).fill(0));
        const dv = Array.from({ length: L }, () => new Array<number>(D).fill(0));

        // Back through context = A·V and the row-softmax, per query row i.
        for (let i = 0; i < L; i++) {
            const a = attention[i];
            const dA = new Array<number>(L).fill(0);
            for (let j = 0; j < L; j++) {
                let s = 0;
                for (let d = 0; d < D; d++) {
                    s += dContext[i][d] * v[j][d];        // dA[i][j]
                    dv[j][d] += a[j] * dContext[i][d];    // dV[j]
                }
                dA[j] = s;
            }
            // Softmax Jacobian for row i.
            let weighted = 0;
            for (let j = 0; j < L; j++) weighted += a[j] * dA[j];
            const dScores = new Array<number>(L);
            for (let j = 0; j < L; j++) dScores[j] = a[j] * (dA[j] - weighted) * scaling;
            // scores[i][j] = q[i]·k[j].
            for (let j = 0; j < L; j++) {
                for (let d = 0; d < D; d++) {
                    dq[i][d] += dScores[j] * k[j][d];
                    dk[j][d] += dScores[j] * q[i][d];
                }
            }
        }

        // Back through Q,K,V = X·W, accumulating dX.
        const dx = Array.from({ length: L }, () => new Array<number>(D).fill(0));
        accumulateLinear(x, dq, this.wq, g.wq, dx);
        accumulateLinear(x, dk, this.wk, g.wk, dx);
        accumulateLinear(x, dv, this.wv, g.wv, dx);

        // Back into the embeddings.
        for (let t = 0; t < L; t++) {
            for (let d = 0; d < D; d++) {
                g.tokenEmbeddings[tokens[t]][d] += dx[t][d];
                g.positionEmbeddings[t][d] += dx[t][d];
            }
        }
    }

    private applyGradients (g: Gradients, count: number) {
        const step = this.learningRate / count;
        const update = (p: number[][], grad: number[][]) => {
            for (let i = 0; i < p.length; i++) for (let j = 0; j < p[i].length; j++) p[i][j] -= step * grad[i][j];
        };
        update(this.tokenEmbeddings, g.tokenEmbeddings);
        update(this.positionEmbeddings, g.positionEmbeddings);
        update(this.wq, g.wq); update(this.wk, g.wk); update(this.wv, g.wv);
        update(this.wOut, g.wOut);
        for (let c = 0; c < this.classCount; c++) this.outBias[c] -= step * g.outBias[c];
    }

    private zeroGradients (): Gradients {
        const zeros = (r: number, c: number) => Array.from({ length: r }, () => new Array<number>(c).fill(0));
        const D = this.modelDim;
        return {
            tokenEmbeddings: zeros(this.vocabSize, D),
            positionEmbeddings: zeros(this.maxLength, D),
            wq: zeros(D, D), wk: zeros(D, D), wv: zeros(D, D),
            wOut: zeros(D, this.classCount),
            outBias: new Array<number>(this.classCount).fill(0),
        };
    }
}

interface Gradients {
    tokenEmbeddings: number[][];
    positionEmbeddings: number[][];
    wq: number[][]; wk: number[][]; wv: number[][];
    wOut: number[][];
    outBias: number[];
}

/** Y = X·W where X is L×D and W is D×D → L×D. */
function matMul (x: number[][], w: number[][]) {
    const L = x.length;
    const D = w.length;
    const out: number[][] = [];
    for (let i = 0; i < L; i++) {
        const row = new Array<number>(D).fill(0);
        for (let d = 0; d < D; d++) {
            for (let e = 0; e < D; e++) row[e] += x[i][d] * w[d][e];
        }
        out.push(row);
    }
    return out;
}

/** Given upstream gradient dY of Y = X·W: accumulate dW (Xᵀ·dY) and add dX (dY·Wᵀ) into `dx`. */
function accumulateLinear (x: number[][], dY: number[][], w: number[][], dW: number[][], dx: number[][]) {
    const L = x.length;
    const D = w.length;
    for (let i = 0; i < L; i++) {
        for (let d = 0; d < D; d++) {
            const xid = x[i][d];
            let dxid = 0;
            for (let e = 0; e < D; e++) {
                dW[d][e] += xid * dY[i][e];
                dxid += dY[i][e] * w[d][e];
            }
            dx[i][d] += dxid;
        }
    }
}

function softmax (values: number[]) {
    const max = Math.max(...values);
    const exps = values.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}
