import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/**
 * A **variational autoencoder (VAE)** — the {@link Autoencoder} turned into a true *generative* model,
 * and the chapter where the café stops describing data and starts inventing it. A plain autoencoder
 * maps each input to a single point in code space; sample a random code and the decoder usually returns
 * garbage, because the points it learned sit in scattered islands with nonsense in between. The VAE
 * fixes that by making the code space **well-organised and samplable**:
 *
 * - The encoder outputs not a point but a little **distribution** per input — a mean `μ` and a
 *   (log-)variance `logσ²` for each latent dimension.
 * - A code is **sampled** from it via the *reparameterisation trick*: `z = μ + σ·ε`, with `ε ~ N(0, 1)`.
 *   Writing the randomness as an external `ε` keeps `z` differentiable in `μ` and `σ`, so backprop
 *   still works.
 * - The loss adds a **KL-divergence** term that pulls every input's latent distribution toward a
 *   standard normal `N(0, 1)`. This packs the codes into one smooth, gap-free blob centred on the
 *   origin — so afterwards you can **draw `z ~ N(0, 1)`, decode it, and get a brand-new, plausible
 *   sample** ({@link sample}). `β` trades reconstruction sharpness against how tidy that latent space is.
 *
 * Loss = reconstruction (binary cross-entropy, for sigmoid pixels in `[0, 1]`) + β·KL. One hidden layer
 * each side; the full path — including the gradient through the sampling and the KL — is finite-
 * difference checked by {@link checkGradients}.
 *
 * @example
 * const vae = new VariationalAutoencoder().setHiddenSize(32).setCodeSize(2).setNumberOfEpochs(2000);
 * vae.train(images);            // rows of pixels in [0, 1]
 * vae.sample(16);               // 16 freshly generated images, drawn from the prior
 * vae.generate(new Matrix([[0, 0]])); // decode a specific point in latent space
 */
export default class VariationalAutoencoder {

    private inputSize = 0;
    private hiddenSize = 32;
    private codeSize = 2;
    private beta = 1;
    private learningRate = 0.05;
    private numberOfEpochs = 2000;
    private seed = 0;

    private encoderHidden: Matrix;  // (inputSize + 1) × hiddenSize
    private meanHead: Matrix;       // (hiddenSize + 1) × codeSize
    private logVarHead: Matrix;     // (hiddenSize + 1) × codeSize
    private decoderHidden: Matrix;  // (codeSize + 1) × hiddenSize
    private decoderOutput: Matrix;  // (hiddenSize + 1) × inputSize
    private random: () => number;
    private built = false;

    public constructor () {}

    /** Train to reconstruct `inputs` (rows of features in [0, 1]) while shaping the latent space. */
    public train (inputs: Matrix) {
        this.ensureBuilt(inputs.getColumnCount());
        const n = inputs.getRowCount();
        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            const noise = this.sampleNoise(n);
            this.applyGradients(this.computeGradients(inputs, noise));
        }
        return this;
    }

    /** Encode inputs to their latent means μ (the deterministic code, ignoring the sampling noise). */
    public encode (inputs: Matrix) {
        this.ensureBuilt(inputs.getColumnCount());
        const hidden = Matrix.transform(this.affine(inputs, this.encoderHidden), tanh);
        return this.affine(hidden, this.meanHead);
    }

    /** Decode latent codes back into inputs (the generator). */
    public generate (codes: Matrix) {
        const hidden = Matrix.transform(this.affine(codes, this.decoderHidden), tanh);
        return Matrix.transform(this.affine(hidden, this.decoderOutput), sigmoid);
    }

    /** Reconstruct inputs by encoding to the mean and decoding (no sampling noise). */
    public reconstruct (inputs: Matrix) {
        return this.generate(this.encode(inputs));
    }

    /** Draw `count` brand-new samples: z ~ N(0, 1) → decode. Pass a seed for reproducibility. */
    public sample (count: number, seed?: number) {
        this.ensureBuilt(this.inputSize);
        const random = seed === undefined ? this.random : mulberry32(seed);
        const codes: number[][] = [];
        for (let i = 0; i < count; i++) {
            const row: number[] = [];
            for (let k = 0; k < this.codeSize; k++) row.push(gaussian(random));
            codes.push(row);
        }
        return this.generate(new Matrix(codes));
    }

    /** Per-row reconstruction (binary cross-entropy) — usable as an anomaly / typicality score. */
    public reconstructionError (inputs: Matrix): number[] {
        const reconstructed = this.reconstruct(inputs).toArray();
        const original = inputs.toArray();
        return original.map((row, i) => {
            let sum = 0;
            for (let j = 0; j < row.length; j++) sum += crossEntropy(reconstructed[i][j], row[j]);
            return sum / row.length;
        });
    }

    /** The training objective at the latent means (reconstruction BCE + β·KL), averaged over rows. */
    public computeLoss (inputs: Matrix): number {
        this.ensureBuilt(inputs.getColumnCount());
        const hidden = Matrix.transform(this.affine(inputs, this.encoderHidden), tanh);
        const mean = this.affine(hidden, this.meanHead);
        const logVar = this.affine(hidden, this.logVarHead);
        const output = this.generate(mean); // decode the mean (zero noise) for a stable monitor
        return this.loss(output, inputs, mean, logVar);
    }

    /**
     * Verifies the analytic gradients against numerical ones on a small random problem, with the
     * sampling noise held fixed so the objective is deterministic. Returns true if every gradient
     * matches to a tight tolerance — the proof the reparameterised + KL backprop is correct.
     */
    public checkGradients (): boolean {
        const probe = new VariationalAutoencoder().setHiddenSize(5).setCodeSize(2).setBeta(0.7).setSeed(2);
        const inputs = Matrix.rand(4, 6, 0.4, 3).transform(v => v + 0.5); // 4 rows of 6 features in (0, 1)
        probe.ensureBuilt(6);
        const noise = probe.sampleNoise(4);

        const analytic = probe.computeGradients(inputs, noise);
        const epsilon = 1e-4;
        let maxError = 0;

        for (const name of GRADIENT_ORDER) {
            const weight = probe.weightFor(name);
            const gradient = analytic[name];
            for (let r = 0; r < weight.getRowCount(); r++) {
                for (let c = 0; c < weight.getColumnCount(); c++) {
                    const original = weight.getElement(r, c);
                    weight.setElement(r, c, original + epsilon);
                    const lossPlus = probe.lossWith(inputs, noise);
                    weight.setElement(r, c, original - epsilon);
                    const lossMinus = probe.lossWith(inputs, noise);
                    weight.setElement(r, c, original);

                    const numeric = (lossPlus - lossMinus) / (2 * epsilon);
                    const denominator = Math.max(1e-8, Math.abs(numeric) + Math.abs(gradient.getElement(r, c)));
                    maxError = Math.max(maxError, Math.abs(numeric - gradient.getElement(r, c)) / denominator);
                }
            }
        }
        return maxError < 1e-4;
    }

    /* Parameter setters */

    public setInputSize (inputSize: number) { this.inputSize = inputSize; return this; }
    public setHiddenSize (hiddenSize: number) { this.hiddenSize = hiddenSize; return this; }
    public setCodeSize (codeSize: number) { this.codeSize = codeSize; return this; }
    public setBeta (beta: number) { this.beta = beta; return this; }
    public setLearningRate (learningRate: number) { this.learningRate = learningRate; return this; }
    public setNumberOfEpochs (numberOfEpochs: number) { this.numberOfEpochs = numberOfEpochs; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getInputSize () { return this.inputSize; }
    public getHiddenSize () { return this.hiddenSize; }
    public getCodeSize () { return this.codeSize; }
    public getBeta () { return this.beta; }
    public getLearningRate () { return this.learningRate; }
    public getNumberOfEpochs () { return this.numberOfEpochs; }
    public getSeed () { return this.seed; }

    /* Private methods */

    private ensureBuilt (inputSize: number) {
        if (this.built && this.inputSize === inputSize) return;
        if (inputSize > 0) this.inputSize = inputSize;
        const i = this.inputSize, h = this.hiddenSize, k = this.codeSize;
        this.encoderHidden = Matrix.rand(i + 1, h, Math.sqrt(6) / Math.sqrt(i + h), this.seed + 1);
        this.meanHead = Matrix.rand(h + 1, k, Math.sqrt(6) / Math.sqrt(h + k), this.seed + 2);
        this.logVarHead = Matrix.rand(h + 1, k, 0.01, this.seed + 3); // tiny → start near logσ²=0 (σ≈1), stable
        this.decoderHidden = Matrix.rand(k + 1, h, Math.sqrt(6) / Math.sqrt(k + h), this.seed + 4);
        this.decoderOutput = Matrix.rand(h + 1, i, Math.sqrt(6) / Math.sqrt(h + i), this.seed + 5);
        this.random = mulberry32(this.seed);
        this.built = true;
    }

    /** Append a bias column and multiply by a weight matrix: [1 | input] · weights. */
    private affine (input: Matrix, weights: Matrix) {
        return Matrix.multiply(Matrix.appendLeft(input, Matrix.ones(input.getRowCount(), 1)), weights);
    }

    private sampleNoise (n: number): Matrix {
        const rows: number[][] = [];
        for (let i = 0; i < n; i++) {
            const row: number[] = [];
            for (let k = 0; k < this.codeSize; k++) row.push(gaussian(this.random));
            rows.push(row);
        }
        return new Matrix(rows);
    }

    private computeGradients (inputs: Matrix, noise: Matrix): Record<GradientName, Matrix> {
        const n = inputs.getRowCount();

        // ---- Forward, keeping every intermediate the backward pass needs.
        const encHiddenZ = this.affine(inputs, this.encoderHidden);
        const encHidden = Matrix.transform(encHiddenZ, tanh);
        const mean = this.affine(encHidden, this.meanHead);
        const logVar = this.affine(encHidden, this.logVarHead);
        const std = Matrix.transform(logVar, v => Math.exp(0.5 * v));
        const z = Matrix.add(mean, hadamard(std, noise)); // reparameterisation
        const decHiddenZ = this.affine(z, this.decoderHidden);
        const decHidden = Matrix.transform(decHiddenZ, tanh);
        const outputZ = this.affine(decHidden, this.decoderOutput);
        const output = Matrix.transform(outputZ, sigmoid);

        // ---- Backward. Gradients are summed over the batch, then scaled by 1/n at the end.
        // Decoder: BCE + sigmoid output ⇒ output-layer error is simply (output − input).
        const dOutput = Matrix.subtract(output, inputs);
        const gradDecoderOutput = Matrix.multiply(Matrix.transpose(withBias(decHidden)), dOutput);

        let dDecHidden = hadamard(Matrix.multiply(dOutput, Matrix.transpose(dropBias(this.decoderOutput))), Matrix.transform(decHiddenZ, tanhPrime));
        const gradDecoderHidden = Matrix.multiply(Matrix.transpose(withBias(z)), dDecHidden);

        const dz = Matrix.multiply(dDecHidden, Matrix.transpose(dropBias(this.decoderHidden))); // recon grad wrt z

        // z = mean + std·noise ⇒ split into the two heads, then add the KL term's gradients.
        // KL = −½ Σ(1 + logVar − mean² − e^logVar): dKL/dmean = mean, dKL/dlogVar = ½(e^logVar − 1).
        const dMean = Matrix.add(dz, Matrix.multiply(mean, this.beta));
        const dLogVarRecon = hadamard(dz, Matrix.multiply(hadamard(std, noise), 0.5)); // ∂z/∂logVar = ½·std·ε
        const dLogVarKl = Matrix.multiply(Matrix.transform(logVar, v => 0.5 * (Math.exp(v) - 1)), this.beta);
        const dLogVar = Matrix.add(dLogVarRecon, dLogVarKl);

        const gradMeanHead = Matrix.multiply(Matrix.transpose(withBias(encHidden)), dMean);
        const gradLogVarHead = Matrix.multiply(Matrix.transpose(withBias(encHidden)), dLogVar);

        const dEncHidden = hadamard(
            Matrix.add(Matrix.multiply(dMean, Matrix.transpose(dropBias(this.meanHead))),
                       Matrix.multiply(dLogVar, Matrix.transpose(dropBias(this.logVarHead)))),
            Matrix.transform(encHiddenZ, tanhPrime),
        );
        const gradEncoderHidden = Matrix.multiply(Matrix.transpose(withBias(inputs)), dEncHidden);

        const scale = 1 / n;
        return {
            encoderHidden: Matrix.multiply(gradEncoderHidden, scale),
            meanHead: Matrix.multiply(gradMeanHead, scale),
            logVarHead: Matrix.multiply(gradLogVarHead, scale),
            decoderHidden: Matrix.multiply(gradDecoderHidden, scale),
            decoderOutput: Matrix.multiply(gradDecoderOutput, scale),
        };
    }

    private applyGradients (gradients: Record<GradientName, Matrix>) {
        for (const name of GRADIENT_ORDER) {
            const updated = Matrix.subtract(this.weightFor(name), Matrix.multiply(gradients[name], this.learningRate));
            this.setWeightFor(name, updated);
        }
    }

    /** The loss for a fixed sampling noise — used by both the monitor-free gradient check and forward. */
    private lossWith (inputs: Matrix, noise: Matrix): number {
        const encHidden = Matrix.transform(this.affine(inputs, this.encoderHidden), tanh);
        const mean = this.affine(encHidden, this.meanHead);
        const logVar = this.affine(encHidden, this.logVarHead);
        const std = Matrix.transform(logVar, v => Math.exp(0.5 * v));
        const z = Matrix.add(mean, hadamard(std, noise));
        const output = this.generate(z);
        return this.loss(output, inputs, mean, logVar);
    }

    private loss (output: Matrix, target: Matrix, mean: Matrix, logVar: Matrix): number {
        const o = output.toArray(), t = target.toArray(), m = mean.toArray(), lv = logVar.toArray();
        const n = o.length;
        let recon = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < o[i].length; j++) recon += crossEntropy(o[i][j], t[i][j]);
        let kl = 0;
        for (let i = 0; i < n; i++) for (let k = 0; k < m[i].length; k++) kl += -0.5 * (1 + lv[i][k] - m[i][k] ** 2 - Math.exp(lv[i][k]));
        return (recon + this.beta * kl) / n;
    }

    private weightFor (name: GradientName): Matrix {
        return this[name];
    }

    private setWeightFor (name: GradientName, value: Matrix) {
        this[name] = value;
    }
}

type GradientName = 'encoderHidden' | 'meanHead' | 'logVarHead' | 'decoderHidden' | 'decoderOutput';
const GRADIENT_ORDER: GradientName[] = ['encoderHidden', 'meanHead', 'logVarHead', 'decoderHidden', 'decoderOutput'];

function withBias (m: Matrix) {
    return Matrix.appendLeft(m, Matrix.ones(m.getRowCount(), 1));
}

function dropBias (weights: Matrix) {
    return weights.getRows(1); // drop the bias row (row 0)
}

function hadamard (a: Matrix, b: Matrix) {
    return a.getClone().multiplyElementWise(b);
}

function tanh (value: number) {
    return Math.tanh(value);
}

function tanhPrime (value: number) {
    const t = Math.tanh(value);
    return 1 - t * t;
}

function sigmoid (value: number) {
    return 1 / (1 + Math.exp(-value));
}

function crossEntropy (prediction: number, target: number) {
    const p = Math.min(1 - 1e-7, Math.max(1e-7, prediction));
    return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
}

/** One standard-normal sample via Box–Muller, drawn from the given uniform generator. */
function gaussian (random: () => number) {
    let u = 0;
    while (u === 0) u = random();
    let v = 0;
    while (v === 0) v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
