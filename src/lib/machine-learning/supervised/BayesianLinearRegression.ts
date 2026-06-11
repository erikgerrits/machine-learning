import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/** Which basis functions the regression expands its single input into before fitting a linear model. */
export type BayesianBasis = "gaussian" | "polynomial";

/**
 * **Bayesian linear regression** — the course finale, and the model that finally admits what it
 * *doesn't* know. Every regressor so far returned a single best-fit line; this one keeps a whole
 * **distribution** over the possible lines and reports, for any input, not just a prediction but how
 * *sure* it is. Near your data it's confident; far from it — extrapolating past the edge, or in a gap —
 * the uncertainty fans out, exactly as honesty demands before a big bet.
 *
 * The machinery is conjugate Bayesian inference, in closed form (no sampling needed to fit). The input
 * `x` is expanded through basis functions `φ(x)` (Gaussian bumps or polynomial powers) so the "line" can
 * curve. A Gaussian **prior** `w ~ N(0, α⁻¹I)` on the weights meets the data through a Gaussian noise
 * model (precision `β`), and the **posterior** over weights is again Gaussian:
 *
 * ```
 * Sₙ = (α·I + β·ΦᵀΦ)⁻¹           // posterior covariance — shrinks as data accumulates
 * mₙ = β·Sₙ·Φᵀy                  // posterior mean weights (the best-fit curve)
 * ```
 *
 * For a new input the **predictive distribution** is Gaussian too, with variance `1/β + φ(x)ᵀSₙφ(x)`:
 * the noise floor `1/β` plus an *epistemic* term that grows wherever the data didn't pin the weights
 * down. {@link predict} returns the mean curve, {@link predictiveStandardDeviation} the error bars, and
 * {@link sample} draws whole plausible curves from the posterior — tight bundles near data, spreading
 * wildly where there's none.
 *
 * @example
 * const model = new BayesianLinearRegression().setBasis('gaussian').setNumberOfBases(6);
 * model.train(x, y);
 * model.predict(grid);                          // the mean fit
 * model.predictiveStandardDeviation(grid);      // ± how sure, point by point
 */
export default class BayesianLinearRegression {

    private basis: BayesianBasis = "gaussian";
    private degree = 4;          // polynomial basis: highest power
    private numberOfBases = 6;    // gaussian basis: number of bumps
    private basisWidth = 0.15;    // gaussian bump width (in input units)
    private alpha = 1;            // prior precision — larger ⇒ stronger pull toward a flat line
    private beta = 25;            // noise precision (1/variance); noise std = 1/√β
    private seed = 0;

    private centers: number[] = [];
    private meanWeights: Matrix;     // M × 1
    private covariance: Matrix;      // M × M (posterior Sₙ)

    public constructor () {}

    /** Fit the posterior over weights from inputs (N × 1) and targets (N × 1). */
    public train (inputs: Matrix, targets: Matrix) {
        this.prepareCenters(inputs);
        const phi = this.designMatrix(inputs);          // N × M
        const phiT = Matrix.transpose(phi);
        const m = phi.getColumnCount();

        // A = α·I + β·ΦᵀΦ ; posterior covariance Sₙ = A⁻¹.
        const a = Matrix.add(Matrix.multiply(Matrix.identity(m), this.alpha), Matrix.multiply(Matrix.multiply(phiT, phi), this.beta));
        this.covariance = a.getInverse();
        // Posterior mean mₙ = β·Sₙ·Φᵀ·y.
        this.meanWeights = Matrix.multiply(Matrix.multiply(Matrix.multiply(this.covariance, phiT), targets), this.beta);
        return this;
    }

    /** The posterior-mean prediction for each input (N × 1). */
    public predict (inputs: Matrix) {
        return Matrix.multiply(this.designMatrix(inputs), this.meanWeights);
    }

    /** The predictive standard deviation at each input — the error bar that grows away from the data. */
    public predictiveStandardDeviation (inputs: Matrix): number[] {
        const phi = this.designMatrix(inputs);
        const sPhiT = Matrix.multiply(this.covariance, Matrix.transpose(phi)); // M × N
        const noiseVariance = 1 / this.beta;
        const result: number[] = [];
        for (let r = 0; r < phi.getRowCount(); r++) {
            let epistemic = 0; // φ(x)ᵀ Sₙ φ(x)
            for (let c = 0; c < phi.getColumnCount(); c++) epistemic += phi.getElement(r, c) * sPhiT.getElement(c, r);
            result.push(Math.sqrt(Math.max(0, noiseVariance + epistemic)));
        }
        return result;
    }

    /** Draw `count` whole curves from the posterior over functions, evaluated at `inputs` (count × N). */
    public sample (inputs: Matrix, count: number, seed?: number): number[][] {
        const random = mulberry32(seed === undefined ? this.seed : seed);
        const phi = this.designMatrix(inputs);
        const lower = cholesky(this.covariance.toArray()); // Sₙ = L·Lᵀ
        const mean = this.meanWeights.toArray().map(row => row[0]);
        const m = mean.length;

        const curves: number[][] = [];
        for (let s = 0; s < count; s++) {
            const z = Array.from({ length: m }, () => gaussian(random));
            const weights = mean.map((mu, i) => {
                let lz = 0;
                for (let j = 0; j <= i; j++) lz += lower[i][j] * z[j]; // (L·z)_i, L lower-triangular
                return mu + lz;
            });
            const curve: number[] = [];
            for (let r = 0; r < phi.getRowCount(); r++) {
                let value = 0;
                for (let c = 0; c < m; c++) value += phi.getElement(r, c) * weights[c];
                curve.push(value);
            }
            curves.push(curve);
        }
        return curves;
    }

    /* Parameter setters */

    public setBasis (basis: BayesianBasis) { this.basis = basis; return this; }
    public setDegree (degree: number) { this.degree = degree; return this; }
    public setNumberOfBases (numberOfBases: number) { this.numberOfBases = numberOfBases; return this; }
    public setBasisWidth (basisWidth: number) { this.basisWidth = basisWidth; return this; }
    public setAlpha (alpha: number) { this.alpha = alpha; return this; }
    public setBeta (beta: number) { this.beta = beta; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getBasis () { return this.basis; }
    public getDegree () { return this.degree; }
    public getNumberOfBases () { return this.numberOfBases; }
    public getBasisWidth () { return this.basisWidth; }
    public getAlpha () { return this.alpha; }
    public getBeta () { return this.beta; }
    public getSeed () { return this.seed; }
    /** The posterior-mean weight vector. */
    public getWeights () { return this.meanWeights.toArray().map(row => row[0]); }

    /* Private methods */

    private prepareCenters (inputs: Matrix) {
        if (this.basis !== "gaussian") return;
        let min = Infinity, max = -Infinity;
        for (let r = 0; r < inputs.getRowCount(); r++) {
            min = Math.min(min, inputs.getElement(r, 0));
            max = Math.max(max, inputs.getElement(r, 0));
        }
        if (!isFinite(min)) { min = 0; max = 1; }
        this.centers = [];
        for (let i = 0; i < this.numberOfBases; i++) {
            this.centers.push(this.numberOfBases === 1 ? (min + max) / 2 : min + (max - min) * (i / (this.numberOfBases - 1)));
        }
    }

    /** Expand each input row into its basis-function features (N × M, including a bias column). */
    private designMatrix (inputs: Matrix): Matrix {
        const rows: number[][] = [];
        for (let r = 0; r < inputs.getRowCount(); r++) rows.push(this.features(inputs.getElement(r, 0)));
        return new Matrix(rows);
    }

    private features (x: number): number[] {
        if (this.basis === "polynomial") {
            const row = [1];
            for (let d = 1; d <= this.degree; d++) row.push(x ** d);
            return row;
        }
        const row = [1]; // bias
        for (const center of this.centers) row.push(Math.exp(-((x - center) ** 2) / (2 * this.basisWidth ** 2)));
        return row;
    }
}

/** Cholesky decomposition of a symmetric positive-definite matrix: returns lower-triangular L with L·Lᵀ = A. */
function cholesky (a: number[][]): number[][] {
    const n = a.length;
    const l: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) sum += l[i][k] * l[j][k];
            if (i === j) l[i][j] = Math.sqrt(Math.max(1e-12, a[i][i] - sum));
            else l[i][j] = (a[i][j] - sum) / l[j][j];
        }
    }
    return l;
}

/** One standard-normal sample via Box–Muller. */
function gaussian (random: () => number): number {
    let u = 0;
    while (u === 0) u = random();
    let v = 0;
    while (v === 0) v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
