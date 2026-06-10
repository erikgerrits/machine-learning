import Matrix from "../../math/linear-algebra/Matrix";

/**
 * **Anomaly detection** by fitting a multivariate Gaussian to the data and flagging the points it
 * finds least likely. The premise is the mirror image of clustering: instead of asking "which group
 * is this in?", it learns what **normal** looks like — a centre and a spread — and then asks "how
 * far outside normal is this?". Rare events (fraud, a spoiled batch, a sensor glitch) sit far out in
 * the tails and get flagged; the bulk of ordinary points do not.
 *
 * "How far out" is measured by the **Mahalanobis distance** — Euclidean distance warped by the data's
 * covariance, so it counts standard deviations *in the data's own shape*, not raw units. A point two
 * steps off along the direction the data naturally spreads is unremarkable; the same two steps across
 * that grain is glaring. `train` estimates the mean and (slightly regularised) covariance; `score`
 * returns each point's Mahalanobis distance; `predict` flags the ones past `threshold` as `1`,
 * everything else `0`.
 *
 * Like the clustering models it takes inputs with **no targets** (it assumes the training data is
 * mostly normal). Fully deterministic.
 *
 * @example
 * const detector = new AnomalyDetector().setThreshold(3);
 * detector.train(new Matrix([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]));
 * detector.predict(new Matrix([[0, 0], [8, 8]])).toArray(); // [[0], [1]] — the far point is anomalous
 */
export default class AnomalyDetector {

    private threshold = 3;

    private mean: number[] = [];
    private inverseCovariance: number[][] = [];

    public constructor () {}

    public train (inputs: Matrix) {
        const rows = inputs.toArray();
        const n = rows.length;
        const d = n > 0 ? rows[0].length : 0;

        this.mean = columnMeans(rows, d);
        const centered = rows.map(row => row.map((value, j) => value - this.mean[j]));

        // Covariance, with a small ridge on the diagonal so it's always invertible (e.g. when a
        // feature is constant or two features are perfectly collinear).
        const covariance = covarianceMatrix(centered, d);
        for (let i = 0; i < d; i++) {
            covariance[i][i] += 1e-6;
        }

        this.inverseCovariance = new Matrix(covariance).getInverse().toArray();
        return this;
    }

    /** Mahalanobis distance of each input from the fitted centre — the anomaly score. */
    public score (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => [this.mahalanobis(row)]));
    }

    /** Flags each input: `1` if its score exceeds `threshold` (anomalous), `0` otherwise. */
    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => [this.mahalanobis(row) > this.threshold ? 1 : 0]));
    }

    /* Parameter setters */

    public setThreshold (threshold: number) {
        this.threshold = threshold;
        return this;
    }

    /* Parameter getters */

    public getThreshold () {
        return this.threshold;
    }

    /** The fitted centre of "normal" (the per-feature mean). */
    public getMean () {
        return new Matrix([this.mean.slice()]);
    }

    /* Private methods */

    private mahalanobis (row: number[]) {
        const difference = row.map((value, j) => value - this.mean[j]);
        const inverse = this.inverseCovariance;

        // squared distance = diffᵀ · Σ⁻¹ · diff
        let squared = 0;
        for (let i = 0; i < difference.length; i++) {
            let inner = 0;
            for (let j = 0; j < difference.length; j++) {
                inner += inverse[i][j] * difference[j];
            }
            squared += difference[i] * inner;
        }

        return Math.sqrt(Math.max(0, squared));
    }
}

function columnMeans (rows: number[][], d: number) {
    const means = new Array<number>(d).fill(0);
    for (const row of rows) {
        for (let j = 0; j < d; j++) {
            means[j] += row[j];
        }
    }
    return means.map(sum => (rows.length > 0 ? sum / rows.length : 0));
}

function covarianceMatrix (centered: number[][], d: number) {
    const n = centered.length;
    const covariance = Array.from({ length: d }, () => new Array<number>(d).fill(0));
    for (const row of centered) {
        for (let i = 0; i < d; i++) {
            for (let j = i; j < d; j++) {
                covariance[i][j] += row[i] * row[j];
            }
        }
    }
    const divisor = n > 1 ? n - 1 : 1;
    for (let i = 0; i < d; i++) {
        for (let j = i; j < d; j++) {
            covariance[i][j] /= divisor;
            covariance[j][i] = covariance[i][j];
        }
    }
    return covariance;
}
