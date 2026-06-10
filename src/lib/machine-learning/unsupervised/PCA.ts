import Matrix from "../../math/linear-algebra/Matrix";

/**
 * **Principal Component Analysis** — the library's dimensionality-reduction model. Clustering asks
 * *"which group?"*; PCA asks a different unsupervised question: *"what are the few directions along
 * which the data actually varies?"* Those directions — the **principal components** — let you
 * squeeze many correlated features down to a handful you can plot, with the least information lost.
 *
 * It works by finding the axes of greatest variance. Centre the data, build its covariance matrix,
 * and take that matrix's eigenvectors: the eigenvector with the largest eigenvalue is the direction
 * the data spreads most (the 1st component), the next-largest is the best direction perpendicular to
 * it, and so on. Keeping the top `k` and projecting onto them turns each `d`-dimensional row into a
 * `k`-dimensional one. The eigenvalues say how much variance each component captures — so you can
 * read off exactly how much you kept.
 *
 * Like the clustering models it takes inputs with **no targets**. `predict` returns the projection
 * (one row of `k` numbers per input); {@link reconstruct} maps a projection back to the original
 * space (lossily, unless `k` equals the feature count). Fully deterministic, with component signs
 * normalised so repeated fits agree.
 *
 * @example
 * const pca = new PCA().setNumberOfComponents(1);
 * pca.train(new Matrix([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]]));
 * pca.predict(new Matrix([[2, 2]]));        // ≈ [[2.83]]  — the single axis along the diagonal
 * pca.getExplainedVarianceRatio();          // ≈ [1]       — that one axis holds all the variance
 */
export default class PCA {

    private numberOfComponents = 2;

    private mean: number[] = [];
    private components: number[][] = [];   // top-k eigenvectors, one per row (k × d)
    private explainedVariance: number[] = []; // top-k eigenvalues
    private totalVariance = 0;             // sum of all eigenvalues

    public constructor () {}

    public train (inputs: Matrix) {
        const rows = inputs.toArray();
        const n = rows.length;
        const d = n > 0 ? rows[0].length : 0;

        this.mean = columnMeans(rows, d);
        const centered = rows.map(row => row.map((value, j) => value - this.mean[j]));

        // Covariance matrix: Cᵢⱼ = average product of centred features i and j (symmetric, d × d).
        const covariance = covarianceMatrix(centered, d);

        const { values, vectors } = jacobiEigen(covariance);

        // Sort eigenpairs by variance (eigenvalue), largest first.
        const order = values.map((value, index) => index).sort((a, b) => values[b] - values[a]);

        this.totalVariance = values.reduce((sum, value) => sum + Math.max(0, value), 0);

        const k = Math.min(this.numberOfComponents, d);
        this.components = [];
        this.explainedVariance = [];
        for (let c = 0; c < k; c++) {
            const index = order[c];
            this.components.push(normaliseSign(vectors.map(row => row[index]))); // eigenvector = column of V
            this.explainedVariance.push(Math.max(0, values[index]));
        }

        return this;
    }

    /** Projects each input onto the principal components: a `k`-dimensional row per input. */
    public predict (inputs: Matrix) {
        const rows = inputs.toArray();
        return new Matrix(rows.map(row => this.project(row)));
    }

    /**
     * Maps points back from component space to the original feature space. Round-tripping
     * `reconstruct(predict(x))`-style is lossy unless `numberOfComponents` equals the feature count —
     * the gap is exactly the information PCA discarded.
     */
    public reconstruct (inputs: Matrix) {
        const rows = inputs.toArray();
        return new Matrix(rows.map(row => this.project(row)).map(projection => this.unproject(projection)));
    }

    /* Parameter setters */

    public setNumberOfComponents (numberOfComponents: number) {
        this.numberOfComponents = numberOfComponents;
        return this;
    }

    /* Parameter getters */

    public getNumberOfComponents () {
        return this.numberOfComponents;
    }

    /** The principal components themselves — one eigenvector per row (`k × d`). */
    public getComponents () {
        return new Matrix(this.components.map(component => component.slice()));
    }

    /** Variance captured by each kept component (its eigenvalue). */
    public getExplainedVariance () {
        return this.explainedVariance.slice();
    }

    /** Fraction of the data's total variance captured by each kept component. */
    public getExplainedVarianceRatio () {
        if (this.totalVariance === 0) {
            return this.explainedVariance.map(() => 0);
        }
        return this.explainedVariance.map(value => value / this.totalVariance);
    }

    /** The per-feature mean subtracted before projecting. */
    public getMean () {
        return new Matrix([this.mean.slice()]);
    }

    /* Private methods */

    private project (row: number[]) {
        const centered = row.map((value, j) => value - this.mean[j]);
        return this.components.map(component => dot(centered, component));
    }

    private unproject (projection: number[]) {
        return this.mean.map((meanValue, j) => {
            let value = meanValue;
            for (let c = 0; c < this.components.length; c++) {
                value += projection[c] * this.components[c][j];
            }
            return value;
        });
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

/**
 * Eigendecomposition of a symmetric matrix by the **cyclic Jacobi** method: repeatedly apply a
 * plane rotation that zeroes the largest off-diagonal pair, until the matrix is (numerically)
 * diagonal. The diagonal then holds the eigenvalues and the accumulated rotations (`V`) hold the
 * eigenvectors as columns. Clear and reliable for the small, symmetric covariance matrices here.
 */
function jacobiEigen (input: number[][]) {
    const n = input.length;
    const a = input.map(row => row.slice());
    const v: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

    for (let sweep = 0; sweep < 100; sweep++) {
        let offDiagonal = 0;
        for (let p = 0; p < n; p++) {
            for (let q = p + 1; q < n; q++) {
                offDiagonal += a[p][q] * a[p][q];
            }
        }
        if (offDiagonal < 1e-20) {
            break;
        }

        for (let p = 0; p < n; p++) {
            for (let q = p + 1; q < n; q++) {
                if (a[p][q] === 0) {
                    continue;
                }

                const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
                const c = Math.cos(phi);
                const s = Math.sin(phi);

                const app = a[p][p];
                const aqq = a[q][q];
                const apq = a[p][q];

                a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                a[p][q] = 0;
                a[q][p] = 0;

                for (let i = 0; i < n; i++) {
                    if (i !== p && i !== q) {
                        const aip = a[i][p];
                        const aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[p][i] = a[i][p];
                        a[i][q] = s * aip + c * aiq;
                        a[q][i] = a[i][q];
                    }
                    const vip = v[i][p];
                    const viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }

    return { values: a.map((row, i) => row[i]), vectors: v };
}

/** Fix an eigenvector's sign so its largest-magnitude entry is positive — makes fits reproducible. */
function normaliseSign (vector: number[]) {
    let maxIndex = 0;
    for (let i = 1; i < vector.length; i++) {
        if (Math.abs(vector[i]) > Math.abs(vector[maxIndex])) {
            maxIndex = i;
        }
    }
    return vector[maxIndex] < 0 ? vector.map(value => -value) : vector;
}

function dot (a: number[], b: number[]) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
