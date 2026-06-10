import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/** The kernels the machine can score similarity with — a straight dot product, or one that bends space. */
export type Kernel = "linear" | "rbf" | "polynomial";

/**
 * Binary **support vector machine**, trained with simplified SMO (sequential minimal optimization).
 *
 * Where logistic regression settles for *any* line that separates the two classes, an SVM hunts for
 * the **max-margin** one — the boundary with the widest empty street between it and the nearest
 * points. Only those nearest points, the **support vectors**, end up mattering; everything else
 * could be deleted and the boundary wouldn't budge. The "softness" of the margin is set by `C`: a
 * small `C` tolerates points inside the street (more slack, a wider, simpler boundary), a large `C`
 * insists on getting them right.
 *
 * The whole thing is phrased in terms of a **kernel** — a similarity score between two points. With
 * the `linear` kernel that's just the dot product and the boundary is a straight line. Swap in the
 * `rbf` (Gaussian) or `polynomial` kernel — the **kernel trick** — and the very same algorithm
 * carves curved boundaries no straight line could, without ever leaving the original features.
 *
 * Targets are a single column of `0`/`1` (like {@link LogisticRegression}); internally the classes
 * are ±1. `predict` returns the raw **decision score** `f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b` per row:
 * its sign is the predicted class (≥ 0 → class 1) and its magnitude is how far past the margin the
 * point sits. Training is deterministic for a fixed seed.
 */
export default class SupportVectorMachine {

    private kernel: Kernel = "linear";
    private regularization = 1;     // C — the soft-margin penalty for points inside/over the street
    private gamma = 1;              // RBF width
    private degree = 3;            // polynomial degree
    private coefficient = 1;      // polynomial constant term (coef0)
    private tolerance = 1e-3;     // KKT tolerance: how far a point may violate the margin before we act
    private numberOfIterations = 20; // SMO sweeps over the data per train() call
    private seed = 0;

    // Learned state. The model is fully described by a coefficient αᵢ per training example (non-zero
    // only for support vectors), the examples themselves, their ±1 labels, and the bias b.
    private alphas: number[];
    private bias = 0;
    private inputs: number[][];
    private labels: number[];
    private kernelMatrix: number[][]; // cached Gram matrix K[i][j] over the training set

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const rows = inputs.toArray();
        const labels = targets.toArray().map(row => (row[0] >= 0.5 ? 1 : -1));
        const exampleCount = rows.length;

        // Initialise (or re-initialise when the dataset size changes) the dual coefficients. Like the
        // regression models, repeated train() calls *continue* from where the last left off — handy
        // for animating the optimisation a few sweeps at a time.
        if (this.alphas === undefined || this.alphas.length !== exampleCount) {
            this.alphas = new Array(exampleCount).fill(0);
            this.bias = 0;
            this.kernelMatrix = undefined;
        }

        this.inputs = rows;
        this.labels = labels;

        if (this.kernelMatrix === undefined) {
            this.kernelMatrix = this.computeKernelMatrix(rows);
        }

        const random = mulberry32(this.seed);
        for (let iteration = 0; iteration < this.numberOfIterations; iteration++) {
            this.smoSweep(random);
        }

        return this;
    }

    public predict (inputs: Matrix) {
        return new Matrix(inputs.toArray().map(row => [this.score(row)]));
    }

    public reset () {
        this.alphas = undefined;
        this.bias = 0;
        this.kernelMatrix = undefined;
        return this;
    }

    /* Parameter setters */

    public setKernel (kernel: Kernel) {
        this.kernel = kernel;
        return this;
    }

    public setRegularization (regularization: number) {
        this.regularization = regularization;
        return this;
    }

    public setGamma (gamma: number) {
        this.gamma = gamma;
        return this;
    }

    public setDegree (degree: number) {
        this.degree = degree;
        return this;
    }

    public setCoefficient (coefficient: number) {
        this.coefficient = coefficient;
        return this;
    }

    public setTolerance (tolerance: number) {
        this.tolerance = tolerance;
        return this;
    }

    public setNumberOfIterations (numberOfIterations: number) {
        this.numberOfIterations = numberOfIterations;
        return this;
    }

    public setSeed (seed: number) {
        this.seed = seed;
        return this;
    }

    /* Parameter getters */

    public getKernel () {
        return this.kernel;
    }

    public getRegularization () {
        return this.regularization;
    }

    public getGamma () {
        return this.gamma;
    }

    public getDegree () {
        return this.degree;
    }

    public getCoefficient () {
        return this.coefficient;
    }

    public getTolerance () {
        return this.tolerance;
    }

    public getNumberOfIterations () {
        return this.numberOfIterations;
    }

    public getSeed () {
        return this.seed;
    }

    public getBias () {
        return this.bias;
    }

    /** Indices of the training rows that ended up as support vectors (a non-negligible αᵢ). */
    public getSupportVectorIndices () {
        if (this.alphas === undefined) {
            return [];
        }

        const indices: number[] = [];
        for (let i = 0; i < this.alphas.length; i++) {
            if (this.alphas[i] > 1e-6) {
                indices.push(i);
            }
        }
        return indices;
    }

    /**
     * Norm of the (implicit) weight vector, ‖w‖ = √(Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)). The geometric
     * margin — the half-width of the street — is `1 / ‖w‖`, so a *smaller* norm means a *wider*
     * margin. Even with a non-linear kernel this is well defined (it lives in the kernel's feature
     * space), it just stops being something you can draw as a single straight gap.
     */
    public getWeightNorm () {
        const supportVectors = this.getSupportVectorIndices();
        let sum = 0;
        for (const i of supportVectors) {
            for (const j of supportVectors) {
                sum += this.alphas[i] * this.alphas[j] * this.labels[i] * this.labels[j] * this.kernelMatrix[i][j];
            }
        }
        return Math.sqrt(Math.max(0, sum));
    }

    /* Private methods */

    /**
     * One SMO sweep. For each example whose margin violates the KKT conditions, pick a second
     * example, then jointly nudge that pair of αs the most the box constraints `0 ≤ α ≤ C` allow —
     * the smallest step the dual problem can take. Many such pairwise steps add up to the global
     * max-margin solution. (Simplified SMO, after Platt 1998 / the CS229 notes: a random partner
     * rather than the heuristic second-choice, which is plenty for teaching-scale data.)
     */
    private smoSweep (random: () => number) {
        const exampleCount = this.inputs.length;
        const alphas = this.alphas;
        const labels = this.labels;
        const kernelMatrix = this.kernelMatrix;
        const c = this.regularization;
        const tolerance = this.tolerance;

        for (let i = 0; i < exampleCount; i++) {
            const errorI = this.decision(i) - labels[i];

            // Does example i break the KKT conditions enough to be worth a step?
            if (!((labels[i] * errorI < -tolerance && alphas[i] < c) || (labels[i] * errorI > tolerance && alphas[i] > 0))) {
                continue;
            }

            // A random partner j ≠ i, drawn uniformly.
            let j = Math.floor(random() * (exampleCount - 1));
            if (j >= i) {
                j++;
            }

            const errorJ = this.decision(j) - labels[j];

            const alphaIOld = alphas[i];
            const alphaJOld = alphas[j];

            // The box the pair must stay inside (so both αs keep 0 ≤ α ≤ C while αᵢyᵢ + αⱼyⱼ is held).
            let low: number;
            let high: number;
            if (labels[i] !== labels[j]) {
                low = Math.max(0, alphaJOld - alphaIOld);
                high = Math.min(c, c + alphaJOld - alphaIOld);
            } else {
                low = Math.max(0, alphaIOld + alphaJOld - c);
                high = Math.min(c, alphaIOld + alphaJOld);
            }
            if (low === high) {
                continue;
            }

            // The second derivative of the objective along this pair; eta ≥ 0 means no proper minimum.
            const eta = 2 * kernelMatrix[i][j] - kernelMatrix[i][i] - kernelMatrix[j][j];
            if (eta >= 0) {
                continue;
            }

            // Step αⱼ to the unconstrained optimum, then clip it back into the box.
            let alphaJ = alphaJOld - (labels[j] * (errorI - errorJ)) / eta;
            alphaJ = Math.min(high, Math.max(low, alphaJ));
            if (Math.abs(alphaJ - alphaJOld) < 1e-5) {
                continue;
            }

            // αᵢ moves the opposite way to keep the constraint Σ αₖyₖ = const intact.
            const alphaI = alphaIOld + labels[i] * labels[j] * (alphaJOld - alphaJ);

            alphas[i] = alphaI;
            alphas[j] = alphaJ;

            // Re-centre the bias so the two just-moved points sit correctly against the margin.
            const biasI = this.bias - errorI - labels[i] * (alphaI - alphaIOld) * kernelMatrix[i][i] - labels[j] * (alphaJ - alphaJOld) * kernelMatrix[i][j];
            const biasJ = this.bias - errorJ - labels[i] * (alphaI - alphaIOld) * kernelMatrix[i][j] - labels[j] * (alphaJ - alphaJOld) * kernelMatrix[j][j];

            if (alphaI > 0 && alphaI < c) {
                this.bias = biasI;
            } else if (alphaJ > 0 && alphaJ < c) {
                this.bias = biasJ;
            } else {
                this.bias = (biasI + biasJ) / 2;
            }
        }
    }

    /** Decision score for training example i, reusing the cached kernel matrix. */
    private decision (i: number) {
        const alphas = this.alphas;
        const labels = this.labels;
        const kernelRow = this.kernelMatrix;

        let sum = this.bias;
        for (let k = 0; k < alphas.length; k++) {
            if (alphas[k] !== 0) {
                sum += alphas[k] * labels[k] * kernelRow[k][i];
            }
        }
        return sum;
    }

    /** Decision score for an arbitrary point — only the support vectors contribute. */
    private score (point: number[]) {
        const alphas = this.alphas;
        if (alphas === undefined) {
            return 0; // untrained: no boundary yet
        }

        let sum = this.bias;
        for (let k = 0; k < alphas.length; k++) {
            if (alphas[k] !== 0) {
                sum += alphas[k] * this.labels[k] * this.applyKernel(this.inputs[k], point);
            }
        }
        return sum;
    }

    private computeKernelMatrix (rows: number[][]) {
        const exampleCount = rows.length;
        const matrix: number[][] = [];
        for (let i = 0; i < exampleCount; i++) {
            matrix.push(new Array(exampleCount));
        }
        // K is symmetric, so only the upper triangle is computed and mirrored.
        for (let i = 0; i < exampleCount; i++) {
            for (let j = i; j < exampleCount; j++) {
                const value = this.applyKernel(rows[i], rows[j]);
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
        return matrix;
    }

    private applyKernel (a: number[], b: number[]) {
        switch (this.kernel) {
            case "rbf": {
                let squaredDistance = 0;
                for (let i = 0; i < a.length; i++) {
                    const difference = a[i] - b[i];
                    squaredDistance += difference * difference;
                }
                return Math.exp(-this.gamma * squaredDistance);
            }
            case "polynomial":
                return Math.pow(dotProduct(a, b) + this.coefficient, this.degree);
            case "linear":
            default:
                return dotProduct(a, b);
        }
    }
}

function dotProduct (a: number[], b: number[]) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
