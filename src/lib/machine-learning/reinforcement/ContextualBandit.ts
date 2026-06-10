import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/** How the contextual bandit trades off acting on its current model (exploit) against gathering
 *  information where it's least certain (explore). */
export type ContextualStrategy = "linucb" | "epsilon-greedy";

/**
 * A **contextual bandit** — the step up from the plain {@link MultiArmedBandit}. There, one arm was
 * best for everyone; here the best arm *depends on the situation*. Before each choice the world hands
 * you a **context** — a feature vector describing who's at the counter (regular or tourist, morning or
 * evening, sweet tooth or not) — and the reward an arm pays now depends on that context. So a single
 * running average per arm is too coarse: the cinnamon roll might delight the morning crowd and bore
 * the evening one.
 *
 * The fix is to give **each arm its own little linear model** that predicts its reward from the
 * context, and to learn those models online from the outcomes you observe. This is the classic
 * **LinUCB** algorithm: each arm keeps a ridge-regression fit `θ_a` (reward ≈ `θ_a · context`), and
 * picks are guided by both the prediction *and* how uncertain it is:
 *
 * - **linucb** — choose the arm with the highest `θ_a · x + α · √(xᵀ A_a⁻¹ x)`: its predicted reward
 *   plus an **optimism bonus** that's large where the arm has seen little data *like this context*.
 *   It explores deliberately, aiming curiosity at the gaps in what it knows.
 * - **epsilon-greedy** — use the same per-arm prediction `θ_a · x`, but explore by picking a random
 *   arm with probability `epsilon` instead. Simpler, blind exploration.
 *
 * Like the multi-armed bandit, it learns **online** — there is no `train(inputs, targets)`. You loop
 * {@link selectArm}(context) → observe a reward → {@link update}(arm, context, reward). Seeded for
 * reproducibility.
 *
 * @example
 * const bandit = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('linucb');
 * for (let t = 0; t < 2000; t++) {
 *   const context = describeCustomer();   // e.g. [isMorning, sweetTooth]
 *   const arm = bandit.selectArm(context);
 *   const reward = offer(arm, context);    // 1 = they took it, 0 = they passed
 *   bandit.update(arm, context, reward);
 * }
 */
export default class ContextualBandit {

    private numberOfArms = 3;
    private contextDimensions = 2;
    private strategy: ContextualStrategy = "linucb";
    private alpha = 1;            // LinUCB exploration weight (how much the uncertainty bonus counts)
    private epsilon = 0.1;        // ε-greedy exploration rate
    private regularization = 1;   // ridge λ — each arm's A starts as λ·I, keeping it invertible
    private seed = 0;

    private A: Matrix[];          // per arm: λ·I + Σ x·xᵀ  (d×d)
    private b: Matrix[];          // per arm: Σ reward·x      (d×1)
    private counts: number[];
    private totalSteps = 0;
    private random: () => number;

    public constructor () {}

    /** Choose an arm for this context, according to the strategy. */
    public selectArm (context: number[]) {
        this.ensureInitialized();
        const x = Matrix.columnVector(context);

        if (this.strategy === "epsilon-greedy" && this.random() < this.epsilon) {
            return Math.floor(this.random() * this.numberOfArms); // explore: a random arm
        }

        let best = 0;
        let bestScore = -Infinity;
        for (let arm = 0; arm < this.numberOfArms; arm++) {
            const inverse = this.A[arm].getInverse();
            const theta = Matrix.multiply(inverse, this.b[arm]); // ridge weights θ_a = A_a⁻¹ b_a
            let score = dot(theta, x);                           // predicted reward θ_a · x
            if (this.strategy === "linucb") {
                // Optimism bonus: α·√(xᵀ A_a⁻¹ x) — wide where this arm has little data like x.
                score += this.alpha * Math.sqrt(Math.max(0, dot(Matrix.multiply(inverse, x), x)));
            }
            if (score > bestScore) {
                bestScore = score;
                best = arm;
            }
        }
        return best;
    }

    /** Fold one observed (context, reward) outcome into the chosen arm's model. */
    public update (arm: number, context: number[], reward: number) {
        this.ensureInitialized();
        const x = Matrix.columnVector(context);
        this.A[arm] = Matrix.add(this.A[arm], Matrix.multiply(x, Matrix.transpose(x))); // A += x·xᵀ
        this.b[arm] = Matrix.add(this.b[arm], Matrix.multiply(x, reward));              // b += reward·x
        this.counts[arm]++;
        this.totalSteps++;
        return this;
    }

    public reset () {
        this.A = undefined;
        this.b = undefined;
        this.counts = undefined;
        this.totalSteps = 0;
        this.random = undefined;
        return this;
    }

    /* Parameter setters */

    public setNumberOfArms (numberOfArms: number) { this.numberOfArms = numberOfArms; return this; }
    public setContextDimensions (contextDimensions: number) { this.contextDimensions = contextDimensions; return this; }
    public setStrategy (strategy: ContextualStrategy) { this.strategy = strategy; return this; }
    public setAlpha (alpha: number) { this.alpha = alpha; return this; }
    public setEpsilon (epsilon: number) { this.epsilon = epsilon; return this; }
    public setRegularization (regularization: number) { this.regularization = regularization; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getNumberOfArms () { return this.numberOfArms; }
    public getContextDimensions () { return this.contextDimensions; }
    public getStrategy () { return this.strategy; }
    public getAlpha () { return this.alpha; }
    public getEpsilon () { return this.epsilon; }
    public getRegularization () { return this.regularization; }
    public getSeed () { return this.seed; }

    /** An arm's learned weight vector θ_a — how strongly it expects each context feature to lift its reward. */
    public getWeights (arm: number) {
        this.ensureInitialized();
        return columnToArray(Matrix.multiply(this.A[arm].getInverse(), this.b[arm]));
    }

    /** An arm's predicted reward θ_a · context for a given context. */
    public predict (arm: number, context: number[]) {
        this.ensureInitialized();
        const theta = Matrix.multiply(this.A[arm].getInverse(), this.b[arm]);
        return dot(theta, Matrix.columnVector(context));
    }

    /** An arm's LinUCB uncertainty width α·√(xᵀ A_a⁻¹ x) for a context — the optimism bonus driving exploration. */
    public getConfidence (arm: number, context: number[]) {
        this.ensureInitialized();
        const x = Matrix.columnVector(context);
        const inverse = this.A[arm].getInverse();
        return this.alpha * Math.sqrt(Math.max(0, dot(Matrix.multiply(inverse, x), x)));
    }

    /** How many times each arm has been played. */
    public getCounts () { this.ensureInitialized(); return this.counts.slice(); }
    /** Total interactions so far. */
    public getTotalSteps () { return this.totalSteps; }

    /* Private methods */

    private ensureInitialized () {
        const d = this.contextDimensions;
        if (this.A === undefined || this.A.length !== this.numberOfArms || this.A[0].getRowCount() !== d) {
            this.A = [];
            this.b = [];
            for (let arm = 0; arm < this.numberOfArms; arm++) {
                this.A.push(Matrix.multiply(Matrix.identity(d), this.regularization)); // λ·I, always invertible
                this.b.push(Matrix.zeros(d, 1));
            }
            this.counts = new Array<number>(this.numberOfArms).fill(0);
            this.totalSteps = 0;
            this.random = mulberry32(this.seed);
        }
    }
}

/** Dot product of two column vectors (d×1 matrices). */
function dot (a: Matrix, b: Matrix): number {
    let sum = 0;
    for (let i = 0; i < a.getRowCount(); i++) {
        sum += a.getElement(i, 0) * b.getElement(i, 0);
    }
    return sum;
}

function columnToArray (column: Matrix): number[] {
    const values: number[] = [];
    for (let i = 0; i < column.getRowCount(); i++) {
        values.push(column.getElement(i, 0));
    }
    return values;
}
