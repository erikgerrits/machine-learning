import mulberry32 from "../../math/random/mulberry32";

/** How the bandit trades off trying new arms (explore) against playing its current best (exploit). */
export type BanditStrategy = "epsilon-greedy" | "ucb";

/**
 * A **multi-armed bandit** — the gentle entry into reinforcement learning, and the first model here
 * that *acts* instead of predicting from a fixed dataset. Picture a row of slot machines (or daily
 * café specials): each "arm" pays out from its own unknown reward distribution, and on every turn you
 * must choose one. The catch is the **explore/exploit dilemma** — play the arm that's looked best so
 * far (exploit) and you might be ignoring a better one you've barely tried (explore).
 *
 * Unlike every other model in this library, there is no `train(inputs, targets)`: a bandit learns
 * **online**, one interaction at a time. You loop {@link selectArm} → observe a reward from the world
 * → {@link update}. Each arm keeps a running average of the rewards it has returned; the strategy
 * decides how much to gamble on the under-explored ones:
 *
 * - **epsilon-greedy** — play the best-estimated arm, but with probability `epsilon` pick one at
 *   random instead. Simple, and never quite stops exploring.
 * - **ucb** (upper confidence bound) — add an "optimism" bonus that's large for rarely-tried arms,
 *   so it explores deliberately (the less-certain an arm, the more it's worth a look) and tapers off
 *   as evidence accumulates.
 *
 * Seeded for reproducibility.
 *
 * @example
 * const bandit = new MultiArmedBandit().setNumberOfArms(3).setStrategy('ucb');
 * for (let t = 0; t < 1000; t++) {
 *   const arm = bandit.selectArm();
 *   const reward = pull(arm);        // ask the world (1 = the special sold, 0 = it didn't)
 *   bandit.update(arm, reward);
 * }
 * bandit.getValues();                // each arm's learned average reward
 */
export default class MultiArmedBandit {

    private numberOfArms = 4;
    private strategy: BanditStrategy = "epsilon-greedy";
    private epsilon = 0.1;
    private confidence = 2; // UCB exploration weight
    private seed = 0;

    private counts: number[];
    private values: number[];
    private totalSteps = 0;
    private random: () => number;

    public constructor () {}

    /** Choose an arm to play this turn, according to the strategy. */
    public selectArm () {
        this.ensureInitialized();
        return this.strategy === "ucb" ? this.selectUcb() : this.selectEpsilonGreedy();
    }

    /** Record the reward the chosen arm returned, folding it into that arm's running average. */
    public update (arm: number, reward: number) {
        this.ensureInitialized();
        this.counts[arm]++;
        this.totalSteps++;
        // Incremental mean: newAverage = old + (reward − old) / count.
        this.values[arm] += (reward - this.values[arm]) / this.counts[arm];
        return this;
    }

    public reset () {
        this.counts = undefined;
        this.values = undefined;
        this.totalSteps = 0;
        this.random = undefined;
        return this;
    }

    /* Parameter setters */

    public setNumberOfArms (numberOfArms: number) { this.numberOfArms = numberOfArms; return this; }
    public setStrategy (strategy: BanditStrategy) { this.strategy = strategy; return this; }
    public setEpsilon (epsilon: number) { this.epsilon = epsilon; return this; }
    public setConfidence (confidence: number) { this.confidence = confidence; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getNumberOfArms () { return this.numberOfArms; }
    public getStrategy () { return this.strategy; }
    public getEpsilon () { return this.epsilon; }
    public getConfidence () { return this.confidence; }
    public getSeed () { return this.seed; }

    /** Each arm's estimated average reward (its running mean of observed rewards). */
    public getValues () { this.ensureInitialized(); return this.values.slice(); }
    /** How many times each arm has been played. */
    public getCounts () { this.ensureInitialized(); return this.counts.slice(); }
    /** Total interactions so far. */
    public getTotalSteps () { return this.totalSteps; }

    /* Private methods */

    private ensureInitialized () {
        if (this.counts === undefined || this.counts.length !== this.numberOfArms) {
            this.counts = new Array<number>(this.numberOfArms).fill(0);
            this.values = new Array<number>(this.numberOfArms).fill(0);
            this.totalSteps = 0;
            this.random = mulberry32(this.seed);
        }
    }

    private selectEpsilonGreedy () {
        if (this.random() < this.epsilon) {
            return Math.floor(this.random() * this.numberOfArms); // explore: a random arm
        }
        return argmax(this.values); // exploit: the best-looking arm
    }

    private selectUcb () {
        // Any never-tried arm is infinitely optimistic — try each at least once first.
        for (let arm = 0; arm < this.numberOfArms; arm++) {
            if (this.counts[arm] === 0) {
                return arm;
            }
        }
        const logStep = Math.log(Math.max(1, this.totalSteps));
        let best = 0;
        let bestScore = -Infinity;
        for (let arm = 0; arm < this.numberOfArms; arm++) {
            const bonus = this.confidence * Math.sqrt(logStep / this.counts[arm]);
            const score = this.values[arm] + bonus;
            if (score > bestScore) {
                bestScore = score;
                best = arm;
            }
        }
        return best;
    }
}

function argmax (values: number[]) {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}
