import mulberry32 from "../../math/random/mulberry32";

/**
 * **Tabular Q-learning** — the leap from bandits to full reinforcement learning. A bandit's reward
 * depended only on the arm it pulled *now*; here actions have **delayed consequences**. The world is a
 * **Markov decision process**: you're in a **state**, you take an **action**, you get a **reward** and
 * land in a **new state**, and a choice now reshapes the rewards far down the line (a move toward the
 * goal, a restock that empties tomorrow's shelf). The job is to learn a **policy** — which action to
 * take in each state — that maximises the *total discounted reward* over time, not just the next step.
 *
 * Q-learning estimates the **action-value** `Q(s, a)`: the long-run reward of taking action `a` in
 * state `s` and then acting well thereafter. It keeps a table of these and refines it from raw
 * experience with one update per transition — the **temporal-difference** rule:
 *
 * ```
 * Q(s, a) ← Q(s, a) + α · [ r + γ · maxₐ′ Q(s′, a′) − Q(s, a) ]
 * ```
 *
 * The bracket is the **TD error**: the gap between what we thought `Q(s, a)` was and a fresher
 * estimate — the reward we just saw plus the (discounted) value of the best move available in the
 * state we landed in. Bootstrapping off `maxₐ′ Q(s′, a′)` is what carries value **backward** from
 * rewards through the states that lead to them. `γ` (the **discount factor**) sets how much the future
 * counts; `α` is the learning rate. It's **model-free** (no map of the world needed) and **off-policy**
 * (it learns the optimal greedy policy even while exploring randomly).
 *
 * Like the bandits, it learns **online** — no `train(inputs, targets)`. You loop {@link selectAction}
 * → step the environment → {@link update}. Seeded for reproducibility.
 *
 * @example
 * const agent = new QLearning().setNumberOfStates(16).setNumberOfActions(4).setDiscountFactor(0.95);
 * let state = env.start();
 * for (let step = 0; step < 10000; step++) {
 *   const action = agent.selectAction(state);
 *   const { nextState, reward, done } = env.step(state, action);
 *   agent.update(state, action, reward, nextState, done);
 *   state = done ? env.start() : nextState;   // new episode when one ends
 * }
 * agent.getPolicy();                           // the learned best action for every state
 */
export default class QLearning {

    private numberOfStates = 1;
    private numberOfActions = 1;
    private learningRate = 0.1;     // α — how far each update moves toward the new estimate
    private discountFactor = 0.95;   // γ — how much future reward counts vs. immediate
    private epsilon = 0.1;           // exploration rate for ε-greedy action selection
    private seed = 0;

    private q: number[][];           // q[state][action] — the action-value table
    private random: () => number;

    public constructor () {}

    /** Choose an action in `state`: explore at random with probability ε, otherwise act greedily. */
    public selectAction (state: number) {
        this.ensureInitialized();
        if (this.random() < this.epsilon) {
            return Math.floor(this.random() * this.numberOfActions); // explore
        }
        return this.bestAction(state); // exploit the current best estimate
    }

    /** The greedy action in `state` — the learned policy's choice (no exploration). */
    public bestAction (state: number) {
        this.ensureInitialized();
        return argmax(this.q[state]);
    }

    /**
     * Fold one observed transition into the table via the temporal-difference rule. Pass `done = true`
     * for a transition into a terminal state, so the future term is dropped (nothing comes after).
     */
    public update (state: number, action: number, reward: number, nextState: number, done = false) {
        this.ensureInitialized();
        const futureValue = done ? 0 : this.q[nextState][argmax(this.q[nextState])];
        const target = reward + this.discountFactor * futureValue;
        this.q[state][action] += this.learningRate * (target - this.q[state][action]);
        return this;
    }

    public reset () {
        this.q = undefined;
        this.random = undefined;
        return this;
    }

    /* Parameter setters */

    public setNumberOfStates (numberOfStates: number) { this.numberOfStates = numberOfStates; return this; }
    public setNumberOfActions (numberOfActions: number) { this.numberOfActions = numberOfActions; return this; }
    public setLearningRate (learningRate: number) { this.learningRate = learningRate; return this; }
    public setDiscountFactor (discountFactor: number) { this.discountFactor = discountFactor; return this; }
    public setEpsilon (epsilon: number) { this.epsilon = epsilon; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getNumberOfStates () { return this.numberOfStates; }
    public getNumberOfActions () { return this.numberOfActions; }
    public getLearningRate () { return this.learningRate; }
    public getDiscountFactor () { return this.discountFactor; }
    public getEpsilon () { return this.epsilon; }
    public getSeed () { return this.seed; }

    /** Every action's Q-value in a state. */
    public getQValues (state: number) { this.ensureInitialized(); return this.q[state].slice(); }
    /** A single Q(state, action). */
    public getQValue (state: number, action: number) { this.ensureInitialized(); return this.q[state][action]; }
    /** A state's value V(s) = maxₐ Q(s, a) — how good it is to be there under the learned policy. */
    public getValue (state: number) { this.ensureInitialized(); return this.q[state][argmax(this.q[state])]; }
    /** The greedy policy: the best action for every state. */
    public getPolicy () { this.ensureInitialized(); return this.q.map(row => argmax(row)); }

    /* Private methods */

    private ensureInitialized () {
        if (this.q === undefined || this.q.length !== this.numberOfStates || this.q[0].length !== this.numberOfActions) {
            this.q = Array.from({ length: this.numberOfStates }, () => new Array<number>(this.numberOfActions).fill(0));
            this.random = mulberry32(this.seed);
        }
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
