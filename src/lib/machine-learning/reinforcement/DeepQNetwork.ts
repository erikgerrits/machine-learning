import Matrix from "../../math/linear-algebra/Matrix";
import FeedforwardNeuralNetwork from "../supervised/FeedforwardNeuralNetwork";
import mulberry32 from "../../math/random/mulberry32";

interface Transition {
    state: number[];
    action: number;
    reward: number;
    nextState: number[];
    done: boolean;
}

/**
 * A **Deep Q-Network (DQN)** — the close of the reinforcement-learning arc, and the bridge back to
 * the neural networks of Part 4. Tabular {@link QLearning} kept one number per (state, action), which
 * only works when you can *enumerate* the states. The moment a state is a vector of real numbers — a
 * position on a smooth floor, the stock levels across a chain of cafés — that table can't be built.
 *
 * DQN's fix: stop storing Q and start **predicting** it. A neural network maps a state vector to a
 * Q-value for every action, so it **generalises** — it gives sensible values for states it has never
 * visited, by interpolating from ones it has. It's the same value idea as Chapter 26, with a function
 * approximator in place of the lookup table. Two tricks (from DeepMind's Atari work) keep the
 * bootstrapping from spiralling:
 *
 * - **Experience replay** — every transition is stored in a buffer, and learning happens on *random
 *   minibatches* drawn from it. This breaks the correlation between consecutive steps and lets each
 *   experience teach more than once.
 * - **A target network** — a periodically-frozen copy of the network supplies the `maxₐ′ Q(s′, a′)`
 *   in the update target. Chasing a fixed target for a while, rather than one that shifts every step,
 *   is far more stable.
 *
 * The network has a sigmoid output, so it represents values in `(0, 1)` — design rewards to stay in
 * that range (e.g. `+1` at the goal, `0` elsewhere, discounted by `γ < 1`). Learns **online**: loop
 * {@link selectAction} → step the world → {@link observe}. Seeded for reproducibility.
 *
 * @example
 * const dqn = new DeepQNetwork().setStateSize(2).setNumberOfActions(4).setDiscountFactor(0.9);
 * let state = env.start();
 * for (let step = 0; step < 20000; step++) {
 *   const action = dqn.selectAction(state);
 *   const { nextState, reward, done } = env.step(state, action);
 *   dqn.observe(state, action, reward, nextState, done);  // remember + learn from a minibatch
 *   state = done ? env.start() : nextState;
 * }
 * dqn.getValue([0.5, 0.5]);   // the network's value for any state — even unseen ones
 */
export default class DeepQNetwork {

    private stateSize = 2;
    private numberOfActions = 4;
    private hiddenSizes = [24, 24];
    private learningRate = 0.05;
    private discountFactor = 0.95;
    private epsilon = 0.1;
    private batchSize = 32;
    private replayCapacity = 5000;
    private targetSyncInterval = 250; // learning steps between freezes of the target network
    private seed = 0;

    private online: FeedforwardNeuralNetwork;
    private target: FeedforwardNeuralNetwork;
    private buffer: Transition[];
    private writeIndex = 0;
    private learnSteps = 0;
    private random: () => number;

    public constructor () {}

    /** Choose an action in `state`: explore at random with probability ε, else act greedily. */
    public selectAction (state: number[]) {
        this.ensureInitialized();
        if (this.random() < this.epsilon) {
            return Math.floor(this.random() * this.numberOfActions);
        }
        return this.bestAction(state);
    }

    /** The greedy action — argmax of the network's predicted Q-values for `state`. */
    public bestAction (state: number[]) {
        return argmax(this.getQValues(state));
    }

    /** The network's predicted Q-value for every action in `state`. */
    public getQValues (state: number[]) {
        this.ensureInitialized();
        return this.online.predict(new Matrix([state])).toArray()[0];
    }

    /** A state's value V(s) = maxₐ Q(s, a) under the current network. */
    public getValue (state: number[]) {
        const q = this.getQValues(state);
        return q[argmax(q)];
    }

    /** Q-values for many states at once, in a single forward pass — handy for sweeping a whole grid. */
    public getQValuesBatch (states: number[][]) {
        this.ensureInitialized();
        return this.online.predict(new Matrix(states)).toArray();
    }

    /** Remember one transition and take a learning step on a replayed minibatch. */
    public observe (state: number[], action: number, reward: number, nextState: number[], done: boolean) {
        this.ensureInitialized();
        this.remember({ state, action, reward, nextState, done });
        this.learn();
        return this;
    }

    public reset () {
        this.online = undefined;
        this.target = undefined;
        this.buffer = undefined;
        this.writeIndex = 0;
        this.learnSteps = 0;
        this.random = undefined;
        return this;
    }

    /* Parameter setters */

    public setStateSize (stateSize: number) { this.stateSize = stateSize; return this; }
    public setNumberOfActions (numberOfActions: number) { this.numberOfActions = numberOfActions; return this; }
    public setHiddenSizes (hiddenSizes: number[]) { this.hiddenSizes = hiddenSizes; return this; }
    public setLearningRate (learningRate: number) { this.learningRate = learningRate; return this; }
    public setDiscountFactor (discountFactor: number) { this.discountFactor = discountFactor; return this; }
    public setEpsilon (epsilon: number) { this.epsilon = epsilon; return this; }
    public setBatchSize (batchSize: number) { this.batchSize = batchSize; return this; }
    public setReplayCapacity (replayCapacity: number) { this.replayCapacity = replayCapacity; return this; }
    public setTargetSyncInterval (targetSyncInterval: number) { this.targetSyncInterval = targetSyncInterval; return this; }
    public setSeed (seed: number) { this.seed = seed; return this; }

    /* Parameter getters */

    public getStateSize () { return this.stateSize; }
    public getNumberOfActions () { return this.numberOfActions; }
    public getHiddenSizes () { return this.hiddenSizes.slice(); }
    public getLearningRate () { return this.learningRate; }
    public getDiscountFactor () { return this.discountFactor; }
    public getEpsilon () { return this.epsilon; }
    public getBatchSize () { return this.batchSize; }
    public getReplayCapacity () { return this.replayCapacity; }
    public getTargetSyncInterval () { return this.targetSyncInterval; }
    public getSeed () { return this.seed; }
    /** How many learning steps (replayed minibatches) have been taken. */
    public getLearnSteps () { return this.learnSteps; }

    /* Private methods */

    private ensureInitialized () {
        if (this.online !== undefined) return;
        const layers = [this.stateSize, ...this.hiddenSizes, this.numberOfActions];
        this.online = new FeedforwardNeuralNetwork(layers, this.seed).setLearningRate(this.learningRate).setNumberOfEpochs(1).setBatchSize(0);
        this.target = new FeedforwardNeuralNetwork(layers, this.seed).setLearningRate(this.learningRate).setNumberOfEpochs(1).setBatchSize(0);
        this.target.setWeightMatrices(this.online.getWeightMatrices());
        this.buffer = [];
        this.writeIndex = 0;
        this.learnSteps = 0;
        this.random = mulberry32(this.seed);
    }

    private remember (transition: Transition) {
        if (this.buffer.length < this.replayCapacity) {
            this.buffer.push(transition);
        } else {
            this.buffer[this.writeIndex] = transition;          // ring buffer: overwrite the oldest
            this.writeIndex = (this.writeIndex + 1) % this.replayCapacity;
        }
    }

    private learn () {
        if (this.buffer.length < this.batchSize) return; // wait until there's a minibatch to draw

        const batch: Transition[] = [];
        for (let i = 0; i < this.batchSize; i++) {
            batch.push(this.buffer[Math.floor(this.random() * this.buffer.length)]);
        }

        const states = batch.map(t => t.state);
        const nextStates = batch.map(t => t.nextState);
        // Start the targets from the network's current outputs, so untaken actions carry zero error,
        // then overwrite the taken action with its TD target r + γ·maxₐ′ Q_target(s′, a′).
        const targets = this.online.predict(new Matrix(states)).toArray();
        const nextQ = this.target.predict(new Matrix(nextStates)).toArray();

        for (let i = 0; i < this.batchSize; i++) {
            const t = batch[i];
            const future = t.done ? 0 : nextQ[i][argmax(nextQ[i])];
            targets[i][t.action] = t.reward + this.discountFactor * future;
        }

        this.online.train(new Matrix(states), new Matrix(targets));
        this.learnSteps++;

        if (this.learnSteps % this.targetSyncInterval === 0) {
            this.target.setWeightMatrices(this.online.getWeightMatrices()); // freeze a fresh target
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
