import { describe, it, expect } from 'vitest';
import DeepQNetwork from '../lib/machine-learning/reinforcement/DeepQNetwork';

// A continuous 1-D line: state is [x] with x in [0, 1]. Action 0 steps left, 1 steps right (by 0.12).
// Reaching x ≥ 0.85 pays +1 and ends the episode; every other step pays 0. The state is a real number,
// not a table index — so the network has to *generalise* a value across the whole line.
const STEP = 0.12;
const GOAL = 0.85;
function lineStep(x: number, action: number) {
    const next = Math.min(1, Math.max(0, x + (action === 1 ? STEP : -STEP)));
    return next >= GOAL ? { next: [next], reward: 1, done: true } : { next: [next], reward: 0, done: false };
}

function trainOnLine(dqn: DeepQNetwork, episodes: number, envSeed = 7) {
    let a = envSeed;
    const rand = () => (a = (a * 48271) % 2147483647) / 2147483647;
    for (let e = 0; e < episodes; e++) {
        let state = [rand() * 0.6];
        for (let t = 0; t < 30; t++) {
            const action = dqn.selectAction(state);
            const { next, reward, done } = lineStep(state[0], action);
            dqn.observe(state, action, reward, next, done);
            if (done) break;
            state = next;
        }
    }
}

describe('DeepQNetwork', () => {

    it('learns to move toward the goal and values states by closeness to it', () => {
        const dqn = new DeepQNetwork().setStateSize(1).setNumberOfActions(2).setHiddenSizes([16, 16])
            .setLearningRate(0.2).setDiscountFactor(0.9).setEpsilon(0.3).setSeed(0);
        trainOnLine(dqn, 800);

        // From anywhere short of the goal, the learned move is "right" (toward it).
        expect(dqn.bestAction([0.3])).toBe(1);
        expect(dqn.bestAction([0.55])).toBe(1);
        // And the value rises as you near the goal.
        expect(dqn.getValue([0.75])).toBeGreaterThan(dqn.getValue([0.15]) + 0.2);
    });

    it('gives a value for any state, including ones it never visited', () => {
        const dqn = new DeepQNetwork().setStateSize(1).setNumberOfActions(2).setSeed(0);

        // The network is defined everywhere — a fresh agent already answers for an arbitrary state.
        const q = dqn.getQValues([0.123]);
        expect(q.length).toBe(2);
        expect(q.every(v => Number.isFinite(v))).toBe(true);

        // Batched lookups match the per-state ones (used to paint a value heatmap in one pass).
        const batch = dqn.getQValuesBatch([[0.1], [0.42], [0.9]]);
        expect(batch.length).toBe(3);
        expect(batch[1]).toEqual(dqn.getQValues([0.42]));
    });

    it('waits for a full minibatch before learning, then steps', () => {
        const dqn = new DeepQNetwork().setStateSize(1).setNumberOfActions(2).setBatchSize(8).setSeed(0);
        for (let i = 0; i < 7; i++) dqn.observe([i / 10], i % 2, 0, [(i + 1) / 10], false);
        expect(dqn.getLearnSteps()).toBe(0); // only 7 transitions buffered — no batch yet

        for (let i = 0; i < 20; i++) dqn.observe([i / 10], i % 2, 0, [(i + 1) / 10], false);
        expect(dqn.getLearnSteps()).toBeGreaterThan(0);
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const dqn = new DeepQNetwork().setStateSize(1).setNumberOfActions(2).setBatchSize(8).setSeed(3);
            for (let i = 0; i < 200; i++) dqn.observe([(i % 9) / 10], i % 2, i % 9 === 8 ? 1 : 0, [((i % 9) + 1) / 10], i % 9 === 8);
            return dqn.getQValues([0.4]);
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const dqn = new DeepQNetwork();
        expect(dqn.getDiscountFactor()).toBe(0.95);

        const returned = dqn.setStateSize(3).setNumberOfActions(5).setHiddenSizes([8, 8]).setLearningRate(0.01)
            .setDiscountFactor(0.8).setEpsilon(0.05).setBatchSize(16).setReplayCapacity(1000).setTargetSyncInterval(100).setSeed(9);
        expect(returned).toBe(dqn);
        expect(dqn.getStateSize()).toBe(3);
        expect(dqn.getNumberOfActions()).toBe(5);
        expect(dqn.getHiddenSizes()).toEqual([8, 8]);
        expect(dqn.getLearningRate()).toBe(0.01);
        expect(dqn.getDiscountFactor()).toBe(0.8);
        expect(dqn.getEpsilon()).toBe(0.05);
        expect(dqn.getBatchSize()).toBe(16);
        expect(dqn.getReplayCapacity()).toBe(1000);
        expect(dqn.getTargetSyncInterval()).toBe(100);
        expect(dqn.getSeed()).toBe(9);
    });
});
