import { describe, it, expect } from 'vitest';
import ContextualBandit from '../lib/machine-learning/reinforcement/ContextualBandit';
import mulberry32 from '../lib/math/random/mulberry32';

// A world where the best arm DEPENDS on the customer. Two customer types, encoded as a context:
//   morning regular = [1, 0]   →   arm 0 (cinnamon roll) sells at 0.8, the rest at 0.2
//   evening tourist = [0, 1]   →   arm 1 (espresso tonic) sells at 0.8, the rest at 0.2
// No single arm is best for everyone, so a context-blind bandit can't do better than guessing which
// type showed up. A contextual bandit should learn to read the context and switch its pick.
const TYPES = [[1, 0], [0, 1]];
const TRUE_RATE = [
    [0.8, 0.2], // arm 0: great for type A, poor for type B
    [0.2, 0.8], // arm 1: poor for type A, great for type B
    [0.3, 0.3], // arm 2: a mediocre all-rounder
];

const rateOf = (arm: number, context: number[]) => TRUE_RATE[arm][context[1] === 1 ? 1 : 0];

/** Run the bandit against the seeded world for `steps` turns; returns the back-half hit rate (sells). */
function play(bandit: ContextualBandit, steps: number, seed = 7) {
    const rand = mulberry32(seed);
    let recentSales = 0;
    for (let t = 0; t < steps; t++) {
        const context = TYPES[Math.floor(rand() * TYPES.length)];
        const arm = bandit.selectArm(context);
        const sold = rand() < rateOf(arm, context) ? 1 : 0;
        bandit.update(arm, context, sold);
        if (t >= steps / 2) recentSales += sold; // measure once it has had time to learn
    }
    return recentSales / (steps / 2);
}

describe('ContextualBandit', () => {

    it('learns a per-context policy with LinUCB', () => {
        const bandit = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('linucb').setAlpha(1).setSeed(0);
        play(bandit, 3000);

        // The whole point: a DIFFERENT arm is chosen depending on who's at the counter.
        expect(bandit.selectArm([1, 0])).toBe(0);
        expect(bandit.selectArm([0, 1])).toBe(1);
    });

    it('recovers each arm\'s context-to-reward weights', () => {
        const bandit = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('linucb').setSeed(0);
        play(bandit, 4000);

        // θ_a · type ≈ that arm's true sell-rate for the type, so the weights track TRUE_RATE.
        expect(bandit.getWeights(0)[0]).toBeCloseTo(0.8, 1); // arm 0, feature for type A
        expect(bandit.getWeights(1)[1]).toBeCloseTo(0.8, 1); // arm 1, feature for type B
        expect(bandit.predict(0, [1, 0])).toBeCloseTo(0.8, 1);
        // Arm 0 strongly prefers type A. (It can't pin down its type-B reward — a good policy almost
        // never features arm 0 for type B, so that weight just stays near the ridge prior.)
        expect(bandit.predict(0, [1, 0])).toBeGreaterThan(bandit.predict(0, [0, 1]) + 0.4);
    });

    it('beats a context-blind bandit when the best arm flips with context', () => {
        const contextual = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('linucb').setSeed(1);
        const contextualScore = play(contextual, 3000);

        // A bandit fed a constant context can't tell the types apart — it can only chase one average
        // arm, capped near (0.8 + 0.2) / 2 = 0.5 because the winner keeps flipping under it.
        const blind = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('linucb').setSeed(1);
        const rand = mulberry32(7);
        let blindSales = 0;
        for (let t = 0; t < 3000; t++) {
            const context = TYPES[Math.floor(rand() * TYPES.length)];
            const arm = blind.selectArm([1, 1]);          // always the SAME context → context-blind
            const sold = rand() < rateOf(arm, context) ? 1 : 0;
            blind.update(arm, [1, 1], sold);
            if (t >= 1500) blindSales += sold;
        }

        expect(contextualScore).toBeGreaterThan(0.65);     // close to the 0.8 optimum
        expect(contextualScore).toBeGreaterThan(blindSales / 1500 + 0.1);
    });

    it('also learns a policy with epsilon-greedy', () => {
        const bandit = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('epsilon-greedy').setEpsilon(0.1).setSeed(0);
        play(bandit, 3000);

        expect(bandit.selectArm([1, 0])).toBe(0);
        expect(bandit.selectArm([0, 1])).toBe(1);
    });

    it('reports a shrinking uncertainty bonus as an arm gathers data', () => {
        const bandit = new ContextualBandit().setNumberOfArms(2).setContextDimensions(2).setStrategy('linucb').setSeed(0);
        const before = bandit.getConfidence(0, [1, 0]);
        for (let t = 0; t < 50; t++) bandit.update(0, [1, 0], 1);
        const after = bandit.getConfidence(0, [1, 0]);
        expect(after).toBeLessThan(before); // more evidence → narrower confidence
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const b = new ContextualBandit().setNumberOfArms(3).setContextDimensions(2).setStrategy('epsilon-greedy').setEpsilon(0.2).setSeed(3);
            play(b, 500);
            return [b.predict(0, [1, 0]), b.predict(1, [0, 1]), b.getCounts()];
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const bandit = new ContextualBandit();
        expect(bandit.getStrategy()).toBe('linucb');

        const returned = bandit.setNumberOfArms(5).setContextDimensions(4).setStrategy('epsilon-greedy').setAlpha(0.5).setEpsilon(0.05).setRegularization(2).setSeed(9);
        expect(returned).toBe(bandit);
        expect(bandit.getNumberOfArms()).toBe(5);
        expect(bandit.getContextDimensions()).toBe(4);
        expect(bandit.getStrategy()).toBe('epsilon-greedy');
        expect(bandit.getAlpha()).toBe(0.5);
        expect(bandit.getEpsilon()).toBe(0.05);
        expect(bandit.getRegularization()).toBe(2);
        expect(bandit.getSeed()).toBe(9);
    });
});
