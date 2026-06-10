import { describe, it, expect } from 'vitest';
import MultiArmedBandit from '../lib/machine-learning/reinforcement/MultiArmedBandit';
import mulberry32 from '../lib/math/random/mulberry32';

// Three café specials with hidden sell-rates; arm 2 is the clear winner.
const MEANS = [0.2, 0.5, 0.8];

/** Run the bandit against a seeded Bernoulli world (reward 1 = "it sold") for `steps` turns. */
function play(bandit: MultiArmedBandit, steps: number, envSeed = 99) {
    const envRandom = mulberry32(envSeed);
    let total = 0;
    for (let t = 0; t < steps; t++) {
        const arm = bandit.selectArm();
        const reward = envRandom() < MEANS[arm] ? 1 : 0;
        bandit.update(arm, reward);
        total += reward;
    }
    return total;
}

const argmax = (values: number[]) => values.indexOf(Math.max(...values));

describe('MultiArmedBandit', () => {

    it('epsilon-greedy learns the best arm and its value', () => {
        const bandit = new MultiArmedBandit().setNumberOfArms(3).setStrategy('epsilon-greedy').setEpsilon(0.1).setSeed(0);
        play(bandit, 3000);

        const values = bandit.getValues();
        expect(argmax(values)).toBe(2);                 // identifies the winner
        expect(values[2]).toBeCloseTo(MEANS[2], 1);     // its estimate ≈ the true 0.8
        expect(bandit.getCounts()[2]).toBeGreaterThan(bandit.getCounts()[0]); // and plays it most
    });

    it('UCB also homes in on the best arm', () => {
        const bandit = new MultiArmedBandit().setNumberOfArms(3).setStrategy('ucb').setConfidence(2).setSeed(0);
        play(bandit, 3000);

        expect(argmax(bandit.getValues())).toBe(2);
        const counts = bandit.getCounts();
        expect(counts[2]).toBeGreaterThan(counts[0]);
        expect(counts[2]).toBeGreaterThan(counts[1]);
    });

    it('beats a no-learning random policy on total reward', () => {
        const learner = new MultiArmedBandit().setNumberOfArms(3).setStrategy('ucb').setSeed(1);
        const learnerReward = play(learner, 2000);

        // A policy that always explores (epsilon = 1) just picks at random — the baseline to beat.
        const random = new MultiArmedBandit().setNumberOfArms(3).setStrategy('epsilon-greedy').setEpsilon(1).setSeed(1);
        const randomReward = play(random, 2000);

        expect(learnerReward).toBeGreaterThan(randomReward);
    });

    it('tries every arm before exploiting (UCB)', () => {
        const bandit = new MultiArmedBandit().setNumberOfArms(4).setStrategy('ucb').setSeed(0);
        const firstFour = [bandit.selectArm()];
        bandit.update(firstFour[0], 0);
        for (let i = 1; i < 4; i++) {
            const arm = bandit.selectArm();
            firstFour.push(arm);
            bandit.update(arm, 0);
        }
        expect(new Set(firstFour)).toEqual(new Set([0, 1, 2, 3])); // each arm once first
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const b = new MultiArmedBandit().setNumberOfArms(3).setEpsilon(0.2).setSeed(7);
            play(b, 500);
            return b.getValues();
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const bandit = new MultiArmedBandit();
        expect(bandit.getStrategy()).toBe('epsilon-greedy');

        const returned = bandit.setNumberOfArms(5).setStrategy('ucb').setEpsilon(0.05).setConfidence(1.5).setSeed(3);
        expect(returned).toBe(bandit);
        expect(bandit.getNumberOfArms()).toBe(5);
        expect(bandit.getStrategy()).toBe('ucb');
        expect(bandit.getEpsilon()).toBe(0.05);
        expect(bandit.getConfidence()).toBe(1.5);
        expect(bandit.getSeed()).toBe(3);
    });
});
