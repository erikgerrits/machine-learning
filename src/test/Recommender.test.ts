import { describe, it, expect } from 'vitest';
import Recommender from '../lib/machine-learning/unsupervised/Recommender';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { RECOMMENDER_INPUTS } from './helpers/fixtures';

const trained = () => {
    const model = new Recommender().setNumberOfFactors(2).setNumberOfEpochs(500).setLearningRate(0.02).setRegularization(0.02).setSeed(0);
    model.train(new Matrix(RECOMMENDER_INPUTS));
    return model;
};

describe('Recommender', () => {

    it('fills hidden ratings in line with the user\'s taste group', () => {
        const predictions = trained().predict().toArray();

        // User 1 (group A) hid item 1 → should predict high; user 3 (group B) hid item 0 → low.
        expect(predictions[1][1]).toBeGreaterThan(3.5);
        expect(predictions[3][0]).toBeLessThan(2.5);
        // User 0's hidden item 3 (a group-B item) should come out low.
        expect(predictions[0][3]).toBeLessThan(2.5);
    });

    it('reconstructs the observed ratings closely', () => {
        const predictions = trained().predict().toArray();
        let error = 0;
        let count = 0;
        RECOMMENDER_INPUTS.forEach((row, u) => row.forEach((rating, i) => {
            if (rating !== 0) {
                error += Math.abs(predictions[u][i] - rating);
                count++;
            }
        }));
        expect(error / count).toBeLessThan(0.5); // mean absolute error on known ratings
    });

    it('recommends a user\'s unrated items, best first, and never a rated one', () => {
        const model = trained();
        const recommendations = model.recommend(1); // user 1 has rated items 0, 2, 3; item 1 is open

        expect(recommendations.map(r => r.item)).toEqual([1]); // only the unrated item
        // Scores are sorted descending.
        for (let i = 1; i < recommendations.length; i++) {
            expect(recommendations[i - 1].score).toBeGreaterThanOrEqual(recommendations[i].score);
        }
    });

    it('exposes latent factors of the right shape', () => {
        const model = trained();
        expect(model.getUserFactors().getRowCount()).toBe(4);
        expect(model.getUserFactors().getColumnCount()).toBe(2);
        expect(model.getItemFactors().getRowCount()).toBe(4);
        expect(model.getItemFactors().getColumnCount()).toBe(2);
    });

    it('is deterministic for a fixed seed', () => {
        expect(trained().predict().toArray()).toEqual(trained().predict().toArray());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new Recommender();
        expect(model.getNumberOfFactors()).toBe(2);

        const returned = model.setNumberOfFactors(3).setNumberOfEpochs(50).setLearningRate(0.05).setRegularization(0.1).setSeed(7);
        expect(returned).toBe(model);
        expect(model.getNumberOfFactors()).toBe(3);
        expect(model.getNumberOfEpochs()).toBe(50);
        expect(model.getLearningRate()).toBe(0.05);
        expect(model.getRegularization()).toBe(0.1);
        expect(model.getSeed()).toBe(7);
    });
});
