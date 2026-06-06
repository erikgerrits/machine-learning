import { describe, it, expect } from 'vitest';
import LinearRegression from '../lib/machine-learning/supervised/LinearRegression';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { LINEAR_INPUTS, LINEAR_TARGETS } from './helpers/fixtures';

describe('LinearRegression', () => {

    it('converges to the underlying linear relationship (y = 1000 + 200x)', () => {
        const regression = new LinearRegression();
        regression.setNumberOfEpochs(10000).setLearningRate(0.02);

        regression.train(new Matrix(LINEAR_INPUTS), new Matrix(LINEAR_TARGETS));
        const predictions = regression.predict(new Matrix(LINEAR_INPUTS)).toArray();

        predictions.forEach((row, i) => {
            expect(Math.abs(row[0] - LINEAR_TARGETS[i][0])).toBeLessThan(1);
        });
    });

    it('predict applies the bias-enriched hypothesis directly', () => {
        const regression = new LinearRegression();
        // Hypothesis is [intercept, slope] against inputs enriched with a leading bias column.
        regression.setHypothesis(new Matrix([[1000], [200]]));

        const predictions = regression.predict(new Matrix([[5], [7]])).toArray();

        expect(predictions).toEqual([[2000], [2400]]);
    });

    it('resetHypothesis clears the learned weights', () => {
        const regression = new LinearRegression();
        regression.setNumberOfEpochs(100).train(new Matrix(LINEAR_INPUTS), new Matrix(LINEAR_TARGETS));

        expect(regression.getHypothesis()).toBeDefined();

        regression.resetHypothesis();

        expect(regression.getHypothesis()).toBeUndefined();
    });

    it('exposes its hyperparameters', () => {
        const regression = new LinearRegression();
        regression.setLearningRate(0.05).setNumberOfEpochs(42);

        expect(regression.getLearningRate()).toBe(0.05);
        expect(regression.getNumberOfEpochs()).toBe(42);
    });
});
