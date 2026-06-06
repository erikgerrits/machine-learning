import { describe, it, expect } from 'vitest';
import LogisticRegression from '../lib/machine-learning/supervised/LogisticRegression';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { LOGISTIC_INPUTS, LOGISTIC_TARGETS, LOGISTIC_EXPECTED_CLASSES } from './helpers/fixtures';

describe('LogisticRegression', () => {

    it('classifies the training set correctly', () => {
        const regression = new LogisticRegression();
        regression.setNumberOfEpochs(1000).setLearningRate(0.01);

        regression.train(new Matrix(LOGISTIC_INPUTS), new Matrix(LOGISTIC_TARGETS));
        const predictions = regression.predict(new Matrix(LOGISTIC_INPUTS)).toArray();

        predictions.forEach((row, i) => {
            const predictedClass = row[0] >= 0.5 ? 1 : 0;
            expect(predictedClass).toBe(LOGISTIC_EXPECTED_CLASSES[i]);
        });
    });

    it('produces probabilities in the (0, 1] range', () => {
        const regression = new LogisticRegression();
        regression.setNumberOfEpochs(1000).setLearningRate(0.01);

        regression.train(new Matrix(LOGISTIC_INPUTS), new Matrix(LOGISTIC_TARGETS));
        const values = regression.predict(new Matrix(LOGISTIC_INPUTS)).toArray().flat();

        for (const value of values) {
            expect(value).toBeGreaterThanOrEqual(0);
            expect(value).toBeLessThanOrEqual(1);
        }
    });

    it('applies the sigmoid of the bias-enriched hypothesis in predict', () => {
        const regression = new LogisticRegression();
        regression.setHypothesis(new Matrix([[0], [0]]));

        // With a zero hypothesis every sigmoid output is exactly 0.5.
        const predictions = regression.predict(new Matrix([[10], [-10]])).toArray();

        expect(predictions).toEqual([[0.5], [0.5]]);
    });
});
