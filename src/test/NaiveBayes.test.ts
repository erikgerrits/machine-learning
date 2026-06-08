import { describe, it, expect } from 'vitest';
import NaiveBayes from '../lib/machine-learning/supervised/NaiveBayes';
import Matrix from '../lib/math/linear-algebra/Matrix';
import {
    NAIVE_BAYES_INPUTS,
    NAIVE_BAYES_TARGETS,
    NAIVE_BAYES_EXPECTED_CLASSES,
    NAIVE_BAYES_QUERIES,
    NAIVE_BAYES_QUERY_CLASSES,
} from './helpers/fixtures';

describe('NaiveBayes', () => {

    it('classifies every training document correctly (argmax)', () => {
        const model = new NaiveBayes();
        model.train(new Matrix(NAIVE_BAYES_INPUTS), new Matrix(NAIVE_BAYES_TARGETS));

        const predictions = model.predict(new Matrix(NAIVE_BAYES_INPUTS));

        expect(predictions.getMaximumRowIndeces().toArray()).toEqual(NAIVE_BAYES_EXPECTED_CLASSES);
    });

    it('generalises to unseen messages', () => {
        const model = new NaiveBayes();
        model.train(new Matrix(NAIVE_BAYES_INPUTS), new Matrix(NAIVE_BAYES_TARGETS));

        const predictions = model.predict(new Matrix(NAIVE_BAYES_QUERIES));

        expect(predictions.getMaximumRowIndeces().toArray()).toEqual(NAIVE_BAYES_QUERY_CLASSES);
    });

    it('produces one normalised probability column per class', () => {
        const model = new NaiveBayes();
        model.train(new Matrix(NAIVE_BAYES_INPUTS), new Matrix(NAIVE_BAYES_TARGETS));

        const predictions = model.predict(new Matrix(NAIVE_BAYES_INPUTS));

        expect(predictions.getRowCount()).toBe(NAIVE_BAYES_INPUTS.length);
        expect(predictions.getColumnCount()).toBe(NAIVE_BAYES_TARGETS[0].length);

        // Every row is a probability distribution summing to 1.
        for (const row of predictions.toArray()) {
            const sum = row.reduce((total, value) => total + value, 0);
            expect(sum).toBeCloseTo(1, 9);
            for (const value of row) {
                expect(Number.isFinite(value)).toBe(true);
                expect(value).toBeGreaterThanOrEqual(0);
            }
        }
    });

    it('smoothing keeps an unseen word from zeroing out a class', () => {
        const model = new NaiveBayes();
        model.train(new Matrix(NAIVE_BAYES_INPUTS), new Matrix(NAIVE_BAYES_TARGETS));

        // "free" + "table" — a mix neither class saw together. With Laplace smoothing the
        // posterior stays finite and well-formed rather than collapsing to NaN.
        const predictions = model.predict(new Matrix([[1, 0, 1, 0]]));
        const [row] = predictions.toArray();

        expect(row.every(value => Number.isFinite(value))).toBe(true);
        expect(row[0] + row[1]).toBeCloseTo(1, 9);
    });

    it('round-trips its smoothing hyperparameter and chains setters', () => {
        const model = new NaiveBayes();
        expect(model.getSmoothing()).toBe(1); // Laplace by default

        expect(model.setSmoothing(0.5)).toBe(model); // chainable
        expect(model.getSmoothing()).toBe(0.5);
    });

    it('exposes one log-prior per class and one likelihood row per class after training', () => {
        const model = new NaiveBayes();
        model.train(new Matrix(NAIVE_BAYES_INPUTS), new Matrix(NAIVE_BAYES_TARGETS));

        expect(model.getLogPriors()).toHaveLength(NAIVE_BAYES_TARGETS[0].length);
        expect(model.getLogLikelihoods()).toHaveLength(NAIVE_BAYES_TARGETS[0].length);
        expect(model.getLogLikelihoods()[0]).toHaveLength(NAIVE_BAYES_INPUTS[0].length);
    });
});
