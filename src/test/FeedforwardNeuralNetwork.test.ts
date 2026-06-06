import { describe, it, expect } from 'vitest';
import FeedforwardNeuralNetwork from '../lib/machine-learning/supervised/FeedforwardNeuralNetwork';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { XNOR_INPUTS, XNOR_TARGETS, FIXED_SEED } from './helpers/fixtures';

describe('FeedforwardNeuralNetwork', () => {

    describe('checkGradients', () => {
        // Numerically verifies analytic backprop against finite differences (tolerance 1e-4).
        // The math is correct, so this is deterministic in outcome despite using unseeded weights.
        it('confirms analytic gradients match numeric gradients for a small network', () => {
            expect(new FeedforwardNeuralNetwork([2, 5, 1]).checkGradients()).toBe(true);
        });

        it('confirms gradients for a deeper network', () => {
            expect(new FeedforwardNeuralNetwork([3, 4, 4, 2]).checkGradients()).toBe(true);
        });
    });

    describe('learning XNOR', () => {
        const trainXnor = (seed: number) => {
            const network = new FeedforwardNeuralNetwork([2, 5, 1], seed);
            network.setNumberOfEpochs(1000);
            network.setLearningRate(1);
            network.train(new Matrix(XNOR_INPUTS), new Matrix(XNOR_TARGETS));
            return network;
        };

        it('classifies all four XNOR cases correctly with a fixed seed', () => {
            const predictions = trainXnor(FIXED_SEED).predict(new Matrix(XNOR_INPUTS)).toArray();

            // XNOR targets are [1, 0, 0, 1]; assert each prediction lands on the correct side of 0.5.
            expect(predictions[0][0]).toBeGreaterThan(0.5);
            expect(predictions[1][0]).toBeLessThan(0.5);
            expect(predictions[2][0]).toBeLessThan(0.5);
            expect(predictions[3][0]).toBeGreaterThan(0.5);
        });

        it('is reproducible: the same seed yields identical predictions', () => {
            const first = trainXnor(FIXED_SEED).predict(new Matrix(XNOR_INPUTS)).toArray();
            const second = trainXnor(FIXED_SEED).predict(new Matrix(XNOR_INPUTS)).toArray();

            expect(first).toEqual(second);
        });
    });

    describe('parameter accessors', () => {
        it('round-trips the configurable hyperparameters', () => {
            const network = new FeedforwardNeuralNetwork([2, 2, 1]);

            network.setLearningRate(0.5);
            network.setNumberOfEpochs(250);
            network.setBatchSize(4);

            expect(network.getLearningRate()).toBe(0.5);
            expect(network.getNumberOfEpochs()).toBe(250);
            expect(network.getBatchSize()).toBe(4);
        });
    });

    describe('getWeightMatrices', () => {
        it('returns one clone per layer transition with bias-augmented shapes', () => {
            const network = new FeedforwardNeuralNetwork([2, 5, 1]);

            const weights = network.getWeightMatrices();

            expect(weights).toHaveLength(2);
            // (incomingNodes + 1) × outgoingNodes — row 0 holds the bias weights.
            expect([weights[0].getRowCount(), weights[0].getColumnCount()]).toEqual([3, 5]);
            expect([weights[1].getRowCount(), weights[1].getColumnCount()]).toEqual([6, 1]);
        });

        it('returns clones — mutating them does not affect the network', () => {
            const network = new FeedforwardNeuralNetwork([2, 3, 1], FIXED_SEED);
            const before = network.predict(new Matrix(XNOR_INPUTS)).toArray();

            network.getWeightMatrices().forEach(weightMatrix => weightMatrix.transform(() => 999));

            expect(network.predict(new Matrix(XNOR_INPUTS)).toArray()).toEqual(before);
        });
    });

    describe('computeLoss', () => {
        it('returns a finite, non-negative loss', () => {
            const network = new FeedforwardNeuralNetwork([2, 5, 1], FIXED_SEED);

            const loss = network.computeLoss(new Matrix(XNOR_INPUTS), new Matrix(XNOR_TARGETS));

            expect(Number.isFinite(loss)).toBe(true);
            expect(loss).toBeGreaterThanOrEqual(0);
        });

        it('decreases after training on XNOR', () => {
            const network = new FeedforwardNeuralNetwork([2, 5, 1], FIXED_SEED);
            const inputs = new Matrix(XNOR_INPUTS);
            const targets = new Matrix(XNOR_TARGETS);

            const lossBefore = network.computeLoss(inputs, targets);

            network.setNumberOfEpochs(1000);
            network.setLearningRate(1);
            network.train(inputs, targets);

            expect(network.computeLoss(inputs, targets)).toBeLessThan(lossBefore);
        });

        it('does not change predictions (read-only)', () => {
            const network = new FeedforwardNeuralNetwork([2, 4, 1], FIXED_SEED);
            const inputs = new Matrix(XNOR_INPUTS);

            const before = network.predict(inputs).toArray();
            network.computeLoss(inputs, new Matrix(XNOR_TARGETS));

            expect(network.predict(inputs).toArray()).toEqual(before);
        });
    });
});
