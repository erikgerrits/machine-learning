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
});
