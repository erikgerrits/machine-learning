import { describe, it, expect } from 'vitest';
import Perceptron from '../lib/machine-learning/supervised/Perceptron';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { GATE_INPUTS, AND_TARGETS, OR_TARGETS, XOR_TARGETS } from './helpers/fixtures';

const classify = (targets: number[][]) => {
    const model = new Perceptron().setLearningRate(0.1).setNumberOfEpochs(100);
    model.train(new Matrix(GATE_INPUTS), new Matrix(targets));
    return model.predict(new Matrix(GATE_INPUTS)).toArray();
};

describe('Perceptron', () => {

    it('learns AND (linearly separable)', () => {
        expect(classify(AND_TARGETS)).toEqual(AND_TARGETS);
    });

    it('learns OR (linearly separable)', () => {
        expect(classify(OR_TARGETS)).toEqual(OR_TARGETS);
    });

    it('cannot solve XOR — one neuron is one straight line', () => {
        // No matter how long it trains, a single perceptron mis-labels at least one XOR point.
        const predictions = classify(XOR_TARGETS);
        const wrong = predictions.filter((row, i) => row[0] !== XOR_TARGETS[i][0]).length;
        expect(wrong).toBeGreaterThan(0);
    });

    it('is deterministic (zero-initialised)', () => {
        expect(classify(AND_TARGETS)).toEqual(classify(AND_TARGETS));
    });

    it('exposes a weight per feature and round-trips its hyperparameters', () => {
        const model = new Perceptron();
        expect(model.getLearningRate()).toBe(0.1);

        const returned = model.setLearningRate(0.5).setNumberOfEpochs(30);
        expect(returned).toBe(model);
        expect(model.getLearningRate()).toBe(0.5);
        expect(model.getNumberOfEpochs()).toBe(30);

        model.train(new Matrix(GATE_INPUTS), new Matrix(AND_TARGETS));
        expect(model.getWeights().getColumnCount()).toBe(2);
    });
});
