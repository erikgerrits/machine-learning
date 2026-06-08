import { describe, it, expect } from 'vitest';
import GradientBoosting from '../lib/machine-learning/supervised/GradientBoosting';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { TREE_INPUTS, TREE_TARGETS, TREE_EXPECTED_CLASSES } from './helpers/fixtures';

describe('GradientBoosting', () => {

    it('boosts into the AND rule (argmax)', () => {
        const model = new GradientBoosting();
        model.setNumberOfTrees(60).setLearningRate(0.3).setMinSamplesSplit(2);
        model.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const predictions = model.predict(new Matrix(TREE_INPUTS));

        expect(predictions.getMaximumRowIndeces().toArray()).toEqual(TREE_EXPECTED_CLASSES);
    });

    it('returns calibrated-ish probabilities: two columns per row, each summing to 1', () => {
        const model = new GradientBoosting();
        model.setNumberOfTrees(40).setMinSamplesSplit(2);
        model.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const predictions = model.predict(new Matrix(TREE_INPUTS));

        expect(predictions.getRowCount()).toBe(TREE_INPUTS.length);
        expect(predictions.getColumnCount()).toBe(2);
        for (const row of predictions.toArray()) {
            expect(row[0] + row[1]).toBeCloseTo(1, 9);
            expect(row[1]).toBeGreaterThanOrEqual(0);
            expect(row[1]).toBeLessThanOrEqual(1);
        }
    });

    it('drives the training error down as rounds accumulate', () => {
        const inputs = new Matrix(TREE_INPUTS);
        const targets = new Matrix(TREE_TARGETS);

        const accuracyAfter = (rounds: number) => {
            const model = new GradientBoosting().setNumberOfTrees(rounds).setLearningRate(0.3).setMinSamplesSplit(2);
            model.train(inputs, targets);
            const predicted = model.predict(inputs).getMaximumRowIndeces().toArray().map(row => row[0]);
            const truth = TREE_EXPECTED_CLASSES.map(row => row[0]);
            return predicted.filter((cls, i) => cls === truth[i]).length / predicted.length;
        };

        expect(accuracyAfter(40)).toBeGreaterThanOrEqual(accuracyAfter(1));
    });

    it('is deterministic for fixed hyperparameters', () => {
        const train = () => {
            const model = new GradientBoosting().setNumberOfTrees(20).setMinSamplesSplit(2);
            model.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));
            return model.predict(new Matrix(TREE_INPUTS)).toArray();
        };

        expect(train()).toEqual(train());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new GradientBoosting();
        expect(model.getLearningRate()).toBe(0.3);

        expect(model.setNumberOfTrees(80).setLearningRate(0.1).setMaxDepth(2).setMinSamplesSplit(5)).toBe(model);
        expect(model.getNumberOfTrees()).toBe(80);
        expect(model.getLearningRate()).toBe(0.1);
        expect(model.getMaxDepth()).toBe(2);
        expect(model.getMinSamplesSplit()).toBe(5);
    });
});
