import { describe, it, expect } from 'vitest';
import MulticlassLogisticRegression from '../lib/machine-learning/supervised/MulticlassLogisticRegression';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { MULTICLASS_INPUTS, MULTICLASS_TARGETS, MULTICLASS_EXPECTED_CLASSES } from './helpers/fixtures';

describe('MulticlassLogisticRegression', () => {

    it('predicts the correct class (argmax) for every training example', () => {
        const regression = new MulticlassLogisticRegression();
        regression.setNumberOfEpochs(10000);
        regression.setLearningRate(0.1);

        regression.train(new Matrix(MULTICLASS_INPUTS), new Matrix(MULTICLASS_TARGETS));
        const predictions = regression.predict(new Matrix(MULTICLASS_INPUTS));

        expect(predictions.getMaximumRowIndeces().toArray()).toEqual(MULTICLASS_EXPECTED_CLASSES);
    });

    it('produces one prediction column per class', () => {
        const regression = new MulticlassLogisticRegression();
        regression.setNumberOfEpochs(100);
        regression.setLearningRate(0.1);

        regression.train(new Matrix(MULTICLASS_INPUTS), new Matrix(MULTICLASS_TARGETS));
        const predictions = regression.predict(new Matrix(MULTICLASS_INPUTS));

        expect(predictions.getRowCount()).toBe(MULTICLASS_INPUTS.length);
        expect(predictions.getColumnCount()).toBe(MULTICLASS_TARGETS[0].length);
    });

    it('exposes one hypothesis per class after training', () => {
        const regression = new MulticlassLogisticRegression();
        regression.setNumberOfEpochs(100);

        regression.train(new Matrix(MULTICLASS_INPUTS), new Matrix(MULTICLASS_TARGETS));

        expect(regression.getHypothesis()).toHaveLength(MULTICLASS_TARGETS[0].length);
    });

    it('round-trips its hyperparameters', () => {
        const regression = new MulticlassLogisticRegression();
        regression.setBatchSize(4).setLearningRate(0.05).setNumberOfEpochs(42).setRegularizationFactor(0.2);

        expect(regression.getBatchSize()).toBe(4);
        expect(regression.getLearningRate()).toBe(0.05);
        expect(regression.getNumberOfEpochs()).toBe(42);
        expect(regression.getRegularizationFactor()).toBe(0.2);
    });

    it('round-trips per-class hypotheses and resetHypothesis clears them', () => {
        const inputs = new Matrix(MULTICLASS_INPUTS);
        const targets = new Matrix(MULTICLASS_TARGETS);

        const regression = new MulticlassLogisticRegression();
        regression.setNumberOfEpochs(50).train(inputs, targets);

        const learned = regression.getHypothesis();
        regression.setHypothesis(learned);
        expect(regression.getHypothesis().map(hypothesis => hypothesis.toArray()))
            .toEqual(learned.map(hypothesis => hypothesis.toArray()));

        // After a reset the next train() rebuilds one classifier per class from scratch.
        regression.resetHypothesis();
        regression.train(inputs, targets);
        expect(regression.getHypothesis()).toHaveLength(MULTICLASS_TARGETS[0].length);
    });
});
