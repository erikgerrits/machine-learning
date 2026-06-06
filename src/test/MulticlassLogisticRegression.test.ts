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
});
