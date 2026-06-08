import { describe, it, expect } from 'vitest';
import RandomForest from '../lib/machine-learning/supervised/RandomForest';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { TREE_INPUTS, TREE_TARGETS, TREE_EXPECTED_CLASSES } from './helpers/fixtures';

describe('RandomForest', () => {

    it('classifies the AND rule correctly (a committee of bagged trees)', () => {
        const forest = new RandomForest();
        forest.setNumberOfTrees(25).setMaxDepth(4).setSeed(0);
        forest.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const predictions = forest.predict(new Matrix(TREE_INPUTS));

        expect(predictions.getMaximumRowIndeces().toArray()).toEqual(TREE_EXPECTED_CLASSES);
    });

    it('averages into one probability distribution per class, each row summing to 1', () => {
        const forest = new RandomForest();
        forest.setNumberOfTrees(15).setSeed(1);
        forest.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const predictions = forest.predict(new Matrix(TREE_INPUTS));

        expect(predictions.getRowCount()).toBe(TREE_INPUTS.length);
        expect(predictions.getColumnCount()).toBe(TREE_TARGETS[0].length);
        for (const row of predictions.toArray()) {
            expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 9);
        }
    });

    it('is reproducible for a fixed seed and grows the requested number of trees', () => {
        const train = () => {
            const forest = new RandomForest();
            forest.setNumberOfTrees(12).setMaxFeatures(1).setSeed(7);
            forest.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));
            return forest;
        };

        const a = train();
        const b = train();

        expect(a.getTrees()).toHaveLength(12);
        expect(a.predict(new Matrix(TREE_INPUTS)).toArray()).toEqual(b.predict(new Matrix(TREE_INPUTS)).toArray());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const forest = new RandomForest();
        expect(forest.getNumberOfTrees()).toBe(50);

        expect(forest.setNumberOfTrees(10).setMaxDepth(6).setMaxFeatures(1).setMinSamplesSplit(3).setSeed(4)).toBe(forest);
        expect(forest.getNumberOfTrees()).toBe(10);
        expect(forest.getMaxDepth()).toBe(6);
        expect(forest.getMaxFeatures()).toBe(1);
        expect(forest.getMinSamplesSplit()).toBe(3);
        expect(forest.getSeed()).toBe(4);
    });
});
