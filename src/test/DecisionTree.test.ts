import { describe, it, expect } from 'vitest';
import DecisionTree from '../lib/machine-learning/supervised/DecisionTree';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { TREE_INPUTS, TREE_TARGETS, TREE_EXPECTED_CLASSES } from './helpers/fixtures';

describe('DecisionTree', () => {

    it('fits an axis-aligned AND rule perfectly (argmax)', () => {
        const tree = new DecisionTree();
        tree.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const predictions = tree.predict(new Matrix(TREE_INPUTS));

        expect(predictions.getMaximumRowIndeces().toArray()).toEqual(TREE_EXPECTED_CLASSES);
    });

    it('produces one class-probability column per class', () => {
        const tree = new DecisionTree();
        tree.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const predictions = tree.predict(new Matrix(TREE_INPUTS));

        expect(predictions.getRowCount()).toBe(TREE_INPUTS.length);
        expect(predictions.getColumnCount()).toBe(TREE_TARGETS[0].length);

        // A fully grown tree reaches pure leaves, so each row is a hard one-hot distribution.
        for (const row of predictions.toArray()) {
            expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 9);
        }
    });

    it('limits the tree to maxDepth', () => {
        const tree = new DecisionTree();
        tree.setMaxDepth(1);
        tree.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        // Depth 1 means a single split whose two children are already leaves.
        const root = tree.getRoot();
        expect(root.featureIndex).not.toBeUndefined();
        expect(root.left?.distribution).not.toBeUndefined();
        expect(root.right?.distribution).not.toBeUndefined();
    });

    it('cannot fit the AND rule at depth 1 but can at depth 2', () => {
        const inputs = new Matrix(TREE_INPUTS);
        const targets = new Matrix(TREE_TARGETS);

        const shallow = new DecisionTree().setMaxDepth(1);
        shallow.train(inputs, targets);
        const shallowCorrect = shallow.predict(inputs).getMaximumRowIndeces().toArray();

        const deep = new DecisionTree().setMaxDepth(2);
        deep.train(inputs, targets);
        const deepCorrect = deep.predict(inputs).getMaximumRowIndeces().toArray();

        expect(deepCorrect).toEqual(TREE_EXPECTED_CLASSES);
        expect(shallowCorrect).not.toEqual(TREE_EXPECTED_CLASSES);
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const tree = new DecisionTree();
        expect(tree.getMaxDepth()).toBe(5);
        expect(tree.getMinSamplesSplit()).toBe(2);

        expect(tree.setMaxDepth(3).setMinSamplesSplit(4)).toBe(tree); // chainable
        expect(tree.getMaxDepth()).toBe(3);
        expect(tree.getMinSamplesSplit()).toBe(4);
    });

    it('is deterministic by default and reproducible when feature-subsampling with a seed', () => {
        const inputs = new Matrix(TREE_INPUTS);
        const targets = new Matrix(TREE_TARGETS);

        // Default (maxFeatures 0) considers every feature → identical trees with no seed.
        const plainA = new DecisionTree();
        const plainB = new DecisionTree();
        plainA.train(inputs, targets);
        plainB.train(inputs, targets);
        expect(plainA.predict(inputs).toArray()).toEqual(plainB.predict(inputs).toArray());

        // Feature subsampling is randomised but reproducible for a fixed seed.
        const seededA = new DecisionTree().setMaxFeatures(1).setSeed(3);
        const seededB = new DecisionTree().setMaxFeatures(1).setSeed(3);
        seededA.train(inputs, targets);
        seededB.train(inputs, targets);
        expect(seededA.predict(inputs).toArray()).toEqual(seededB.predict(inputs).toArray());
    });

    it('exposes the trained tree as readable nodes', () => {
        const tree = new DecisionTree();
        tree.train(new Matrix(TREE_INPUTS), new Matrix(TREE_TARGETS));

        const root = tree.getRoot();
        // The root splits on one of the two features at a sensible threshold.
        expect([0, 1]).toContain(root.featureIndex);
        expect(typeof root.threshold).toBe('number');
    });
});
