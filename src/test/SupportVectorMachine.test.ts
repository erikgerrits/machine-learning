import { describe, it, expect } from 'vitest';
import SupportVectorMachine from '../lib/machine-learning/supervised/SupportVectorMachine';
import Matrix from '../lib/math/linear-algebra/Matrix';
import {
    SVM_LINEAR_INPUTS,
    SVM_LINEAR_TARGETS,
    SVM_LINEAR_EXPECTED_CLASSES,
    XNOR_INPUTS,
    SVM_XOR_TARGETS,
    SVM_XOR_EXPECTED_CLASSES,
} from './helpers/fixtures';

/** The sign of the decision score is the predicted class: ≥ 0 → class 1, otherwise class 0. */
const classes = (scores: number[][]) => scores.map(row => (row[0] >= 0 ? 1 : 0));

describe('SupportVectorMachine', () => {

    it('separates a linearly separable set with a linear kernel', () => {
        const model = new SupportVectorMachine().setKernel('linear').setRegularization(10).setNumberOfIterations(50);
        model.train(new Matrix(SVM_LINEAR_INPUTS), new Matrix(SVM_LINEAR_TARGETS));

        const scores = model.predict(new Matrix(SVM_LINEAR_INPUTS)).toArray();
        expect(classes(scores)).toEqual(SVM_LINEAR_EXPECTED_CLASSES);
    });

    it('keeps only the boundary-hugging points as support vectors', () => {
        const model = new SupportVectorMachine().setKernel('linear').setRegularization(10).setNumberOfIterations(50);
        model.train(new Matrix(SVM_LINEAR_INPUTS), new Matrix(SVM_LINEAR_TARGETS));

        const supportVectors = model.getSupportVectorIndices();
        // At least one from each class anchors the margin, but not every point should be a support vector.
        expect(supportVectors.length).toBeGreaterThanOrEqual(2);
        expect(supportVectors.length).toBeLessThan(SVM_LINEAR_INPUTS.length);
    });

    it('solves XOR with an RBF kernel where a linear kernel cannot', () => {
        const inputs = new Matrix(XNOR_INPUTS);
        const targets = new Matrix(SVM_XOR_TARGETS);

        const linear = new SupportVectorMachine().setKernel('linear').setRegularization(10).setNumberOfIterations(80);
        linear.train(inputs, targets);
        const linearClasses = classes(linear.predict(inputs).toArray());

        const rbf = new SupportVectorMachine().setKernel('rbf').setGamma(1).setRegularization(100).setNumberOfIterations(200);
        rbf.train(inputs, targets);
        const rbfClasses = classes(rbf.predict(inputs).toArray());

        expect(rbfClasses).toEqual(SVM_XOR_EXPECTED_CLASSES);
        expect(linearClasses).not.toEqual(SVM_XOR_EXPECTED_CLASSES);
    });

    it('widens the margin (shrinks ‖w‖) when C is smaller', () => {
        // Overlapping classes along x: two points (the last in each row) sit on the wrong side, so the
        // problem is *not* linearly separable and the soft-margin C genuinely bites.
        const overlapInputs = new Matrix([[-1, 0], [-0.7, 0], [-0.4, 0], [-0.2, 0], [1, 0], [0.7, 0], [0.4, 0], [0.2, 0]]);
        const overlapTargets = new Matrix([[0], [0], [0], [1], [1], [1], [1], [0]]);

        const norm = (c: number) => {
            const model = new SupportVectorMachine().setKernel('linear').setRegularization(c).setNumberOfIterations(200);
            model.train(overlapInputs, overlapTargets);
            return model.getWeightNorm();
        };

        // A looser penalty tolerates more slack, so the optimiser is free to pick a wider street.
        expect(norm(0.05)).toBeLessThan(norm(50));
    });

    it('is deterministic for a fixed seed', () => {
        const train = () => {
            const model = new SupportVectorMachine().setKernel('rbf').setGamma(1).setRegularization(10).setSeed(7).setNumberOfIterations(60);
            model.train(new Matrix(SVM_LINEAR_INPUTS), new Matrix(SVM_LINEAR_TARGETS));
            return model.predict(new Matrix(SVM_LINEAR_INPUTS)).toArray();
        };

        expect(train()).toEqual(train());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new SupportVectorMachine();
        expect(model.getKernel()).toBe('linear');
        expect(model.getRegularization()).toBe(1);

        const returned = model
            .setKernel('polynomial')
            .setRegularization(5)
            .setGamma(0.5)
            .setDegree(2)
            .setCoefficient(0)
            .setTolerance(1e-4)
            .setNumberOfIterations(100)
            .setSeed(3);

        expect(returned).toBe(model);
        expect(model.getKernel()).toBe('polynomial');
        expect(model.getRegularization()).toBe(5);
        expect(model.getGamma()).toBe(0.5);
        expect(model.getDegree()).toBe(2);
        expect(model.getCoefficient()).toBe(0);
        expect(model.getTolerance()).toBe(1e-4);
        expect(model.getNumberOfIterations()).toBe(100);
        expect(model.getSeed()).toBe(3);
    });
});
