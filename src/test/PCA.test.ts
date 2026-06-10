import { describe, it, expect } from 'vitest';
import PCA from '../lib/machine-learning/unsupervised/PCA';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { PCA_DIAGONAL_INPUTS } from './helpers/fixtures';

const SQRT_HALF = Math.SQRT1_2; // 0.7071…

describe('PCA', () => {

    it('finds the diagonal as the first principal component', () => {
        const model = new PCA().setNumberOfComponents(2);
        model.train(new Matrix(PCA_DIAGONAL_INPUTS));

        const [pc1] = model.getComponents().toArray();
        // PC1 points along y = x (sign normalised to positive): ~[0.707, 0.707].
        expect(Math.abs(pc1[0])).toBeCloseTo(SQRT_HALF, 6);
        expect(Math.abs(pc1[1])).toBeCloseTo(SQRT_HALF, 6);
        expect(Math.sign(pc1[0])).toBe(Math.sign(pc1[1])); // both components share a sign on this line
    });

    it('reports almost all variance on the first component', () => {
        const model = new PCA().setNumberOfComponents(2);
        model.train(new Matrix(PCA_DIAGONAL_INPUTS));

        const ratio = model.getExplainedVarianceRatio();
        expect(ratio[0]).toBeCloseTo(1, 6);
        expect(ratio[1]).toBeCloseTo(0, 6);
        expect(ratio[0] + ratio[1]).toBeCloseTo(1, 6);
    });

    it('projects to k dimensions', () => {
        const model = new PCA().setNumberOfComponents(1);
        model.train(new Matrix(PCA_DIAGONAL_INPUTS));

        const projected = model.predict(new Matrix(PCA_DIAGONAL_INPUTS));
        expect(projected.getColumnCount()).toBe(1);
        expect(projected.getRowCount()).toBe(PCA_DIAGONAL_INPUTS.length);
        // The point (2, 2) sits sqrt(8) ≈ 2.83 along the diagonal from the centre (the origin here).
        expect(Math.abs(projected.toArray()[4][0])).toBeCloseTo(Math.sqrt(8), 6);
    });

    it('reconstructs a 1-D shadow that lands back on the line (data is genuinely 1-D)', () => {
        const model = new PCA().setNumberOfComponents(1);
        model.train(new Matrix(PCA_DIAGONAL_INPUTS));

        // The data lies exactly on y = x, so even a single component rebuilds it with no loss.
        const reconstructed = model.reconstruct(new Matrix(PCA_DIAGONAL_INPUTS)).toArray();
        reconstructed.forEach((row, i) => {
            expect(row[0]).toBeCloseTo(PCA_DIAGONAL_INPUTS[i][0], 6);
            expect(row[1]).toBeCloseTo(PCA_DIAGONAL_INPUTS[i][1], 6);
        });
    });

    it('reconstructs losslessly when keeping every component', () => {
        const inputs = new Matrix([[1, 2], [3, 1], [-2, 4], [0, -3], [2, 2]]);
        const model = new PCA().setNumberOfComponents(2);
        model.train(inputs);

        const reconstructed = model.reconstruct(inputs).toArray();
        inputs.toArray().forEach((row, i) => {
            expect(reconstructed[i][0]).toBeCloseTo(row[0], 6);
            expect(reconstructed[i][1]).toBeCloseTo(row[1], 6);
        });
    });

    it('centres on the data mean', () => {
        const model = new PCA().train(new Matrix([[0, 10], [2, 14], [4, 18]]));
        expect(model.getMean().toArray()).toEqual([[2, 14]]);
    });

    it('is deterministic and chains setters', () => {
        const run = () => new PCA().setNumberOfComponents(1).train(new Matrix(PCA_DIAGONAL_INPUTS)).predict(new Matrix(PCA_DIAGONAL_INPUTS)).toArray();
        expect(run()).toEqual(run());

        const model = new PCA();
        expect(model.getNumberOfComponents()).toBe(2);
        expect(model.setNumberOfComponents(3)).toBe(model);
        expect(model.getNumberOfComponents()).toBe(3);
    });
});
