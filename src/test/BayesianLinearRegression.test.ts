import { describe, it, expect } from 'vitest';
import BayesianLinearRegression from '../lib/machine-learning/supervised/BayesianLinearRegression';
import Matrix from '../lib/math/linear-algebra/Matrix';

// A hidden sine, sampled (with a little noise) only over the middle of the range — leaving the edges
// unobserved, so we can watch the model stay confident where it has data and grow unsure where it doesn't.
const f = (x: number) => Math.sin(2 * Math.PI * x) * 0.5;
function dataset(count: number, seed = 1): { X: Matrix; Y: Matrix } {
    let s = seed;
    const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
    const xs: number[][] = [], ys: number[][] = [];
    for (let i = 0; i < count; i++) {
        const x = 0.2 + rand() * 0.45; // observed only within [0.2, 0.65]
        xs.push([x]);
        ys.push([f(x) + (rand() - 0.5) * 0.1]);
    }
    return { X: new Matrix(xs), Y: new Matrix(ys) };
}

const std = (model: BayesianLinearRegression, x: number) => model.predictiveStandardDeviation(new Matrix([[x]]))[0];

describe('BayesianLinearRegression', () => {

    it('fits the mean curve through the data', () => {
        const { X, Y } = dataset(30);
        const model = new BayesianLinearRegression().setBasis('gaussian').setNumberOfBases(6).setBasisWidth(0.15).setBeta(40).setSeed(0);
        model.train(X, Y);

        for (const x of [0.3, 0.45, 0.6]) {
            const predicted = model.predict(new Matrix([[x]])).getElement(0, 0);
            expect(Math.abs(predicted - f(x))).toBeLessThan(0.12); // close to the true sine within the data
        }
    });

    it('grows less certain away from the data (the error bars fan out)', () => {
        const { X, Y } = dataset(30);
        const model = new BayesianLinearRegression().setBasis('gaussian').setNumberOfBases(6).setBasisWidth(0.15).setBeta(40).setSeed(0);
        model.train(X, Y);

        expect(std(model, 0.0)).toBeGreaterThan(std(model, 0.4));   // extrapolating left
        expect(std(model, 0.95)).toBeGreaterThan(std(model, 0.4));  // extrapolating right
    });

    it('gets more certain as more data arrives (the posterior contracts)', () => {
        const train = (count: number) => {
            const { X, Y } = dataset(count);
            const model = new BayesianLinearRegression().setBasis('gaussian').setNumberOfBases(6).setBasisWidth(0.15).setBeta(40).setSeed(0);
            return model.train(X, Y);
        };
        expect(std(train(40), 0.4)).toBeLessThan(std(train(4), 0.4));
    });

    it('draws posterior curves that bunch up on the data and spread off it', () => {
        const { X, Y } = dataset(30);
        const model = new BayesianLinearRegression().setBasis('gaussian').setNumberOfBases(6).setBasisWidth(0.15).setBeta(40).setSeed(0);
        model.train(X, Y);

        const curves = model.sample(new Matrix([[0.4], [0.95]]), 150, 3);
        expect(curves.length).toBe(150);
        const spread = (col: number) => {
            const values = curves.map(c => c[col]);
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            return Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
        };
        expect(spread(1)).toBeGreaterThan(spread(0) * 2); // far-from-data curves vary much more
    });

    it('also fits with a polynomial basis', () => {
        const { X, Y } = dataset(30);
        const model = new BayesianLinearRegression().setBasis('polynomial').setDegree(4).setAlpha(0.5).setBeta(40);
        model.train(X, Y);

        expect(Math.abs(model.predict(new Matrix([[0.45]])).getElement(0, 0) - f(0.45))).toBeLessThan(0.15);
        expect(std(model, 1.1)).toBeGreaterThan(std(model, 0.45)); // polynomials extrapolate very unsurely
    });

    it('is deterministic and shaped correctly', () => {
        const { X, Y } = dataset(20);
        const model = new BayesianLinearRegression().setNumberOfBases(5).setSeed(7).train(X, Y);
        const grid = new Matrix([[0.1], [0.5], [0.9]]);

        expect(model.predict(grid).getRowCount()).toBe(3);
        expect(model.predictiveStandardDeviation(grid).length).toBe(3);
        expect(model.sample(grid, 4, 2)).toEqual(model.sample(grid, 4, 2)); // same seed → same curves
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new BayesianLinearRegression();
        expect(model.getBasis()).toBe('gaussian');

        const returned = model.setBasis('polynomial').setDegree(3).setNumberOfBases(8).setBasisWidth(0.2).setAlpha(2).setBeta(10).setSeed(5);
        expect(returned).toBe(model);
        expect(model.getBasis()).toBe('polynomial');
        expect(model.getDegree()).toBe(3);
        expect(model.getNumberOfBases()).toBe(8);
        expect(model.getBasisWidth()).toBe(0.2);
        expect(model.getAlpha()).toBe(2);
        expect(model.getBeta()).toBe(10);
        expect(model.getSeed()).toBe(5);
    });
});
