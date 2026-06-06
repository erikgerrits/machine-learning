import { describe, it, expect } from 'vitest';
import NearestNeighbors from '../lib/machine-learning/supervised/NearestNeighbors';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { expectMatrixClose } from './helpers/matrix-assert';
import { KNN_INPUTS, KNN_TARGETS, KNN_QUERIES, KNN_EXPECTED } from './helpers/fixtures';

describe('NearestNeighbors', () => {

    it('reproduces the documented k=1 predictions on equidistant data', () => {
        const knn = new NearestNeighbors();
        knn.setNumberOfNeighbors(1);
        knn.train(new Matrix(KNN_INPUTS), new Matrix(KNN_TARGETS));

        expectMatrixClose(knn.predict(new Matrix(KNN_QUERIES)), KNN_EXPECTED);
    });

    it('returns a point\'s own target when the query matches it exactly', () => {
        const knn = new NearestNeighbors();
        knn.train(new Matrix([[0, 0], [10, 10]]), new Matrix([[1, 0], [0, 1]]));

        expect(knn.predict(new Matrix([[0, 0]])).toArray()).toEqual([[1, 0]]);
    });

    it('averages the k nearest neighbours when k > 1', () => {
        const knn = new NearestNeighbors();
        knn.setNumberOfNeighbors(2);
        knn.train(new Matrix([[0], [10], [20]]), new Matrix([[2], [4], [100]]));

        // Query 1 is closest to inputs 0 and 10 → average of targets 2 and 4 = 3.
        expect(knn.predict(new Matrix([[1]])).toArray()).toEqual([[3]]);
    });

    it('uses the configured distance function', () => {
        const train = () => {
            const knn = new NearestNeighbors();
            knn.train(new Matrix([[0], [10]]), new Matrix([[1, 0], [0, 1]]));
            return knn;
        };

        // Default (euclidean-squared): query 3 is nearest to input 0 → first target.
        expect(train().predict(new Matrix([[3]])).toArray()).toEqual([[1, 0]]);

        // Negated distance inverts "nearest" → the farthest point (input 10) wins.
        const inverted = train();
        inverted.setDistanceFunction((x, y) => -Matrix.subtract(x, y).transform(v => v * v).getSum());
        expect(inverted.predict(new Matrix([[3]])).toArray()).toEqual([[0, 1]]);
    });

    it('exposes its configuration', () => {
        const knn = new NearestNeighbors();
        const distance = (x: Matrix, y: Matrix) => Math.abs(x.getElement(0, 0) - y.getElement(0, 0));

        knn.setNumberOfNeighbors(3);
        knn.setDistanceFunction(distance);

        expect(knn.getNumberOfNeighbors()).toBe(3);
        expect(knn.getDistanceFunction()).toBe(distance);
    });
});
