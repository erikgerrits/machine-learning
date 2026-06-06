import { describe, it, expect } from 'vitest';
import KMeans from '../lib/machine-learning/unsupervised/KMeans';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { KMEANS_INPUTS, KMEANS_LOW_BLOB_MEAN, KMEANS_HIGH_BLOB_MEAN, FIXED_SEED } from './helpers/fixtures';

describe('KMeans', () => {

    const trainBlobs = (seed = FIXED_SEED) => {
        const kmeans = new KMeans().setNumberOfClusters(2).setSeed(seed);
        kmeans.train(new Matrix(KMEANS_INPUTS));
        return kmeans;
    };

    it('groups two well-separated blobs into two clusters', () => {
        const kmeans = trainBlobs();
        const labels = kmeans.predict(new Matrix(KMEANS_INPUTS)).getMaximumRowIndeces().toArray().flat();

        // The three low points share one cluster, the three high points share the other.
        expect(labels[0]).toBe(labels[1]);
        expect(labels[1]).toBe(labels[2]);
        expect(labels[3]).toBe(labels[4]);
        expect(labels[4]).toBe(labels[5]);
        expect(labels[0]).not.toBe(labels[3]);
    });

    it('converges its centroids to the blob means', () => {
        const centroids = trainBlobs().getCentroids().toArray();

        // Cluster order depends on the random init, so compare after sorting by x.
        const sorted = centroids.slice().sort((a, b) => a[0] - b[0]);

        expect(sorted[0][0]).toBeCloseTo(KMEANS_LOW_BLOB_MEAN[0], 9);
        expect(sorted[0][1]).toBeCloseTo(KMEANS_LOW_BLOB_MEAN[1], 9);
        expect(sorted[1][0]).toBeCloseTo(KMEANS_HIGH_BLOB_MEAN[0], 9);
        expect(sorted[1][1]).toBeCloseTo(KMEANS_HIGH_BLOB_MEAN[1], 9);
    });

    it('is reproducible: the same seed yields identical centroids', () => {
        expect(trainBlobs().getCentroids().toArray()).toEqual(trainBlobs().getCentroids().toArray());
    });

    it('predicts a one-hot membership row per input', () => {
        const memberships = trainBlobs().predict(new Matrix(KMEANS_INPUTS));

        expect(memberships.getRowCount()).toBe(KMEANS_INPUTS.length);
        expect(memberships.getColumnCount()).toBe(2);
        for (const row of memberships.toArray()) {
            expect(row.reduce((sum, value) => sum + value, 0)).toBe(1);
        }
    });

    it('assigns points to provided centroids without training', () => {
        const kmeans = new KMeans().setNumberOfClusters(2).setCentroids(new Matrix([[0, 0], [10, 10]]));

        const memberships = kmeans.predict(new Matrix([[1, 1], [9, 9]])).toArray();

        expect(memberships).toEqual([[1, 0], [0, 1]]);
    });

    it('keeps a centroid that wins no points instead of producing NaNs', () => {
        const kmeans = new KMeans().setNumberOfClusters(2).setNumberOfIterations(1);
        kmeans.setCentroids(new Matrix([[0, 0], [100, 100]]));

        // Every point is nearest to [0, 0], so the second cluster ends up empty.
        kmeans.train(new Matrix([[0, 0], [1, 1], [2, 2]]));
        const centroids = kmeans.getCentroids().toArray();

        expect(centroids[0]).toEqual([1, 1]);     // mean of the three points
        expect(centroids[1]).toEqual([100, 100]); // empty cluster keeps its old centroid
    });

    it('uses the configured distance function', () => {
        const manhattan = (x: Matrix, y: Matrix) => Matrix.subtract(x, y).transform(Math.abs).getSum();
        const kmeans = new KMeans().setNumberOfClusters(2).setDistanceFunction(manhattan).setCentroids(new Matrix([[6, 8], [0, 11]]));

        // For the origin the two metrics disagree: Euclidean picks [6, 8] (100 < 121), but
        // Manhattan picks [0, 11] (11 < 14) — so cluster 1 proves the custom distance is used.
        expect(kmeans.predict(new Matrix([[0, 0]])).toArray()).toEqual([[0, 1]]);
    });

    it('round-trips its configuration', () => {
        const distance = (x: Matrix, y: Matrix) => Math.abs(x.getElement(0, 0) - y.getElement(0, 0));
        const kmeans = new KMeans()
            .setNumberOfClusters(3)
            .setNumberOfIterations(25)
            .setSeed(7)
            .setDistanceFunction(distance);

        expect(kmeans.getNumberOfClusters()).toBe(3);
        expect(kmeans.getNumberOfIterations()).toBe(25);
        expect(kmeans.getSeed()).toBe(7);
        expect(kmeans.getDistanceFunction()).toBe(distance);
    });

    it('resetCentroids clears the learned centroids', () => {
        const kmeans = trainBlobs();
        expect(kmeans.getCentroids()).toBeDefined();

        kmeans.resetCentroids();

        expect(kmeans.getCentroids()).toBeUndefined();
    });
});
