import { describe, it, expect } from 'vitest';
import HierarchicalClustering from '../lib/machine-learning/unsupervised/HierarchicalClustering';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { KMEANS_INPUTS, HIERARCHICAL_LINE_INPUTS } from './helpers/fixtures';

describe('HierarchicalClustering', () => {

    it('cuts two well-separated blobs into the right two groups', () => {
        const model = new HierarchicalClustering().setNumberOfClusters(2);
        model.train(new Matrix(KMEANS_INPUTS));

        const labels = model.getClusterLabels();
        // Points 0–2 are the origin blob, 3–5 the far blob.
        expect(labels[0]).toBe(labels[1]);
        expect(labels[1]).toBe(labels[2]);
        expect(labels[3]).toBe(labels[4]);
        expect(labels[4]).toBe(labels[5]);
        expect(labels[0]).not.toBe(labels[3]);
    });

    it('records n-1 merges with non-decreasing heights, for every linkage', () => {
        for (const linkage of ['single', 'complete', 'average'] as const) {
            const model = new HierarchicalClustering().setLinkage(linkage);
            model.train(new Matrix(KMEANS_INPUTS));

            const merges = model.getMergeHistory();
            expect(merges.length).toBe(KMEANS_INPUTS.length - 1);
            for (let t = 1; t < merges.length; t++) {
                expect(merges[t].distance).toBeGreaterThanOrEqual(merges[t - 1].distance - 1e-9);
            }
            // The final merge folds everything into one cluster of size n.
            expect(merges[merges.length - 1].size).toBe(KMEANS_INPUTS.length);
        }
    });

    it('merges the closest points first (unambiguous line)', () => {
        const model = new HierarchicalClustering().setLinkage('single');
        model.train(new Matrix(HIERARCHICAL_LINE_INPUTS));

        const merges = model.getMergeHistory();
        // The 0–1 pair (distance 1) is the closest, so it merges first.
        expect(merges[0].distance).toBeCloseTo(1, 9);
        expect(new Set([merges[0].left, merges[0].right])).toEqual(new Set([0, 1]));
        // Heights climb as clusters get coarser: 1 (0–1) → 2 (join point 2) → 5 (join point 8).
        expect(merges.map(m => m.distance)).toEqual([1, 2, 5]);
    });

    it('single vs complete linkage land the same 2-cut but at different heights', () => {
        const single = new HierarchicalClustering().setLinkage('single');
        const complete = new HierarchicalClustering().setLinkage('complete');
        single.train(new Matrix(KMEANS_INPUTS));
        complete.train(new Matrix(KMEANS_INPUTS));

        const top = (model: HierarchicalClustering) => model.getMergeHistory().slice(-1)[0].distance;
        // Complete linkage measures clusters by their farthest pair, so the final join sits higher.
        expect(top(complete)).toBeGreaterThan(top(single));
    });

    it('predicts one-hot membership with k columns summing to 1', () => {
        const model = new HierarchicalClustering().setNumberOfClusters(2);
        model.train(new Matrix(KMEANS_INPUTS));

        const predictions = model.predict(new Matrix(KMEANS_INPUTS));
        expect(predictions.getColumnCount()).toBe(2);
        for (const row of predictions.toArray()) {
            expect(row.reduce((a, b) => a + b, 0)).toBe(1);
        }
    });

    it('is deterministic', () => {
        const run = () => {
            const model = new HierarchicalClustering().setNumberOfClusters(3).setLinkage('average');
            model.train(new Matrix(KMEANS_INPUTS));
            return model.getClusterLabels();
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new HierarchicalClustering();
        expect(model.getNumberOfClusters()).toBe(2);
        expect(model.getLinkage()).toBe('average');

        const returned = model.setNumberOfClusters(4).setLinkage('complete');
        expect(returned).toBe(model);
        expect(model.getNumberOfClusters()).toBe(4);
        expect(model.getLinkage()).toBe('complete');
    });
});
