import { describe, it, expect } from 'vitest';
import AnomalyDetector from '../lib/machine-learning/unsupervised/AnomalyDetector';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { ANOMALY_NORMAL, ANOMALY_CORRELATED } from './helpers/fixtures';

describe('AnomalyDetector', () => {

    it('flags a far-out point and leaves normal ones alone', () => {
        const model = new AnomalyDetector().setThreshold(3);
        model.train(new Matrix(ANOMALY_NORMAL));

        const flags = model.predict(new Matrix([[0, 0], [0.5, 0.5], [8, 8]])).toArray().map(row => row[0]);
        expect(flags).toEqual([0, 0, 1]);
    });

    it('scores points farther from the centre as more anomalous', () => {
        const model = new AnomalyDetector().train(new Matrix(ANOMALY_NORMAL));

        const scores = model.score(new Matrix([[0, 0], [2, 2], [6, 6]])).toArray().map(row => row[0]);
        expect(scores[0]).toBeLessThan(scores[1]);
        expect(scores[1]).toBeLessThan(scores[2]);
        expect(scores[0]).toBeLessThan(0.5); // near the centre → barely anomalous
    });

    it('uses Mahalanobis distance: across the grain is more anomalous than along it', () => {
        const model = new AnomalyDetector().train(new Matrix(ANOMALY_CORRELATED));

        const along = model.score(new Matrix([[2, 2]])).toArray()[0][0];   // along y = x
        const across = model.score(new Matrix([[2, -2]])).toArray()[0][0]; // same Euclidean distance, across the grain
        expect(across).toBeGreaterThan(along);
    });

    it('lowering the threshold flags more points', () => {
        const queries = new Matrix([[0, 0], [2, 2], [3, 3], [5, 5]]);
        const flagged = (t: number) => {
            const model = new AnomalyDetector().setThreshold(t).train(new Matrix(ANOMALY_NORMAL));
            return model.predict(queries).toArray().reduce((sum, row) => sum + row[0], 0);
        };
        expect(flagged(1.5)).toBeGreaterThanOrEqual(flagged(4));
    });

    it('centres on the data mean', () => {
        const model = new AnomalyDetector().train(new Matrix([[0, 10], [2, 14], [4, 18]]));
        expect(model.getMean().toArray()).toEqual([[2, 14]]);
    });

    it('is deterministic and chains setters', () => {
        const run = () => new AnomalyDetector().setThreshold(2.5).train(new Matrix(ANOMALY_NORMAL)).score(new Matrix(ANOMALY_NORMAL)).toArray();
        expect(run()).toEqual(run());

        const model = new AnomalyDetector();
        expect(model.getThreshold()).toBe(3);
        expect(model.setThreshold(2)).toBe(model);
        expect(model.getThreshold()).toBe(2);
    });
});
