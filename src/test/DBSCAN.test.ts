import { describe, it, expect } from 'vitest';
import DBSCAN from '../lib/machine-learning/unsupervised/DBSCAN';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { DBSCAN_INPUTS, DBSCAN_EXPECTED_LABELS } from './helpers/fixtures';

describe('DBSCAN', () => {

    it('finds the dense clusters and flags the outlier as noise', () => {
        const model = new DBSCAN().setEpsilon(0.5).setMinPoints(3);
        model.train(new Matrix(DBSCAN_INPUTS));

        expect(model.getLabels()).toEqual(DBSCAN_EXPECTED_LABELS);
        expect(model.getClusterCount()).toBe(2);
    });

    it('predicts the same labels for the training points', () => {
        const model = new DBSCAN().setEpsilon(0.5).setMinPoints(3);
        model.train(new Matrix(DBSCAN_INPUTS));

        const predicted = model.predict(new Matrix(DBSCAN_INPUTS)).toArray().map(row => row[0]);
        expect(predicted).toEqual(DBSCAN_EXPECTED_LABELS);
    });

    it('assigns a near point to a cluster and a far point to noise', () => {
        const model = new DBSCAN().setEpsilon(0.5).setMinPoints(3);
        model.train(new Matrix(DBSCAN_INPUTS));

        const predicted = model.predict(new Matrix([[0.05, 0.05], [20, 20]])).toArray().map(row => row[0]);
        expect(predicted[0]).toBe(0); // inside blob A
        expect(predicted[1]).toBe(DBSCAN.NOISE); // nowhere near a core point
    });

    it('merges everything into one cluster as epsilon grows', () => {
        const model = new DBSCAN().setEpsilon(100).setMinPoints(3);
        model.train(new Matrix(DBSCAN_INPUTS));

        // With a huge radius every point is a core neighbour of every other, so it's all one cluster.
        expect(model.getClusterCount()).toBe(1);
        expect(model.getLabels().every(label => label === 0)).toBe(true);
    });

    it('calls everything noise when minPoints is unreachable', () => {
        const model = new DBSCAN().setEpsilon(0.5).setMinPoints(50);
        model.train(new Matrix(DBSCAN_INPUTS));

        expect(model.getClusterCount()).toBe(0);
        expect(model.getLabels().every(label => label === DBSCAN.NOISE)).toBe(true);
    });

    it('is deterministic', () => {
        const run = () => {
            const model = new DBSCAN().setEpsilon(0.5).setMinPoints(3);
            model.train(new Matrix(DBSCAN_INPUTS));
            return model.getLabels();
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new DBSCAN();
        expect(model.getEpsilon()).toBe(0.5);
        expect(model.getMinPoints()).toBe(4);

        const returned = model.setEpsilon(0.3).setMinPoints(6);
        expect(returned).toBe(model);
        expect(model.getEpsilon()).toBe(0.3);
        expect(model.getMinPoints()).toBe(6);
    });
});
