import { describe, it, expect } from 'vitest';
import ExponentialSmoothing from '../lib/machine-learning/time-series/ExponentialSmoothing';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { TS_CONSTANT, TS_TREND, TS_SEASONAL } from './helpers/fixtures';

const flat = (matrix: Matrix) => matrix.toArray().map(row => row[0]);

describe('ExponentialSmoothing', () => {

    it('forecasts a constant series as that constant', () => {
        const model = new ExponentialSmoothing().setAlpha(0.5);
        model.train(new Matrix(TS_CONSTANT));

        for (const value of flat(model.predict(3))) {
            expect(value).toBeCloseTo(5, 6);
        }
    });

    it('extends a linear trend upward when trend smoothing is on', () => {
        const model = new ExponentialSmoothing().setAlpha(0.5).setBeta(0.5);
        model.train(new Matrix(TS_TREND));

        const forecast = flat(model.predict(3));
        // The series ends at 6 and climbs by ~1 each step, so the forecast keeps climbing.
        expect(forecast[0]).toBeGreaterThan(6);
        expect(forecast[1]).toBeGreaterThan(forecast[0]);
        expect(forecast[2]).toBeGreaterThan(forecast[1]);
        expect(model.getTrend()).toBeGreaterThan(0.5);
    });

    it('ignores trend by default (a flat forecast for the same trend series)', () => {
        const model = new ExponentialSmoothing().setAlpha(0.5); // beta defaults to 0
        model.train(new Matrix(TS_TREND));

        const forecast = flat(model.predict(2));
        expect(forecast[0]).toBeCloseTo(forecast[1], 6); // no trend term → flat
    });

    it('learns a repeating seasonal cycle', () => {
        const model = new ExponentialSmoothing().setAlpha(0.5).setGamma(0.5).setSeasonLength(2);
        model.train(new Matrix(TS_SEASONAL));

        const forecast = flat(model.predict(4));
        // Series ends on a "20"; next should continue 10, 20, 10, 20.
        expect(forecast[0]).toBeCloseTo(10, 4);
        expect(forecast[1]).toBeCloseTo(20, 4);
        expect(forecast[2]).toBeCloseTo(10, 4);
        expect(forecast[3]).toBeCloseTo(20, 4);
    });

    it('produces one fitted value per training point', () => {
        const model = new ExponentialSmoothing().train(new Matrix(TS_SEASONAL));
        expect(model.getFitted().getRowCount()).toBe(TS_SEASONAL.length);
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new ExponentialSmoothing();
        expect(model.getAlpha()).toBe(0.5);
        expect(model.getBeta()).toBe(0);
        expect(model.getSeasonLength()).toBe(1);

        const returned = model.setAlpha(0.3).setBeta(0.2).setGamma(0.4).setSeasonLength(7);
        expect(returned).toBe(model);
        expect(model.getAlpha()).toBe(0.3);
        expect(model.getBeta()).toBe(0.2);
        expect(model.getGamma()).toBe(0.4);
        expect(model.getSeasonLength()).toBe(7);
    });
});
