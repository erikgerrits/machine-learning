import { gaussian, mulberry32 } from './rng';

// Daily café-demand series for the time-series playground (Chapter 19). Each is eight weeks of one
// number per day — built from a base level, an optional trend, a weekly rhythm (quiet midweek, busy
// weekends), and noise — so exponential smoothing has a real level/trend/season to recover. The
// recommended alpha/beta/gamma and forecast horizon are sensible starting points per series.
export interface TimeSeriesDataset {
    id: string;
    label: string;
    blurb: string;
    alpha: number;
    beta: number;
    gamma: number;
    seasonLength: number;
    horizon: number;
    generate: (seed: number) => { series: number[] };
}

const DAYS = 56; // eight weeks
// Per-weekday offset (Mon…Sun): quiet start of the week, a Friday/Saturday rush, a calm Sunday.
const WEEKLY = [-10, -7, -4, -2, 22, 34, 10];

function demand(base: number, trend: number, weekly: number[], noise: number) {
    return (seed: number) => {
        const rand = mulberry32(seed);
        const series: number[] = [];
        for (let d = 0; d < DAYS; d++) {
            const value = base + trend * d + weekly[d % 7] + gaussian(rand) * noise;
            series.push(Math.max(0, Math.round(value)));
        }
        return { series };
    };
}

export const TIME_SERIES_DATASETS: TimeSeriesDataset[] = [
    {
        id: 'weekly',
        label: 'Weekly rhythm',
        blurb: 'Demand drifts up gently but every week looks the same — quiet Monday, packed Saturday. Smoothing tracks the level and the weekly season together.',
        alpha: 0.3,
        beta: 0.05,
        gamma: 0.4,
        seasonLength: 7,
        horizon: 14,
        generate: demand(50, 0.15, WEEKLY, 4),
    },
    {
        id: 'growing',
        label: 'Going up',
        blurb: 'The café is taking off — a strong upward trend under the same weekly rhythm. Without the trend term the forecast would lag behind reality; turn beta up and it keeps pace.',
        alpha: 0.3,
        beta: 0.15,
        gamma: 0.4,
        seasonLength: 7,
        horizon: 14,
        generate: demand(35, 0.7, WEEKLY, 5),
    },
    {
        id: 'flat',
        label: 'No real pattern',
        blurb: 'Just a noisy flat line — no trend, no weekly shape worth speaking of. Smoothing can only sensibly return the average; there is nothing here to forecast.',
        alpha: 0.3,
        beta: 0,
        gamma: 0.2,
        seasonLength: 7,
        horizon: 14,
        generate: demand(50, 0, [0, 0, 0, 0, 0, 0, 0], 11),
    },
];
