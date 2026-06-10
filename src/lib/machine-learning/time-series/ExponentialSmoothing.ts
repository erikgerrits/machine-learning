import Matrix from "../../math/linear-algebra/Matrix";

/**
 * **Exponential smoothing** for time-series forecasting — Holt-Winters, the workhorse for "what
 * happens next week?". Plain regression treats rows as an unordered bag of points; a time series is
 * the opposite — the *order* is the signal, and the recent past matters more than the distant past.
 * Exponential smoothing leans into that: every forecast is a weighted average of history where the
 * weights fade exponentially into the past.
 *
 * It tracks up to three things, each its own running estimate updated as it walks the series:
 * - **level** (`alpha`) — where the series is right now,
 * - **trend** (`beta`) — which way and how fast it's drifting (set `beta = 0` to ignore trend), and
 * - **seasonality** (`gamma`, `seasonLength`) — a repeating cycle, e.g. the weekly rhythm of a café
 *   (`seasonLength = 7`); set `seasonLength` to 1 to ignore it.
 *
 * `train` takes the series as a column (one value per time step) and walks it once, settling those
 * estimates. `predict(steps)` then extends them into the future: level, plus `steps` of trend, plus
 * the matching point in the seasonal cycle. {@link getFitted} returns the one-step-ahead predictions
 * over the history, for plotting the fit. Fully deterministic.
 *
 * @example
 * const model = new ExponentialSmoothing().setAlpha(0.5).setSeasonLength(2);
 * model.train(new Matrix([[10], [20], [10], [20], [10], [20]]));
 * model.predict(2).toArray(); // ≈ [[10], [20]] — it learned the two-step cycle
 */
export default class ExponentialSmoothing {

    private alpha = 0.5;     // level smoothing
    private beta = 0;        // trend smoothing (0 → no trend term)
    private gamma = 0.3;     // seasonal smoothing
    private seasonLength = 1; // 1 → no seasonal term

    private level = 0;
    private trend = 0;
    private seasonals: number[] = [];
    private seriesLength = 0;
    private fitted: number[] = [];

    public constructor () {}

    public train (inputs: Matrix) {
        const series = inputs.toArray().map(row => row[0]);
        const n = series.length;
        const m = this.seasonLength;
        const seasonal = m >= 2;
        const useTrend = this.beta > 0;

        this.seriesLength = n;
        this.fitted = [];
        this.seasonals = [];

        if (n === 0) {
            return this;
        }

        // Seed the estimates: level from the first season's average, trend from the gap to the next
        // season, and one seasonal offset per phase.
        let level: number;
        let trend = 0;
        const seasonals = new Array<number>(seasonal ? m : 0).fill(0);

        if (seasonal) {
            level = mean(series.slice(0, m));
            if (useTrend && n >= 2 * m) {
                trend = (mean(series.slice(m, 2 * m)) - mean(series.slice(0, m))) / m;
            }
            for (let i = 0; i < m; i++) {
                seasonals[i] = series[i] - level;
            }
        } else {
            level = series[0];
            if (useTrend && n >= 2) {
                trend = series[1] - series[0];
            }
        }

        // Walk the series, updating level / trend / season and recording each one-step-ahead forecast.
        for (let t = 0; t < n; t++) {
            const previousLevel = level;
            const previousTrend = trend;
            const season = seasonal ? seasonals[t % m] : 0;

            this.fitted.push(previousLevel + (useTrend ? previousTrend : 0) + season);

            level = this.alpha * (series[t] - season) + (1 - this.alpha) * (previousLevel + (useTrend ? previousTrend : 0));
            if (useTrend) {
                trend = this.beta * (level - previousLevel) + (1 - this.beta) * previousTrend;
            }
            if (seasonal) {
                seasonals[t % m] = this.gamma * (series[t] - level) + (1 - this.gamma) * season;
            }
        }

        this.level = level;
        this.trend = trend;
        this.seasonals = seasonals;
        return this;
    }

    /** Forecast the next `steps` values: level + accumulated trend + the matching seasonal offset. */
    public predict (steps: number) {
        const m = this.seasonLength;
        const seasonal = m >= 2;
        const useTrend = this.beta > 0;

        const forecast: number[][] = [];
        for (let h = 1; h <= steps; h++) {
            const season = seasonal ? this.seasonals[(this.seriesLength + h - 1) % m] : 0;
            forecast.push([this.level + (useTrend ? h * this.trend : 0) + season]);
        }
        return new Matrix(forecast);
    }

    /* Parameter setters */

    public setAlpha (alpha: number) {
        this.alpha = alpha;
        return this;
    }

    public setBeta (beta: number) {
        this.beta = beta;
        return this;
    }

    public setGamma (gamma: number) {
        this.gamma = gamma;
        return this;
    }

    public setSeasonLength (seasonLength: number) {
        this.seasonLength = seasonLength;
        return this;
    }

    /* Parameter getters */

    public getAlpha () {
        return this.alpha;
    }

    public getBeta () {
        return this.beta;
    }

    public getGamma () {
        return this.gamma;
    }

    public getSeasonLength () {
        return this.seasonLength;
    }

    /** The one-step-ahead predictions over the training series, for plotting the fit. */
    public getFitted () {
        return new Matrix(this.fitted.map(value => [value]));
    }

    /** The final level estimate (where the series ended up). */
    public getLevel () {
        return this.level;
    }

    /** The final trend estimate (per-step drift). */
    public getTrend () {
        return this.trend;
    }
}

function mean (values: number[]) {
    if (values.length === 0) {
        return 0;
    }
    let sum = 0;
    for (const value of values) {
        sum += value;
    }
    return sum / values.length;
}
