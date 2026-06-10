import { useCallback, useEffect, useRef, useState } from 'react';
import { ExponentialSmoothing, Matrix } from 'machine-learning';
import { TIME_SERIES_DATASETS } from '../ml/timeSeriesDatasets';
import { fitCanvas } from '../viz/canvas';
import { drawForecast } from '../viz/timeSeries';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, Select, Slider } from './controls/Controls';
import styles from './TimeSeriesPlayground.module.css';

export function TimeSeriesPlayground() {
    const [datasetId, setDatasetId] = useState(TIME_SERIES_DATASETS[0].id);
    const [alphaSlider, setAlphaSlider] = useState(TIME_SERIES_DATASETS[0].alpha * 100);
    const [betaSlider, setBetaSlider] = useState(TIME_SERIES_DATASETS[0].beta * 100);
    const [gammaSlider, setGammaSlider] = useState(TIME_SERIES_DATASETS[0].gamma * 100);
    const [weekly, setWeekly] = useState(true);
    const [horizon, setHorizon] = useState(TIME_SERIES_DATASETS[0].horizon);
    const [seed, setSeed] = useState(0);
    const [mae, setMae] = useState(0);

    const alpha = alphaSlider / 100;
    const beta = betaSlider / 100;
    const gamma = gammaSlider / 100;

    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const historyRef = useRef<number[]>([]);
    const fittedRef = useRef<number[]>([]);
    const forecastRef = useRef<number[]>([]);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const { ctx, width, height } = fitCanvas(canvas);
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, width, height);
        drawForecast(ctx, width, height, historyRef.current, fittedRef.current, forecastRef.current);
    }, []);

    const rebuild = useCallback(() => {
        const dataset = TIME_SERIES_DATASETS.find(d => d.id === datasetId) ?? TIME_SERIES_DATASETS[0];
        const { series } = dataset.generate(seed);

        const model = new ExponentialSmoothing()
            .setAlpha(alpha)
            .setBeta(beta)
            .setGamma(gamma)
            .setSeasonLength(weekly ? dataset.seasonLength : 1);
        model.train(new Matrix(series.map(v => [v])));

        const fitted = model.getFitted().toArray().map(row => row[0]);
        const forecast = model.predict(horizon).toArray().map(row => row[0]);

        historyRef.current = series;
        fittedRef.current = fitted;
        forecastRef.current = forecast;

        // In-sample mean absolute error, skipping the first season (warm-up).
        const warmup = weekly ? dataset.seasonLength : 1;
        let sum = 0;
        let count = 0;
        for (let i = warmup; i < series.length; i++) {
            sum += Math.abs(fitted[i] - series[i]);
            count++;
        }
        setMae(count > 0 ? sum / count : 0);
        draw();
    }, [datasetId, alpha, beta, gamma, weekly, horizon, seed, draw]);

    // Debounced so dragging the smoothing sliders (each re-fits + redraws) stays smooth.
    useEffect(() => {
        const timer = setTimeout(rebuild, 60);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const handleDataset = (id: string) => {
        const next = TIME_SERIES_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setAlphaSlider(next.alpha * 100);
        setBetaSlider(next.beta * 100);
        setGammaSlider(next.gamma * 100);
        setHorizon(next.horizon);
    };

    const dataset = TIME_SERIES_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Daily demand"
                    value={datasetId}
                    options={TIME_SERIES_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider label="α · level" value={alphaSlider} display={alpha.toFixed(2)} min={1} max={100} onChange={setAlphaSlider} />
                <Slider label="β · trend" value={betaSlider} display={beta.toFixed(2)} min={0} max={100} onChange={setBetaSlider} />
                {weekly && <Slider label="γ · season" value={gammaSlider} display={gamma.toFixed(2)} min={0} max={100} onChange={setGammaSlider} />}
                <Checkbox label="Weekly season (period 7)" checked={weekly} onChange={setWeekly} />
                <Slider label="Forecast days" value={horizon} display={`${horizon} days`} min={3} max={28} onChange={setHorizon} />
                <Hint>α weights recent days; β follows the trend; γ learns the weekly shape.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.chartWrap}>
                    <canvas ref={canvasRef} className={styles.chart} />
                    <div className={styles.legend}>
                        <span style={{ color: '#38bdf8' }}>▬ actual</span>
                        <span style={{ color: '#fb923c' }}>▬ forecast</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Fit error (MAE)" value={mae.toFixed(1)} />
                        <Metric label="Forecast" value={`${horizon} days`} />
                    </MetricsRow>

                    <Card title="Level, trend, season" subtitle="three running estimates">
                        <p className={styles.note}>
                            Smoothing keeps a running <strong>level</strong> (where demand is now), a{' '}
                            <strong>trend</strong> (which way it's drifting), and a <strong>seasonal</strong>{' '}
                            offset for each weekday. The forecast is just level + trend + the matching day's
                            season — extended past the dashed "now" line.
                        </p>
                    </Card>

                    <Card title="Turn off the season" subtitle="why order matters">
                        <p className={styles.note}>
                            Untick the weekly season and the forecast flattens into a trend line — it can no
                            longer tell a Saturday from a Tuesday. That's the structure plain regression
                            misses when it ignores the order of the days.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
