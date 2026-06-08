import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Domain } from '../ml/datasets';
import { mse } from '../ml/metrics';
import { gaussian, mulberry32 } from '../ml/rng';
import { fitCanvas } from '../viz/canvas';
import { drawRegression } from '../viz/regressionLine';
import { Card, ControlPanel, Hint, Metric, MetricsRow, Slider } from './controls/Controls';
import styles from './BaselinePlayground.module.css';

// Chapter 0's "beat the baseline" demo. No training, no library model — just a column of past
// daily sales and a constant guess (the flat line) you slide by hand. The point: the squared
// error bottoms out *exactly* at the average, so the mean is the bar every real model must beat.

const DAYS = 14;
const DOMAIN: Domain = { xMin: -0.5, xMax: DAYS - 0.5, yMin: 30, yMax: 130 };

// Two weeks of croissant sales, seeded so the picture is stable. Centred near 80 — the same
// "eighty a day" Nadia's aunt scrawled in the till, which is exactly the average of this data.
function makeSales(): { inputs: number[][]; targets: number[][] } {
    const rand = mulberry32(7);
    const inputs: number[][] = [];
    const targets: number[][] = [];
    for (let day = 0; day < DAYS; day++) {
        const sales = Math.round(80 + gaussian(rand) * 15);
        inputs.push([day]);
        targets.push([Math.max(DOMAIN.yMin + 5, Math.min(DOMAIN.yMax - 5, sales))]);
    }
    return { inputs, targets };
}

export function BaselinePlayground() {
    const data = useMemo(makeSales, []);
    const mean = useMemo(
        () => data.targets.reduce((sum, [v]) => sum + v, 0) / data.targets.length,
        [data],
    );
    const meanMse = useMemo(() => mse(data.targets.map(() => [mean]), data.targets), [data, mean]);

    const [guess, setGuess] = useState(60);
    const guessMse = useMemo(() => mse(data.targets.map(() => [guess]), data.targets), [data, guess]);

    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const { ctx, width, height } = fitCanvas(canvas);
        const predicted = data.targets.map(() => guess);
        const line: [[number, number], [number, number]] = [
            [DOMAIN.xMin, guess],
            [DOMAIN.xMax, guess],
        ];
        drawRegression(ctx, width, height, DOMAIN, data.inputs, data.targets, predicted, line, 'day →', 'croissants sold →');
    }, [data, guess]);

    useEffect(() => {
        draw();
    }, [draw]);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Slider
                    label="Your guess"
                    value={guess}
                    display={`${guess} / day`}
                    min={DOMAIN.yMin + 5}
                    max={DOMAIN.yMax - 5}
                    onChange={setGuess}
                />
                <Hint>
                    Slide the flat line up and down — it's one number for every day. Watch the error
                    bottom out, and notice exactly where it lands.
                </Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.plotWrap}>
                    <canvas ref={canvasRef} className={styles.plot} />
                    <div className={styles.legend}>
                        <span style={{ color: 'var(--accent)' }}>● a day's sales</span>
                        <span style={{ color: 'var(--accent-2)' }}>— your guess</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Your guess" value={`${guess}`} />
                        <Metric label="Your error" value={guessMse.toFixed(0)} />
                        <Metric label="Best possible" value={meanMse.toFixed(0)} />
                    </MetricsRow>

                    <Card title="The baseline" subtitle={`the average is ${mean.toFixed(0)}`}>
                        <p className={styles.note}>
                            One number for every day, no matter the weather. The error is smallest
                            <strong> exactly at the average</strong> — no constant guess can beat it.
                            That flat line is the bar every smarter model has to clear.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
