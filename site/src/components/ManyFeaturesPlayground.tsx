import { useCallback, useEffect, useRef, useState } from 'react';
import { LinearRegression, Matrix } from 'machine-learning';
import { mse } from '../ml/metrics';
import { gaussian, mulberry32 } from '../ml/rng';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawPredictedVsActual } from '../viz/predictedActual';
import { drawLossCurve } from '../viz/lossCurve';
import {
    Card,
    Checkbox,
    ControlPanel,
    Hint,
    Metric,
    MetricsRow,
    NumberField,
    RunControls,
    Slider,
} from './controls/Controls';
import styles from './ManyFeaturesPlayground.module.css';

const POINTS = 120;
const SALES_MIN = 20;
const SALES_MAX = 140;

// The hidden "true recipe" the data is generated from — the model should recover these.
const TRUE = { bias: 80, temp: 18, weekend: 22, footfall: 12 };

// Three features per day, each in a friendly O(1) range so plain gradient descent converges
// without rescaling. weekend is a 0/1 flag (~2 days in 7); temperature and foot traffic sit in
// [-1, 1] relative to a normal day.
const FEATURES = [
    { key: 'temp', label: 'Temperature', col: 0 },
    { key: 'weekend', label: 'Weekend', col: 1 },
    { key: 'footfall', label: 'Foot traffic', col: 2 },
] as const;

const sliderToLr = (slider: number) => Math.pow(10, -3 + 4 * (slider / 1000));
const lrToSlider = (lr: number) => Math.round(((Math.log10(lr) + 3) / 4) * 1000);

function generate(seed: number): { rows: number[][]; targets: number[][] } {
    const rand = mulberry32(seed);
    const rows: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < POINTS; i++) {
        const temp = rand() * 2 - 1;
        const weekend = rand() < 2 / 7 ? 1 : 0;
        const footfall = rand() * 2 - 1;
        const sold =
            TRUE.bias + TRUE.temp * temp + TRUE.weekend * weekend + TRUE.footfall * footfall + gaussian(rand) * 5;
        rows.push([temp, weekend, footfall]);
        targets.push([sold]);
    }
    return { rows, targets };
}

interface TrainingData {
    inputMatrix: Matrix | null; // null when no features are active (the baseline / predict-the-mean)
    targetMatrix: Matrix | null;
    targets: number[][];
    mean: number;
}

export function ManyFeaturesPlayground() {
    const [active, setActive] = useState({ temp: true, weekend: false, footfall: false });
    const [sliderLR, setSliderLR] = useState(lrToSlider(0.3));
    const [seed, setSeed] = useState(0);
    const [stepsPerFrame, setStepsPerFrame] = useState(4);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epoch: 0, loss: 0 });
    const [baseValue, setBaseValue] = useState(0);
    const [weights, setWeights] = useState<{ label: string; value: number }[]>([]);

    const learningRate = sliderToLr(sliderLR);
    const lrRef = useRef(learningRate);
    const stepsRef = useRef(stepsPerFrame);
    lrRef.current = learningRate;
    stepsRef.current = stepsPerFrame;

    const modelRef = useRef<LinearRegression | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const activeLabelsRef = useRef<string[]>([]);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const plotCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const predictionsOf = (data: TrainingData): number[] => {
        const model = modelRef.current;
        if (!model || !data.inputMatrix) return data.targets.map(() => data.mean);
        return model.predict(data.inputMatrix).toArray().map(row => row[0]);
    };

    const setReadout = (model: LinearRegression | null, mean: number) => {
        if (!model) {
            setBaseValue(mean);
            setWeights([]);
            return;
        }
        const h = model.getHypothesis().toArray().map(row => row[0]);
        setBaseValue(h[0]);
        setWeights(activeLabelsRef.current.map((label, i) => ({ label, value: h[i + 1] })));
    };

    const drawAll = useCallback(() => {
        const data = dataRef.current;
        if (!data) return;
        const predicted = predictionsOf(data);
        const actual = data.targets.map(row => row[0]);

        const plot = plotCanvasRef.current;
        if (plot) {
            const { ctx, width, height } = fitCanvas(plot);
            drawPredictedVsActual(ctx, width, height, SALES_MIN, SALES_MAX, actual, predicted);
        }
        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const { rows, targets } = generate(seed);
        const mean = targets.reduce((sum, [v]) => sum + v, 0) / targets.length;
        const cols = FEATURES.filter(f => active[f.key]);
        activeLabelsRef.current = cols.map(f => f.label);

        let inputMatrix: Matrix | null = null;
        let targetMatrix: Matrix | null = null;
        let model: LinearRegression | null = null;
        if (cols.length > 0) {
            inputMatrix = new Matrix(rows.map(row => cols.map(f => row[f.col])));
            targetMatrix = new Matrix(targets);
            model = new LinearRegression();
            model.setNumberOfEpochs(1); // one epoch per train() call → epochs driven by the loop
            model.setLearningRate(lrRef.current);
            // Start at the baseline: bias = the mean, every feature weight 0. Training pulls the
            // weights up from there, so the cloud begins as a flat band at predicted = average.
            const hypothesis = Matrix.zeros(cols.length + 1, 1);
            hypothesis.setElement(0, 0, mean);
            model.setHypothesis(hypothesis);
        }

        modelRef.current = model;
        dataRef.current = { inputMatrix, targetMatrix, targets, mean };
        epochRef.current = 0;

        const predicted = model && inputMatrix ? model.predict(inputMatrix).toArray().map(r => r[0]) : targets.map(() => mean);
        lossRef.current = [mse(predicted.map(p => [p]), targets)];

        setMetrics({ epoch: 0, loss: lossRef.current[0] });
        setReadout(model, mean);
        drawAll();
    }, [seed, active, drawAll]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    useEffect(() => {
        modelRef.current?.setLearningRate(learningRate);
    }, [learningRate]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data || !data.inputMatrix || !data.targetMatrix) return; // baseline: nothing to train

        const steps = stepsRef.current;
        for (let i = 0; i < steps; i++) model.train(data.inputMatrix, data.targetMatrix);
        epochRef.current += steps;

        const predicted = model.predict(data.inputMatrix).toArray().map(r => r[0]);
        const loss = mse(predicted.map(p => [p]), data.targets);
        lossRef.current.push(loss);
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawAll();

        frameRef.current += 1;
        if (frameRef.current % 4 === 0) {
            setMetrics({ epoch: epochRef.current, loss });
            setReadout(model, data.mean);
        }
    }, [drawAll]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const featureCount = FEATURES.filter(f => active[f.key]).length;

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls
                    running={running}
                    onToggle={() => setRunning(r => !r)}
                    onStep={handleStep}
                    onReset={handleReset}
                />
                <Hint>
                    Switch features on and off, then press Train. Each clue you add lets the model
                    give a different answer — watch the cloud pull onto the diagonal and the error
                    fall. With none on, it can only guess the average.
                </Hint>
                <Checkbox label="Temperature" checked={active.temp} onChange={v => setActive(a => ({ ...a, temp: v }))} />
                <Checkbox label="Weekend" checked={active.weekend} onChange={v => setActive(a => ({ ...a, weekend: v }))} />
                <Checkbox label="Foot traffic" checked={active.footfall} onChange={v => setActive(a => ({ ...a, footfall: v }))} />
                <Slider
                    label="Learning rate"
                    value={sliderLR}
                    display={learningRate.toFixed(3)}
                    min={0}
                    max={1000}
                    onChange={setSliderLR}
                />
                <Slider
                    label="Speed"
                    value={stepsPerFrame}
                    display={`${stepsPerFrame} epochs / frame`}
                    min={1}
                    max={20}
                    onChange={setStepsPerFrame}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.plotWrap}>
                    <canvas ref={plotCanvasRef} className={styles.plot} />
                    <div className={styles.legend}>
                        <span style={{ color: 'var(--accent)' }}>● a day</span>
                        <span style={{ color: 'var(--muted)' }}>-- perfect prediction</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="MSE" value={metrics.loss.toFixed(1)} />
                        <Metric label="Features" value={String(featureCount)} />
                    </MetricsRow>

                    <Card title="The recipe" subtitle="croissants / day">
                        <ul className={styles.recipe}>
                            <li>
                                <span>Baseline day</span>
                                <strong>{baseValue.toFixed(0)}</strong>
                            </li>
                            {weights.map(w => (
                                <li key={w.label}>
                                    <span>{w.label}</span>
                                    <strong>
                                        {w.value >= 0 ? '+' : '−'}
                                        {Math.abs(w.value).toFixed(1)}
                                    </strong>
                                </li>
                            ))}
                            {weights.length === 0 && (
                                <li>
                                    <span>No features — just the average</span>
                                </li>
                            )}
                        </ul>
                    </Card>

                    <Card title="Loss" subtitle="mean squared error per epoch">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
