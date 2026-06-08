import { useCallback, useEffect, useRef, useState } from 'react';
import { LinearRegression, Matrix } from 'machine-learning';
import { REGRESSION_DATASETS } from '../ml/regressionDatasets';
import type { Domain } from '../ml/datasets';
import { mse } from '../ml/metrics';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawRegression } from '../viz/regressionLine';
import { drawLossCurve } from '../viz/lossCurve';
import {
    Card,
    ControlPanel,
    Hint,
    Metric,
    MetricsRow,
    NumberField,
    RunControls,
    Select,
    Slider,
} from './controls/Controls';
import styles from './LinearRegressionPlayground.module.css';

const POINTS = 80;

const sliderToLr = (slider: number) => Math.pow(10, -3 + 4 * (slider / 1000));
const lrToSlider = (lr: number) => Math.round(((Math.log10(lr) + 3) / 4) * 1000);

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
    targetMatrix: Matrix;
}

export function LinearRegressionPlayground() {
    const [datasetId, setDatasetId] = useState(REGRESSION_DATASETS[0].id);
    const [sliderLR, setSliderLR] = useState(lrToSlider(REGRESSION_DATASETS[0].recommendedLr));
    const [seed, setSeed] = useState(0);
    const [stepsPerFrame, setStepsPerFrame] = useState(3);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epoch: 0, loss: 0, slope: 0, intercept: 0 });

    const learningRate = sliderToLr(sliderLR);

    const lrRef = useRef(learningRate);
    const stepsRef = useRef(stepsPerFrame);
    lrRef.current = learningRate;
    stepsRef.current = stepsPerFrame;

    const modelRef = useRef<LinearRegression | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const domainRef = useRef<Domain>(REGRESSION_DATASETS[0].domain);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const plotCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const readModel = () => {
        const model = modelRef.current;
        const data = dataRef.current;
        const domain = domainRef.current;
        if (!model || !data) return null;

        const predicted = model.predict(data.inputMatrix).toArray().map(row => row[0]);
        const ends = model.predict(new Matrix([[domain.xMin], [domain.xMax]])).toArray();
        const line: [[number, number], [number, number]] = [
            [domain.xMin, ends[0][0]],
            [domain.xMax, ends[1][0]],
        ];
        const [[intercept], [slope]] = model.getHypothesis().toArray();
        return { predicted, line, slope, intercept };
    };

    const drawAll = useCallback(() => {
        const data = dataRef.current;
        const read = readModel();
        if (!data || !read) return;

        const plotCanvas = plotCanvasRef.current;
        if (plotCanvas) {
            const { ctx, width, height } = fitCanvas(plotCanvas);
            drawRegression(ctx, width, height, domainRef.current, data.inputs, data.targets, read.predicted, read.line, 'temperature →', 'items sold →');
        }

        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = REGRESSION_DATASETS.find(d => d.id === datasetId) ?? REGRESSION_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);

        const model = new LinearRegression();
        model.setNumberOfEpochs(1); // one epoch per train() call → epochs driven by the loop
        model.setLearningRate(lrRef.current);
        // Start from a flat line at zero (intercept 0, slope 0); inputs are bias-enriched to 2 cols.
        model.setHypothesis(Matrix.zeros(2, 1));

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix, targetMatrix };
        domainRef.current = dataset.domain;
        epochRef.current = 0;
        lossRef.current = [mse(model.predict(inputMatrix).toArray(), targets)];

        setMetrics({ epoch: 0, loss: lossRef.current[0], slope: 0, intercept: 0 });
        drawAll();
    }, [datasetId, seed, drawAll]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    useEffect(() => {
        modelRef.current?.setLearningRate(learningRate);
    }, [learningRate]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        const steps = stepsRef.current;
        for (let i = 0; i < steps; i++) model.train(data.inputMatrix, data.targetMatrix);
        epochRef.current += steps;

        const loss = mse(model.predict(data.inputMatrix).toArray(), data.targets);
        lossRef.current.push(loss);
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawAll();

        frameRef.current += 1;
        if (frameRef.current % 4 === 0) {
            const read = readModel();
            setMetrics({
                epoch: epochRef.current,
                loss,
                slope: read?.slope ?? 0,
                intercept: read?.intercept ?? 0,
            });
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
    const handleDataset = (id: string) => {
        const next = REGRESSION_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setSliderLR(lrToSlider(next.recommendedLr));
    };

    const dataset = REGRESSION_DATASETS.find(d => d.id === datasetId);
    const sign = metrics.intercept >= 0 ? '+' : '−';
    const equation = `y = ${metrics.slope.toFixed(2)} · x ${sign} ${Math.abs(metrics.intercept).toFixed(2)}`;

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls
                    running={running}
                    onToggle={() => setRunning(r => !r)}
                    onStep={handleStep}
                    onReset={handleReset}
                />
                <Select
                    label="Dataset"
                    value={datasetId}
                    options={REGRESSION_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
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
                        <span style={{ color: 'var(--accent)' }}>● daily sales</span>
                        <span style={{ color: 'var(--accent-2)' }}>— predicted demand</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="MSE" value={metrics.loss.toFixed(4)} />
                        <Metric label="Slope" value={metrics.slope.toFixed(2)} />
                    </MetricsRow>

                    <Card title="Fitted line" subtitle="intercept + slope · x">
                        <code className={styles.equation}>{equation}</code>
                    </Card>

                    <Card title="Loss" subtitle="mean squared error per epoch">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
