import { useCallback, useEffect, useRef, useState } from 'react';
import { Perceptron, Matrix } from 'machine-learning';
import type { Domain } from '../ml/datasets';
import { GATE_DATASETS } from '../ml/perceptronDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, paintBoundary, drawPoints, type Grid } from '../viz/decisionBoundary';
import { drawMisclassified } from '../viz/perceptron';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './PerceptronPlayground.module.css';

const GRID = 64;
const POINTS = 120;

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
    targetMatrix: Matrix;
}

export function PerceptronPlayground() {
    const [datasetId, setDatasetId] = useState(GATE_DATASETS[0].id);
    const [rateSlider, setRateSlider] = useState(20); // learning rate = rateSlider / 100
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epoch: 0, wrong: 0 });

    const learningRate = rateSlider / 100;

    const modelRef = useRef<Perceptron | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(GATE_DATASETS[0].domain);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const epochRef = useRef(0);
    const frameRef = useRef(0);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        const { ctx, width, height } = fitCanvas(canvas);
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, width, height);

        const regions = model.predict(grid.matrix).toArray().map(row => row[0]); // 0 / 1 half-planes
        if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
        paintBoundary(offscreenRef.current, regions, grid.size);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(offscreenRef.current, 0, 0, width, height);

        drawPoints(ctx, data.inputs, data.targets, domainRef.current, width, height);

        const predictions = model.predict(data.inputMatrix).toArray().map(row => row[0]);
        const labels = data.targets.map(row => row[0]);
        drawMisclassified(ctx, data.inputs, labels, predictions, domainRef.current, width, height);
    }, []);

    const rebuild = useCallback(() => {
        const dataset = GATE_DATASETS.find(d => d.id === datasetId) ?? GATE_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);

        const model = new Perceptron().setLearningRate(learningRate).setNumberOfEpochs(1);
        model.train(inputMatrix, targetMatrix); // one pass so the boundary exists to draw

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix, targetMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        epochRef.current = 1;
        frameRef.current = 0;

        const predictions = model.predict(inputMatrix).toArray().map(row => row[0]);
        const wrong = predictions.filter((p, i) => p !== targets[i][0]).length;
        setMetrics({ epoch: 1, wrong });
        setRunning(false);
        draw();
    }, [datasetId, learningRate, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        model.train(data.inputMatrix, data.targetMatrix);
        epochRef.current += 1;

        const predictions = model.predict(data.inputMatrix).toArray().map(row => row[0]);
        const wrong = predictions.filter((p, i) => p !== data.targets[i][0]).length;

        draw();
        frameRef.current += 1;
        if (frameRef.current % 2 === 0) {
            setMetrics({ epoch: epochRef.current, wrong });
        }
    }, [draw]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const dataset = GATE_DATASETS.find(d => d.id === datasetId);

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
                    label="Gate"
                    value={datasetId}
                    options={GATE_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={setDatasetId}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider label="Learning rate" value={rateSlider} display={learningRate.toFixed(2)} min={1} max={100} onChange={setRateSlider} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>One neuron draws one straight line. On XOR, watch the line never stop moving.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={canvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span style={{ color: 'var(--accent)' }}>● off (0)</span>
                        <span style={{ color: 'var(--accent-2)' }}>● fires (1)</span>
                        <span style={{ color: '#f87171' }}>◯ wrong</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="Misclassified" value={String(metrics.wrong)} />
                    </MetricsRow>

                    <Card title="Weighted sum + step" subtitle="one neuron">
                        <p className={styles.note}>
                            The perceptron adds up its inputs times their weights, plus a bias, and fires
                            if the total clears zero. That makes its boundary a single straight line — and
                            it learns by nudging that line toward any point it gets wrong.
                        </p>
                    </Card>

                    <Card title="The XOR wall" subtitle="why we need layers">
                        <p className={styles.note}>
                            On <strong>AND</strong> and <strong>OR</strong> the misclassified count falls
                            to zero and the line stops. On <strong>XOR</strong> it never does — no straight
                            line can split those corners. Stacking neurons into layers is what breaks the
                            wall (next chapter).
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
