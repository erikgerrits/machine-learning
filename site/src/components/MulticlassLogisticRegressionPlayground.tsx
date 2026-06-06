import { useCallback, useEffect, useRef, useState } from 'react';
import { MulticlassLogisticRegression, Matrix } from 'machine-learning';
import { MULTICLASS_DATASETS } from '../ml/multiclassDatasets';
import type { Domain } from '../ml/datasets';
import { argmaxAccuracy, crossEntropyMulticlass } from '../ml/metrics';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, type Grid } from '../viz/decisionBoundary';
import { CLASS_HEX, drawClassPoints, paintMulticlass } from '../viz/multiclass';
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
import styles from './LogisticRegressionPlayground.module.css';

const POINTS = 240;

const sliderToLr = (slider: number) => Math.pow(10, -3 + 4 * (slider / 1000));
const lrToSlider = (lr: number) => Math.round(((Math.log10(lr) + 3) / 4) * 1000);

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
    targetMatrix: Matrix;
    trueClass: number[];
}

export function MulticlassLogisticRegressionPlayground() {
    const [datasetId, setDatasetId] = useState(MULTICLASS_DATASETS[0].id);
    const [sliderLR, setSliderLR] = useState(lrToSlider(MULTICLASS_DATASETS[0].recommendedLr));
    const [seed, setSeed] = useState(0);
    const [stepsPerFrame, setStepsPerFrame] = useState(8);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epoch: 0, loss: 0, acc: 0 });

    const learningRate = sliderToLr(sliderLR);
    const lrRef = useRef(learningRate);
    const stepsRef = useRef(stepsPerFrame);
    lrRef.current = learningRate;
    stepsRef.current = stepsPerFrame;

    const modelRef = useRef<MulticlassLogisticRegression | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(MULTICLASS_DATASETS[0].domain);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const frameRef = useRef(0);

    const boundaryCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const drawAll = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        const boundaryCanvas = boundaryCanvasRef.current;
        if (boundaryCanvas) {
            const { ctx, width, height } = fitCanvas(boundaryCanvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const values = model.predict(grid.matrix).toArray();
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintMulticlass(offscreenRef.current, values, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);
            drawClassPoints(ctx, data.inputs, data.trueClass, domainRef.current, width, height);
        }

        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = MULTICLASS_DATASETS.find(d => d.id === datasetId) ?? MULTICLASS_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);
        const trueClass = targetMatrix.getMaximumRowIndeces().toArray().map(row => row[0]);

        const model = new MulticlassLogisticRegression();
        model.setNumberOfEpochs(1); // one epoch per train() call → epochs driven by the loop
        model.setLearningRate(lrRef.current);
        // The first train() lazily creates the per-class classifiers (so predict() is valid).
        model.train(inputMatrix, targetMatrix);

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix, targetMatrix, trueClass };
        gridRef.current = makeGrid(dataset.domain);
        domainRef.current = dataset.domain;
        epochRef.current = 0;

        const predictions = model.predict(inputMatrix);
        lossRef.current = [crossEntropyMulticlass(predictions.toArray(), targets)];
        const predClass = predictions.getMaximumRowIndeces().toArray().map(row => row[0]);
        setMetrics({ epoch: 0, loss: lossRef.current[0], acc: argmaxAccuracy(predClass, trueClass) });
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

        const predictions = model.predict(data.inputMatrix);
        const loss = crossEntropyMulticlass(predictions.toArray(), data.targets);
        lossRef.current.push(loss);
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawAll();

        frameRef.current += 1;
        if (frameRef.current % 4 === 0) {
            const predClass = predictions.getMaximumRowIndeces().toArray().map(row => row[0]);
            setMetrics({ epoch: epochRef.current, loss, acc: argmaxAccuracy(predClass, data.trueClass) });
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
        const next = MULTICLASS_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setSliderLR(lrToSlider(next.recommendedLr));
    };

    const dataset = MULTICLASS_DATASETS.find(d => d.id === datasetId);

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
                    options={MULTICLASS_DATASETS.map(d => ({ value: d.id, label: d.label }))}
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
                    max={30}
                    onChange={setStepsPerFrame}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={boundaryCanvasRef} className={styles.boundary} />
                    <div className={styles.activation}>
                        <span style={{ color: CLASS_HEX[0] }}>● class 0</span>
                        <span style={{ color: CLASS_HEX[1] }}>● class 1</span>
                        <span style={{ color: CLASS_HEX[2] }}>● class 2</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="Loss" value={metrics.loss.toFixed(3)} />
                        <Metric label="Accuracy" value={`${(metrics.acc * 100).toFixed(0)}%`} />
                    </MetricsRow>

                    <Card title="One-vs-rest" subtitle="3 binary classifiers">
                        <p className={styles.note}>
                            One logistic classifier is trained per class — "this class vs. everything
                            else" — and each point is given to whichever classifier is most confident.
                        </p>
                    </Card>

                    <Card title="Loss" subtitle="cross-entropy per epoch">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
