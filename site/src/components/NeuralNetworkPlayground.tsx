import { useCallback, useEffect, useRef, useState } from 'react';
import { FeedforwardNeuralNetwork, Matrix } from 'machine-learning';
import { DATASETS, type Domain } from '../ml/datasets';
import { accuracy } from '../ml/metrics';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawPoints, makeGrid, paintBoundary, type Grid } from '../viz/decisionBoundary';
import { drawNetwork } from '../viz/network';
import { drawLossCurve } from '../viz/lossCurve';
import styles from './NeuralNetworkPlayground.module.css';

const POINTS = 240;

const HIDDEN_PRESETS = [
    { label: '1 × 4', layers: [4] },
    { label: '1 × 8', layers: [8] },
    { label: '1 × 12', layers: [12] },
    { label: '2 × 12', layers: [12, 12] },
    { label: '2 × 16', layers: [16, 16] },
];

const BATCH_MODES = [
    { label: 'Batch', value: 0 },
    { label: 'Mini-batch', value: 16 },
    { label: 'SGD', value: 1 },
];

// Learning rate is exposed on a log slider spanning 1e-3 … 10.
const sliderToLr = (slider: number) => Math.pow(10, -3 + 4 * (slider / 1000));
const lrToSlider = (lr: number) => Math.round(((Math.log10(lr) + 3) / 4) * 1000);

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
    targetMatrix: Matrix;
}

export function NeuralNetworkPlayground() {
    const [datasetId, setDatasetId] = useState(DATASETS[0].id);
    const [hidden, setHidden] = useState<number[]>(DATASETS[0].recommendedHidden);
    const [sliderLR, setSliderLR] = useState(lrToSlider(DATASETS[0].recommendedLr));
    const [batchSize, setBatchSize] = useState(0);
    const [seed, setSeed] = useState(0);
    const [stepsPerFrame, setStepsPerFrame] = useState(8);
    const [running, setRunning] = useState(false);
    const [gradOk, setGradOk] = useState<boolean | null>(null);
    const [metrics, setMetrics] = useState({ epoch: 0, loss: 0, acc: 0 });

    const learningRate = sliderToLr(sliderLR);

    // Latest control values, mirrored into refs so the animation loop avoids stale closures.
    const lrRef = useRef(learningRate);
    const batchRef = useRef(batchSize);
    const stepsRef = useRef(stepsPerFrame);
    lrRef.current = learningRate;
    batchRef.current = batchSize;
    stepsRef.current = stepsPerFrame;

    // Mutable training state, kept out of React so 60fps redraws don't trigger re-renders.
    const netRef = useRef<FeedforwardNeuralNetwork | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const archRef = useRef<number[]>([2, 1]);
    const domainRef = useRef<Domain>(DATASETS[0].domain);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const frameRef = useRef(0);

    const boundaryCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const netCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const drawAll = useCallback(() => {
        const net = netRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!net || !data || !grid) return;

        const boundaryCanvas = boundaryCanvasRef.current;
        if (boundaryCanvas) {
            const { ctx, width, height } = fitCanvas(boundaryCanvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const values = net.predict(grid.matrix).toArray().map(row => row[0]);
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintBoundary(offscreenRef.current, values, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);
            drawPoints(ctx, data.inputs, data.targets, domainRef.current, width, height);
        }

        const netCanvas = netCanvasRef.current;
        if (netCanvas) {
            const { ctx, width, height } = fitCanvas(netCanvas);
            drawNetwork(ctx, width, height, net.getWeightMatrices().map(m => m.toArray()), archRef.current);
        }

        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = DATASETS.find(d => d.id === datasetId) ?? DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);
        const arch = [2, ...hidden, 1];

        const net = new FeedforwardNeuralNetwork(arch, seed);
        net.setNumberOfEpochs(1); // one epoch per train() call → we drive epochs from the loop
        net.setLearningRate(lrRef.current);
        net.setBatchSize(batchRef.current);

        netRef.current = net;
        dataRef.current = { inputs, targets, inputMatrix, targetMatrix };
        gridRef.current = makeGrid(dataset.domain);
        archRef.current = arch;
        domainRef.current = dataset.domain;
        lossRef.current = [net.computeLoss(inputMatrix, targetMatrix)];
        epochRef.current = 0;

        setGradOk(net.checkGradients());
        setMetrics({
            epoch: 0,
            loss: lossRef.current[0],
            acc: accuracy(net.predict(inputMatrix).toArray(), targets),
        });
        drawAll();
    }, [datasetId, hidden, seed, drawAll]);

    // Rebuild whenever the network's identity changes (dataset / architecture / seed).
    useEffect(() => {
        rebuild();
    }, [rebuild]);

    // Learning rate and batch size apply live, without resetting the weights.
    useEffect(() => {
        netRef.current?.setLearningRate(learningRate);
    }, [learningRate]);
    useEffect(() => {
        netRef.current?.setBatchSize(batchSize);
    }, [batchSize]);

    const step = useCallback(() => {
        const net = netRef.current;
        const data = dataRef.current;
        if (!net || !data) return;

        const steps = stepsRef.current;
        for (let i = 0; i < steps; i++) net.train(data.inputMatrix, data.targetMatrix);
        epochRef.current += steps;

        const loss = net.computeLoss(data.inputMatrix, data.targetMatrix);
        lossRef.current.push(loss);
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawAll();

        // Throttle the (re-rendering) metrics text; the canvases already update every frame.
        frameRef.current += 1;
        if (frameRef.current % 4 === 0) {
            setMetrics({
                epoch: epochRef.current,
                loss,
                acc: accuracy(net.predict(data.inputMatrix).toArray(), data.targets),
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

    return (
        <div className={styles.playground}>
            <aside className={styles.controls}>
                <div className={styles.transport}>
                    <button
                        className={`${styles.btn} ${styles.primary}`}
                        onClick={() => setRunning(r => !r)}
                    >
                        {running ? '❚❚ Pause' : '▶ Train'}
                    </button>
                    <button className={styles.btn} onClick={handleStep} disabled={running}>
                        Step
                    </button>
                    <button className={styles.btn} onClick={handleReset}>
                        Reset
                    </button>
                </div>

                <label className={styles.field}>
                    <span>Dataset</span>
                    <select
                        value={datasetId}
                        onChange={e => {
                            const next = DATASETS.find(d => d.id === e.target.value);
                            if (!next) return;
                            // Selecting a dataset applies a known-good architecture + learning rate
                            // so it converges impressively out of the box (still fully tweakable).
                            setDatasetId(next.id);
                            setHidden(next.recommendedHidden);
                            setSliderLR(lrToSlider(next.recommendedLr));
                        }}
                    >
                        {DATASETS.map(d => (
                            <option key={d.id} value={d.id}>{d.label}</option>
                        ))}
                    </select>
                </label>
                <p className={styles.blurb}>{DATASETS.find(d => d.id === datasetId)?.blurb}</p>

                <label className={styles.field}>
                    <span>Hidden layers</span>
                    <select
                        value={hidden.join('-')}
                        onChange={e => {
                            const preset = HIDDEN_PRESETS.find(p => p.layers.join('-') === e.target.value);
                            if (preset) setHidden(preset.layers);
                        }}
                    >
                        {HIDDEN_PRESETS.map(p => (
                            <option key={p.label} value={p.layers.join('-')}>{p.label}</option>
                        ))}
                    </select>
                </label>

                <label className={styles.field}>
                    <span>Learning rate <em>{learningRate.toFixed(3)}</em></span>
                    <input
                        type="range"
                        min={0}
                        max={1000}
                        value={sliderLR}
                        onChange={e => setSliderLR(Number(e.target.value))}
                    />
                </label>

                <label className={styles.field}>
                    <span>Speed <em>{stepsPerFrame} epochs / frame</em></span>
                    <input
                        type="range"
                        min={1}
                        max={30}
                        value={stepsPerFrame}
                        onChange={e => setStepsPerFrame(Number(e.target.value))}
                    />
                </label>

                <label className={styles.field}>
                    <span>Gradient descent</span>
                    <select value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}>
                        {BATCH_MODES.map(m => (
                            <option key={m.label} value={m.value}>{m.label}</option>
                        ))}
                    </select>
                </label>

                <label className={styles.field}>
                    <span>Random seed</span>
                    <input
                        type="number"
                        value={seed}
                        onChange={e => setSeed(Number(e.target.value))}
                    />
                </label>
            </aside>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={boundaryCanvasRef} className={styles.boundary} />
                    <div className={styles.activation}>
                        <span style={{ color: 'var(--accent)' }}>● class 0</span>
                        <span style={{ color: 'var(--accent-2)' }}>● class 1</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <div className={styles.metrics}>
                        <div className={styles.metric}>
                            <span className={styles.metricLabel}>Epoch</span>
                            <span className={styles.metricValue}>{metrics.epoch}</span>
                        </div>
                        <div className={styles.metric}>
                            <span className={styles.metricLabel}>Loss</span>
                            <span className={styles.metricValue}>{metrics.loss.toFixed(4)}</span>
                        </div>
                        <div className={styles.metric}>
                            <span className={styles.metricLabel}>Accuracy</span>
                            <span className={styles.metricValue}>{(metrics.acc * 100).toFixed(0)}%</span>
                        </div>
                    </div>

                    {gradOk && (
                        <div className={styles.badge} title="Backprop verified against finite-difference gradients">
                            ✓ Gradients verified
                        </div>
                    )}

                    <div className={styles.card}>
                        <h3>Network <span>weights pulse as it learns</span></h3>
                        <canvas ref={netCanvasRef} className={styles.diagramCanvas} />
                    </div>

                    <div className={styles.card}>
                        <h3>Loss <span>cross-entropy per epoch</span></h3>
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </div>
                </div>
            </div>
        </div>
    );
}
