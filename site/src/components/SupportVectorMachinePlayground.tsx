import { useCallback, useEffect, useRef, useState } from 'react';
import { SupportVectorMachine, Matrix, type Kernel } from 'machine-learning';
import type { Domain } from '../ml/datasets';
import { SVM_DATASETS } from '../ml/svmDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, drawPoints, type Grid } from '../viz/decisionBoundary';
import { paintMargins, drawSupportVectors } from '../viz/svm';
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
import styles from './SupportVectorMachinePlayground.module.css';

const POINTS = 100;
const GRID = 64;
const SWEEPS_PER_FRAME = 1; // SMO sweeps per animation frame — the boundary tightens, then settles

// C and gamma live on log sliders so the interesting range (fractions → tens) is easy to reach.
const sliderToC = (slider: number) => Math.pow(10, -1 + 3 * (slider / 1000));
const sliderToGamma = (slider: number) => Math.pow(10, -1 + 2 * (slider / 1000));

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
}

export function SupportVectorMachinePlayground() {
    const [datasetId, setDatasetId] = useState(SVM_DATASETS[0].id);
    const [kernel, setKernel] = useState<Kernel>('linear');
    const [cSlider, setCSlider] = useState(700); // ≈ C = 12.6
    const [gammaSlider, setGammaSlider] = useState(600); // ≈ gamma = 1.6
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ sweeps: 0, supportVectors: 0, acc: 0 });

    const c = sliderToC(cSlider);
    const gamma = sliderToGamma(gammaSlider);

    const modelRef = useRef<SupportVectorMachine | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(SVM_DATASETS[0].domain);
    const lossRef = useRef<number[]>([]);
    const sweepRef = useRef(0);
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

            const scores = model.predict(grid.matrix).toArray().map(row => row[0]);
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintMargins(offscreenRef.current, scores, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);

            const supportVectors = model.getSupportVectorIndices().map(i => data.inputs[i]);
            drawSupportVectors(ctx, supportVectors, domainRef.current, width, height);
            drawPoints(ctx, data.inputs, data.targets, domainRef.current, width, height);
        }

        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = SVM_DATASETS.find(d => d.id === datasetId) ?? SVM_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);

        const model = new SupportVectorMachine()
            .setKernel(kernel)
            .setRegularization(c)
            .setGamma(gamma)
            .setSeed(seed)
            .setNumberOfIterations(SWEEPS_PER_FRAME);

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix: new Matrix(inputs) };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        sweepRef.current = 0;
        lossRef.current = [];

        setMetrics({ sweeps: 0, supportVectors: 0, acc: 0 });
        drawAll();
    }, [datasetId, kernel, c, gamma, seed, drawAll]);

    // Debounced so dragging C / gamma (each rebuilds the model) stays smooth.
    useEffect(() => {
        const timer = setTimeout(rebuild, 60);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const stats = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return { hinge: 0, supportVectors: 0, acc: 0 };

        const scores = model.predict(data.inputMatrix).toArray();
        let correct = 0;
        let hinge = 0;
        for (let i = 0; i < scores.length; i++) {
            const score = scores[i][0];
            const label = data.targets[i][0] === 1 ? 1 : -1;
            if ((score >= 0 ? 1 : 0) === data.targets[i][0]) correct++;
            hinge += Math.max(0, 1 - label * score);
        }
        return {
            hinge: hinge / scores.length,
            supportVectors: model.getSupportVectorIndices().length,
            acc: correct / scores.length,
        };
    }, []);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        model.train(data.inputMatrix, new Matrix(data.targets));
        sweepRef.current += SWEEPS_PER_FRAME;

        const { hinge, supportVectors, acc } = stats();
        lossRef.current.push(hinge);
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawAll();

        frameRef.current += 1;
        if (frameRef.current % 3 === 0) {
            setMetrics({ sweeps: sweepRef.current, supportVectors, acc });
        }
    }, [drawAll, stats]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const dataset = SVM_DATASETS.find(d => d.id === datasetId);

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
                    label="Policy"
                    value={datasetId}
                    options={SVM_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={setDatasetId}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Select
                    label="Kernel"
                    value={kernel}
                    options={[
                        { value: 'linear', label: 'Linear (straight line)' },
                        { value: 'rbf', label: 'RBF (curved)' },
                    ]}
                    onChange={value => setKernel(value as Kernel)}
                />
                <Slider
                    label="C (margin softness)"
                    value={cSlider}
                    display={c < 1 ? c.toFixed(2) : c.toFixed(0)}
                    min={0}
                    max={1000}
                    onChange={setCSlider}
                />
                {kernel === 'rbf' && (
                    <Slider
                        label="Gamma (reach)"
                        value={gammaSlider}
                        display={gamma.toFixed(2)}
                        min={0}
                        max={1000}
                        onChange={setGammaSlider}
                    />
                )}
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={boundaryCanvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span style={{ color: 'var(--accent)' }}>● no comp</span>
                        <span style={{ color: 'var(--accent-2)' }}>● comp</span>
                        <span style={{ color: '#f8fafc' }}>◯ support vector</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Sweeps" value={String(metrics.sweeps)} />
                        <Metric label="Support vectors" value={String(metrics.supportVectors)} />
                        <Metric label="Accuracy" value={`${(metrics.acc * 100).toFixed(0)}%`} />
                    </MetricsRow>

                    <Card title="The margin" subtitle="the widest empty street">
                        <p className={styles.note}>
                            The dark band between the colours is the <strong>margin</strong> — the empty
                            street the SVM makes as wide as it can. Only the ringed{' '}
                            <strong>support vectors</strong> touch its edges and pin it in place; every
                            other point could be deleted and the boundary wouldn't move.
                        </p>
                    </Card>

                    <Card title="Hinge loss" subtitle="penalty for points in the street">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
