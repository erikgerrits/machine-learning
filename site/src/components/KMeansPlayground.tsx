import { useCallback, useEffect, useRef, useState } from 'react';
import { KMeans, Matrix } from 'machine-learning';
import { CLUSTERING_DATASETS } from '../ml/clusteringDatasets';
import type { Domain } from '../ml/datasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, type Grid } from '../viz/decisionBoundary';
import { centroidsMoved, drawCentroids, drawClusterPoints, inertia, paintClusters } from '../viz/clusters';
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

// The Voronoi map is recomputed only when the centroids move (one Lloyd step at a time), so an
// 80×80 grid stays cheap while reading crisply.
const GRID = 80;
const POINTS = 220;

// Each cluster index column from predict() is one-hot, so argmax recovers the assigned cluster.
const argmaxRows = (matrix: Matrix): number[] => matrix.getMaximumRowIndeces().toArray().map(row => row[0]);

interface TrainingData {
    inputs: number[][];
    inputMatrix: Matrix;
}

export function KMeansPlayground() {
    const [datasetId, setDatasetId] = useState(CLUSTERING_DATASETS[0].id);
    const [k, setK] = useState(CLUSTERING_DATASETS[0].recommendedClusters);
    const [seed, setSeed] = useState(0);
    const [speed, setSpeed] = useState(4); // Lloyd steps per second
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ iteration: 0, inertia: 0, converged: false });

    const speedRef = useRef(speed);
    speedRef.current = speed;

    const modelRef = useRef<KMeans | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(CLUSTERING_DATASETS[0].domain);
    const inertiaRef = useRef<number[]>([]);
    const iterationRef = useRef(0);
    const convergedRef = useRef(false);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const frameRef = useRef(0);

    const boundaryCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const inertiaCanvasRef = useRef<HTMLCanvasElement | null>(null);

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

            const cellClusters = argmaxRows(model.predict(grid.matrix));
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintClusters(offscreenRef.current, cellClusters, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);

            const pointClusters = argmaxRows(model.predict(data.inputMatrix));
            drawClusterPoints(ctx, data.inputs, pointClusters, domainRef.current, width, height);
            drawCentroids(ctx, model.getCentroids().toArray(), domainRef.current, width, height);
        }

        const inertiaCanvas = inertiaCanvasRef.current;
        if (inertiaCanvas) {
            const { ctx, width, height } = fitCanvas(inertiaCanvas);
            drawLossCurve(ctx, width, height, inertiaRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = CLUSTERING_DATASETS.find(d => d.id === datasetId) ?? CLUSTERING_DATASETS[0];
        const { inputs } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);

        const model = new KMeans();
        model.setNumberOfClusters(k);
        model.setSeed(seed);
        model.setNumberOfIterations(0); // place the initial centroids without stepping yet…
        model.train(inputMatrix);
        model.setNumberOfIterations(1); // …so each later train() call is exactly one Lloyd step

        modelRef.current = model;
        dataRef.current = { inputs, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        iterationRef.current = 0;
        convergedRef.current = false;
        frameRef.current = 0;

        const assignments = argmaxRows(model.predict(inputMatrix));
        inertiaRef.current = [inertia(inputs, model.getCentroids().toArray(), assignments)];
        setMetrics({ iteration: 0, inertia: inertiaRef.current[0], converged: false });
        drawAll();
    }, [datasetId, seed, k, drawAll]);

    // Re-initialise whenever the dataset, seed, or k changes (all are captured by rebuild).
    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const doIteration = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;
        if (convergedRef.current) {
            setRunning(false);
            return;
        }

        const before = model.getCentroids().toArray();
        model.train(data.inputMatrix); // one Lloyd step (assign → move centroids)
        iterationRef.current += 1;
        const after = model.getCentroids().toArray();

        const assignments = argmaxRows(model.predict(data.inputMatrix));
        const score = inertia(data.inputs, after, assignments);
        inertiaRef.current.push(score);
        if (inertiaRef.current.length > 1200) inertiaRef.current.shift();

        if (centroidsMoved(before, after) === 0) convergedRef.current = true;

        drawAll();
        setMetrics({ iteration: iterationRef.current, inertia: score, converged: convergedRef.current });
        if (convergedRef.current) setRunning(false);
    }, [drawAll]);

    // The animation runs at ~60fps; throttle it down to the chosen number of steps per second so
    // each discrete Lloyd iteration is actually watchable.
    const tick = useCallback(() => {
        frameRef.current += 1;
        const framesPerStep = Math.max(1, Math.round(60 / speedRef.current));
        if (frameRef.current % framesPerStep === 0) doIteration();
    }, [doIteration]);

    useAnimationFrame(tick, running);

    const handleToggle = () => {
        if (convergedRef.current) return; // nothing left to run until Reset
        setRunning(r => !r);
    };
    const handleStep = () => {
        if (!running) doIteration();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };
    const handleDataset = (id: string) => {
        const next = CLUSTERING_DATASETS.find(d => d.id === id);
        if (!next) return;
        setRunning(false);
        setDatasetId(next.id);
        setK(next.recommendedClusters);
    };
    const handleK = (value: number) => {
        setRunning(false);
        setK(value);
    };

    const dataset = CLUSTERING_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls
                    running={running}
                    onToggle={handleToggle}
                    onStep={handleStep}
                    onReset={handleReset}
                />
                <Select
                    label="Dataset"
                    value={datasetId}
                    options={CLUSTERING_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider
                    label="Clusters (k)"
                    value={k}
                    display={String(k)}
                    min={2}
                    max={6}
                    onChange={handleK}
                />
                <Slider
                    label="Speed"
                    value={speed}
                    display={`${speed} steps / sec`}
                    min={1}
                    max={20}
                    onChange={setSpeed}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>No labels — nobody told it who's who. k-means finds the segments on its own.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={boundaryCanvasRef} className={styles.boundary} />
                    <div className={styles.activation}>
                        <span>◆ segment centre</span>
                        <span>● customers by segment</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Iteration" value={String(metrics.iteration)} />
                        <Metric label="Inertia" value={metrics.inertia.toFixed(3)} />
                        <Metric label="Status" value={metrics.converged ? 'converged' : 'running'} />
                    </MetricsRow>

                    <Card title="Lloyd's algorithm" subtitle="assign → move, repeat">
                        <p className={styles.note}>
                            Each step does two things: every customer is <strong>assigned</strong> to the
                            nearest segment centre (the coloured regions), then every centre{' '}
                            <strong>moves</strong> to the average of its customers. Repeat until nothing
                            moves — that's convergence. Change <strong>k</strong> or the{' '}
                            <strong>seed</strong> to see it land in a different grouping.
                        </p>
                    </Card>

                    <Card title="Inertia" subtitle="mean squared distance to centroid">
                        <canvas ref={inertiaCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
