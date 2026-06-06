import { useCallback, useEffect, useRef, useState } from 'react';
import { NearestNeighbors, Matrix } from 'machine-learning';
import { DATASETS, type Domain } from '../ml/datasets';
import { accuracy } from '../ml/metrics';
import { fitCanvas } from '../viz/canvas';
import { drawPoints, makeGrid, paintBoundary, type Grid } from '../viz/decisionBoundary';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './LogisticRegressionPlayground.module.css';

// k-NN over a grid is O(cells × points); 70² keeps each recompute well under ~150ms.
const GRID = 70;
const POINTS = 200;

// k-NN shines exactly where a straight line fails — show the curved datasets.
const KNN_DATASETS = ['moons', 'circles', 'spiral', 'blobs']
    .map(id => DATASETS.find(d => d.id === id))
    .filter((d): d is (typeof DATASETS)[number] => Boolean(d));

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
}

export function NearestNeighborsPlayground() {
    const [datasetId, setDatasetId] = useState(KNN_DATASETS[0].id);
    const [k, setK] = useState(1);
    const [seed, setSeed] = useState(0);
    const [acc, setAcc] = useState(0);

    const kRef = useRef(k);
    kRef.current = k;

    const modelRef = useRef<NearestNeighbors | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(KNN_DATASETS[0].domain);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const boundaryCanvasRef = useRef<HTMLCanvasElement | null>(null);

    // k-NN has no training loop: it just stores the data and votes. So we recompute the whole
    // boundary on demand (dataset / k / seed) rather than animating epochs.
    const recompute = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        model.setNumberOfNeighbors(kRef.current);

        const boundaryCanvas = boundaryCanvasRef.current;
        if (boundaryCanvas) {
            const { ctx, width, height } = fitCanvas(boundaryCanvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const values = model.predict(grid.matrix).toArray().map(row => row[0]);
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintBoundary(offscreenRef.current, values, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);
            drawPoints(ctx, data.inputs, data.targets, domainRef.current, width, height);
        }

        setAcc(accuracy(model.predict(data.inputMatrix).toArray(), data.targets));
    }, []);

    const rebuild = useCallback(() => {
        const dataset = KNN_DATASETS.find(d => d.id === datasetId) ?? KNN_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);

        const model = new NearestNeighbors();
        model.setNumberOfNeighbors(kRef.current);
        model.train(inputMatrix, new Matrix(targets)); // "training" = memorising the examples

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        recompute();
    }, [datasetId, seed, recompute]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    // Recompute when k changes, debounced so dragging the slider stays smooth.
    useEffect(() => {
        const timer = setTimeout(() => recompute(), 60);
        return () => clearTimeout(timer);
    }, [k, recompute]);

    const dataset = KNN_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Dataset"
                    value={datasetId}
                    options={KNN_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={setDatasetId}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider
                    label="Neighbours (k)"
                    value={k}
                    display={String(k)}
                    min={1}
                    max={25}
                    onChange={setK}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>No training to run — k-NN classifies straight from the stored points.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={boundaryCanvasRef} className={styles.boundary} />
                    <div className={styles.activation}>
                        <span style={{ color: 'var(--accent)' }}>● class 0</span>
                        <span style={{ color: 'var(--accent-2)' }}>● class 1</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Neighbours" value={String(k)} />
                        <Metric label="Train acc" value={`${(acc * 100).toFixed(0)}%`} />
                        <Metric label="Points" value={String(POINTS)} />
                    </MetricsRow>

                    <Card title="Lazy learning" subtitle="no model, just data">
                        <p className={styles.note}>
                            k-NN draws no equation and runs no gradient descent — it simply keeps every
                            example and labels each new point by a vote of its <strong>k</strong> nearest
                            neighbours. Small <strong>k</strong> gives jagged, overfit regions; larger
                            <strong> k</strong> smooths the boundary. Unlike a straight-line classifier,
                            it bends around moons, rings, and spirals with ease.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
