import { useCallback, useEffect, useRef, useState } from 'react';
import { DBSCAN, Matrix } from 'machine-learning';
import { DBSCAN_DATASETS } from '../ml/dbscanDatasets';
import type { Domain } from '../ml/datasets';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, type Grid } from '../viz/decisionBoundary';
import { paintDensity, drawDbscanPoints } from '../viz/dbscan';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './DBSCANPlayground.module.css';

const GRID = 64;
const POINTS = 150;

interface TrainingData {
    inputs: number[][];
    inputMatrix: Matrix;
}

export function DBSCANPlayground() {
    const [datasetId, setDatasetId] = useState(DBSCAN_DATASETS[0].id);
    const [epsilonSlider, setEpsilonSlider] = useState(DBSCAN_DATASETS[0].epsilon * 100);
    const [minPoints, setMinPoints] = useState(DBSCAN_DATASETS[0].minPoints);
    const [seed, setSeed] = useState(0);
    const [metrics, setMetrics] = useState({ clusters: 0, noise: 0 });

    const epsilon = epsilonSlider / 100;

    const modelRef = useRef<DBSCAN | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(DBSCAN_DATASETS[0].domain);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const mapCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        const canvas = mapCanvasRef.current;
        if (canvas) {
            const { ctx, width, height } = fitCanvas(canvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const cellLabels = model.predict(grid.matrix).toArray().map(row => row[0]);
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintDensity(offscreenRef.current, cellLabels, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);

            drawDbscanPoints(ctx, data.inputs, model.getLabels(), domainRef.current, width, height);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = DBSCAN_DATASETS.find(d => d.id === datasetId) ?? DBSCAN_DATASETS[0];
        const { inputs } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);

        const model = new DBSCAN().setEpsilon(epsilon).setMinPoints(minPoints);
        model.train(inputMatrix);

        modelRef.current = model;
        dataRef.current = { inputs, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;

        const noise = model.getLabels().filter(label => label < 0).length;
        setMetrics({ clusters: model.getClusterCount(), noise });
        draw();
    }, [datasetId, epsilon, minPoints, seed, draw]);

    // Debounced so dragging epsilon / minPoints (each re-runs DBSCAN) stays smooth.
    useEffect(() => {
        const timer = setTimeout(rebuild, 60);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const handleDataset = (id: string) => {
        const next = DBSCAN_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setEpsilonSlider(next.epsilon * 100);
        setMinPoints(next.minPoints);
    };

    const dataset = DBSCAN_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Customers"
                    value={datasetId}
                    options={DBSCAN_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider
                    label="Epsilon (radius)"
                    value={epsilonSlider}
                    display={epsilon.toFixed(2)}
                    min={4}
                    max={60}
                    onChange={setEpsilonSlider}
                />
                <Slider
                    label="Min points"
                    value={minPoints}
                    display={String(minPoints)}
                    min={2}
                    max={12}
                    onChange={setMinPoints}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>No k. The number of groups falls out of the density — and stragglers are left as noise.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={mapCanvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span>● clustered</span>
                        <span style={{ color: 'rgba(148, 163, 184, 0.95)' }}>✕ noise</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Clusters found" value={String(metrics.clusters)} />
                        <Metric label="Noise points" value={String(metrics.noise)} />
                    </MetricsRow>

                    <Card title="Density, not distance" subtitle="core → reachable → noise">
                        <p className={styles.note}>
                            A point is a <strong>core</strong> point when at least <em>min points</em>
                            neighbours sit within <em>epsilon</em>. Clusters grow by chaining through
                            core points, so they take any dense shape. Points reachable from no core are{' '}
                            <strong>noise</strong> — drawn as grey ✕.
                        </p>
                    </Card>

                    <Card title="Two knobs" subtitle="epsilon × min points">
                        <p className={styles.note}>
                            Raise <strong>epsilon</strong> and groups reach farther and merge; lower it
                            and they splinter and more points become noise. Raise <strong>min points</strong>
                            and it takes a denser pile to count as a cluster. There is no k to set.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
