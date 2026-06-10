import { useCallback, useEffect, useRef, useState } from 'react';
import { AnomalyDetector, Matrix } from 'machine-learning';
import { ANOMALY_DATASETS } from '../ml/anomalyDatasets';
import type { Domain } from '../ml/datasets';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, type Grid } from '../viz/decisionBoundary';
import { paintAnomalyRegion, drawAnomalyPoints } from '../viz/anomaly';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './AnomalyDetectionPlayground.module.css';

const GRID = 64;
const POINTS = 200;

interface TrainingData {
    inputs: number[][];
    inputMatrix: Matrix;
}

export function AnomalyDetectionPlayground() {
    const [datasetId, setDatasetId] = useState(ANOMALY_DATASETS[0].id);
    const [thresholdSlider, setThresholdSlider] = useState(ANOMALY_DATASETS[0].threshold * 10);
    const [seed, setSeed] = useState(0);
    const [anomalies, setAnomalies] = useState(0);

    const threshold = thresholdSlider / 10;

    const modelRef = useRef<AnomalyDetector | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(ANOMALY_DATASETS[0].domain);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        const canvas = canvasRef.current;
        if (canvas) {
            const { ctx, width, height } = fitCanvas(canvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const cellScores = model.score(grid.matrix).toArray().map(row => row[0]);
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintAnomalyRegion(offscreenRef.current, cellScores, model.getThreshold(), grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);

            const flags = model.predict(data.inputMatrix).toArray().map(row => row[0]);
            drawAnomalyPoints(ctx, data.inputs, flags, domainRef.current, width, height);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = ANOMALY_DATASETS.find(d => d.id === datasetId) ?? ANOMALY_DATASETS[0];
        const { inputs } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);

        const model = new AnomalyDetector().setThreshold(threshold);
        model.train(inputMatrix);

        modelRef.current = model;
        dataRef.current = { inputs, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;

        const flagged = model.predict(inputMatrix).toArray().reduce((sum, row) => sum + row[0], 0);
        setAnomalies(flagged);
        draw();
    }, [datasetId, threshold, seed, draw]);

    // Debounced so dragging the threshold (each re-scores + redraws) stays smooth.
    useEffect(() => {
        const timer = setTimeout(rebuild, 60);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const handleDataset = (id: string) => {
        const next = ANOMALY_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setThresholdSlider(next.threshold * 10);
    };

    const dataset = ANOMALY_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Transactions"
                    value={datasetId}
                    options={ANOMALY_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider
                    label="Threshold (sensitivity)"
                    value={thresholdSlider}
                    display={threshold.toFixed(1)}
                    min={15}
                    max={50}
                    onChange={setThresholdSlider}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>Lower the threshold and the normal region shrinks — more transactions get flagged.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={canvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span style={{ color: '#38bdf8' }}>● normal</span>
                        <span style={{ color: '#f87171' }}>● flagged</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Flagged" value={`${anomalies} / ${POINTS}`} />
                        <Metric label="Threshold" value={threshold.toFixed(1)} />
                    </MetricsRow>

                    <Card title="Modelling normal" subtitle="mean + covariance">
                        <p className={styles.note}>
                            The detector fits one Gaussian to all the data — a centre and a spread. The
                            blue region is where that bell curve says points are normal; its shape (round
                            or tilted) follows the data's own covariance.
                        </p>
                    </Card>

                    <Card title="Mahalanobis distance" subtitle="standard deviations, not units">
                        <p className={styles.note}>
                            A point is scored by how many standard deviations it sits from the centre — in
                            the data's own shape. The threshold is the ring where that score tips a
                            transaction from normal (blue) to flagged (red).
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
