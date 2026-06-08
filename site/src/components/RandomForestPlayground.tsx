import { useCallback, useEffect, useRef, useState } from 'react';
import { RandomForest, Matrix } from 'machine-learning';
import { POLICY_DATASETS } from '../ml/policyDatasets';
import type { Domain } from '../ml/datasets';
import { fitCanvas } from '../viz/canvas';
import { drawPoints, makeGrid, paintBoundary, type Grid } from '../viz/decisionBoundary';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './RandomForestPlayground.module.css';

const GRID = 70;
const POINTS = 120;
const MAX_DEPTH = 6; // deep enough that a single tree is jagged, so the crowd's smoothing shows

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
}

export function RandomForestPlayground() {
    const [datasetId, setDatasetId] = useState(POLICY_DATASETS[0].id);
    const [numberOfTrees, setNumberOfTrees] = useState(30);
    const [seed, setSeed] = useState(0);
    const [accuracy, setAccuracy] = useState(0);

    const modelRef = useRef<RandomForest | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(POLICY_DATASETS[0].domain);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);
    const boundaryCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        const canvas = boundaryCanvasRef.current;
        if (canvas) {
            const { ctx, width, height } = fitCanvas(canvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const values = model.predict(grid.matrix).toArray().map(row => row[1]); // averaged P(comp)
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintBoundary(offscreenRef.current, values, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);
            drawPoints(ctx, data.inputs, data.targets, domainRef.current, width, height);
        }

        const predicted = model.predict(data.inputMatrix).getMaximumRowIndeces().toArray().map(row => row[0]);
        const correct = predicted.filter((cls, i) => cls === data.targets[i][0]).length;
        setAccuracy(correct / predicted.length);
    }, []);

    const rebuild = useCallback(() => {
        const dataset = POLICY_DATASETS.find(d => d.id === datasetId) ?? POLICY_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const oneHot = targets.map(([label]) => (label === 1 ? [0, 1] : [1, 0]));

        const model = new RandomForest()
            .setNumberOfTrees(numberOfTrees)
            .setMaxDepth(MAX_DEPTH)
            .setSeed(seed);
        model.train(inputMatrix, new Matrix(oneHot));

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        draw();
    }, [datasetId, numberOfTrees, seed, draw]);

    // Debounced so dragging the "trees" slider (each step retrains the whole forest) stays smooth.
    useEffect(() => {
        const timer = setTimeout(rebuild, 50);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const dataset = POLICY_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Policy"
                    value={datasetId}
                    options={POLICY_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={setDatasetId}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider
                    label="Trees"
                    value={numberOfTrees}
                    display={String(numberOfTrees)}
                    min={1}
                    max={60}
                    onChange={setNumberOfTrees}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>At 1 tree it's a single jagged judge (Chapter 8). Add trees and the vote smooths.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={boundaryCanvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span style={{ color: 'var(--accent)' }}>● no comp</span>
                        <span style={{ color: 'var(--accent-2)' }}>● comp</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Trees" value={String(numberOfTrees)} />
                        <Metric label="Train acc" value={`${(accuracy * 100).toFixed(0)}%`} />
                        <Metric label="Depth" value={String(MAX_DEPTH)} />
                    </MetricsRow>

                    <Card title="The price of the crowd" subtitle="stability vs. a rulebook">
                        <p className={styles.note}>
                            Each tree trains on a different random resample of the cases, then they
                            average their votes — so one tree's quirks cancel out and the boundary
                            barely flinches when you reseed. The cost: there's no single tree to read
                            any more. You traded the rulebook for steadiness.
                        </p>
                    </Card>

                    <Card title="Watch it settle" subtitle="drag the trees up">
                        <p className={styles.note}>
                            One tree is all hard edges and little islands. As the committee grows, the
                            soft blend of many votes rounds the policy into something stable.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
