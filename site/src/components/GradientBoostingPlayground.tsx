import { useCallback, useEffect, useRef, useState } from 'react';
import { GradientBoosting, Matrix } from 'machine-learning';
import { POLICY_DATASETS } from '../ml/policyDatasets';
import type { Domain } from '../ml/datasets';
import { fitCanvas } from '../viz/canvas';
import { drawPoints, makeGrid, paintBoundary, type Grid } from '../viz/decisionBoundary';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './GradientBoostingPlayground.module.css';

const GRID = 70;
const POINTS = 120;
const MAX_DEPTH = 3; // shallow trees, but deep enough to catch feature interactions (e.g. quadrants)

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
}

export function GradientBoostingPlayground() {
    const [datasetId, setDatasetId] = useState(POLICY_DATASETS[0].id);
    const [rounds, setRounds] = useState(40);
    const [rateSlider, setRateSlider] = useState(30); // learning rate = rateSlider / 100
    const [seed, setSeed] = useState(0);
    const [accuracy, setAccuracy] = useState(0);

    const learningRate = rateSlider / 100;

    const modelRef = useRef<GradientBoosting | null>(null);
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

            const values = model.predict(grid.matrix).toArray().map(row => row[1]); // P(comp)
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

        const model = new GradientBoosting()
            .setNumberOfTrees(rounds)
            .setLearningRate(learningRate)
            .setMaxDepth(MAX_DEPTH);
        model.train(inputMatrix, new Matrix(oneHot));

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        draw();
    }, [datasetId, rounds, learningRate, seed, draw]);

    // Debounced so dragging the rounds/rate sliders (each step re-boosts from scratch) stays smooth.
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
                    label="Rounds"
                    value={rounds}
                    display={String(rounds)}
                    min={1}
                    max={80}
                    onChange={setRounds}
                />
                <Slider
                    label="Learning rate"
                    value={rateSlider}
                    display={learningRate.toFixed(2)}
                    min={1}
                    max={100}
                    onChange={setRateSlider}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>At a few rounds it's a vague blur; each round sharpens it. Push it far and it starts chasing noise.</Hint>
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
                        <Metric label="Rounds" value={String(rounds)} />
                        <Metric label="Train acc" value={`${(accuracy * 100).toFixed(0)}%`} />
                        <Metric label="Rate" value={learningRate.toFixed(2)} />
                    </MetricsRow>

                    <Card title="Fixing the leftovers" subtitle="trees in sequence">
                        <p className={styles.note}>
                            It starts at the base rate — one flat guess — then each round fits a small
                            tree to whatever's still wrong and adds it in, shrunk by the learning rate.
                            Round by round the boundary forms out of the errors of the round before.
                        </p>
                    </Card>

                    <Card title="Knowing when to stop" subtitle="rounds × rate">
                        <p className={styles.note}>
                            More rounds (or a bigger rate) fit harder — too hard, eventually, boxing in
                            the noisy cases just like a too-deep tree. A small rate with the right number
                            of rounds is the sweet spot.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
