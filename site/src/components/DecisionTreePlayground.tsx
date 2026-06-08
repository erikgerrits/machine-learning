import { useCallback, useEffect, useRef, useState } from 'react';
import { DecisionTree, Matrix, type DecisionTreeNode } from 'machine-learning';
import { POLICY_DATASETS } from '../ml/policyDatasets';
import type { Domain } from '../ml/datasets';
import { fitCanvas } from '../viz/canvas';
import { drawPoints, makeGrid, paintBoundary, type Grid } from '../viz/decisionBoundary';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './DecisionTreePlayground.module.css';

const GRID = 80;
const POINTS = 160;
const FEATURES = ['wait', 'trouble'];
const DISPLAY_DEPTH = 3; // the rulebook reads cleanly to here; deeper subtrees collapse to "…"

const argmax = (values: number[]) => values.reduce((best, v, i) => (v > values[best] ? i : best), 0);

function countLeaves(node: DecisionTreeNode): number {
    if (node.distribution !== undefined) return 1;
    return countLeaves(node.left!) + countLeaves(node.right!);
}

// Render the trained tree as a nested, readable rulebook (the chapter's whole point).
function renderNode(node: DecisionTreeNode, depth: number) {
    if (node.distribution !== undefined) {
        const comp = argmax(node.distribution) === 1;
        const confidence = Math.round(Math.max(...node.distribution) * 100);
        return (
            <div className={styles.leaf} style={{ color: comp ? 'var(--accent-2)' : 'var(--muted)' }}>
                → {comp ? 'comp' : 'no comp'}
                <span className={styles.conf}>{confidence}%</span>
            </div>
        );
    }
    if (depth >= DISPLAY_DEPTH) {
        return <div className={styles.more}>… more rules</div>;
    }
    return (
        <div>
            <div className={styles.rule}>
                if {FEATURES[node.featureIndex!]} &lt; {node.threshold!.toFixed(2)}
            </div>
            <div className={styles.branch}>{renderNode(node.left!, depth + 1)}</div>
            <div className={styles.elseRule}>else</div>
            <div className={styles.branch}>{renderNode(node.right!, depth + 1)}</div>
        </div>
    );
}

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
}

export function DecisionTreePlayground() {
    const [datasetId, setDatasetId] = useState(POLICY_DATASETS[0].id);
    const [maxDepth, setMaxDepth] = useState(4);
    const [seed, setSeed] = useState(0);
    const [stats, setStats] = useState({ accuracy: 0, leaves: 0 });
    const [root, setRoot] = useState<DecisionTreeNode | null>(null);

    const modelRef = useRef<DecisionTree | null>(null);
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
        setStats({ accuracy: correct / predicted.length, leaves: countLeaves(model.getRoot()) });
        setRoot(model.getRoot());
    }, []);

    const rebuild = useCallback(() => {
        const dataset = POLICY_DATASETS.find(d => d.id === datasetId) ?? POLICY_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const oneHot = targets.map(([label]) => (label === 1 ? [0, 1] : [1, 0]));

        const model = new DecisionTree().setMaxDepth(maxDepth);
        model.train(inputMatrix, new Matrix(oneHot));

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        draw();
    }, [datasetId, maxDepth, seed, draw]);

    useEffect(() => {
        rebuild();
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
                    label="Max depth"
                    value={maxDepth}
                    display={String(maxDepth)}
                    min={1}
                    max={8}
                    onChange={setMaxDepth}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>Shallow trees underfit; deep ones carve a tiny box around every noisy case.</Hint>
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
                        <Metric label="Depth" value={String(maxDepth)} />
                        <Metric label="Train acc" value={`${(stats.accuracy * 100).toFixed(0)}%`} />
                        <Metric label="Leaves" value={String(stats.leaves)} />
                    </MetricsRow>

                    <Card title="The rulebook" subtitle="the whole policy, in plain rules">
                        <div className={styles.rulebook}>{root && renderNode(root, 0)}</div>
                    </Card>

                    <Card title="Why it's blocky" subtitle="axis-aligned splits">
                        <p className={styles.note}>
                            Every question is "is one feature above a cut?", so the regions are always
                            rectangles. The tree can box in any shape — give it enough depth — but a
                            curve only ever becomes a staircase of little boxes.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
