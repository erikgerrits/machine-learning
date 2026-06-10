import { useCallback, useEffect, useRef, useState } from 'react';
import { HierarchicalClustering, Matrix, type Linkage } from 'machine-learning';
import { HIERARCHICAL_DATASETS } from '../ml/hierarchicalDatasets';
import type { Domain } from '../ml/datasets';
import { fitCanvas } from '../viz/canvas';
import { makeGrid, type Grid } from '../viz/decisionBoundary';
import { drawClusterPoints, paintClusters } from '../viz/clusters';
import { drawDendrogram } from '../viz/dendrogram';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './HierarchicalClusteringPlayground.module.css';

const GRID = 64;
const POINTS = 140;

const argmaxRows = (matrix: Matrix): number[] => matrix.getMaximumRowIndeces().toArray().map(row => row[0]);

const LINKAGE_OPTIONS: { value: Linkage; label: string }[] = [
    { value: 'single', label: 'Single (nearest pair)' },
    { value: 'complete', label: 'Complete (farthest pair)' },
    { value: 'average', label: 'Average (mean pair)' },
];

interface TrainingData {
    inputs: number[][];
    inputMatrix: Matrix;
}

export function HierarchicalClusteringPlayground() {
    const [datasetId, setDatasetId] = useState(HIERARCHICAL_DATASETS[0].id);
    const [linkage, setLinkage] = useState<Linkage>('average');
    const [k, setK] = useState(HIERARCHICAL_DATASETS[0].recommendedClusters);
    const [seed, setSeed] = useState(0);

    const modelRef = useRef<HierarchicalClustering | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const gridRef = useRef<Grid | null>(null);
    const domainRef = useRef<Domain>(HIERARCHICAL_DATASETS[0].domain);
    const offscreenRef = useRef<HTMLCanvasElement | null>(null);

    const mapCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const dendroCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const grid = gridRef.current;
        if (!model || !data || !grid) return;

        const labels = model.getClusterLabels();

        const mapCanvas = mapCanvasRef.current;
        if (mapCanvas) {
            const { ctx, width, height } = fitCanvas(mapCanvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);

            const cellClusters = argmaxRows(model.predict(grid.matrix));
            if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
            paintClusters(offscreenRef.current, cellClusters, grid.size);
            ctx.imageSmoothingEnabled = true;
            ctx.drawImage(offscreenRef.current, 0, 0, width, height);

            drawClusterPoints(ctx, data.inputs, labels, domainRef.current, width, height);
        }

        const dendroCanvas = dendroCanvasRef.current;
        if (dendroCanvas) {
            const { ctx, width, height } = fitCanvas(dendroCanvas);
            drawDendrogram(ctx, width, height, model.getMergeHistory(), data.inputs.length, k, labels);
        }
    }, [k]);

    const rebuild = useCallback(() => {
        const dataset = HIERARCHICAL_DATASETS.find(d => d.id === datasetId) ?? HIERARCHICAL_DATASETS[0];
        const { inputs } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);

        const model = new HierarchicalClustering().setLinkage(linkage).setNumberOfClusters(k);
        model.train(inputMatrix);

        modelRef.current = model;
        dataRef.current = { inputs, inputMatrix };
        gridRef.current = makeGrid(dataset.domain, GRID);
        domainRef.current = dataset.domain;
        draw();
    }, [datasetId, linkage, seed, k, draw]);

    // Debounced so dragging k (each re-cut + redraw) stays smooth. The tree itself only changes with
    // dataset / linkage / seed; rebuilding on k too is cheap at this scale and keeps the code simple.
    useEffect(() => {
        const timer = setTimeout(rebuild, 60);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const handleDataset = (id: string) => {
        const next = HIERARCHICAL_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setK(next.recommendedClusters);
    };

    const dataset = HIERARCHICAL_DATASETS.find(d => d.id === datasetId);
    const linkageLabel = linkage.charAt(0).toUpperCase() + linkage.slice(1);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Customers"
                    value={datasetId}
                    options={HIERARCHICAL_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Select
                    label="Linkage"
                    value={linkage}
                    options={LINKAGE_OPTIONS}
                    onChange={value => setLinkage(value as Linkage)}
                />
                <Slider
                    label="Clusters (k)"
                    value={k}
                    display={String(k)}
                    min={1}
                    max={6}
                    onChange={setK}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>The tree is built once; k just chooses where to slice it. No need to fix it up front.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={mapCanvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span>● customers by group</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Clusters" value={String(k)} />
                        <Metric label="Linkage" value={linkageLabel} />
                    </MetricsRow>

                    <Card title="The dendrogram" subtitle="merge tree + cut line">
                        <canvas ref={dendroCanvasRef} className={styles.dendroCanvas} />
                        <p className={styles.note}>
                            Every leaf is one customer; each bracket is a merge, drawn at the distance the
                            two groups joined. The dashed line is the <strong>cut</strong> for the current
                            k — slide k and watch it rise or fall.
                        </p>
                    </Card>

                    <Card title="Linkage" subtitle="how 'close' two groups are">
                        <p className={styles.note}>
                            <strong>Single</strong> measures groups by their nearest pair (it can chain
                            along a curve), <strong>complete</strong> by their farthest pair (tight, round
                            groups), <strong>average</strong> by the mean over all pairs. Same data,
                            different trees.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
