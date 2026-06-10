import { useCallback, useEffect, useRef, useState } from 'react';
import { PCA, Matrix } from 'machine-learning';
import { PCA_DATASETS } from '../ml/pcaDatasets';
import type { Domain } from '../ml/datasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawPcaAxes, drawPcaPoints, drawPc1Line } from '../viz/pca';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select } from './controls/Controls';
import styles from './PCAPlayground.module.css';

const POINTS = 160;

interface FitResult {
    points: number[][];
    mean: number[];
    components: number[][];
    stds: number[];
    ratio: number[];
}

export function PCAPlayground() {
    const [datasetId, setDatasetId] = useState(PCA_DATASETS[0].id);
    const [reduce, setReduce] = useState(false);
    const [seed, setSeed] = useState(0);
    const [ratio, setRatio] = useState([0, 0]);

    const fitRef = useRef<FitResult | null>(null);
    const domainRef = useRef<Domain>(PCA_DATASETS[0].domain);
    const tRef = useRef(0);
    const targetRef = useRef(0);
    const [animating, setAnimating] = useState(false);

    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const fit = fitRef.current;
        const canvas = canvasRef.current;
        if (!fit || !canvas) return;

        const { ctx, width, height } = fitCanvas(canvas);
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, width, height);

        const t = tRef.current;
        if (t > 0.01) {
            drawPc1Line(ctx, fit.mean, fit.components[0], domainRef.current, width, height);
        }
        drawPcaPoints(ctx, fit.points, fit.mean, fit.components[0], domainRef.current, width, height, t);
        drawPcaAxes(ctx, fit.mean, fit.components, fit.stds, domainRef.current, width, height);
    }, []);

    const rebuild = useCallback(() => {
        const dataset = PCA_DATASETS.find(d => d.id === datasetId) ?? PCA_DATASETS[0];
        const { inputs } = dataset.generate(seed, POINTS);

        const model = new PCA().setNumberOfComponents(2);
        model.train(new Matrix(inputs));

        const variance = model.getExplainedVariance();
        fitRef.current = {
            points: inputs,
            mean: model.getMean().toArray()[0],
            components: model.getComponents().toArray(),
            stds: variance.map(v => Math.sqrt(v)),
            ratio: model.getExplainedVarianceRatio(),
        };
        domainRef.current = dataset.domain;
        setRatio(fitRef.current.ratio);
        draw();
    }, [datasetId, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    // Animate the points collapsing onto PC1 (or springing back) whenever the toggle flips.
    useEffect(() => {
        targetRef.current = reduce ? 1 : 0;
        setAnimating(true);
    }, [reduce]);

    const tick = useCallback(() => {
        const target = targetRef.current;
        const step = 0.06;
        if (tRef.current < target) tRef.current = Math.min(target, tRef.current + step);
        else tRef.current = Math.max(target, tRef.current - step);

        draw();
        if (tRef.current === target) setAnimating(false);
    }, [draw]);

    useAnimationFrame(tick, animating);

    const dataset = PCA_DATASETS.find(d => d.id === datasetId);
    const pct = (value: number) => `${(value * 100).toFixed(0)}%`;

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Customers"
                    value={datasetId}
                    options={PCA_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={setDatasetId}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Checkbox label="Reduce to 1 axis (project onto PC1)" checked={reduce} onChange={setReduce} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>PC1 is the direction of greatest spread; PC2 is whatever's left, at a right angle.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={canvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span style={{ color: '#fb923c' }}>▬ PC1</span>
                        <span style={{ color: '#94a3b8' }}>▬ PC2</span>
                        <span style={{ color: '#38bdf8' }}>● customers</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="PC1 variance" value={pct(ratio[0] ?? 0)} />
                        <Metric label="PC2 variance" value={pct(ratio[1] ?? 0)} />
                    </MetricsRow>

                    <Card title="Principal components" subtitle="axes of greatest variance">
                        <p className={styles.note}>
                            PCA rotates to find the axis the data spreads along most (<strong>PC1</strong>,
                            the long amber arrow), then the best one at a right angle to it
                            (<strong>PC2</strong>). The arrows' lengths are how much variance each holds.
                        </p>
                    </Card>

                    <Card title="Keeping just PC1" subtitle="2-D → 1-D">
                        <p className={styles.note}>
                            Tick the box and every point collapses onto the PC1 line — that's the reduction.
                            When PC1 holds most of the variance, the squashed points barely move and almost
                            nothing is lost; when the cloud is round, they fall a long way and you can see
                            the cost.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
