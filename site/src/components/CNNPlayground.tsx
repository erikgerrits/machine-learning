import { useCallback, useEffect, useRef, useState } from 'react';
import { ConvolutionalNeuralNetwork, Matrix } from 'machine-learning';
import { CNN_DATASETS } from '../ml/cnnDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawGrid, label } from '../viz/cnn';
import { drawLossCurve } from '../viz/lossCurve';
import { Badge, Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './CNNPlayground.module.css';

const POINTS = 90;
const FILTER_PRESETS = [4, 6, 8];

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
    targetMatrix: Matrix;
    size: number;
    classNames: string[];
}

export function CNNPlayground() {
    const [datasetId, setDatasetId] = useState(CNN_DATASETS[0].id);
    const [filters, setFilters] = useState(CNN_DATASETS[0].recommendedFilters);
    const [rateSlider, setRateSlider] = useState(Math.round(CNN_DATASETS[0].recommendedLr * 100));
    const [stepsPerFrame, setStepsPerFrame] = useState(3);
    const [example, setExample] = useState(0);
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [gradOk, setGradOk] = useState<boolean | null>(null);
    const [metrics, setMetrics] = useState({ epoch: 0, loss: 0, acc: 0 });

    const learningRate = rateSlider / 100;

    const lrRef = useRef(learningRate);
    const stepsRef = useRef(stepsPerFrame);
    const exampleRef = useRef(example);
    lrRef.current = learningRate;
    stepsRef.current = stepsPerFrame;
    exampleRef.current = example;

    const modelRef = useRef<ConvolutionalNeuralNetwork | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const vizCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const reshape = (row: number[], size: number) => {
        const g: number[][] = [];
        for (let i = 0; i < size; i++) g.push(row.slice(i * size, (i + 1) * size));
        return g;
    };

    const drawViz = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        const canvas = vizCanvasRef.current;
        if (!model || !data || !canvas) return;

        const { ctx, width, height } = fitCanvas(canvas);
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, width, height);

        const idx = Math.min(exampleRef.current, data.inputs.length - 1);
        const image = data.inputs[idx];
        const learnedFilters = model.getFilters();
        const maps = model.getConvMaps(image);

        // Input image (top-left) + its predicted class.
        label(ctx, 'Input image', 8, 16);
        drawGrid(ctx, 8, 22, 96, reshape(image, data.size));
        const probs = model.predict(new Matrix([image])).toArray()[0];
        const predicted = probs.indexOf(Math.max(...probs));
        const trueClass = data.targets[idx].indexOf(1);
        ctx.fillStyle = predicted === trueClass ? '#34d399' : '#f87171';
        ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(`pred: ${data.classNames[predicted]}`, 8, 140);
        ctx.fillStyle = 'rgba(148,163,184,0.8)';
        ctx.fillText(`true: ${data.classNames[trueClass]}`, 8, 156);

        // Learned filters (a row).
        const fx = 130;
        label(ctx, `Learned filters (${learnedFilters.length})`, fx, 16);
        const fbox = 34;
        learnedFilters.forEach((f, k) => drawGrid(ctx, fx + k * (fbox + 8), 22, fbox, f, true));

        // Feature maps for this image (a row), aligned under the filters.
        label(ctx, 'Feature maps (what each filter sees here)', fx, 86);
        const mbox = 44;
        maps.forEach((m, k) => drawGrid(ctx, fx + k * (mbox + 8), 92, mbox, m));
    }, []);

    const drawLoss = useCallback(() => {
        const canvas = lossCanvasRef.current;
        if (!canvas) return;
        const { ctx, width, height } = fitCanvas(canvas);
        drawLossCurve(ctx, width, height, lossRef.current);
    }, []);

    const rebuild = useCallback(() => {
        const dataset = CNN_DATASETS.find(d => d.id === datasetId) ?? CNN_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);

        const model = new ConvolutionalNeuralNetwork()
            .setInputShape(dataset.size, dataset.size)
            .setFilterCount(filters)
            .setLearningRate(lrRef.current)
            .setNumberOfEpochs(1)
            .setSeed(seed);
        model.train(inputMatrix, targetMatrix); // one pass to initialise the weights + shapes

        modelRef.current = model;
        dataRef.current = { inputs, targets, inputMatrix, targetMatrix, size: dataset.size, classNames: dataset.classNames };
        lossRef.current = [model.computeLoss(inputMatrix, targetMatrix)];
        epochRef.current = 1;
        frameRef.current = 0;

        setGradOk(model.checkGradients());
        setMetrics({ epoch: 1, loss: lossRef.current[0], acc: accuracy(model, inputMatrix, targets) });
        drawViz();
        drawLoss();
    }, [datasetId, filters, seed, drawViz, drawLoss]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    useEffect(() => {
        // Live LR changes shouldn't reset training.
        if (modelRef.current) modelRef.current.setLearningRate(learningRate);
    }, [learningRate]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        for (let i = 0; i < stepsRef.current; i++) model.train(data.inputMatrix, data.targetMatrix);
        epochRef.current += stepsRef.current;

        const loss = model.computeLoss(data.inputMatrix, data.targetMatrix);
        lossRef.current.push(loss);
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawViz();
        drawLoss();

        frameRef.current += 1;
        if (frameRef.current % 3 === 0) {
            setMetrics({ epoch: epochRef.current, loss, acc: accuracy(model, data.inputMatrix, data.targets) });
        }
    }, [drawViz, drawLoss]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };
    const handleDataset = (id: string) => {
        const next = CNN_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setFilters(next.recommendedFilters);
        setRateSlider(Math.round(next.recommendedLr * 100));
    };
    // Re-draw the viz when the selected example changes (no retrain).
    useEffect(() => { drawViz(); }, [example, drawViz]);

    const dataset = CNN_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls running={running} onToggle={() => setRunning(r => !r)} onStep={handleStep} onReset={handleReset} />
                <Select
                    label="Images"
                    value={datasetId}
                    options={CNN_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Select
                    label="Filters"
                    value={String(filters)}
                    options={FILTER_PRESETS.map(f => ({ value: String(f), label: `${f} filters` }))}
                    onChange={v => setFilters(Number(v))}
                />
                <Slider label="Learning rate" value={rateSlider} display={learningRate.toFixed(2)} min={2} max={60} onChange={setRateSlider} />
                <Slider label="Speed" value={stepsPerFrame} display={`${stepsPerFrame} epochs / frame`} min={1} max={8} onChange={setStepsPerFrame} />
                <Slider label="Show example" value={example} display={`#${example}`} min={0} max={POINTS - 1} onChange={setExample} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.vizWrap}>
                    <canvas ref={vizCanvasRef} className={styles.viz} />
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="Loss" value={metrics.loss.toFixed(3)} />
                        <Metric label="Accuracy" value={`${(metrics.acc * 100).toFixed(0)}%`} />
                    </MetricsRow>

                    {gradOk && <Badge>✓ Gradients verified</Badge>}

                    <Card title="Loss" subtitle="cross-entropy per epoch">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>

                    <Card title="What to watch" subtitle="filters become detectors">
                        <p className={styles.note}>
                            Each filter starts as random noise and sharpens into a little edge detector —
                            one for horizontal strokes, one for vertical, and so on. The feature maps light
                            up wherever a filter's pattern appears, so the same stroke is found no matter
                            where it sits.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}

function accuracy(model: ConvolutionalNeuralNetwork, inputMatrix: Matrix, targets: number[][]): number {
    const predicted = model.predict(inputMatrix).getMaximumRowIndeces().toArray().map(row => row[0]);
    let correct = 0;
    for (let i = 0; i < predicted.length; i++) {
        if (predicted[i] === targets[i].indexOf(1)) correct++;
    }
    return predicted.length > 0 ? correct / predicted.length : 0;
}
