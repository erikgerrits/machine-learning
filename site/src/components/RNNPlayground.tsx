import { useCallback, useEffect, useRef, useState } from 'react';
import { RecurrentNeuralNetwork, Matrix } from 'machine-learning';
import { RNN_DATASETS } from '../ml/rnnDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawEmbeddings } from '../viz/rnn';
import { drawLossCurve } from '../viz/lossCurve';
import { Badge, Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './RNNPlayground.module.css';

const POINTS = 80;
const SAMPLE_COUNT = 4;

interface TrainingData {
    inputs: number[][];
    targets: number[][];
    inputMatrix: Matrix;
    targetMatrix: Matrix;
    vocab: string[];
    classNames: string[];
    positive: Set<number>;
    negative: Set<number>;
}

export function RNNPlayground() {
    const [datasetId, setDatasetId] = useState(RNN_DATASETS[0].id);
    const [rateSlider, setRateSlider] = useState(10); // lr = rateSlider / 100
    const [stepsPerFrame, setStepsPerFrame] = useState(4);
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [gradOk, setGradOk] = useState<boolean | null>(null);
    const [metrics, setMetrics] = useState({ epoch: 0, loss: 0, acc: 0 });
    const [samples, setSamples] = useState<{ words: string; predicted: number; correct: boolean }[]>([]);

    const learningRate = rateSlider / 100;
    const lrRef = useRef(learningRate);
    const stepsRef = useRef(stepsPerFrame);
    lrRef.current = learningRate;
    stepsRef.current = stepsPerFrame;

    const modelRef = useRef<RecurrentNeuralNetwork | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const embedCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const refreshStats = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        const preds = model.predict(data.inputMatrix).getMaximumRowIndeces().toArray().map(r => r[0]);
        const truth = data.targets.map(t => t.indexOf(1));
        const correct = preds.filter((p, i) => p === truth[i]).length;

        const sample = [];
        for (let i = 0; i < SAMPLE_COUNT; i++) {
            const idx = Math.floor((i / SAMPLE_COUNT) * data.inputs.length);
            const words = data.inputs[idx].filter(t => t > 0).map(t => data.vocab[Math.round(t)]).join(' ');
            sample.push({ words, predicted: preds[idx], correct: preds[idx] === truth[idx] });
        }
        setSamples(sample);
        setMetrics({ epoch: epochRef.current, loss: model.computeLoss(data.inputMatrix, data.targetMatrix), acc: correct / preds.length });
    }, []);

    const drawAll = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        const canvas = embedCanvasRef.current;
        if (canvas) {
            const { ctx, width, height } = fitCanvas(canvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);
            drawEmbeddings(ctx, width, height, model.getEmbeddings(), data.vocab, data.positive, data.negative);
        }
        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const dataset = RNN_DATASETS.find(d => d.id === datasetId) ?? RNN_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);

        const model = new RecurrentNeuralNetwork()
            .setVocabSize(dataset.vocab.length)
            .setEmbeddingDim(2)
            .setHiddenSize(12)
            .setLearningRate(lrRef.current)
            .setNumberOfEpochs(1)
            .setSeed(seed);
        model.train(inputMatrix, targetMatrix); // one pass to initialise weights/shapes

        modelRef.current = model;
        dataRef.current = {
            inputs, targets, inputMatrix, targetMatrix,
            vocab: dataset.vocab, classNames: dataset.classNames,
            positive: new Set(dataset.positiveTokens), negative: new Set(dataset.negativeTokens),
        };
        lossRef.current = [model.computeLoss(inputMatrix, targetMatrix)];
        epochRef.current = 1;
        frameRef.current = 0;

        setGradOk(model.checkGradients());
        drawAll();
        refreshStats();
    }, [datasetId, seed, drawAll, refreshStats]);

    useEffect(() => { rebuild(); }, [rebuild]);
    useEffect(() => { if (modelRef.current) modelRef.current.setLearningRate(learningRate); }, [learningRate]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        for (let i = 0; i < stepsRef.current; i++) model.train(data.inputMatrix, data.targetMatrix);
        epochRef.current += stepsRef.current;

        lossRef.current.push(model.computeLoss(data.inputMatrix, data.targetMatrix));
        if (lossRef.current.length > 1200) lossRef.current.shift();

        drawAll();
        frameRef.current += 1;
        if (frameRef.current % 3 === 0) refreshStats();
    }, [drawAll, refreshStats]);

    useAnimationFrame(step, running);

    const handleStep = () => { if (!running) step(); };
    const handleReset = () => { setRunning(false); rebuild(); };
    const handleDataset = (id: string) => { setDatasetId(id); };

    const dataset = RNN_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls running={running} onToggle={() => setRunning(r => !r)} onStep={handleStep} onReset={handleReset} />
                <Select
                    label="Reviews"
                    value={datasetId}
                    options={RNN_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider label="Learning rate" value={rateSlider} display={learningRate.toFixed(2)} min={2} max={40} onChange={setRateSlider} />
                <Slider label="Speed" value={stepsPerFrame} display={`${stepsPerFrame} epochs / frame`} min={1} max={10} onChange={setStepsPerFrame} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>The scatter is the learned word vectors. Watch positive and negative words drift apart.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={embedCanvasRef} className={styles.boundary} />
                    <div className={styles.legend}>
                        <span style={{ color: '#fb923c' }}>● positive</span>
                        <span style={{ color: '#38bdf8' }}>● negative</span>
                        <span style={{ color: '#94a3b8' }}>● filler</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="Loss" value={metrics.loss.toFixed(3)} />
                        <Metric label="Accuracy" value={`${(metrics.acc * 100).toFixed(0)}%`} />
                    </MetricsRow>

                    {gradOk && <Badge>✓ Gradients verified</Badge>}

                    <Card title="Reading reviews" subtitle="prediction per sample">
                        <ul className={styles.sampleList}>
                            {samples.map((s, i) => (
                                <li key={i} className={styles.sample}>
                                    <span className={styles.sampleText}>"{s.words}"</span>
                                    <span style={{ color: s.correct ? '#34d399' : '#f87171' }}>
                                        {dataset?.classNames[s.predicted]}
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </Card>

                    <Card title="Loss" subtitle="cross-entropy per epoch">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
