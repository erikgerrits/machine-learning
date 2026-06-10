import { useCallback, useEffect, useRef, useState } from 'react';
import { Transformer, Matrix } from 'machine-learning';
import { TRANSFORMER_DATASETS } from '../ml/transformerDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawAttention } from '../viz/transformer';
import { drawLossCurve } from '../viz/lossCurve';
import { Badge, Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './TransformerPlayground.module.css';

const POINTS = 60;
const MODEL_DIM = 16;

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

export function TransformerPlayground() {
    const [datasetId, setDatasetId] = useState(TRANSFORMER_DATASETS[0].id);
    const [rateSlider, setRateSlider] = useState(20); // lr = rateSlider / 100
    const [stepsPerFrame, setStepsPerFrame] = useState(8);
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

    const modelRef = useRef<Transformer | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const lossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const attnCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const polarityOf = (data: TrainingData, token: number): 'pos' | 'neg' | 'filler' =>
        data.positive.has(token) ? 'pos' : data.negative.has(token) ? 'neg' : 'filler';

    const drawAll = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        const canvas = attnCanvasRef.current;
        if (canvas) {
            const { ctx, width, height } = fitCanvas(canvas);
            ctx.fillStyle = '#0b1120';
            ctx.fillRect(0, 0, width, height);
            const idx = Math.min(exampleRef.current, data.inputs.length - 1);
            const seq = data.inputs[idx];
            const words = seq.map(t => data.vocab[Math.round(t)]);
            const polarity = seq.map(t => polarityOf(data, Math.round(t)));
            drawAttention(ctx, width, height, words, model.getAttention(seq)[0], polarity);
        }
        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawLossCurve(ctx, width, height, lossRef.current);
        }
    }, []);

    const accuracy = useCallback((model: Transformer, data: TrainingData) => {
        const preds = model.predict(data.inputMatrix).getMaximumRowIndeces().toArray().map(r => r[0]);
        const truth = data.targets.map(t => t.indexOf(1));
        return preds.filter((p, i) => p === truth[i]).length / preds.length;
    }, []);

    const rebuild = useCallback(() => {
        const dataset = TRANSFORMER_DATASETS.find(d => d.id === datasetId) ?? TRANSFORMER_DATASETS[0];
        const { inputs, targets } = dataset.generate(seed, POINTS);
        const inputMatrix = new Matrix(inputs);
        const targetMatrix = new Matrix(targets);

        const model = new Transformer()
            .setVocabSize(dataset.vocab.length)
            .setModelDim(MODEL_DIM)
            .setMaxLength(dataset.sequenceLength)
            .setLearningRate(lrRef.current)
            .setNumberOfEpochs(1)
            .setSeed(seed);
        model.train(inputMatrix, targetMatrix); // one pass to initialise

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
        setMetrics({ epoch: 1, loss: lossRef.current[0], acc: accuracy(model, dataRef.current) });
        drawAll();
    }, [datasetId, seed, drawAll, accuracy]);

    useEffect(() => { rebuild(); }, [rebuild]);
    useEffect(() => { if (modelRef.current) modelRef.current.setLearningRate(learningRate); }, [learningRate]);
    useEffect(() => { drawAll(); }, [example, drawAll]);

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
        if (frameRef.current % 3 === 0) {
            setMetrics({ epoch: epochRef.current, loss: lossRef.current[lossRef.current.length - 1], acc: accuracy(model, data) });
        }
    }, [drawAll, accuracy]);

    useAnimationFrame(step, running);

    const handleStep = () => { if (!running) step(); };
    const handleReset = () => { setRunning(false); rebuild(); };
    const handleDataset = (id: string) => { setDatasetId(id); setExample(0); };

    const dataset = TRANSFORMER_DATASETS.find(d => d.id === datasetId);

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls running={running} onToggle={() => setRunning(r => !r)} onStep={handleStep} onReset={handleReset} />
                <Select
                    label="Reviews"
                    value={datasetId}
                    options={TRANSFORMER_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider label="Learning rate" value={rateSlider} display={learningRate.toFixed(2)} min={5} max={40} onChange={setRateSlider} />
                <Slider label="Speed" value={stepsPerFrame} display={`${stepsPerFrame} epochs / frame`} min={1} max={16} onChange={setStepsPerFrame} />
                <Slider label="Show review" value={example} display={`#${example}`} min={0} max={POINTS - 1} onChange={setExample} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>One sentiment word hides among fillers. Watch its attention bar grow tallest.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.attnWrap}>
                    <canvas ref={attnCanvasRef} className={styles.attn} />
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

                    <Card title="What to watch" subtitle="attention finds the word">
                        <p className={styles.note}>
                            Every position can look at every other at once. Early on the attention is
                            spread evenly; as it learns, the [CLS] slot's bar over the one sentiment word
                            grows tallest — the model deciding *that's* the word that settles the verdict,
                            wherever it sits.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
