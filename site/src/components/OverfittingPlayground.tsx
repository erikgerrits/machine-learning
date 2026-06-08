import { useCallback, useEffect, useRef, useState } from 'react';
import { LinearRegression, Matrix } from 'machine-learning';
import { mse } from '../ml/metrics';
import { gaussian, mulberry32 } from '../ml/rng';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { fitCanvas } from '../viz/canvas';
import { drawTrainTestFit } from '../viz/trainTestFit';
import { drawDualLoss } from '../viz/dualLoss';
import {
    Card,
    ControlPanel,
    Hint,
    Metric,
    MetricsRow,
    NumberField,
    RunControls,
    Slider,
} from './controls/Controls';
import styles from './OverfittingPlayground.module.css';

const POOL = 150;
const TEST_DAYS = 60;
const REAL_FEATURES = 3;
const JUNK_MAX = 40;
const SALES_MIN = 20;
const SALES_MAX = 140;

// The hidden recipe behind sales. Only the first three features matter; everything else is noise.
const TRUE = { bias: 80, temp: 18, weekend: 22, footfall: 12 };

const sliderToLr = (slider: number) => Math.pow(10, -3 + 4 * (slider / 1000));
const lrToSlider = (lr: number) => Math.round(((Math.log10(lr) + 3) / 4) * 1000);
const sliderToLambda = (slider: number) => slider / 2; // 0 … 50

// A pool of days: 3 real features that drive sales, plus JUNK_MAX pure-noise "features" (the
// pigeons on the awning) that don't. The model is later given only the first few of the junk.
function generatePool(seed: number): { rows: number[][]; targets: number[][] } {
    const rand = mulberry32(seed);
    const rows: number[][] = [];
    const targets: number[][] = [];
    for (let i = 0; i < POOL; i++) {
        const temp = rand() * 2 - 1;
        const weekend = rand() < 2 / 7 ? 1 : 0;
        const footfall = rand() * 2 - 1;
        const junk = Array.from({ length: JUNK_MAX }, () => gaussian(rand));
        const sold = TRUE.bias + TRUE.temp * temp + TRUE.weekend * weekend + TRUE.footfall * footfall + gaussian(rand) * 5;
        rows.push([temp, weekend, footfall, ...junk]);
        targets.push([sold]);
    }
    return { rows, targets };
}

const predict = (model: LinearRegression, input: Matrix) =>
    model.predict(input).toArray().map(row => row[0]);

interface TrainingData {
    trainInput: Matrix;
    trainTarget: Matrix;
    trainTargets: number[][];
    testInput: Matrix;
    testTargets: number[][];
}

export function OverfittingPlayground() {
    const [junkCount, setJunkCount] = useState(30);
    const [trainDays, setTrainDays] = useState(20);
    const [lambdaSlider, setLambdaSlider] = useState(0);
    const [sliderLR, setSliderLR] = useState(lrToSlider(0.1));
    const [seed, setSeed] = useState(0);
    const [stepsPerFrame, setStepsPerFrame] = useState(6);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epoch: 0, trainLoss: 0, testLoss: 0 });

    const learningRate = sliderToLr(sliderLR);
    const lambda = sliderToLambda(lambdaSlider);
    const lrRef = useRef(learningRate);
    const lambdaRef = useRef(lambda);
    const stepsRef = useRef(stepsPerFrame);
    lrRef.current = learningRate;
    lambdaRef.current = lambda;
    stepsRef.current = stepsPerFrame;

    const modelRef = useRef<LinearRegression | null>(null);
    const dataRef = useRef<TrainingData | null>(null);
    const trainLossRef = useRef<number[]>([]);
    const testLossRef = useRef<number[]>([]);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const plotCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const drawAll = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        const plot = plotCanvasRef.current;
        if (plot) {
            const { ctx, width, height } = fitCanvas(plot);
            drawTrainTestFit(
                ctx,
                width,
                height,
                SALES_MIN,
                SALES_MAX,
                data.trainTargets.map(r => r[0]),
                predict(model, data.trainInput),
                data.testTargets.map(r => r[0]),
                predict(model, data.testInput),
            );
        }
        const lossCanvas = lossCanvasRef.current;
        if (lossCanvas) {
            const { ctx, width, height } = fitCanvas(lossCanvas);
            drawDualLoss(ctx, width, height, trainLossRef.current, testLossRef.current);
        }
    }, []);

    const rebuild = useCallback(() => {
        const { rows, targets } = generatePool(seed);
        const featureCount = REAL_FEATURES + junkCount;

        const trainRows = rows.slice(0, trainDays).map(r => r.slice(0, featureCount));
        const trainTargets = targets.slice(0, trainDays);
        const testRows = rows.slice(POOL - TEST_DAYS).map(r => r.slice(0, featureCount));
        const testTargets = targets.slice(POOL - TEST_DAYS);

        const trainInput = new Matrix(trainRows);
        const trainTarget = new Matrix(trainTargets);
        const testInput = new Matrix(testRows);

        const mean = trainTargets.reduce((sum, [v]) => sum + v, 0) / trainTargets.length;

        const model = new LinearRegression();
        model.setNumberOfEpochs(1); // one step per train() call → epochs driven by the loop
        model.setLearningRate(lrRef.current);
        model.setRegularizationFactor(lambdaRef.current);
        const hypothesis = Matrix.zeros(featureCount + 1, 1);
        hypothesis.setElement(0, 0, mean); // bias starts at the average; weights start at zero
        model.setHypothesis(hypothesis);

        modelRef.current = model;
        dataRef.current = { trainInput, trainTarget, trainTargets, testInput, testTargets };
        epochRef.current = 0;
        trainLossRef.current = [mse(predict(model, trainInput).map(p => [p]), trainTargets)];
        testLossRef.current = [mse(predict(model, testInput).map(p => [p]), testTargets)];

        setMetrics({ epoch: 0, trainLoss: trainLossRef.current[0], testLoss: testLossRef.current[0] });
        drawAll();
    }, [seed, junkCount, trainDays, drawAll]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    useEffect(() => {
        modelRef.current?.setLearningRate(learningRate);
    }, [learningRate]);

    useEffect(() => {
        modelRef.current?.setRegularizationFactor(lambda);
    }, [lambda]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const data = dataRef.current;
        if (!model || !data) return;

        const steps = stepsRef.current;
        for (let i = 0; i < steps; i++) model.train(data.trainInput, data.trainTarget);
        epochRef.current += steps;

        const trainLoss = mse(predict(model, data.trainInput).map(p => [p]), data.trainTargets);
        const testLoss = mse(predict(model, data.testInput).map(p => [p]), data.testTargets);
        trainLossRef.current.push(trainLoss);
        testLossRef.current.push(testLoss);
        if (trainLossRef.current.length > 1200) {
            trainLossRef.current.shift();
            testLossRef.current.shift();
        }

        drawAll();

        frameRef.current += 1;
        if (frameRef.current % 4 === 0) {
            setMetrics({ epoch: epochRef.current, trainLoss, testLoss });
        }
    }, [drawAll]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const gap = metrics.testLoss - metrics.trainLoss;

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls
                    running={running}
                    onToggle={() => setRunning(r => !r)}
                    onStep={handleStep}
                    onReset={handleReset}
                />
                <Hint>
                    Press Train. With lots of junk features and few days, the model memorises the
                    studied days (blue) and flubs the unseen ones (orange). Then raise the
                    regularization to pull them back together.
                </Hint>
                <Slider
                    label="Junk features"
                    value={junkCount}
                    display={`${junkCount} noise + ${REAL_FEATURES} real`}
                    min={0}
                    max={JUNK_MAX}
                    onChange={setJunkCount}
                />
                <Slider
                    label="Regularization (λ)"
                    value={lambdaSlider}
                    display={lambda.toFixed(1)}
                    min={0}
                    max={100}
                    onChange={setLambdaSlider}
                />
                <Slider
                    label="Training days"
                    value={trainDays}
                    display={`${trainDays} days`}
                    min={12}
                    max={POOL - TEST_DAYS}
                    onChange={setTrainDays}
                />
                <Slider
                    label="Learning rate"
                    value={sliderLR}
                    display={learningRate.toFixed(3)}
                    min={0}
                    max={1000}
                    onChange={setSliderLR}
                />
                <Slider
                    label="Speed"
                    value={stepsPerFrame}
                    display={`${stepsPerFrame} epochs / frame`}
                    min={1}
                    max={20}
                    onChange={setStepsPerFrame}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.plotWrap}>
                    <canvas ref={plotCanvasRef} className={styles.plot} />
                    <div className={styles.legend}>
                        <span style={{ color: 'var(--accent)' }}>● studied days</span>
                        <span style={{ color: 'var(--accent-2)' }}>● unseen days</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Train error" value={metrics.trainLoss.toFixed(0)} />
                        <Metric label="Test error" value={metrics.testLoss.toFixed(0)} />
                        <Metric label="The gap" value={gap.toFixed(0)} />
                    </MetricsRow>

                    <Card title="The gap is the lie" subtitle="train vs. test">
                        <p className={styles.note}>
                            <strong>Train error</strong> is how well the model memorised the past;
                            <strong> test error</strong> is what tomorrow will actually cost. A big
                            gap means it's fooling itself. Regularization trades a little train
                            error for a lot less test error — until too much λ makes it forget the
                            real recipe too.
                        </p>
                    </Card>

                    <Card title="Error over time" subtitle="train (blue) vs. test (orange)">
                        <canvas ref={lossCanvasRef} className={styles.lossCanvas} />
                    </Card>
                </div>
            </div>
        </div>
    );
}
