import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { Recommender, Matrix } from 'machine-learning';
import { RECOMMENDER_DATASETS } from '../ml/recommenderDatasets';
import { ratingColor, ratingTextColor } from '../viz/recommender';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './RecommenderPlayground.module.css';

const STEPS_PER_FRAME = 4;

export function RecommenderPlayground() {
    const [datasetId, setDatasetId] = useState(RECOMMENDER_DATASETS[0].id);
    const [factors, setFactors] = useState(2);
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [selectedUser, setSelectedUser] = useState(0);
    const [predicted, setPredicted] = useState<number[][]>([]);
    const [metrics, setMetrics] = useState({ epoch: 0, rmse: 0 });

    const dataset = RECOMMENDER_DATASETS.find(d => d.id === datasetId) ?? RECOMMENDER_DATASETS[0];

    const modelRef = useRef<Recommender | null>(null);
    const ratingsRef = useRef<number[][]>([]);
    const matrixRef = useRef<Matrix | null>(null);
    const epochRef = useRef(0);
    const frameRef = useRef(0);

    const observedRmse = useCallback((prediction: number[][]) => {
        const ratings = ratingsRef.current;
        let sum = 0;
        let count = 0;
        for (let u = 0; u < ratings.length; u++) {
            for (let i = 0; i < ratings[u].length; i++) {
                if (ratings[u][i] !== 0) {
                    const error = prediction[u][i] - ratings[u][i];
                    sum += error * error;
                    count++;
                }
            }
        }
        return count > 0 ? Math.sqrt(sum / count) : 0;
    }, []);

    const rebuild = useCallback(() => {
        const active = RECOMMENDER_DATASETS.find(d => d.id === datasetId) ?? RECOMMENDER_DATASETS[0];
        const { inputs } = active.generate(seed);
        const matrix = new Matrix(inputs);

        const model = new Recommender()
            .setNumberOfFactors(factors)
            .setLearningRate(0.02)
            .setRegularization(0.05)
            .setSeed(seed)
            .setNumberOfEpochs(0);
        model.train(matrix); // initialise factors without stepping yet
        model.setNumberOfEpochs(STEPS_PER_FRAME);

        modelRef.current = model;
        ratingsRef.current = inputs;
        matrixRef.current = matrix;
        epochRef.current = 0;
        frameRef.current = 0;
        setRunning(false);

        const prediction = model.predict().toArray();
        setPredicted(prediction);
        setMetrics({ epoch: 0, rmse: observedRmse(prediction) });
    }, [datasetId, factors, seed, observedRmse]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const model = modelRef.current;
        const matrix = matrixRef.current;
        if (!model || !matrix) return;

        model.train(matrix);
        epochRef.current += STEPS_PER_FRAME;

        const prediction = model.predict().toArray();
        setPredicted(prediction);

        frameRef.current += 1;
        if (frameRef.current % 2 === 0) {
            setMetrics({ epoch: epochRef.current, rmse: observedRmse(prediction) });
        }
    }, [observedRmse]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => rebuild();
    const handleDataset = (id: string) => {
        setSelectedUser(0);
        setDatasetId(id);
    };

    const ratings = ratingsRef.current;
    const recommendations = predicted.length > 0
        ? predicted[selectedUser]
            .map((score, item) => ({ item, score }))
            .filter(({ item }) => ratings[selectedUser]?.[item] === 0)
            .sort((a, b) => b.score - a.score)
            .slice(0, 3)
        : [];

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls
                    running={running}
                    onToggle={() => setRunning(r => !r)}
                    onStep={handleStep}
                    onReset={handleReset}
                />
                <Select
                    label="Ratings"
                    value={datasetId}
                    options={RECOMMENDER_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                <Hint>{dataset.blurb}</Hint>
                <Slider label="Taste factors (k)" value={factors} display={String(factors)} min={1} max={4} onChange={setFactors} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>Click a name to see that person's top picks. Bright = predicted high; ★ = top suggestion.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.gridWrap}>
                    <div className={styles.grid} style={{ gridTemplateColumns: `84px repeat(${dataset.items.length}, 1fr)` }}>
                        <div className={styles.corner} />
                        {dataset.items.map(item => (
                            <div key={item} className={styles.itemHead}>{item}</div>
                        ))}

                        {dataset.users.map((user, u) => (
                            <Fragment key={user}>
                                <button
                                    className={`${styles.userHead} ${u === selectedUser ? styles.userActive : ''}`}
                                    onClick={() => setSelectedUser(u)}
                                >
                                    {user}
                                </button>
                                {dataset.items.map((_, i) => {
                                    const observed = ratings[u]?.[i] !== 0 && ratings[u]?.[i] !== undefined;
                                    const value = observed ? ratings[u][i] : (predicted[u]?.[i] ?? 0);
                                    const isTopPick = u === selectedUser && recommendations[0]?.item === i;
                                    return (
                                        <div
                                            key={i}
                                            className={`${styles.cell} ${observed ? styles.observed : styles.hidden} ${isTopPick ? styles.topPick : ''}`}
                                            style={{ background: ratingColor(value), color: ratingTextColor(value) }}
                                            title={observed ? `rated ${value}` : `predicted ${value.toFixed(1)}`}
                                        >
                                            {observed ? value : (isTopPick ? '★' : '')}
                                        </div>
                                    );
                                })}
                            </Fragment>
                        ))}
                    </div>
                    <div className={styles.legend}>
                        <span>bordered = rated</span>
                        <span>plain = predicted</span>
                        <span style={{ color: '#fbbf24' }}>★ top pick</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epoch" value={String(metrics.epoch)} />
                        <Metric label="RMSE" value={metrics.rmse.toFixed(2)} />
                    </MetricsRow>

                    <Card title={`Top picks · ${dataset.users[selectedUser]}`} subtitle="best untried items">
                        {recommendations.length === 0 ? (
                            <p className={styles.note}>This regular has tried everything on the menu.</p>
                        ) : (
                            <ul className={styles.pickList}>
                                {recommendations.map(({ item, score }) => (
                                    <li key={item} className={styles.pick}>
                                        <span>{dataset.items[item]}</span>
                                        <span className={styles.pickScore}>{score.toFixed(1)}</span>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </Card>

                    <Card title="Latent taste factors" subtitle="learned, not given">
                        <p className={styles.note}>
                            Press <strong>Train</strong> and watch the blanks fill in. Each person and each
                            item gets a short vector of hidden factors; a prediction is how well a taste
                            lines up with an item. With more <strong>factors</strong> the model captures
                            finer taste — until, with too few ratings, it just memorises noise.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
