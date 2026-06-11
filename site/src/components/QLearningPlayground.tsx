import { useCallback, useEffect, useRef, useState } from 'react';
import { QLearning } from 'machine-learning';
import { GRIDWORLD_MAPS, makeGridworld, type Gridworld } from '../ml/gridworldDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawGridworld } from '../viz/gridworld';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './QLearningPlayground.module.css';

const STEPS_PER_FRAME = 6; // environment steps per animation frame — enough to learn in a few seconds
const MAX_EPISODE_STEPS = 80;

export function QLearningPlayground() {
    const [mapId, setMapId] = useState(GRIDWORLD_MAPS[0].id);
    const [alphaSlider, setAlphaSlider] = useState(40); // α = slider / 100
    const [gammaSlider, setGammaSlider] = useState(95); // γ = slider / 100
    const [epsilonSlider, setEpsilonSlider] = useState(20); // ε = slider / 100
    const [seed, setSeed] = useState(0);
    const [showPolicy, setShowPolicy] = useState(true);
    const [showValues, setShowValues] = useState(true);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ episodes: 0, lastReturn: 0, lastSteps: 0, success: 0 });

    const alpha = alphaSlider / 100;
    const gamma = gammaSlider / 100;
    const epsilon = epsilonSlider / 100;

    const agentRef = useRef<QLearning | null>(null);
    const envRef = useRef<Gridworld | null>(null);
    const stateRef = useRef(0);
    const epReturnRef = useRef(0);
    const epStepsRef = useRef(0);
    const episodesRef = useRef(0);
    const recentRef = useRef<number[]>([]); // 1 = reached goal, 0 = fell in / timed out
    const frameRef = useRef(0);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const agent = agentRef.current;
        const env = envRef.current;
        const canvas = canvasRef.current;
        if (!agent || !env || !canvas) return;
        const states = Array.from({ length: env.numberOfStates }, (_, s) => s);
        drawGridworld(canvas, {
            rows: env.rows,
            cols: env.cols,
            cellTypes: env.cellTypes,
            values: states.map(s => agent.getValue(s)),
            policy: agent.getPolicy(),
            agent: stateRef.current,
            showPolicy,
            showValues,
        });
    }, [showPolicy, showValues]);

    const rebuild = useCallback(() => {
        const map = GRIDWORLD_MAPS.find(m => m.id === mapId) ?? GRIDWORLD_MAPS[0];
        const env = makeGridworld(map);
        const agent = new QLearning()
            .setNumberOfStates(env.numberOfStates)
            .setNumberOfActions(env.numberOfActions)
            .setLearningRate(alpha)
            .setDiscountFactor(gamma)
            .setEpsilon(epsilon)
            .setSeed(seed);

        agentRef.current = agent;
        envRef.current = env;
        stateRef.current = env.start;
        epReturnRef.current = 0;
        epStepsRef.current = 0;
        episodesRef.current = 0;
        recentRef.current = [];
        frameRef.current = 0;

        setMetrics({ episodes: 0, lastReturn: 0, lastSteps: 0, success: 0 });
        setRunning(false);
        draw();
    }, [mapId, alpha, gamma, epsilon, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const agent = agentRef.current;
        const env = envRef.current;
        if (!agent || !env) return;

        for (let i = 0; i < STEPS_PER_FRAME; i++) {
            const state = stateRef.current;
            const action = agent.selectAction(state);
            const { nextState, reward, done } = env.step(state, action);
            agent.update(state, action, reward, nextState, done);
            epReturnRef.current += reward;
            epStepsRef.current += 1;

            if (done || epStepsRef.current >= MAX_EPISODE_STEPS) {
                const reachedGoal = env.cellTypes[nextState] === 'goal' ? 1 : 0;
                recentRef.current.push(reachedGoal);
                if (recentRef.current.length > 50) recentRef.current.shift();
                episodesRef.current += 1;
                setMetrics({
                    episodes: episodesRef.current,
                    lastReturn: Math.round(epReturnRef.current * 100) / 100,
                    lastSteps: epStepsRef.current,
                    success: Math.round((recentRef.current.reduce((a, b) => a + b, 0) / recentRef.current.length) * 100),
                });
                stateRef.current = env.start;
                epReturnRef.current = 0;
                epStepsRef.current = 0;
            } else {
                stateRef.current = nextState;
            }
        }

        draw();
        frameRef.current += 1;
    }, [draw]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const map = GRIDWORLD_MAPS.find(m => m.id === mapId);

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
                    label="Floor plan"
                    value={mapId}
                    options={GRIDWORLD_MAPS.map(m => ({ value: m.id, label: m.label }))}
                    onChange={setMapId}
                />
                {map && <Hint>{map.blurb}</Hint>}
                <Slider label="Learning rate α" value={alphaSlider} display={alpha.toFixed(2)} min={5} max={100} onChange={setAlphaSlider} />
                <Slider label="Discount γ" value={gammaSlider} display={gamma.toFixed(2)} min={50} max={99} onChange={setGammaSlider} />
                <Slider label="Explore rate ε" value={epsilonSlider} display={epsilon.toFixed(2)} min={0} max={60} onChange={setEpsilonSlider} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Checkbox label="Value heatmap" checked={showValues} onChange={setShowValues} />
                <Checkbox label="Policy arrows" checked={showPolicy} onChange={setShowPolicy} />
                <Hint>Hit Train and watch value spread out from the table (★) while the arrows lock into a route. The yellow dot is the runner, still exploring.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.gridWrap}>
                    <canvas ref={canvasRef} className={styles.grid} />
                    <div className={styles.legend}>
                        <span><i style={{ background: '#38bdf8' }} /> kitchen</span>
                        <span><i style={{ background: '#34d399' }} /> table</span>
                        <span><i style={{ background: '#f43f5e' }} /> spill</span>
                        <span><i style={{ background: '#facc15', borderRadius: '50%' }} /> runner</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Episodes" value={String(metrics.episodes)} />
                        <Metric label="Success" value={`${metrics.success}%`} />
                    </MetricsRow>
                    <MetricsRow>
                        <Metric label="Last return" value={metrics.lastReturn.toFixed(2)} />
                        <Metric label="Last steps" value={String(metrics.lastSteps)} />
                    </MetricsRow>

                    <Card title="Value, flowing backward" subtitle="the heatmap">
                        <p className={styles.note}>
                            Each cell&apos;s colour is its value <strong>V(s) = maxₐ Q(s, a)</strong> — the best
                            long-run reward reachable from there. Only the table pays off, so early on the floor
                            is dark; with each visit the goal&apos;s value seeps one cell further out, until a
                            warm gradient points the whole floor home.
                        </p>
                    </Card>

                    <Card title="The policy" subtitle="the arrows">
                        <p className={styles.note}>
                            Each arrow is the greedy action — the runner&apos;s learned move in that cell. Watch
                            them swing into a coherent route as the values settle. On <em>Cliff walk</em>, the
                            arrows deliberately steer up and over the spills; on <em>The back room</em>, the far
                            room stays dark and arrow-less until value finally reaches it through the doorway.
                        </p>
                    </Card>

                    <Card title="Explore vs. exploit" subtitle="ε and γ">
                        <p className={styles.note}>
                            <strong>ε</strong> is how often the runner wanders off-policy to discover new routes —
                            needed early, a nuisance once it knows the way. <strong>γ</strong> sets how far ahead
                            it plans: near 1 it values the distant table almost as much as a step saved now; turn
                            it down and far cells fade, the runner growing short-sighted.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
