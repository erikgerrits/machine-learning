import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DeepQNetwork } from 'machine-learning';
import { NAV_SCENARIOS, makeNavEnv, type NavEnv } from '../ml/navDatasets';
import { mulberry32 } from '../ml/rng';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawDeepRl } from '../viz/deepRl';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './DeepQNetworkPlayground.module.css';

const STEPS_PER_FRAME = 12;
const MAX_EPISODE_STEPS = 60;
const HEAT_RES = 26;   // value-heatmap resolution
const ARROW_RES = 9;   // policy-arrow resolution

// Pre-tuned to learn reliably in a couple of minutes in the browser.
const DEFAULT_LR = 20;   // /100
const DEFAULT_GAMMA = 90; // /100
const DEFAULT_EPS = 30;   // /100

export function DeepQNetworkPlayground() {
    const [scenarioId, setScenarioId] = useState(NAV_SCENARIOS[0].id);
    const [lrSlider, setLrSlider] = useState(DEFAULT_LR);
    const [gammaSlider, setGammaSlider] = useState(DEFAULT_GAMMA);
    const [epsilonSlider, setEpsilonSlider] = useState(DEFAULT_EPS);
    const [seed, setSeed] = useState(1);
    const [showValues, setShowValues] = useState(true);
    const [showPolicy, setShowPolicy] = useState(true);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ episodes: 0, success: 0, lastSteps: 0 });

    const lr = lrSlider / 100;
    const gamma = gammaSlider / 100;
    const epsilon = epsilonSlider / 100;

    // Fixed sample points for the heatmap and arrow grids (arena coords, y increasing upward).
    const heatPoints = useMemo(() => {
        const pts: number[][] = [];
        for (let r = 0; r < HEAT_RES; r++) for (let c = 0; c < HEAT_RES; c++) pts.push([(c + 0.5) / HEAT_RES, (r + 0.5) / HEAT_RES]);
        return pts;
    }, []);
    const arrowPoints = useMemo(() => {
        const pts: number[][] = [];
        for (let r = 0; r < ARROW_RES; r++) for (let c = 0; c < ARROW_RES; c++) pts.push([(c + 0.5) / ARROW_RES, (r + 0.5) / ARROW_RES]);
        return pts;
    }, []);

    const dqnRef = useRef<DeepQNetwork | null>(null);
    const envRef = useRef<NavEnv | null>(null);
    const stateRef = useRef<number[]>([0, 0]);
    const trailRef = useRef<number[][]>([]);
    const epStepsRef = useRef(0);
    const episodesRef = useRef(0);
    const recentRef = useRef<number[]>([]);
    const envRandomRef = useRef<() => number>(() => 0);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const dqn = dqnRef.current;
        const env = envRef.current;
        const canvas = canvasRef.current;
        if (!dqn || !env || !canvas) return;
        const heatValues = dqn.getQValuesBatch(heatPoints).map(q => Math.max(...q));
        const arrowActions = dqn.getQValuesBatch(arrowPoints).map(q => q.indexOf(Math.max(...q)));
        drawDeepRl(canvas, {
            goal: env.goal,
            radius: env.radius,
            heatRes: HEAT_RES,
            heatValues,
            arrowRes: ARROW_RES,
            arrowActions,
            agent: stateRef.current,
            trail: trailRef.current,
            showValues,
            showPolicy,
        });
    }, [heatPoints, arrowPoints, showValues, showPolicy]);

    const rebuild = useCallback(() => {
        const scenario = NAV_SCENARIOS.find(s => s.id === scenarioId) ?? NAV_SCENARIOS[0];
        const env = makeNavEnv(scenario);
        const dqn = new DeepQNetwork()
            .setStateSize(2).setNumberOfActions(4).setHiddenSizes([24, 24])
            .setLearningRate(lr).setDiscountFactor(gamma).setEpsilon(epsilon).setSeed(seed);

        dqnRef.current = dqn;
        envRef.current = env;
        envRandomRef.current = mulberry32(seed + 101);
        stateRef.current = env.randomStart(envRandomRef.current);
        trailRef.current = [stateRef.current];
        epStepsRef.current = 0;
        episodesRef.current = 0;
        recentRef.current = [];

        setMetrics({ episodes: 0, success: 0, lastSteps: 0 });
        setRunning(false);
        draw();
    }, [scenarioId, lr, gamma, epsilon, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const dqn = dqnRef.current;
        const env = envRef.current;
        if (!dqn || !env) return;

        for (let i = 0; i < STEPS_PER_FRAME; i++) {
            const state = stateRef.current;
            const action = dqn.selectAction(state);
            const { nextState, reward, done } = env.step(state, action);
            dqn.observe(state, action, reward, nextState, done);
            epStepsRef.current += 1;

            const trail = trailRef.current;
            trail.push(nextState);
            if (trail.length > 40) trail.shift();

            if (done || epStepsRef.current >= MAX_EPISODE_STEPS) {
                recentRef.current.push(done ? 1 : 0); // only the goal is terminal → done means success
                if (recentRef.current.length > 50) recentRef.current.shift();
                episodesRef.current += 1;
                setMetrics({
                    episodes: episodesRef.current,
                    success: Math.round((recentRef.current.reduce((a, b) => a + b, 0) / recentRef.current.length) * 100),
                    lastSteps: epStepsRef.current,
                });
                stateRef.current = env.randomStart(envRandomRef.current);
                trailRef.current = [stateRef.current];
                epStepsRef.current = 0;
            } else {
                stateRef.current = nextState;
            }
        }

        draw();
    }, [draw]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const scenario = NAV_SCENARIOS.find(s => s.id === scenarioId);

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
                    label="Table"
                    value={scenarioId}
                    options={NAV_SCENARIOS.map(s => ({ value: s.id, label: s.label }))}
                    onChange={setScenarioId}
                />
                {scenario && <Hint>{scenario.blurb}</Hint>}
                <Slider label="Learning rate α" value={lrSlider} display={lr.toFixed(2)} min={5} max={40} onChange={setLrSlider} />
                <Slider label="Discount γ" value={gammaSlider} display={gamma.toFixed(2)} min={50} max={98} onChange={setGammaSlider} />
                <Slider label="Explore rate ε" value={epsilonSlider} display={epsilon.toFixed(2)} min={0} max={60} onChange={setEpsilonSlider} />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Checkbox label="Value heatmap" checked={showValues} onChange={setShowValues} />
                <Checkbox label="Policy arrows" checked={showPolicy} onChange={setShowPolicy} />
                <Hint>Reinforcement learning is slow — give it a minute or two. The heatmap sharpens and the arrows lock onto the table as the network learns.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.gridWrap}>
                    <canvas ref={canvasRef} className={styles.grid} />
                    <div className={styles.legend}>
                        <span><i style={{ background: '#34d399' }} /> table</span>
                        <span><i style={{ background: '#facc15', borderRadius: '50%' }} /> runner</span>
                        <span><i style={{ background: 'linear-gradient(90deg,#1e2959,#fbbf24)' }} /> low→high value</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Episodes" value={String(metrics.episodes)} />
                        <Metric label="Success" value={`${metrics.success}%`} />
                        <Metric label="Last steps" value={String(metrics.lastSteps)} />
                    </MetricsRow>

                    <Card title="A function, not a table" subtitle="why this scales">
                        <p className={styles.note}>
                            Chapter 26 stored one value per cell. Here the floor is continuous — infinitely many
                            states — so a neural network <strong>predicts</strong> V(s) instead. The smooth heatmap
                            is that network evaluated everywhere at once, including spots the runner never stepped:
                            it <em>generalises</em>. That&apos;s what lets RL scale past anything you could tabulate.
                        </p>
                    </Card>

                    <Card title="Replay & a target net" subtitle="keeping it stable">
                        <p className={styles.note}>
                            Bootstrapping a network off its own shifting predictions is unstable, so DQN adds two
                            fixes: an <strong>experience-replay</strong> buffer it learns from in random minibatches
                            (breaking the correlation between consecutive steps), and a periodically-frozen
                            <strong> target network</strong> to supply a steady learning target. Both are running
                            under the hood here.
                        </p>
                    </Card>

                    <Card title="Be patient" subtitle="and watch it form">
                        <p className={styles.note}>
                            Early on the surface is flat and noisy and the runner wanders. As reward from the table
                            propagates back through the network, a clean gradient appears and the arrows snap into a
                            field that points home from everywhere. Nudge <strong>γ</strong> down to watch the far
                            floor go short-sighted, or drop <strong>ε</strong> once it knows the way.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
