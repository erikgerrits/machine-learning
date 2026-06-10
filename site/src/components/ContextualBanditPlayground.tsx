import { useCallback, useEffect, useRef, useState } from 'react';
import { ContextualBandit } from 'machine-learning';
import type { ContextualStrategy } from 'machine-learning';
import { CONTEXTUAL_SCENARIOS, contextOf, bestRateForType, type ContextualScenario } from '../ml/contextualBanditDatasets';
import { mulberry32 } from '../ml/rng';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawContextualBandit } from '../viz/contextualBandit';
import { CLUSTER_HEX } from '../viz/clusters';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './ContextualBanditPlayground.module.css';

const STEPS_PER_FRAME = 4;

export function ContextualBanditPlayground() {
    const [scenarioId, setScenarioId] = useState(CONTEXTUAL_SCENARIOS[0].id);
    const [strategy, setStrategy] = useState<ContextualStrategy>('linucb');
    const [alphaSlider, setAlphaSlider] = useState(10); // alpha = slider / 10
    const [epsilonSlider, setEpsilonSlider] = useState(10); // epsilon = slider / 100
    const [seed, setSeed] = useState(0);
    const [reveal, setReveal] = useState(true);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ days: 0, sales: 0, regret: 0, serving: '—' });

    const alpha = alphaSlider / 10;
    const epsilon = epsilonSlider / 100;

    const banditRef = useRef<ContextualBandit | null>(null);
    const envRef = useRef<() => number>(() => 0);
    const scenarioRef = useRef<ContextualScenario>(CONTEXTUAL_SCENARIOS[0]);
    const servingRef = useRef(-1);
    const salesRef = useRef(0);
    const optimalRef = useRef(0);
    const frameRef = useRef(0);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const bandit = banditRef.current;
        const canvas = canvasRef.current;
        const scenario = scenarioRef.current;
        if (!bandit || !canvas) return;

        const predicted = scenario.types.map((_, t) => {
            const ctx = contextOf(scenario, t);
            return scenario.arms.map((_, a) => bandit.predict(a, ctx));
        });
        const rates = scenario.types.map((_, t) => scenario.arms.map((_, a) => scenario.rates[a][t]));

        drawContextualBandit(canvas, {
            arms: scenario.arms,
            types: scenario.types,
            predicted,
            rates,
            serving: servingRef.current,
            revealRates: reveal,
        });
    }, [reveal]);

    const rebuild = useCallback(() => {
        const scenario = CONTEXTUAL_SCENARIOS.find(s => s.id === scenarioId) ?? CONTEXTUAL_SCENARIOS[0];
        const bandit = new ContextualBandit()
            .setNumberOfArms(scenario.arms.length)
            .setContextDimensions(scenario.types.length)
            .setStrategy(strategy)
            .setAlpha(alpha)
            .setEpsilon(epsilon)
            .setSeed(seed);

        banditRef.current = bandit;
        scenarioRef.current = scenario;
        envRef.current = mulberry32(seed + 101);
        servingRef.current = -1;
        salesRef.current = 0;
        optimalRef.current = 0;
        frameRef.current = 0;

        setMetrics({ days: 0, sales: 0, regret: 0, serving: '—' });
        setRunning(false);
        draw();
    }, [scenarioId, strategy, alpha, epsilon, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const bandit = banditRef.current;
        const scenario = scenarioRef.current;
        if (!bandit) return;

        for (let s = 0; s < STEPS_PER_FRAME; s++) {
            const type = Math.floor(envRef.current() * scenario.types.length); // who walks in
            const context = contextOf(scenario, type);
            const arm = bandit.selectArm(context);                              // pick an offer for them
            const sold = envRef.current() < scenario.rates[arm][type] ? 1 : 0;  // did they take it?
            bandit.update(arm, context, sold);                                  // learn from the outcome
            servingRef.current = type;
            salesRef.current += sold;
            optimalRef.current += bestRateForType(scenario, type);              // a perfect chooser's take
        }

        draw();
        frameRef.current += 1;
        if (frameRef.current % 2 === 0) {
            setMetrics({
                days: bandit.getTotalSteps(),
                sales: salesRef.current,
                regret: Math.round(optimalRef.current - salesRef.current),
                serving: scenario.types[servingRef.current] ?? '—',
            });
        }
    }, [draw]);

    useAnimationFrame(step, running);

    const handleStep = () => {
        if (!running) step();
    };
    const handleReset = () => {
        setRunning(false);
        rebuild();
    };

    const scenario = CONTEXTUAL_SCENARIOS.find(s => s.id === scenarioId);

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
                    label="Scenario"
                    value={scenarioId}
                    options={CONTEXTUAL_SCENARIOS.map(s => ({ value: s.id, label: s.label }))}
                    onChange={setScenarioId}
                />
                {scenario && <Hint>{scenario.blurb}</Hint>}
                <Select
                    label="Strategy"
                    value={strategy}
                    options={[
                        { value: 'linucb', label: 'LinUCB' },
                        { value: 'epsilon-greedy', label: 'ε-greedy' },
                    ]}
                    onChange={v => setStrategy(v as ContextualStrategy)}
                />
                {strategy === 'linucb' ? (
                    <Slider label="Optimism α" value={alphaSlider} display={alpha.toFixed(1)} min={0} max={30} onChange={setAlphaSlider} />
                ) : (
                    <Slider label="Explore rate ε" value={epsilonSlider} display={epsilon.toFixed(2)} min={0} max={50} onChange={setEpsilonSlider} />
                )}
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Checkbox label="Show true rates" checked={reveal} onChange={setReveal} />
                <Hint>Each panel is one kind of customer. Hit Train and watch a different ★ rise in each — one policy, tailored per context.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.chartWrap}>
                    <canvas ref={canvasRef} className={styles.chart} />
                </div>

                <div className={styles.side}>
                    {scenario && (
                        <div className={styles.legend}>
                            {scenario.arms.map((arm, a) => (
                                <span key={arm}>
                                    <i style={{ background: CLUSTER_HEX[a % CLUSTER_HEX.length] }} />
                                    {arm}
                                </span>
                            ))}
                        </div>
                    )}

                    <MetricsRow>
                        <Metric label="Days" value={String(metrics.days)} />
                        <Metric label="Sales" value={String(metrics.sales)} />
                        <Metric label="Regret" value={String(metrics.regret)} />
                    </MetricsRow>

                    <Card title="Serving now" subtitle="the lit-up panel">
                        <p className={styles.note}>
                            <strong>{metrics.serving}</strong> — the customer at the counter this turn. The
                            bandit reads the context, picks that panel&apos;s ★ offer, and folds the result
                            into <em>that offer&apos;s</em> model. <em>Regret</em> is the sales lost versus a
                            chooser who already knew every customer&apos;s favourite.
                        </p>
                    </Card>

                    <Card title="One model per offer" subtitle="LinUCB">
                        <p className={styles.note}>
                            Each offer learns a little linear model: how much each customer feature lifts its
                            sell-rate. <strong>LinUCB</strong> picks the offer with the best predicted rate
                            <em> plus</em> an optimism bonus where it has little data on that kind of customer —
                            <strong> α</strong> sets how boldly it probes. On <em>One size fits all</em> the ★
                            lands on the same offer in every panel: context turned out not to matter.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
