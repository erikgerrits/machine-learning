import { useCallback, useEffect, useRef, useState } from 'react';
import { MultiArmedBandit } from 'machine-learning';
import type { BanditStrategy } from 'machine-learning';
import { BANDIT_MENUS, bestRate, type BanditMenu } from '../ml/banditDatasets';
import { mulberry32 } from '../ml/rng';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawBandit } from '../viz/bandit';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, NumberField, RunControls, Select, Slider } from './controls/Controls';
import styles from './BanditPlayground.module.css';

const STEPS_PER_FRAME = 4; // a few interactions per frame so a run converges in a watchable span

export function BanditPlayground() {
    const [menuId, setMenuId] = useState(BANDIT_MENUS[0].id);
    const [strategy, setStrategy] = useState<BanditStrategy>('epsilon-greedy');
    const [epsilonSlider, setEpsilonSlider] = useState(10); // epsilon = slider / 100
    const [confSlider, setConfSlider] = useState(14);        // confidence = slider / 10
    const [seed, setSeed] = useState(0);
    const [reveal, setReveal] = useState(true);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ days: 0, sales: 0, regret: 0, featuring: '—' });

    const epsilon = epsilonSlider / 100;
    const confidence = confSlider / 10;

    const banditRef = useRef<MultiArmedBandit | null>(null);
    const envRef = useRef<() => number>(() => 0);
    const menuRef = useRef<BanditMenu>(BANDIT_MENUS[0]);
    const selectedRef = useRef(-1);
    const salesRef = useRef(0);
    const optimalRef = useRef(0);
    const frameRef = useRef(0);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const uppers = useCallback((estimates: number[], counts: number[], steps: number): number[] => {
        if (strategy !== 'ucb') return estimates.slice(); // ε-greedy has no optimism band
        const logStep = Math.log(Math.max(1, steps));
        return estimates.map((v, i) =>
            counts[i] === 0 ? 1 : Math.min(1, v + confidence * Math.sqrt(logStep / counts[i])),
        );
    }, [strategy, confidence]);

    const draw = useCallback(() => {
        const bandit = banditRef.current;
        const canvas = canvasRef.current;
        if (!bandit || !canvas) return;
        const estimates = bandit.getValues();
        const counts = bandit.getCounts();
        drawBandit(canvas, {
            names: menuRef.current.arms.map(a => a.name),
            rates: menuRef.current.arms.map(a => a.rate),
            estimates,
            uppers: uppers(estimates, counts, bandit.getTotalSteps()),
            counts,
            selected: selectedRef.current,
            revealRates: reveal,
        });
    }, [uppers, reveal]);

    const rebuild = useCallback(() => {
        const menu = BANDIT_MENUS.find(m => m.id === menuId) ?? BANDIT_MENUS[0];
        const bandit = new MultiArmedBandit()
            .setNumberOfArms(menu.arms.length)
            .setStrategy(strategy)
            .setEpsilon(epsilon)
            .setConfidence(confidence)
            .setSeed(seed);

        banditRef.current = bandit;
        menuRef.current = menu;
        envRef.current = mulberry32(seed + 101); // reward stream, independent of the bandit's own RNG
        selectedRef.current = -1;
        salesRef.current = 0;
        optimalRef.current = 0;
        frameRef.current = 0;

        setMetrics({ days: 0, sales: 0, regret: 0, featuring: '—' });
        setRunning(false);
        draw();
    }, [menuId, strategy, epsilon, confidence, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    // Redraw once the canvas actually has a size (its CSS aspect-ratio settles a frame after mount) and
    // on any later resize — otherwise the first paint can land while the canvas is still zero-sized.
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || typeof ResizeObserver === 'undefined') return;
        const observer = new ResizeObserver(() => draw());
        observer.observe(canvas);
        return () => observer.disconnect();
    }, [draw]);

    const step = useCallback(() => {
        const bandit = banditRef.current;
        const menu = menuRef.current;
        if (!bandit) return;

        const best = bestRate(menu);
        for (let s = 0; s < STEPS_PER_FRAME; s++) {
            const arm = bandit.selectArm();              // choose what to feature
            const sold = envRef.current() < menu.arms[arm].rate ? 1 : 0; // ask the world
            bandit.update(arm, sold);                    // learn from the outcome
            selectedRef.current = arm;
            salesRef.current += sold;
            optimalRef.current += best;                  // what a perfect chooser would have earned
        }

        draw();
        frameRef.current += 1;
        if (frameRef.current % 2 === 0) {
            const days = bandit.getTotalSteps();
            setMetrics({
                days,
                sales: salesRef.current,
                regret: Math.round(optimalRef.current - salesRef.current),
                featuring: menu.arms[selectedRef.current]?.name ?? '—',
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

    const menu = BANDIT_MENUS.find(m => m.id === menuId);

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
                    label="Menu"
                    value={menuId}
                    options={BANDIT_MENUS.map(m => ({ value: m.id, label: m.label }))}
                    onChange={setMenuId}
                />
                {menu && <Hint>{menu.blurb}</Hint>}
                <Select
                    label="Strategy"
                    value={strategy}
                    options={[
                        { value: 'epsilon-greedy', label: 'ε-greedy' },
                        { value: 'ucb', label: 'UCB' },
                    ]}
                    onChange={v => setStrategy(v as BanditStrategy)}
                />
                {strategy === 'epsilon-greedy' ? (
                    <Slider label="Explore rate ε" value={epsilonSlider} display={epsilon.toFixed(2)} min={0} max={50} onChange={setEpsilonSlider} />
                ) : (
                    <Slider label="Confidence c" value={confSlider} display={confidence.toFixed(1)} min={0} max={30} onChange={setConfSlider} />
                )}
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Checkbox label="Show true rates" checked={reveal} onChange={setReveal} />
                <Hint>Hit Train and watch the bars climb toward the white ticks — the bandit&apos;s guesses closing in on each special&apos;s real sell-rate.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.chartWrap}>
                    <canvas ref={canvasRef} className={styles.chart} />
                    <div className={styles.legend}>
                        <span>▮ estimate</span>
                        <span style={{ opacity: 0.7 }}>▯ exploration headroom</span>
                        <span>│ true rate</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Days" value={String(metrics.days)} />
                        <Metric label="Sales" value={String(metrics.sales)} />
                        <Metric label="Regret" value={String(metrics.regret)} />
                    </MetricsRow>

                    <Card title="Featuring today" subtitle="the latest pick">
                        <p className={styles.note}>
                            <strong>{metrics.featuring}</strong> — the special the bandit chose to put on the
                            board this turn (ringed in gold). <em>Regret</em> counts the sales it gave up by
                            not always knowing the best choice; a good strategy keeps it growing slowly.
                        </p>
                    </Card>

                    <Card title="Explore vs. exploit" subtitle="the whole game">
                        <p className={styles.note}>
                            With <strong>ε-greedy</strong>, ε is the slice of days it gambles on a random
                            special instead of its favourite — turn it to 0 and it can lock onto a wrong early
                            guess forever. <strong>UCB</strong> instead chases the faint headroom on each bar:
                            big for rarely-tried specials, shrinking as it learns. On <em>Photo finish</em>,
                            watch which one separates the near-tied leaders faster.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
