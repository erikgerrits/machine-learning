import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BayesianLinearRegression, Matrix } from 'machine-learning';
import type { BayesianBasis } from 'machine-learning';
import { BAYESIAN_SCENARIOS, type BayesianScenario } from '../ml/bayesianDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawBayesian } from '../viz/bayesian';
import { Card, Checkbox, ControlPanel, Hint, Metric, MetricsRow, RunControls, Select, Slider } from './controls/Controls';
import styles from './BayesianLinearRegressionPlayground.module.css';

const GRID = 120;
const SAMPLES = 8;
const REVEAL_EVERY = 5; // frames between revealing one more observation

export function BayesianLinearRegressionPlayground() {
    const [scenarioId, setScenarioId] = useState(BAYESIAN_SCENARIOS[0].id);
    const [basis, setBasis] = useState<BayesianBasis>('gaussian');
    const [complexity, setComplexity] = useState(6);  // gaussian bases, or polynomial degree
    const [alphaSlider, setAlphaSlider] = useState(10); // alpha = slider / 10
    const [betaSlider, setBetaSlider] = useState(40);   // beta = slider
    const [seed, setSeed] = useState(0);
    const [showSamples, setShowSamples] = useState(true);
    const [showTruth, setShowTruth] = useState(true);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ points: 0, avgStd: 0 });

    const alpha = alphaSlider / 10;
    const beta = betaSlider;

    const scenario = useMemo<BayesianScenario>(() => BAYESIAN_SCENARIOS.find(s => s.id === scenarioId) ?? BAYESIAN_SCENARIOS[0], [scenarioId]);
    const gridX = useMemo(() => Array.from({ length: GRID }, (_, i) => scenario.domain[0] + (scenario.domain[1] - scenario.domain[0]) * (i / (GRID - 1))), [scenario]);
    const trueY = useMemo(() => gridX.map(scenario.truth), [gridX, scenario]);

    const pointsRef = useRef<{ x: number; y: number }[]>([]);
    const revealedRef = useRef(1);
    const frameRef = useRef(0);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const fitAndDraw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const shown = pointsRef.current.slice(0, revealedRef.current);

        const model = new BayesianLinearRegression()
            .setBasis(basis)
            .setNumberOfBases(complexity).setDegree(complexity)
            .setBasisWidth((scenario.domain[1] - scenario.domain[0]) / complexity)
            .setAlpha(alpha).setBeta(beta).setSeed(seed + 1);
        model.train(new Matrix(shown.map(p => [p.x])), new Matrix(shown.map(p => [p.y])));

        const gridMatrix = new Matrix(gridX.map(x => [x]));
        const meanY = model.predict(gridMatrix).toArray().map(r => r[0]);
        const stdY = model.predictiveStandardDeviation(gridMatrix);
        const samples = showSamples ? model.sample(gridMatrix, SAMPLES, seed + 7) : [];

        drawBayesian(canvas, {
            domain: scenario.domain, yRange: scenario.yRange,
            gridX, meanY, stdY, samples, trueY, points: shown, showSamples, showTruth,
        });
        setMetrics({ points: shown.length, avgStd: Math.round((stdY.reduce((a, b) => a + b, 0) / stdY.length) * 1000) / 1000 });
    }, [basis, complexity, alpha, beta, seed, showSamples, showTruth, scenario, gridX, trueY]);

    const reset = useCallback(() => {
        pointsRef.current = scenario.generate(seed + 1);
        revealedRef.current = 1;
        frameRef.current = 0;
        setRunning(false);
        fitAndDraw();
    }, [scenario, seed, fitAndDraw]);

    useEffect(() => { reset(); }, [reset]);
    // Refit (without re-revealing) when only the model knobs change.
    useEffect(() => { fitAndDraw(); }, [fitAndDraw]);

    const tick = useCallback(() => {
        frameRef.current += 1;
        if (frameRef.current % REVEAL_EVERY !== 0) return;
        if (revealedRef.current >= pointsRef.current.length) { setRunning(false); return; }
        revealedRef.current += 1;
        fitAndDraw();
    }, [fitAndDraw]);

    useAnimationFrame(tick, running);

    const handleStep = () => {
        if (running) return;
        if (revealedRef.current < pointsRef.current.length) revealedRef.current += 1;
        fitAndDraw();
    };
    const handleReset = () => { setRunning(false); reset(); };

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <RunControls running={running} onToggle={() => setRunning(r => !r)} onStep={handleStep} onReset={handleReset} />
                <Select label="Data" value={scenarioId} options={BAYESIAN_SCENARIOS.map(s => ({ value: s.id, label: s.label }))} onChange={setScenarioId} />
                <Hint>{scenario.blurb}</Hint>
                <Select
                    label="Basis"
                    value={basis}
                    options={[{ value: 'gaussian', label: 'Gaussian bumps' }, { value: 'polynomial', label: 'Polynomial' }]}
                    onChange={v => setBasis(v as BayesianBasis)}
                />
                <Slider label={basis === 'gaussian' ? 'Bumps' : 'Degree'} value={complexity} display={String(complexity)} min={2} max={9} onChange={setComplexity} />
                <Slider label="Prior strength α" value={alphaSlider} display={alpha.toFixed(1)} min={1} max={60} onChange={setAlphaSlider} />
                <Slider label="Noise precision β" value={betaSlider} display={String(beta)} min={5} max={120} onChange={setBetaSlider} />
                <Slider label="Random seed" value={seed} display={String(seed)} min={0} max={9} onChange={setSeed} />
                <Checkbox label="Posterior samples" checked={showSamples} onChange={setShowSamples} />
                <Checkbox label="Show hidden truth" checked={showTruth} onChange={setShowTruth} />
                <Hint>Press Train to reveal the data point by point — watch the shaded band collapse onto the evidence and stay wide where there is none.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.plotWrap}>
                    <canvas ref={canvasRef} className={styles.plot} />
                    <div className={styles.legend}>
                        <span style={{ color: '#fb923c' }}>● data</span>
                        <span style={{ color: '#38bdf8' }}>— mean fit</span>
                        <span style={{ color: 'rgba(56,189,248,0.7)' }}>▭ ±2σ band</span>
                        <span style={{ color: '#94a3b8' }}>┄ truth</span>
                    </div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Observations" value={String(metrics.points)} />
                        <Metric label="Avg ±σ" value={metrics.avgStd.toFixed(3)} />
                    </MetricsRow>

                    <Card title="The band is the point" subtitle="honest doubt">
                        <p className={styles.note}>
                            The shaded region is the model&apos;s <strong>95% credible band</strong> — where it
                            believes the true curve lies. It hugs the data tightly and flares open wherever points
                            are missing. That admitted uncertainty, not just the blue mean line, is what tells you
                            when the model is guessing.
                        </p>
                    </Card>

                    <Card title="Prior vs. evidence" subtitle="α and β">
                        <p className={styles.note}>
                            Before any data the band is the <em>prior</em> — a vague flat guess. Each point is
                            <em> evidence</em> that sharpens it. <strong>α</strong> sets how strongly the prior pulls
                            toward a flat line (more α = steadier but stiffer); <strong>β</strong> is how much the
                            model trusts each point (more β = it hugs the data harder). The posterior is their balance.
                        </p>
                    </Card>

                    <Card title="The whole course, in one band" subtitle="the finale">
                        <p className={styles.note}>
                            Chapter 1 fit a single line. This one fits a <em>distribution</em> over lines and reports
                            what it doesn&apos;t know — the difference between a confident guess and an honest one.
                            From a shoebox of receipts to reasoning under uncertainty: that&apos;s the whole arc.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
