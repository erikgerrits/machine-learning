import { useCallback, useEffect, useRef, useState } from 'react';
import { Autoencoder, Matrix } from 'machine-learning';
import { AUTOENCODER_SCENARIOS, type AutoencoderScenario } from '../ml/autoencoderDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawLatentManifold, drawTriptych } from '../viz/autoencoder';
import { Card, ControlPanel, Hint, Metric, MetricsRow, RunControls, Select, Slider } from './controls/Controls';
import styles from './AutoencoderPlayground.module.css';

const EPOCHS_PER_FRAME = 20;
const DATA_COUNT = 150;
const MANIFOLD_RES = 8;

export function AutoencoderPlayground() {
    const [scenarioId, setScenarioId] = useState(AUTOENCODER_SCENARIOS[0].id);
    const [noiseSlider, setNoiseSlider] = useState(50); // noise = slider / 100
    const [lrSlider, setLrSlider] = useState(100);       // lr = slider / 100
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epochs: 0, error: 0 });

    const noise = noiseSlider / 100;
    const lr = lrSlider / 100;

    const aeRef = useRef<Autoencoder | null>(null);
    const dataRef = useRef<Matrix | null>(null);
    const colorsRef = useRef<number[]>([]);
    const scenarioRef = useRef<AutoencoderScenario>(AUTOENCODER_SCENARIOS[0]);
    const cleanRef = useRef<number[]>([]);
    const noiseVecRef = useRef<number[]>([]); // fixed unit-noise pattern, scaled by the slider
    const epochsRef = useRef(0);
    const manifoldCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const triptychCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const ae = aeRef.current;
        const data = dataRef.current;
        const scenario = scenarioRef.current;
        const manifoldCanvas = manifoldCanvasRef.current;
        const triptychCanvas = triptychCanvasRef.current;
        if (!ae || !data || !manifoldCanvas || !triptychCanvas) return;

        // Latent manifold: where the real images land, then a grid of decoded thumbnails over that range.
        const codes = ae.encode(data).toArray();
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const [a, b] of codes) { minX = Math.min(minX, a); maxX = Math.max(maxX, a); minY = Math.min(minY, b); maxY = Math.max(maxY, b); }
        const padX = (maxX - minX) * 0.15 + 1e-3;
        const padY = (maxY - minY) * 0.15 + 1e-3;
        minX -= padX; maxX += padX; minY -= padY; maxY += padY;

        const gridCodes: number[][] = [];
        for (let r = 0; r < MANIFOLD_RES; r++) {
            for (let c = 0; c < MANIFOLD_RES; c++) {
                gridCodes.push([minX + (maxX - minX) * (c / (MANIFOLD_RES - 1)), minY + (maxY - minY) * (r / (MANIFOLD_RES - 1))]);
            }
        }
        const tiles = ae.decode(new Matrix(gridCodes)).toArray();
        drawLatentManifold(manifoldCanvas, {
            gridRes: MANIFOLD_RES,
            tiles,
            imageWidth: scenario.width,
            dataCodes: codes,
            dataColors: colorsRef.current,
            codeMin: [minX, minY],
            codeMax: [maxX, maxY],
            showData: true,
        });

        // Denoising triptych: clean → noisy → reconstruction of the noisy one.
        const clean = cleanRef.current;
        const noisy = clean.map((v, i) => Math.min(1, Math.max(0, v + noiseVecRef.current[i] * noise)));
        const reconstructed = ae.reconstruct(new Matrix([noisy])).toArray()[0];
        drawTriptych(triptychCanvas, {
            images: [clean, noisy, reconstructed],
            labels: ['clean', 'noisy', 'rebuilt'],
            imageWidth: scenario.width,
        });
    }, [noise]);

    const rebuild = useCallback(() => {
        const scenario = AUTOENCODER_SCENARIOS.find(s => s.id === scenarioId) ?? AUTOENCODER_SCENARIOS[0];
        const { images, colors } = scenario.generate(seed + 1, DATA_COUNT);
        const ae = new Autoencoder().setHiddenSizes([24]).setCodeSize(2).setLearningRate(lr).setNumberOfEpochs(EPOCHS_PER_FRAME).setSeed(seed);

        aeRef.current = ae;
        dataRef.current = new Matrix(images);
        colorsRef.current = colors;
        scenarioRef.current = scenario;
        cleanRef.current = scenario.example();

        // A fixed random noise pattern (in [-1, 1]) so dragging the noise slider is smooth, not jumpy.
        let s = seed + 7;
        const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
        noiseVecRef.current = scenario.example().map(() => rand() * 2 - 1);

        epochsRef.current = 0;
        setMetrics({ epochs: 0, error: 0 });
        setRunning(false);
        ae.train(dataRef.current); // one chunk so the network exists to draw
        epochsRef.current = EPOCHS_PER_FRAME;
        draw();
    }, [scenarioId, lr, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const ae = aeRef.current;
        const data = dataRef.current;
        if (!ae || !data) return;

        ae.train(data); // EPOCHS_PER_FRAME more epochs
        epochsRef.current += EPOCHS_PER_FRAME;
        draw();
        if (epochsRef.current % (EPOCHS_PER_FRAME * 4) === 0) {
            const error = ae.reconstructionError(data).reduce((a, b) => a + b, 0) / data.getRowCount();
            setMetrics({ epochs: epochsRef.current, error: Math.round(error * 10000) / 10000 });
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

    // Redraw the triptych live when the noise slider moves, even while paused.
    useEffect(() => {
        draw();
    }, [noise, draw]);

    const scenario = AUTOENCODER_SCENARIOS.find(s => s.id === scenarioId);

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
                    label="Data"
                    value={scenarioId}
                    options={AUTOENCODER_SCENARIOS.map(s => ({ value: s.id, label: s.label }))}
                    onChange={setScenarioId}
                />
                {scenario && <Hint>{scenario.blurb}</Hint>}
                <Slider label="Noise" value={noiseSlider} display={noise.toFixed(2)} min={0} max={100} onChange={setNoiseSlider} />
                <Slider label="Learning rate" value={lrSlider} display={lr.toFixed(2)} min={20} max={150} onChange={setLrSlider} />
                <Slider label="Random seed" value={seed} display={String(seed)} min={0} max={9} onChange={setSeed} />
                <Hint>Press Train and watch the tiles morph from noise into smooth images sweeping across the data — the 2-D map the bottleneck learned.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.gridWrap}>
                    <canvas ref={manifoldCanvasRef} className={styles.grid} />
                    <div className={styles.caption}>latent manifold · each tile is a code decoded back to an image; dots are real data</div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epochs" value={String(metrics.epochs)} />
                        <Metric label="Recon error" value={metrics.error.toFixed(4)} />
                    </MetricsRow>

                    <div className={styles.triptychWrap}>
                        <canvas ref={triptychCanvasRef} className={styles.triptych} />
                    </div>

                    <Card title="Squeeze, then rebuild" subtitle="denoising">
                        <p className={styles.note}>
                            The middle image is the clean one with the <strong>Noise</strong> slider&apos;s speckle
                            added; the right is the autoencoder&apos;s reconstruction. Because noise can&apos;t fit
                            through the 2-number bottleneck, it gets dropped — the network rebuilds only the clean
                            blob it knows. Crank the noise and watch how much it can still recover.
                        </p>
                    </Card>

                    <Card title="A learned map" subtitle="the manifold">
                        <p className={styles.note}>
                            The big panel decodes a grid of codes back into images, so you see what the 2-D code
                            <em> means</em>: move across it and the picture sweeps smoothly through the data. The
                            dots are the real images&apos; codes. On <em>Sliding bar</em> they fan into a clear colour
                            gradient — one dominant direction, the data&apos;s single degree of freedom.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
