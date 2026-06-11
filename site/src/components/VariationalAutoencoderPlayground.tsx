import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { VariationalAutoencoder, Matrix } from 'machine-learning';
import { AUTOENCODER_SCENARIOS, type AutoencoderScenario } from '../ml/autoencoderDatasets';
import { useAnimationFrame } from '../hooks/useAnimationFrame';
import { drawGenerativeManifold, drawSampleGallery } from '../viz/vae';
import { Card, ControlPanel, Hint, Metric, MetricsRow, RunControls, Select, Slider } from './controls/Controls';
import styles from './VariationalAutoencoderPlayground.module.css';

const EPOCHS_PER_FRAME = 15;
const DATA_COUNT = 160;
const MANIFOLD_RES = 9;     // tiles per side over the prior
const PRIOR_SPAN = 2.2;     // grid spans [-PRIOR_SPAN, PRIOR_SPAN]² of the N(0,1) latent
const GALLERY = 16;

// Box–Muller standard-normal from a uniform generator.
function gaussian(rand: () => number) {
    let u = 0; while (u === 0) u = rand();
    let v = 0; while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function VariationalAutoencoderPlayground() {
    const [scenarioId, setScenarioId] = useState(AUTOENCODER_SCENARIOS[0].id);
    const [betaSlider, setBetaSlider] = useState(100); // beta = slider / 100
    const [lrSlider, setLrSlider] = useState(6);        // lr = slider / 100
    const [seed, setSeed] = useState(0);
    const [running, setRunning] = useState(false);
    const [metrics, setMetrics] = useState({ epochs: 0, loss: 0 });

    const beta = betaSlider / 100;
    const lr = lrSlider / 100;

    // Fixed prior grid (constant) for the generative manifold.
    const gridCodes = useMemo(() => {
        const pts: number[][] = [];
        for (let r = 0; r < MANIFOLD_RES; r++) {
            for (let c = 0; c < MANIFOLD_RES; c++) {
                pts.push([
                    -PRIOR_SPAN + (2 * PRIOR_SPAN) * (c / (MANIFOLD_RES - 1)),
                    -PRIOR_SPAN + (2 * PRIOR_SPAN) * (r / (MANIFOLD_RES - 1)),
                ]);
            }
        }
        return pts;
    }, []);

    const vaeRef = useRef<VariationalAutoencoder | null>(null);
    const dataRef = useRef<Matrix | null>(null);
    const colorsRef = useRef<number[]>([]);
    const scenarioRef = useRef<AutoencoderScenario>(AUTOENCODER_SCENARIOS[0]);
    const galleryCodesRef = useRef<number[][]>([]);
    const epochsRef = useRef(0);
    const manifoldCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const galleryCanvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const vae = vaeRef.current;
        const data = dataRef.current;
        const scenario = scenarioRef.current;
        const manifoldCanvas = manifoldCanvasRef.current;
        const galleryCanvas = galleryCanvasRef.current;
        if (!vae || !data || !manifoldCanvas || !galleryCanvas) return;

        const tiles = vae.generate(new Matrix(gridCodes)).toArray();
        drawGenerativeManifold(manifoldCanvas, {
            gridRes: MANIFOLD_RES,
            tiles,
            imageWidth: scenario.width,
            dataCodes: vae.encode(data).toArray(),
            dataColors: colorsRef.current,
            codeMin: [-PRIOR_SPAN, -PRIOR_SPAN],
            codeMax: [PRIOR_SPAN, PRIOR_SPAN],
            showData: true,
        });

        const samples = vae.generate(new Matrix(galleryCodesRef.current)).toArray();
        drawSampleGallery(galleryCanvas, { images: samples, imageWidth: scenario.width, cols: 4 });
    }, [gridCodes]);

    const rebuild = useCallback(() => {
        const scenario = AUTOENCODER_SCENARIOS.find(s => s.id === scenarioId) ?? AUTOENCODER_SCENARIOS[0];
        const { images, colors } = scenario.generate(seed + 1, DATA_COUNT);
        const vae = new VariationalAutoencoder()
            .setHiddenSize(32).setCodeSize(2).setBeta(beta).setLearningRate(lr).setNumberOfEpochs(EPOCHS_PER_FRAME).setSeed(seed);

        vaeRef.current = vae;
        dataRef.current = new Matrix(images);
        colorsRef.current = colors;
        scenarioRef.current = scenario;

        // A fixed set of random latent codes, so the gallery shows the *same* draws sharpening over time.
        let s = seed + 31;
        const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
        galleryCodesRef.current = Array.from({ length: GALLERY }, () => [gaussian(rand), gaussian(rand)]);

        epochsRef.current = 0;
        setMetrics({ epochs: 0, loss: 0 });
        setRunning(false);
        vae.train(dataRef.current); // one chunk so the network exists to draw
        epochsRef.current = EPOCHS_PER_FRAME;
        draw();
    }, [scenarioId, beta, lr, seed, draw]);

    useEffect(() => {
        rebuild();
    }, [rebuild]);

    const step = useCallback(() => {
        const vae = vaeRef.current;
        const data = dataRef.current;
        if (!vae || !data) return;

        vae.train(data);
        epochsRef.current += EPOCHS_PER_FRAME;
        draw();
        if (epochsRef.current % (EPOCHS_PER_FRAME * 4) === 0) {
            setMetrics({ epochs: epochsRef.current, loss: Math.round(vae.computeLoss(data) * 1000) / 1000 });
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
                <Slider label="KL weight β" value={betaSlider} display={beta.toFixed(2)} min={0} max={300} onChange={setBetaSlider} />
                <Slider label="Learning rate" value={lrSlider} display={lr.toFixed(2)} min={2} max={15} onChange={setLrSlider} />
                <Slider label="Random seed" value={seed} display={String(seed)} min={0} max={9} onChange={setSeed} />
                <Hint>Press Train: the gallery&apos;s random codes sharpen into real samples, and the big panel fills with images the VAE invents across its latent space.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.gridWrap}>
                    <canvas ref={manifoldCanvasRef} className={styles.grid} />
                    <div className={styles.caption}>generative manifold · every tile is a latent code decoded into a new image; dots are real data</div>
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Epochs" value={String(metrics.epochs)} />
                        <Metric label="Loss" value={metrics.loss.toFixed(3)} />
                    </MetricsRow>

                    <div className={styles.galleryWrap}>
                        <canvas ref={galleryCanvasRef} className={styles.gallery} />
                    </div>

                    <Card title="Samples from nothing" subtitle="z ~ N(0,1) → decode">
                        <p className={styles.note}>
                            These 16 images are decoded from <em>fixed random codes</em> — pure noise drawn from
                            a standard normal. Watch them morph from static into clean samples as training pulls
                            the latent space into shape. None are copied from the data; the VAE <em>invents</em> them.
                        </p>
                    </Card>

                    <Card title="Why a distribution" subtitle="the KL weight β">
                        <p className={styles.note}>
                            A plain autoencoder leaves gaps between codes, so a random draw decodes to garbage.
                            The <strong>KL term</strong> presses every input&apos;s code toward N(0,1), packing them
                            into one smooth blob with no gaps — so sampling works. Raise <strong>β</strong> for a
                            tidier, more samplable space (but blurrier); lower it for sharper images on a gappier map.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
