import { useCallback, useEffect, useRef, useState } from 'react';
import { AssociationRules, Matrix, type AssociationRule } from 'machine-learning';
import { ASSOCIATION_DATASETS } from '../ml/associationDatasets';
import { fitCanvas } from '../viz/canvas';
import { drawAssociationWeb } from '../viz/association';
import { Card, ControlPanel, Hint, Metric, MetricsRow, NumberField, Select, Slider } from './controls/Controls';
import styles from './AssociationRulesPlayground.module.css';

const POINTS = 240;

export function AssociationRulesPlayground() {
    const [datasetId, setDatasetId] = useState(ASSOCIATION_DATASETS[0].id);
    const [supportSlider, setSupportSlider] = useState(ASSOCIATION_DATASETS[0].minSupport * 100);
    const [confidenceSlider, setConfidenceSlider] = useState(ASSOCIATION_DATASETS[0].minConfidence * 100);
    const [seed, setSeed] = useState(0);
    const [rules, setRules] = useState<AssociationRule[]>([]);
    const [itemsetCount, setItemsetCount] = useState(0);

    const minSupport = supportSlider / 100;
    const minConfidence = confidenceSlider / 100;

    const dataset = ASSOCIATION_DATASETS.find(d => d.id === datasetId) ?? ASSOCIATION_DATASETS[0];
    const items = dataset.items;

    const rulesRef = useRef<AssociationRule[]>([]);
    const itemSupportRef = useRef<number[]>([]);
    const itemsRef = useRef<string[]>(items);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const { ctx, width, height } = fitCanvas(canvas);
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(0, 0, width, height);
        drawAssociationWeb(ctx, width, height, itemsRef.current, rulesRef.current, itemSupportRef.current);
    }, []);

    const rebuild = useCallback(() => {
        const active = ASSOCIATION_DATASETS.find(d => d.id === datasetId) ?? ASSOCIATION_DATASETS[0];
        const { inputs } = active.generate(seed, POINTS);

        const model = new AssociationRules().setMinSupport(minSupport).setMinConfidence(minConfidence).setMaxItemsetSize(3);
        model.train(new Matrix(inputs));

        const found = model.getRules();
        rulesRef.current = found;
        itemsRef.current = active.items;
        itemSupportRef.current = active.items.map((_, i) =>
            inputs.reduce((sum, row) => sum + row[i], 0) / inputs.length,
        );

        setRules(found);
        setItemsetCount(model.getFrequentItemsets().length);
        draw();
    }, [datasetId, minSupport, minConfidence, seed, draw]);

    // Debounced so dragging the support / confidence bars (each re-mines the rules) stays smooth.
    useEffect(() => {
        const timer = setTimeout(rebuild, 60);
        return () => clearTimeout(timer);
    }, [rebuild]);

    const handleDataset = (id: string) => {
        const next = ASSOCIATION_DATASETS.find(d => d.id === id);
        if (!next) return;
        setDatasetId(next.id);
        setSupportSlider(next.minSupport * 100);
        setConfidenceSlider(next.minConfidence * 100);
    };

    const names = (indices: number[]) => indices.map(i => items[i]).join(' + ');

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Receipts"
                    value={datasetId}
                    options={ASSOCIATION_DATASETS.map(d => ({ value: d.id, label: d.label }))}
                    onChange={handleDataset}
                />
                {dataset && <Hint>{dataset.blurb}</Hint>}
                <Slider
                    label="Min support"
                    value={supportSlider}
                    display={`${supportSlider}%`}
                    min={3}
                    max={50}
                    onChange={setSupportSlider}
                />
                <Slider
                    label="Min confidence"
                    value={confidenceSlider}
                    display={`${confidenceSlider}%`}
                    min={20}
                    max={100}
                    onChange={setConfidenceSlider}
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} />
                <Hint>Thicker, brighter arrows are stronger associations (higher lift).</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <div className={styles.boundaryWrap}>
                    <canvas ref={canvasRef} className={styles.boundary} />
                </div>

                <div className={styles.side}>
                    <MetricsRow>
                        <Metric label="Rules found" value={String(rules.length)} />
                        <Metric label="Frequent sets" value={String(itemsetCount)} />
                    </MetricsRow>

                    <Card title="Top rules" subtitle="sorted by lift">
                        {rules.length === 0 ? (
                            <p className={styles.note}>No rules clear the bars — lower min support or confidence.</p>
                        ) : (
                            <ul className={styles.ruleList}>
                                {rules.slice(0, 7).map((rule, i) => (
                                    <li key={i} className={styles.rule}>
                                        <span className={styles.ruleText}>
                                            {names(rule.antecedent)} <span className={styles.arrow}>→</span> {names(rule.consequent)}
                                        </span>
                                        <span className={styles.ruleStats}>
                                            conf {(rule.confidence * 100).toFixed(0)}% · lift {rule.lift.toFixed(2)}
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </Card>

                    <Card title="Support, confidence, lift" subtitle="how a rule earns its place">
                        <p className={styles.note}>
                            <strong>Support</strong> is how common the combo is; <strong>confidence</strong>{' '}
                            is how reliably the consequent follows; <strong>lift</strong> is how much more
                            than chance they appear together. Lift above 1 is a real pairing — below 1 they
                            actually repel.
                        </p>
                    </Card>
                </div>
            </div>
        </div>
    );
}
