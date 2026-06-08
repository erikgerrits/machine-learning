import { useMemo, useState } from 'react';
import { NaiveBayes, Matrix } from 'machine-learning';
import { CLASS_HEX } from '../viz/multiclass';
import { Card, ControlPanel, Hint, Select, Slider } from './controls/Controls';
import styles from './NaiveBayesPlayground.module.css';

const CLASSES = ['Reservation', 'Complaint', 'Spam'];

// Nadia's labelled inbox — the training set. Reservations lean on "table / book / reserve",
// complaints on "cold / rude / refund", spam on "free / win / claim / click".
const TRAIN: { text: string; classIndex: number }[] = [
    { text: 'Could we book a table for four on Friday evening?', classIndex: 0 },
    { text: "I'd like to reserve a table for two tomorrow at seven.", classIndex: 0 },
    { text: 'Do you have a table free for six people this weekend?', classIndex: 0 },
    { text: 'Can I make a booking for Saturday lunch please?', classIndex: 0 },
    { text: 'Please reserve a table by the window for tonight.', classIndex: 0 },
    { text: 'My coffee was cold and the service was painfully slow.', classIndex: 1 },
    { text: 'I waited twenty minutes and the order was completely wrong.', classIndex: 1 },
    { text: 'The croissant was stale and I would like a refund.', classIndex: 1 },
    { text: 'Very disappointed, the table was dirty when we arrived.', classIndex: 1 },
    { text: 'The staff were rude and the food was terrible.', classIndex: 1 },
    { text: 'Congratulations you have won a free prize, claim it now!', classIndex: 2 },
    { text: 'Earn money fast working from home, click this link.', classIndex: 2 },
    { text: 'Free crypto investment, double your cash today only.', classIndex: 2 },
    { text: 'You are a winner! Claim your free gift card now.', classIndex: 2 },
    { text: 'Cheap loans guaranteed approval, click here for free money.', classIndex: 2 },
];

const SAMPLES = [
    { id: 'booking', label: 'A booking', text: 'Could we book a table for four this Saturday evening?' },
    { id: 'complaint', label: 'A complaint', text: 'My latte arrived cold and the waiter was rude about it.' },
    { id: 'spam', label: 'Spam', text: 'Congratulations! Claim your free gift card now, click the link.' },
    { id: 'tricky', label: 'A tricky one', text: 'Is the free table by the window still available tonight?' },
];

const tokenize = (text: string) => (text.toLowerCase().match(/[a-z]+/g) ?? []).filter(t => t.length >= 3);

const oneHot = (index: number, classes: number) => {
    const row = new Array(classes).fill(0);
    row[index] = 1;
    return row;
};

const argmax = (values: number[]) => values.reduce((best, v, i) => (v > values[best] ? i : best), 0);

export function NaiveBayesPlayground() {
    const [sampleId, setSampleId] = useState(SAMPLES[0].id);
    const [input, setInput] = useState(SAMPLES[0].text);
    const [smoothingSlider, setSmoothingSlider] = useState(10);
    const smoothing = smoothingSlider / 10; // 0.1 … 5.0, never 0 (so no word can zero a class)

    // Vocabulary and the document-term training matrix — fixed, derived once from the inbox.
    const { vocab, vocabIndex, trainCounts, trainTargets } = useMemo(() => {
        const set = new Set<string>();
        TRAIN.forEach(m => tokenize(m.text).forEach(t => set.add(t)));
        const vocab = Array.from(set).sort();
        const vocabIndex = new Map(vocab.map((word, i) => [word, i]));

        const vectorize = (text: string) => {
            const counts = new Array(vocab.length).fill(0);
            tokenize(text).forEach(t => {
                const i = vocabIndex.get(t);
                if (i !== undefined) counts[i]++;
            });
            return counts;
        };

        return {
            vocab,
            vocabIndex,
            trainCounts: TRAIN.map(m => vectorize(m.text)),
            trainTargets: TRAIN.map(m => oneHot(m.classIndex, CLASSES.length)),
        };
    }, []);

    // Train the model (cheap; recompute when smoothing changes).
    const model = useMemo(() => {
        const nb = new NaiveBayes();
        nb.setSmoothing(smoothing);
        nb.train(new Matrix(trainCounts), new Matrix(trainTargets));
        return nb;
    }, [smoothing, trainCounts, trainTargets]);

    // For each vocabulary word, the class it points to most strongly, and the most telling words
    // per class (highest likelihood relative to the other classes).
    const { wordWinner, topWords } = useMemo(() => {
        const logLikelihoods = model.getLogLikelihoods();
        const wordWinner = new Map<string, number>();
        vocab.forEach((word, w) => {
            wordWinner.set(word, argmax(logLikelihoods.map(perClass => perClass[w])));
        });

        const topWords = CLASSES.map((_, c) => {
            const scored = vocab.map((word, w) => {
                const others = logLikelihoods.filter((_, k) => k !== c).map(perClass => perClass[w]);
                return { word, edge: logLikelihoods[c][w] - Math.max(...others) };
            });
            return scored.sort((a, b) => b.edge - a.edge).slice(0, 5).map(s => s.word);
        });

        return { wordWinner, topWords };
    }, [model, vocab]);

    const { probabilities, recognised } = useMemo(() => {
        const vector = new Array(vocab.length).fill(0);
        tokenize(input).forEach(t => {
            const i = vocabIndex.get(t);
            if (i !== undefined) vector[i]++;
        });
        const probabilities = model.predict(new Matrix([vector])).toArray()[0];
        const recognised = vector.reduce((sum, c) => sum + c, 0);
        return { probabilities, recognised };
    }, [model, input, vocab, vocabIndex]);

    const winner = argmax(probabilities);

    const loadSample = (id: string) => {
        const sample = SAMPLES.find(s => s.id === id);
        if (!sample) return;
        setSampleId(id);
        setInput(sample.text);
    };

    const words = input.split(/(\s+)/); // keep whitespace so the reading stays intact

    return (
        <div className={styles.playground}>
            <ControlPanel>
                <Select
                    label="Load a message"
                    value={sampleId}
                    options={SAMPLES.map(s => ({ value: s.id, label: s.label }))}
                    onChange={loadSample}
                />
                <Hint>…or just type your own below. The verdict updates as you write.</Hint>
                <Slider
                    label="Smoothing (α)"
                    value={smoothingSlider}
                    display={smoothing.toFixed(1)}
                    min={1}
                    max={50}
                    onChange={setSmoothingSlider}
                />
                <Hint>Smoothing keeps a single unfamiliar word from slamming a class to zero.</Hint>
            </ControlPanel>

            <div className={styles.stage}>
                <textarea
                    className={styles.textarea}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    spellCheck={false}
                    aria-label="Message to classify"
                />

                <Card title="Verdict" subtitle={recognised === 0 ? 'no familiar words — going on the base rate' : `${CLASSES[winner]}`}>
                    <div className={styles.bars}>
                        {CLASSES.map((label, c) => (
                            <div className={styles.barRow} key={label}>
                                <span className={styles.barLabel} style={c === winner ? { color: CLASS_HEX[c], fontWeight: 600 } : undefined}>
                                    {label}
                                </span>
                                <span className={styles.barTrack}>
                                    <span
                                        className={styles.barFill}
                                        style={{ width: `${(probabilities[c] * 100).toFixed(1)}%`, background: CLASS_HEX[c] }}
                                    />
                                </span>
                                <span className={styles.barPct}>{(probabilities[c] * 100).toFixed(0)}%</span>
                            </div>
                        ))}
                    </div>
                </Card>

                <Card title="What it reads" subtitle="words coloured by the class they point to">
                    <p className={styles.reading}>
                        {words.map((word, i) => {
                            const cleaned = word.toLowerCase().replace(/[^a-z]/g, '');
                            const owner = cleaned ? wordWinner.get(cleaned) : undefined;
                            if (owner === undefined) {
                                return <span key={i} className={styles.muted}>{word}</span>;
                            }
                            return (
                                <span key={i} className={styles.tell} style={{ color: CLASS_HEX[owner] }}>
                                    {word}
                                </span>
                            );
                        })}
                    </p>
                    <p className={styles.caption}>Grey words carry no signal — they weren't in the training inbox.</p>
                </Card>

                <Card title="What tips it off" subtitle="the words most telling of each class">
                    <div className={styles.tells}>
                        {CLASSES.map((label, c) => (
                            <div className={styles.tellRow} key={label}>
                                <span className={styles.tellClass} style={{ color: CLASS_HEX[c] }}>{label}</span>
                                <span className={styles.tellWords}>{topWords[c].join(' · ')}</span>
                            </div>
                        ))}
                    </div>
                </Card>
            </div>
        </div>
    );
}
