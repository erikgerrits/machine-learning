import { describe, it, expect } from 'vitest';
import Transformer from '../lib/machine-learning/supervised/Transformer';
import Matrix from '../lib/math/linear-algebra/Matrix';

// A task that genuinely needs attention: each length-5 sequence is [CLS, w, w, w, w] where the four
// words are fillers (1) except for exactly one keyword — A (2) or B (3) — at a random position. The
// class is which keyword is present, so the model must find the salient word *wherever it sits* and
// route it to the CLS slot. Vocab: 0=CLS, 1=filler, 2=keyA, 3=keyB.
const SEQUENCES: number[][] = [];
const LABELS: number[][] = [];
for (let pos = 1; pos <= 4; pos++) {
    for (const key of [2, 3]) {
        const seq = [0, 1, 1, 1, 1];
        seq[pos] = key;
        SEQUENCES.push(seq);
        LABELS.push(key === 2 ? [1, 0] : [0, 1]);
    }
}

describe('Transformer', () => {

    it('verifies its attention backprop against finite differences', () => {
        const small = new Transformer().setVocabSize(4).setModelDim(4).setMaxLength(4).setSeed(1);
        expect(small.checkGradients()).toBe(true);

        const bigger = new Transformer().setVocabSize(6).setModelDim(8).setMaxLength(5).setSeed(5);
        expect(bigger.checkGradients()).toBe(true);
    });

    it('learns to find the keyword wherever it sits', () => {
        const model = new Transformer()
            .setVocabSize(4).setModelDim(8).setMaxLength(5)
            .setLearningRate(0.05).setNumberOfEpochs(600).setSeed(0);
        model.train(new Matrix(SEQUENCES), new Matrix(LABELS));

        const predicted = model.predict(new Matrix(SEQUENCES)).getMaximumRowIndeces().toArray().map(r => r[0]);
        const truth = LABELS.map(row => (row[0] === 1 ? 0 : 1));
        expect(predicted.filter((p, i) => p === truth[i]).length).toBe(SEQUENCES.length);
    });

    it('returns class probabilities that sum to 1', () => {
        const model = new Transformer().setVocabSize(4).setModelDim(8).setMaxLength(5).setSeed(0);
        model.train(new Matrix(SEQUENCES), new Matrix(LABELS));

        const predictions = model.predict(new Matrix(SEQUENCES)).toArray();
        expect(predictions[0].length).toBe(2);
        for (const row of predictions) expect(row[0] + row[1]).toBeCloseTo(1, 6);
    });

    it('exposes an attention matrix whose rows are distributions', () => {
        const model = new Transformer().setVocabSize(4).setModelDim(8).setMaxLength(5).setSeed(0);
        model.train(new Matrix(SEQUENCES), new Matrix(LABELS));

        const attention = model.getAttention(SEQUENCES[0]);
        expect(attention.length).toBe(5);       // L × L
        expect(attention[0].length).toBe(5);
        for (const row of attention) {
            expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6); // each row softmaxes to 1
        }
    });

    it('is deterministic and the loss falls', () => {
        const run = () => {
            const model = new Transformer().setVocabSize(4).setModelDim(8).setMaxLength(5).setNumberOfEpochs(80).setSeed(2);
            model.train(new Matrix(SEQUENCES), new Matrix(LABELS));
            return model.predict(new Matrix(SEQUENCES)).toArray();
        };
        expect(run()).toEqual(run());

        const fresh = new Transformer().setVocabSize(4).setModelDim(8).setMaxLength(5).setSeed(2);
        const initial = fresh.setNumberOfEpochs(0).train(new Matrix(SEQUENCES), new Matrix(LABELS)).computeLoss(new Matrix(SEQUENCES), new Matrix(LABELS));
        const after = fresh.setNumberOfEpochs(200).train(new Matrix(SEQUENCES), new Matrix(LABELS)).computeLoss(new Matrix(SEQUENCES), new Matrix(LABELS));
        expect(after).toBeLessThan(initial);
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new Transformer();
        expect(model.getModelDim()).toBe(8);

        const returned = model.setVocabSize(30).setModelDim(16).setMaxLength(12).setLearningRate(0.01).setNumberOfEpochs(50).setSeed(7);
        expect(returned).toBe(model);
        expect(model.getVocabSize()).toBe(30);
        expect(model.getModelDim()).toBe(16);
        expect(model.getMaxLength()).toBe(12);
        expect(model.getLearningRate()).toBe(0.01);
        expect(model.getNumberOfEpochs()).toBe(50);
        expect(model.getSeed()).toBe(7);
    });
});
