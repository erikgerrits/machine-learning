import { describe, it, expect } from 'vitest';
import RecurrentNeuralNetwork from '../lib/machine-learning/supervised/RecurrentNeuralNetwork';
import Matrix from '../lib/math/linear-algebra/Matrix';

// A small order-sensitive task over tokens {1,2,3} (0 = padding): class A if the sequence STARTS
// with token 1, else class B. The label can't be read from any single position — the net has to
// carry "what did I see first?" forward through the whole sequence.
const SEQUENCES = [
    [1, 2, 3, 0], [1, 3, 2, 0], [1, 2, 2, 0], [1, 3, 3, 0],
    [2, 1, 3, 0], [3, 2, 1, 0], [2, 3, 3, 0], [3, 1, 1, 0],
];
const LABELS = SEQUENCES.map(s => (s[0] === 1 ? [1, 0] : [0, 1]));

describe('RecurrentNeuralNetwork', () => {

    it('verifies its backprop-through-time against finite differences', () => {
        const small = new RecurrentNeuralNetwork().setVocabSize(4).setEmbeddingDim(3).setHiddenSize(4).setSeed(1);
        expect(small.checkGradients()).toBe(true);

        const bigger = new RecurrentNeuralNetwork().setVocabSize(6).setEmbeddingDim(2).setHiddenSize(8).setSeed(4);
        expect(bigger.checkGradients()).toBe(true);
    });

    it('learns an order-sensitive sequence task', () => {
        const model = new RecurrentNeuralNetwork()
            .setVocabSize(4).setEmbeddingDim(4).setHiddenSize(10)
            .setLearningRate(0.1).setNumberOfEpochs(800).setSeed(0);
        model.train(new Matrix(SEQUENCES), new Matrix(LABELS));

        const predicted = model.predict(new Matrix(SEQUENCES)).getMaximumRowIndeces().toArray().map(row => row[0]);
        const truth = LABELS.map(row => (row[0] === 1 ? 0 : 1));
        const correct = predicted.filter((p, i) => p === truth[i]).length;
        expect(correct).toBe(SEQUENCES.length);
    });

    it('returns class probabilities that sum to 1', () => {
        const model = new RecurrentNeuralNetwork().setVocabSize(4).setHiddenSize(6).setSeed(0);
        model.train(new Matrix(SEQUENCES), new Matrix(LABELS));

        const predictions = model.predict(new Matrix(SEQUENCES)).toArray();
        expect(predictions[0].length).toBe(2);
        for (const row of predictions) {
            expect(row[0] + row[1]).toBeCloseTo(1, 6);
        }
    });

    it('exposes learned embeddings of the right shape, and the loss falls', () => {
        const model = new RecurrentNeuralNetwork().setVocabSize(4).setEmbeddingDim(2).setHiddenSize(8).setSeed(2);
        const initial = model.setNumberOfEpochs(0).train(new Matrix(SEQUENCES), new Matrix(LABELS)).computeLoss(new Matrix(SEQUENCES), new Matrix(LABELS));
        const after = model.setNumberOfEpochs(300).train(new Matrix(SEQUENCES), new Matrix(LABELS)).computeLoss(new Matrix(SEQUENCES), new Matrix(LABELS));

        expect(after).toBeLessThan(initial);
        const embeddings = model.getEmbeddings();
        expect(embeddings.length).toBe(4);     // vocab size
        expect(embeddings[0].length).toBe(2);  // embedding dim
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const model = new RecurrentNeuralNetwork().setVocabSize(4).setHiddenSize(8).setNumberOfEpochs(100).setSeed(3);
            model.train(new Matrix(SEQUENCES), new Matrix(LABELS));
            return model.predict(new Matrix(SEQUENCES)).toArray();
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new RecurrentNeuralNetwork();
        expect(model.getHiddenSize()).toBe(12);

        const returned = model.setVocabSize(30).setEmbeddingDim(4).setHiddenSize(16).setLearningRate(0.05).setNumberOfEpochs(40).setSeed(9);
        expect(returned).toBe(model);
        expect(model.getVocabSize()).toBe(30);
        expect(model.getEmbeddingDim()).toBe(4);
        expect(model.getHiddenSize()).toBe(16);
        expect(model.getLearningRate()).toBe(0.05);
        expect(model.getNumberOfEpochs()).toBe(40);
        expect(model.getSeed()).toBe(9);
    });
});
