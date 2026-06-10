import { describe, it, expect } from 'vitest';
import ConvolutionalNeuralNetwork from '../lib/machine-learning/supervised/ConvolutionalNeuralNetwork';
import Matrix from '../lib/math/linear-algebra/Matrix';

// A tiny translation-invariant task: an 8×8 image with a single horizontal line (class 0) or a
// single vertical line (class 1), at varying positions. A CNN should learn to spot the orientation
// wherever it sits — exactly what a dense net struggles with.
const SIZE = 8;
function line(orientation: 'h' | 'v', position: number): number[] {
    const image = new Array<number>(SIZE * SIZE).fill(0);
    for (let k = 0; k < SIZE; k++) {
        image[orientation === 'h' ? position * SIZE + k : k * SIZE + position] = 1;
    }
    return image;
}
const POSITIONS = [1, 2, 4, 5, 6];
const IMAGES = [...POSITIONS.map(p => line('h', p)), ...POSITIONS.map(p => line('v', p))];
const LABELS = [...POSITIONS.map(() => [1, 0]), ...POSITIONS.map(() => [0, 1])];

describe('ConvolutionalNeuralNetwork', () => {

    it('verifies its convolution backprop against finite differences', () => {
        const small = new ConvolutionalNeuralNetwork().setInputShape(6, 6).setFilterCount(2).setSeed(1);
        expect(small.checkGradients()).toBe(true);

        // …and at the default (larger) configuration too.
        const big = new ConvolutionalNeuralNetwork().setInputShape(8, 8).setFilterCount(4).setSeed(3);
        expect(big.checkGradients()).toBe(true);
    });

    it('learns horizontal-vs-vertical regardless of position', () => {
        const model = new ConvolutionalNeuralNetwork()
            .setInputShape(SIZE, SIZE)
            .setFilterCount(4)
            .setLearningRate(0.3)
            .setNumberOfEpochs(400)
            .setSeed(0);
        model.train(new Matrix(IMAGES), new Matrix(LABELS));

        const predicted = model.predict(new Matrix(IMAGES)).getMaximumRowIndeces().toArray().map(row => row[0]);
        const truth = LABELS.map(row => (row[0] === 1 ? 0 : 1));
        const correct = predicted.filter((p, i) => p === truth[i]).length;
        expect(correct).toBe(IMAGES.length); // all of them
    });

    it('returns class probabilities that sum to 1', () => {
        const model = new ConvolutionalNeuralNetwork().setInputShape(SIZE, SIZE).setFilterCount(3).setSeed(0);
        model.train(new Matrix(IMAGES), new Matrix(LABELS));

        const predictions = model.predict(new Matrix(IMAGES)).toArray();
        expect(predictions[0].length).toBe(2);
        for (const row of predictions) {
            expect(row[0] + row[1]).toBeCloseTo(1, 6);
        }
    });

    it('is deterministic for a fixed seed, and the loss falls', () => {
        const train = () => {
            const model = new ConvolutionalNeuralNetwork().setInputShape(SIZE, SIZE).setFilterCount(4).setNumberOfEpochs(50).setSeed(2);
            const before = model.train(new Matrix(IMAGES), new Matrix(LABELS)).computeLoss(new Matrix(IMAGES), new Matrix(LABELS));
            return { predictions: model.predict(new Matrix(IMAGES)).toArray(), before };
        };
        const a = train();
        const b = train();
        expect(a.predictions).toEqual(b.predictions);

        const fresh = new ConvolutionalNeuralNetwork().setInputShape(SIZE, SIZE).setFilterCount(4).setSeed(2);
        const initialLoss = fresh.setNumberOfEpochs(0).train(new Matrix(IMAGES), new Matrix(LABELS)).computeLoss(new Matrix(IMAGES), new Matrix(LABELS));
        expect(a.before).toBeLessThan(initialLoss);
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const model = new ConvolutionalNeuralNetwork();
        expect(model.getFilterCount()).toBe(6);

        const returned = model.setInputShape(10, 10).setFilterCount(8).setFilterSize(5).setLearningRate(0.05).setNumberOfEpochs(20).setSeed(9);
        expect(returned).toBe(model);
        expect(model.getFilterCount()).toBe(8);
        expect(model.getFilterSize()).toBe(5);
        expect(model.getLearningRate()).toBe(0.05);
        expect(model.getNumberOfEpochs()).toBe(20);
        expect(model.getSeed()).toBe(9);
    });
});
