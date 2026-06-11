import { describe, it, expect } from 'vitest';
import Autoencoder from '../lib/machine-learning/unsupervised/Autoencoder';
import Matrix from '../lib/math/linear-algebra/Matrix';

// 7×7 "blobs": a soft bright spot at some (cx, cy). The images live in 49 pixels but really only vary
// in 2 ways — where the spot is — so an autoencoder with a 2-number code should capture them well.
const W = 7;
const SIZE = W * W;
function blob(cx: number, cy: number): number[] {
    const img: number[] = [];
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) img.push(Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 5));
    return img;
}

function blobDataset(count: number, seed = 1): Matrix {
    let s = seed;
    const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
    const rows: number[][] = [];
    for (let i = 0; i < count; i++) rows.push(blob(1.5 + rand() * 4, 1.5 + rand() * 4));
    return new Matrix(rows);
}

const mse = (a: number[], b: number[]) => a.reduce((sum, _, i) => sum + (a[i] - b[i]) ** 2, 0) / a.length;

// One trained autoencoder, reused across the reconstruction / compression / denoising / anomaly tests.
const data = blobDataset(60);
const trained = new Autoencoder().setHiddenSizes([24]).setCodeSize(2).setLearningRate(1).setNumberOfEpochs(800).setSeed(0);
const errorBeforeTraining = average(trained.reconstructionError(data));
trained.train(data);

function average(values: number[]) {
    return values.reduce((a, b) => a + b, 0) / values.length;
}

describe('Autoencoder', () => {

    it('verifies its backprop against finite differences', () => {
        expect(new Autoencoder().checkGradients()).toBe(true);
    });

    it('learns to reconstruct its input through the bottleneck', () => {
        const errorAfter = average(trained.reconstructionError(data));
        expect(errorAfter).toBeLessThan(errorBeforeTraining); // training helped
        expect(errorAfter).toBeLessThan(0.05);                // and reconstructs well
    });

    it('compresses to a code of the requested size and decodes back', () => {
        const codes = trained.encode(data);
        expect(codes.getColumnCount()).toBe(2);                 // 49 pixels → 2 numbers
        expect(codes.getRowCount()).toBe(60);

        const rebuilt = trained.decode(codes);
        expect(rebuilt.getColumnCount()).toBe(SIZE);
        expect(trained.reconstruct(data).getColumnCount()).toBe(SIZE);
    });

    it('denoises by squeezing noise out through the bottleneck', () => {
        let s = 99;
        const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
        const clean = blob(3, 3);
        const noisy = clean.map(v => Math.min(1, Math.max(0, v + (rand() - 0.5) * 0.7)));

        const reconstructed = trained.reconstruct(new Matrix([noisy])).toArray()[0];
        expect(mse(reconstructed, clean)).toBeLessThan(mse(noisy, clean)); // closer to clean than the noisy input
    });

    it('flags anomalies by their high reconstruction error', () => {
        let s = 5;
        const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
        const anomaly = Array.from({ length: SIZE }, () => rand()); // pure noise — nothing like a blob

        const normalError = trained.reconstructionError(new Matrix([blob(3, 3)]))[0];
        const anomalyError = trained.reconstructionError(new Matrix([anomaly]))[0];
        expect(anomalyError).toBeGreaterThan(normalError * 3);
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const ae = new Autoencoder().setHiddenSizes([8]).setCodeSize(2).setNumberOfEpochs(100).setSeed(4);
            ae.train(blobDataset(20));
            return ae.reconstruct(blobDataset(3, 2)).toArray();
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const ae = new Autoencoder();
        expect(ae.getCodeSize()).toBe(2);

        const returned = ae.setInputSize(64).setHiddenSizes([32, 16]).setCodeSize(3).setLearningRate(0.2).setNumberOfEpochs(500).setSeed(7);
        expect(returned).toBe(ae);
        expect(ae.getInputSize()).toBe(64);
        expect(ae.getHiddenSizes()).toEqual([32, 16]);
        expect(ae.getCodeSize()).toBe(3);
        expect(ae.getLearningRate()).toBe(0.2);
        expect(ae.getNumberOfEpochs()).toBe(500);
        expect(ae.getSeed()).toBe(7);
    });
});
