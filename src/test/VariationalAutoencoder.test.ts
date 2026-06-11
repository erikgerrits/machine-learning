import { describe, it, expect } from 'vitest';
import VariationalAutoencoder from '../lib/machine-learning/unsupervised/VariationalAutoencoder';
import Matrix from '../lib/math/linear-algebra/Matrix';

// 7×7 blobs again — a soft spot that only moves, so the data really has 2 degrees of freedom. A VAE
// should both reconstruct them and organise its latent space into a standard normal we can sample.
const W = 7;
const SIZE = W * W;
function blob(cx: number, cy: number): number[] {
    const img: number[] = [];
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) img.push(Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 4));
    return img;
}
function blobDataset(count: number, seed = 1): Matrix {
    let s = seed;
    const rand = () => (s = (s * 48271) % 2147483647) / 2147483647;
    const rows: number[][] = [];
    for (let i = 0; i < count; i++) rows.push(blob(1.5 + rand() * 4, 1.5 + rand() * 4));
    return new Matrix(rows);
}
const average = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

// One trained VAE, reused across the reconstruction / latent / sampling tests.
const data = blobDataset(80);
const trained = new VariationalAutoencoder().setHiddenSize(24).setCodeSize(2).setBeta(1).setLearningRate(0.06).setNumberOfEpochs(1200).setSeed(0);
const errorBeforeTraining = average(trained.reconstructionError(data));
trained.train(data);

describe('VariationalAutoencoder', () => {

    it('verifies its reparameterised + KL backprop against finite differences', () => {
        expect(new VariationalAutoencoder().checkGradients()).toBe(true);
    });

    it('learns to reconstruct its inputs', () => {
        expect(average(trained.reconstructionError(data))).toBeLessThan(errorBeforeTraining);
        expect(average(trained.reconstructionError(data))).toBeLessThan(0.45);
    });

    it('shapes the latent space into a standard normal', () => {
        const codes = trained.encode(data).toArray();
        for (let k = 0; k < 2; k++) {
            const column = codes.map(c => c[k]);
            const mean = average(column);
            const std = Math.sqrt(average(column.map(v => (v - mean) ** 2)));
            expect(Math.abs(mean)).toBeLessThan(0.4);          // centred near the origin…
            expect(std).toBeGreaterThan(0.6);                  // …with roughly unit spread — the N(0,1)
            expect(std).toBeLessThan(1.5);                     //   prior the KL term pulls it toward.
        }
    });

    it('generates new, varied, blob-like samples from the prior', () => {
        const samples = trained.sample(24, 9).toArray();
        expect(samples.length).toBe(24);
        expect(samples[0].length).toBe(SIZE);

        // Each sample has a clear bright spot (not washed-out noise)…
        const peaks = samples.map(img => Math.max(...img));
        expect(average(peaks)).toBeGreaterThan(0.5);

        // …and the blobs land in different places (the generator isn't collapsed to one image).
        const centroids = samples.map(img => {
            let cx = 0, sum = 0;
            for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { cx += x * img[y * W + x]; sum += img[y * W + x]; }
            return cx / (sum || 1);
        });
        const mean = average(centroids);
        const spread = Math.sqrt(average(centroids.map(v => (v - mean) ** 2)));
        expect(spread).toBeGreaterThan(0.3);
    });

    it('encodes, generates, and samples with the right shapes', () => {
        expect(trained.encode(data).getColumnCount()).toBe(2);
        expect(trained.generate(new Matrix([[0, 0], [1, -1]])).getColumnCount()).toBe(SIZE);
        const drawn = trained.sample(5, 3);
        expect(drawn.getRowCount()).toBe(5);
        expect(drawn.getColumnCount()).toBe(SIZE);
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const vae = new VariationalAutoencoder().setHiddenSize(8).setCodeSize(2).setNumberOfEpochs(80).setSeed(4);
            vae.train(blobDataset(20));
            return vae.sample(3, 1).toArray();
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const vae = new VariationalAutoencoder();
        expect(vae.getCodeSize()).toBe(2);

        const returned = vae.setInputSize(64).setHiddenSize(40).setCodeSize(3).setBeta(2).setLearningRate(0.01).setNumberOfEpochs(500).setSeed(7);
        expect(returned).toBe(vae);
        expect(vae.getInputSize()).toBe(64);
        expect(vae.getHiddenSize()).toBe(40);
        expect(vae.getCodeSize()).toBe(3);
        expect(vae.getBeta()).toBe(2);
        expect(vae.getLearningRate()).toBe(0.01);
        expect(vae.getNumberOfEpochs()).toBe(500);
        expect(vae.getSeed()).toBe(7);
    });
});
