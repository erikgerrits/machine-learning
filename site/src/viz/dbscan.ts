import type { Domain } from '../ml/datasets';
import { CLUSTER_RGB, clusterHex } from './clusters';

/**
 * Paints the density map into a size×size offscreen canvas. Each cell carries the integer label
 * DBSCAN's `predict` returned: a cluster index gets that cluster's colour, while **noise** (`-1`)
 * is left fully transparent so the dark backdrop shows through — the empty space between dense
 * regions reads as exactly that, empty.
 */
export function paintDensity(offscreen: HTMLCanvasElement, labels: number[], size: number): void {
    if (offscreen.width !== size || offscreen.height !== size) {
        offscreen.width = size;
        offscreen.height = size;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(size, size);
    for (let k = 0; k < labels.length; k++) {
        const label = labels[k];
        const o = k * 4;
        if (label < 0) {
            image.data[o + 3] = 0; // noise → transparent
            continue;
        }
        const [r, g, b] = CLUSTER_RGB[label % CLUSTER_RGB.length];
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 150;
    }
    ctx.putImageData(image, 0, 0);
}

/** Draws the points: clustered ones as filled coloured dots, noise as small grey ✕ marks. */
export function drawDbscanPoints(
    ctx: CanvasRenderingContext2D,
    inputs: number[][],
    labels: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (let i = 0; i < inputs.length; i++) {
        const px = ((inputs[i][0] - domain.xMin) / (domain.xMax - domain.xMin)) * width;
        const py = (1 - (inputs[i][1] - domain.yMin) / (domain.yMax - domain.yMin)) * height;

        if (labels[i] < 0) {
            // Noise: a faint grey cross, clearly not part of any group.
            const r = 3;
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.7)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(px - r, py - r);
            ctx.lineTo(px + r, py + r);
            ctx.moveTo(px + r, py - r);
            ctx.lineTo(px - r, py + r);
            ctx.stroke();
            continue;
        }

        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = clusterHex(labels[i]);
        ctx.fill();
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();
    }
}
