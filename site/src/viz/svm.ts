import type { Domain } from '../ml/datasets';

// Class hues, matching the rest of the site: sky-blue = class 0 (no comp), amber = class 1 (comp).
const C0: [number, number, number] = [56, 189, 248];
const C1: [number, number, number] = [251, 146, 60];

/**
 * Paints the SVM decision scores into a size×size offscreen canvas. Unlike the logistic heatmap
 * (which fades toward a 50% seam), this one renders the **margin** directly: each cell is coloured
 * by the sign of the score `f` and its opacity ramps with `|f|`, clamped at the margin edge `|f| = 1`.
 * Cells inside the street (`|f| < 1`) stay translucent over the dark backdrop, so the empty margin
 * shows up as a dark band — and the wider the SVM's margin, the wider that band.
 */
export function paintMargins(offscreen: HTMLCanvasElement, scores: number[], size: number): void {
    if (offscreen.width !== size || offscreen.height !== size) {
        offscreen.width = size;
        offscreen.height = size;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(size, size);
    for (let k = 0; k < scores.length; k++) {
        const f = scores[k];
        const [r, g, b] = f >= 0 ? C1 : C0;
        const magnitude = Math.min(1, Math.abs(f)); // 0 at the boundary → 1 at/beyond the margin edge
        const alpha = Math.round(30 + 205 * magnitude);
        const o = k * 4;
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = alpha;
    }
    ctx.putImageData(image, 0, 0);
}

/**
 * Rings the support vectors — the points the boundary actually balances on. Drawn as a bright
 * hollow circle so the underlying class-coloured dot stays visible inside.
 */
export function drawSupportVectors(
    ctx: CanvasRenderingContext2D,
    points: number[][],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (const [x, y] of points) {
        const px = ((x - domain.xMin) / (domain.xMax - domain.xMin)) * width;
        const py = (1 - (y - domain.yMin) / (domain.yMax - domain.yMin)) * height;

        ctx.beginPath();
        ctx.arc(px, py, 8, 0, 2 * Math.PI);
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#f8fafc';
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.9)';
        ctx.stroke();
    }
}
