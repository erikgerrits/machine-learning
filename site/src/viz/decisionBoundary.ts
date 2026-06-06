import { Matrix } from 'machine-learning';
import type { Domain } from '../ml/datasets';
import { boundaryRGBA, classColor } from './colors';

/** Grid resolution for the decision-boundary heatmap (GRID × GRID cells). */
export const GRID = 100;

export interface Grid {
    /** A (GRID*GRID) × 2 Matrix of feature-space points, row-major (gy outer, gx inner). */
    matrix: Matrix;
    size: number;
}

/**
 * Builds the sampling grid once per domain. Each cell's center is mapped back into feature
 * space; row 0 is the top of the canvas (yMax). Reused every frame — `predict()` is
 * non-destructive on its input, so the same Matrix can be fed to the network repeatedly.
 */
export function makeGrid(domain: Domain, size = GRID): Grid {
    const points: number[][] = [];
    for (let gy = 0; gy < size; gy++) {
        for (let gx = 0; gx < size; gx++) {
            const x = domain.xMin + ((gx + 0.5) / size) * (domain.xMax - domain.xMin);
            const y = domain.yMax - ((gy + 0.5) / size) * (domain.yMax - domain.yMin);
            points.push([x, y]);
        }
    }
    return { matrix: new Matrix(points), size };
}

/** Paints the per-cell model outputs into a GRID×GRID offscreen canvas via ImageData. */
export function paintBoundary(offscreen: HTMLCanvasElement, values: number[], size: number): void {
    if (offscreen.width !== size || offscreen.height !== size) {
        offscreen.width = size;
        offscreen.height = size;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(size, size);
    for (let k = 0; k < values.length; k++) {
        const [r, g, b, a] = boundaryRGBA(values[k]);
        const o = k * 4;
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = a;
    }
    ctx.putImageData(image, 0, 0);
}

/** Draws the training points on top of the heatmap, colored by their true label. */
export function drawPoints(
    ctx: CanvasRenderingContext2D,
    inputs: number[][],
    targets: number[][],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (let i = 0; i < inputs.length; i++) {
        const [x, y] = inputs[i];
        const px = ((x - domain.xMin) / (domain.xMax - domain.xMin)) * width;
        const py = (1 - (y - domain.yMin) / (domain.yMax - domain.yMin)) * height;

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fillStyle = classColor(targets[i][0]);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();
    }
}
