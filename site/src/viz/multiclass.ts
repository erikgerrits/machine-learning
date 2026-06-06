import type { Domain } from '../ml/datasets';

// Three-class palette: sky, amber, violet.
export const CLASS_RGB: [number, number, number][] = [
    [56, 189, 248],
    [251, 146, 60],
    [167, 139, 250],
];
export const CLASS_HEX = ['#38bdf8', '#fb923c', '#a78bfa'];

export function classHex(index: number): string {
    return CLASS_HEX[index % CLASS_HEX.length];
}

/**
 * Paints a multiclass decision map into a size×size offscreen canvas: each cell is colored by
 * the winning class (argmax of the per-class scores), with opacity driven by the margin between
 * the top two classes — so contested borders read as soft seams.
 */
export function paintMulticlass(offscreen: HTMLCanvasElement, values: number[][], size: number): void {
    if (offscreen.width !== size || offscreen.height !== size) {
        offscreen.width = size;
        offscreen.height = size;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(size, size);
    for (let k = 0; k < values.length; k++) {
        const row = values[k];
        let top = -Infinity;
        let second = -Infinity;
        let argmax = 0;
        for (let c = 0; c < row.length; c++) {
            if (row[c] > top) {
                second = top;
                top = row[c];
                argmax = c;
            } else if (row[c] > second) {
                second = row[c];
            }
        }
        const [r, g, b] = CLASS_RGB[argmax % CLASS_RGB.length];
        const margin = Math.min(1, Math.max(0, top - second));
        const a = Math.round(120 + 115 * Math.min(1, margin * 3));
        const o = k * 4;
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = a;
    }
    ctx.putImageData(image, 0, 0);
}

/** Draws training points colored by their (integer) class index. */
export function drawClassPoints(
    ctx: CanvasRenderingContext2D,
    inputs: number[][],
    classIndex: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (let i = 0; i < inputs.length; i++) {
        const px = ((inputs[i][0] - domain.xMin) / (domain.xMax - domain.xMin)) * width;
        const py = (1 - (inputs[i][1] - domain.yMin) / (domain.yMax - domain.yMin)) * height;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, 2 * Math.PI);
        ctx.fillStyle = classHex(classIndex[i]);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();
    }
}
