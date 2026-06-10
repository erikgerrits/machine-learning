import type { Domain } from '../ml/datasets';

const NORMAL: [number, number, number] = [56, 189, 248]; // sky
const ANOMALY: [number, number, number] = [248, 113, 113]; // red

/**
 * Paints the anomaly map: each cell is coloured by its Mahalanobis score relative to the threshold.
 * Inside the threshold (`ratio ≤ 1`) the cell is cool blue and fades out toward the boundary — the
 * fitted "normal" region, an ellipse following the data's shape. Past it (`ratio > 1`) the cell
 * turns red and deepens with distance, so the threshold ring where they meet reads as the decision
 * boundary between normal and anomalous.
 */
export function paintAnomalyRegion(
    offscreen: HTMLCanvasElement,
    scores: number[],
    threshold: number,
    size: number,
): void {
    if (offscreen.width !== size || offscreen.height !== size) {
        offscreen.width = size;
        offscreen.height = size;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(size, size);
    for (let k = 0; k < scores.length; k++) {
        const ratio = scores[k] / threshold;
        const o = k * 4;
        if (ratio <= 1) {
            image.data[o] = NORMAL[0];
            image.data[o + 1] = NORMAL[1];
            image.data[o + 2] = NORMAL[2];
            image.data[o + 3] = Math.round(25 + 60 * (1 - ratio)); // deepest at the centre
        } else {
            image.data[o] = ANOMALY[0];
            image.data[o + 1] = ANOMALY[1];
            image.data[o + 2] = ANOMALY[2];
            image.data[o + 3] = Math.round(35 + Math.min(165, 110 * (ratio - 1)));
        }
    }
    ctx.putImageData(image, 0, 0);
}

/** Draws the points: normal ones as sky dots, flagged anomalies as red dots with a white ring. */
export function drawAnomalyPoints(
    ctx: CanvasRenderingContext2D,
    inputs: number[][],
    flags: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (let i = 0; i < inputs.length; i++) {
        const px = ((inputs[i][0] - domain.xMin) / (domain.xMax - domain.xMin)) * width;
        const py = (1 - (inputs[i][1] - domain.yMin) / (domain.yMax - domain.yMin)) * height;
        const anomaly = flags[i] === 1;

        ctx.beginPath();
        ctx.arc(px, py, anomaly ? 4.5 : 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = anomaly ? '#f87171' : '#38bdf8';
        ctx.fill();
        ctx.lineWidth = anomaly ? 2 : 1.25;
        ctx.strokeStyle = anomaly ? '#f8fafc' : 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();
    }
}
