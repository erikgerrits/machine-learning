import type { Domain } from '../ml/datasets';

/**
 * Rings the misclassified points so the perceptron's mistakes are obvious — on XOR these never go
 * away, which is the whole point of the interlude.
 */
export function drawMisclassified(
    ctx: CanvasRenderingContext2D,
    inputs: number[][],
    labels: number[],
    predictions: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (let i = 0; i < inputs.length; i++) {
        if (predictions[i] === labels[i]) {
            continue;
        }
        const px = ((inputs[i][0] - domain.xMin) / (domain.xMax - domain.xMin)) * width;
        const py = (1 - (inputs[i][1] - domain.yMin) / (domain.yMax - domain.yMin)) * height;

        ctx.beginPath();
        ctx.arc(px, py, 8, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#f87171'; // red ring = "got this one wrong"
        ctx.stroke();
    }
}
