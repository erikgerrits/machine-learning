import type { Domain } from '../ml/datasets';

// Renders the regression view: a scatter of (x, y) points, faint vertical "residual" lines from
// each point to the current fit (the squared lengths are what training shrinks), and the fitted
// line itself. Redrawn every frame, so the line visibly rotates into place as the model learns.
export function drawRegression(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    domain: Domain,
    inputs: number[][],
    targets: number[][],
    /** The model's predicted y at each input (same order), for the residual lines. */
    predicted: number[],
    /** Two endpoints of the fitted line in data coordinates: [[x0, y0], [x1, y1]]. */
    line: [[number, number], [number, number]],
    /** Optional axis captions (e.g. "temperature →" / "items sold →"). */
    xLabel?: string,
    yLabel?: string,
): void {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const sx = (x: number) => ((x - domain.xMin) / (domain.xMax - domain.xMin)) * width;
    const sy = (y: number) => (1 - (y - domain.yMin) / (domain.yMax - domain.yMin)) * height;

    // Zero axes for orientation.
    ctx.strokeStyle = 'rgba(148, 164, 189, 0.18)';
    ctx.lineWidth = 1;
    if (domain.yMin < 0 && domain.yMax > 0) {
        const y0 = sy(0);
        ctx.beginPath();
        ctx.moveTo(0, y0);
        ctx.lineTo(width, y0);
        ctx.stroke();
    }
    if (domain.xMin < 0 && domain.xMax > 0) {
        const x0 = sx(0);
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x0, height);
        ctx.stroke();
    }

    // Residuals: the gap each point contributes to the loss.
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.28)';
    ctx.lineWidth = 1;
    for (let i = 0; i < inputs.length; i++) {
        const x = sx(inputs[i][0]);
        ctx.beginPath();
        ctx.moveTo(x, sy(targets[i][0]));
        ctx.lineTo(x, sy(predicted[i]));
        ctx.stroke();
    }

    // Data points.
    for (let i = 0; i < inputs.length; i++) {
        ctx.beginPath();
        ctx.arc(sx(inputs[i][0]), sy(targets[i][0]), 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.8)';
        ctx.stroke();
    }

    // The fitted line.
    ctx.beginPath();
    ctx.moveTo(sx(line[0][0]), sy(line[0][1]));
    ctx.lineTo(sx(line[1][0]), sy(line[1][1]));
    ctx.strokeStyle = '#fb923c';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Axis captions: what the axes mean, without committing to exact units.
    if (xLabel || yLabel) {
        ctx.fillStyle = 'rgba(148, 164, 189, 0.55)';
        ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
        if (xLabel) {
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(xLabel, width / 2, height - 6);
        }
        if (yLabel) {
            ctx.save();
            ctx.translate(14, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(yLabel, 0, 0);
            ctx.restore();
        }
    }
}
