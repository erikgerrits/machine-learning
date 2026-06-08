// Renders a predicted-vs-actual scatter: one dot per day at (actual sales, predicted sales).
// A perfect model lands every dot on the dashed 45° diagonal; the vertical gap to that line is
// the prediction error. This view works for ANY number of input features — which is exactly why
// it's the right picture for multi-feature regression, where the "fit" is a hyperplane you can't
// draw directly.
export function drawPredictedVsActual(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    min: number,
    max: number,
    actual: number[],
    predicted: number[],
): void {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const pad = 8;
    const span = max - min || 1;
    const sx = (v: number) => pad + ((v - min) / span) * (width - 2 * pad);
    const sy = (v: number) => height - pad - ((v - min) / span) * (height - 2 * pad);

    // The "perfect prediction" diagonal: predicted === actual.
    ctx.strokeStyle = 'rgba(148, 164, 189, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(sx(min), sy(min));
    ctx.lineTo(sx(max), sy(max));
    ctx.stroke();
    ctx.setLineDash([]);

    // Residuals: the vertical gap from each dot down/up to the diagonal (predicted − actual).
    ctx.strokeStyle = 'rgba(251, 146, 60, 0.30)';
    ctx.lineWidth = 1;
    for (let i = 0; i < actual.length; i++) {
        ctx.beginPath();
        ctx.moveTo(sx(actual[i]), sy(predicted[i]));
        ctx.lineTo(sx(actual[i]), sy(actual[i]));
        ctx.stroke();
    }

    // The dots.
    for (let i = 0; i < actual.length; i++) {
        ctx.beginPath();
        ctx.arc(sx(actual[i]), sy(predicted[i]), 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.8)';
        ctx.stroke();
    }

    // Axis captions.
    ctx.fillStyle = 'rgba(148, 164, 189, 0.6)';
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('actual sales →', width / 2, height - 4);
    ctx.save();
    ctx.translate(13, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('predicted sales →', 0, 0);
    ctx.restore();
}
