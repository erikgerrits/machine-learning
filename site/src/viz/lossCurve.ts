// A minimal canvas line chart of loss vs. epoch — no charting dependency. The Y axis auto-
// scales to the largest loss seen so far, so the descent stays in frame as it falls.

export function drawLossCurve(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    history: number[],
): void {
    ctx.clearRect(0, 0, width, height);

    // Baseline.
    ctx.strokeStyle = 'rgba(148, 164, 189, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 1);
    ctx.lineTo(width, height - 1);
    ctx.stroke();

    if (history.length < 2) return;

    const maxLoss = Math.max(...history, 1e-6);
    const stepX = width / (history.length - 1);
    const toY = (loss: number) => height - (loss / maxLoss) * (height - 8) - 4;

    // Filled area under the curve.
    ctx.beginPath();
    ctx.moveTo(0, height);
    history.forEach((loss, i) => ctx.lineTo(i * stepX, toY(loss)));
    ctx.lineTo((history.length - 1) * stepX, height);
    ctx.closePath();
    ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
    ctx.fill();

    // The curve itself.
    ctx.beginPath();
    history.forEach((loss, i) => {
        const x = i * stepX;
        const y = toY(loss);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();
}
