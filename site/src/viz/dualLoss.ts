// Two loss curves on one chart: training error (blue) and held-out test error (orange), sharing
// a y-scale so the gap between them is honest. When a model overfits, the training curve keeps
// sliding down while the test curve bottoms out and turns back up — the gap opening before your
// eyes is the picture of overfitting.
export function drawDualLoss(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    train: number[],
    test: number[],
): void {
    ctx.clearRect(0, 0, width, height);

    // Baseline.
    ctx.strokeStyle = 'rgba(148, 164, 189, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 1);
    ctx.lineTo(width, height - 1);
    ctx.stroke();

    const length = Math.min(train.length, test.length);
    if (length < 2) return;

    const maxLoss = Math.max(...train.slice(0, length), ...test.slice(0, length), 1e-6);
    const stepX = width / (length - 1);
    const toY = (loss: number) => height - (loss / maxLoss) * (height - 8) - 4;

    const drawSeries = (series: number[], color: string) => {
        ctx.beginPath();
        for (let i = 0; i < length; i++) {
            const x = i * stepX;
            const y = toY(series[i]);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
    };

    drawSeries(train, '#38bdf8'); // studied days
    drawSeries(test, '#fb923c'); // unseen days
}
