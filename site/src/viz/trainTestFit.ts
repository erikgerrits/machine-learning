// Predicted-vs-actual scatter for the overfitting chapter, drawn with TWO sets of days: the
// days the model trained on ("studied", blue) and held-out days it has never seen ("unseen",
// orange). A model that has truly learned puts both on the dashed diagonal. A model that has
// merely memorised glues the studied days to the diagonal while the unseen days scatter — and
// that visible gap is overfitting.
export function drawTrainTestFit(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    min: number,
    max: number,
    trainActual: number[],
    trainPredicted: number[],
    testActual: number[],
    testPredicted: number[],
): void {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const pad = 8;
    const span = max - min || 1;
    const sx = (v: number) => pad + ((v - min) / span) * (width - 2 * pad);
    const sy = (v: number) => height - pad - ((v - min) / span) * (height - 2 * pad);

    // The "perfect prediction" diagonal.
    ctx.strokeStyle = 'rgba(148, 164, 189, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(sx(min), sy(min));
    ctx.lineTo(sx(max), sy(max));
    ctx.stroke();
    ctx.setLineDash([]);

    // Test residuals — the cost of every unseen day the model got wrong.
    ctx.strokeStyle = 'rgba(251, 146, 60, 0.28)';
    ctx.lineWidth = 1;
    for (let i = 0; i < testActual.length; i++) {
        ctx.beginPath();
        ctx.moveTo(sx(testActual[i]), sy(testPredicted[i]));
        ctx.lineTo(sx(testActual[i]), sy(testActual[i]));
        ctx.stroke();
    }

    // Studied days (blue) — small, since they're expected to sit on the line.
    ctx.fillStyle = 'rgba(56, 189, 248, 0.85)';
    for (let i = 0; i < trainActual.length; i++) {
        ctx.beginPath();
        ctx.arc(sx(trainActual[i]), sy(trainPredicted[i]), 2.6, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Unseen days (orange) — the ones that matter.
    for (let i = 0; i < testActual.length; i++) {
        ctx.beginPath();
        ctx.arc(sx(testActual[i]), sy(testPredicted[i]), 3.4, 0, 2 * Math.PI);
        ctx.fillStyle = '#fb923c';
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
