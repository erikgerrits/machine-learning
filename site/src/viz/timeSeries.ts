// Line-chart viz for the time-series playground: history, the in-sample fit, and the forecast.

const HISTORY = '#38bdf8'; // sky — the actual past
const FORECAST = '#fb923c'; // amber — the forecast ahead
const FIT = 'rgba(251, 146, 60, 0.5)'; // faint amber — the fitted line over history

/**
 * Draws the series as a line chart: the actual `history` in sky, the model's one-step-ahead `fitted`
 * line faintly over it, and the `forecast` continuing past a dashed "now" divider in amber. The y-axis
 * auto-scales to all three so the forecast's seasonal swings stay on-canvas.
 */
export function drawForecast(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    history: number[],
    fitted: number[],
    forecast: number[],
): void {
    const padX = 12;
    const padY = 14;
    const plotW = width - 2 * padX;
    const plotH = height - 2 * padY;

    const total = history.length + forecast.length;
    if (total < 2) return;

    const all = history.concat(fitted, forecast);
    let min = Math.min(...all);
    let max = Math.max(...all);
    if (max - min < 1e-9) {
        max = min + 1;
    }
    const margin = (max - min) * 0.08;
    min -= margin;
    max += margin;

    const x = (i: number) => padX + (i / (total - 1)) * plotW;
    const y = (v: number) => padY + (1 - (v - min) / (max - min)) * plotH;

    const line = (values: number[], startIndex: number, color: string, dashed: boolean, lineWidth: number) => {
        if (values.length === 0) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashed ? [5, 4] : []);
        ctx.beginPath();
        values.forEach((v, k) => {
            const px = x(startIndex + k);
            const py = y(v);
            if (k === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.setLineDash([]);
    };

    // The "now" divider between history and forecast.
    const dividerX = x(history.length - 0.5);
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(dividerX, padY);
    ctx.lineTo(dividerX, height - padY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Fitted line over the history, then the actual history on top.
    line(fitted, 0, FIT, false, 1.5);
    line(history, 0, HISTORY, false, 2);

    // Forecast, joined to the last actual point so the line is continuous.
    const lastActual = history[history.length - 1];
    line([lastActual, ...forecast], history.length - 1, FORECAST, true, 2);

    // Dots on the forecast so individual future days read clearly.
    ctx.fillStyle = FORECAST;
    forecast.forEach((v, k) => {
        ctx.beginPath();
        ctx.arc(x(history.length + k), y(v), 2.5, 0, 2 * Math.PI);
        ctx.fill();
    });
}
