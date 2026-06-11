import { fitCanvas } from './canvas';

// A 1-D regression plot built to make *uncertainty* the star: a shaded 95% credible band (mean ± 2σ)
// that pinches onto the data and flares where there's none, a few posterior sample curves threading
// through it, the posterior-mean fit, the revealed observations, and the hidden truth for reference.

export interface BayesianView {
    domain: [number, number];
    yRange: [number, number];
    gridX: number[];
    meanY: number[];
    stdY: number[];
    samples: number[][];      // posterior function samples over gridX
    trueY: number[];          // hidden truth over gridX
    points: { x: number; y: number }[];
    showSamples: boolean;
    showTruth: boolean;
}

export function drawBayesian(canvas: HTMLCanvasElement, view: BayesianView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const padL = 8, padR = 8, padT = 10, padB = 10;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const [x0, x1] = view.domain;
    const [y0, y1] = view.yRange;
    const px = (x: number) => padL + ((x - x0) / (x1 - x0)) * plotW;
    const py = (y: number) => padT + (1 - (y - y0) / (y1 - y0)) * plotH;

    // Zero line.
    ctx.strokeStyle = 'rgba(148,163,184,0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, py(0));
    ctx.lineTo(width - padR, py(0));
    ctx.stroke();

    const n = view.gridX.length;

    // 95% credible band: mean ± 2σ.
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = px(view.gridX[i]); const y = py(view.meanY[i] + 2 * view.stdY[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(px(view.gridX[i]), py(view.meanY[i] - 2 * view.stdY[i]));
    ctx.closePath();
    ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
    ctx.fill();

    // Posterior sample curves.
    if (view.showSamples) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
        ctx.lineWidth = 1;
        for (const sample of view.samples) {
            ctx.beginPath();
            for (let i = 0; i < n; i++) { const x = px(view.gridX[i]), y = py(sample[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
            ctx.stroke();
        }
    }

    // Hidden truth (dashed).
    if (view.showTruth) {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        for (let i = 0; i < n; i++) { const x = px(view.gridX[i]), y = py(view.trueY[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Posterior mean fit.
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = px(view.gridX[i]), y = py(view.meanY[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();

    // Observations.
    ctx.fillStyle = '#fb923c';
    for (const point of view.points) {
        ctx.beginPath();
        ctx.arc(px(point.x), py(point.y), 3.5, 0, Math.PI * 2);
        ctx.fill();
    }
}
