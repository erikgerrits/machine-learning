import { fitCanvas } from './canvas';

// Renders the continuous floor: a fine value heatmap the network paints over the whole arena (the
// star of the chapter — a *function*, not a table, so it's defined at every point), a coarser field of
// greedy-policy arrows, the table (goal disc), and the runner with a fading trail. The whole point is
// that the heatmap is smooth and covers everywhere, including spots the runner never actually stepped.

const ARROWS = ['↑', '→', '↓', '←'];

export interface DeepRlView {
    goal: [number, number];
    radius: number;
    heatRes: number;        // heatmap grid resolution (heatRes × heatRes)
    heatValues: number[];   // V at each heat cell, row-major with y increasing upward
    arrowRes: number;       // policy grid resolution
    arrowActions: number[]; // greedy action at each arrow point, row-major with y increasing upward
    agent: number[] | null; // runner position [x, y]
    trail: number[][];      // recent positions
    showValues: boolean;
    showPolicy: boolean;
}

export function drawDeepRl(canvas: HTMLCanvasElement, view: DeepRlView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const pad = 8;
    const size = Math.min(width, height) - pad * 2;
    const originX = (width - size) / 2;
    const originY = (height - size) / 2;

    // Arena coords (x right, y UP) → canvas pixels (y flipped so up is up).
    const px = (x: number) => originX + x * size;
    const py = (y: number) => originY + (1 - y) * size;

    // Value heatmap.
    if (view.showValues) {
        let lo = Infinity;
        let hi = -Infinity;
        for (const v of view.heatValues) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        const span = hi - lo > 1e-6 ? hi - lo : 1;
        const cell = size / view.heatRes;
        for (let r = 0; r < view.heatRes; r++) {
            for (let c = 0; c < view.heatRes; c++) {
                const v = view.heatValues[r * view.heatRes + c];
                ctx.fillStyle = valueColor((v - lo) / span);
                // Row 0 is the bottom (y = 0); flip to canvas rows.
                ctx.fillRect(originX + c * cell, originY + (view.heatRes - 1 - r) * cell, cell + 1, cell + 1);
            }
        }
    }

    // Policy arrows.
    if (view.showPolicy) {
        ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
        ctx.font = `${Math.round(size / view.arrowRes * 0.5)}px system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (let r = 0; r < view.arrowRes; r++) {
            for (let c = 0; c < view.arrowRes; c++) {
                const x = (c + 0.5) / view.arrowRes;
                const y = (r + 0.5) / view.arrowRes;
                ctx.fillText(ARROWS[view.arrowActions[r * view.arrowRes + c]], px(x), py(y));
            }
        }
    }

    // The table (goal).
    ctx.beginPath();
    ctx.arc(px(view.goal[0]), py(view.goal[1]), view.radius * size, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(52, 211, 153, 0.35)';
    ctx.fill();
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#022c22';
    ctx.font = `${Math.round(view.radius * size * 0.9)}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('★', px(view.goal[0]), py(view.goal[1]));

    // Trail.
    if (view.trail.length > 1) {
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.45)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px(view.trail[0][0]), py(view.trail[0][1]));
        for (let i = 1; i < view.trail.length; i++) ctx.lineTo(px(view.trail[i][0]), py(view.trail[i][1]));
        ctx.stroke();
    }

    // The runner.
    if (view.agent) {
        ctx.beginPath();
        ctx.arc(px(view.agent[0]), py(view.agent[1]), size * 0.018, 0, Math.PI * 2);
        ctx.fillStyle = '#facc15';
        ctx.fill();
        ctx.strokeStyle = '#0b1120';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// Low value → deep indigo, high → warm amber: cold far from the table, hot near it.
function valueColor(t: number): string {
    const u = Math.max(0, Math.min(1, t));
    const lo: [number, number, number] = [30, 41, 89];
    const hi: [number, number, number] = [251, 191, 36];
    return `rgb(${Math.round(lo[0] + (hi[0] - lo[0]) * u)}, ${Math.round(lo[1] + (hi[1] - lo[1]) * u)}, ${Math.round(lo[2] + (hi[2] - lo[2]) * u)})`;
}
