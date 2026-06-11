import { fitCanvas } from './canvas';
import type { CellType } from '../ml/gridworldDatasets';

// Draws a grid world the way reinforcement-learning textbooks do: each open cell tinted by its learned
// value V(s) (a heatmap that fills in from the goal outward as value propagates back), an arrow per
// cell for the greedy action (the policy), and a dot for the runner's current position. Goal, spill
// and wall cells get their own fixed colours.

const ARROWS = ['↑', '→', '↓', '←'];

export interface GridworldView {
    rows: number;
    cols: number;
    cellTypes: CellType[];
    values: number[];     // V(s) per state (max-Q); only meaningful for open cells
    policy: number[];     // greedy action per state
    agent: number;        // current state of the runner (-1 to hide)
    showPolicy: boolean;
    showValues: boolean;
}

export function drawGridworld(canvas: HTMLCanvasElement, view: GridworldView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const pad = 10;
    const cell = Math.floor(Math.min((width - pad * 2) / view.cols, (height - pad * 2) / view.rows));
    const gridW = cell * view.cols;
    const gridH = cell * view.rows;
    const originX = (width - gridW) / 2;
    const originY = (height - gridH) / 2;

    // Value range across open cells, for normalising the heatmap.
    let lo = Infinity;
    let hi = -Infinity;
    for (let s = 0; s < view.values.length; s++) {
        if (view.cellTypes[s] === 'empty' || view.cellTypes[s] === 'start') {
            lo = Math.min(lo, view.values[s]);
            hi = Math.max(hi, view.values[s]);
        }
    }
    const span = hi - lo > 1e-6 ? hi - lo : 1;

    for (let s = 0; s < view.cellTypes.length; s++) {
        const row = Math.floor(s / view.cols);
        const col = s % view.cols;
        const x = originX + col * cell;
        const y = originY + row * cell;
        const type = view.cellTypes[s];

        // Cell fill.
        if (type === 'wall') {
            ctx.fillStyle = '#1e293b';
        } else if (type === 'goal') {
            ctx.fillStyle = '#34d399';
        } else if (type === 'hazard') {
            ctx.fillStyle = '#f43f5e';
        } else if (view.showValues) {
            ctx.fillStyle = valueColor((view.values[s] - lo) / span);
        } else {
            ctx.fillStyle = '#0f172a';
        }
        ctx.fillRect(x, y, cell, cell);

        // Grid line.
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cell, cell);

        const cx = x + cell / 2;
        const cy = y + cell / 2;

        if (type === 'goal' || type === 'hazard') {
            ctx.fillStyle = '#0b1120';
            ctx.font = `${Math.round(cell * 0.4)}px system-ui, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(type === 'goal' ? '★' : '✕', cx, cy);
        } else if (type !== 'wall' && view.showPolicy) {
            // Greedy-action arrow.
            ctx.fillStyle = 'rgba(226, 232, 240, 0.85)';
            ctx.font = `${Math.round(cell * 0.42)}px system-ui, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(ARROWS[view.policy[s]], cx, cy);
        }

        if (type === 'start') {
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.strokeRect(x + 2, y + 2, cell - 4, cell - 4);
        }
    }

    // The runner.
    if (view.agent >= 0) {
        const row = Math.floor(view.agent / view.cols);
        const col = view.agent % view.cols;
        ctx.fillStyle = '#facc15';
        ctx.beginPath();
        ctx.arc(originX + col * cell + cell / 2, originY + row * cell + cell / 2, cell * 0.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#0b1120';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// Low value → deep indigo, high value → warm amber: the classic "cold far from the goal, hot near it".
function valueColor(t: number): string {
    const u = Math.max(0, Math.min(1, t));
    const lo: [number, number, number] = [30, 41, 89];   // indigo-ish
    const hi: [number, number, number] = [251, 191, 36];  // amber-400
    const r = Math.round(lo[0] + (hi[0] - lo[0]) * u);
    const g = Math.round(lo[1] + (hi[1] - lo[1]) * u);
    const b = Math.round(lo[2] + (hi[2] - lo[2]) * u);
    return `rgb(${r}, ${g}, ${b})`;
}
