import { fitCanvas } from './canvas';
import { CLUSTER_HEX } from './clusters';

// One horizontal bar per arm, drawn on a [0, 1] sell-rate track. Each bar shows:
//   • a solid fill   — the arm's *estimated* value (its running average reward so far)
//   • a faint tip    — the exploration headroom (for UCB, the optimism bonus; how far above its
//                       estimate the strategy is still willing to gamble). Shrinks as evidence grows.
//   • a white tick   — the arm's *true* hidden rate, so you can see the estimate close in on it
//   • a pull count   — how many times the bandit has featured this special
// The arm chosen on the latest turn is ringed, and the current front-runner is marked ★.

export interface BanditView {
    names: string[];
    rates: number[];      // true hidden rates (revealed in the viz so you can grade the bandit)
    estimates: number[];  // bandit.getValues()
    uppers: number[];     // estimate + exploration headroom, clamped to [0, 1] (== estimate for ε-greedy)
    counts: number[];     // bandit.getCounts()
    selected: number;     // arm chosen this turn (-1 before the first turn)
    revealRates: boolean; // show the true-rate ticks?
}

export function drawBandit(canvas: HTMLCanvasElement, view: BanditView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const n = view.names.length;
    const padX = 16;
    const labelW = 132;
    const countW = 64;
    const trackX0 = padX + labelW;
    const trackX1 = width - padX - countW;
    const trackW = trackX1 - trackX0;

    const topPad = 14;
    const botPad = 14;
    const rowH = (height - topPad - botPad) / n;
    const barH = Math.min(26, rowH * 0.5);

    const leader = argmax(view.estimates);
    const toX = (v: number) => trackX0 + Math.max(0, Math.min(1, v)) * trackW;

    ctx.textBaseline = 'middle';

    for (let i = 0; i < n; i++) {
        const cy = topPad + rowH * i + rowH / 2;
        const barY = cy - barH / 2;
        const color = CLUSTER_HEX[i % CLUSTER_HEX.length];

        // Track background.
        ctx.fillStyle = 'rgba(148, 163, 184, 0.12)';
        roundRect(ctx, trackX0, barY, trackW, barH, 5);
        ctx.fill();

        // Exploration headroom (estimate → upper): the faint tip the strategy is still chasing.
        if (view.uppers[i] > view.estimates[i] + 1e-4) {
            ctx.fillStyle = hexToRgba(color, 0.28);
            roundRect(ctx, toX(view.estimates[i]), barY, toX(view.uppers[i]) - toX(view.estimates[i]), barH, 0);
            ctx.fill();
        }

        // Solid estimate fill.
        ctx.fillStyle = color;
        const estW = toX(view.estimates[i]) - trackX0;
        if (estW > 0) {
            roundRect(ctx, trackX0, barY, estW, barH, 5);
            ctx.fill();
        }

        // True-rate tick.
        if (view.revealRates) {
            const tx = toX(view.rates[i]);
            ctx.strokeStyle = '#f8fafc';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(tx, barY - 4);
            ctx.lineTo(tx, barY + barH + 4);
            ctx.stroke();
        }

        // Ring the arm chosen this turn.
        if (i === view.selected) {
            ctx.strokeStyle = '#facc15';
            ctx.lineWidth = 2;
            roundRect(ctx, trackX0 - 3, barY - 3, trackW + 6, barH + 6, 7);
            ctx.stroke();
        }

        // Name (★ marks the current front-runner).
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '13px system-ui, sans-serif';
        ctx.textAlign = 'left';
        const star = i === leader && view.counts[i] > 0 ? '★ ' : '';
        ctx.fillText(star + view.names[i], padX, cy);

        // Estimate value over the bar, and pull count on the right.
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px ui-monospace, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(view.counts[i] > 0 ? view.estimates[i].toFixed(2) : '—', trackX1 - 6, barY - 8 < topPad ? cy : barY - 9);
        ctx.fillText(`${view.counts[i]}×`, width - padX, cy);
    }
}

function argmax(values: number[]): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
}

function hexToRgba(hex: string, alpha: number): string {
    const v = parseInt(hex.slice(1), 16);
    return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${alpha})`;
}
