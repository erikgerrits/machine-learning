import { fitCanvas } from './canvas';
import { CLUSTER_HEX } from './clusters';

// One panel per customer type, side by side. Within a panel, one vertical bar per offer (arm), its
// height the bandit's *predicted* sell-rate for that offer to that customer. A white tick marks the
// offer's true hidden rate, and a ★ crowns the offer the bandit would currently pick for that type —
// so you can watch a genuinely *different* winner emerge in each column. The customer being served on
// the latest turn has their panel lit up.

export interface ContextualView {
    arms: string[];           // offer names (shared across panels; coloured consistently)
    types: string[];          // customer type names (one panel each)
    predicted: number[][];    // predicted[type][arm] — bandit's estimated sell-rate, clamped to [0, 1]
    rates: number[][];        // true rates[type][arm]
    serving: number;          // index of the customer type served this turn (-1 before the first)
    revealRates: boolean;
}

export function drawContextualBandit(canvas: HTMLCanvasElement, view: ContextualView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const nTypes = view.types.length;
    const nArms = view.arms.length;
    const outerPad = 12;
    const panelGap = 10;
    const panelW = (width - outerPad * 2 - panelGap * (nTypes - 1)) / nTypes;

    const titleH = 26;
    const plotTop = outerPad + titleH;
    const plotBottom = height - outerPad - 4;
    const plotH = plotBottom - plotTop;

    ctx.textBaseline = 'middle';

    for (let t = 0; t < nTypes; t++) {
        const panelX = outerPad + t * (panelW + panelGap);

        // Light the panel for the customer currently being served.
        if (t === view.serving) {
            ctx.fillStyle = 'rgba(250, 204, 21, 0.08)';
            ctx.fillRect(panelX - 4, outerPad - 2, panelW + 8, height - outerPad * 2 + 4);
            ctx.strokeStyle = 'rgba(250, 204, 21, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(panelX - 4, outerPad - 2, panelW + 8, height - outerPad * 2 + 4);
        }

        // Panel title (customer type).
        ctx.fillStyle = t === view.serving ? '#facc15' : '#e2e8f0';
        ctx.font = '600 12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(view.types[t], panelX + panelW / 2, outerPad + titleH / 2);

        const chosen = argmax(view.predicted[t]);
        const barSlot = panelW / nArms;
        const barW = Math.min(64, barSlot * 0.6);

        for (let a = 0; a < nArms; a++) {
            const cx = panelX + barSlot * (a + 0.5);
            const color = CLUSTER_HEX[a % CLUSTER_HEX.length];

            // Track.
            ctx.fillStyle = 'rgba(148, 163, 184, 0.10)';
            ctx.fillRect(cx - barW / 2, plotTop, barW, plotH);

            // Predicted-reward bar.
            const h = clamp01(view.predicted[t][a]) * plotH;
            ctx.fillStyle = color;
            ctx.fillRect(cx - barW / 2, plotBottom - h, barW, h);

            // True-rate tick.
            if (view.revealRates) {
                const ty = plotBottom - clamp01(view.rates[t][a]) * plotH;
                ctx.strokeStyle = '#f8fafc';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cx - barW / 2 - 3, ty);
                ctx.lineTo(cx + barW / 2 + 3, ty);
                ctx.stroke();
            }

            // ★ over the offer the bandit would pick for this customer.
            if (a === chosen) {
                ctx.fillStyle = '#facc15';
                ctx.font = '13px system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('★', cx, plotBottom - h - 10);
            }
        }
    }
}

function argmax(values: number[]): number {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
}

function clamp01(v: number): number {
    return Math.max(0, Math.min(1, v));
}
