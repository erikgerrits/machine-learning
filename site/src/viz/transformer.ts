// Attention viz for the transformer playground: the [CLS] token's attention over the sequence, as a
// bar per word. As training proceeds the bar over the one sentiment word grows tallest — the model
// learning to look at the word that decides the class.

const POS = '#fb923c';   // positive word
const NEG = '#38bdf8';   // negative word
const FILLER = '#64748b'; // filler / [CLS]
const BAR = 'rgba(251, 191, 36, 0.9)'; // attention bar (amber)

export function drawAttention(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    words: string[],
    attentionRow: number[],
    polarity: ('pos' | 'neg' | 'filler')[],
): void {
    const n = words.length;
    if (n === 0) return;

    // Margins scale with the canvas so the title and value labels keep their headroom on a tall box.
    const padX = 10;
    const cellW = (width - 2 * padX) / n;
    const baseY = height - Math.max(28, height * 0.08); // words sit here
    const barTop = Math.max(26, height * 0.1);          // bars grow down to baseY from here
    const barAreaH = baseY - barTop;

    ctx.fillStyle = 'rgba(226, 232, 240, 0.7)';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('[CLS] attention — which word the model looks at', padX, 14);

    // The row is a softmax (sums to 1). Scale against a reference weight rather than the row's own max:
    // uniform early on → short, equal bars; the winning word's bar visibly grows as it locks on, and a
    // strongly-attended word (≳ FULL) fills the plot. (Normalising to the row max instead pinned the
    // tallest bar to the ceiling every frame, so nothing ever appeared to "grow".)
    const FULL = 0.6;

    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
        const cx = padX + (i + 0.5) * cellW;
        const barH = Math.min(1, Math.max(0, attentionRow[i]) / FULL) * barAreaH;

        // Attention bar.
        ctx.fillStyle = BAR;
        const barW = Math.min(cellW * 0.6, 44);
        ctx.fillRect(cx - barW / 2, baseY - barH, barW, barH);

        // Weight label above the bar.
        ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
        ctx.font = '9px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(attentionRow[i].toFixed(2), cx, baseY - barH - 4);

        // Word, coloured by polarity.
        ctx.fillStyle = polarity[i] === 'pos' ? POS : polarity[i] === 'neg' ? NEG : FILLER;
        ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(words[i], cx, baseY + 16);
    }
    ctx.textAlign = 'left';
}
