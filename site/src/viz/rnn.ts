// Scatter of the learned 2-D word embeddings — the heart of the RNN chapter. As training proceeds,
// words that swing sentiment the same way drift together, so positive and negative words separate
// while fillers stay in the muddle.

const POS = '#fb923c';   // amber — positive words
const NEG = '#38bdf8';   // sky — negative words
const NEUTRAL = '#64748b'; // slate — fillers / nouns

export function drawEmbeddings(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    embeddings: number[][],
    vocab: string[],
    positive: Set<number>,
    negative: Set<number>,
): void {
    const pad = 26;
    // Bounds over real tokens (skip <pad> at index 0).
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let t = 1; t < embeddings.length; t++) {
        minX = Math.min(minX, embeddings[t][0]);
        maxX = Math.max(maxX, embeddings[t][0]);
        minY = Math.min(minY, embeddings[t][1]);
        maxY = Math.max(maxY, embeddings[t][1]);
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const toX = (x: number) => pad + ((x - minX) / spanX) * (width - 2 * pad);
    const toY = (y: number) => pad + (1 - (y - minY) / spanY) * (height - 2 * pad);

    ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (let t = 1; t < embeddings.length; t++) {
        const px = toX(embeddings[t][0]);
        const py = toY(embeddings[t][1]);
        const color = positive.has(t) ? POS : negative.has(t) ? NEG : NEUTRAL;

        ctx.beginPath();
        ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.fillStyle = color === NEUTRAL ? 'rgba(148,163,184,0.85)' : color;
        ctx.fillText(vocab[t], px + 5, py);
    }
}
