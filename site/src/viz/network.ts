// Draws the network as nodes-and-edges, with each edge's thickness ∝ |weight| and its color
// keyed to the weight's sign. Called every frame, so the edges visibly "pulse" as the network
// learns — the synchronized counterpart to the morphing decision boundary.

interface Node {
    x: number;
    y: number;
}

/**
 * @param weights  one matrix per layer transition as plain arrays, shape (inputs+1) × outputs;
 *                 row 0 holds the bias weights.
 * @param arch     node count per layer, e.g. [2, 8, 1].
 */
export function drawNetwork(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    weights: number[][][],
    arch: number[],
): void {
    ctx.clearRect(0, 0, width, height);

    const padX = 34;
    const padY = 26;
    const layers = arch.length;
    const columnGap = layers > 1 ? (width - padX * 2) / (layers - 1) : 0;

    // Node positions per layer, vertically centered.
    const positions: Node[][] = arch.map((count, layer) => {
        const x = padX + columnGap * layer;
        const slot = (height - padY * 2) / Math.max(count, 1);
        return Array.from({ length: count }, (_, j) => ({ x, y: padY + slot * (j + 0.5) }));
    });

    // A bias node sits at the bottom of every layer that feeds a next layer.
    const biasNodes: (Node | null)[] = arch.map((_, layer) =>
        layer < layers - 1 ? { x: padX + columnGap * layer, y: height - padY * 0.5 } : null,
    );

    // Normalize edge widths against the largest current weight magnitude.
    let maxAbs = 1e-6;
    for (const matrix of weights) {
        for (const row of matrix) {
            for (const w of row) maxAbs = Math.max(maxAbs, Math.abs(w));
        }
    }

    // Edges (drawn under the nodes).
    for (let layer = 0; layer < weights.length; layer++) {
        const matrix = weights[layer];
        for (let s = 0; s < matrix.length; s++) {
            const source = s === 0 ? biasNodes[layer] : positions[layer][s - 1];
            if (!source) continue;
            for (let o = 0; o < matrix[s].length; o++) {
                const target = positions[layer + 1][o];
                const norm = Math.abs(matrix[s][o]) / maxAbs;
                const channel = matrix[s][o] >= 0 ? '56, 189, 248' : '251, 146, 60';

                ctx.beginPath();
                ctx.moveTo(source.x, source.y);
                ctx.lineTo(target.x, target.y);
                ctx.lineWidth = 0.4 + norm * 3;
                ctx.strokeStyle = `rgba(${channel}, ${0.12 + 0.6 * norm})`;
                ctx.stroke();
            }
        }
    }

    // Bias nodes.
    for (const bias of biasNodes) {
        if (!bias) continue;
        ctx.beginPath();
        ctx.arc(bias.x, bias.y, 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#1e293b';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#475569';
        ctx.stroke();
    }

    // Neuron nodes.
    for (const layer of positions) {
        for (const node of layer) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 7, 0, 2 * Math.PI);
            ctx.fillStyle = '#0b1120';
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#64748b';
            ctx.stroke();
        }
    }
}
