import { clusterHex } from './clusters';

/** One merge of the dendrogram (matches the library's `getMergeHistory()` shape). */
export interface DendrogramMerge {
    left: number;
    right: number;
    distance: number;
}

/**
 * Draws the dendrogram — the tree of merges hierarchical clustering builds. Leaves (the original
 * `n` points) sit along the bottom; each merge is an inverted-U bracket drawn at the height
 * (distance) the two groups joined, so taller brackets = groups that were farther apart.
 *
 * A dashed **cut line** marks where the tree is sliced for the current `k`: branches *below* it
 * each sit inside one final cluster and are coloured to match the map, while the merges *above* it
 * (the ones the cut discards) are greyed. Moving `k` slides that line — the whole point of the
 * chapter: you pick the number of clusters by choosing a height, after the tree is built.
 */
export function drawDendrogram(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    merges: DendrogramMerge[],
    n: number,
    k: number,
    labels: number[],
): void {
    ctx.clearRect(0, 0, width, height);
    if (n < 2 || merges.length === 0) return;

    const padX = 10;
    const padTop = 16;
    const padBottom = 10;
    const plotW = width - 2 * padX;
    const plotH = height - padTop - padBottom;

    const totalNodes = 2 * n - 1;
    const childLeft = new Array<number>(totalNodes).fill(-1);
    const childRight = new Array<number>(totalNodes).fill(-1);
    const nodeHeight = new Array<number>(totalNodes).fill(0);
    for (let t = 0; t < merges.length; t++) {
        const id = n + t;
        childLeft[id] = merges[t].left;
        childRight[id] = merges[t].right;
        nodeHeight[id] = merges[t].distance;
    }
    const root = n + merges.length - 1;
    const maxHeight = nodeHeight[root] || 1;

    // Lay out leaves left→right in tree order (no crossing branches); each internal node sits above
    // the midpoint of its two children. `repLeaf` lets us colour a branch by its cluster.
    const xpos = new Array<number>(totalNodes).fill(0);
    const repLeaf = new Array<number>(totalNodes).fill(0);
    let slot = 0;
    const assign = (node: number): void => {
        if (node < n) {
            xpos[node] = (slot + 0.5) / n;
            repLeaf[node] = node;
            slot++;
            return;
        }
        assign(childLeft[node]);
        assign(childRight[node]);
        xpos[node] = (xpos[childLeft[node]] + xpos[childRight[node]]) / 2;
        repLeaf[node] = repLeaf[childLeft[node]];
    };
    assign(root);

    const toX = (x: number) => padX + x * plotW;
    const toY = (h: number) => padTop + (1 - h / maxHeight) * plotH; // height 0 at the bottom

    const below = n - k; // the first `below` merges sit under the cut, inside single clusters

    for (let t = 0; t < merges.length; t++) {
        const id = n + t;
        const left = childLeft[id];
        const right = childRight[id];
        const yMerge = toY(nodeHeight[id]);
        const xLeft = toX(xpos[left]);
        const xRight = toX(xpos[right]);

        const inCluster = t < below;
        ctx.strokeStyle = inCluster ? clusterHex(labels[repLeaf[id]]) : 'rgba(148, 163, 184, 0.55)';
        ctx.lineWidth = inCluster ? 2 : 1.25;
        ctx.beginPath();
        ctx.moveTo(xLeft, toY(nodeHeight[left]));
        ctx.lineTo(xLeft, yMerge);
        ctx.moveTo(xRight, toY(nodeHeight[right]));
        ctx.lineTo(xRight, yMerge);
        ctx.moveTo(xLeft, yMerge);
        ctx.lineTo(xRight, yMerge);
        ctx.stroke();
    }

    // The cut sits between the last below-cut merge and the first above-cut one.
    const lower = below > 0 ? nodeHeight[n + below - 1] : 0;
    const upper = below < merges.length ? nodeHeight[n + below] : maxHeight * 1.05;
    const cutY = toY((lower + upper) / 2);

    ctx.strokeStyle = 'rgba(248, 250, 252, 0.7)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(padX, cutY);
    ctx.lineTo(width - padX, cutY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(248, 250, 252, 0.8)';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`cut → ${k} cluster${k === 1 ? '' : 's'}`, padX + 2, cutY - 3);
}
