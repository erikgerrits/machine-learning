import type { Domain } from '../ml/datasets';

// A six-way cluster palette (sky, amber, violet, emerald, pink, yellow) — enough distinct hues
// for every k the playground allows. Colours wrap if k somehow exceeds the palette length.
export const CLUSTER_RGB: [number, number, number][] = [
    [56, 189, 248],
    [251, 146, 60],
    [167, 139, 250],
    [52, 211, 153],
    [244, 114, 182],
    [250, 204, 21],
];
export const CLUSTER_HEX = ['#38bdf8', '#fb923c', '#a78bfa', '#34d399', '#f472b6', '#facc15'];

export function clusterHex(index: number): string {
    return CLUSTER_HEX[index % CLUSTER_HEX.length];
}

function project(x: number, y: number, domain: Domain, width: number, height: number): [number, number] {
    const px = ((x - domain.xMin) / (domain.xMax - domain.xMin)) * width;
    const py = (1 - (y - domain.yMin) / (domain.yMax - domain.yMin)) * height;
    return [px, py];
}

/**
 * Paints the cluster map into a size×size offscreen canvas: each cell is filled with the colour
 * of the centroid it belongs to, so the plane reads as a set of Voronoi regions. `assignments`
 * holds one cluster index per cell (row-major, matching the sampling grid).
 */
export function paintClusters(offscreen: HTMLCanvasElement, assignments: number[], size: number): void {
    if (offscreen.width !== size || offscreen.height !== size) {
        offscreen.width = size;
        offscreen.height = size;
    }
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const image = ctx.createImageData(size, size);
    for (let i = 0; i < assignments.length; i++) {
        const [r, g, b] = CLUSTER_RGB[assignments[i] % CLUSTER_RGB.length];
        const o = i * 4;
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 150; // translucent so points and centroids stay legible on top
    }
    ctx.putImageData(image, 0, 0);
}

/** Draws the data points, each coloured by the cluster it is currently assigned to. */
export function drawClusterPoints(
    ctx: CanvasRenderingContext2D,
    inputs: number[][],
    assignments: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    for (let i = 0; i < inputs.length; i++) {
        const [px, py] = project(inputs[i][0], inputs[i][1], domain, width, height);
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = clusterHex(assignments[i]);
        ctx.fill();
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();
    }
}

/** Draws each centroid as a bold, white-ringed diamond so it stands out from the round points. */
export function drawCentroids(
    ctx: CanvasRenderingContext2D,
    centroids: number[][],
    domain: Domain,
    width: number,
    height: number,
): void {
    const s = 8;
    for (let c = 0; c < centroids.length; c++) {
        const [px, py] = project(centroids[c][0], centroids[c][1], domain, width, height);

        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(Math.PI / 4); // a rotated square reads as a diamond
        ctx.lineJoin = 'round';
        ctx.fillStyle = clusterHex(c);
        ctx.beginPath();
        ctx.rect(-s, -s, 2 * s, 2 * s);
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#f8fafc';
        ctx.stroke();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.9)';
        ctx.stroke();
        ctx.restore();
    }
}

/** Mean squared distance from each point to its assigned centroid — the quantity k-means minimises. */
export function inertia(inputs: number[][], centroids: number[][], assignments: number[]): number {
    if (inputs.length === 0) return 0;

    let sum = 0;
    for (let i = 0; i < inputs.length; i++) {
        const centroid = centroids[assignments[i]];
        for (let j = 0; j < inputs[i].length; j++) {
            const diff = inputs[i][j] - centroid[j];
            sum += diff * diff;
        }
    }
    return sum / inputs.length;
}

/** Total distance the centroids travelled between two iterations — 0 means k-means has converged. */
export function centroidsMoved(before: number[][], after: number[][]): number {
    let total = 0;
    for (let c = 0; c < after.length; c++) {
        let squared = 0;
        for (let j = 0; j < after[c].length; j++) {
            const diff = after[c][j] - before[c][j];
            squared += diff * diff;
        }
        total += Math.sqrt(squared);
    }
    return total;
}
