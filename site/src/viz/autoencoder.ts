import { fitCanvas } from './canvas';
import { CLUSTER_HEX } from './clusters';

// Two views of an autoencoder. The latent manifold tiles code-space with *decoded* thumbnails — move
// across the grid and the reconstructed image sweeps smoothly through the data (the learned 2-D map of
// what the bottleneck represents) — with the real images' codes scattered on top. The triptych shows
// one image clean, corrupted, and reconstructed: denoising made literal.

function drawImage(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, img: number[], width: number) {
    const cell = size / width;
    for (let r = 0; r < width; r++) {
        for (let c = 0; c < width; c++) {
            const v = Math.max(0, Math.min(1, img[r * width + c]));
            const shade = Math.round(v * 255);
            ctx.fillStyle = `rgb(${shade}, ${Math.round(shade * 0.95 + 10)}, ${Math.round(shade * 0.8 + 20)})`;
            ctx.fillRect(x + c * cell, y + r * cell, cell + 0.5, cell + 0.5);
        }
    }
}

export interface ManifoldView {
    gridRes: number;        // tiles per side
    tiles: number[][];      // decoded image per grid point, row-major (z2 ascending down → flipped to draw)
    imageWidth: number;
    dataCodes: number[][];  // 2-D codes of the real images
    dataColors: number[];   // 0..1 colour per data point
    codeMin: [number, number];
    codeMax: [number, number];
    showData: boolean;
}

export function drawLatentManifold(canvas: HTMLCanvasElement, view: ManifoldView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const pad = 6;
    const size = Math.min(width, height) - pad * 2;
    const originX = (width - size) / 2;
    const originY = (height - size) / 2;
    const tile = size / view.gridRes;

    // Tiles: column c spans code-x, row r spans code-y; draw with y increasing upward.
    for (let r = 0; r < view.gridRes; r++) {
        for (let c = 0; c < view.gridRes; c++) {
            const img = view.tiles[r * view.gridRes + c];
            drawImage(ctx, originX + c * tile, originY + (view.gridRes - 1 - r) * tile, tile, img, view.imageWidth);
        }
    }

    // Real data codes on top, positioned by where they fall in code-space.
    if (view.showData) {
        const spanX = view.codeMax[0] - view.codeMin[0] || 1;
        const spanY = view.codeMax[1] - view.codeMin[1] || 1;
        for (let i = 0; i < view.dataCodes.length; i++) {
            const px = originX + ((view.dataCodes[i][0] - view.codeMin[0]) / spanX) * size;
            const py = originY + (1 - (view.dataCodes[i][1] - view.codeMin[1]) / spanY) * size;
            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI * 2);
            ctx.fillStyle = colorRamp(view.dataColors[i]);
            ctx.globalAlpha = 0.85;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
    }
}

export interface TriptychView {
    images: number[][]; // [clean, noisy, reconstructed]
    labels: string[];
    imageWidth: number;
}

export function drawTriptych(canvas: HTMLCanvasElement, view: TriptychView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const gap = 10;
    const labelH = 16;
    const n = view.images.length;
    const size = Math.min((width - gap * (n + 1)) / n, height - labelH - gap);
    const totalW = size * n + gap * (n - 1);
    const startX = (width - totalW) / 2;
    const top = labelH;

    for (let i = 0; i < n; i++) {
        const x = startX + i * (size + gap);
        drawImage(ctx, x, top, size, view.images[i], view.imageWidth);
        ctx.strokeStyle = 'rgba(148,163,184,0.3)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, top + 0.5, size, size);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(view.labels[i], x + size / 2, top / 2);
    }
}

// 0 → sky, 1 → amber, blending the two cluster hues so the latent scatter reads as a smooth gradient.
function colorRamp(t: number): string {
    const u = Math.max(0, Math.min(1, t));
    const a = hexToRgb(CLUSTER_HEX[0]);
    const b = hexToRgb(CLUSTER_HEX[1]);
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * u)}, ${Math.round(a[1] + (b[1] - a[1]) * u)}, ${Math.round(a[2] + (b[2] - a[2]) * u)})`;
}

function hexToRgb(hex: string): [number, number, number] {
    const v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
