import { fitCanvas } from './canvas';
import { CLUSTER_HEX } from './clusters';

// Two views of a VAE. The generative manifold decodes a fixed grid of latent codes spanning the
// N(0,1) prior — every tile is an *invented* image, and because the latent space is smooth they sweep
// continuously into one another, the real data's codes scattered on top. The gallery decodes a fixed
// handful of random codes; watch them morph from noise into clean samples as training proceeds.

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
    gridRes: number;
    tiles: number[][];      // decoded image at each prior-grid point, row-major (z2 ascending up)
    imageWidth: number;
    dataCodes: number[][];  // latent means of the real data
    dataColors: number[];
    codeMin: [number, number];
    codeMax: [number, number];
    showData: boolean;
}

export function drawGenerativeManifold(canvas: HTMLCanvasElement, view: ManifoldView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const pad = 6;
    const size = Math.min(width, height) - pad * 2;
    const originX = (width - size) / 2;
    const originY = (height - size) / 2;
    const tile = size / view.gridRes;

    for (let r = 0; r < view.gridRes; r++) {
        for (let c = 0; c < view.gridRes; c++) {
            drawImage(ctx, originX + c * tile, originY + (view.gridRes - 1 - r) * tile, tile, view.tiles[r * view.gridRes + c], view.imageWidth);
        }
    }

    if (view.showData) {
        const spanX = view.codeMax[0] - view.codeMin[0] || 1;
        const spanY = view.codeMax[1] - view.codeMin[1] || 1;
        for (let i = 0; i < view.dataCodes.length; i++) {
            const px = originX + ((view.dataCodes[i][0] - view.codeMin[0]) / spanX) * size;
            const py = originY + (1 - (view.dataCodes[i][1] - view.codeMin[1]) / spanY) * size;
            if (px < originX || px > originX + size || py < originY || py > originY + size) continue;
            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = colorRamp(view.dataColors[i]);
            ctx.globalAlpha = 0.8;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
    }
}

export interface GalleryView {
    images: number[][];
    imageWidth: number;
    cols: number;
}

export function drawSampleGallery(canvas: HTMLCanvasElement, view: GalleryView): void {
    const { ctx, width, height } = fitCanvas(canvas);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, width, height);

    const rows = Math.ceil(view.images.length / view.cols);
    const gap = 4;
    const size = Math.min((width - gap * (view.cols + 1)) / view.cols, (height - gap * (rows + 1)) / rows);
    const totalW = size * view.cols + gap * (view.cols - 1);
    const totalH = size * rows + gap * (rows - 1);
    const startX = (width - totalW) / 2;
    const startY = (height - totalH) / 2;

    for (let i = 0; i < view.images.length; i++) {
        const r = Math.floor(i / view.cols);
        const c = i % view.cols;
        drawImage(ctx, startX + c * (size + gap), startY + r * (size + gap), size, view.images[i], view.imageWidth);
    }
}

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
