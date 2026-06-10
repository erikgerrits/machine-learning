// Drawing helpers for the CNN playground: render small 2-D grids (images, feature maps, filters).

const SKY: [number, number, number] = [56, 189, 248];
const AMBER: [number, number, number] = [251, 146, 60];
const DARK: [number, number, number] = [11, 17, 32];

function lerp(a: [number, number, number], b: [number, number, number], t: number): string {
    const r = Math.round(a[0] + (b[0] - a[0]) * t);
    const g = Math.round(a[1] + (b[1] - a[1]) * t);
    const bch = Math.round(a[2] + (b[2] - a[2]) * t);
    return `rgb(${r}, ${g}, ${bch})`;
}

/**
 * Draws a 2-D array into a `box`-pixel square at (x, y). Grayscale for images / feature maps (dark →
 * bright with value); a diverging blue↔amber map for filters (negative weights blue, positive amber),
 * so a learned edge detector reads at a glance.
 */
export function drawGrid(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    box: number,
    grid: number[][],
    diverging = false,
): void {
    const rows = grid.length;
    const cols = grid[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    const cell = box / Math.max(rows, cols);

    let scale = 1e-6;
    for (const row of grid) {
        for (const v of row) {
            scale = Math.max(scale, diverging ? Math.abs(v) : v);
        }
    }

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const v = grid[i][j];
            let color: string;
            if (diverging) {
                const t = Math.max(-1, Math.min(1, v / scale));
                color = t >= 0 ? lerp(DARK, AMBER, t) : lerp(DARK, SKY, -t);
            } else {
                color = lerp(DARK, [255, 255, 255], Math.max(0, Math.min(1, v / scale)));
            }
            ctx.fillStyle = color;
            ctx.fillRect(x + j * cell, y + i * cell, Math.ceil(cell), Math.ceil(cell));
        }
    }

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, box, box);
}

/** Small caption above a grid. */
export function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number): void {
    ctx.fillStyle = 'rgba(226, 232, 240, 0.85)';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(text, x, y);
}
