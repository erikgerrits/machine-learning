import type { Domain } from '../ml/datasets';

// PCA scene colours.
const POINT = '#38bdf8';   // data points (sky)
const PC1_COLOR = '#fb923c'; // first principal component (amber)
const PC2_COLOR = '#64748b'; // second component (slate)

function toPx(x: number, y: number, domain: Domain, width: number, height: number): [number, number] {
    const px = ((x - domain.xMin) / (domain.xMax - domain.xMin)) * width;
    const py = (1 - (y - domain.yMin) / (domain.yMax - domain.yMin)) * height;
    return [px, py];
}

/** The dashed line through the mean along PC1 — the axis points collapse onto when reducing to 1-D. */
export function drawPc1Line(
    ctx: CanvasRenderingContext2D,
    mean: number[],
    pc1: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    const reach = 6; // long enough to span the plot
    const [ax, ay] = toPx(mean[0] - pc1[0] * reach, mean[1] - pc1[1] * reach, domain, width, height);
    const [bx, by] = toPx(mean[0] + pc1[0] * reach, mean[1] + pc1[1] * reach, domain, width, height);

    ctx.strokeStyle = 'rgba(251, 146, 60, 0.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
}

/**
 * Draws the data points, interpolated `t` of the way (0 → 1) toward their projection onto PC1. At
 * `t = 0` they sit where they really are; at `t = 1` they've collapsed onto the PC1 line — the 2-D →
 * 1-D reduction, made visible. Faint connectors show how far each point moved (the lost variance).
 */
export function drawPcaPoints(
    ctx: CanvasRenderingContext2D,
    points: number[][],
    mean: number[],
    pc1: number[],
    domain: Domain,
    width: number,
    height: number,
    t: number,
): void {
    for (const [x, y] of points) {
        const scalar = (x - mean[0]) * pc1[0] + (y - mean[1]) * pc1[1];
        const projX = mean[0] + scalar * pc1[0];
        const projY = mean[1] + scalar * pc1[1];
        const cx = x + (projX - x) * t;
        const cy = y + (projY - y) * t;

        if (t > 0.01) {
            const [ox, oy] = toPx(x, y, domain, width, height);
            const [px, py] = toPx(cx, cy, domain, width, height);
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(ox, oy);
            ctx.lineTo(px, py);
            ctx.stroke();
        }

        const [dx, dy] = toPx(cx, cy, domain, width, height);
        ctx.beginPath();
        ctx.arc(dx, dy, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = POINT;
        ctx.fill();
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();
    }
}

/**
 * Draws the principal components as arrows from the mean, each ±2 standard deviations long, so the
 * longer arrow (PC1) literally points along the data's main spread and the shorter one (PC2) across it.
 */
export function drawPcaAxes(
    ctx: CanvasRenderingContext2D,
    mean: number[],
    components: number[][],
    standardDeviations: number[],
    domain: Domain,
    width: number,
    height: number,
): void {
    const colors = [PC1_COLOR, PC2_COLOR];
    for (let c = Math.min(components.length, 2) - 1; c >= 0; c--) {
        const unit = components[c];
        const len = 2 * standardDeviations[c];
        const [mx, my] = toPx(mean[0], mean[1], domain, width, height);
        const [ex, ey] = toPx(mean[0] + unit[0] * len, mean[1] + unit[1] * len, domain, width, height);
        const [sx, sy] = toPx(mean[0] - unit[0] * len, mean[1] - unit[1] * len, domain, width, height);

        ctx.strokeStyle = colors[c];
        ctx.lineWidth = c === 0 ? 3 : 2;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // A small arrowhead on the positive end.
        const angle = Math.atan2(ey - my, ex - mx);
        const head = 9;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - head * Math.cos(angle - 0.4), ey - head * Math.sin(angle - 0.4));
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - head * Math.cos(angle + 0.4), ey - head * Math.sin(angle + 0.4));
        ctx.stroke();
    }
}
