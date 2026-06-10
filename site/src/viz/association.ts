import type { AssociationRule } from 'machine-learning';

/**
 * Draws the **co-occurrence web**: menu items as labelled nodes around a circle, and each
 * single-item → single-item rule as a curved arrow from antecedent to consequent. Stronger
 * associations (higher lift) draw thicker and brighter, so the habits worth knowing pop out. Node
 * size reflects how often each item is bought at all.
 */
export function drawAssociationWeb(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    items: string[],
    rules: AssociationRule[],
    itemSupport: number[],
): void {
    const n = items.length;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.34;

    const nodeAt = (i: number): [number, number] => {
        const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
        return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
    };

    // Edges: pairwise rules only (single item → single item), strongest drawn last so they sit on top.
    const pairwise = rules
        .filter(rule => rule.antecedent.length === 1 && rule.consequent.length === 1)
        .sort((a, b) => a.lift - b.lift);

    for (const rule of pairwise) {
        const [ax, ay] = nodeAt(rule.antecedent[0]);
        const [bx, by] = nodeAt(rule.consequent[0]);

        // Bow the curve so opposite directions (A→B vs B→A) don't overlap.
        const midX = (ax + bx) / 2;
        const midY = (ay + by) / 2;
        const bow = rule.antecedent[0] < rule.consequent[0] ? 0.16 : -0.16;
        const controlX = midX + (cy - midY) * 0 + (by - ay) * bow;
        const controlY = midY + (ax - bx) * bow;

        const intensity = Math.min(1, Math.max(0, (rule.lift - 1) / 1.5));
        ctx.strokeStyle = `rgba(251, 146, 60, ${0.25 + 0.65 * intensity})`;
        ctx.lineWidth = 1 + 4 * intensity;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.quadraticCurveTo(controlX, controlY, bx, by);
        ctx.stroke();

        // Arrowhead near the consequent.
        const angle = Math.atan2(by - controlY, bx - controlX);
        const head = 8;
        ctx.fillStyle = `rgba(251, 146, 60, ${0.4 + 0.6 * intensity})`;
        ctx.beginPath();
        ctx.moveTo(bx - head * Math.cos(angle - 0.4), by - head * Math.sin(angle - 0.4));
        ctx.lineTo(bx, by);
        ctx.lineTo(bx - head * Math.cos(angle + 0.4), by - head * Math.sin(angle + 0.4));
        ctx.fill();
    }

    // Nodes + labels.
    ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
    for (let i = 0; i < n; i++) {
        const [x, y] = nodeAt(i);
        const r = 5 + 10 * Math.min(1, itemSupport[i]);

        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = 'rgba(11, 17, 32, 0.85)';
        ctx.stroke();

        const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
        const outward = Math.cos(angle);
        const lx = x + outward * 14;
        const ly = y + Math.sin(angle) * 14;
        ctx.fillStyle = '#e2e8f0';
        ctx.textAlign = outward >= 0 ? 'left' : 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(items[i], lx, ly);
    }
    ctx.textAlign = 'left';
}
