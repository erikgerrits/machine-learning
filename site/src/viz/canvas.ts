/**
 * Matches a canvas's backing store to its actual laid-out CSS box × devicePixelRatio (crisp on
 * retina, and responsive — it never forces a fixed size that its container would clip) and
 * returns a context scaled so callers draw in CSS pixels, plus that measured width/height.
 * Safe to call every frame: it only resizes the backing store when the box actually changes.
 */
export function fitCanvas(canvas: HTMLCanvasElement): {
    ctx: CanvasRenderingContext2D;
    width: number;
    height: number;
} {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
}
