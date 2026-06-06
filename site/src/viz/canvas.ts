/**
 * Sizes a canvas's backing store to its CSS box × devicePixelRatio (for crisp rendering on
 * retina displays) and returns a 2D context already scaled so callers can draw in CSS pixels.
 * Safe to call every frame — it only resizes when the dimensions actually change.
 */
export function fitCanvas(canvas: HTMLCanvasElement, cssWidth: number, cssHeight: number): CanvasRenderingContext2D {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(cssWidth * dpr);
    const h = Math.round(cssHeight * dpr);

    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = `${cssWidth}px`;
        canvas.style.height = `${cssHeight}px`;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
}
