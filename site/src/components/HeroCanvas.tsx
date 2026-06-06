import { useEffect, useRef } from 'react';
import { drawNetwork } from '../viz/network';

const ARCH = [3, 6, 6, 2];
const W = 520;
const H = 300;

// An idle "thinking" animation: a network whose weights breathe in and out via sine waves —
// purely decorative, but it sets the tone that this site is about networks in motion.
export function HeroCanvas() {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Stable per-edge phase offsets so each connection oscillates independently.
        const phases = ARCH.slice(0, -1).map((inCount, layer) =>
            Array.from({ length: inCount + 1 }, (_, s) =>
                Array.from({ length: ARCH[layer + 1] }, (_, o) => s * 7 + o * 13 + layer * 29),
            ),
        );

        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let tick = 0;
        let stopped = false;
        let frame = 0;

        const render = () => {
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            const weights = phases.map(layer =>
                layer.map(row => row.map(phase => Math.sin(tick * 0.03 + phase))),
            );
            drawNetwork(ctx, W, H, weights, ARCH);
            if (stopped || reduced) return;
            tick += 1;
            frame = requestAnimationFrame(render);
        };

        render();
        return () => {
            stopped = true;
            cancelAnimationFrame(frame);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            aria-hidden="true"
            style={{ width: '100%', maxWidth: W, height: 'auto' }}
        />
    );
}
