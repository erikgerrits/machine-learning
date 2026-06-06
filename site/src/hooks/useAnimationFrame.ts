import { useEffect, useRef } from 'react';

/**
 * Runs `callback` once per animation frame while `active` is true. The callback is kept in a
 * ref so the loop always sees the latest closure (current hyperparameters) without restarting,
 * and frames are skipped while the tab is hidden to avoid a runaway backlog.
 */
export function useAnimationFrame(callback: () => void, active: boolean): void {
    const callbackRef = useRef(callback);

    useEffect(() => {
        callbackRef.current = callback;
    });

    useEffect(() => {
        if (!active) return;

        let frame = 0;
        let stopped = false;

        const loop = () => {
            if (stopped) return;
            if (!document.hidden) callbackRef.current();
            frame = requestAnimationFrame(loop);
        };

        frame = requestAnimationFrame(loop);
        return () => {
            stopped = true;
            cancelAnimationFrame(frame);
        };
    }, [active]);
}
