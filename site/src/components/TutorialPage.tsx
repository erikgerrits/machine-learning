import type { ReactNode } from 'react';
import '../styles/prose.css';

/** Wraps an MDX tutorial in the readable "prose" column with embedded-demo break-outs. */
export function TutorialPage({ children }: { children: ReactNode }) {
    return <article className="prose">{children}</article>;
}
