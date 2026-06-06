import { copyFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import mdx from '@mdx-js/rollup';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
    // GitHub Pages serves this project site under /machine-learning/.
    base: '/machine-learning/',
    plugins: [
        // MDX must run before the React plugin so its JSX output gets the React transform.
        {
            enforce: 'pre',
            ...mdx({
                remarkPlugins: [remarkMath],
                rehypePlugins: [rehypeKatex, rehypeSlug, [rehypeAutolinkHeadings, { behavior: 'wrap' }]],
                providerImportSource: '@mdx-js/react',
            }),
        },
        react({ include: /\.(mdx|js|jsx|ts|tsx)$/ }),
        // GitHub Pages has no SPA rewrite, so serve the app for unknown deep links too:
        // a copy of index.html as 404.html lets client-side routing take over on refresh.
        {
            name: 'spa-fallback',
            closeBundle() {
                const out = resolve(here, 'dist');
                if (existsSync(`${out}/index.html`)) copyFileSync(`${out}/index.html`, `${out}/404.html`);
            },
        },
    ],
    resolve: {
        alias: {
            // Import the library directly from its TypeScript source so edits hot-reload and
            // full types flow through — Vite/esbuild compiles the TS on the fly.
            'machine-learning': resolve(here, '../src/lib/index.ts'),
        },
    },
});
