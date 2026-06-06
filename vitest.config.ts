import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        include: ['src/test/**/*.test.ts'],
        coverage: {
            provider: 'v8',
            include: ['src/lib/**'],
        },
    },
});
