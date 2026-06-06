import { expect } from 'vitest';
import Matrix from '../../lib/math/linear-algebra/Matrix';

/**
 * Assert that a Matrix has the expected shape and element values (within a
 * floating-point tolerance). `precision` is the number of decimal digits passed
 * to Vitest's `toBeCloseTo` (default 9 ≈ exact for our deterministic math).
 */
export function expectMatrixClose(actual: Matrix, expected: number[][], precision = 9): void {
    const rows = actual.toArray();

    expect(rows.length).toBe(expected.length);

    for (let i = 0; i < rows.length; i++) {
        expect(rows[i].length).toBe(expected[i].length);

        for (let j = 0; j < rows[i].length; j++) {
            expect(rows[i][j]).toBeCloseTo(expected[i][j], precision);
        }
    }
}
