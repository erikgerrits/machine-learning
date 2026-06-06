import { describe, it, expect } from 'vitest';
import Matrix from '../lib/math/linear-algebra/Matrix';
import { expectMatrixClose } from './helpers/matrix-assert';

describe('Matrix', () => {

    describe('construction', () => {
        it('builds from a 2D array and round-trips through toArray', () => {
            const matrix = new Matrix([[1, 2, 3], [4, 5, 6]]);

            expect(matrix.getRowCount()).toBe(2);
            expect(matrix.getColumnCount()).toBe(3);
            expect(matrix.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
        });

        it('builds from a Float64Array with explicit dimensions', () => {
            const matrix = new Matrix(new Float64Array([1, 2, 3, 4]), 2, 2);

            expect(matrix.toArray()).toEqual([[1, 2], [3, 4]]);
        });

        it('treats an empty array as a 0x0 matrix', () => {
            const matrix = new Matrix([]);

            expect(matrix.getRowCount()).toBe(0);
            expect(matrix.getColumnCount()).toBe(0);
        });
    });

    describe('factory methods', () => {
        it('ones / zeros fill the requested shape', () => {
            expect(Matrix.ones(2, 3).toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
            expect(Matrix.zeros(2, 2).toArray()).toEqual([[0, 0], [0, 0]]);
        });

        it('identity has ones on the diagonal', () => {
            expect(Matrix.identity(3).toArray()).toEqual([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]);
        });

        it('columnVector turns a flat array into a column', () => {
            expect(Matrix.columnVector([1, 2, 3]).toArray()).toEqual([[1], [2], [3]]);
        });

        describe('rand', () => {
            it('is deterministic for the same seed', () => {
                const a = Matrix.rand(3, 4, 1, 42);
                const b = Matrix.rand(3, 4, 1, 42);

                expect(a.toArray()).toEqual(b.toArray());
            });

            it('produces different output for different seeds', () => {
                const a = Matrix.rand(3, 4, 1, 1);
                const b = Matrix.rand(3, 4, 1, 2);

                expect(a.toArray()).not.toEqual(b.toArray());
            });

            it('keeps every value within [-epsilon, epsilon)', () => {
                const epsilon = 0.25;
                const values = Matrix.rand(10, 10, epsilon, 7).toArray().flat();

                for (const value of values) {
                    expect(value).toBeGreaterThanOrEqual(-epsilon);
                    expect(value).toBeLessThan(epsilon);
                }
            });
        });
    });

    describe('arithmetic', () => {
        it('adds two matrices', () => {
            const result = Matrix.add(new Matrix([[1, 2], [3, 4]]), new Matrix([[10, 20], [30, 40]]));

            expect(result.toArray()).toEqual([[11, 22], [33, 44]]);
        });

        it('adds a scalar to every element', () => {
            expect(Matrix.add(new Matrix([[1, 2], [3, 4]]), 1).toArray()).toEqual([[2, 3], [4, 5]]);
        });

        it('subtracts matrices and scalars', () => {
            expect(Matrix.subtract(new Matrix([[5, 5]]), new Matrix([[1, 2]])).toArray()).toEqual([[4, 3]]);
            expect(Matrix.subtract(new Matrix([[5, 5]]), 2).toArray()).toEqual([[3, 3]]);
        });

        it('multiplies two matrices (2x3 · 3x2)', () => {
            const a = new Matrix([[1, 2, 3], [4, 5, 6]]);
            const b = new Matrix([[7, 8], [9, 10], [11, 12]]);

            expect(Matrix.multiply(a, b).toArray()).toEqual([[58, 64], [139, 154]]);
        });

        it('multiplies by a scalar', () => {
            expect(Matrix.multiply(new Matrix([[1, 2], [3, 4]]), 2).toArray()).toEqual([[2, 4], [6, 8]]);
        });

        it('multiplies element-wise (Hadamard)', () => {
            const a = new Matrix([[1, 2], [3, 4]]);
            const b = new Matrix([[5, 6], [7, 8]]);

            expect(a.multiplyElementWise(b).toArray()).toEqual([[5, 12], [21, 32]]);
        });

        it('transposes', () => {
            expect(Matrix.transpose(new Matrix([[1, 2, 3], [4, 5, 6]])).toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
        });

        it('transform applies a function with row and column indices', () => {
            const matrix = new Matrix([[1, 2], [3, 4]]);

            const result = Matrix.transform(matrix, (value, row, column) => value * 10 + row + column);

            expect(result.toArray()).toEqual([[10, 21], [31, 42]]);
        });
    });

    describe('mutation semantics', () => {
        it('static operations return a clone and leave the operand unchanged', () => {
            const original = new Matrix([[1, 2], [3, 4]]);

            const result = Matrix.add(original, 100);

            expect(result.toArray()).toEqual([[101, 102], [103, 104]]);
            expect(original.toArray()).toEqual([[1, 2], [3, 4]]);
        });

        it('instance operations mutate the receiver in place', () => {
            const matrix = new Matrix([[1, 2], [3, 4]]);

            const returned = matrix.add(100);

            expect(returned).toBe(matrix);
            expect(matrix.toArray()).toEqual([[101, 102], [103, 104]]);
        });

        it('getClone produces an independent deep copy', () => {
            const original = new Matrix([[1, 2], [3, 4]]);
            const clone = original.getClone();

            clone.setElement(0, 0, 99);

            expect(clone.getElement(0, 0)).toBe(99);
            expect(original.getElement(0, 0)).toBe(1);
        });
    });

    describe('appending', () => {
        it('appends rows on the bottom and top', () => {
            expect(Matrix.appendBottom(new Matrix([[1, 2]]), new Matrix([[3, 4]])).toArray()).toEqual([[1, 2], [3, 4]]);
            expect(Matrix.appendTop(new Matrix([[1, 2]]), new Matrix([[3, 4]])).toArray()).toEqual([[3, 4], [1, 2]]);
        });

        it('appends columns on the left and right', () => {
            expect(Matrix.appendRight(new Matrix([[1], [2]]), new Matrix([[3], [4]])).toArray()).toEqual([[1, 3], [2, 4]]);
            expect(Matrix.appendLeft(new Matrix([[1], [2]]), new Matrix([[3], [4]])).toArray()).toEqual([[3, 1], [4, 2]]);
        });

        it('appending onto an empty matrix copies the operand', () => {
            expect(new Matrix([]).appendBottom(new Matrix([[1, 2]])).toArray()).toEqual([[1, 2]]);
            expect(new Matrix([]).appendRight(new Matrix([[1], [2]])).toArray()).toEqual([[1], [2]]);
        });
    });

    describe('accessors and slicing', () => {
        const matrix = () => new Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

        it('reads and writes individual elements', () => {
            const m = matrix();

            expect(m.getElement(1, 2)).toBe(6);
            m.setElement(1, 2, 60);
            expect(m.getElement(1, 2)).toBe(60);
        });

        it('extracts a single row and a range of rows', () => {
            expect(matrix().getRow(1).toArray()).toEqual([[4, 5, 6]]);
            expect(matrix().getRows(0, 1).toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
            expect(matrix().getRows(1).toArray()).toEqual([[4, 5, 6], [7, 8, 9]]); // default end = last row
        });

        it('extracts a single column and a range of columns', () => {
            expect(matrix().getColumn(0).toArray()).toEqual([[1], [4], [7]]);
            expect(matrix().getColumns(1, 2).toArray()).toEqual([[2, 3], [5, 6], [8, 9]]);
            expect(matrix().getColumns(1).toArray()).toEqual([[2, 3], [5, 6], [8, 9]]); // default end = last column
        });
    });

    describe('aggregates', () => {
        const matrix = new Matrix([[1, 5, 2], [9, 4, 6]]);

        it('sums all elements', () => {
            expect(matrix.getSum()).toBe(27);
        });

        it('reports the maximum and minimum values', () => {
            expect(matrix.getMaximumValue()).toBe(9);
            expect(matrix.getMinimumValue()).toBe(1);
        });

        it('returns the column index of the per-row maximum, breaking ties toward the first', () => {
            const tied = new Matrix([[1, 3, 2], [5, 4, 6], [9, 9, 1]]);

            expect(tied.getMaximumRowIndeces().toArray()).toEqual([[1], [2], [0]]);
        });
    });

    describe('determinant and inverse', () => {
        it('computes the determinant for 1x1, 2x2 and 3x3 matrices', () => {
            expect(new Matrix([[7]]).getDeterminant()).toBe(7);
            expect(new Matrix([[4, 7], [2, 6]]).getDeterminant()).toBe(10);
            expect(new Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]]).getDeterminant()).toBeCloseTo(1, 9);
        });

        it('inverts a 2x2 matrix', () => {
            expectMatrixClose(new Matrix([[4, 7], [2, 6]]).getInverse(), [[0.6, -0.7], [-0.2, 0.4]]);
        });

        it('A · A⁻¹ ≈ identity for a 3x3 matrix', () => {
            const a = new Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]]);

            expectMatrixClose(Matrix.multiply(a, a.getInverse()), Matrix.identity(3).toArray());
        });
    });

    describe('error handling', () => {
        it('throws when adding matrices of mismatched shape', () => {
            expect(() => new Matrix([[1, 2]]).add(new Matrix([[1], [2]]))).toThrow(/different number of rows/);
            expect(() => new Matrix([[1, 2]]).add(new Matrix([[1, 2, 3]]))).toThrow(/different number of columns/);
        });

        it('throws when subtracting matrices of mismatched shape', () => {
            expect(() => new Matrix([[1, 2]]).subtract(new Matrix([[1, 2, 3]]))).toThrow(/Cannot subtract/);
        });

        it('throws when inner dimensions do not match for multiplication', () => {
            expect(() => new Matrix([[1, 2]]).multiply(new Matrix([[1, 2]]))).toThrow(/Cannot multiply/);
        });

        it('throws on element-wise multiplication of mismatched shapes', () => {
            expect(() => new Matrix([[1, 2]]).multiplyElementWise(new Matrix([[1, 2, 3]]))).toThrow(/element-wise/);
        });

        it('throws when appending mismatched dimensions', () => {
            expect(() => new Matrix([[1, 2]]).appendBottom(new Matrix([[1, 2, 3]]))).toThrow(/appendBottom/);
            expect(() => new Matrix([[1, 2]]).appendTop(new Matrix([[1, 2, 3]]))).toThrow(/appendTop/);
            expect(() => new Matrix([[1], [2]]).appendLeft(new Matrix([[1]]))).toThrow(/appendLeft/);
            expect(() => new Matrix([[1], [2]]).appendRight(new Matrix([[1]]))).toThrow(/appendRight/);
        });

        it('throws when computing the determinant or inverse of a non-square matrix', () => {
            expect(() => new Matrix([[1, 2, 3]]).getDeterminant()).toThrow(/non-square/);
            expect(() => new Matrix([[1, 2, 3]]).getInverse()).toThrow(/must be square/);
        });

        it('throws when inverting a singular matrix', () => {
            expect(() => new Matrix([[1, 2], [2, 4]]).getInverse()).toThrow(/zero determinant/);
        });
    });
});
