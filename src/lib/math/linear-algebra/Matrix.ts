import * as nblas from 'nblas';
import { Chance } from 'chance';

export default class Matrix {

    private data: Float64Array;
    private rowCount: number;
    private columnCount: number;
    private blasEnabled = true;

    public constructor (data: Float64Array, rowCount: number, columnCount: number);
    public constructor (elements: number[][]);
    public constructor (data: Float64Array|number[][], rowCount?: number, columnCount?: number) {
        if (data instanceof Float64Array) {
            this.data = data;
        
        } else {
            rowCount = data.length;
            columnCount = rowCount > 0 ? data[0].length : 0;
            this.data = new Float64Array(rowCount * columnCount);
            let dataIndex = 0;
            for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
                for (let columnIndex = 0; columnIndex < columnCount; columnIndex++) {
                    this.data[dataIndex++] = data[rowIndex][columnIndex];
                }
            }
        }

        this.rowCount = rowCount;
        this.columnCount = columnCount;
    }


    public static add (matrix1: Matrix, matrix2: Matrix): Matrix;
    public static add (matrix: Matrix, scalar: number): Matrix;
    public static add (matrix: Matrix, value: any) {
        return matrix.getClone().add(value);
    }

    public static columnVector (elements: number[]) {
        return new Matrix(elements.map((element) => [element]));
    }

    public static identity (dimensions: number) {
        const size = dimensions * dimensions;
        const data = new Float64Array(size);

        const increment = dimensions + 1;
        for (let diagonalIndex = 0; diagonalIndex < size; diagonalIndex += increment) {
            data[diagonalIndex] = 1;
        }

        return new Matrix(data, dimensions, dimensions);
    }

    public static multiply (matrix1: Matrix, matrix2: Matrix): Matrix;
    public static multiply (matrix: Matrix, scalar: number): Matrix;
    public static multiply (matrix: Matrix, value: any) {
        return matrix.getClone().multiply(value);
    }

    public static ones (rowCount: number, columnCount: number) {
        const size = rowCount * columnCount;
        const data = new Float64Array(size);

        for (let index = 0; index < size; index++) {
            data[index] = 1;
        }

        return new Matrix(data, rowCount, columnCount);
    }

    public static rand (rowCount: number, columnCount: number, epsilon = 1, seed: number = undefined) {
        const chance = seed !== undefined ? new Chance(seed) : new Chance();
        const size = rowCount * columnCount;
        const data = new Float64Array(size);

        for (let index = 0; index < size; index++) {
            data[index] = chance.floating({min: 0, max: 1}) * 2 * epsilon - epsilon;
        }

        return new Matrix(data, rowCount, columnCount);
    }

    public static subtract (matrix1: Matrix, matrix2: Matrix): Matrix;
    public static subtract (matrix: Matrix, scalar: number): Matrix;
    public static subtract (matrix: Matrix, value: any) {
        return matrix.getClone().subtract(value);
    }

    public static transform (matrix: Matrix, transform: (value: number, rowIndex?: number, columnIndex?: number) => number) {
        return matrix.getClone().transform(transform);
    }

    public static transpose (matrix: Matrix) {
        return matrix.getClone().transpose();
    }

    public static zeros (rowCount: number, columnCount: number) {
        const data = new Float64Array(rowCount * columnCount);
        return new Matrix(data, rowCount, columnCount);
    }

    /* Command methods */

    public add (matrix: Matrix): Matrix;
    public add (scalar: number): Matrix;
    public add (operand: Matrix | number) {
        return operand instanceof Matrix ? this.addMatrix(operand) : this.addScalar(operand);
    }

    public appendBottom (matrix: Matrix) {
        const rowCount = this.rowCount;

        if (rowCount === 0) {
            this.data = matrix.data.slice();
            this.rowCount = matrix.rowCount;
            this.columnCount = matrix.columnCount;
            return this;
        }

        const columnCount = this.columnCount;

        if (columnCount !== matrix.columnCount) {
            throw Error('Cannot appendBottom a matrix with ' + matrix.columnCount + ' columns to a matrix with ' + columnCount + ' columns.');
        }

        const newData = new Float64Array((rowCount + matrix.rowCount) * columnCount);

        newData.set(this.data);
        newData.set(matrix.data, rowCount * columnCount);

        this.data = newData;
        this.rowCount = rowCount + matrix.rowCount;

        return this;
    }

    public appendLeft (matrix: Matrix) {
        const rowCount = this.rowCount;

        if (rowCount === 0) {
            this.data = matrix.data.slice();
            this.rowCount = matrix.rowCount;
            this.columnCount = matrix.columnCount;
            return this;
        }

        if (rowCount !== matrix.rowCount) {
            throw Error('Cannot appendLeft a matrix with ' + matrix.rowCount + ' rows to a matrix with ' + rowCount + ' rows.');
        }

        const thisColumnCount = this.columnCount;
        const otherColumnCount = matrix.columnCount;
        
        const data = this.data;
        const otherData = matrix.data;
        const newData = new Float64Array(rowCount * (thisColumnCount + otherColumnCount));

        let thisStart = 0;
        let otherStart = 0;
        let offset = -thisColumnCount;
        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            newData.set(otherData.slice(otherStart, otherStart += otherColumnCount), offset += thisColumnCount);
            newData.set(data.slice(thisStart, thisStart += thisColumnCount), offset += otherColumnCount);
        }

        this.data = newData;
        this.columnCount = thisColumnCount + otherColumnCount;

        return this;
    }

    public appendRight (matrix: Matrix) {
        const rowCount = this.rowCount;

        if (rowCount === 0) {
            this.data = matrix.data.slice();
            this.rowCount = matrix.rowCount;
            this.columnCount = matrix.columnCount;
            return this;
        }

        if (rowCount !== matrix.rowCount) {
            throw Error('Cannot appendRight a matrix with ' + matrix.rowCount + ' rows to a matrix with ' + rowCount + ' rows.');
        }

        const thisColumnCount = this.columnCount;
        const otherColumnCount = matrix.columnCount;
        
        const data = this.data;
        const otherData = matrix.data;
        const newData = new Float64Array(rowCount * (thisColumnCount + otherColumnCount));

        let thisStart = 0;
        let otherStart = 0;
        let offset = -otherColumnCount;
        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            newData.set(data.slice(thisStart, thisStart += thisColumnCount), offset += otherColumnCount);
            newData.set(otherData.slice(otherStart, otherStart += otherColumnCount), offset += thisColumnCount);
        }

        this.data = newData;
        this.columnCount = thisColumnCount + otherColumnCount;

        return this;
    }

    public appendTop (matrix: Matrix) {
        const rowCount = this.rowCount;

        if (rowCount === 0) {
            this.data = matrix.data.slice();
            this.rowCount = matrix.rowCount;
            this.columnCount = matrix.columnCount;
            return this;
        }

        const columnCount = this.columnCount;

        if (columnCount !== matrix.columnCount) {
            throw Error('Cannot appendTop a matrix with ' + matrix.columnCount + ' columns to a matrix with ' + columnCount + ' columns.');
        }

        const newData = new Float64Array((rowCount + matrix.rowCount) * columnCount);

        newData.set(matrix.data);
        newData.set(this.data, matrix.rowCount * columnCount);

        this.data = newData;
        this.rowCount = rowCount + matrix.rowCount;

        return this;
    }

    public deleteRows (rowIndeces: number[]) {
        const newRowCount = this.rowCount - rowIndeces.length;
        const columnCount = this.columnCount;

        const data = new Float64Array(newRowCount * columnCount);

        let offset = 0;
        for (let rowIndex = 0; rowIndex < this.rowCount; rowIndex++) {
            if (rowIndeces.includes(rowIndex)) {
                continue;
            }

            const start = rowIndex * columnCount;
            data.set(this.data.slice(start, start + columnCount), offset);

            offset += columnCount;
        }

        this.data = data;
        this.rowCount = newRowCount;
    }

    public multiply (matrix: Matrix): Matrix;
    public multiply (scalar: number): Matrix;
    public multiply (operand: any) {
        return operand instanceof Matrix ? this.multiplyWithMatrix(operand) : this.multiplyWithScalar(operand);
    }

    public multiplyElementWise (matrix: Matrix) {
        if (this.rowCount !== matrix.rowCount || this.columnCount !== matrix.columnCount) {
            throw Error('Cannot perform element-wise multiplication between ' + this.rowCount + 'x' + this.columnCount + ' matrix and ' + matrix.rowCount + 'x' + matrix.columnCount + ' matrix');
        }

        return this.transform((element, rowIndex, columnIndex) => element * matrix.getElement(rowIndex, columnIndex));
    }

    public removeRows (rowIndeces: number[]) {
        const rows = this.getRows(rowIndeces);

        this.deleteRows(rowIndeces);

        return rows;
    }

    public setElement (rowIndex: number, columnIndex: number, value: number) {
        this.data[rowIndex * this.columnCount + columnIndex] = value;
        return this;
    }

    public subtract (matrix: Matrix): Matrix;
    public subtract (scalar: number): Matrix;
    public subtract (operand: any) {
        return operand instanceof Matrix ? this.subtractMatrix(operand) : this.addScalar(-operand);
    }

    public transform (transform: (value: number, rowIndex?: number, columnIndex?: number) => number) {
        const rowCount = this.getRowCount();
        const columnCount = this.getColumnCount();

        let elementIndex = 0;
        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            for (let columnIndex = 0; columnIndex < columnCount; columnIndex++, elementIndex++) {
                this.data[elementIndex] = transform(this.data[elementIndex], rowIndex, columnIndex);
            }
        }

        return this;
    }

    public transpose () {
        const rowCount = this.getRowCount();
        const columnCount = this.getColumnCount();
        
        const currentData = this.data;
        const newData = new Float64Array(rowCount * columnCount);
        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            for (let columnIndex = 0; columnIndex < columnCount; columnIndex++) {
                newData[columnIndex * rowCount + rowIndex] = currentData[rowIndex * columnCount + columnIndex];
            }
        }

        this.data = newData;
        this.rowCount = columnCount;
        this.columnCount = rowCount;

        return this;
    }

    /* Query methods */

    public getClone () {
        return new Matrix(this.data.slice(), this.rowCount, this.columnCount);
    }

    public getColumn (columnIndex: number) {
        return this.getColumns(columnIndex, columnIndex);
    }

    public getColumns (startColumnIndex: number, endColumnIndex?: number) {
        if (endColumnIndex === undefined) {
            endColumnIndex = this.columnCount - 1;
        }

        const rowCount = this.rowCount;
        const columnCount = this.columnCount;
        const newColumnCount = endColumnIndex - startColumnIndex + 1;

        const data = new Float64Array(rowCount * newColumnCount);

        let elementIndex = 0;
        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            let rowFirstElementIndex = rowIndex * columnCount;
            for (let columnIndex = startColumnIndex; columnIndex <= endColumnIndex; columnIndex++, elementIndex++) {
                data[elementIndex] = this.data[rowFirstElementIndex + columnIndex];
            }
        }

        return new Matrix(data, rowCount, newColumnCount);
    }

    public getColumnCount () {
        return this.columnCount;
    }

    public getData () {
        const columnCount = this.columnCount;
        const size = this.rowCount * columnCount;

        const dataArray = Array.from(this.data);
        const data: number[][] = [];

        for (let start = 0; start < size; start += columnCount) {
            data.push(dataArray.slice(start, start + columnCount));
        }

        return data;
    }

    public getDeterminant (): number {
        const rowCount = this.rowCount;
        const columnCount = this.columnCount;

        if (rowCount !== columnCount) {
            throw Error('Cannot compute determinant of a non-square matrix.');
        }

        if (rowCount === 1) {
            return this.data[0];
        }

        if (rowCount === 2) {
            return this.data[0] * this.data[3] - this.data[1] * this.data[2];
        }

        let sum = 0;
        for (let j = 0; j < columnCount; j++) {
            sum += this.data[j] * this.getCofactor(0, j);
        }

        return sum;
    }

    public getElement (rowIndex: number, columnIndex: number) {
        return this.data[rowIndex * this.columnCount + columnIndex];
    }

    public getInverse (): Matrix {
        const rowCount = this.rowCount;
        const columnCount = this.columnCount;

        if (rowCount !== columnCount) {
            throw new Error('Cannot calculate the inverse of a ' + rowCount + 'x' + 'columnCount matrix. The matrix must be square.');
        }

        const determinant = this.getDeterminant();

        if (determinant === 0) {
            throw new Error('Cannot compute the inverse of a matrix with a zero determinant.');
        }

        return this.getAdjoint().multiply(1 / determinant);
    }

    public getMaximumRowIndeces () {
        const rowCount = this.rowCount;
        const columnCount = this.columnCount;

        const data = new Float64Array(this.rowCount);

        let thisData = this.data;
        let elementIndex = 0;
        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
            let maxValue = Number.NEGATIVE_INFINITY;
            let maxIndex = -1;
            
            for (let columnIndex = 0; columnIndex < columnCount; columnIndex++) {
                const value = thisData[elementIndex++];

                if (value > maxValue) {
                    maxValue = value;
                    maxIndex = columnIndex;
                }
            }

            data[rowIndex] = maxIndex;
        }

        return new Matrix(data, rowCount, 1);
    }

    public getMaximumValue () {
        const thisData = this.data;
        const size = thisData.length;

        let max = Number.NEGATIVE_INFINITY;

        for (let elementIndex = 0; elementIndex < size; elementIndex++) {
            const value = thisData[elementIndex];

            if (value > max) {
                max = value;
            }
        }

        return max;
    }

    public getMinimumValue () {
        const thisData = this.data;
        const size = thisData.length;

        let max = Number.POSITIVE_INFINITY;

        for (let elementIndex = 0; elementIndex < size; elementIndex++) {
            const value = thisData[elementIndex];

            if (value < max) {
                max = value;
            }
        }

        return max;
    }

    public getRow (rowIndex: number) {
        const columnCount = this.columnCount;
        const start = rowIndex * columnCount;
        return new Matrix(this.data.slice(start, start + columnCount), 1, columnCount);
    }

    public getRows (rowIndeces: number[]) {
        const resultRowCount = rowIndeces.length;
        const columnCount = this.columnCount;

        const data = new Float64Array(resultRowCount * columnCount);

        for (let rowIndecesIndex = 0; rowIndecesIndex < resultRowCount; rowIndecesIndex++) {
            const start = rowIndeces[rowIndecesIndex] * columnCount;
            data.set(this.data.slice(start, start + columnCount), rowIndecesIndex * columnCount);
        }

        return new Matrix(data, resultRowCount, columnCount);
    }

    public getRowCount () {
        return this.rowCount;
    }

    public getSum () {
        const thisData = this.data;
        const size = this.data.length;

        let sum = 0;
        for (let elementIndex = 0; elementIndex < size; elementIndex++) {
            sum += thisData[elementIndex];
        }

        return sum;
    }

    /* Helper methods */

    private addMatrix (matrix: Matrix) {
        if (this.rowCount !== matrix.rowCount) {
            throw Error('Cannot add matrices with different number of rows.');
        }

        if (this.columnCount !== matrix.columnCount) {
            throw Error('Cannot add matrices with different number of columns.');
        }

        if (this.blasEnabled) {
            nblas.axpy(matrix.data, this.data, 1);

        } else {
            const size = this.data.length;
            for (let elementIndex = 0; elementIndex < size; elementIndex++) {
                this.data[elementIndex] += matrix.data[elementIndex]
            }
        }

        return this;
    }

    private addScalar (scalar: number) {
        const size = this.data.length;
        for (let elementIndex = 0; elementIndex < size; elementIndex++) {
            this.data[elementIndex] += scalar;
        }

        return this;
    }

    private checkerboard () {
        const rowCount = this.rowCount;
        const columnCount = this.columnCount;

        for (let i = 0; i < rowCount; i++) {
            const rowStartIndex = i * columnCount;

            for (let j = (i % 2 - 1) * -1; j < columnCount; j += 2) {
                this.data[rowStartIndex + j] = -this.data[rowStartIndex + j];
            }
        }


        return this;
    }

    private getAdjoint () {
        return this.getCofactorMatrix().transpose();
    }

    private getCofactor (rowIndex: number, columnIndex: number) {
        return Math.pow(-1, rowIndex + columnIndex) * this.getMinor(rowIndex, columnIndex);
    }

    private getCofactorMatrix () {
        return this.getMatrixOfMinors().checkerboard();
    }

    private getMatrixOfMinors () {
        const rowCount = this.rowCount;
        const columnCount = this.columnCount;

        const matrixOfMinorsData = new Float64Array(rowCount * columnCount);

        for (let i = 0; i < rowCount; i++) {
            const rowStartIndex = i * columnCount;

            for (let j = 0; j < columnCount; j++) {
                matrixOfMinorsData[rowStartIndex + j] = this.getMinor(i, j);
            }
        }

        return new Matrix(matrixOfMinorsData, rowCount, columnCount);
    }

    private getMinor (rowIndex: number, columnIndex: number) {
        const rowCount = this.rowCount;

        if (rowCount === 1) {
            return 1;
        }

        const columnCount = this.columnCount;

        const minorRowCount = this.rowCount - 1;
        const minorColumnCount = this.columnCount - 1;
        const minorData = new Float64Array(minorRowCount * minorColumnCount);

        let dataIndex = 0;
        for (let i = 0; i < rowCount; i++) {
            for (let j = 0; j < columnCount; j++) {
                if (i !== rowIndex && j !== columnIndex) {
                    minorData[dataIndex++] = this.data[i * rowCount + j];
                }
            }
        }

        const minorMatrix = new Matrix(minorData, minorRowCount, minorColumnCount);

        return minorMatrix.getDeterminant();
    }

    private multiplyWithMatrix (other: Matrix) {
        const rowCount = this.rowCount;
        const columnCount = this.columnCount;
        const otherRowCont = other.rowCount;
        const otherColumnCount = other.columnCount;

        if (columnCount !== otherRowCont) {
            throw Error('Cannot scale ' + rowCount + 'x' + columnCount + ' matrix with ' + otherRowCont + 'x' + otherColumnCount + ' matrix.');
        }

        const newData = new Float64Array(rowCount * otherColumnCount);

        if (this.blasEnabled) {
            nblas.gemm(this.data, other.data, newData, rowCount, otherColumnCount, columnCount);

        } else {
            const newData = new Float64Array(rowCount * otherColumnCount);

            let elementIndex = 0;
            for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
                for (let otherColumnIndex = 0; otherColumnIndex < otherColumnCount; otherColumnIndex++) {
                    let sum = 0;
                    for (let columnIndex = 0; columnIndex < columnCount; columnIndex++) {
                        sum += this.getElement(rowIndex, columnIndex) * other.getElement(columnIndex, otherColumnIndex);
                    }
                    newData[elementIndex++] = sum;
                }
            }
        }

        this.columnCount = otherColumnCount;
        this.data = newData;

        return this;
    }

    private multiplyWithScalar (scalar: number) {
        if (this.blasEnabled) {
            nblas.scal(this.data, scalar);
        } else {
            const size = this.data.length;
            for (let elementIndex = 0; elementIndex < size; elementIndex++) {
                this.data[elementIndex] *= scalar;
            }
        }

        return this;
    }

    private subtractMatrix (matrix: Matrix) {
        if (this.rowCount !== matrix.rowCount || this.columnCount !== matrix.columnCount) {
            throw Error('Cannot subtract ' + matrix.rowCount + 'x' + matrix.columnCount + ' matrix from ' + this.rowCount + 'x' + this.columnCount + ' matrix.');
        }

        if (this.blasEnabled) {
            nblas.axpy(matrix.data, this.data, -1);

        } else {
            const size = this.data.length;
            for (let elementIndex = 0; elementIndex < size; elementIndex++) {
                this.data[elementIndex] -= matrix.data[elementIndex]
            }
        }

        return this;
    }
}