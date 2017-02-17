import Matrix from "../../math/linear-algebra/Matrix";

export default class NearestNeighbors {

    private distanceFunction = (x: Matrix, y: Matrix) => Matrix.subtract(x, y).transform(value => value * value).getSum();
    private numberOfNeighbors = 1;

    private inputs: Matrix;
    private targets: Matrix;

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public predict (inputs: Matrix) {
        const outputs = new Matrix([]);

        for (let i = 0; i < inputs.getRowCount(); i++) {
            outputs.appendBottom(this.predictOne(inputs.getRow(i)));
        }

        return outputs;
    }

    /* Parameter setters */

    public setDistanceFunction (distanceFunction: (x: Matrix, y: Matrix) => number) {
        this.distanceFunction = distanceFunction;
    }

    public setNumberOfNeighbors (numberOfNeighbors: number) {
        this.numberOfNeighbors = numberOfNeighbors;
    }

    /* Parameter getters */

    public getDistanceFunction () {
        return this.distanceFunction;
    }

    public getNumberOfNeighbors () {
        return this.numberOfNeighbors;
    }

    /* Private methods */

    private predictOne (input: Matrix) {
        let neighbors: Neighbor[] = [];
        let furthestNeighborDistance = 0;

        for (let exampleIndex = 0; exampleIndex < this.inputs.getRowCount(); exampleIndex++) {
            const exampleInput = this.inputs.getRow(exampleIndex);
            const exampleOutput = this.targets.getRow(exampleIndex);

            const distance = this.distanceFunction(input, exampleInput);

            if (exampleIndex < this.numberOfNeighbors) {
                neighbors.push(new Neighbor(distance, exampleOutput));

                if (distance > furthestNeighborDistance) {
                    furthestNeighborDistance = distance;
                }

                continue;
            }

            if (distance > furthestNeighborDistance) {
                continue;
            }

            if (neighbors.length >= this.numberOfNeighbors) {
                neighbors = neighbors.filter(neighbor => neighbor.getDistance() <= distance);
            }

            neighbors.push(new Neighbor(distance, exampleOutput));

            furthestNeighborDistance = neighbors.reduce((furthestDistance, neighbor) => neighbor.getDistance() > furthestDistance ? neighbor.getDistance() : furthestDistance, 0);
        }

        return neighbors.reduce((prediction, neighbor) => prediction.add(neighbor.getOutput()), Matrix.zeros(1, neighbors[0].getOutput().getColumnCount())).multiply(1 / neighbors.length);
    }
}

class Neighbor {

    public constructor (private distance: number, private output: Matrix) {}

    public getDistance () {
        return this.distance;
    }

    public getOutput () {
        return this.output;
    }
}