import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/**
 * k-means clustering — the library's first *unsupervised* model. Unlike every other algorithm
 * here it is handed inputs with **no targets**: it discovers structure on its own by grouping the
 * points into `k` clusters of nearby neighbours.
 *
 * It runs Lloyd's algorithm. Start with `k` centroids (each a randomly chosen data point), then
 * repeat two steps until the assignments stop changing:
 *   1. **Assign** every point to its nearest centroid (Euclidean distance by default).
 *   2. **Update** each centroid to the mean of the points now assigned to it.
 *
 * Like the iterative supervised models, the centroids are instance state and are NOT reset by
 * `train()`, so calling `setNumberOfIterations(1)` and then `train()` in a loop steps the
 * algorithm one round at a time — handy for watching the centroids migrate in a live visualisation.
 * Pass a seed via {@link setSeed} to make the random initialisation (and therefore the whole
 * clustering) reproducible.
 *
 * @example
 * const kmeans = new KMeans().setNumberOfClusters(2).setSeed(0);
 * kmeans.train(new Matrix([[0, 0], [0, 1], [10, 10], [10, 11]]));
 * kmeans.predict(new Matrix([[0, 0], [10, 10]])); // one-hot membership, e.g. [[1, 0], [0, 1]]
 * kmeans.getCentroids();                          // the two cluster centres
 */
export default class KMeans {

    private distanceFunction = (x: Matrix, y: Matrix) => Matrix.subtract(x, y).transform(value => value * value).getSum();
    private numberOfClusters = 2;
    private numberOfIterations = 100;
    private seed: number = undefined;

    private centroids: Matrix;

    public constructor () {}

    public train (inputs: Matrix) {
        if (this.centroids === undefined) {
            this.centroids = this.initialiseCentroids(inputs);
        }

        for (let iteration = 0; iteration < this.numberOfIterations; iteration++) {
            const assignments = this.assign(inputs);
            const newCentroids = this.recomputeCentroids(inputs, assignments);

            // Stop early once the centroids stop moving — the clustering has converged.
            const converged = this.centroidsEqual(this.centroids, newCentroids);
            this.centroids = newCentroids;

            if (converged) {
                break;
            }
        }
    }

    /**
     * Assigns each input to its nearest centroid, returned as a one-hot membership matrix
     * (one column per cluster). Call `.getMaximumRowIndeces()` on the result for the raw
     * cluster index of each point.
     */
    public predict (inputs: Matrix) {
        const outputs = Matrix.zeros(inputs.getRowCount(), this.numberOfClusters);

        for (let i = 0; i < inputs.getRowCount(); i++) {
            outputs.setElement(i, this.nearestCentroidIndex(inputs.getRow(i)), 1);
        }

        return outputs;
    }

    /* Parameter setters */

    public setDistanceFunction (distanceFunction: (x: Matrix, y: Matrix) => number) {
        this.distanceFunction = distanceFunction;
        return this;
    }

    public setNumberOfClusters (numberOfClusters: number) {
        this.numberOfClusters = numberOfClusters;
        return this;
    }

    public setNumberOfIterations (numberOfIterations: number) {
        this.numberOfIterations = numberOfIterations;
        return this;
    }

    public setSeed (seed: number) {
        this.seed = seed;
        return this;
    }

    public setCentroids (centroids: Matrix) {
        this.centroids = centroids;
        return this;
    }

    public resetCentroids () {
        this.centroids = undefined;
        return this;
    }

    /* Parameter getters */

    public getDistanceFunction () {
        return this.distanceFunction;
    }

    public getNumberOfClusters () {
        return this.numberOfClusters;
    }

    public getNumberOfIterations () {
        return this.numberOfIterations;
    }

    public getSeed () {
        return this.seed;
    }

    public getCentroids () {
        return this.centroids;
    }

    /* Private methods */

    private initialiseCentroids (inputs: Matrix) {
        // Forgy initialisation: pick `k` distinct data points as the starting centroids.
        const random = mulberry32(this.seed !== undefined ? this.seed : Math.floor(Math.random() * 0x100000000));
        const rowCount = inputs.getRowCount();

        const chosen = new Set<number>();
        const centroids = new Matrix([]);

        while (chosen.size < this.numberOfClusters && chosen.size < rowCount) {
            const index = Math.floor(random() * rowCount);

            if (chosen.has(index)) {
                continue;
            }

            chosen.add(index);
            centroids.appendBottom(inputs.getRow(index));
        }

        return centroids;
    }

    private assign (inputs: Matrix) {
        const assignments: number[] = [];

        for (let i = 0; i < inputs.getRowCount(); i++) {
            assignments.push(this.nearestCentroidIndex(inputs.getRow(i)));
        }

        return assignments;
    }

    private nearestCentroidIndex (input: Matrix) {
        let nearestIndex = 0;
        let nearestDistance = Number.POSITIVE_INFINITY;

        for (let cluster = 0; cluster < this.centroids.getRowCount(); cluster++) {
            const distance = this.distanceFunction(input, this.centroids.getRow(cluster));

            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = cluster;
            }
        }

        return nearestIndex;
    }

    private recomputeCentroids (inputs: Matrix, assignments: number[]) {
        const featureCount = inputs.getColumnCount();
        const clusterCount = this.centroids.getRowCount();

        const sums: Matrix[] = [];
        const counts: number[] = [];
        for (let cluster = 0; cluster < clusterCount; cluster++) {
            sums.push(Matrix.zeros(1, featureCount));
            counts.push(0);
        }

        for (let i = 0; i < inputs.getRowCount(); i++) {
            const cluster = assignments[i];
            sums[cluster].add(inputs.getRow(i));
            counts[cluster]++;
        }

        const newCentroids = new Matrix([]);
        for (let cluster = 0; cluster < clusterCount; cluster++) {
            // An empty cluster keeps its previous centroid rather than collapsing to a NaN row.
            const centroid = counts[cluster] === 0 ? this.centroids.getRow(cluster) : sums[cluster].multiply(1 / counts[cluster]);
            newCentroids.appendBottom(centroid);
        }

        return newCentroids;
    }

    private centroidsEqual (a: Matrix, b: Matrix) {
        const aValues = a.toArray().flat();
        const bValues = b.toArray().flat();

        return aValues.every((value, index) => value === bValues[index]);
    }
}
