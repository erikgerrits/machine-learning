import Matrix from "../../math/linear-algebra/Matrix";

/**
 * **DBSCAN** (density-based spatial clustering of applications with noise) — clustering by *density*
 * instead of distance to a centre. Where k-means and hierarchical clustering force every point into
 * some group, DBSCAN's whole idea is that some points are just **noise**: stragglers and one-offs
 * that belong to no cluster at all.
 *
 * Two knobs define "dense enough": a radius `epsilon` and a count `minPoints`. A point is a **core**
 * point if at least `minPoints` points (itself included) lie within `epsilon`. Clusters grow by
 * chaining through core points — any point within `epsilon` of a core joins its cluster — so a
 * cluster can be any shape that stays dense (curves, rings, blobs). Points reachable from a core but
 * not core themselves are **border** points; points reachable from no core are **noise**.
 *
 * Unlike k-means there is no `k`: the number of clusters falls out of the data. Like the other
 * unsupervised models it is handed inputs with **no targets**. `predict` returns one integer label
 * per row — `0..clusterCount-1` for a cluster, or `-1` for noise; a query point takes the cluster of
 * its nearest core point if that core is within `epsilon`, otherwise noise.
 *
 * @example
 * const dbscan = new DBSCAN().setEpsilon(0.5).setMinPoints(3);
 * dbscan.train(new Matrix([[0, 0], [0.1, 0], [0, 0.1], [9, 9]]));
 * dbscan.getLabels();        // e.g. [0, 0, 0, -1]  ← three dense points, one noise
 * dbscan.getClusterCount();  // 1
 */
export default class DBSCAN {

    /** Label for points that belong to no cluster. */
    public static readonly NOISE = -1;

    private epsilon = 0.5;
    private minPoints = 4;

    private points: number[][] = [];
    private pointCount = 0;
    private labels: number[] = [];
    private coreIndices: number[] = [];
    private clusterCount = 0;

    public constructor () {}

    public train (inputs: Matrix) {
        const points = inputs.toArray();
        const n = points.length;

        this.points = points;
        this.pointCount = n;
        this.clusterCount = 0;
        this.coreIndices = [];

        const UNCLASSIFIED = -2;
        this.labels = new Array<number>(n).fill(UNCLASSIFIED);

        if (n === 0) {
            return this;
        }

        // Pre-compute each point's epsilon-neighbourhood (including itself) and flag the core points.
        const neighbours: number[][] = [];
        const isCore: boolean[] = [];
        for (let i = 0; i < n; i++) {
            const nb: number[] = [];
            for (let j = 0; j < n; j++) {
                if (euclidean(points[i], points[j]) <= this.epsilon) {
                    nb.push(j);
                }
            }
            neighbours.push(nb);
            isCore.push(nb.length >= this.minPoints);
        }

        let clusterId = 0;
        for (let p = 0; p < n; p++) {
            // Only an unclassified *core* point can start a new cluster; non-core points are left to
            // be claimed as borders during expansion (or fall through to noise).
            if (this.labels[p] !== UNCLASSIFIED || !isCore[p]) {
                continue;
            }

            this.labels[p] = clusterId;

            // Flood-fill the cluster: walk the neighbourhood, and wherever it passes through another
            // core point, fold that point's neighbours into the frontier too.
            const seeds = [...neighbours[p]];
            const queued = new Set<number>(seeds);
            for (let s = 0; s < seeds.length; s++) {
                const q = seeds[s];
                if (this.labels[q] !== UNCLASSIFIED) {
                    continue;
                }
                this.labels[q] = clusterId;
                if (isCore[q]) {
                    for (const r of neighbours[q]) {
                        if (!queued.has(r)) {
                            queued.add(r);
                            seeds.push(r);
                        }
                    }
                }
            }

            clusterId++;
        }

        for (let i = 0; i < n; i++) {
            if (this.labels[i] === UNCLASSIFIED) {
                this.labels[i] = DBSCAN.NOISE;
            }
            if (isCore[i]) {
                this.coreIndices.push(i);
            }
        }

        this.clusterCount = clusterId;
        return this;
    }

    /**
     * One integer label per input row: `0..clusterCount-1` for a cluster, or `-1` for noise. A query
     * is assigned the cluster of its nearest core point when that core lies within `epsilon`.
     */
    public predict (inputs: Matrix) {
        const queries = inputs.toArray();
        return new Matrix(queries.map(query => [this.labelFor(query)]));
    }

    /* Parameter setters */

    public setEpsilon (epsilon: number) {
        this.epsilon = epsilon;
        return this;
    }

    public setMinPoints (minPoints: number) {
        this.minPoints = minPoints;
        return this;
    }

    /* Parameter getters */

    public getEpsilon () {
        return this.epsilon;
    }

    public getMinPoints () {
        return this.minPoints;
    }

    /** The cluster label of each training point (`-1` for noise). */
    public getLabels () {
        return this.labels.slice();
    }

    /** How many clusters DBSCAN discovered (noise is not a cluster). */
    public getClusterCount () {
        return this.clusterCount;
    }

    /* Private methods */

    private labelFor (query: number[]) {
        let nearestDistance = Number.POSITIVE_INFINITY;
        let nearestLabel = DBSCAN.NOISE;

        for (const coreIndex of this.coreIndices) {
            const distance = euclidean(query, this.points[coreIndex]);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestLabel = this.labels[coreIndex];
            }
        }

        return nearestDistance <= this.epsilon ? nearestLabel : DBSCAN.NOISE;
    }
}

function euclidean (a: number[], b: number[]) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const difference = a[i] - b[i];
        sum += difference * difference;
    }
    return Math.sqrt(sum);
}
