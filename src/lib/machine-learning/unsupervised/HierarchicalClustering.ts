import Matrix from "../../math/linear-algebra/Matrix";

/** How the distance between two *clusters* is measured from the distances between their members. */
export type Linkage = "single" | "complete" | "average";

/**
 * One merge in the dendrogram: the two nodes that joined, the distance at which they joined, and
 * the size of the resulting cluster. Leaf nodes are ids `0..n-1` (the original points); merge `t`
 * creates internal node id `n + t`. `distance` is non-decreasing across the merge sequence.
 */
export interface HierarchicalMerge {
    left: number;
    right: number;
    distance: number;
    size: number;
}

/**
 * **Agglomerative hierarchical clustering** — the second unsupervised model. Where k-means needs to
 * be told `k` up front and only finds round blobs, this one **grows** the groups bottom-up and
 * defers the choice of how many: start with every point as its own cluster, then repeatedly merge
 * the two **closest** clusters until only one remains. The whole history of merges is a tree — a
 * **dendrogram** — and you read off any number of clusters by slicing it at a height.
 *
 * "Closest" depends on the **linkage**: `single` uses the nearest pair of members (so clusters can
 * chain along curves), `complete` the farthest pair (compact, round-ish clusters), `average` the
 * mean over all member pairs (a balance). Distances are merged efficiently with the Lance–Williams
 * update, so the full tree costs one pass — after which cutting to any `k` is instant.
 *
 * Like {@link KMeans} it is handed inputs with **no targets**. `predict` returns one-hot cluster
 * membership (the tree cut at {@link setNumberOfClusters}); a query point takes the cluster of its
 * nearest training point, so the regions follow the data's shape rather than collapsing to round cells.
 *
 * @example
 * const hc = new HierarchicalClustering().setNumberOfClusters(2).setLinkage('average');
 * hc.train(new Matrix([[0, 0], [0, 1], [10, 10], [10, 11]]));
 * hc.predict(new Matrix([[0, 0], [10, 10]])); // one-hot membership, e.g. [[1, 0], [0, 1]]
 * hc.getMergeHistory();                        // the dendrogram, as a list of merges
 */
export default class HierarchicalClustering {

    private numberOfClusters = 2;
    private linkage: Linkage = "average";

    private points: number[][] = [];
    private pointCount = 0;
    private merges: Merge[] = []; // length n-1, in merge order (ascending distance)

    public constructor () {}

    public train (inputs: Matrix) {
        const points = inputs.toArray();
        const n = points.length;

        this.points = points;
        this.pointCount = n;
        this.merges = [];

        if (n < 2) {
            return this;
        }

        // Each point starts as its own active cluster. We track, per active cluster id: its size,
        // a representative original point (for the union-find cut later), and its distance to every
        // other active cluster.
        const active = new Set<number>();
        const size = new Map<number, number>();
        const representative = new Map<number, number>();
        const distances = new Map<number, Map<number, number>>();

        for (let i = 0; i < n; i++) {
            active.add(i);
            size.set(i, 1);
            representative.set(i, i);
            distances.set(i, new Map());
        }
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const distance = euclidean(points[i], points[j]);
                distances.get(i)!.set(j, distance);
                distances.get(j)!.set(i, distance);
            }
        }

        let nextId = n;
        while (active.size > 1) {
            const [a, b, distance] = this.closestPair(active, distances);
            const newId = nextId++;
            const sizeA = size.get(a)!;
            const sizeB = size.get(b)!;

            this.merges.push({
                left: a,
                right: b,
                distance,
                size: sizeA + sizeB,
                representativeA: representative.get(a)!,
                representativeB: representative.get(b)!,
            });

            // Lance–Williams: the merged cluster's distance to each other cluster is a simple
            // combination of the two old distances, so we never revisit the raw points.
            const newDistances = new Map<number, number>();
            distances.set(newId, newDistances);
            for (const c of active) {
                if (c === a || c === b) {
                    continue;
                }
                const merged = this.linkageDistance(distances.get(a)!.get(c)!, distances.get(b)!.get(c)!, sizeA, sizeB);
                newDistances.set(c, merged);
                distances.get(c)!.set(newId, merged);
            }

            size.set(newId, sizeA + sizeB);
            representative.set(newId, representative.get(a)!);

            active.delete(a);
            active.delete(b);
            active.add(newId);
            distances.delete(a);
            distances.delete(b);
            for (const c of active) {
                if (c !== newId) {
                    distances.get(c)!.delete(a);
                    distances.get(c)!.delete(b);
                }
            }
        }

        return this;
    }

    /**
     * One-hot cluster membership (one column per cluster), the tree cut at `numberOfClusters`.
     * Arbitrary query points are assigned the cluster of their nearest training point.
     */
    public predict (inputs: Matrix) {
        const labels = this.clusterLabels(this.numberOfClusters);
        const queries = inputs.toArray();

        const outputs = Matrix.zeros(queries.length, this.numberOfClusters);
        for (let i = 0; i < queries.length; i++) {
            const nearest = this.nearestTrainingIndex(queries[i]);
            outputs.setElement(i, labels[nearest], 1);
        }
        return outputs;
    }

    /* Parameter setters */

    public setNumberOfClusters (numberOfClusters: number) {
        this.numberOfClusters = numberOfClusters;
        return this;
    }

    public setLinkage (linkage: Linkage) {
        this.linkage = linkage;
        return this;
    }

    /* Parameter getters */

    public getNumberOfClusters () {
        return this.numberOfClusters;
    }

    public getLinkage () {
        return this.linkage;
    }

    /** The cluster index of each training point, for the tree cut at `numberOfClusters`. */
    public getClusterLabels () {
        return this.clusterLabels(this.numberOfClusters);
    }

    /** The full merge sequence (the dendrogram), in ascending order of merge distance. */
    public getMergeHistory (): HierarchicalMerge[] {
        return this.merges.map(merge => ({ left: merge.left, right: merge.right, distance: merge.distance, size: merge.size }));
    }

    /* Private methods */

    /** The two active clusters separated by the smallest linkage distance. */
    private closestPair (active: Set<number>, distances: Map<number, Map<number, number>>): [number, number, number] {
        const ids = [...active];
        let bestA = ids[0];
        let bestB = ids[1];
        let best = Number.POSITIVE_INFINITY;

        for (let x = 0; x < ids.length; x++) {
            for (let y = x + 1; y < ids.length; y++) {
                const distance = distances.get(ids[x])!.get(ids[y])!;
                if (distance < best) {
                    best = distance;
                    bestA = ids[x];
                    bestB = ids[y];
                }
            }
        }

        return [bestA, bestB, best];
    }

    private linkageDistance (distanceAC: number, distanceBC: number, sizeA: number, sizeB: number) {
        switch (this.linkage) {
            case "single":
                return Math.min(distanceAC, distanceBC);
            case "complete":
                return Math.max(distanceAC, distanceBC);
            case "average":
            default:
                // UPGMA: weight each side by its cluster size so it tracks the true mean over pairs.
                return (sizeA * distanceAC + sizeB * distanceBC) / (sizeA + sizeB);
        }
    }

    /**
     * Cuts the dendrogram to `k` clusters and labels every training point. Replaying the first
     * `n − k` merges with a union-find leaves exactly `k` connected components; they are relabelled
     * `0..k-1` in order of first appearance (stable colours).
     */
    private clusterLabels (k: number) {
        const n = this.pointCount;
        const parent = Array.from({ length: n }, (_, i) => i);

        const find = (x: number): number => {
            while (parent[x] !== x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        };

        const mergeCount = Math.min(this.merges.length, Math.max(0, n - k));
        for (let t = 0; t < mergeCount; t++) {
            parent[find(this.merges[t].representativeA)] = find(this.merges[t].representativeB);
        }

        const labelByRoot = new Map<number, number>();
        let next = 0;
        const labels = new Array<number>(n);
        for (let i = 0; i < n; i++) {
            const root = find(i);
            if (!labelByRoot.has(root)) {
                labelByRoot.set(root, next++);
            }
            labels[i] = labelByRoot.get(root)!;
        }
        return labels;
    }

    private nearestTrainingIndex (query: number[]) {
        let nearestIndex = 0;
        let nearestDistance = Number.POSITIVE_INFINITY;

        for (let i = 0; i < this.points.length; i++) {
            const distance = euclidean(query, this.points[i]);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = i;
            }
        }
        return nearestIndex;
    }
}

/** Internal merge record — the public {@link HierarchicalMerge} plus the union-find representatives. */
interface Merge extends HierarchicalMerge {
    representativeA: number;
    representativeB: number;
}

function euclidean (a: number[], b: number[]) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const difference = a[i] - b[i];
        sum += difference * difference;
    }
    return Math.sqrt(sum);
}
