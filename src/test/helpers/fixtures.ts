// Canonical datasets shared across the model test suites, mirroring examples/demo.ts.
// Keeping them here lets each test assert against the same inputs the demo documents.

/** XNOR (the opposite of XOR) — the classic non-linearly-separable problem. */
export const XNOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
export const XNOR_TARGETS = [[1], [0], [0], [1]];

/** A perfectly linear relationship: y = 1000 + 200 * x. */
export const LINEAR_INPUTS = [[5], [7], [9], [11], [13]];
export const LINEAR_TARGETS = [[2000], [2400], [2800], [3200], [3600]];

/** Binary classification: is the second feature greater than the first? */
export const LOGISTIC_INPUTS = [[1000, 1100], [4500, 3000], [700, 1300], [1150, 700], [1300, 1200], [600, 650]];
export const LOGISTIC_TARGETS = [[1], [0], [1], [0], [0], [1]];
/** Expected class per row (argmax / threshold), independent of saturation. */
export const LOGISTIC_EXPECTED_CLASSES = [1, 0, 1, 0, 0, 1];

/** Multiclass: which of the three features is the largest? (one-hot targets) */
export const MULTICLASS_INPUTS = [[4500, 1200, 3000], [700, 890, 800], [700, 1200, 1300], [1150, 600, 700], [600, 1500, 1650], [400, 401, 400]];
export const MULTICLASS_TARGETS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]];
/** Expected winning class index per row (argmax of the one-hot targets). */
export const MULTICLASS_EXPECTED_CLASSES = [[0], [1], [2], [0], [2], [1]];

/** k-NN with equidistant examples; demo breaks ties via multiple neighbors. */
export const KNN_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [2, 2]];
export const KNN_TARGETS = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]];
export const KNN_QUERIES = [[0.5, 0.5], [1.5, 1.5], [1.75, 1.75]];
export const KNN_EXPECTED = [
    [0.4, 0.2, 0.2, 0.2],
    [0.6666666666666666, 0, 0, 0.3333333333333333],
    [0, 0, 0, 1],
];

/**
 * Multinomial Naive Bayes: a tiny bag-of-words spam filter. Columns are word counts for the
 * vocabulary [free, money, table, tonight]; spam leans on "free money", real bookings on
 * "table tonight". Targets are one-hot [spam, ham].
 */
export const NAIVE_BAYES_INPUTS = [
    [2, 1, 0, 0],
    [1, 2, 0, 0],
    [3, 1, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 1, 2],
    [0, 0, 1, 1],
];
export const NAIVE_BAYES_TARGETS = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]];
/** Expected class per training row (argmax): the first three are spam (0), the rest ham (1). */
export const NAIVE_BAYES_EXPECTED_CLASSES = [[0], [0], [0], [1], [1], [1]];
/** Unseen messages: "free money" → spam, "table tonight" → ham. */
export const NAIVE_BAYES_QUERIES = [[1, 1, 0, 0], [0, 0, 1, 1]];
export const NAIVE_BAYES_QUERY_CLASSES = [[0], [1]];

/**
 * Decision tree: an AND rule — class 1 iff both features are "high" (> 0.5). Cleanly separable by
 * two axis-aligned splits, so a tree of depth ≥ 2 fits it perfectly. Targets are one-hot.
 */
export const TREE_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1], [0.2, 0.2], [0.9, 0.9], [0.1, 0.8], [0.8, 0.1]];
export const TREE_TARGETS = [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0]];
/** Expected class per row (argmax): only the two "both high" rows are class 1. */
export const TREE_EXPECTED_CLASSES = [[0], [0], [0], [1], [0], [1], [0], [0]];

/**
 * Support vector machine — a cleanly separable 2D set: class 1 sits in the upper-right, class 0 in
 * the lower-right's mirror (lower-left). A straight max-margin line (roughly x + y = 0) splits them
 * with room to spare, so a linear-kernel SVM nails every point and only the inner points become
 * support vectors. Targets are a single 0/1 column, like logistic regression.
 */
export const SVM_LINEAR_INPUTS = [[2, 2], [3, 3], [3, 1], [1, 3], [-2, -2], [-3, -3], [-3, -1], [-1, -3]];
export const SVM_LINEAR_TARGETS = [[1], [1], [1], [1], [0], [0], [0], [0]];
export const SVM_LINEAR_EXPECTED_CLASSES = [1, 1, 1, 1, 0, 0, 0, 0];

/**
 * XOR — the textbook non-linearly-separable problem (opposite corners share a class). No straight
 * line can split it, so a linear SVM fails; an RBF kernel carves it cleanly. Reuses {@link
 * XNOR_INPUTS} for the four corners with XOR targets.
 */
export const SVM_XOR_TARGETS = [[0], [1], [1], [0]];
export const SVM_XOR_EXPECTED_CLASSES = [0, 1, 1, 0];

/**
 * Four points on a line for hierarchical clustering: a tight pair at 0–1, a lone point at 3, and a
 * far point at 8. The merge order is unambiguous (0–1 first, then 2 joins, then 3 last), which lets
 * tests pin the dendrogram's shape and its non-decreasing merge heights.
 */
export const HIERARCHICAL_LINE_INPUTS = [[0, 0], [1, 0], [3, 0], [8, 0]];

/**
 * DBSCAN: two tight blobs and one far-flung outlier. With epsilon 0.5 / minPoints 3, each blob is a
 * dense cluster and the lone point at (10, 0) is reachable from nobody — so it's labelled noise (-1).
 */
export const DBSCAN_INPUTS = [
    [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1],
    [5, 5], [5.1, 5], [5, 5.1], [5.1, 5.1],
    [10, 0],
];
/** Expected labels: blob A = cluster 0, blob B = cluster 1, the outlier = noise. */
export const DBSCAN_EXPECTED_LABELS = [0, 0, 0, 0, 1, 1, 1, 1, -1];

/**
 * PCA: five points lying exactly on the line y = x. All the variance is along one diagonal axis, so
 * the first principal component should be ~[0.707, 0.707] and capture ~100% of the variance; the
 * second captures ~0. A clean, fully determined case for the eigendecomposition.
 */
export const PCA_DIAGONAL_INPUTS = [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]];

/** Anomaly detection: a tight, roughly round cloud of "normal" points around the origin. */
export const ANOMALY_NORMAL = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [0.5, -0.5]];

/**
 * Anomaly detection on a correlated cloud (points hug the line y = x). Used to show the Mahalanobis
 * distance respects the data's shape: a point off the diagonal is far more anomalous than one the
 * same Euclidean distance away *along* it.
 */
export const ANOMALY_CORRELATED = [[-2, -2.1], [-1, -0.9], [0, 0.1], [1, 1.1], [2, 1.9], [-1.5, -1.6], [1.5, 1.4], [0.5, 0.6]];

/**
 * Association rules: a tiny basket matrix over 4 items (columns 0–3), 6 baskets. Item 1 almost
 * always rides along with item 0, so "1 → 0" is a confident rule. Supports: 0 in 5/6 baskets,
 * 1 in 4/6, {0,1} in 4/6 — so confidence(1→0)=1.0, confidence(0→1)=0.8.
 */
export const ASSOCIATION_INPUTS = [
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
];

/**
 * Recommender: a 4-user × 4-item ratings matrix with two clear taste groups (0 = not rated yet).
 * Users 0–1 love items 0–1 and dislike 2–3; users 2–3 are the opposite. A few ratings are hidden
 * so matrix factorization has blanks to fill — and the group structure says exactly what they
 * should be (e.g. user 1's hidden item 1 → high; user 3's hidden item 0 → low).
 */
export const RECOMMENDER_INPUTS = [
    [5, 5, 1, 0],
    [5, 0, 1, 1],
    [1, 1, 5, 5],
    [0, 1, 5, 5],
];

/** Time series (one value per time step, as a column): a flat line, a rising trend, a 2-step cycle. */
export const TS_CONSTANT = [[5], [5], [5], [5], [5]];
export const TS_TREND = [[1], [2], [3], [4], [5], [6]];
export const TS_SEASONAL = [[10], [20], [10], [20], [10], [20], [10], [20]];

/** Two well-separated blobs for clustering: three points near the origin, three near (10, 10). */
export const KMEANS_INPUTS = [[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]];
/** The mean of each blob — what the two centroids should converge to. */
export const KMEANS_LOW_BLOB_MEAN = [1 / 3, 1 / 3];
export const KMEANS_HIGH_BLOB_MEAN = [31 / 3, 31 / 3];

/** A fixed seed makes seeded models (Matrix.rand, FeedforwardNeuralNetwork, KMeans) reproducible. */
export const FIXED_SEED = 0;
