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

/** Two well-separated blobs for clustering: three points near the origin, three near (10, 10). */
export const KMEANS_INPUTS = [[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]];
/** The mean of each blob — what the two centroids should converge to. */
export const KMEANS_LOW_BLOB_MEAN = [1 / 3, 1 / 3];
export const KMEANS_HIGH_BLOB_MEAN = [31 / 3, 31 / 3];

/** A fixed seed makes seeded models (Matrix.rand, FeedforwardNeuralNetwork, KMeans) reproducible. */
export const FIXED_SEED = 0;
