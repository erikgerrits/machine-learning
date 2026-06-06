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

/** A fixed seed makes seeded models (Matrix.rand, FeedforwardNeuralNetwork) reproducible. */
export const FIXED_SEED = 0;
