/**
 * Fraction of examples classified correctly, thresholding the network's sigmoid output at 0.5.
 * (The library doesn't expose accuracy — it's a presentation concern, computed here.)
 */
export function accuracy(predictions: number[][], targets: number[][]): number {
    if (predictions.length === 0) return 0;

    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
        const predictedClass = predictions[i][0] >= 0.5 ? 1 : 0;
        if (predictedClass === targets[i][0]) correct++;
    }
    return correct / predictions.length;
}

/**
 * Mean squared error — the average squared gap between prediction and target. This is what
 * linear regression minimises; the library's Regression models don't expose it, so we compute
 * it here for the loss curve.
 */
export function mse(predictions: number[][], targets: number[][]): number {
    if (predictions.length === 0) return 0;

    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        const error = predictions[i][0] - targets[i][0];
        sum += error * error;
    }
    return sum / predictions.length;
}

/**
 * Average binary cross-entropy between sigmoid predictions and 0/1 targets — the loss logistic
 * regression minimises. Predictions are clamped away from 0 and 1 so log() stays finite.
 */
export function crossEntropy(predictions: number[][], targets: number[][]): number {
    if (predictions.length === 0) return 0;

    const eps = 1e-7;
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        const p = Math.min(1 - eps, Math.max(eps, predictions[i][0]));
        const y = targets[i][0];
        sum -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
    }
    return sum / predictions.length;
}

/** Mean per-example cross-entropy summed across all class columns (one-vs-rest). */
export function crossEntropyMulticlass(predictions: number[][], targets: number[][]): number {
    if (predictions.length === 0) return 0;

    const eps = 1e-7;
    const classes = predictions[0].length;
    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        for (let c = 0; c < classes; c++) {
            const p = Math.min(1 - eps, Math.max(eps, predictions[i][c]));
            const y = targets[i][c];
            sum -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
        }
    }
    return sum / predictions.length;
}

/** Argmax-match accuracy: fraction of rows whose predicted class equals the target class. */
export function argmaxAccuracy(predictedClass: number[], trueClass: number[]): number {
    if (predictedClass.length === 0) return 0;

    let correct = 0;
    for (let i = 0; i < predictedClass.length; i++) {
        if (predictedClass[i] === trueClass[i]) correct++;
    }
    return correct / predictedClass.length;
}
