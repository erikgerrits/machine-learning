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
