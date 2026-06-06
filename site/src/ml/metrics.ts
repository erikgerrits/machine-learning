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
