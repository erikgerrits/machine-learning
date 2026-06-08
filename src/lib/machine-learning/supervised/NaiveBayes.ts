import Matrix from "../../math/linear-algebra/Matrix";

/**
 * Multinomial Naive Bayes — the classic text classifier. Each input row is a vector of feature
 * **counts** (for text, how often each vocabulary word appears in a document) and each target row
 * is a **one-hot** class label, one column per class — the same convention as
 * {@link MulticlassLogisticRegression}.
 *
 * "Training" is a single counting pass: how often each class occurs (the priors) and how often
 * each feature occurs within each class (the likelihoods). Prediction applies **Bayes' rule**
 * under the "naive" assumption that features are independent given the class, so a class's score
 * is its prior times the per-feature likelihoods. The work is done in log-space to avoid numeric
 * underflow, with **Laplace (add-α) smoothing** so a single unseen word can never zero out a whole
 * class. `predict` returns one normalised posterior probability per class (each row sums to 1);
 * the predicted class is the argmax across the columns.
 */
export default class NaiveBayes {

    private smoothing = 1; // Laplace / add-alpha smoothing

    private logPriors: number[] = [];        // log P(class) per class
    private logLikelihoods: number[][] = []; // log P(feature | class), indexed [class][feature]

    public constructor () {}

    public train (inputs: Matrix, targets: Matrix) {
        const documents = inputs.toArray();
        const labels = targets.toArray();
        const documentCount = documents.length;
        const featureCount = inputs.getColumnCount();
        const classCount = targets.getColumnCount();

        // Tally, per class, how many documents it has and the total count of each feature.
        const documentsPerClass = new Array(classCount).fill(0);
        const featureCountsPerClass = Array.from({ length: classCount }, () => new Array(featureCount).fill(0));

        for (let i = 0; i < documentCount; i++) {
            const classIndex = argmax(labels[i]);
            documentsPerClass[classIndex]++;
            for (let f = 0; f < featureCount; f++) {
                featureCountsPerClass[classIndex][f] += documents[i][f];
            }
        }

        this.logPriors = documentsPerClass.map(count => Math.log(count / documentCount));

        // P(feature | class) with add-α smoothing: (count + α) / (classTotal + α · vocabularySize).
        this.logLikelihoods = featureCountsPerClass.map(featureCounts => {
            const classTotal = featureCounts.reduce((sum, count) => sum + count, 0);
            const denominator = classTotal + this.smoothing * featureCount;
            return featureCounts.map(count => Math.log((count + this.smoothing) / denominator));
        });
    }

    public predict (inputs: Matrix) {
        const posteriors = inputs.toArray().map(document => {
            // Score per class = log P(class) + Σ count_f · log P(feature_f | class).
            const logScores = this.logPriors.map((logPrior, classIndex) => {
                const likelihoods = this.logLikelihoods[classIndex];
                let score = logPrior;
                for (let f = 0; f < document.length; f++) {
                    score += document[f] * likelihoods[f];
                }
                return score;
            });

            return normaliseLogScores(logScores);
        });

        return new Matrix(posteriors);
    }

    /* Parameter setters */

    public setSmoothing (smoothing: number) {
        this.smoothing = smoothing;
        return this;
    }

    /* Parameter getters */

    public getSmoothing () {
        return this.smoothing;
    }

    public getLogPriors () {
        return this.logPriors;
    }

    public getLogLikelihoods () {
        return this.logLikelihoods;
    }
}

function argmax (values: number[]) {
    let bestIndex = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[bestIndex]) {
            bestIndex = i;
        }
    }
    return bestIndex;
}

/** Turn unnormalised log-scores into a probability distribution (log-sum-exp, for stability). */
function normaliseLogScores (logScores: number[]) {
    const maximum = Math.max(...logScores);
    const exponentials = logScores.map(score => Math.exp(score - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map(value => value / total);
}
