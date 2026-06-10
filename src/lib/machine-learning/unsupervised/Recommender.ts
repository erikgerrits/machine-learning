import Matrix from "../../math/linear-algebra/Matrix";
import mulberry32 from "../../math/random/mulberry32";

/**
 * **Recommender** by matrix factorization — collaborative filtering, the "customers like you also
 * liked…" model. Association rules found patterns true of the *average* customer; this learns each
 * customer (and each item) individually, so it can suggest what *one specific person* would love but
 * hasn't tried.
 *
 * The input is a sparse **ratings matrix**: one row per user, one column per item, the rating where
 * it's known and `0` where it isn't (the gaps are the whole point — they're what we want to fill).
 * The model explains the known ratings as a product of two skinny matrices, `R ≈ globalMean + U Vᵀ`:
 * each user gets a short vector of **latent factors** (their taste — how much they lean "coffee vs
 * tea", "sweet vs savoury", whatever the data implies) and each item gets one too (how much it
 * embodies each factor). A predicted rating is just how well a user's taste lines up with an item's
 * profile — the dot product — and *that* fills the blanks for items nobody told us about.
 *
 * Training is plain SGD over the observed entries (with L2 regularisation), so — like the regression
 * models — repeated `train()` calls continue from where they left off. `predict()` returns the full
 * completed ratings matrix; {@link recommend} ranks a user's unrated items by predicted score.
 *
 * @example
 * const reco = new Recommender().setNumberOfFactors(2).setNumberOfEpochs(300);
 * reco.train(new Matrix([[5, 5, 1, 0], [5, 0, 1, 1], [1, 1, 5, 5], [0, 1, 5, 5]]));
 * reco.recommend(1); // user 1's unrated items, best first — predicts they'll love item 1
 */
export default class Recommender {

    private numberOfFactors = 2;
    private numberOfEpochs = 100;
    private learningRate = 0.01;
    private regularization = 0.05;
    private seed = 0;

    private userFactors: number[][];
    private itemFactors: number[][];
    private globalMean = 0;
    private ratings: number[][] = [];

    public constructor () {}

    public train (inputs: Matrix) {
        const ratings = inputs.toArray();
        const userCount = ratings.length;
        const itemCount = userCount > 0 ? ratings[0].length : 0;
        this.ratings = ratings;

        // Observed entries only (0 means "not rated yet" — those are what we're trying to predict).
        const observed: [number, number, number][] = [];
        for (let u = 0; u < userCount; u++) {
            for (let i = 0; i < itemCount; i++) {
                if (ratings[u][i] !== 0) {
                    observed.push([u, i, ratings[u][i]]);
                }
            }
        }

        // Initialise factors (and the global mean) on the first train, like the regression models'
        // hypothesis — so calling train() repeatedly continues the same optimisation.
        if (this.userFactors === undefined || this.userFactors.length !== userCount || this.itemFactors.length !== itemCount) {
            const random = mulberry32(this.seed);
            this.userFactors = randomFactors(userCount, this.numberOfFactors, random);
            this.itemFactors = randomFactors(itemCount, this.numberOfFactors, random);
            this.globalMean = observed.length > 0 ? observed.reduce((sum, [, , r]) => sum + r, 0) / observed.length : 0;
        }

        const k = this.numberOfFactors;
        const lr = this.learningRate;
        const reg = this.regularization;

        for (let epoch = 0; epoch < this.numberOfEpochs; epoch++) {
            for (const [u, i, rating] of observed) {
                const userVector = this.userFactors[u];
                const itemVector = this.itemFactors[i];

                const error = rating - this.globalMean - dot(userVector, itemVector);

                // Step both vectors toward the residual, shrinking them a little (L2) as we go.
                for (let f = 0; f < k; f++) {
                    const uf = userVector[f];
                    const vf = itemVector[f];
                    userVector[f] += lr * (error * vf - reg * uf);
                    itemVector[f] += lr * (error * uf - reg * vf);
                }
            }
        }

        return this;
    }

    /** The full completed ratings matrix: a predicted score for every user × item, gaps included. */
    public predict () {
        return new Matrix(this.userFactors.map(userVector =>
            this.itemFactors.map(itemVector => this.globalMean + dot(userVector, itemVector)),
        ));
    }

    /**
     * A user's unrated items, ranked best-first by predicted score. `count` caps the list (default:
     * all unrated items).
     */
    public recommend (userIndex: number, count?: number) {
        const userVector = this.userFactors[userIndex];
        const scored: { item: number; score: number }[] = [];
        for (let i = 0; i < this.itemFactors.length; i++) {
            if (this.ratings[userIndex][i] === 0) {
                scored.push({ item: i, score: this.globalMean + dot(userVector, this.itemFactors[i]) });
            }
        }
        scored.sort((a, b) => b.score - a.score);
        return count === undefined ? scored : scored.slice(0, count);
    }

    public reset () {
        this.userFactors = undefined;
        this.itemFactors = undefined;
        this.globalMean = 0;
        return this;
    }

    /* Parameter setters */

    public setNumberOfFactors (numberOfFactors: number) {
        this.numberOfFactors = numberOfFactors;
        return this;
    }

    public setNumberOfEpochs (numberOfEpochs: number) {
        this.numberOfEpochs = numberOfEpochs;
        return this;
    }

    public setLearningRate (learningRate: number) {
        this.learningRate = learningRate;
        return this;
    }

    public setRegularization (regularization: number) {
        this.regularization = regularization;
        return this;
    }

    public setSeed (seed: number) {
        this.seed = seed;
        return this;
    }

    /* Parameter getters */

    public getNumberOfFactors () {
        return this.numberOfFactors;
    }

    public getNumberOfEpochs () {
        return this.numberOfEpochs;
    }

    public getLearningRate () {
        return this.learningRate;
    }

    public getRegularization () {
        return this.regularization;
    }

    public getSeed () {
        return this.seed;
    }

    /** The learned latent vector for each user (users × factors). */
    public getUserFactors () {
        return new Matrix(this.userFactors.map(row => row.slice()));
    }

    /** The learned latent vector for each item (items × factors). */
    public getItemFactors () {
        return new Matrix(this.itemFactors.map(row => row.slice()));
    }

    public getGlobalMean () {
        return this.globalMean;
    }
}

function randomFactors (rows: number, factors: number, random: () => number) {
    const matrix: number[][] = [];
    for (let r = 0; r < rows; r++) {
        const row = new Array<number>(factors);
        for (let f = 0; f < factors; f++) {
            row[f] = (random() * 2 - 1) * 0.1;
        }
        matrix.push(row);
    }
    return matrix;
}

function dot (a: number[], b: number[]) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
