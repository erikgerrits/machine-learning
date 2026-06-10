# machine-learning

[![CI](https://github.com/erikgerrits/machine-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/erikgerrits/machine-learning/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/machine-learning.svg)](https://www.npmjs.com/package/machine-learning)
[![license](https://img.shields.io/npm/l/machine-learning.svg)](./LICENSE)

A small, dependency-free TypeScript machine learning library — built to be **read, understood, and watched**.

## 🚀 Interactive playground

**[Open the live playground & tutorials →](https://erikgerrits.github.io/machine-learning/)**

Train each model live in your browser and watch the decision boundary form, the loss fall, and
the network's weights pulse in real time. Each algorithm comes with a tutorial that maps the math
straight onto the library's source. (The site lives in [`site/`](./site) and deploys to GitHub Pages.)

## Important notes
This library is in an early development phase and many **breaking changes are to be expected**.

The TypeScript source files can be found on [GitHub](https://github.com/erikgerrits/machine-learning) and the JavaScript production files(including .ts.d files) can be found as an [npm package](https://www.npmjs.com/package/machine-learning).

## Documentation

Below are some simple code usage examples.

* [Feedforward Neural Network](#feedforward-neural-network)
* [Linear Regression](#linear-regression)
* [Logistic Regression](#logistic-regression)
* [Multiclass Logistic Regression](#multiclass-logistic-regression)
* [Nearest Neighbors](#nearest-neighbors)
* [Perceptron](#perceptron)
* [Naive Bayes](#naive-bayes)
* [Decision Tree](#decision-tree)
* [Random Forest](#random-forest)
* [Gradient Boosting](#gradient-boosting)
* [Support Vector Machine](#support-vector-machine)
* [Convolutional Neural Network](#convolutional-neural-network)
* [Recurrent Neural Network](#recurrent-neural-network)
* [Transformer](#transformer)
* [k-Means Clustering](#k-means-clustering)
* [Hierarchical Clustering](#hierarchical-clustering)
* [DBSCAN](#dbscan)
* [PCA](#pca)
* [Anomaly Detection](#anomaly-detection)
* [Association Rules](#association-rules)
* [Recommender](#recommender)
* [Exponential Smoothing (time series)](#exponential-smoothing-time-series)
* [Multi-Armed Bandit (reinforcement learning)](#multi-armed-bandit-reinforcement-learning)

### Feedforward Neural Network
```TypeScript
import * as ml from 'machine-learning';

// Feedforward neural network: solve XNOR problem (opposite of XOR)
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);
const targets = new ml.Matrix([[1], [0], [0], [1]]);

const feedforwardNeuralNetwork = new ml.FeedforwardNeuralNetwork([2, 5, 1], 0);
feedforwardNeuralNetwork.setNumberOfEpochs(1000);
feedforwardNeuralNetwork.setLearningRate(1);

feedforwardNeuralNetwork.train(inputs, targets);
const predictions = feedforwardNeuralNetwork.predict(inputs);
console.log(predictions.toArray());
// [ [ 0.9943559154265011 ], [ 0.012148393118769857 ], [ 0.013640408487437417 ], [ 0.9816837627444868 ] ]

```

### Linear Regression

```TypeScript
import * as ml from 'machine-learning';

// Linear Regression: y = 1000 + 200 * x
const inputs = new ml.Matrix([[5], [7], [9], [11], [13]]);
const targets = new ml.Matrix([[2000], [2400], [2800], [3200], [3600]]);

const linearRegression = new ml.LinearRegression();
linearRegression.setNumberOfEpochs(10000);
linearRegression.setLearningRate(0.02);

linearRegression.train(inputs, targets);
const predictions = linearRegression.predict(inputs);
console.log(predictions.toArray());
// [ [ 1999.999991189672 ], [ 2399.9999948012005 ], [ 2799.999998412729 ], [ 3200.0000020242574 ], [ 3600.000005635786 ] ]

```

### Logistic Regression
```TypeScript
import * as ml from 'machine-learning';

// Logistic Regression: determine if second input is higher than first input
const inputs = new ml.Matrix([[1000, 1100], [4500, 3000], [700, 1300], [1150, 700], [1300, 1200], [600, 650]]);
const targets = new ml.Matrix([[1], [0], [1], [0], [0], [1]]);

const logisticRegression = new ml.LogisticRegression();
logisticRegression.setNumberOfEpochs(1000);
logisticRegression.setLearningRate(0.01);

logisticRegression.train(inputs, targets);
const predictions = logisticRegression.predict(inputs);
console.log(predictions.toArray());
// [ [ 1 ], [ 0 ], [ 1 ], [ 0 ], [ 0 ], [ 1 ] ]

```

### Multiclass Logistic Regression
```TypeScript
import * as ml from 'machine-learning';

// Multiclass Logistic Regression: determine the highest value
const inputs = new ml.Matrix([[4500, 1200, 3000], [700, 890, 800], [700, 1200, 1300], [1150, 600, 700], [600, 1500, 1650], [400, 401, 400]]);
const targets = new ml.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]);

const multiclassLogisticRegression = new ml.MulticlassLogisticRegression();
multiclassLogisticRegression.setNumberOfEpochs(10000);
multiclassLogisticRegression.setLearningRate(0.1);

multiclassLogisticRegression.train(inputs, targets);
const predictions = multiclassLogisticRegression.predict(inputs);
console.log(predictions.toArray());
// [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ]

```

### Nearest Neighbors
```TypeScript
import * as ml from 'machine-learning';

// Nearest neighbors: Equidistant examples, breaks ties by considering multiple neighbors even though number set to 1
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [2, 2]]);
const targets = new ml.Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]);

const nearestNeighbors = new ml.NearestNeighbors();
nearestNeighbors.setNumberOfNeighbors(1);

nearestNeighbors.train(inputs, targets);

const unknowns = new ml.Matrix([[0.5, 0.5], [1.5, 1.5], [1.75, 1.75]]);

const predictions = nearestNeighbors.predict(unknowns);
console.log(predictions.toArray());
// [ [ 0.4, 0.2, 0.2, 0.2 ], [ 0.6666666666666666, 0, 0, 0.3333333333333333 ], [ 0, 0, 0, 1 ] ]

```

### Perceptron

```TypeScript
import * as ml from 'machine-learning';

// Perceptron (a single neuron): learns AND, but one straight line can't solve XOR.
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);

const perceptron = new ml.Perceptron();
perceptron.setLearningRate(0.1).setNumberOfEpochs(100);

perceptron.train(inputs, new ml.Matrix([[0], [0], [0], [1]])); // AND
console.log(perceptron.predict(inputs).toArray());
// [ [ 0 ], [ 0 ], [ 0 ], [ 1 ] ]  <- solves AND (linearly separable)

perceptron.reset();
perceptron.train(inputs, new ml.Matrix([[0], [1], [1], [0]])); // XOR
console.log(perceptron.predict(inputs).toArray());
// [ [ 1 ], [ 1 ], [ 0 ], [ 0 ] ] != [0,1,1,0]  <- one neuron is one line; XOR needs a layered network

```

### Naive Bayes

```TypeScript
import * as ml from 'machine-learning';

// Naive Bayes (multinomial): classify messages by word counts.
// Vocabulary [free, money, table, tonight]; one-hot classes [spam, ham].
const inputs = new ml.Matrix([[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]]);
const targets = new ml.Matrix([[1, 0], [1, 0], [0, 1], [0, 1]]);

const naiveBayes = new ml.NaiveBayes();
naiveBayes.train(inputs, targets);

const unknowns = new ml.Matrix([[1, 1, 0, 0], [0, 0, 1, 1]]); // "free money" / "table tonight"
const predictions = naiveBayes.predict(unknowns);
console.log(predictions.getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 1 ] ]  ← spam, then ham

```

### Decision Tree

```TypeScript
import * as ml from 'machine-learning';

// Decision tree: class 1 iff both features are "high" (an AND rule).
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.8, 0.1]]);
const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]);

const decisionTree = new ml.DecisionTree();
decisionTree.setMaxDepth(3);

decisionTree.train(inputs, targets);
const predictions = decisionTree.predict(inputs);
console.log(predictions.getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]  ← only the "both high" rows are class 1

```

### Random Forest

```TypeScript
import * as ml from 'machine-learning';

// Random forest: a committee of trees over bootstrap samples, votes averaged.
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.8, 0.1]]);
const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]);

const randomForest = new ml.RandomForest();
randomForest.setNumberOfTrees(30).setMaxDepth(3).setSeed(0);

randomForest.train(inputs, targets);
const predictions = randomForest.predict(inputs);
console.log(predictions.getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]

```

### Gradient Boosting

```TypeScript
import * as ml from 'machine-learning';

// Gradient boosting: trees built in sequence, each fitting the leftover error.
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.8, 0.1]]);
const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]);

const gradientBoosting = new ml.GradientBoosting();
gradientBoosting.setNumberOfTrees(60).setLearningRate(0.3).setMinSamplesSplit(2);

gradientBoosting.train(inputs, targets);
const predictions = gradientBoosting.predict(inputs);
console.log(predictions.getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]

```

### Support Vector Machine

```TypeScript
import * as ml from 'machine-learning';

// Support vector machine: the widest-margin boundary between two classes.
// A linear kernel draws a straight max-margin line; .setKernel('rbf') carves curves (the kernel trick).
const inputs = new ml.Matrix([[2, 2], [3, 3], [3, 1], [1, 3], [-2, -2], [-3, -3], [-3, -1], [-1, -3]]);
const targets = new ml.Matrix([[1], [1], [1], [1], [0], [0], [0], [0]]);

const supportVectorMachine = new ml.SupportVectorMachine();
supportVectorMachine.setKernel('linear').setRegularization(10).setNumberOfIterations(50);

supportVectorMachine.train(inputs, targets);

// predict returns the raw decision score per row; its sign is the class (>= 0 -> class 1).
const predictions = supportVectorMachine.predict(inputs);
console.log(predictions.transform(score => (score >= 0 ? 1 : 0)).toArray());
// [ [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ] ]

console.log(supportVectorMachine.getSupportVectorIndices());
// [ 0, 2, 3, 4, 6 ]  <- the boundary-hugging points the line balances on (the rest could be deleted)

```

### Convolutional Neural Network

```TypeScript
import * as ml from 'machine-learning';

// CNN: conv -> ReLU -> max-pool -> dense -> softmax, trained from scratch by backprop.
// 8x8 images, a horizontal line (class 0) or a vertical line (class 1) at various positions.
const size = 8;
const line = (orientation: 'h' | 'v', pos: number) => {
    const image = new Array(size * size).fill(0);
    for (let k = 0; k < size; k++) image[orientation === 'h' ? pos * size + k : k * size + pos] = 1;
    return image;
};
const images = new ml.Matrix([line('h', 1), line('h', 4), line('h', 6), line('v', 1), line('v', 4), line('v', 6)]);
const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);

const cnn = new ml.ConvolutionalNeuralNetwork();
cnn.setInputShape(8, 8).setFilterCount(4).setLearningRate(0.3).setNumberOfEpochs(300).setSeed(0);

cnn.train(images, targets);

console.log(cnn.predict(images).getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 1 ] ]  <- horizontals -> class 0, verticals -> class 1

console.log('gradients verified:', cnn.checkGradients());
// gradients verified: true  <- finite-difference check of the convolution backprop

```

### Recurrent Neural Network

```TypeScript
import * as ml from 'machine-learning';

// RNN: embedding -> recurrent hidden state -> dense softmax, trained by backprop-through-time.
// Read a tiny review and classify sentiment. Vocab: 0=<pad> 1=the 2=coffee 3=service 4=was 5=great 6=terrible
const reviews = new ml.Matrix([
    [1, 2, 4, 5, 0], // the coffee was great    -> positive
    [1, 3, 4, 5, 0], // the service was great   -> positive
    [1, 2, 4, 6, 0], // the coffee was terrible -> negative
    [1, 3, 4, 6, 0], // the service was terrible -> negative
]);
const sentiment = new ml.Matrix([[1, 0], [1, 0], [0, 1], [0, 1]]); // [positive, negative]

const rnn = new ml.RecurrentNeuralNetwork();
rnn.setVocabSize(7).setEmbeddingDim(2).setHiddenSize(8).setLearningRate(0.1).setNumberOfEpochs(400).setSeed(0);

rnn.train(reviews, sentiment);

console.log(rnn.predict(reviews).getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 0 ], [ 1 ], [ 1 ] ]  <- "great" -> positive (0), "terrible" -> negative (1)

console.log('gradients verified:', rnn.checkGradients());
// gradients verified: true  <- finite-difference check of backprop-through-time

```

### Transformer

```TypeScript
import * as ml from 'machine-learning';

// Transformer (single self-attention block): find the salient word anywhere in the sequence.
// Vocab: 0=<cls> 1=filler 2=good 3=bad. Class = which keyword appears, at any position.
const make = (pos: number, key: number) => { const s = [0, 1, 1, 1, 1]; s[pos] = key; return s; };
const sequences = new ml.Matrix([make(1, 2), make(3, 2), make(2, 3), make(4, 3), make(4, 2), make(1, 3)]);
const targets = new ml.Matrix([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]);

const transformer = new ml.Transformer();
transformer.setVocabSize(4).setModelDim(8).setMaxLength(5).setLearningRate(0.05).setNumberOfEpochs(500).setSeed(0);

transformer.train(sequences, targets);

console.log(transformer.predict(sequences).getMaximumRowIndeces().toArray());
// [ [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ], [ 1 ] ]  <- "good" -> 0, "bad" -> 1, wherever it appears

console.log(transformer.getAttention(make(3, 2))[0].map(a => Number(a.toFixed(2))));
// [ 0.08, 0.13, 0.1, 0.55, 0.14 ]  <- the [CLS] token attends mostly to index 3, where the keyword sits

console.log('gradients verified:', transformer.checkGradients());
// gradients verified: true  <- finite-difference check of the attention backprop

```

### k-Means Clustering

```TypeScript
import * as ml from 'machine-learning';

// k-Means clustering (unsupervised): group points into 2 clusters — note there are no targets.
const inputs = new ml.Matrix([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]]);

const kMeans = new ml.KMeans();
kMeans.setNumberOfClusters(2);
kMeans.setSeed(0); // makes the random centroid initialisation reproducible

kMeans.train(inputs);

const predictions = kMeans.predict(inputs);
console.log(predictions.toArray());
// one-hot cluster membership per point (the low blob is cluster 1, the high blob cluster 0)
// [ [ 0, 1 ], [ 0, 1 ], [ 0, 1 ], [ 1, 0 ], [ 1, 0 ], [ 1, 0 ] ]

console.log(kMeans.getCentroids().toArray());
// the two cluster centres (each blob's mean)
// [ [ 10.333333333333332, 10.333333333333332 ], [ 0.3333333333333333, 0.3333333333333333 ] ]

```

### Hierarchical Clustering

```TypeScript
import * as ml from 'machine-learning';

// Hierarchical clustering (unsupervised): merge the closest groups bottom-up into a tree,
// then cut it to k clusters — no need to fix k before building.
const inputs = new ml.Matrix([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]]);

const hierarchical = new ml.HierarchicalClustering();
hierarchical.setNumberOfClusters(2).setLinkage('average'); // 'single' | 'complete' | 'average'

hierarchical.train(inputs);

const predictions = hierarchical.predict(inputs);
console.log(predictions.toArray());
// one-hot cluster membership (the low blob is one cluster, the high blob the other)
// [ [ 1, 0 ], [ 1, 0 ], [ 1, 0 ], [ 0, 1 ], [ 0, 1 ], [ 0, 1 ] ]

console.log(hierarchical.getMergeHistory().map(m => Number(m.distance.toFixed(2))));
// the dendrogram's merge heights, climbing until the two far-apart blobs join at the top
// [ 1, 1, 1.21, 1.21, 14.17 ]

```

### DBSCAN

```TypeScript
import * as ml from 'machine-learning';

// DBSCAN (unsupervised): clusters by density and flags stragglers as noise — no k needed.
// Two tight blobs and one far-flung outlier.
const inputs = new ml.Matrix([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [5, 5], [5.1, 5], [5, 5.1], [5.1, 5.1], [10, 0]]);

const dbscan = new ml.DBSCAN();
dbscan.setEpsilon(0.5).setMinPoints(3); // a point is "core" with >= minPoints neighbours within epsilon

dbscan.train(inputs);

console.log(dbscan.getLabels());
// [ 0, 0, 0, 0, 1, 1, 1, 1, -1 ]  <- two dense blobs (clusters 0 and 1), the lone point is noise (-1)

console.log(dbscan.getClusterCount());
// 2  <- discovered, not specified up front

```

### PCA

```TypeScript
import * as ml from 'machine-learning';

// PCA (unsupervised): find the axes of greatest variance and project onto the top ones.
// These points lie on the line y = x, so a single axis captures all the variance.
const inputs = new ml.Matrix([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]]);

const pca = new ml.PCA();
pca.setNumberOfComponents(1);

pca.train(inputs);

console.log(pca.predict(inputs).toArray());
// each point's position along the one axis: [ [ -2.83 ], [ -1.41 ], [ 0 ], [ 1.41 ], [ 2.83 ] ]

console.log(pca.getExplainedVarianceRatio());
// [ 1 ]  <- that single axis holds 100% of the variance (the 2nd dimension was redundant)

```

### Anomaly Detection

```TypeScript
import * as ml from 'machine-learning';

// Anomaly detection (unsupervised): fit a Gaussian to "normal" data, flag the unlikely points.
const inputs = new ml.Matrix([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [0.5, -0.5]]);

const detector = new ml.AnomalyDetector();
detector.setThreshold(3); // flag points more than 3 Mahalanobis "std devs" from the centre

detector.train(inputs);

console.log(detector.predict(new ml.Matrix([[0, 0], [0.5, 0.5], [8, 8]])).toArray());
// [ [ 0 ], [ 0 ], [ 1 ] ]  <- the far-out point is the anomaly

console.log(detector.score(new ml.Matrix([[8, 8]])).toArray());
// [ [ 12.22 ] ]  <- its Mahalanobis distance, far past the threshold

```

### Association Rules

```TypeScript
import * as ml from 'machine-learning';

// Association rules (unsupervised): mine "basket" data for "buys X also buys Y".
// Items: 0=coffee, 1=croissant, 2=tea, 3=cookie. One row per receipt, 1 = item present.
const inputs = new ml.Matrix([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
]);

const rules = new ml.AssociationRules();
rules.setMinSupport(0.3).setMinConfidence(0.6); // support/confidence bars for keeping a rule

rules.train(inputs);

const top = rules.getRules()[0]; // rules come sorted strongest (highest lift) first
console.log(top.antecedent, '->', top.consequent, '| confidence', top.confidence, 'lift', top.lift);
// [ 1 ] -> [ 0 ] | confidence 1 lift 1.5  <- everyone who bought a croissant also bought coffee

```

### Recommender

```TypeScript
import * as ml from 'machine-learning';

// Recommender (matrix factorization): fill in ratings nobody gave, suggest what each person likes.
// 4 users x 4 items, 0 = not rated. Two taste groups: users 0-1 vs users 2-3.
const ratings = new ml.Matrix([
    [5, 5, 1, 0],
    [5, 0, 1, 1],
    [1, 1, 5, 5],
    [0, 1, 5, 5],
]);

const recommender = new ml.Recommender();
recommender.setNumberOfFactors(2).setNumberOfEpochs(500).setLearningRate(0.02).setSeed(0);

recommender.train(ratings);

console.log(recommender.recommend(1)); // user 1's unrated items, best first
// [ { item: 1, score: 5.24 } ]  <- predicts user 1 will love item 1 (their taste group does)

console.log(recommender.predict().toArray()[3][0].toFixed(2)); // user 3's hidden item 0
// 1.29  <- correctly low: user 3 is in the other taste group

```

### Exponential Smoothing (time series)

```TypeScript
import * as ml from 'machine-learning';

// Exponential smoothing (Holt-Winters): forecast a series with a weekly rhythm.
// Two weeks of daily croissant demand (Mon..Sun) - quiet midweek, busy weekends, slight uptrend.
const demand = new ml.Matrix([
    [40], [42], [45], [50], [80], [95], [70],
    [44], [46], [49], [55], [85], [100], [74],
]);

const model = new ml.ExponentialSmoothing();
model.setAlpha(0.4).setBeta(0.1).setGamma(0.5).setSeasonLength(7); // level, trend, season, period

model.train(demand);

console.log(model.predict(7).toArray().map(row => Math.round(row[0])));
// [ 47, 49, 52, 57, 87, 103, 78 ]  <- next week, continuing the weekly rhythm (busy Fri/Sat) + uptrend

```

### Multi-Armed Bandit (reinforcement learning)

```TypeScript
import * as ml from 'machine-learning';

// Multi-armed bandit: which daily special sells best? Unlike every other model here, a bandit has no
// train(inputs, targets) - it learns online by *acting*: selectArm -> see a reward -> update, and must
// balance exploring under-tried specials against exploiting the one that looks best so far.
const sellRate = [0.3, 0.55, 0.8]; // hidden truth the bandit must discover

const bandit = new ml.MultiArmedBandit();
bandit.setNumberOfArms(3).setStrategy('ucb').setSeed(0); // or 'epsilon-greedy'

for (let day = 0; day < 2000; day++) {
    const special = bandit.selectArm();                    // choose what to feature
    const sold = Math.random() < sellRate[special] ? 1 : 0; // ask the world
    bandit.update(special, sold);                          // learn from the outcome
}

console.log(bandit.getValues().map(v => Number(v.toFixed(2))));
// [ 0.30, 0.55, 0.80 ]  <- learned sell-rates converge to the hidden truth
console.log(bandit.getCounts());
// [ 80, 246, 1674 ]  <- and it played the winner (special 2) far the most, sampling the rest to be sure

```

## Development

This project is written in TypeScript and tested with [Vitest](https://vitest.dev/).

```bash
yarn install        # install dependencies
yarn test           # run the test suite
yarn test:watch     # run the tests in watch mode
yarn coverage       # run the tests with a coverage report
yarn typecheck      # type-check without emitting
yarn build          # compile the library to dist/lib
yarn demo           # run the runnable examples in examples/demo.ts
```

### Playground site

The interactive playground and tutorials are a self-contained Vite + React app in
[`site/`](./site) with its own dependencies (it imports the library straight from source via a
Vite alias, so it never ships to npm).

```bash
yarn build              # build the library first — the site type-checks against dist/lib
cd site
yarn install            # install the site's dependencies
yarn dev                # start the playground at http://localhost:5173/machine-learning/
yarn build              # production build (deployed to GitHub Pages by deploy-site.yml)
```

> **Enabling the live site:** in the repo, go to **Settings → Pages → Build and deployment →
> Source: "GitHub Actions"**. Pushes to `master` then deploy automatically.

### Releasing

The npm package is published by CI when a version tag is pushed:

```bash
# bump "version" in package.json, commit as "Release X.Y.Z", merge to master, then:
git tag vX.Y.Z
git push --tags        # publish.yml type-checks, tests, builds, and runs `npm publish`
```

> Requires an **`NPM_TOKEN`** repository secret (an npm *automation* token for the package owner):
> **Settings → Secrets and variables → Actions → New repository secret**.
