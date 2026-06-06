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
* [k-Means Clustering](#k-means-clustering)

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
