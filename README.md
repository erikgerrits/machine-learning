# machine-learning

## Important notes
This library has a dependency on the **nblas** package for fast matrix operations.
It should work by default on OSX, but on Linux you may need to run ``apt-get install libblas-dev`` first.
On Windows you may need to install [LAPACK](http://www.netlib.org/lapack/#_lapack_version_3_7_0_2).

This library is in an early development phase and many **breaking changes are to be expected**.

The TypeScript source files can be found on [GitHub](https://github.com/erikgerrits/machine-learning) and the JavaScript production files(including .ts.d files) can be found as an [npm package](https://www.npmjs.com/package/machine-learning).

## Documentation

TypeDocs for all classes can be found [here](http://platformj.com). Below are some simple code usage examples.

### Linear Regression

```TypeScript
import * as ml from 'machine-learning';

// Linear Regression: y = 1000 + 200 * x
const inputs = new ml.Matrix([[5], [7], [9], [11], [13]]);
const outputs = new ml.Matrix([[2000], [2400], [2800], [3200], [3600]]);

const linearRegression = new ml.LinearRegression(inputs, outputs);
linearRegression.setMaximumIterations(10000);
linearRegression.setLearningRate(0.02);

const predictionsBeforeTraining = linearRegression.predict(inputs);
console.log(predictionsBeforeTraining.toArray());
// [ [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ] ]

linearRegression.train();
const predictionsAfterTraining = linearRegression.predict(inputs);
console.log(predictionsAfterTraining.toArray());
// [ [ 1999.999991189672 ], [ 2399.9999948012005 ], [ 2799.999998412729 ], [ 3200.0000020242574 ], [ 3600.000005635786 ] ]

```

For more detailed information, access the [LinearRegression class documentation](http://platformj.com/classes/_machine_learning_supervised_linearregression_.linearregression.html)

### Logistic Regression
```TypeScript
import * as ml from 'machine-learning';

// Logistic Regression: determine if second input is higher than first input
const inputs = new ml.Matrix([[1000, 1100], [4500, 3000], [700, 1300], [1150, 700], [1300, 1200], [600, 650]]);
const outputs = new ml.Matrix([[1], [0], [1], [0], [0], [1]]);

const logisticRegression = new ml.LogisticRegression(inputs, outputs);
logisticRegression.setMaximumIterations(1000);
logisticRegression.setLearningRate(0.01);

const predictionsBeforeTraining = logisticRegression.predict(inputs);
console.log(predictionsBeforeTraining.toArray());
// [ [ 0.5 ], [ 0.5 ], [ 0.5 ], [ 0.5 ], [ 0.5 ], [ 0.5 ] ]

logisticRegression.train();
const predictionsAfterTraining = logisticRegression.predict(inputs);
console.log(predictionsAfterTraining.toArray());
// [ [ 1 ], [ 0 ], [ 1 ], [ 0 ], [ 0 ], [ 1 ] ]

```

For more detailed information, access the [LogisticRegression class documentation](http://platformj.com/classes/_machine_learning_supervised_logisticregression_.logisticregression.html)

### Multiclass Logistic Regression
```TypeScript
import * as ml from 'machine-learning';

// Multiclass Logistic Regression: determine the highest value
const inputs = new ml.Matrix([[4500, 1200, 3000], [700, 890, 800], [700, 1200, 1300], [1150, 600, 700], [600, 1500, 1650], [400, 401, 400]]);
const outputs = new ml.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]);

const multiclassLogisticRegression = new ml.MulticlassLogisticRegression(inputs, outputs);
multiclassLogisticRegression.setMaximumIterations(10000);
multiclassLogisticRegression.setLearningRate(0.1);

const predictionsBeforeTraining = multiclassLogisticRegression.predict(inputs);
console.log(predictionsBeforeTraining.toArray());
// [ [ 0.5, 0.5, 0.5 ], [ 0.5, 0.5, 0.5 ], [ 0.5, 0.5, 0.5 ], [ 0.5, 0.5, 0.5 ], [ 0.5, 0.5, 0.5 ], [ 0.5, 0.5, 0.5 ] ]

multiclassLogisticRegression.train();
const predictionsAfterTraining = multiclassLogisticRegression.predict(inputs);
console.log(predictionsAfterTraining.toArray());
// [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ]

```

For more detailed information, access the [MulticlassLogisticRegression class documentation](http://platformj.com/classes/_machine_learning_supervised_multiclasslogisticregression_.multiclasslogisticregression.html)

### Feedforward Neural Network
```TypeScript
import * as ml from 'machine-learning';

// Feedforward neural network: solve XNOR problem (opposite of XOR)
const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);
const outputs = new ml.Matrix([[1], [0], [0], [1]]);

const feedforwardNeuralNetwork = new ml.FeedforwardNeuralNetwork(inputs, outputs, [5]);
feedforwardNeuralNetwork.setMaximumIterations(1000);
feedforwardNeuralNetwork.setLearningRate(1);

const predictionsBeforeTraining = feedforwardNeuralNetwork.predict(inputs);
console.log(predictionsBeforeTraining.toArray());
// [ [ 0.7124268075556096 ], [ 0.6816362785223121 ], [ 0.7143507126826376 ], [ 0.68576576898992 ] ]

feedforwardNeuralNetwork.train();
const predictionsAfterTraining = feedforwardNeuralNetwork.predict(inputs);
console.log(predictionsAfterTraining.toArray());
// [ [ 0.9959984961377983 ], [ 0.013960357357609482 ], [ 0.011093933572618063 ], [ 0.9798749896755027 ] ]

```

For more detailed information, access the [FeedforwardNeuralNetwork class documentation](http://platformj.com/classes/_machine_learning_supervised_feedforwardneuralnetwork_.feedforwardneuralnetwork.html)
