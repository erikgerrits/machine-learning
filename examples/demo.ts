import * as ml from '../src/lib';

{
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
}

{
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
}

{
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
}

{
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
}

{
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
}

{
    // k-Means clustering (unsupervised): group points into 2 clusters — note there are no targets.
    const inputs = new ml.Matrix([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]]);

    const kMeans = new ml.KMeans();
    kMeans.setNumberOfClusters(2);
    kMeans.setSeed(0);

    kMeans.train(inputs);
    const predictions = kMeans.predict(inputs);
    console.log(predictions.toArray());
    // one-hot cluster membership per point (the low blob is cluster 1, the high blob cluster 0):
    // [ [ 0, 1 ], [ 0, 1 ], [ 0, 1 ], [ 1, 0 ], [ 1, 0 ], [ 1, 0 ] ]
    console.log(kMeans.getCentroids().toArray());
    // the two cluster centres (each blob's mean):
    // [ [ 10.333333333333332, 10.333333333333332 ], [ 0.3333333333333333, 0.3333333333333333 ] ]
}

{
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
}

{
    // Decision tree: class 1 iff both features are "high" (an AND rule).
    const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.8, 0.1]]);
    const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]);

    const decisionTree = new ml.DecisionTree();
    decisionTree.setMaxDepth(3);

    decisionTree.train(inputs, targets);
    const predictions = decisionTree.predict(inputs);
    console.log(predictions.getMaximumRowIndeces().toArray());
    // [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]  ← only the "both high" rows are class 1
}

{
    // Random forest: a committee of trees over bootstrap samples, votes averaged. Same AND rule.
    const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.8, 0.1]]);
    const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]);

    const randomForest = new ml.RandomForest();
    randomForest.setNumberOfTrees(30).setMaxDepth(3).setSeed(0);

    randomForest.train(inputs, targets);
    const predictions = randomForest.predict(inputs);
    console.log(predictions.getMaximumRowIndeces().toArray());
    // [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]  ← the committee agrees with the single tree
}

{
    // Gradient boosting: trees built in sequence, each fitting the leftover error. Same AND rule.
    const inputs = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.8, 0.1]]);
    const targets = new ml.Matrix([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]);

    const gradientBoosting = new ml.GradientBoosting();
    gradientBoosting.setNumberOfTrees(60).setLearningRate(0.3).setMinSamplesSplit(2);

    gradientBoosting.train(inputs, targets);
    const predictions = gradientBoosting.predict(inputs);
    console.log(predictions.getMaximumRowIndeces().toArray());
    // [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ] ]  ← boosted into the same AND rule
}

{
    const nn = new ml.FeedforwardNeuralNetwork([40, 40, 40, 40, 40]);
    console.log('');
    console.log('Checking FeedforwardNeuralNetwork gradients...');
    console.log(nn.checkGradients() ? 'OK' : 'GRADIENTS NOT OK!!!');
}
