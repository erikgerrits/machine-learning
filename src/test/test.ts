import * as ml from '../lib/index';

{
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
}


{
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
}