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
    // Hierarchical clustering (unsupervised): merge the closest groups bottom-up into a tree, then
    // cut it to k clusters — no need to fix k before building. Same two blobs as k-means.
    const inputs = new ml.Matrix([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]]);

    const hierarchical = new ml.HierarchicalClustering();
    hierarchical.setNumberOfClusters(2).setLinkage('average');

    hierarchical.train(inputs);
    console.log(hierarchical.predict(inputs).toArray());
    // one-hot membership: the low blob is one cluster, the high blob the other
    // [ [ 1, 0 ], [ 1, 0 ], [ 1, 0 ], [ 0, 1 ], [ 0, 1 ], [ 0, 1 ] ]
    console.log(hierarchical.getMergeHistory().map(m => Number(m.distance.toFixed(2))));
    // [ 1, 1, 1.21, 1.21, 14.17 ]  ← merge heights climbing until the two far-apart blobs join at the top
}

{
    // DBSCAN (unsupervised): clusters by density and flags stragglers as noise — no k needed.
    const inputs = new ml.Matrix([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [5, 5], [5.1, 5], [5, 5.1], [5.1, 5.1], [10, 0]]);

    const dbscan = new ml.DBSCAN();
    dbscan.setEpsilon(0.5).setMinPoints(3);

    dbscan.train(inputs);
    console.log(dbscan.getLabels());
    // [ 0, 0, 0, 0, 1, 1, 1, 1, -1 ]  ← two dense blobs (clusters 0 and 1), the lone point is noise (-1)
    console.log(dbscan.getClusterCount());
    // 2
}

{
    // PCA (unsupervised): find the axes of greatest variance and project onto the top ones.
    // These points lie exactly on the line y = x, so a single axis captures all the variance.
    const inputs = new ml.Matrix([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]]);

    const pca = new ml.PCA();
    pca.setNumberOfComponents(1);

    pca.train(inputs);
    console.log(pca.predict(inputs).toArray());
    // each point's position along the one axis: [ [ -2.83 ], [ -1.41 ], [ 0 ], [ 1.41 ], [ 2.83 ] ]
    console.log(pca.getExplainedVarianceRatio());
    // [ 1 ]  ← that single axis holds 100% of the variance (the 2nd dimension was redundant)
}

{
    // Anomaly detection (unsupervised): fit a Gaussian to "normal" data, flag the unlikely points.
    const inputs = new ml.Matrix([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [0.5, -0.5]]);

    const detector = new ml.AnomalyDetector();
    detector.setThreshold(3); // flag points more than 3 Mahalanobis "std devs" from the centre

    detector.train(inputs);
    console.log(detector.predict(new ml.Matrix([[0, 0], [0.5, 0.5], [8, 8]])).toArray());
    // [ [ 0 ], [ 0 ], [ 1 ] ]  ← the far-out point is the anomaly
    console.log(detector.score(new ml.Matrix([[8, 8]])).toArray());
    // [ [ 12.22 ] ]  ← its Mahalanobis distance, far past the threshold of 3
}

{
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
    rules.setMinSupport(0.3).setMinConfidence(0.6);

    rules.train(inputs);
    const top = rules.getRules()[0];
    console.log(top.antecedent, '->', top.consequent, '| confidence', top.confidence.toFixed(2), 'lift', top.lift.toFixed(2));
    // [ 1 ] -> [ 0 ] | confidence 1.00 lift 1.50  ← everyone who bought a croissant also bought coffee
}

{
    // Recommender (matrix factorization): fill in ratings nobody gave, suggest what each person likes.
    // 4 users x 4 items, 0 = not rated. Two taste groups: users 0–1 vs users 2–3.
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
    // [ { item: 1, score: 5.24 } ]  ← predicts user 1 will love item 1 (their taste group does)
    console.log(recommender.predict().toArray()[3][0].toFixed(2)); // user 3's hidden item 0 → low
    // 1.29
}

{
    // Exponential smoothing (Holt-Winters): forecast a series with a weekly rhythm.
    // Two weeks of daily croissant demand (Mon..Sun) — quiet midweek, busy weekends, slight uptrend.
    const demand = new ml.Matrix([
        [40], [42], [45], [50], [80], [95], [70],
        [44], [46], [49], [55], [85], [100], [74],
    ]);

    const model = new ml.ExponentialSmoothing();
    model.setAlpha(0.4).setBeta(0.1).setGamma(0.5).setSeasonLength(7);

    model.train(demand);
    console.log(model.predict(7).toArray().map(row => Math.round(row[0])));
    // [ 47, 49, 52, 57, 87, 103, 78 ]  ← next week, continuing the weekly rhythm (busy Fri/Sat) + uptrend
}

{
    // Convolutional neural network: learns to spot a shape anywhere in a tiny image.
    // 8×8 images — a horizontal line (class 0) or a vertical line (class 1), at various positions.
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
    // [ [ 0 ], [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 1 ] ]  ← horizontals → class 0, verticals → class 1
    console.log('CNN gradients verified:', cnn.checkGradients());
    // CNN gradients verified: true  ← finite-difference check of the convolution backprop
}

{
    // Recurrent neural network: read a sequence (a tiny review) and classify its sentiment.
    // Vocabulary: 0=<pad> 1=the 2=coffee 3=service 4=was 5=great 6=terrible
    const reviews = new ml.Matrix([
        [1, 2, 4, 5, 0], // the coffee was great   → positive
        [1, 3, 4, 5, 0], // the service was great  → positive
        [1, 2, 4, 6, 0], // the coffee was terrible → negative
        [1, 3, 4, 6, 0], // the service was terrible → negative
    ]);
    const sentiment = new ml.Matrix([[1, 0], [1, 0], [0, 1], [0, 1]]); // [positive, negative]

    const rnn = new ml.RecurrentNeuralNetwork();
    rnn.setVocabSize(7).setEmbeddingDim(2).setHiddenSize(8).setLearningRate(0.1).setNumberOfEpochs(400).setSeed(0);

    rnn.train(reviews, sentiment);
    console.log(rnn.predict(reviews).getMaximumRowIndeces().toArray());
    // [ [ 0 ], [ 0 ], [ 1 ], [ 1 ] ]  ← "great" reviews → positive (0), "terrible" → negative (1)
    console.log('RNN gradients verified:', rnn.checkGradients());
    // RNN gradients verified: true  ← finite-difference check of backprop-through-time
}

{
    // Transformer (self-attention): find the salient word anywhere in the sequence.
    // Vocab: 0=<cls> 1=filler 2=good 3=bad. Class = which keyword appears, at any position.
    const make = (pos: number, key: number) => { const s = [0, 1, 1, 1, 1]; s[pos] = key; return s; };
    const sequences = new ml.Matrix([make(1, 2), make(3, 2), make(2, 3), make(4, 3), make(4, 2), make(1, 3)]);
    const targets = new ml.Matrix([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]);

    const transformer = new ml.Transformer();
    transformer.setVocabSize(4).setModelDim(8).setMaxLength(5).setLearningRate(0.05).setNumberOfEpochs(500).setSeed(0);

    transformer.train(sequences, targets);
    console.log(transformer.predict(sequences).getMaximumRowIndeces().toArray());
    // [ [ 0 ], [ 0 ], [ 1 ], [ 1 ], [ 0 ], [ 1 ] ]  ← "good" → 0, "bad" → 1, wherever it appears
    // The [CLS] token's attention row concentrates on the keyword's position (here index 3):
    console.log(transformer.getAttention(make(3, 2))[0].map(a => Number(a.toFixed(2))));
    // [ 0.08, 0.13, 0.1, 0.55, 0.14 ]  ← CLS attends mostly to index 3, where the keyword sits
    console.log('Transformer gradients verified:', transformer.checkGradients());
    // Transformer gradients verified: true  ← finite-difference check of the attention backprop
}

{
    // Multi-armed bandit (reinforcement learning): which daily special sells best?
    // Three specials with hidden sell-rates; the bandit learns by trying them and watching outcomes.
    const sellRate = [0.3, 0.55, 0.8]; // hidden truth the bandit must discover

    const bandit = new ml.MultiArmedBandit();
    bandit.setNumberOfArms(3).setStrategy('ucb').setSeed(0);

    for (let day = 0; day < 2000; day++) {
        const special = bandit.selectArm();                          // choose what to feature
        const sold = Math.random() < sellRate[special] ? 1 : 0;       // ask the world
        bandit.update(special, sold);                                 // learn from the outcome
    }
    console.log(bandit.getValues().map(v => Number(v.toFixed(2))));
    // ≈ [ 0.30, 0.55, 0.80 ]  ← learned sell-rates converge to the hidden truth
    console.log('best special:', bandit.getValues().indexOf(Math.max(...bandit.getValues())), '(plays', bandit.getCounts(), ')');
    // best special: 2  ← and it played special 2 by far the most (exploit), sampling the others to be sure
}

{
    // Contextual bandit (LinUCB): the best offer now depends on WHO is at the counter.
    // Context = [isMorningRegular, isEveningTourist]. Two arms: a cinnamon roll vs an espresso tonic.
    const types = [[1, 0], [0, 1]];                  // morning regular / evening tourist
    const rate = [[0.8, 0.2], [0.2, 0.8]];           // hidden: roll wins mornings, tonic wins evenings

    const bandit = new ml.ContextualBandit().setNumberOfArms(2).setContextDimensions(2).setStrategy('linucb').setSeed(0);

    let lcg = 1;
    const rng = () => (lcg = (lcg * 48271) % 2147483647) / 2147483647; // tiny seeded generator
    for (let t = 0; t < 3000; t++) {
        const context = types[rng() < 0.5 ? 0 : 1];                  // who walked in
        const arm = bandit.selectArm(context);                       // pick an offer for them
        const took = rng() < rate[arm][context[1] === 1 ? 1 : 0] ? 1 : 0;
        bandit.update(arm, context, took);                           // learn from the outcome
    }
    console.log('morning →', bandit.selectArm([1, 0]), ' evening →', bandit.selectArm([0, 1]));
    // morning → 0  evening → 1  ← it learned a per-customer policy: roll for regulars, tonic for tourists
}

{
    // Q-learning (reinforcement learning): a café runner learns the quickest tray route across the
    // floor, from the kitchen (S) to a table (G), dodging a spill (#). Actions: up/right/down/left.
    // Only the goal pays off (+1), the spill stings (−1), each step costs a little (−0.04) — so the
    // agent must propagate that goal value *backward* through the floor to know which way to walk.
    const COLS = 4, ROWS = 3, GOAL = 3, HAZARD = 5, START = 8; // state = row*COLS + col
    //   . . . G        (top-right is the table)
    //   . # . .        (centre-left is the spill)
    //   S . . .        (bottom-left is the kitchen)
    const step = (s: number, a: number) => {
        let r = Math.floor(s / COLS), c = s % COLS;
        if (a === 0) r--; else if (a === 1) c++; else if (a === 2) r++; else c--;
        if (r < 0 || r >= ROWS || c < 0 || c >= COLS) { r = Math.floor(s / COLS); c = s % COLS; } // bump a wall, stay
        const next = r * COLS + c;
        if (next === GOAL) return { next, reward: 1, done: true };
        if (next === HAZARD) return { next, reward: -1, done: true };
        return { next, reward: -0.04, done: false };
    };

    const agent = new ml.QLearning().setNumberOfStates(ROWS * COLS).setNumberOfActions(4)
        .setLearningRate(0.4).setDiscountFactor(0.95).setEpsilon(0.2).setSeed(0);
    for (let episode = 0; episode < 3000; episode++) {
        let state = START;
        for (let t = 0; t < 50; t++) {
            const action = agent.selectAction(state);
            const { next, reward, done } = step(state, action);
            agent.update(state, action, reward, next, done);
            if (done) break;
            state = next;
        }
    }

    const arrows = ['↑', '→', '↓', '←'];
    const policy = agent.getPolicy().map((a, s) =>
        s === GOAL ? 'G' : s === HAZARD ? '#' : arrows[a]);
    for (let r = 0; r < ROWS; r++) console.log(policy.slice(r * COLS, r * COLS + COLS).join(' '));
    // → → → G     the learned policy — every cell points along a quickest safe route to the table.
    // ↑ # ↑ ↑     From the kitchen it walks up the left edge, then right across the top, steering
    // ↑ → ↑ ↑     around the spill (#) rather than through it.
}

{
    // Deep Q-network (DQN): the same value idea, but the floor is now *continuous*. The runner's state
    // is a real position (x, y) in [0,1]², far too many states to tabulate — so a neural network learns
    // to predict Q(state) and generalises to spots it never stepped on. Goal: reach the table near the
    // top-right corner (+1, ends the trip); every other step pays 0, discounted by γ.
    const STEP = 0.12, GOAL: [number, number] = [0.85, 0.85], RADIUS = 0.18;
    const dist = (x: number, y: number) => Math.hypot(x - GOAL[0], y - GOAL[1]);
    const step = (s: number[], a: number) => {
        const x = Math.min(1, Math.max(0, s[0] + (a === 1 ? STEP : a === 3 ? -STEP : 0)));
        const y = Math.min(1, Math.max(0, s[1] + (a === 0 ? STEP : a === 2 ? -STEP : 0)));
        return dist(x, y) < RADIUS ? { next: [x, y], reward: 1, done: true } : { next: [x, y], reward: 0, done: false };
    };

    const dqn = new ml.DeepQNetwork().setStateSize(2).setNumberOfActions(4).setHiddenSizes([24, 24])
        .setLearningRate(0.2).setDiscountFactor(0.9).setEpsilon(0.3).setSeed(1);
    let lcg = 3;
    const rng = () => (lcg = (lcg * 48271) % 2147483647) / 2147483647;
    for (let episode = 0; episode < 4000; episode++) {
        let state = [rng(), rng()]; // start anywhere on the floor, so the goal is found often
        for (let t = 0; t < 60; t++) {
            const action = dqn.selectAction(state);
            const { next, reward, done } = step(state, action);
            dqn.observe(state, action, reward, next, done);
            if (done) break;
            state = next;
        }
    }

    // Sample the learned value V(x, y) on a coarse grid — high (▓) near the table, low (·) far away.
    const shades = ['·', '░', '▒', '▓', '█'];
    for (let row = 4; row >= 0; row--) {
        let line = '';
        for (let col = 0; col < 5; col++) {
            const v = dqn.getValue([col / 4, row / 4]);
            line += shades[Math.min(4, Math.floor(v * 5))] + ' ';
        }
        console.log(line);
    }
    // ▓ ▓ █ █ █     the value surface the network painted over the whole floor — brightest at the
    // ▒ ▓ ▓ █ █     table (top-right) and fading with distance. The runner just walks uphill: in every
    // ░ ▒ ▓ ▓ █     state it picks the action whose Q is highest, and that leads it home — even from
    // ░ ░ ▒ ▓ █     spots it never actually visited while training, because the net interpolates
    // ░ ░ ░ ▒ ▓     between the ones it did.
}

{
    // Autoencoder: compress, then denoise. Each "image" is a 7×7 soft blob at some position — 49
    // pixels that really only vary 2 ways (where the blob is). The autoencoder squeezes each through a
    // 2-number bottleneck and rebuilds it; noise can't fit through that squeeze, so it gets cleaned up.
    const W = 7;
    const blob = (cx: number, cy: number) => {
        const img: number[] = [];
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) img.push(Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 5));
        return img;
    };
    let lcg = 1;
    const rng = () => (lcg = (lcg * 48271) % 2147483647) / 2147483647;
    const images = Array.from({ length: 80 }, () => blob(1.5 + rng() * 4, 1.5 + rng() * 4));

    const autoencoder = new ml.Autoencoder().setHiddenSizes([24]).setCodeSize(2).setLearningRate(1).setNumberOfEpochs(1500).setSeed(0);
    autoencoder.train(new ml.Matrix(images)); // 49 pixels -> 2 numbers -> 49 pixels

    const clean = blob(3, 3);
    const noisy = clean.map(v => Math.min(1, Math.max(0, v + (rng() - 0.5) * 0.8))); // speckle it
    const cleaned = autoencoder.reconstruct(new ml.Matrix([noisy])).toArray()[0];

    const render = (img: number[]) => img.map(v => ' .:+#'[Math.min(4, Math.floor(v * 5))]).join('');
    const row = (y: number, img: number[]) => render(img.slice(y * W, y * W + W));
    console.log('   noisy        →  reconstructed');
    for (let y = 0; y < W; y++) console.log('  ' + row(y, noisy) + '      ' + row(y, cleaned));
    // The left is a speckled mess; the right is a clean blob back near the centre — the network kept
    // only what fit through the 2-number code (roughly "a blob, here") and dropped the noise.
}

{
    // Variational autoencoder (VAE): the autoencoder turned into a *generator*. It learns a latent
    // space shaped like a standard normal, so we can draw a random code z ~ N(0, 1), decode it, and get
    // a brand-new blob that was never in the training set. Same 7×7 blobs as before.
    const W = 7;
    const blob = (cx: number, cy: number) => {
        const img: number[] = [];
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) img.push(Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 4));
        return img;
    };
    let lcg = 1;
    const rng = () => (lcg = (lcg * 48271) % 2147483647) / 2147483647;
    const images = Array.from({ length: 200 }, () => blob(1.5 + rng() * 4, 1.5 + rng() * 4));

    const vae = new ml.VariationalAutoencoder().setHiddenSize(32).setCodeSize(2).setBeta(1).setLearningRate(0.06).setNumberOfEpochs(2000).setSeed(0);
    vae.train(new ml.Matrix(images));

    const samples = vae.sample(3, 7).toArray(); // 3 fresh blobs, drawn from the prior — not memorised
    const render = (img: number[], y: number) => img.slice(y * W, y * W + W).map(v => ' .:+#'[Math.min(4, Math.floor(v * 5))]).join('');
    console.log('three invented blobs (z ~ N(0,1) → decode):');
    for (let y = 0; y < W; y++) console.log('  ' + [0, 1, 2].map(i => render(samples[i], y)).join('    '));
    // Each column is a clean, plausible blob at a different spot — none of them copied from the data.
    // The decoder became a little generator: feed it a point in latent space, get a new image out.
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
    // Support vector machine: the widest-margin boundary between two classes. A linear kernel draws
    // a straight max-margin line; swap in .setKernel('rbf') to carve curved boundaries (the kernel trick).
    const inputs = new ml.Matrix([[2, 2], [3, 3], [3, 1], [1, 3], [-2, -2], [-3, -3], [-3, -1], [-1, -3]]);
    const targets = new ml.Matrix([[1], [1], [1], [1], [0], [0], [0], [0]]);

    const supportVectorMachine = new ml.SupportVectorMachine();
    supportVectorMachine.setKernel('linear').setRegularization(10).setNumberOfIterations(50);

    supportVectorMachine.train(inputs, targets);
    // predict returns the raw decision score per row; its sign is the class (≥ 0 → class 1).
    const predictions = supportVectorMachine.predict(inputs);
    console.log(predictions.transform(score => (score >= 0 ? 1 : 0)).toArray());
    // [ [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ] ]
    console.log(supportVectorMachine.getSupportVectorIndices());
    // [ 0, 2, 3, 4, 6 ]  ← the boundary-hugging points the line balances on (the rest could be deleted)
}

{
    // Perceptron (a single neuron): learns AND, but one straight line can't solve XOR.
    const gates = new ml.Matrix([[0, 0], [0, 1], [1, 0], [1, 1]]);

    const perceptron = new ml.Perceptron();
    perceptron.setLearningRate(0.1).setNumberOfEpochs(100);

    perceptron.train(gates, new ml.Matrix([[0], [0], [0], [1]])); // AND
    console.log(perceptron.predict(gates).toArray());
    // [ [ 0 ], [ 0 ], [ 0 ], [ 1 ] ]  ← solves AND (linearly separable)

    perceptron.reset();
    perceptron.train(gates, new ml.Matrix([[0], [1], [1], [0]])); // XOR
    console.log(perceptron.predict(gates).toArray());
    // [ [ 1 ], [ 1 ], [ 0 ], [ 0 ] ] ≠ [0,1,1,0] — XOR isn't linearly separable; it takes a layered network (below)
}

{
    const nn = new ml.FeedforwardNeuralNetwork([40, 40, 40, 40, 40]);
    console.log('');
    console.log('Checking FeedforwardNeuralNetwork gradients...');
    console.log(nn.checkGradients() ? 'OK' : 'GRADIENTS NOT OK!!!');
}
