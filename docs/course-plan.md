# The Drifting Leaf — A Story-Driven Machine Learning Course (Master Plan)

> Working title. The café is **The Drifting Leaf**, on **Maple Lane**. Our protagonist is **Nadia**.
> This is a *global, adaptable outline* — a living map we extend release by release, not a one-pass build.

---

## Context — why we're doing this

When this plan was written, the site had six excellent, technically-honest tutorials
(Linear/Logistic/Multiclass Regression, k‑NN, a neural net, k‑Means), each mapping real math onto
the real library source via live Canvas playgrounds. The weakness was **motivation**: the problems were abstract toy datasets
("here is a cloud of points"). Nothing pulls you from one algorithm to the next.

This plan reframes the whole thing as a **narrative course**: one character, Nadia, revives a
failing café and grows it into something bigger. Every chapter she hits a wall — her current tool
*visibly fails* at a new problem — and she reasons her way to a new method. The method she
"invents" **is the real library class**; the playground **is her tool**. The result keeps the
existing math‑on‑code rigor but wraps it in a page‑turner that makes you *want* the next algorithm
before you're taught it.

We also use this as the excuse to **widen the curriculum massively** — simpler starting points,
the missing in-between methods, advanced models, and whole other paradigms (ensembles,
dimensionality reduction, recommenders, sequences, deep learning, reinforcement learning,
generative). The built algorithms become the spine; everything else is roadmap.

**Goal:** engaging, fun, genuinely educational, and very cool.

**Status (this release).** Woven through **Ch 0–21** with no built-but-unwoven gaps: all of Part 1,
the whole of Part 2 (k‑NN, Naive Bayes, Decision Trees, Random Forests, Gradient Boosting, Support
Vector Machines), all of Part 3 (k‑Means, Hierarchical Clustering, DBSCAN, PCA, Anomaly Detection,
Association Rules, Recommender Systems), and Part 4 — Time Series, the Perceptron interlude, Neural
Networks, Convolutional Networks, Recurrent Networks, and now **Ch 23 Transformers & Attention**
(the newest chapter: a from-scratch trainable single self-attention block — Q/K/V, scaled
dot-product attention, a CLS head, learned token + positional embeddings — with full backprop and
gradient checking). **All three deep-learning frontier chapters (CNN, RNN, Transformer) became full
trainable builds**, each gradient-checked, rather than the planned "adopts" explainers. That closes
the built deep-learning arc (Ch 0–23). Still roadmap: the reinforcement-learning arc (Ch 24–27),
generative (Ch 28–29), and Bayesian (Ch 30). See the status column below.

---

## The core idea: the "old tool breaks" engine

Every chapter follows the same emotional beat — this *is* the request, made into a repeatable
template. The cliffhanger at the end of each chapter is the hook into the next.

**Per-chapter template (Nadia's Ledger):**
1. **The Problem** — a scene at the café. A concrete business need, with stakes.
2. **The Wall** — she reaches for her *current* tool and it fails. We show the failure **live in
   the playground** (the line overshoots, the straight boundary can't carve a spiral, there are no
   labels to learn from…). The reader *feels* the limitation.
3. **The Idea** — she reasons out loud: *what's missing, and what would fix it?* This is where the
   new algorithm is born from need, not decree.
4. **How It Works** — the intuition, then the math (just-in-time), then the **real library code**
   it maps onto.
5. **Try It** — operate the live playground; sliders let you reproduce both the failure and the fix.
6. **What Broke** — a teaser: the new tool's own limitation, which becomes the *next* chapter's wall.
7. **Field Note** (optional sidebar) — a reusable concept (evaluation, overfitting, ethics…).

This template makes the course consistent to write and turns the syllabus into a story with
momentum: *predict → the average is too dumb → fit a line → lines can't say yes/no → probabilities →
two classes isn't enough → many classes → straight fences fail → ask neighbors → neighbors are slow
& dumb in high dimensions → learn features → … → there are no labels at all → discover structure →
… → don't just predict, act and learn from outcomes → reinforcement learning.*

---

## Design principles (what keeps the café a durable home)

The café holds up across the whole ML landscape — but only because of two load-bearing principles.
A growing business is the one setting that naturally emits *every* species of data problem (numbers
to predict, decisions to make, unlabeled groups, sequences, text, images, recommendations, repeated
decisions with feedback, high-stakes bets under uncertainty), which is exactly why "a business" is
the canonical setting for applied-ML courses. To make that work all the way to the frontier:

1. **The growth arc is load-bearing, not decoration.** Indie café → small chain → institution
   (the "seasons"). A transformer or CNN is absurd for a corner café but obvious for a chain with
   an app. As long as we honor the arc, the café scales to the whole landscape. If a method ever
   *feels too big*, the fix is almost always "we're further along the arc than this chapter
   assumed" — advance the business, don't abandon the setting.

2. **The "builds → adopts" pivot.** Early/mid chapters: Nadia **invents** the method to solve her
   problem — she's the maker, and it's fully implemented in the library. Frontier chapters: Nadia
   **adopts** a tool and we **open the hood** — she's the user/explorer, and the chapter is an
   honest explainer rather than a from-scratch build. This keeps us truthful (we're not pretending
   a café trains GPT from scratch) *and* it's a lovely character arc: founder → leader who
   understands the tools she relies on. The pivot is narratively motivated (she hires/partners with
   Priya's data work; the chain has resources).

**How new methods find a home.** The course is organized by **problem type × business stage**, not
by algorithm, so any future method has two coordinates to slot into (which problem family + which
season). The framing is natural enough that it actively *suggests* good additions — see the
candidate chapters under Open Questions.

---

## Story bible — character & world

**Nadia** — late 20s, ex–line cook, sharp and skeptical of hype. She inherits her aunt's failing
café, *The Drifting Leaf*, with a shoebox of receipts and no idea how to run a business. She's not
a mathematician; she learns by *doing* and writes every method she figures out into a battered
**ledger** (this is the model registry — each algorithm is a page; each playground is the tool she
built from that page).

**Recurring cast** (gives every problem stakes and warmth):
- **Tomás** — the gruff supplier who needs *exact* orders by 6am → drives demand forecasting.
- **Priya** — Nadia's teenage niece who codes; occasionally hands Nadia a "tool" → lets us
  introduce code and libraries naturally, in-world.
- **Cogwheel Coffee** — the slick data-driven chain that opens across the street → the rival who
  keeps raising the stakes (better forecasts, an app, dynamic pricing).
- **Mr. Alvarez & the regulars** — the human face of "customers" / data points.
- **Bea** — the baker; quality and recipes.

**Why problems escalate (the business arc / "seasons"):** as the café grows, the *amount* of data
grows and the *kind* of question changes. This is the engine that keeps the syllabus widening:

| Season | Business stage | Problem family | ML paradigm |
|---|---|---|---|
| **1 — Keep the lights on** | One failing café | Predict & decide | Regression, classification (the basics) |
| **2 — Know your regulars** | Loyal crowd forms | Sort, segment, recommend | More classifiers, ensembles, unsupervised, recsys |
| **3 — Go wide** | 2nd location + an app | Photos, reviews, sequences, scale | Deep learning, time series, NLP, vision |
| **4 — The smart café** | A small chain | Act & optimize over time | Reinforcement learning, bandits, optimization |
| **5 — The frontier** | An institution | Create & reason under uncertainty | Generative, Bayesian, advanced |

---

## The course map (the wide, adaptable outline)

Ordered simplest → most advanced. Three states:

- **✅ DONE** — built *and* woven into the café story: the algorithm lives in `src/lib` with a
  playground, and its tutorial follows the per-chapter template in Nadia's voice.
- **◐ BUILT** — the algorithm + playground exist in the code, but the tutorial is still the older
  generic explainer, *not yet re-themed* into the story. (Currently: none — every built algorithm is woven.)
- **○ ROADMAP** — a future chapter; the library grows into it.

Every row's "Wall" is the previous tool failing.

### Part 0 — Foundations (the data mindset)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 0 | **The Ledger** | Make sense of a shoebox of receipts | No method yet → learn to think in *observations, features, train vs. predict, error*; meet the baseline predictors (**predict the mean** / **predict the majority**) and the rule "you must beat the baseline" | ✅ DONE (intro) |

### Part 1 — Predicting & deciding (Season 1)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 1 | **Linear Regression** | How many croissants to bake tomorrow? | Baking the daily *average* wastes stock on slow days and sells out on sunny Saturdays → fit a **line** through demand vs. temperature; MSE; gradient descent | ✅ `LinearRegression` |
| 2 | **Many features** | Demand depends on temp **and** weekday **and** foot traffic | One input is too crude → **multiple** features as a weighted sum; feature scaling; the **Matrix** as "a page of measurements" | ✅ `LinearRegression` (multi-feature) |
| 3 | **Overfitting & Regularization** | She adds 30 features incl. silly ones; nails last week, flops on next week | "More features = better?" No → **train/test split**, overfitting, **L2/ridge** to shrink weights | ✅ `Regression` (`regularizationFactor`) |
| 4 | **Logistic Regression** | Will this dough batch rise? (good / bad) | Regression-on-0/1 overshoots past 1, one bad batch tilts the line, "0.5 risen" is nonsense → **sigmoid** → probability; decision boundary; log-loss | ✅ `LogisticRegression` |
| 5 | **Multiclass / Softmax** | Auto-sort pastries into 5 types on the line | Yes/no can't pick among 5 → **one-vs-rest**, one-hot, argmax | ✅ `MulticlassLogisticRegression` |

### Part 2 — A wider toolbox (Season 2)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 6 | **k-Nearest Neighbors** | Lookalike customers/items that swirl together | Straight boundaries *cannot* carve a spiral → **ask the neighbors**; Euclidean distance; lazy learning; choosing k | ✅ `NearestNeighbors` |
| 7 | **Naive Bayes** | Filter spam reservations & complaint emails | k-NN/logistic need numbers; this is *text* → **Bayes' rule**, bag-of-words, fast & probabilistic | ✅ `NaiveBayes` |
| 8 | **Decision Trees** | Staff need an explainable "comp this order?" rulebook | Linear models & k-NN are opaque → a **flowchart of yes/no splits**; entropy/info-gain; interpretability | ✅ `DecisionTree` |
| 9 | **Random Forests / Bagging** | One tree's calls are jumpy & overfit | A single tree is unstable → **many trees vote**; bagging; feature importance | ✅ `RandomForest` |
| 10 | **Gradient Boosting** | Cogwheel's forecasts beat hers | Need the tabular workhorse → **boost**: each model fixes the last one's mistakes | ✅ `GradientBoosting` |
| 11 | **Support Vector Machines** | Find the *safest* boundary, not just *a* boundary | Logistic gives any separating line → **max-margin** + the **kernel trick** for nonlinearity without nets | ✅ `SupportVectorMachine` |

### Part 3 — Understanding customers (Season 2, unsupervised)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 12 | **k-Means Clustering** | Who *are* my regulars? (no labels) | Every method so far needed an answer key — here there's none → **unsupervised**; Lloyd's algorithm; inertia; choosing k | ✅ `KMeans` |
| 13 | **Hierarchical Clustering** | A nested family tree of menu items / customer groups | k is unknown & structure is nested → **dendrograms** | ✅ `HierarchicalClustering` |
| 14 | **DBSCAN** | A weird late-night cluster (possible fraud) | k-means forces every point into a round blob → **density-based** clusters + outliers | ✅ `DBSCAN` |
| 15 | **PCA / Dimensionality Reduction** | A 30-question taste survey you can't visualize | You can't see 30 dimensions; features are redundant → **principal components**: 30 → 2 "flavor axes" | ✅ `PCA` |
| 16 | **Anomaly Detection** | Catch the fraudulent transaction / spoiled batch | Rare events drown in normal data → model "normal," flag the rest | ✅ `AnomalyDetector` |
| 17 | **Association Rules** | "Coffee + muffin" basket combos | Need co-occurrence patterns → Apriori; support/confidence/lift | ✅ `AssociationRules` |
| 18 | **Recommender Systems** | The loyalty app suggests items per person | Basket rules are global, not personal → **collaborative filtering / matrix factorization**; latent taste factors | ✅ `Recommender` |

### Part 4 — Scale, sequences & deep learning (Season 3)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 19 | **Time Series Forecasting** | *Forecasting numbers* across weeks & seasons | Plain regression ignores order & seasonality → moving averages, exponential smoothing, trend/seasonality (ARIMA-lite) | ✅ `ExponentialSmoothing` |
| — | *Interlude: The Perceptron* | A single artificial "brain cell" | Built as a short history interlude (Priya shows Nadia where neural nets came from): weighted sum + activation; why one neuron can't do XOR | ✅ `Perceptron` (interlude) |
| 20 | **Neural Networks & Backprop** | Danger/delight hides in *combinations* of cues (XNOR-like) | k-NN is slow, memoryless, and dumb in high dimensions; linear can't combine → **stack layers that learn features**; backprop ("blame flows backward"); dropout; SGD/Adam | ✅ `FeedforwardNeuralNetwork` |
| 21 | **Convolutional Nets (CNNs)** | Grade latte-art photos / spot pastry defects / scan receipts | Dense nets ignore spatial structure → **convolutions**, filters, pooling (computer vision) | ✅ `ConvolutionalNeuralNetwork` (built from scratch — exceeded the "adopts" plan) |
| 22 | **Recurrent Nets / LSTMs** | Learn *sequence & text representations* (reviews, chat logs) — **not** forecasting (that's Ch 19) | Feedforward has no memory of order; this is the bridge to language → recurrence, memory, vanishing gradients. *Embeddings are introduced here as connective tissue* — learned word/item **vectors** where "similar sits close" (also reused by recsys, Ch 18) | ✅ `RecurrentNeuralNetwork` (built from scratch — RNN + embeddings + BPTT; exceeded the "adopts" plan) |
| 23 | **Transformers & Attention** | The café's AI assistant / chatbot | RNNs forget long context & don't parallelize → **attention**; the modern backbone (and a nod to the model writing this) | ✅ `Transformer` (built from scratch — single self-attention block + BPTT; exceeded the "adopts" plan) |

### Part 5 — Learning by doing: reinforcement learning (Season 4)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 24 | **Multi-Armed Bandits** | Which daily special sells best? | Static prediction never *acts* or learns from outcomes → **explore vs. exploit**; regret; ε-greedy/UCB (the gentle RL entry) | ○ |
| 25 | **Contextual Bandits** | Personalized offers per customer context | One best arm for everyone is too coarse → condition the choice on context | ○ |
| 26 | **MDPs & Q-Learning** | A restocking / pricing **policy** over time | Actions have *delayed* consequences → states, actions, rewards, value, Bellman, Q-learning | ○ |
| 27 | **Deep RL (DQN / Policy Gradients)** | **Chain-scale** optimization where the state space explodes (many stores × SKUs × conditions) | *Tabular Q-learning's table is now impossibly large* → neural nets approximate value/policy. **NB: this is the weakest café hook** — keep it tied to chain-scale optimization, not a gimmicky "robot barista" | ○ (adopts) |

### Part 6 — The frontier (Season 5)
| # | Chapter | Café problem | The Wall → The Idea | Status |
|---|---|---|---|---|
| 28 | **Autoencoders** | Compress & denoise data; anomaly detection with nets | Need compact learned representations → encode→decode | ○ |
| 29 | **Generative Models (VAE/GAN/Diffusion)** | Invent new recipe ideas & marketing images | Don't classify — *create* → generative modeling | ○ (adopts) |
| 30 | **Bayesian / Probabilistic Models** | "How *sure* are we?" before a big bet (signing the lease for store #3, a huge catering order) | Point predictions hide uncertainty → priors, posteriors, reasoning under uncertainty | ○ |

**Cross-cutting "Field Notes"** (recurring sidebars, threaded through the story, not standalone
chapters): evaluation metrics (accuracy, precision/recall, RMSE, ROC), cross-validation,
bias–variance, feature engineering & scaling, data leakage, hyperparameter tuning, and
ethics/fairness. They surface exactly when the story needs them (e.g., precision/recall lands the
moment a "spoiled batch" false negative would be costly).

> This map is deliberately over-complete and **meant to change.** We can reorder, split, merge, or
> drop chapters as the library grows. The point is a shared north star.

---

## Math: what we assume vs. what we teach

**Approach: layered.** The main thread assumes only high-school algebra and stays
picture-and-intuition first, mapped onto real code (matching the site's current tone). Each chapter
then offers an optional collapsible **▸ Go deeper** block with the real derivatives / matrix forms
for readers who want them. *(Adjustable per chapter — some advanced chapters may lean more
technical.)*

**Assumed (prerequisites):** arithmetic & fractions/percentages, exponents & square roots; basic
algebra (evaluate and rearrange `y = mx + b`, substitution); reading an x–y graph and plotting
points; the idea of a function `f(x) → y`. *That's it.*

**Taught just-in-time** (each concept arrives the chapter it's first needed, always with a picture
and the real code):

| Concept | Introduced in | Taught as… |
|---|---|---|
| Σ summation & averages | Ch 1 | adding up the squared misses (MSE) |
| Vectors & matrices (tables of numbers) | Ch 2 | the `Matrix` class = "a page of measurements" |
| Dot product / weighted sum | Ch 2 | features × weights |
| Slope → gradient; learning rate | Ch 1 (deepened Ch 20) | "which way is downhill?"; step size on the error valley |
| Convex vs. non-convex | Ch 1 / 20 | one valley vs. many valleys |
| Sigmoid & probability [0,1]; log-loss | Ch 4 | a soft switch; punishing confident-and-wrong |
| One-hot encoding & argmax | Ch 5 | the answer key as a row of 0s with one 1 |
| Euclidean distance | Ch 6 (reused Ch 12) | Pythagoras, extended to many dimensions |
| Bayes' rule / conditional probability | Ch 7 | updating belief from evidence |
| Entropy / information gain | Ch 8 | how much a question "splits the room" |
| Variance, covariance, principal axes | Ch 15 | the directions data spreads most |
| Chain rule (intuitive) | Ch 20 | blame flowing backward through layers |
| Convolution (intuitive) | Ch 21 | a little stamp sliding over an image |
| Expectation, reward, discounting, Bellman | Ch 24–27 | "what's this choice worth over time?" |
| Probability distributions / uncertainty | Ch 30 | a range of beliefs, not one number |

**Calculus is never a prerequisite** — gradients and the chain rule are taught as intuition
("downhill," "blame flows back"), with the actual derivatives available only in the optional
deep-dive blocks.

---

## How it slots into the codebase

We already know the exact extension points from the existing site (registry-driven MDX +
Canvas playgrounds). Realization is staged so we never block on the whole map:

**Phase 1 — Re-frame the built algorithms into café chapters (no new algorithms). [mostly done]**
- Rewrite the tutorials in `site/src/content/*.mdx` to follow the per-chapter template (Problem
  → Wall → Idea → How It Works → Try It → What Broke), in Nadia's voice. *Done for every built
  algorithm (Ch 0–20 plus the Perceptron interlude). Phase 1 is complete; future chapters are new
  builds (Phase 2+).*
- Re-theme datasets in `site/src/ml/*` (e.g. `datasets.ts`, `clusteringDatasets.ts`) to café data
  (demand, batches, pastries, regulars) **with the same underlying shapes/labels**, so the existing
  Canvas playgrounds and viz keep working unchanged.
- Turn the registry `site/src/algorithms.ts` into an ordered **course outline**: add `chapter`
  number, `part`/`season`, and `status: 'built' | 'roadmap'`; render the sidebar in
  `site/src/components/Layout.tsx` as chapters grouped by Part, with roadmap chapters shown as
  "coming soon."
- Add a course **landing page** (the cold-open: Nadia and the shoebox of receipts) plus the
  Part 0 "Ledger" intro.

**Phase 2+ — Grow into the roadmap, one release at a time.**
- Each new chapter ships with a library release and follows the established add-a-chapter recipe:
  implement the algorithm in `src/lib/machine-learning/{supervised|unsupervised|...}` (reusing
  `Matrix` and the chainable `train`/`predict` interface) → add a themed dataset in `site/src/ml/`
  → add viz in `site/src/viz/` → build a `*Playground.tsx` → register in `algorithms.ts` → write the
  MDX chapter. A roadmap chapter flips to `status: 'built'`.
- Some advanced chapters (Transformers, deep RL, diffusion) may be **explainer chapters** with
  bespoke illustrative playgrounds rather than full from-scratch library implementations — flagged
  per chapter so scope stays honest. New `src/lib` directories (e.g. `reinforcement/`,
  `dimensionality-reduction/`) get added as paradigms arrive.

---

## Verification

- **Preview locally:** `cd site && yarn dev`, then read each re-themed chapter end to end and
  confirm every playground still trains/animates with the café datasets.
- **Builds green:** `cd site && yarn build && yarn typecheck`; library `yarn test` still passes
  (Phase 1 changes are prose + datasets, not algorithm logic).
- **Narrative check:** each chapter's "What Broke" teaser names a limitation that the *next*
  chapter's "Wall" pays off — read consecutive chapters to confirm the chain holds.
- **Deploy:** existing `deploy-site.yml` ships it to GitHub Pages on merge.

---

## Candidate future chapters (the framing suggests these)

Because the course files by **problem type × business stage**, the café setting actively proposes
strong additions. Leading candidate first:

- **Causal Inference / Uplift Modeling** — *"Did the promo **cause** the sales bump, or were those
  customers coming anyway?"* An excellent café fit and a notable gap in the current map; slots into
  Part 5 (decisions) or a new mini-part "Did it actually work?". Pairs naturally with the bandits
  chapters.
- **Gaussian Mixture Models** — soft/probabilistic clustering, a natural follow-on to k-means (Part 3).
- **Online / Streaming Learning** — *"the café never closes; data keeps arriving"* — models that
  update continuously (Part 4).
- **Graph ML / GNNs** — the supplier network or the social graph of customers (Part 4).
- **Optimizers & gradient-descent variants** (SGD/Adam) — could be a Field Note or a deeper block
  inside the neural-network chapter rather than its own chapter.

## Open questions to revisit later (non-blocking)

- Final course/café/character names (placeholders above are good but swappable).
- Exact chapter ordering in Part 2 vs Part 3 (trees/ensembles before or after first clustering?).
- How many "frontier" chapters are full implementations vs. illustrated explainers (see the
  *builds → adopts* principle — chapters tagged "adopts" lean explainer).
- Whether to add a light per-chapter "exercise / your turn" beyond the playground.
