import { lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { TutorialPage } from './components/TutorialPage';
import { Landing } from './pages/Landing';
import { NotFound } from './pages/NotFound';

// Each chapter is lazy-loaded so its playground + MDX land in their own chunk, fetched only when
// the chapter is opened. The Layout (header + sidebar) and Landing stay in the initial bundle; the
// Suspense boundary that catches these lazy loads lives in Layout, around the content <Outlet>.
const TheLedgerTutorial = lazy(() => import('./content/the-ledger.mdx'));
const NeuralNetworkTutorial = lazy(() => import('./content/neural-network.mdx'));
const LinearRegressionTutorial = lazy(() => import('./content/linear-regression.mdx'));
const ManyFeaturesTutorial = lazy(() => import('./content/many-features.mdx'));
const OverfittingTutorial = lazy(() => import('./content/overfitting.mdx'));
const LogisticRegressionTutorial = lazy(() => import('./content/logistic-regression.mdx'));
const MulticlassLogisticRegressionTutorial = lazy(() => import('./content/multiclass-logistic-regression.mdx'));
const NearestNeighborsTutorial = lazy(() => import('./content/nearest-neighbors.mdx'));
const NaiveBayesTutorial = lazy(() => import('./content/naive-bayes.mdx'));
const DecisionTreesTutorial = lazy(() => import('./content/decision-trees.mdx'));
const RandomForestsTutorial = lazy(() => import('./content/random-forests.mdx'));
const GradientBoostingTutorial = lazy(() => import('./content/gradient-boosting.mdx'));
const SupportVectorMachinesTutorial = lazy(() => import('./content/support-vector-machines.mdx'));
const KMeansTutorial = lazy(() => import('./content/k-means.mdx'));
const HierarchicalClusteringTutorial = lazy(() => import('./content/hierarchical-clustering.mdx'));
const DBSCANTutorial = lazy(() => import('./content/dbscan.mdx'));
const PCATutorial = lazy(() => import('./content/pca.mdx'));
const AnomalyDetectionTutorial = lazy(() => import('./content/anomaly-detection.mdx'));
const AssociationRulesTutorial = lazy(() => import('./content/association-rules.mdx'));
const RecommenderSystemsTutorial = lazy(() => import('./content/recommender-systems.mdx'));
const TimeSeriesTutorial = lazy(() => import('./content/time-series.mdx'));
const PerceptronTutorial = lazy(() => import('./content/perceptron.mdx'));
const CNNTutorial = lazy(() => import('./content/cnn.mdx'));
const RNNTutorial = lazy(() => import('./content/rnn.mdx'));
const TransformerTutorial = lazy(() => import('./content/transformer.mdx'));

export function App() {
    return (
        <Routes>
            <Route element={<Layout />}>
                <Route index element={<Landing />} />
                <Route
                    path="the-ledger"
                    element={
                        <TutorialPage>
                            <TheLedgerTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="neural-network"
                    element={
                        <TutorialPage>
                            <NeuralNetworkTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="linear-regression"
                    element={
                        <TutorialPage>
                            <LinearRegressionTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="many-features"
                    element={
                        <TutorialPage>
                            <ManyFeaturesTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="overfitting"
                    element={
                        <TutorialPage>
                            <OverfittingTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="logistic-regression"
                    element={
                        <TutorialPage>
                            <LogisticRegressionTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="multiclass-logistic-regression"
                    element={
                        <TutorialPage>
                            <MulticlassLogisticRegressionTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="nearest-neighbors"
                    element={
                        <TutorialPage>
                            <NearestNeighborsTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="naive-bayes"
                    element={
                        <TutorialPage>
                            <NaiveBayesTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="decision-trees"
                    element={
                        <TutorialPage>
                            <DecisionTreesTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="random-forests"
                    element={
                        <TutorialPage>
                            <RandomForestsTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="gradient-boosting"
                    element={
                        <TutorialPage>
                            <GradientBoostingTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="support-vector-machines"
                    element={
                        <TutorialPage>
                            <SupportVectorMachinesTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="k-means"
                    element={
                        <TutorialPage>
                            <KMeansTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="hierarchical-clustering"
                    element={
                        <TutorialPage>
                            <HierarchicalClusteringTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="dbscan"
                    element={
                        <TutorialPage>
                            <DBSCANTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="pca"
                    element={
                        <TutorialPage>
                            <PCATutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="anomaly-detection"
                    element={
                        <TutorialPage>
                            <AnomalyDetectionTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="association-rules"
                    element={
                        <TutorialPage>
                            <AssociationRulesTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="recommender-systems"
                    element={
                        <TutorialPage>
                            <RecommenderSystemsTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="time-series"
                    element={
                        <TutorialPage>
                            <TimeSeriesTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="perceptron"
                    element={
                        <TutorialPage>
                            <PerceptronTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="cnn"
                    element={
                        <TutorialPage>
                            <CNNTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="rnn"
                    element={
                        <TutorialPage>
                            <RNNTutorial />
                        </TutorialPage>
                    }
                />
                <Route
                    path="transformer"
                    element={
                        <TutorialPage>
                            <TransformerTutorial />
                        </TutorialPage>
                    }
                />
                <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
    );
}
