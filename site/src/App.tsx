import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { TutorialPage } from './components/TutorialPage';
import { Landing } from './pages/Landing';
import { NotFound } from './pages/NotFound';
import NeuralNetworkTutorial from './content/neural-network.mdx';
import LinearRegressionTutorial from './content/linear-regression.mdx';
import LogisticRegressionTutorial from './content/logistic-regression.mdx';
import MulticlassLogisticRegressionTutorial from './content/multiclass-logistic-regression.mdx';
import NearestNeighborsTutorial from './content/nearest-neighbors.mdx';

export function App() {
    return (
        <Routes>
            <Route element={<Layout />}>
                <Route index element={<Landing />} />
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
                <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
    );
}
