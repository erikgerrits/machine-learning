import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { TutorialPage } from './components/TutorialPage';
import { Landing } from './pages/Landing';
import { NotFound } from './pages/NotFound';
import NeuralNetworkTutorial from './content/neural-network.mdx';
import LinearRegressionTutorial from './content/linear-regression.mdx';

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
                <Route path="*" element={<NotFound />} />
            </Route>
        </Routes>
    );
}
