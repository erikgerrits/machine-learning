import type { ComponentType } from 'react';
import { NeuralNetworkPlayground } from './components/NeuralNetworkPlayground';

/**
 * The catalog of algorithms the site covers. The neural network is live (flagship); the rest
 * are scaffolded as "coming soon" and slot in by flipping `status` and adding a Playground +
 * tutorial page. The sidebar nav, landing cards, and routes all derive from this list.
 */
export interface AlgorithmEntry {
    id: string;
    path: string;
    title: string;
    tagline: string;
    status: 'live' | 'soon';
    Playground?: ComponentType;
}

export const ALGORITHMS: AlgorithmEntry[] = [
    {
        id: 'neural-network',
        path: '/neural-network',
        title: 'Neural Network',
        tagline: 'Bends curved boundaries no straight line could draw.',
        status: 'live',
        Playground: NeuralNetworkPlayground,
    },
    {
        id: 'linear-regression',
        path: '/linear-regression',
        title: 'Linear Regression',
        tagline: 'Fits the best straight line through a cloud of points.',
        status: 'soon',
    },
    {
        id: 'logistic-regression',
        path: '/logistic-regression',
        title: 'Logistic Regression',
        tagline: 'Splits two classes with a straight decision boundary.',
        status: 'soon',
    },
    {
        id: 'multiclass-logistic-regression',
        path: '/multiclass-logistic-regression',
        title: 'Multiclass Logistic',
        tagline: 'One-vs-rest classifiers carve up many classes.',
        status: 'soon',
    },
    {
        id: 'nearest-neighbors',
        path: '/nearest-neighbors',
        title: 'Nearest Neighbors',
        tagline: 'Classifies each point by the company it keeps.',
        status: 'soon',
    },
];
