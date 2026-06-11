import { describe, it, expect } from 'vitest';
import QLearning from '../lib/machine-learning/reinforcement/QLearning';

// ── A 5-state corridor: 0 1 2 3 [4=goal]. Action 0 = left, 1 = right. Reaching state 4 pays +1 and
// ends the episode; every other step pays 0. The only optimal move anywhere is "right", and the value
// of each state should be the goal reward discounted by how many steps away it is: γ^(distance−1).
function corridorStep(state: number, action: number) {
    const nextState = action === 1 ? Math.min(state + 1, 4) : Math.max(state - 1, 0);
    const done = nextState === 4;
    return { nextState, reward: done ? 1 : 0, done };
}

// ── A 3×3 grid (row-major state = row*3 + col):
//   . . G        start at bottom-left (6), goal at top-right (2), a hazard in the middle (4).
//   . # .        Actions: 0 up, 1 right, 2 down, 3 left. Stepping off-grid stays put. Reaching the
//   S . .        goal pays +1 (done), the hazard −1 (done), any other step −0.04.
const GRID_COLS = 3;
const GOAL = 2;
const HAZARD = 4;
function gridStep(state: number, action: number) {
    let row = Math.floor(state / GRID_COLS);
    let col = state % GRID_COLS;
    if (action === 0) row -= 1;
    else if (action === 1) col += 1;
    else if (action === 2) row += 1;
    else col -= 1;
    if (row < 0 || row > 2 || col < 0 || col > 2) { row = Math.floor(state / GRID_COLS); col = state % GRID_COLS; }
    const nextState = row * GRID_COLS + col;
    if (nextState === GOAL) return { nextState, reward: 1, done: true };
    if (nextState === HAZARD) return { nextState, reward: -1, done: true };
    return { nextState, reward: -0.04, done: false };
}

function train(agent: QLearning, step: (s: number, a: number) => { nextState: number; reward: number; done: boolean }, start: number, episodes: number, maxSteps = 50) {
    for (let e = 0; e < episodes; e++) {
        let state = start;
        for (let t = 0; t < maxSteps; t++) {
            const action = agent.selectAction(state);
            const { nextState, reward, done } = step(state, action);
            agent.update(state, action, reward, nextState, done);
            if (done) break;
            state = nextState;
        }
    }
}

describe('QLearning', () => {

    it('learns to walk toward the goal and values states by discounted distance', () => {
        const agent = new QLearning().setNumberOfStates(5).setNumberOfActions(2)
            .setLearningRate(0.5).setDiscountFactor(0.9).setEpsilon(0.2).setSeed(0);
        train(agent, corridorStep, 0, 3000);

        // Optimal policy: go right from every non-goal state.
        expect(agent.bestAction(0)).toBe(1);
        expect(agent.bestAction(3)).toBe(1);

        // Values fall off as γ^(distance−1): one step away ≈ 1, then 0.9, 0.81, 0.729.
        expect(agent.getValue(3)).toBeCloseTo(1, 1);
        expect(agent.getValue(2)).toBeCloseTo(0.9, 1);
        expect(agent.getValue(1)).toBeCloseTo(0.81, 1);
        expect(agent.getValue(0)).toBeCloseTo(0.729, 1);
    });

    it('finds a path through a grid that reaches the goal and avoids the hazard', () => {
        const agent = new QLearning().setNumberOfStates(9).setNumberOfActions(4)
            .setLearningRate(0.4).setDiscountFactor(0.95).setEpsilon(0.2).setSeed(1);
        train(agent, gridStep, 6, 4000);

        // Follow the learned greedy policy from the start; it should march to the goal, never the hazard.
        let state = 6;
        const visited = [state];
        for (let t = 0; t < 12 && state !== GOAL; t++) {
            state = gridStep(state, agent.bestAction(state)).nextState;
            visited.push(state);
        }
        expect(state).toBe(GOAL);
        expect(visited).not.toContain(HAZARD);
    });

    it('drops the future term on terminal transitions', () => {
        // Give state 1 a big value, then feed state 0 → state 1 transitions that are flagged terminal.
        // A terminal update must NOT bootstrap off state 1's value, so Q(0,0) should settle at the
        // reward (1), not reward + γ·V(1).
        const terminal = new QLearning().setNumberOfStates(2).setNumberOfActions(1).setLearningRate(0.5).setDiscountFactor(0.9);
        for (let i = 0; i < 50; i++) terminal.update(1, 0, 5, 1, true);   // V(1) ≈ 5
        for (let i = 0; i < 50; i++) terminal.update(0, 0, 1, 1, true);   // terminal: ignore V(1)
        expect(terminal.getQValue(0, 0)).toBeCloseTo(1, 5);

        // The same transitions treated as non-terminal DO bootstrap → reward + γ·5 = 5.5.
        const bootstrap = new QLearning().setNumberOfStates(2).setNumberOfActions(1).setLearningRate(0.5).setDiscountFactor(0.9);
        for (let i = 0; i < 50; i++) bootstrap.update(1, 0, 5, 1, true);
        for (let i = 0; i < 80; i++) bootstrap.update(0, 0, 1, 1, false);
        expect(bootstrap.getQValue(0, 0)).toBeCloseTo(5.5, 1);
    });

    it('is deterministic for a fixed seed', () => {
        const run = () => {
            const agent = new QLearning().setNumberOfStates(5).setNumberOfActions(2).setEpsilon(0.3).setSeed(4);
            train(agent, corridorStep, 0, 200);
            return agent.getPolicy().concat(agent.getQValues(2));
        };
        expect(run()).toEqual(run());
    });

    it('round-trips its hyperparameters and chains setters', () => {
        const agent = new QLearning();
        expect(agent.getDiscountFactor()).toBe(0.95);

        const returned = agent.setNumberOfStates(20).setNumberOfActions(4).setLearningRate(0.2).setDiscountFactor(0.8).setEpsilon(0.05).setSeed(6);
        expect(returned).toBe(agent);
        expect(agent.getNumberOfStates()).toBe(20);
        expect(agent.getNumberOfActions()).toBe(4);
        expect(agent.getLearningRate()).toBe(0.2);
        expect(agent.getDiscountFactor()).toBe(0.8);
        expect(agent.getEpsilon()).toBe(0.05);
        expect(agent.getSeed()).toBe(6);
    });
});
