// Scenarios for the deep-RL playground. Unlike Chapter 26's tiled grid, the floor here is *continuous*:
// the runner's state is a real position (x, y) in [0,1]², so there's no table of cells to fill — a
// neural network has to learn a value for the whole surface. The scenarios differ only in where the
// table (the goal) sits, which gives each a distinct value landscape. Actions: 0 up, 1 right, 2 down,
// 3 left (each a fixed step). Reaching within `radius` of the table pays +1 and ends the trip; every
// other step pays 0, so discounting alone rewards shorter routes — and values stay in (0,1), which is
// what the network's sigmoid output can represent.

export interface NavScenario {
    id: string;
    label: string;
    blurb: string;
    goal: [number, number];
    radius: number;
    step: number;
}

export const NAV_SCENARIOS: NavScenario[] = [
    {
        id: 'far-table',
        label: 'Corner table',
        blurb: 'The table sits in the far corner. Watch the value surface bloom out from it across the whole floor and the arrows everywhere swing to point home.',
        goal: [0.85, 0.85],
        radius: 0.18,
        step: 0.12,
    },
    {
        id: 'window-seat',
        label: 'Window seat',
        blurb: 'The table moves to the middle of the top wall. Same runner, same network — a completely different value landscape learned from scratch.',
        goal: [0.5, 0.9],
        radius: 0.18,
        step: 0.12,
    },
    {
        id: 'corner-booth',
        label: 'Corner booth',
        blurb: 'A table low on the right. Notice the network gives sensible values for far-off spots it rarely visits — the generalisation a lookup table can’t.',
        goal: [0.85, 0.15],
        radius: 0.18,
        step: 0.12,
    },
];

const ACTION_DELTAS = [
    [0, 1],  // 0 up    (+y)
    [1, 0],  // 1 right (+x)
    [0, -1], // 2 down  (−y)
    [-1, 0], // 3 left  (−x)
];

export interface NavEnv {
    stateSize: number;
    numberOfActions: number;
    goal: [number, number];
    radius: number;
    randomStart: (rand: () => number) => number[];
    step: (state: number[], action: number) => { nextState: number[]; reward: number; done: boolean };
}

/** Compile a scenario into a continuous navigation environment. */
export function makeNavEnv(scenario: NavScenario): NavEnv {
    const { goal, radius, step } = scenario;
    const dist = (x: number, y: number) => Math.hypot(x - goal[0], y - goal[1]);

    return {
        stateSize: 2,
        numberOfActions: 4,
        goal,
        radius,
        randomStart: (rand: () => number) => [rand(), rand()], // anywhere on the floor
        step: (state: number[], action: number) => {
            const [dx, dy] = ACTION_DELTAS[action];
            const x = Math.min(1, Math.max(0, state[0] + dx * step)); // bumping a wall just clamps
            const y = Math.min(1, Math.max(0, state[1] + dy * step));
            if (dist(x, y) < radius) return { nextState: [x, y], reward: 1, done: true };
            return { nextState: [x, y], reward: 0, done: false };
        },
    };
}
