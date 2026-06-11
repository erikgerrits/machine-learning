// Grid-world maps for the Q-learning playground, plus a tiny environment that turns a map into the
// (state, action) → (nextState, reward, done) transitions the agent learns from. Each map is drawn as
// ASCII for readability:
//   S = start (the kitchen)    G = goal (the table, +goalReward, ends the episode)
//   X = hazard (a spill, hazardReward, ends the episode)    # = wall (can't enter; bumping it stays)
//   . = open floor (each step costs stepReward, a small "hurry up" nudge)
// State is row-major: state = row * cols + col. Actions: 0 up, 1 right, 2 down, 3 left.

export interface GridworldMap {
    id: string;
    label: string;
    blurb: string;
    layout: string[];
    stepReward: number;
    goalReward: number;
    hazardReward: number;
}

export type CellType = 'empty' | 'wall' | 'start' | 'goal' | 'hazard';

export const GRIDWORLD_MAPS: GridworldMap[] = [
    {
        id: 'cafe-floor',
        label: 'Café floor',
        blurb: 'An open floor with a single spill. Watch the value heatmap glow outward from the table and the arrows all swing to point home.',
        layout: [
            '.....G',
            '......',
            '..X...',
            '......',
            'S.....',
        ],
        stepReward: -0.04,
        goalReward: 1,
        hazardReward: -1,
    },
    {
        id: 'cliff-walk',
        label: 'Cliff walk',
        blurb: 'The short route home runs right along a row of spills. One slip ends the trip — so the runner learns to climb up and around rather than risk the edge.',
        layout: [
            '.......',
            '.......',
            '.......',
            'SXXXXXG',
        ],
        stepReward: -0.04,
        goalReward: 1,
        hazardReward: -1,
    },
    {
        id: 'back-room',
        label: 'The back room',
        blurb: 'A wall splits the floor in two with a single gap. The reward only makes sense once value has flowed back through that doorway — watch the far room stay dark until it does.',
        layout: [
            'S....',
            '.....',
            '###.#',
            '.....',
            '....G',
        ],
        stepReward: -0.04,
        goalReward: 1,
        hazardReward: -1,
    },
];

const ACTION_DELTAS = [
    [-1, 0], // 0 up
    [0, 1],  // 1 right
    [1, 0],  // 2 down
    [0, -1], // 3 left
];

export interface Gridworld {
    rows: number;
    cols: number;
    numberOfStates: number;
    numberOfActions: number;
    start: number;
    cellTypes: CellType[];
    step: (state: number, action: number) => { nextState: number; reward: number; done: boolean };
    isTerminal: (state: number) => boolean;
}

const SYMBOL_TO_TYPE: Record<string, CellType> = {
    '.': 'empty', '#': 'wall', 'S': 'start', 'G': 'goal', 'X': 'hazard',
};

/** Compile a map into a deterministic grid-world environment. */
export function makeGridworld(map: GridworldMap): Gridworld {
    const rows = map.layout.length;
    const cols = map.layout[0].length;
    const cellTypes: CellType[] = [];
    let start = 0;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const type = SYMBOL_TO_TYPE[map.layout[r][c]] ?? 'empty';
            cellTypes.push(type);
            if (type === 'start') start = r * cols + c;
        }
    }

    const isTerminal = (state: number) => cellTypes[state] === 'goal' || cellTypes[state] === 'hazard';

    const step = (state: number, action: number) => {
        const [dr, dc] = ACTION_DELTAS[action];
        let row = Math.floor(state / cols) + dr;
        let col = (state % cols) + dc;
        let nextState = row * cols + col;

        // Off the grid or into a wall → stay put (but still pay the step cost).
        if (row < 0 || row >= rows || col < 0 || col >= cols || cellTypes[nextState] === 'wall') {
            nextState = state;
        }

        if (cellTypes[nextState] === 'goal') return { nextState, reward: map.goalReward, done: true };
        if (cellTypes[nextState] === 'hazard') return { nextState, reward: map.hazardReward, done: true };
        return { nextState, reward: map.stepReward, done: false };
    };

    return { rows, cols, numberOfStates: rows * cols, numberOfActions: 4, start, cellTypes, step, isTerminal };
}
