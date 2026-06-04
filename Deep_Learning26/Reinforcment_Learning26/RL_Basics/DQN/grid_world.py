from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable, Callable, Optional

State = Tuple[int, int]            # (row, col)
Action = str                       # 'U','D','L','R'

@dataclass
class GridWorld:
    rows: int = 3
    cols: int = 3
    terminal: List[State] = ((2, 2),)                
    obstacles: List[State] = ()                     
    reward_map: Dict[State, float] = None           
    slip_prob: float = 0.0                          
    step_cost: float = -1.0                         
    treasure_bonus: float = 10.0                    

    def __post_init__(self):
        # Convert JSON-loaded lists of lists to tuples of ints
        self.terminal = [tuple(s) for s in self.terminal]
        self.obstacles = [tuple(s) for s in self.obstacles]
        
        # sanity checks
        for s in self.terminal + self.obstacles:
            assert 0 <= s[0] < self.rows and 0 <= s[1] < self.cols, f"Invalid cell {s}"
        self.actions: List[Action] = ['U', 'D', 'L', 'R']

        # default reward map (only terminal gets extra bonus)
        if self.reward_map is None:
            self.reward_map = {}
            for s in self.terminal:
                self.reward_map[s] = self.step_cost + self.treasure_bonus

        # pre‑compute neighbours for deterministic case
        self._deterministic_next: Dict[Tuple[State, Action], State] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                s = (r, c)
                if s in self.obstacles:
                    continue
                for a in self.actions:
                    self._deterministic_next[(s, a)] = self._move(s, a)

    def _move(self, state: State, a: Action) -> State:
        if state in self.terminal:
            return state
        r, c = state
        if   a == 'U': r = max(r - 1, 0)
        elif a == 'D': r = min(r + 1, self.rows - 1)
        elif a == 'L': c = max(c - 1, 0)
        elif a == 'R': c = min(c + 1, self.cols - 1)

        nxt = (r, c)
        if nxt in self.obstacles:
            nxt = state
        return nxt

    def transition(self, state: State, a: Action) -> List[Tuple[float, State, float]]:
        if state in self.terminal:
            return [(1.0, state, 0.0)]

        main_next = self._deterministic_next[(state, a)]

        if self.slip_prob == 0.0:
            return [(1.0, main_next, self._reward(main_next))]

        orthogonal = {'U': ('L', 'R'), 'D': ('L', 'R'),
                      'L': ('U', 'D'), 'R': ('U', 'D')}[a]

        probs = [
            (1.0 - self.slip_prob, main_next),
            (self.slip_prob / 2.0, self._deterministic_next[(state, orthogonal[0])]),
            (self.slip_prob / 2.0, self._deterministic_next[(state, orthogonal[1])])
        ]

        # collapse duplicate next-states
        outcome: Dict[State, float] = {}
        for p, nxt in probs:
            outcome[nxt] = outcome.get(nxt, 0.0) + p

        return [(p, s, self._reward(s)) for s, p in outcome.items()]

    def _reward(self, nxt: State) -> float:
        base = self.step_cost
        if nxt in self.reward_map:
            base = self.reward_map[nxt]          
        return base

    @property
    def all_states(self) -> List[State]:
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) not in self.obstacles]

    @property
    def non_terminal_states(self) -> List[State]:
        return [s for s in self.all_states if s not in self.terminal]