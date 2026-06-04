import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random
import logging
from typing import Dict, Tuple, List
from tqdm import tqdm
from grid_world import GridWorld, State, Action

log = logging.getLogger(__name__)

class PolicyIteration:
    def __init__(self,
                 env: GridWorld,
                 gamma: float = 0.9,
                 max_iters: int = 1000,
                 eps: float = 1e-8):
        self.env = env
        self.gamma = gamma
        self.max_iters = max_iters
        self.eps = eps

        self.state_idx = {s: i for i, s in enumerate(env.non_terminal_states)}
        self.N = len(self.state_idx)

        # Initialize a deterministic random policy
        self.policy: Dict[State, Action] = {
            s: random.choice(self.env.actions) for s in env.non_terminal_states
        }

    def _evaluate_policy(self) -> np.ndarray:
        rows, cols, data = [], [], []
        b = np.zeros(self.N)

        for s, i in self.state_idx.items():
            a = self.policy[s]
            transitions = self.env.transition(s, a)

            rows.append(i); cols.append(i); data.append(1.0)

            for prob, nxt, reward in transitions:
                b[i] += prob * reward
                if nxt in self.env.terminal:
                    continue
                j = self.state_idx[nxt]
                rows.append(i); cols.append(j); data.append(-self.gamma * prob)

        A = sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        V = spla.spsolve(A, b)

        if np.isnan(V).any():
            raise RuntimeError("Linear system produced NaNs – check parameters.")

        return V

    def _improve_policy(self, V_vec: np.ndarray) -> bool:
        stable = True

        V = {s: V_vec[i] for s, i in self.state_idx.items()}
        V.update({t: 0.0 for t in self.env.terminal})   

        for s in self.env.non_terminal_states:
            q_vals: Dict[Action, float] = {}
            for a in self.env.actions:
                q = 0.0
                for prob, nxt, reward in self.env.transition(s, a):
                    q += prob * (reward + self.gamma * V[nxt])
                q_vals[a] = q

            best_a = max(q_vals, key=q_vals.get)
            if best_a != self.policy[s]:
                self.policy[s] = best_a
                stable = False

        return stable

    def run(self) -> Tuple[Dict[State, float], Dict[State, Action]]:
        for it in tqdm(range(self.max_iters), desc="Policy-Iteration"):
            V_vec = self._evaluate_policy()
            if self._improve_policy(V_vec):
                log.info(f"Converged after {it+1} iterations.")
                break
        else:
            log.warning("Reached max_iters without convergence.")

        V = {t: 0.0 for t in self.env.terminal}
        V.update({s: V_vec[i] for s, i in self.state_idx.items()})
        return V, self.policy