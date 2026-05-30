#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# --------------------------------------------------------------
# 1.  Grid world definition
# --------------------------------------------------------------
rows, cols = 3, 3
states = [(r, c) for r in range(rows) for c in range(cols)]

terminal = (2, 2)                 # the treasure cell
actions = ['U', 'D', 'L', 'R']
gamma   = 0.9                     # discount factor

def step(state, a):
    """Deterministic transition function."""
    if state == terminal:
        return terminal, 0                # stay in terminal, no reward

    r, c = state
    if   a == 'U': nr, nc = max(r - 1, 0), c
    elif a == 'D': nr, nc = min(r + 1, rows - 1), c
    elif a == 'L': nr, nc = r, max(c - 1, 0)
    elif a == 'R': nr, nc = r, min(c + 1, cols - 1)

    nxt = (nr, nc)

    # reward model
    reward = -1                     # cost of taking a step
    if nxt == terminal:
        reward = 9                  # -1 + 10 (the treasure bonus)

    return nxt, reward

# --------------------------------------------------------------
# 2.  Helper structures for the linear system
# --------------------------------------------------------------
# Map each (non‑terminal) state to a unique index 0 … N‑1
non_term_states = [s for s in states if s != terminal]
idx_of = {s: i for i, s in enumerate(non_term_states)}
N = len(non_term_states)          # number of unknown V‑values

# -----------------------------------------------------------------
# 3.  Initialise a (deterministic) policy – here we start with a
#     uniform random choice, later we will store only ONE action per
#     state.
# -----------------------------------------------------------------
policy = {s: np.random.choice(actions) for s in non_term_states} #policy=action??
policy[terminal] = None           # terminal has no action

# --------------------------------------------------------------
# 4.  Policy‑iteration loop
# --------------------------------------------------------------
max_outer_iters = 1000            # safety net
theta = 1e-6                      # not needed for the linear solve,
                                 # but used for an optional iterative
                                 # policy‑evaluation fallback.

for outer in range(max_outer_iters):  #this iterates overe both loops!
    # ----------------------------------------------------------
    # 4.1 Policy evaluation  (solve V = R + gamma * P V)
    # ----------------------------------------------------------
    # Build the linear system A V = b   (A is NxN, b is Nx1)
    A = np.zeros((N, N))
    b = np.zeros(N)

    for s in non_term_states:
        i = idx_of[s]
        a = policy[s]                         # current action for state s
        s_next, r = step(s, a)

        # Left‑hand side: V(s) - gamma * V(s_next) = r
        A[i, i] = 1.0                         # V(s) coefficient

        if s_next != terminal:               # terminal V = 0, drop term
            j = idx_of[s_next]
            A[i, j] -= gamma                 # -gamma * V(s_next)

        b[i] = r

    # Solve the linear system
    V_vec = np.linalg.solve(A, b)

    # Convert back to a dict for easy lookup
    V = {terminal: 0.0}
    V.update({s: V_vec[idx_of[s]] for s in non_term_states})

    # ----------------------------------------------------------
    # 4.2 Policy improvement
    # ----------------------------------------------------------
    policy_stable = True

    for s in non_term_states:
        # compute Q(s,a) for every possible action
        q_vals = {}
        for a in actions:
            s_next, r = step(s, a)
            q_vals[a] = r + gamma * V[s_next]

        # greedy action w.r.t. the *new* V
        best_a = max(q_vals, key=q_vals.get)

        if best_a != policy[s]:
            policy[s] = best_a
            policy_stable = False  #if it keeps changing after max iter reached it does not converge! 

    # ----------------------------------------------------------
    # 4.3 Termination test
    # ----------------------------------------------------------
    if policy_stable:
        print(f"Policy iteration converged after {outer+1} outer iterations.")
        break
else:
    print("WARNING: reached max_outer_iters without convergence.")

# --------------------------------------------------------------
# 5.  Pretty‑print the final policy
# --------------------------------------------------------------
print("\nOptimal policy (T = terminal):")
for r in range(rows):
    row = ''
    for c in range(cols):
        a = policy[(r, c)]
        row += f'{a or "T":4}'
    print(row)

# --------------------------------------------------------------
# 6.  (Optional) pretty‑print the final state‑value function
# --------------------------------------------------------------
print("\nState‑value function V*:")
for r in range(rows):
    row = ''
    for c in range(cols):
        row += f'{V[(r,c)]:6.2f} '
    print(row)