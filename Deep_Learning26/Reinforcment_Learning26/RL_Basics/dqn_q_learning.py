#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q‑learning for the 3×3 deterministic grid‑world from the original script.
No neural networks – just a NumPy Q‑table.

The environment is exactly the same:
    * 9 states (row, col)  – (2,2) is the terminal treasure.
    * 4 deterministic actions: U, D, L, R
    * Reward = -1 per step, +9 when stepping onto the treasure (i.e. -1 + 10).
    * Discount factor γ = 0.9
"""

import numpy as np
import random
from collections import defaultdict

# --------------------------------------------------------------
# 1.  Grid world definition (identical to the original code)
# --------------------------------------------------------------
rows, cols = 3, 3
states = [(r, c) for r in range(rows) for c in range(cols)]

terminal = (2, 2)                 # treasure cell
actions = ['U', 'D', 'L', 'R']
gamma   = 0.9                     # discount factor

def step(state, a):
    """Deterministic transition function (environment simulator)."""
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
# 2.  Q‑learning hyper‑parameters (feel free to tune)
# --------------------------------------------------------------
num_episodes   = 5000          # how many episodes to run
max_steps_per_episode = 100   # safety cap (the grid is tiny)
epsilon_start  = 1.0          # initial exploration prob.
epsilon_end    = 0.05         # final exploration prob.
epsilon_decay  = 0.999        # exponential decay factor

alpha_start    = 0.5          # learning‑rate at episode 0
alpha_end      = 0.01         # learning‑rate at the end
alpha_decay    = 0.999        # exponential decay factor

# --------------------------------------------------------------
# 3.  Initialise a Q‑table: dict(state) -> np.array(len(actions))
# --------------------------------------------------------------
# Use a defaultdict so that any unseen state gets a zero‑filled vector.
Q = defaultdict(lambda: np.zeros(len(actions)))
print(Q)
# Helper to map action string ↔ index
action_to_idx = {a: i for i, a in enumerate(actions)}
idx_to_action = {a: i for a, i in enumerate(actions)}

def epsilon_greedy(state, eps):
    """Return an action (string) using ε‑greedy on the current Q‑table."""
    if random.random() < eps:
        return random.choice(actions)               # explore
    q_vals = Q[state]
    max_idx = np.flatnonzero(q_vals == q_vals.max())  # break ties randomly
    action_idx = int(random.choice(max_idx))
    return idx_to_action[action_idx]               # exploit

# --------------------------------------------------------------
# 4.  Training loop
# --------------------------------------------------------------
epsilon = epsilon_start
alpha   = alpha_start

for ep in range(1, num_episodes + 1):
    state = (0, 0)                     # start each episode in the upper‑left corner
    total_reward = 0

    for step_i in range(max_steps_per_episode):
        # 4.1 Choose action (ε‑greedy)
        a = epsilon_greedy(state, epsilon)   #chooses actions stochastically (epsilon-greedy) so that agent explores unseen state-action pairs. ,-->previous programs never explore, they evaluate all actions analytically. 
        a_idx = action_to_idx[a]

        # 4.2 Interact with environment
        next_state, reward = step(state, a)   #called only once to sample the result of the action the agent actually chose; without Q-learning:inside for each action loop for every state; "gateway to learning"
        total_reward += reward

        # 4.3 TD‑update for Q
        #    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') – Q(s,a) ]
        best_next_q = 0.0 if next_state == terminal else Q[next_state].max()  #looks up the current learned Q-values of sampled successor state, does not ask for transition probabilities, wittout Q-learning: r+ gamma*V[s_next] of known , updatedV
        print(Q)
        td_target = reward + gamma * best_next_q  #TD update ; the only place the learned paraemters change. uses sampled reward and next state, plus Q[next_state].max()
        td_error = td_target - Q[state][a_idx]   #TD update: wihtout Q -learning (previous programs) deterministic Bellman backup : V_new = max_a (r+ gamma*V[s′]); exact backup, no learning-rate, no TD error, and no dependence on a previousl-learned estimate of V[s′] 
        Q[state][a_idx] += alpha * td_error

        # 4.4 Move to next state (or break if terminal)
        if next_state == terminal:
            break
        state = next_state

    # ----------------------------------------------------------
    # 4.5 Decay exploration & learning rates (optional schedule)
    # ----------------------------------------------------------
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    alpha   = max(alpha_end,   alpha   * alpha_decay)

    # (optional) print progress every 500 episodes
    if ep % 500 == 0:
        print(f"Episode {ep:4d} | total_reward = {total_reward:4d} | ε = {epsilon:.3f} | α = {alpha:.3f}")

# --------------------------------------------------------------
# 5.  Extract the greedy policy and the state‑value function
# --------------------------------------------------------------
policy = {}
V = {}
for s in states:
    if s == terminal:
        policy[s] = None
        V[s] = 0.0
    else:
        q_vals = Q[s]
        best_idx = np.flatnonzero(q_vals == q_vals.max())
        action_idx = int(random.choice(best_idx))
        best_a   = idx_to_action[action_idx]   # break ties randomly
        policy[s] = best_a
        V[s] = q_vals.max()

# --------------------------------------------------------------
# 6.  Pretty‑print the results (identical format to your original script)
# --------------------------------------------------------------
print("\nOptimal policy (T = terminal):")
for r in range(rows):
    row = ''
    for c in range(cols):
        a = policy[(r, c)]
        row += f'{a or "T":4}'
    print(row)

print("\nState‑value function V* (rounded to 2 decimals):")
for r in range(rows):
    row = ''
    for c in range(cols):
        row += f'{V[(r,c)]:6.2f} '
    print(row)