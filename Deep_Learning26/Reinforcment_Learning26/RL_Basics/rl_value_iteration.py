# import numpy as np
#find, for every non‑terminal cell, the action that maximises the expected discounted return
# ----- 1. Define the grid and helpers -----
rows, cols = 3, 3
states = [(r, c) for r in range(rows) for c in range(cols)] #r..rows, c..cols
terminal = (2, 2)
actions = ['U', 'D', 'L', 'R']
gamma = 0.9
max_iters = 1000
theta = 1e-6                     # convergence tolerance

def step(state, a):
    """Deterministic transition."""
    r, c = state
    if state == terminal:
        return terminal, 0                # stays terminal, no reward
    # compute candidate next cell
    if a == 'U': nr, nc = max(r-1, 0), c
    if a == 'D': nr, nc = min(r+1, rows-1), c
    if a == 'L': nr, nc = r, max(c-1, 0)
    if a == 'R': nr, nc = r, min(c+1, cols-1)

    next_state = (nr, nc)
    # reward: -1 for any move, +9 if we land on treasure
    reward = -1
    if next_state == terminal:
        reward = 9          # -1 + 10
    return next_state, reward

# ----- 2. Initialise value function -----
V = {s: 0.0 for s in states}
V[terminal] = 0.0   # terminal value is always zero

# ----- 3. Value‑iteration loop -----
for it in range(max_iters):   #all rewards in all steps are already calculated here, these are just -1 at the beginning but through 
    #iterating more and more down the line steps are factored in, because V changes in every iteration, V is zero to start with
    #convergence after 5 itrations makes  sense because from the farthest end it's only 4 steps
    delta = 0.0
    V_new = V.copy()
    for s in states:
        if s == terminal:
            continue
        # compute Q(s,a) for each action
        q_vals = []
        for a in actions:  #deterministic
            s_next, r = step(s, a)
            q = r + gamma * V[s_next]      # deterministic Bellmann
            q_vals.append(q)
        best = max(q_vals)
        delta = max(delta, abs(best - V[s]))
        V_new[s] = best
    V = V_new
    if delta < theta:
        break

print(f"Converged after {it+1} iterations")


policy = {}
for s in states:
    if s == terminal:
        policy[s] = None
        continue
    q_vals = {} #q_vals from before gets zeroes, re-initailized as dict
    for a in actions:
        s_next, r = step(s, a)
        q_vals[a] = r + gamma * V[s_next] # use the *converged* V*
    # choose the action with the highest Q-value (break ties arbitrarily)
    best_a = max(q_vals, key=q_vals.get)   # finds the key that has the highest corresponding value in a dictionary
    policy[s] = best_a

# pretty‑print
for r in range(rows):
    row = ''
    for c in range(cols):
        a = policy[(r, c)]
        row += f'{a or "T":4}'
    print(row)

# 6.  (Optional) pretty‑print the final state‑value function
# --------------------------------------------------------------
print("\nState‑value function V*:")
for r in range(rows):
    row = ''
    for c in range(cols):
        row += f'{V[(r,c)]:6.2f} '
    print(row)



'''In Reinforcement Learning (RL), \(V\) refers to the state-value function. It measures how "good" it is for an agent to 
be in a particular state. Mathematically, it calculates the expected sum 
of discounted future rewards an agent will receive from that state onward, assuming it follows a specific behavioral strategy (policy'''