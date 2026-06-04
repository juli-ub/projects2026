from grid_world import GridWorld
from policy_iteration import PolicyIteration
from visualize import plot_policy

# 5x5 with a wall and two treasure cells
world = GridWorld(
    rows=5,
    cols=5,
    terminal=[(4, 4), (0, 4)],
    obstacles=[(2, 2), (1, 3)],
    slip_prob=0.1,
    step_cost=-0.1,
    treasure_bonus=5.0
)

pi = PolicyIteration(world, gamma=0.9)
V, policy = pi.run()
plot_policy(world, policy, V=V, title="Examples.py 5x5 Grid World")