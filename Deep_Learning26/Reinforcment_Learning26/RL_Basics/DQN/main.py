import logging
logging.basicConfig(level=logging.INFO)

from grid_world import GridWorld
from policy_iteration import PolicyIteration
from visualize import plot_policy

def main():
    world = GridWorld(rows=5,
                      cols=5,
                      terminal=[(4, 4)],
                      obstacles=[(2, 2), (1, 3)],
                      slip_prob=0.2,
                      step_cost=-0.1,
                      treasure_bonus=5.0)

    pi = PolicyIteration(world, gamma=0.9)
    V, optimal_policy = pi.run()

    # Visualize results
    plot_policy(world, optimal_policy, V=V, title="5×5 Grid – Optimal Policy")

if __name__ == '__main__':
    main()