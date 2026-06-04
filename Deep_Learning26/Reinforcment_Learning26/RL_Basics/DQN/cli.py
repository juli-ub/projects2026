import argparse
import json
import logging
from grid_world import GridWorld
from policy_iteration import PolicyIteration
from visualize import plot_policy

def main():
    parser = argparse.ArgumentParser(description="Tabular RL on a configurable GridWorld")
    parser.add_argument('--config', type=str, help="Path to JSON config")
    parser.add_argument('--show', action='store_true', help="Plot the result")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    world = GridWorld(**cfg.get('world', {}))
    pi = PolicyIteration(world,
                         gamma=cfg.get('gamma', 0.9),
                         max_iters=cfg.get('max_iters', 1000),
                         eps=cfg.get('eps', 1e-8))

    V, policy = pi.run()

    if args.show:
        plot_policy(world, policy, V=V, title="Result (CLI)")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()