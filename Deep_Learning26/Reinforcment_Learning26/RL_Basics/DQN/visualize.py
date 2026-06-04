import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Optional, List
from grid_world import GridWorld, State, Action

# Mapping directions to visual 2D plot directions
_ARROW = {'U': (0, 0.3), 'D': (0, -0.3), 'L': (-0.3, 0), 'R': (0.3, 0)}

def plot_policy(env: GridWorld,
                policy: Dict[State, Action],
                V: Optional[Dict[State, float]] = None,
                title: str = "Optimal policy"):
    fig, ax = plt.subplots(figsize=(env.cols, env.rows))
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, env.cols + 1, 1))
    ax.set_yticks(np.arange(0, env.rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # ---- Value heat-map ----
    if V is not None:
        grid = np.full((env.rows, env.cols), np.nan)
        for (r, c), v in V.items():
            grid[r, c] = v
        # Display grid: row 0 is drawn at the top
        im = ax.imshow(grid, cmap='coolwarm', origin='upper',
                       interpolation='none', extent=[0, env.cols, 0, env.rows])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Obstacles & terminals ----
    # Row r in grid maps to y = env.rows - 1 - r in Cartesian space
    for r, c in env.obstacles:
        ax.add_patch(plt.Rectangle((c, env.rows - 1 - r), 1, 1, color='k'))
    for r, c in env.terminal:
        ax.add_patch(plt.Rectangle((c, env.rows - 1 - r), 1, 1, color='gold'))

    # ---- Arrows for policy ----
    for (r, c), a in policy.items():
        dx, dy = _ARROW[a]
        x = c + 0.5
        y = env.rows - 0.5 - r
        # Center the arrow in the grid square
        ax.arrow(x - dx*0.5, y - dy*0.5, dx, dy,
                 head_width=0.15, head_length=0.15,
                 fc='w', ec='w', linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    plt.show()