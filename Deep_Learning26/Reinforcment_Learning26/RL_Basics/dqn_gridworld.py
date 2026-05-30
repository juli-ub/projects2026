# -------------------------------------------------------------
#  Simple DQN on a tiny deterministic GridWorld (CPU only)
# -------------------------------------------------------------
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------------
# 1️⃣  Mock environment ------------------------------------------------
# -------------------------------------------------------------
class GridWorld:
    """
    5x5 grid, start at (0,0), goal at (4,4).
    Actions: 0=up, 1=down, 2=left, 3=right.
    Reward: -1 per step, +10 for reaching goal (episode ends).
    """
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._obs()

    def _obs(self):
        # simple one‑hot flattened representation of the agent location
        obs = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self.pos[0] * self.size + self.pos[1]
        obs[idx] = 1.0
        return obs

    def step(self, action):
        x, y = self.pos
        if action == 0:   # up
            x = max(x-1, 0)
        elif action == 1: # down
            x = min(x+1, self.size-1)
        elif action == 2: # left
            y = max(y-1, 0)
        elif action == 3: # right
            y = min(y+1, self.size-1)

        self.pos = (x, y)
        done = self.pos == self.goal
        reward = 10.0 if done else -1.0
        return self._obs(), reward, done, {}

# -------------------------------------------------------------
# 2️⃣  Replay buffer ------------------------------------------------
# -------------------------------------------------------------
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10_000): 
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------
# 3️⃣  Neural network (Q‑function) ---------------------------------
# -------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 4️⃣  Training hyper‑parameters ------------------------------------
# -------------------------------------------------------------
DEVICE = torch.device("cpu")          # force CPU
ENV = GridWorld(size=5)
STATE_DIM = ENV.size * ENV.size
N_ACTIONS = 4

policy_net   = DQN(STATE_DIM, N_ACTIONS).to(DEVICE)
target_net   = DQN(STATE_DIM, N_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

replay = ReplayBuffer(capacity=5000)

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY_STEPS = 5000
TARGET_UPDATE = 200          # steps

def epsilon_by_step(step):
    """Linear annealing from EPS_START to EPS_END."""
    eps = max(EPS_END,
              EPS_START - (step / EPS_DECAY_STEPS) * (EPS_START - EPS_END))
    return eps

# -------------------------------------------------------------
# 5️⃣  Main training loop -------------------------------------------
# -------------------------------------------------------------
num_episodes = 150 #500
global_step = 0
all_rewards = []

for ep in range(num_episodes):
    state = ENV.reset()
    episode_reward = 0

    while True:
        # ---- ε‑greedy action selection ----
        eps = epsilon_by_step(global_step)
        if random.random() < eps:   #explore
            action = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():   #exploit
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                q_vals = policy_net(state_tensor)
                action = q_vals.argmax(dim=1).item()

        # ---- environment step ----
        next_state, reward, done, _ = ENV.step(action)
        episode_reward += reward

        # ---- store transition ----
        replay.push(state, action, reward, next_state, done)

        state = next_state
        global_step += 1

        # ---- learn from a minibatch if enough data ----
        if len(replay) >= BATCH_SIZE:
            batch = replay.sample(BATCH_SIZE)

            # Convert batch arrays → tensors
            state_batch = torch.from_numpy(np.stack(batch.state)).to(DEVICE)
            action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(DEVICE)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            next_state_batch = torch.from_numpy(np.stack(batch.next_state)).to(DEVICE)
            done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(DEVICE)

            # ---- Q‑learning target ----
            with torch.no_grad():
                next_q = target_net(next_state_batch).max(1, keepdim=True)[0]
                target = reward_batch + GAMMA * (1.0 - done_batch) * next_q

            # ---- current Q estimate ----
            current_q = policy_net(state_batch).gather(1, action_batch)

            # ---- loss + backward ----
            loss = criterion(current_q, target)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (helps stability)
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- periodic target network update ----
        if global_step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    all_rewards.append(episode_reward)
    if (ep + 1) % 50 == 0:
        avg_last = np.mean(all_rewards[-50:])
        print(f"Episode {ep+1:3d} | Avg reward (last 50): {avg_last:6.2f} | ε={eps:.3f}")

# -------------------------------------------------------------
# 6️⃣  Quick evaluation (deterministic greedy policy) ---------------
# -------------------------------------------------------------
def evaluate(policy, env, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                s = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                a = policy(s).argmax(dim=1).item()
            state, r, done, _ = env.step(a)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
    return np.mean(rewards)

greedy_reward = evaluate(policy_net, ENV, n_episodes=30)
print("\n=== Evaluation with greedy policy ===")
print(f"Average return over 30 episodes: {greedy_reward:.2f}")

# -------------------------------------------------------------
# 7️⃣  What you just saw ------------------------------------------------
# -------------------------------------------------------------
"""
- The environment is tiny (25 states) – perfect for a demo.
- The DQN learns to reach the goal in ~200‑300 episodes (you'll see the avg reward rise toward +10).
- No GPU is required; everything runs on the CPU.
- Feel free to tweak:
    * GRID size (make it larger) – you may need a bigger network or more training steps.
    * Change the reward scheme (e.g., sparse reward only at the goal) – you’ll see learning become harder.
    * Replace DQN with a policy‑gradient (REINFORCE) or actor‑critic if you prefer continuous‑action problems.
"""