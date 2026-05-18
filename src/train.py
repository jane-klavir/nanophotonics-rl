from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .env import MieEnv
from .dqn import DQN, ReplayBuffer, Transition


@dataclass
class TrainConfig:
    episodes: int = 3000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 1e-4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.998
    target_update: int = 500
    hidden: int = 256
    buffer_cap: int = 50_000
    min_buffer: int = 1_000
    grad_clip: float = 1.0
    print_every: int = 100


def train_dqn(
    env: MieEnv,
    config: TrainConfig | None = None,
    device: torch.device | str | None = None,
):
    """Train a Double Dueling DQN on the Mie environment."""
    config = config if config is not None else TrainConfig()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    online_net = DQN(MieEnv.STATE_DIM, MieEnv.N_ACTIONS, config.hidden).to(device)
    target_net = DQN(MieEnv.STATE_DIM, MieEnv.N_ACTIONS, config.hidden).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_cap)
    loss_fn = nn.SmoothL1Loss()

    epsilon = config.eps_start
    episode_rewards = []
    episode_errors = []
    losses = []
    step_count = 0

    for ep in range(1, config.episodes + 1):
        state = env.reset()
        ep_ret = 0.0

        while True:
            if random.random() < epsilon:
                action = random.randrange(MieEnv.N_ACTIONS)
            else:
                with torch.no_grad():
                    t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online_net(t).argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_ret += reward
            step_count += 1

            if len(buffer) >= config.min_buffer:
                batch = buffer.sample(config.batch_size)
                b = Transition(*zip(*batch))

                s_b = torch.tensor(np.array(b.state), dtype=torch.float32, device=device)
                a_b = torch.tensor(b.action, dtype=torch.long, device=device).unsqueeze(1)
                r_b = torch.tensor(b.reward, dtype=torch.float32, device=device).unsqueeze(1)
                ns_b = torch.tensor(np.array(b.next_state), dtype=torch.float32, device=device)
                d_b = torch.tensor(b.done, dtype=torch.float32, device=device).unsqueeze(1)

                q_curr = online_net(s_b).gather(1, a_b)

                with torch.no_grad():
                    best_actions = online_net(ns_b).argmax(dim=1, keepdim=True)
                    q_next = target_net(ns_b).gather(1, best_actions)
                    q_targ = r_b + config.gamma * q_next * (1 - d_b)

                loss = loss_fn(q_curr, q_targ)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), config.grad_clip)
                optimizer.step()
                losses.append(float(loss.item()))

                if step_count % config.target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())

            if done:
                break

        epsilon = max(config.eps_end, epsilon * config.eps_decay)
        episode_rewards.append(ep_ret)
        episode_errors.append(info["combined_error"])

        if config.print_every and ep % config.print_every == 0:
            avg_r = np.mean(episode_rewards[-config.print_every:])
            avg_e = np.mean(episode_errors[-config.print_every:])
            print(f"Ep {ep:4d} | eps={epsilon:.3f} | avg_reward={avg_r:7.2f} | avg_combined_error={avg_e:.4f}")

    history = {
        "episode_rewards": episode_rewards,
        "episode_errors": episode_errors,
        "losses": losses,
        "epsilon_final": epsilon,
    }
    print("Training done.")
    return online_net, history
