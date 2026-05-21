import random
import collections
import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden, 1)
        self.adv_stream = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        val = self.value_stream(f)
        adv = self.adv_stream(f)
        return val + (adv - adv.mean(dim=1, keepdim=True))


Transition = collections.namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done"),
)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)
