import torch
import torch.nn as torch_nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from dataclasses import dataclass

@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.5  # Soft update
    batch_size: int = 64
    buffer_size: int = 10000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

class QNetwork(torch_nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(state_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_dim, hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, config: DQNConfig, device: torch.device):
        self.config = config
        self.device = device
        
        self.policy_net = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.buffer_size, device)
        
        self.epsilon = config.eps_start

    def act(self, state: np.ndarray, evaluate=False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)
            
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) > self.config.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
            
        loss = F.huber_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._soft_update(self.policy_net, self.target_net, self.config.tau)
        
    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.config.eps_end, self.epsilon * self.config.eps_decay)
