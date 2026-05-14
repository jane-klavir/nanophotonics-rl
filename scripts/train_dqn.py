#!/usr/bin/env python3

import os
import torch
import numpy as np
from nano_mie.rl_env import NanoparticleEnv
from nano_mie.dqn import DQNAgent, DQNConfig

def evaluate(agent, env, num_episodes=50):
    successes = 0
    final_errors = []
    total_steps = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            
        successes += int(info['success'])
        final_errors.append(info['new_error'])
        total_steps.append(info['steps'])
        
    return {
        "success_rate": successes / num_episodes,
        "mean_error": np.mean(final_errors),
        "median_error": np.median(final_errors),
        "mean_steps": np.mean(total_steps)
    }

def train():
    env = NanoparticleEnv(data_path="data/processed/mie_materials_v1.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    config = DQNConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
        lr=1e-3,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995
    )
    
    agent = DQNAgent(config, device)
    
    num_episodes = 500
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        # Track initial error to print later
        # env._prev_error holds the initial error right after reset
        initial_error = env._prev_error
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
        agent.update_epsilon()
        
        if episode % 10 == 0:
            print(f"Episode: {episode:4d} | Reward: {episode_reward:7.2f} | Init Error: {initial_error:.4f} | Final Error: {info['new_error']:.4f} | Steps: {info['steps']} | Success: {info['success']} | Eps: {agent.epsilon:.3f}")
            
        if episode % 50 == 0:
            print("-" * 80)
            eval_metrics = evaluate(agent, env, num_episodes=50)
            print(f"EVAL (50 eps) | Success Rate: {eval_metrics['success_rate']:.2%} | Mean Err: {eval_metrics['mean_error']:.4f} | Median Err: {eval_metrics['median_error']:.4f} | Mean Steps: {eval_metrics['mean_steps']:.1f}")
            print("-" * 80)

    print("Training finished!")
    os.makedirs("models", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), "models/dqn_baseline.pth")
    print("Model saved to models/dqn_baseline.pth")

if __name__ == "__main__":
    train()
