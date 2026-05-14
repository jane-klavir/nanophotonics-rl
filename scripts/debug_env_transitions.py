import numpy as np
from nano_mie.rl_env import NanoparticleEnv

def main():
    env = NanoparticleEnv(data_path="data/processed/mie_materials_v1.npz")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        state, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 10:
            step += 1
            action = env.action_space.sample()
            
            # Extract current state before action
            mat_before = env._curr_mat
            rad_before = env._curr_rad
            rad_idx_before = env._curr_rad_idx
            row_before = env.mat_to_rows[mat_before][rad_idx_before]
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"Step {step}:")
            print(f"  Action: {action}")
            print(f"  Material: {mat_before} -> {info['material']}")
            print(f"  Radius: {rad_before:.2f} -> {info['radius_nm']:.2f}")
            print(f"  Lookup row: {row_before} -> {info['lookup_row']}")
            print(f"  Error: {info['old_error']:.5f} -> {info['new_error']:.5f}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            print(f"  Success: {info['success']}")

if __name__ == '__main__':
    main()