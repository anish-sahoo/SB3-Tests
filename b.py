import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import Breakout

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import matplotlib.pyplot as plt


def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env = Monitor(env, filename=f"monitor_{rank}.csv")  # Wrap with Monitor
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    ale = ALEInterface()
    ale.loadROM(Breakout)
    
    env_id = "ALE/Breakout-v5"
    num_cpu = 10  # Number of processes to use
    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO("CnnPolicy", vec_env, verbose=1, device="cuda")
    model.learn(total_timesteps=200_000, progress_bar=True)
    print("Training done")
    
    model.save("ppo_breakout")

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

    # After training, aggregate monitor logs and plot rewards
    reward_logs = []
    for i in range(num_cpu):
        df = pd.read_csv(f"monitor_{i}.csv.monitor.csv", skiprows=1)  # Skip header line
        reward_logs.append(df['r'])

    # Combine rewards from all environments
    combined_rewards = pd.concat(reward_logs).reset_index(drop=True)

    # Plotting the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(combined_rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.legend()
    plt.show()
