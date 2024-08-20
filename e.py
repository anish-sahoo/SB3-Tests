import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import os
import matplotlib.pyplot as plt


def make_env(env_id: str, seed: int = 0):
    """
    Utility function for a single environment.

    :param env_id: the environment ID
    :param seed: the initial seed for RNG
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = Monitor(env, filename="monitor_0")
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "ALE/Breakout-v5"
    
    env = DummyVecEnv([make_env(env_id)])

    model = DQN("CnnPolicy", env, verbose=1, device="cuda", buffer_size=50_000,exploration_fraction=0.3, exploration_final_eps=0.01, tensorboard_log="./dqn_breakout_tensorboard/")
    model.learn(total_timesteps=200_000, progress_bar=True)
    
    model.save("dqn_breakout")

    # Load the monitor file
    df = pd.read_csv("monitor_0.monitor.csv", skiprows=1)
    combined_rewards = df['r']
    combined_loss = df['l']

    plt.figure(figsize=(10, 5))
    plt.plot(combined_rewards, label='Reward')
    plt.plot(combined_loss, label='Loss')  # Add the plot for loss
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Reward and Loss over Time')
    plt.legend()
    plt.savefig("dqn_breakout_rewards.png")
