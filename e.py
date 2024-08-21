import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper


def make_env(env_id: str, seed: int = 0):
    """
    Utility function for a single environment.

    :param env_id: the environment ID
    :param seed: the initial seed for RNG
    """
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = Monitor(env, filename="monitor_0.log", allow_early_resets=True)
        env = AtariWrapper(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=True)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "ALE/Breakout-v5"
    num_envs = 20  # Adjust based on your system's capabilities
    env = SubprocVecEnv([make_env(env_id) for _ in range(num_envs)])
    # env = DummyVecEnv([make_env(env_id)])

    model = DQN("CnnPolicy", env, device="cuda", learning_starts=42_000, buffer_size=42_000, exploration_fraction=0.99, learning_rate=0.00025, exploration_final_eps=0.01, tensorboard_log="./dqn_breakout_tensorboard/")
    model.learn(total_timesteps=50_000_000, progress_bar=True)
    
    model.save("dqn_breakout")
