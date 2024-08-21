import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import Breakout
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

ale = ALEInterface()
ale.loadROM(Breakout)

# Create a single environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Wrap the single environment in a VecEnv
vec_env = DummyVecEnv([lambda: env])

model = DQN.load("./dqn_breakout", env=vec_env)
obs, done = vec_env.reset(), False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    vec_env.render()
input("Press Enter to continue...")
