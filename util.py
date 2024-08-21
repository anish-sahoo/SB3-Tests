
import polars as pd
import matplotlib.pyplot as plt

# Load the monitor file
df = pd.read_csv("monitor_0.monitor.csv", skip_rows=1)
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