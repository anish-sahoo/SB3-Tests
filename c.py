import polars as pd
import matplotlib.pyplot as plt

reward_logs = []
for i in range(10):
    df = pd.read_csv(f"monitor_{i}.csv.monitor.csv", skip_rows=1)  # Skip header line
    reward_logs.append(df['r'])

# Combine rewards from all environments
combined_rewards = pd.concat(reward_logs).to_frame()

# Resetting the index by creating a new index column
combined_rewards = combined_rewards.with_columns(pd.Series(name="index", values=range(combined_rewards.height)))

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(combined_rewards['r'], label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Time')
plt.legend()
plt.show()
