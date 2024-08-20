from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_tensorboard_values(logdir: str, tag: str):
    """Extract values for a given tag from TensorBoard logs."""
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    scalar_events = event_acc.Scalars(tag)
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    return steps, values

if __name__ == "__main__":
    # Define the log directory and tag
    tensorboard_logdir = "./dqn_breakout_tensorboard/"
    loss_tag = "l"  # The tag for training loss

    # Extract reward data
    df = pd.read_csv("monitor_0.monitor.csv", skiprows=1)
    combined_rewards = df['r']

    plt.figure(figsize=(10, 5))
    plt.plot(combined_rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.legend()
    plt.savefig("dqn_breakout_rewards.png")

    # Extract loss data
    steps, loss_values = extract_tensorboard_values(tensorboard_logdir, loss_tag)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, loss_values, label='Loss', color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    plt.legend()
    plt.savefig("dqn_breakout_loss.png")

    # Remove the monitor file after saving the plot
    file_path = "monitor_0.monitor.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
