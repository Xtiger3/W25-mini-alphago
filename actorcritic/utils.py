import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


def hold_training_plot():
    """Keep the program alive to display the training plots"""
    plt.ioff()
    plt.show()


def make_training_plots(on=True):
    """Set up two interactive matplotlib graphs:
    1. Learning metrics (average move confidence, losses)
    2. Win rates"""
    if on:
        plt.ion()
    
    # Figure 1: Frequent metrics (updated every episode)
    fig1, axes1 = plt.subplots(2, 2, figsize=(10 , 6))
    fig1.suptitle("Learning Metrics")
    
    # First row
    axes1[0, 0].set_xlabel("Episode")
    axes1[0, 1].set_xlabel("Episode")
    
    # Second row
    axes1[1, 0].set_xlabel("Episode")
    axes1[1, 1].set_xlabel("Episode")
    
    # Figure 2: Win rates (updated less frequently)
    fig2, axes2 = plt.subplots(1, 2, figsize=(7, 3))
    fig2.suptitle("Win Rates")
    
    axes2[0].set_xlabel("Episode")
    axes2[0].set_ylabel("Win Rate vs Random")
    
    axes2[1].set_xlabel("Episode")
    axes2[1].set_ylabel("Win Rate vs Previous")
    
    plt.tight_layout()
    return (fig1, axes1), (fig2, axes2)


def update_learning_metrics(axes, episode, stats, win_stats, on=True):
    """Update the learning metrics plot."""
    # Map metrics to subplots
    
    plot_map = {
        0: (0, 0),   # Avg confidence
        1: (0, 1),   # Avg value
        2: (1, 0),   # Total loss
        3: (1, 1)    # Win rate (updated less frequently)
    }
    
    colors = ['b', 'g', 'r', 'c']
    
    labels = {
        (0, 0): ("Episode", "Average Move Confidence"),
        (0, 1): ("Episode", "Average Advantage"),
        (1, 0): ("Episode", "Total Loss"),
        (1, 1): ("Episode", "Average Reward")
    }
    
    # Plot frequent metrics
    for i in range(3):
        row, col = plot_map[i]
        axes[row, col].clear()
        axes[row, col].plot(
            range(episode - len(stats) + 1, episode + 1),
            [stat[i] for stat in stats],
            linestyle='-',
            marker='o',
            color=colors[i],
            markersize=3
        )
        
        # Re-set labels after clear
        axes[row, col].set_xlabel(labels[(row, col)][0])
        axes[row, col].set_ylabel(labels[(row, col)][1])

    # Plot win rate separately (less frequent)
    if win_stats:
        row, col = plot_map[3]
        axes[row, col].clear()
        win_x = range(episode - len(stats) + 1, episode + 1, 10)
        axes[row, col].plot(
            win_x[:len(win_stats)],
            win_stats,
            linestyle='-',
            marker='o',
            color=colors[3],
            markersize=3
        )
        axes[row, col].set_xlabel(labels[(row, col)][0])
        axes[row, col].set_ylabel(labels[(row, col)][1])
    
    if on:
        plt.pause(0.00001)


def update_win_rates(axes, episode, stats, on=True):
    """Update the win rates plot."""
    colors = ['b', 'g']
    labels = [
        ("Episode", "Win Rate vs Random"),
        ("Episode", "Win Rate vs Previous")
    ]
    
    for i, ax in enumerate(axes):
        ax.plot(
            range(1, episode + 1, 100),
            [stat[i] for stat in stats],
            linestyle='-',
            marker='o',
            color=colors[i],
            markersize=3
        )
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
    
    if on:
        plt.pause(0.00001)


def save_training_plots(base_filename="actorcritic_training", game_number=0):
    """Save both training plots to files."""
    plt.figure(1)
    plt.savefig(f"{base_filename}_metrics.png", dpi=200, bbox_inches='tight')
    
    plt.figure(2)
    plt.savefig(f"winrates_{game_number}.png", dpi=200, bbox_inches='tight')


def save_stats_to_csv(stats, filename):
    """Save training statistics to CSV file"""
    os.makedirs('training_logs', exist_ok=True)
    filepath = f'training_logs/{filename}.csv'
    
    # Convert to dataframe and save
    df = pd.DataFrame(stats, columns=[
        'episode', 'epoch', 
        'reward', 'avg_policy_confidence', 'avg_value_est', 'advantage',
        'policy_loss', 'value_loss', 'total_loss', 'grad_norm', 'time'
    ])

    header = not os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=header)