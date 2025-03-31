import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def hold_training_plot():
    """Keep the program alive to display the training plots"""
    plt.ioff()
    plt.show()


def make_training_plots():
    """Set up two interactive matplotlib graphs:
    1. Learning metrics (average move confidence, losses)
    2. Win rates"""
    plt.ion()
    
    # Figure 1: Frequent metrics (updated every episode)
    fig1, axes1 = plt.subplots(2, 2, figsize=(7, 6))
    fig1.suptitle("Learning Metrics")
    
    # First row
    axes1[0, 0].set_xlabel("Episode")
    axes1[0, 0].set_ylabel("Average Move Confidence")
    
    axes1[0, 1].set_xlabel("Episode")
    axes1[0, 1].set_ylabel("Total Loss")
    
    # Second row
    axes1[1, 0].set_xlabel("Episode")
    axes1[1, 0].set_ylabel("Policy Loss")
    
    axes1[1, 1].set_xlabel("Episode")
    axes1[1, 1].set_ylabel("Value Loss")
    
    # Figure 2: Win rates (updated less frequently)
    fig2, axes2 = plt.subplots(1, 2, figsize=(7, 3))
    fig2.suptitle("Win Rates")
    
    axes2[0].set_xlabel("Episode")
    axes2[0].set_ylabel("Win Rate vs Initial")
    
    axes2[1].set_xlabel("Episode")
    axes2[1].set_ylabel("Win Rate vs Previous")
    
    plt.tight_layout()
    return (fig1, axes1), (fig2, axes2)


def update_learning_metrics(axes, episode, stats):
    """Update the learning metrics plot."""
    # Map metrics to subplots
    plot_map = {
        0: (0, 0),  # Avg confidence
        1: (1, 0),   # Policy loss
        2: (1, 1),   # Value loss
        3: (0, 1)    # Total loss
    }
    
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i in range(4):
        row, col = plot_map[i]
        axes[row, col].plot(
            range(episode - len(stats) + 1, episode + 1),
            [stat[i] for stat in stats],
            linestyle='-',
            marker='o',
            color=colors[i],
            markersize=3
        )
    
    plt.pause(0.00001)


def update_win_rates(axes, episode, stats):
    """Update the win rates plot."""
    colors = ['b', 'g']
    
    for i, ax in enumerate(axes):
        ax.plot(
            range(episode - len(stats) + 1, episode + 1),
            [stat[i] for stat in stats],
            linestyle='-',
            marker='o',
            color=colors[i],
            markersize=3
        )
    
    plt.pause(0.00001)


def save_training_plots(base_filename="actorcritic_training"):
    """Save both training plots to files."""
    plt.figure(1)
    plt.savefig(f"{base_filename}_metrics.png", dpi=200, bbox_inches='tight')
    
    plt.figure(2)
    plt.savefig(f"{base_filename}_winrates.png", dpi=200, bbox_inches='tight')