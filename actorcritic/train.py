import numpy as np
import torch
import warnings
import sys
from pathlib import Path
from collections import deque
from copy import deepcopy
from typing import Tuple
import time

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from network import NeuralNet
from game_node import GameNode
from helper import evaluator
from data_preprocess import *
from utils import *


def compute_returns(next_value, rewards, masks, gamma) -> list:
    r = next_value
    returns = [0] * len(rewards)
    # Loop through the most recent rewards first
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r * masks[step]
        returns[step] = r
    return returns


def make_random_move(state: GameNode) -> Tuple[int, int]:
    """Make a random valid move on the board."""
    valid_mask = torch.tensor(state.available_moves_mask())
    action_probs = torch.where(valid_mask, torch.ones(1, 82), -1e8)
    action_probs = torch.softmax(action_probs, dim=1).squeeze()
    action = torch.distributions.Categorical(action_probs).sample()
    
    row = -1 if action == 81 else action // 9
    col = -1 if action == 81 else action % 9
        
    return row, col


def step(state: GameNode, model: NeuralNet, start: bool, log_probs: list, values: list, rewards: list, masks: list, entropies: list, move_confidence: list) -> Tuple[GameNode, float, bool]:
    """Take a step in the game."""
    # Convert initial state to tensor
    encoded_state = node_to_tensor(state)
    state_tensor = encoded_state.unsqueeze(0).float()
    
     # Get policy and value from network
    action_logits, value = model(state_tensor)
    
    # Apply valid moves mask
    valid_mask = torch.tensor(state.available_moves_mask())
    action_logits = torch.where(valid_mask, action_logits, -1e8)
    action_probs = torch.softmax(action_logits, dim=1).squeeze()
    
    # Create categorical distribution and sample action
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    # Take action
    row = -1 if action == 81 else action // 9
    col = -1 if action == 81 else action % 9
    next_state = state.create_child((row, col))
    
    # Get reward and check if terminal
    done = next_state.is_terminal()
    if done:
        # Reward is NOT from the perspective of current_player
        winner = next_state.compute_winner()
        reward = winner * 1 if start else winner * -1
        # print("winner is", winner, "reward is", reward)
    else:
        reward = 0
        # Small intermediate reward for capturing stones
        # reward = 0.01 * (len(state.groups) - len(next_state.groups))  # Reward for capturing stones

    # Store transition
    log_probs.append(log_prob)
    values.append(value.squeeze())
    rewards.append(reward)
    masks.append(1.0 - float(done))
    entropies.append(entropy)
    move_confidence.append(action_probs.max().item())
    
    return next_state, reward, done


def play_against_random(model: NeuralNet, optimizer: torch.optim.Adam, start: bool = True, num_steps=5, gamma=0.99):
    state = GameNode(size=9)
    episode_reward = 0
    
    # Initialize lists to store trajectory
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropies = []
    move_confidence = []
    
    # Make the random bot play first
    if not start:
        row, col = make_random_move(state)
        state = state.create_child((row, col))
    
    for move in range(500):  # Max moves per episode
        # Update state
        state, reward, done = step(state, model, start, log_probs, values, rewards, masks, entropies, move_confidence)
        encoded_state = node_to_tensor(state)
        state_tensor = encoded_state.unsqueeze(0).float()
        episode_reward += reward
        
        # If we've collected enough steps or reached terminal state, update
        if (move + 1) % num_steps == 0 or done:
            with torch.no_grad():
                _, next_value = model(state_tensor)
                next_value = next_value.squeeze()
            
            # Compute returns
            returns = torch.tensor(compute_returns(next_value, rewards, masks, gamma))
            
            # Convert lists to tensors
            log_probs_t = torch.stack(log_probs)
            values_t = torch.stack(values)
            entropies_t = torch.stack(entropies)
            
            # Compute advantages
            advantages = returns - values_t
            # Compute losses
            mean_policy_loss = -(log_probs_t * advantages.detach()).mean()
            # print(mean_policy_loss)
            mean_value_loss = advantages.pow(2).mean()
            mean_entropy_loss = entropies_t.mean()
            total_loss = mean_policy_loss + 0.5 * mean_value_loss - 0.01 * mean_entropy_loss
            
            # Backpropagate
            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Clear buffers
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropies = []
        
        if done:
            break
        
        # Make the random opponent play
        row, col = make_random_move(state)
        state = state.create_child((row, col))
    # print(f"Game ended with {state.compute_winner()} as winner")
    
    # TODO: measure avg(max(action_probs)) -> decrease lr if it becomes confident too quickly
    return episode_reward, state.move, sum(move_confidence) / (len(move_confidence) + 1), mean_policy_loss.item(), mean_value_loss.item(), total_loss.item()


if __name__ == '__main__':
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    torch.manual_seed(TRAIN_PARAMS["seed"])
    np.random.seed(TRAIN_PARAMS["seed"])

    # Initialize the models
    intial_model = NeuralNet()
    old_model = deepcopy(intial_model)
    model = deepcopy(intial_model)
    
    # TODO: try different learning rates based on the plot (multiply or divide by 2 to ensure learning is happening)
    # decrease learning rate for using small batch sizes
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_games = 10000
    gamma = 0.99
    
    # New parameters for n-step A2C
    num_steps = 10   # Number of steps to unroll
    print_freq = 10  # Print frequency
    save_freq = 100  # Save frequency
    
    game_rewards = deque(maxlen=print_freq)
    game_len = deque(maxlen=print_freq)
    general_stats = []
    win_stats = []
    
    # Initial the plot
    (fig1, axes1), (fig2, axes2) = make_training_plots()
    
    for game in range(num_games):
        # run the self play and training
        game_reward, length, avg_confidence, policy_loss, value_loss, total_loss = play_against_random(model, optimizer, start = game%2 == 0, num_steps=num_steps, gamma=gamma)
        # game_reward, avg_confidence, policy_loss, value_loss, total_loss = self_play(model, optimizer)
        game_rewards.append(game_reward)
        game_len.append(length)
        
        general_stats.append([avg_confidence, total_loss, policy_loss, value_loss])
        update_learning_metrics(axes1, game, general_stats)
        
        if (game + 1) % print_freq == 0:
            print(f"Episode {game + 1}, Avg Reward: {np.mean(game_rewards) if game_rewards else 0}, Avg Length: {np.mean(game_len) if game_len else 0}")
            
        if (game + 1) % save_freq == 0:
            # Plot win rate vs very first model
            win_rate_initial = evaluator(intial_model, model)
            print(f"Win rate against initial model: {win_rate_initial}")
            
            # Plot win rate vs model from 100 episodes ago
            win_rate = evaluator(old_model, model)
            print(f"Win rate against model from {save_freq} episodes ago: {win_rate}")

            old_model = deepcopy(model)
            
            win_stats.append([win_rate_initial, win_rate])
            update_win_rates(axes2, (game + 1) // save_freq, win_stats)
            
            torch.save(model.state_dict(), f"checkpoints/alphazero_model{game + 1}.pth")
            print(f"Saved model at episode {game + 1}")

    save_training_plots("alphazero_training_final")
    print("Training complete!")