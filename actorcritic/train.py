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
from helper import *
from data_preprocess import *
from utils import *


def compute_returns(next_value, rewards, masks, gamma) -> list:
    # print(len(masks), len(rewards))
    # print("masks", masks)
    # print("rewards", rewards)
    r = next_value
    returns = [0] * len(rewards)
    # Loop through the most recent rewards first
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r * masks[step]
        returns[step] = r
    return returns


def step(state: GameNode, model: NeuralNet, start: bool, log_probs: list, values: list, rewards: list, masks: list, entropies: list) -> Tuple[GameNode, float, bool]:
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

    # Store transition
    log_probs.append(log_prob)
    values.append(value.squeeze())
    entropies.append(entropy)
    
    return next_state, action_probs.max().item(), value.squeeze().item(), done


def play_against_random(model: NeuralNet, optimizer: torch.optim.Adam, start: bool = True, num_steps=5, gamma=0.99):
    state = GameNode(size=9)
    # print("start = ", start)

    # slowly increase komi to make the model take advantage of starting first
    # goes from 0.5 to 7.5 over the course of training
    state.komi = komi
    
    # Initialize lists to store trajectory
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropies = []
    avg_move_confidence = 0
    avg_advantage = 0
    
    # Make the random bot play first
    if not start:
        row, col = make_random_move(state)
        state = state.create_child((row, col))
    
    for move in range(500):  # Max moves per episode
        # Update state
        state, move_confidence, value, done = step(state, model, start, log_probs, values, rewards, masks, entropies)
        if avg_move_confidence == 0:
            avg_move_confidence = move_confidence
        else:
            avg_move_confidence = avg_move_confidence * 0.9 + move_confidence * 0.1
        
        if not done:
            # Make the random opponent play
            row, col = make_random_move(state)
            state = state.create_child((row, col))
            done = state.is_terminal()
        
        # Store transition
        reward = state.compute_winner() if done else 0
        masks.append(1.0 - float(done))
        rewards.append(reward)

        encoded_state = node_to_tensor(state)
        state_tensor = encoded_state.unsqueeze(0).float()
        
        # If we've collected enough steps or reached terminal state, update
        if (move + 1) % num_steps == 0 or done:
            with torch.no_grad():
                _, next_value = model(state_tensor)
                next_value = next_value.squeeze()
                # print("next_value", next_value)
            
            # Compute returns
            returns = torch.tensor(compute_returns(next_value, rewards, masks, gamma))
            
            # Convert lists to tensors
            log_probs_t = torch.stack(log_probs)
            values_t = torch.stack(values)
            entropies_t = torch.stack(entropies)
            # print("returns", returns)
            # print("values_t", values_t)
            
            # Compute advantages
            if start:
                advantages = returns - values_t  # Model is player 1
            else:
                advantages = (-returns) - (-values_t)  # Model is player 2 (flip perspective)
            
                
            avg_advantage = advantages.mean() if avg_advantage == 0 else avg_advantage * 0.9 + advantages.mean() * 0.1
            # advantages = returns - values_t
            # Compute losses
            mean_policy_loss = -(log_probs_t * advantages.detach()).mean()
            # print(mean_policy_loss)
            mean_value_loss = advantages.pow(2).mean()
            mean_entropy_loss = entropies_t.mean()
            total_loss = mean_policy_loss + 0.5 * mean_value_loss - 0.01 * mean_entropy_loss
            
            # print("advantages", advantages)
            # print("mean_policy_loss", mean_policy_loss)
            # print("mean_value_loss", mean_value_loss)
            # Backpropagate
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

            save_stats_to_csv([[
                game, state.move,  # general game info
                reward, round(avg_move_confidence, 3), round(values_t.mean().item(), 3), round(avg_advantage.item(), 3),  # moves info
                round(mean_policy_loss.item(), 3), round(mean_value_loss.item(), 3), round(total_loss.item(), 3), 
                round(grad_norm.item(),3), round(time.time() - start_time, 1)  # training info
            ]], "training_log")
            
            
            # Clear buffers
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropies = []
            
            # time.sleep(10)
        
        if done:
            break
    
    # print(f"advantages: {advantages}")
    # print(f"returns: {returns}")
    # print(f"values: {values_t}")
    # print(f"Game ended with {state.compute_winner()} as winner")
    
    # TODO: measure avg(max(action_probs)) -> decrease lr if it becomes confident too quickly
    episode_reward = reward if start else -reward
    return episode_reward, state.move, avg_move_confidence, avg_advantage.item(), total_loss.item()


if __name__ == '__main__':
    # Suppress all warnings
    warnings.filterwarnings("ignore")

    torch.manual_seed(TRAIN_PARAMS["seed"])
    np.random.seed(TRAIN_PARAMS["seed"])

    # Initialize the models
    model = NeuralNet()
    old_model = deepcopy(model)
    
    # TODO: try different learning rates based on the plot (multiply or divide by 2 to ensure learning is happening)
    # decrease learning rate for using small batch sizes
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_games = 50000
    gamma = 0.99
    
    # New parameters for n-step A2C
    num_steps = 10   # Number of steps to unroll
    print_freq = 10  # Print frequency
    save_freq = 100  # Save frequency
    
    game_rewards = deque(maxlen=print_freq)
    game_len = deque(maxlen=print_freq)
    general_stats = deque(maxlen=500)
    win_stats = deque(maxlen=500//print_freq)
    test_stats = []

    start_time = time.time()
    
    # Initial the plot
    interative_plot = False
    (fig1, axes1), (fig2, axes2) = make_training_plots(interative_plot)

    # komi
    komi = 2.5
    
    for game in range(num_games):
        # run the self play and training
        game_reward, length, avg_confidence, avg_advantage, total_loss = play_against_random(model, optimizer, start = game%2 == 0, num_steps=num_steps, gamma=gamma)
        # game_reward, avg_confidence, policy_loss, value_loss, total_loss = self_play(model, optimizer)
        game_rewards.append(game_reward)
        game_len.append(length)
        
        general_stats.append([avg_confidence, avg_advantage, total_loss])
        
        if (game + 1) % print_freq == 0:
            win_stats.append(np.mean(game_rewards))
            # print(f"Episode {game + 1}, Avg Reward: {np.mean(game_rewards) if game_rewards else 0}, Avg Length: {np.mean(game_len) if game_len else 0}")
            
        if (game + 1) % save_freq == 0:
            # only keep the most recent 500 data points
            update_learning_metrics(axes1, game, general_stats, win_stats, interative_plot)

            # Plot win rate vs very first model
            win_rate_random = evaluate_against_random(model, 100, komi)
            
            # Plot win rate vs model from 100 episodes ago
            win_rate = evaluator(old_model, model, 100, komi)

            print(f"win rate vs random: {win_rate_random}\tvs {save_freq} episodes ago: {win_rate}")
            
            old_model = deepcopy(model)
            
            test_stats.append([win_rate_random, win_rate])
            update_win_rates(axes2, game, test_stats, interative_plot)
            
            torch.save(model.state_dict(), f"checkpoints/alphazero_model{game + 1}.pth")
            print(f"Saved model at episode {game + 1}")

            # GameNode.board_index = 0

        if (game + 1) % 500 == 0:
            save_training_plots(f"alphazero_training_{game + 1}", (game+1)//10000)

    save_training_plots("alphazero_training_final", -1)
    print("Training complete!")