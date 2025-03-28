import numpy as np

import torch
import warnings
from network import AlphaZeroNet

from game_node import GameNode
from data_preprocess import *

# Suppress all warnings
warnings.filterwarnings("ignore")

def train():
    # Define the actor and critic networks
    model = AlphaZeroNet(6 + 1, 9 * 9 + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Main training loop
    num_episodes = 10000
    gamma = 0.99

    for episode in range(num_episodes):
        state = GameNode(size=9)  # Initialize a new game
        episode_reward = 0
        
        # Convert state to tensor
        # state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

        for t in range(1, 10000):  # Limit the number of time steps
            # Encode the board state
            encoded_state = node_to_tensor(state)
            state_tensor = torch.tensor(encoded_state).unsqueeze(0).float()
            
            # Get policy and value from network
            action_probs, value = model(state_tensor)
            
            with torch.no_grad():
                action_probs = torch.softmax(action_probs, dim=1).squeeze()  # Shape: (82,)
                valid_mask = torch.tensor(state.available_moves_mask(), dtype=torch.float32)

                # Apply valid moves mask
                action_probs = action_probs * valid_mask
                action_probs = action_probs / action_probs.sum()
    
                # Sample an action directly in PyTorch
                action = torch.multinomial(action_probs, num_samples=1).item()
                
                # Sample an action
                # action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
            row = -1 if action == 81 else action // 9
            col = -1 if action == 81 else action % 9
            
            # Take the chosen action and observe the next state and reward
            next_state = state.create_child((row, col))
            reward = next_state.compute_winner() if next_state.is_terminal() else 0
            
            # Get next state value from network
            with torch.no_grad():
                encoded_state = node_to_tensor(next_state)
                next_state_tensor = torch.tensor(encoded_state).unsqueeze(0).float()
                _, next_value = model(next_state_tensor)
                next_value = next_value.squeeze().detach()
            
            # Compute advantage
            current_value = value.squeeze()
            advantage = reward + gamma * next_value - current_value
            
            # Compute losses
            policy_loss = -torch.log(action_probs[action]) * advantage.detach()
            value_loss = advantage.pow(2)
            total_loss = policy_loss + value_loss
            
            optimizer.zero_grad() # Zero gradients
            total_loss.backward() # Backpropagate
            optimizer.step() # Update network
            
            # Update state and accumulate reward
            state = next_state
            state_tensor = next_state_tensor
            episode_reward += reward
            
            if next_state.is_terminal():
                break

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {episode_reward}')

    # Save model
    torch.save(model.state_dict(), 'alphazero_model.pth')

    print("done!")


train()