from network import AlphaZeroNet
from preprocessing import encode
from game_node import GameNode
from train_helper import *
import numpy as np
import torch.nn.functional as F
import torch
from config import *
from data_preprocess import *
from mcts import *

def self_play(model: AlphaZeroNet, num_games=10, look_back=3):
    game_data = []  # Stores (state, policy, outcome) for training
    
    for _ in range(num_games):
        board = GameNode(9)  # Initialize the game board
        game_history = []  # Stores the history of the game
        
        p1_pass, p2_pass = False, False
        while not board.is_terminal():
            # Encode the board state
            encoded_state = encode(board, look_back)
            state_tensor = torch.tensor(encoded_state).unsqueeze(0).float().to(device)
            
            # Get policy and value predictions from the network
            with torch.no_grad():
                policy, value = model(state_tensor)
            
            policy = policy.cpu().squeeze().numpy()
            
            # apply mask
            mask = board.available_moves_mask() == False
            policy[mask] = -np.inf
            policy = F.softmax(torch.tensor(policy), dim=0).numpy()
            # print(policy)

            # Sample a move from the policy
            # (1 − epsilon) * policy_np + epsilon * 1 / num_actions
            move = np.random.choice(len(policy), p=policy)
            # print(move, policy[move])
            
            row = -1 if move == 81 else move // 9
            col = -1 if move == 81 else move % 9

            if move == 81:
                if board.move % 2 == 0:
                    p1_pass = True
                else:
                    p2_pass = True
            
            if p1_pass and p2_pass:
                break

            # Record the state, policy, and move
            game_history.append((encoded_state, policy))
            
            # Play the move
            board = board.create_child((row, col))
        
        # Determine the game outcome
        outcome = board.compute_winner()  # AReturns 1 if player 1 wins and -1 if player 2 wins
        
        # Add the game history to the training data
        for state, policy in game_history:
            print(policy)
            exit()
            game_data.append((state, policy, outcome))
    
    return game_data


def train(model: AlphaZeroNet, game_data, batch_size=32, lr=0.001, checkpoint_dir="checkpoints"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    prev_val_loss = float('inf')
    patience, curr_patience = 5, 0
    epoch = 0
    
    while curr_patience < patience:
        np.random.shuffle(game_data)
        for i in range(0, len(game_data), batch_size):
            batch = game_data[i:i+batch_size]
            states, policies, outcomes = zip(*batch)
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            policies = torch.tensor(np.array(policies), dtype=torch.float32).to(device)
            outcomes = torch.tensor(np.array(outcomes), dtype=torch.float32).unsqueeze(1).to(device)
            
            # Forward pass
            policy_pred, value_pred = model(states)
            
            # Compute loss
            policy_loss, value_loss = calc_loss(policy_pred, value_pred, policies, outcomes)
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (loss.item() < prev_val_loss):
            curr_patience = 0
            prev_val_loss = loss.item()
        else:
            curr_patience += 1
        
        epoch += 1

        print(f"Epoch {epoch} \tLoss: {loss.item()} \tPolicy Loss: {policy_loss.item()} \tValue Loss: {value_loss.item()}")
        save_checkpoint(model, epoch, checkpoint_dir, {})


torch.manual_seed(TRAIN_PARAMS["seed"])
np.random.seed(TRAIN_PARAMS["seed"])

# # parameters to training
num_games = 50
num_iter = 10

device = torch.device("mps")
model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"]).to(device)
board = GameNode(size=9)

# game_data = []
# for filename in os.listdir('games'):
#     # print(filename)
#     generate_training_data_from_games([f"games/{filename}"], game_data)

# train(model, game_data, batch_size=32, lr=0.001, checkpoint_dir=f"checkpoint_pretrain")

for i in range(num_iter):
    print(f"starting self play {i}...")
    game_data = self_play(model, num_games=num_games, look_back=MODEL_PARAMS["lookback"])
    print(f"starting train {i}...")
    train(model, game_data, batch_size=32, lr=0.01, checkpoint_dir=f"checkpoint_dir_{i}")



