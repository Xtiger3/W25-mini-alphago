from network import NeuralNet
from game_node import GameNode
from alphazero.helper import *
import torch
from config import *
from data_preprocess import *
from mcts import *
from copy import deepcopy
import time

def self_play(model: NeuralNet, num_games: int, simulations=100):
    game_history = { 'states': [], 'policies': [], 'outcomes': []}

    for i in range(num_games):
        print("playing game", i)
        board = GameNode(size=9)  # Initialize a new game
        
        while not board.is_terminal():
            # Run MCTS to get policy and action
            policy_tensor, next_board = run_mcts_sims(model, board, num_simulations=simulations)

            # Store the state, policy, and value
            game_history['states'].append(node_to_tensor(board))
            game_history['policies'].append(policy_tensor)

            print(f"player {board.move % 2 + 1} played {next_board.prev_move}")
            
            board = next_board  # Play the move
            print(board)
            
            # clearly lost games are resigned
            # TODO: disable resignation for 10% of the games
            if board.early_termination():
                print("Game terminated early")
                break

        # Determine the game outcome (value) for each state in the game history
        outcome = board.compute_winner()
        print("winner is", outcome)
        for _ in range(board.move):
            game_history['outcomes'].append(outcome)
    
    # At the end, stack all tensors
    states = torch.stack(game_history['states'])
    policies = torch.stack(game_history['policies'])
    outcomes = torch.tensor(game_history['outcomes'], dtype=torch.float32).unsqueeze(1)
    return states, policies, outcomes


def train_model(model, states, policies, values, epochs, batch_size, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(states, policies, values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            policy_pred, value_pred = model(batch_states)
            
            # Compute losses
            policy_loss = criterion_policy(policy_pred, batch_policies)
            value_loss = criterion_value(value_pred, batch_values)
            total_loss = policy_loss + value_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")
        save_checkpoint(model, iteration, f"checkpoints", {})


def evaluate_model(old_model: NeuralNet, new_model: NeuralNet, num_eval_games: int, num_mcts_sims: int = 20):
    new_model_wins = 0
    for game in range(num_eval_games):
        board = GameNode(size=9)
        # Alternate starting player for each game
        current_player = game % 2  # 0 for new model, 1 for old model
        
        while not board.is_terminal() and not board.early_termination():
            if current_player == 0:
                _, board = run_mcts_sims(new_model, board, num_simulations=num_mcts_sims)
            else:
                _, board = run_mcts_sims(old_model, board, num_simulations=num_mcts_sims)
            print(board)
            current_player = 1 - current_player  # Switch players

        outcome = board.compute_winner()
        if outcome == 1 and current_player == 1:
            new_model_wins += 1
        elif outcome == -1 and current_player == 0:
            new_model_wins += 1
    
    return new_model_wins / num_eval_games


# device = torch.device("cpu")

if __name__ == "__main__":
    
    torch.manual_seed(TRAIN_PARAMS["seed"])
    np.random.seed(TRAIN_PARAMS["seed"])
    
    num_iterations = 10000
    num_self_play_games = 1
    num_mcts_simulations = 100
    num_epochs = 1
    batch_size = 16
    learning_rate = 0.001
    eval_games = 10
    
    best_model = NeuralNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])
    curr_model = deepcopy(best_model)


    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Self-play and data generation
        states, policies, outcomes = self_play(curr_model, num_self_play_games, num_mcts_simulations)
        
        # Train the model
        print("Starting train...")
        train_model(curr_model, states, policies, outcomes, num_epochs, batch_size, learning_rate)
        
        # Evaluate the new model
        print("Starting evaluation...")
        win_rate = evaluate_model(best_model, curr_model, eval_games, 50)
        print(f"Win rate against previous model: {win_rate * 100:.2f}%")
        
        # Save the best model
        if win_rate >= 0.55:  # Only update if the new model is significantly better
            best_model = deepcopy(curr_model)