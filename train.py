from network import AlphaZeroNet
from game_node import GameNode
from train_helper import *
import torch
from config import *
from data_preprocess import *
from mcts import *
import time

def self_play(model: AlphaZeroNet, num_games, simulations=100, look_back=3):
    data = []
    for i in range(num_games):
        print("playing game", i)
        board = GameNode(size=9)  # Initialize a new game
        game_history = []
        p1_pass, p2_pass = False, False
        
        while not board.is_terminal() and board.move < iteration + 100:
            # Run MCTS to get policy and value
            board_action_value = run_mcts_sims(model, board, num_simulations=simulations)
            # print(board_action_value)
            # Normalize visit counts to get policy
            total_visits = sum(board_action_value.values())
            policy = {action: visits / total_visits for action, visits in board_action_value.items()}
            
            policy_array = np.zeros(82)
            # Populate the array
            for (i, j), value in policy.items():
                if (i, j) == (-1, -1):
                    index = 81
                else:
                    index = i * 9 + j
                policy_array[index] = value
            
            # Store the state, policy, and value
            encoded_state = encode(board, look_back)
            game_history.append((encoded_state, policy_array))
            
            # Select the action with the highest visit count
            # try:
            max_action = max(board_action_value, key=board_action_value.get)
            # except:
            #     print(board_action_value)
            #     exit()
            board = board.create_child(max_action)  # Play the move

            print(f"player {board.move % 2 + 1} played {max_action}")
            print(board)
            
            if max_action == (-1, -1):
                print(board_action_value)
            
            if board.move % 2 == 0:
                p1_pass = max_action == (-1, -1)
            else:
                p2_pass = max_action == (-1, -1)
            
            if p1_pass and p2_pass:
                break

        # Determine the game outcome (value) for each state in the game history
        if board.is_terminal():
            print("board is terminal")
        outcome = board.compute_winner()
        print("winner is", outcome)
        for i, (state, policy) in enumerate(game_history):
            data.append((state, policy, outcome))
    
    return data


def train_model(model, data, epochs, batch_size, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.MSELoss()
    states, policies, outcomes = zip(*data)

    # Convert data to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32)
    policies = torch.tensor(np.array(policies), dtype=torch.float32)
    values = torch.tensor(np.array(outcomes), dtype=torch.float32).unsqueeze(1)
    # states = torch.tensor([d[0] for d in data]).float().to(device)
    # policies = torch.tensor([d[1] for d in data]).float().to(device)
    # values = torch.tensor([d[2] for d in data]).float().to(device)
    
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
        del batch_states, batch_policies, batch_values, policy_pred, value_pred
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")
        save_checkpoint(model, iteration, f"checkpoints", {})
        time.sleep(100)
        
        
def evaluate_model(new_model, old_model, num_eval_games):
    new_model_wins = 0
    for _ in range(num_eval_games):
        board = GameNode(size=9)
        current_player = 0  # 0 for new model, 1 for old model
        
        while not board.is_terminal():
            if current_player == 0:
                board_action_value = run_mcts_sims(new_model, board)
            else:
                board_action_value = run_mcts_sims(old_model, board)
            
            max_action = max(board_action_value, key=board_action_value.get)
            board = board.create_child(max_action)
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
    
    # model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"]).to(device)
    best_model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])

    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        # Self-play and data generation
        data = self_play(best_model, num_self_play_games, num_mcts_simulations)
        
        # Train the model
        print("Starting train...")
        train_model(best_model, data, num_epochs, batch_size, learning_rate)
        
        # Evaluate the new model
        # print("Starting evaluation...")
        # win_rate = evaluate_model(best_model, best_model, eval_games)
        # print(f"Win rate against previous model: {win_rate * 100:.2f}%")
        
        # Save the best model
        # if win_rate >= 0.55:  # Only update if the new model is significantly better
        #     best_model = model