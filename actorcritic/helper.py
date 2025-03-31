from network import NeuralNet
from game_node import GameNode
from data_preprocess import *

def evaluator(old_model: NeuralNet, curr_model: NeuralNet, num_games: int = 100):
    num_wins = 0
    for game in range(num_games):
        state = GameNode(size=9)  # Initialize a new game

        for move in range(1000):  # Limit the number of moves
            # Encode the board state
            encoded_state = node_to_tensor(state)
            state_tensor = encoded_state.unsqueeze(0).float()
            
            with torch.no_grad():
                # Get policy and value from network
                if game % 2 == move % 2:    # curr is Black for even games, White for odd games
                    action_logits, _ = curr_model(state_tensor)
                else:                       # old is Black for odd games, White for even games
                    action_logits, _ = old_model(state_tensor)
            
                # Apply valid moves mask
                valid_mask = torch.tensor(state.available_moves_mask())
                action_logits = torch.where(valid_mask, action_logits, -1e8)
                action_probs = torch.softmax(action_logits, dim=1).squeeze()
                action = torch.multinomial(action_probs, num_samples=1).item()

            row = -1 if action == 81 else action // 9
            col = -1 if action == 81 else action % 9
            
            # Take the chosen action and observe the next state and reward
            state = state.create_child((row, col))
            if state.is_terminal():
                break

        current_player = 1 if game % 2 == 0 else -1
        if state.compute_winner() == current_player:
            num_wins += 1

    return num_wins / num_games    
