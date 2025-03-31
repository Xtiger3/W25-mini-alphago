from typing import Tuple
import torch
from network import NeuralNet
from game_node import GameNode
from data_preprocess import *
import numpy as np
import math

# device = torch.device("cpu")

def expand(state: GameNode, action_logits: torch.Tensor):
    """
    Expand the node by creating child nodes for each possible action.
    
    Args:
        state: The game state to expand.
        action_logits: The action logits from the neural network.
    """

    valid_moves_mask = torch.tensor(state.available_moves_mask())
    valid_moves_mask[81] = False
    action_logits = torch.where(valid_moves_mask, action_logits, -1e8)
    action_probs = torch.softmax(action_logits, dim=1).squeeze()

    # TODO: dihedral reflection or rotation selected unifromly at random from i = 1..8
    # TODO: positions in queue are evaluated by the nn using a mini-batch size of 8
    for action, is_valid in enumerate(valid_moves_mask):
        if is_valid:  # Only process valid moves
            row = -1 if action == state.size**2 else action // state.size
            col = -1 if action == state.size**2 else action % state.size
            
            # Create the new state for the valid move
            new_state = state.create_child((row, col))
            new_state.prior = action_probs[action].item()
    if len(state.nexts) == 0:
        new_state = state.create_child((-1, -1))
        new_state.prior = action_probs[81].item()

def select_self_play_move(state: GameNode) -> GameNode:
    """
    Select a move based on the visit counts of the child nodes.
    
    Args:
        state: The current game state.
        
    Returns:
        The selected child node (action).
    """
    visit_counts = [child.visit_count for child in state.nexts]

    selected_action = np.random.choice(state.nexts, p=np.array(visit_counts)/sum(visit_counts))
    return selected_action
    

def run_mcts_sims(model: NeuralNet,
                  root_state: GameNode,
                  exploration_weight=1.0,
                  num_simulations=100) -> Tuple[torch.Tensor, GameNode]:
    """
    Run the MCTS algorithm from the given root state.
    
    Args:
        root_state: The initial game state.
    
    Returns:
        A tuple containing the policy tensor and the next board state.
    """
    root = root_state

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Select: Traverse the tree until a leaf node is reached
        while not node.is_leaf():
            node = max(node.nexts, key=lambda item: item.ucb_score(exploration_weight))
            search_path.append(node)

        # Simulation: Evaluate the leaf node using the neural network
        with torch.no_grad():
            action_logits, value = model(node_to_tensor(node).unsqueeze(0))
        # if node.prev_move == (-1, -1):
        #     value = 0

        # Expansion: Expand the leaf node if the game is not over
        if not node.is_terminal():
            expand(node, action_logits)

        # Backpropagation: Update the values along the search path
        for search_node in reversed(search_path):
            search_node.backup(value)

    # Create policy tensor
    policy_tensor = torch.zeros(82)
    visit_counts = torch.tensor([child.visit_count for child in root.nexts])
    indices = torch.tensor([81 if child.prev_move == (-1, -1) else 
                        child.prev_move[0] * 9 + child.prev_move[1] 
                        for child in root.nexts])
    policy_tensor.scatter_(0, indices, visit_counts / visit_counts.sum())
    
    next_board = select_self_play_move(root)
    root.nexts = [] # Clear the nexts to avoid memory leaks
    
    # Return the visit counts for each action
    return policy_tensor, next_board
    
    
if __name__ == "__main__":
    model = NeuralNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])
    board = GameNode(size=9)
    board_action_value = run_mcts_sims(model, board)
    print(board_action_value, board_action_value[(-1,-1)])

    max_action = max(board_action_value, key=board_action_value.get)
    print(max_action, board_action_value[max_action])