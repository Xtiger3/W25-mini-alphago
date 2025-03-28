from typing import Dict
import torch
from network import AlphaZeroNet
from game_node import GameNode
from data_preprocess import *
import time
import math
# device = torch.device("cpu")

def expand(model: AlphaZeroNet, state: GameNode, action_probabilities: torch.Tensor):
    """
    Expand the node by creating child nodes for each possible action.
    
    Args:
        model: The neural network model.
        state: The game state to expand.
    
    Returns:
        The value for back up.
    """

    valid_moves = state.available_moves_mask()
    # assert(state.is_leaf())
    
    # TODO: dihedral reflection or rotation selected unifromly at random from i = 1..8
    # TODO: positions in queue are evaluated by the nn using a mini-batch size of 8
    for action, is_valid in enumerate(valid_moves):
        if is_valid:  # Only process valid moves
            # if action == state.size**2:
            #     continue
            row = -1 if action == state.size**2 else action // state.size
            col = -1 if action == state.size**2 else action % state.size
            
            # Create the new state for the valid move
            new_state = state.create_child((row, col))
            new_state.prior = action_probabilities[0, action].item()
            # assert(len(new_state.nexts) == 0)

    # TODO: the value v is backed up
    # return value


def run_mcts_sims(model: AlphaZeroNet, root_state: GameNode, exploration_weight=1.0, num_simulations=100) -> Dict:
    """
    Run the MCTS algorithm from the given root state.
    
    Args:
        root_state: The initial game state.
    
    Returns:
        A dictionary mapping actions to their visit counts.
    """
    root = root_state

    for i in range(num_simulations):
        node = root
        search_path = [node]

        # Select: Traverse the tree until a leaf node is reached
        while not node.is_leaf():
            # print("old nodes: ", node, len(node.nexts))
            node = max(node.nexts, key=lambda item: item.ucb_score(exploration_weight))
            search_path.append(node)
            # time.sleep(1)
        
        # print("start of new sim: ", len(node.nexts))
        # print(node)
        # time.sleep(1)
        
        # Simulation: Evaluate the leaf node using the neural network
        with torch.no_grad():
            action_probabilities, value = model(node_to_tensor(node).unsqueeze(0))
        if node.prev_move == (-1, -1):
            value = -math.inf

        # Expansion: Expand the leaf node if the game is not over
        if not node.is_terminal():
            expand(model, node, action_probabilities)
            # parallelize by running multiple simulations in parallel with batch
        # else:
        #     print("board is terminal")

        # Backpropagation: Update the values along the search path
        for search_node in reversed(search_path):
            search_node.backup(value)
 
        # time.sleep(1)

    # Return the visit counts for each action
    return {child.prev_move: child.visit_count for child in root.nexts}
    
    
if __name__ == "__main__":
    model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])
    board = GameNode(size=9)
    board_action_value = run_mcts_sims(model, board)
    print(board_action_value, board_action_value[(-1,-1)])

    max_action = max(board_action_value, key=board_action_value.get)
    print(max_action, board_action_value[max_action])