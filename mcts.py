from preprocessing import encode
from typing import Dict
import torch
from network import AlphaZeroNet

class MCTSNode:
    def __init__(self, state, parent=None, prior=0):
        self.state = state  # The game state associated with this node
        self.parent : MCTSNode = parent  # The parent node
        self.children : Dict[int, MCTSNode] = {}  # A dictionary of child nodes, indexed by action

        self.visit_count = 0
        self.total_action_value = 0
        self.mean_action_value = 0
        self.prior = prior

    def is_leaf(self):
        """Check if the node is a leaf (i.e., no children)."""
        return len(self.children) == 0

    def is_root(self):
        """Check if the node is the root of the tree."""
        return self.parent is None

    def select_child(self, exploration_weight=1.0):
        """
        Select a child node using AlphaGo Zero's PUCT formula.
        
        Args:
            exploration_weight: A constant determining the level of exploration.
        
        Returns:
            The selected child node and the corresponding action.
        """
        
        return max(self.children.items(), key=lambda item: item[1].ucb_score(exploration_weight))

    def ucb_score(self, exploration_weight):
        """
        Calculate the Upper Confidence Bound score for this node.
        
        Args:
            exploration_weight: A constant determining the level of exploration.
        
        Returns:
            The Upper Confidence Bound score.
        """
        if self.visit_count == 0:
            return float('inf')  # Ensure unvisited nodes are prioritized
        exploitation = self.mean_action_value
        exploration = exploration_weight * self.prior * (self.parent.visit_count ** 0.5) / (1 + self.visit_count)
        return exploitation + exploration

    def expand_and_evaluate(self, model:AlphaZeroNet, state):
        """
        Expand the node by creating child nodes for each possible action.
        
        Args:
            action_probs: A list of tuples (action, probability) from the policy network.
        
        Returns:
            The value for back up.
        """
        
        action_probabilities, value = model(state)
        # TODO: dihedral reflection or rotation selected unifromly at random from i = 1..8
        # TODO: positions in queue are evaluated by the nn using a mini-batch size of 8
        for action, prob in enumerate(action_probabilities):
            if action not in self.children:
                self.children[action] = MCTSNode(None, self, prob)
        # TODO: the value v is backed up
        return value
        
    def backup(self, value):
        """
        Update the node's action value and visit count after a simulation.
        
        Args:
            value: The value obtained from the simulation.
        """
        self.visit_count += 1
        self.total_action_value += value
        self.mean_action_value = self.total_action_value / self.visit_count


class MCTS:
    def __init__(self, model, exploration_weight=1.0, num_simulations=100):
        self.model = model  # The neural network model
        self.exploration_weight = exploration_weight  # Exploration weight for UCB
        self.num_simulations = num_simulations  # Number of simulations to run

    def run(self, root_state):
        """
        Run the MCTS algorithm from the given root state.
        
        Args:
            root_state: The initial game state.
        
        Returns:
            A dictionary mapping actions to their visit counts.
        """
        root = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Select: Traverse the tree until a leaf node is reached
            while not node.is_leaf():
                action, node = node.select_child(self.exploration_weight)
                search_path.append(node)

            # Expansion: Expand the leaf node if the game is not over
            if not node.state.is_terminal():
                encoded_state = encode(node.state, lookback=3)
                state_tensor = torch.tensor(encoded_state).unsqueeze(0).float()
                with torch.no_grad():
                    action_probs, _ = self.model(state_tensor)
                action_probs = action_probs.squeeze().numpy()
                node.expand(zip(range(len(action_probs)), action_probs))

            # Simulation: Evaluate the leaf node using the neural network
            encoded_state = encode(node.state, lookback=3)
            state_tensor = torch.tensor(encoded_state).unsqueeze(0).float()
            with torch.no_grad():
                _, value = self.model(state_tensor)
            value = value.item()

            # Backpropagation: Update the values along the search path
            for node in reversed(search_path):
                node.backup(value)

        # Return the visit counts for each action
        return {action: child.visit_count for action, child in root.children.items()}