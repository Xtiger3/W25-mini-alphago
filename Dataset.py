import imported_game
import numpy as np
import os
from data_preprocess import node_to_tensor


def human_target_policy(gameNode):
    """ return a board array with a 1 where the human player moved"""
    target = np.zeros(gameNode.size * gameNode.size + 1)

    # handle pass
    if gameNode.prev_move == (-1, -1) or gameNode.prev_move is None:
        target[-1] = 1

    # handle move
    else:
        row = gameNode.prev_move[0]
        col = gameNode.prev_move[1]
        target[gameNode.size * row + col] = 1

    return target


class Dataset:
    def __init__(self):
        self.positional_data = []

    def parse_game_file(self, game_data):
        """creates (s, z, 𝛑) tuples from a game file"""
        game = imported_game.ImportedGame(game_data)

    def load_games(self, game_directory):
        for i, game_file in enumerate(dirs := os.listdir(game_directory)):
            print(f"[{i}/{len(dirs)}] Loading game {game_file}")
            filepath = os.path.join(game_directory, game_file)
            this_game = imported_game.ImportedGame(filepath)

            # pare through the game
            node = this_game.linked_list()
            last_human_policy = human_target_policy(node)
            while (node := node.prev) is not None:
                s = node_to_tensor(node)
                z = this_game.meta.get('final_eval')
                pi = last_human_policy
                last_human_policy = human_target_policy(node)
                self.positional_data.append((s, z, pi))


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_games('games')

    print(len(dataset.positional_data))
    print(dataset.positional_data)
