from board import Board
from game_node import GameNode
import numpy as np

# shape = channels * 9 * 9
# channels = 2 * n turns + turn matrix
def encode(board: GameNode, look_back: int) -> np.ndarray:
    size = board.size
    num_feature_planes = 2 * look_back + 1

    # Initialize the feature planes
    feature_planes = np.zeros((num_feature_planes, size, size), dtype=np.int32)
    
    # Plane turn
    feature_planes[num_feature_planes - 1] = board.move % 2

    for n in range(look_back):
        # Plane even: Black stones
        feature_planes[2 * n] = (board.grid == 1)

        # Plane odd: White stones
        feature_planes[2 * n + 1] = (board.grid == 2)
        
        board = board.prev
        
        if (board is None):
            break

    return feature_planes


board = GameNode(9)
board = board.create_child((1, 1))
board = board.create_child((1, 2))
board = board.create_child((1, 3))
board = board.create_child((1, 4))
board = board.create_child((1, 5))
# print(board.prev)
encoded_board = encode(board, 3)
# print(encoded_board)
print(encoded_board.shape)

# black vs white?