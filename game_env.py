from typing import Tuple
from game_node import GameNode
from preprocessing import encode
from network import AlphaZeroNet
import numpy as np
import torch.nn.functional as F
import torch
from train_helper import restore_checkpoint
from config import *
import time
model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])
random_model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])

board = GameNode(9)
players = ["bot", "bot2"]
turn = 0

p1_pass, p2_pass = False, False

model, _, _ = restore_checkpoint(model, "checkpoint_dir_6", force=True)

while not board.is_terminal():
    print(board)
    print()
    
    if players[turn] == "bot":
        state_tensor = torch.tensor(encode(board, look_back=MODEL_PARAMS["lookback"])).unsqueeze(0).float()
        with torch.no_grad():
            policy, value = model(state_tensor)
        policy = policy.squeeze().numpy()
        mask = board.available_moves_mask() == False
        policy[mask] = -np.inf
        move = np.argmax(policy)
        row = -1 if move == 81 else move // 9
        col = -1 if move == 81 else move % 9
        p1_pass = move == 81
        board = board.create_child((row, col))
    elif players[turn] == "bot2":
        state_tensor = torch.tensor(encode(board, look_back=MODEL_PARAMS["lookback"])).unsqueeze(0).float()
        with torch.no_grad():
            policy, value = random_model(state_tensor)
        policy = policy.squeeze().numpy()
        mask = board.available_moves_mask() == False
        policy[mask] = -np.inf
        move = np.argmax(policy)
        row = -1 if move == 81 else move // 9
        col = -1 if move == 81 else move % 9
        p2_pass = move == 81
        board = board.create_child((row, col))
    else:
        # print(board.available_moves_mask())
        move = input("Enter move (format it as row, col e.g. 0,0): ")
        row, col = map(int, move.split(','))
        board = board.create_child((int(row), int(col)))
    time.sleep(0.1)
    if p1_pass and p2_pass:
        break
    turn = 1 if turn == 0 else 0
    
print(board.compute_winner())
