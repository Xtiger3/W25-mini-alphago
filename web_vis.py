"""
Usage:
1. Verify flask is installed (run "pip install -r requirements.txt" if not)
2. Run "flask --app web_vis run" in a terminal
"""

# Imports
from flask import Flask, request, jsonify
import torch

from game_node import GameNode
from train_helper import restore_checkpoint
from network import AlphaZeroNet
from config import *
# from data_preprocess import encode

from data_preprocess import node_to_tensor

# Set up game node
SIZE = 9

model = AlphaZeroNet(MODEL_PARAMS["in_channels"], GAME_PARAMS["num_actions"])

model, _, _ = restore_checkpoint(model, "checkpoint_dir_9", force=True)

curr_node = GameNode(SIZE)

# Game node utils
def small_string(node: GameNode):
    global SIZE
    invert = lambda s: s.replace("○", "B").replace("●", "W").replace("W", "○").replace("B", "●")
    return "\n".join([invert(s.replace(" ", "")[-SIZE:]) for s in str(node).split("\n")[3:]])

# Flask things (assumes model behaves well)
app = Flask(__name__, static_folder="web_vis")

@app.route("/")
def main():
    return app.send_static_file("index.html")

@app.route("/play_move", methods=["POST"])
def play_move():
    global curr_node, SIZE

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    if "row" not in data or "col" not in data:
        return jsonify({"error": "JSON data missing row and/or col fields"}), 400
    
    if (data["row"], data["col"]) != (-1, -1) and (not (0 <= data["row"] < SIZE) or not (0 <= data["col"] < SIZE)):
        return jsonify({"error": f"Specified location {data['row'], data['col']} is out of bounds"}), 400

    if not curr_node.is_valid_move(data["row"], data["col"]):
        return jsonify({"error": f"Specified location {data['row'], data['col']} is an invalid move"}), 400

    curr_node = curr_node.create_child((data["row"], data["col"]))

    return "Good", 200

@app.route("/get_board", methods=["POST"])
def get_board():
    return small_string(curr_node), 200

@app.route("/reset", methods=["POST"])
def reset():
    global curr_node, SIZE

    curr_node = GameNode(SIZE)

    return "Good", 200

@app.route("/undo", methods=["POST"])
def undo():
    global curr_node
    
    if curr_node.prev is None:
        return jsonify({"error": "No move to undo"}), 400

    curr_node = curr_node.prev

    return "Good", 200

@app.route("/network", methods=["POST"])
def network():
    global curr_node

    # state_tensor = torch.tensor(encode(curr_node, look_back=MODEL_PARAMS["lookback"])).unsqueeze(0).float()
    # policy, val = model(state_tensor)
    policy, val = model(node_to_tensor(curr_node).unsqueeze(0))

    policy = policy.softmax(1).flatten().detach()

    policy /= policy.max()
    policy = policy / 5

    policy *= torch.tensor(curr_node.available_moves_mask())

    return jsonify({
        "policy": policy.tolist(),
        "value": val.detach().item()
    }), 200
