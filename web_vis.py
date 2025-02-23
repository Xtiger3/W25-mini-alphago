"""
Usage:
1. Verify flask is installed (run "pip install -r requirements.txt" if not)
2. Run "flask --app web_vis run" in a terminal
"""

# Imports
from flask import Flask, request, jsonify
import torch

from game_node import GameNode
from network import NeuralNet

from data_preprocess import node_to_tensor

# Model setup
MODEL_STATE_DICT_PATH = "model.pt" # Update this as needed

model = NeuralNet()

try:
    model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, weights_only=True))
except:
    print(f"Failed to load model at {MODEL_STATE_DICT_PATH}")

    res = ""

    while res not in list("yn"):
        res = input("Load random model (y/n)? ")
        res = res.lower()
    
    if res == "n":
        print("Program exited early: cannot run without model")
        exit(1)

# Set up game node
SIZE = 9
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

    policy, val = model(node_to_tensor(curr_node).unsqueeze(0))

    policy = policy.softmax(1).flatten().detach()

    policy /= policy.max()
    policy = policy / 5

    policy *= torch.tensor(curr_node.available_moves_mask())

    return jsonify({
        "policy": policy.tolist(),
        "value": val.detach().item()
    }), 200
