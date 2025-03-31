import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import itertools
from imported_game import ImportedGame
from data_preprocess import *
import numpy as np


def save_checkpoint(model: nn.Module, epoch: int, checkpoint_dir: str, stats: list):
    """
    Save the 'model' parameters, the cumulative stats, and current epoch number as a checkpoint file (.pth.tar) in 'checkpoint_dir'. 
    Args:
        model: The model to be saved.
        epoch (int): The current epoch number.
        checkpoint_dir (str): Directory where the checkpoint file will be saved.
        stats (list): A cumulative list consisted of all the model accuracy, loss, and AUC for every epoch up to the current epoch. 
             Note: almost always use the last element of stats -- stats[-1] -- which represents the most recent stats. 

    Description:
        This function saves the current state of the model, including its parameters, epoch number, and
        training statistics to a checkpoint file. The checkpoint file is named according to the current
        epoch, and if the specified directory does not exist, it will be created.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir,exist_ok=True)
    torch.save(state, filename)


def restore_checkpoint(model: nn.Module, checkpoint_dir: str, cuda: bool = False, force: bool = False, pretrain: bool = False):
    """
    Restore model from checkpoint if it exists.

    Args:
        model (torch.nn.Module): The model to be restored.
        checkpoint_dir (str): Directory where checkpoint files are stored.
        cuda (bool, optional): Whether to load the model on GPU if available. Defaults to False.
        force (bool, optional): If True, force the user to choose an epoch. Defaults to False.
        pretrain (bool, optional): If True, allows partial loading of the model state (used for pretraining). Defaults to False.

    Returns:
        tuple: The restored model, the starting epoch, and the list of statistics.

    Description:
        This function attempts to restore a saved model from the specified `checkpoint_dir`.
        If no checkpoint is found, the function either raises an exception (if `force` is True) or returns
        the original model and starts from epoch 0. If a checkpoint is available, the user can choose which
        epoch to load from. The model's parameters, epoch number, and training statistics are restored.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        try:
            print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
            inp_epoch = int(input())
            if inp_epoch not in range(1, epoch + 1):
                raise Exception("Invalid epoch number")
        except:
            print("Which epoch to load from?")
            inp_epoch = int(input())

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in checkpoint_dir."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def calc_loss(policy_pred, value_pred, policy_target, value_target):
    policy_loss = F.cross_entropy(policy_pred, policy_target)
    value_loss = F.mse_loss(value_pred, value_target)
    return policy_loss, value_loss


def generate_training_data_from_games(game_paths, game_data, look_back=3):
    training_data = []
    
    for game_path in game_paths:
        game = ImportedGame(game_path)
        node = game.linked_list()
        
        # Move to the head of the linked list
        while node.prev is not None:
            # print(node)
            node = node.prev
        
        # Traverse the game and generate training data
        while True:
            if len(node.nexts) == 0:
                break
            
            # Encode the current state
            encoded_state = node_to_tensor(node)
            
            # Get the next move (policy)
            next_node = node.nexts[0]
            # print(node.nexts)
            policy = next_node.grid.flatten() - node.grid.flatten()
            policy[policy > 0] = 1
            policy = np.append(policy, 0)
            
            # Determine the outcome (winner)
            outcome = game.meta["final_eval"]
            
            # Add to training data
            game_data.append((encoded_state, policy, outcome))
            
            # Move to the next node
            node = next_node
    
    # return training_data