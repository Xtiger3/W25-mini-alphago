# config.py

# Data parameters
GAME_PARAMS = {
    "size": 9,
    "num_actions": 9 * 9 + 1,  # 9x9 board + pass
}

# Model parameters
MODEL_PARAMS = {
    "lookback": 3,
    "in_channels": 3 * 2 + 1,  # 3 * 2 for lookback, 1 for current player
    # "input_size": 9 * 9,  # Example for a 9x9 board
    # "hidden_size": 256,
    # "num_layers": 5,
    # "output_size": 9 * 9 + 1,  # Policy output + value output
}

# Training parameters
TRAIN_PARAMS = {
    "seed": 42,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "patience": 5,  # Early stopping patience
}

# Checkpoint and logging
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


# network parametrs
INPUT_CHANNELS = 3 * 2 + 1
OUTPUT_CHANNELS = 9 * 9 + 1
KERNEL = 3
NUM_RESIDUALS = 2