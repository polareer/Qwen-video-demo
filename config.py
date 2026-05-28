"""
Project configuration
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Model
MODEL_CONFIG = {
    "name": "SimpleCNN",
    "num_classes": 10,
    "input_size": (3, 32, 32),  # channels, height, width
}

# Training
TRAIN_CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",
}

# Dataset
DATASET_CONFIG = {
    "name": "CIFAR10",
    "num_classes": 10,
    "image_size": 32,
}