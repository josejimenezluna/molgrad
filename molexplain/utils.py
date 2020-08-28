import os
import torch

ROOT_PATH = os.path.dirname(__file__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = os.path.join(ROOT_PATH, "data")
RAW_DATA_PATH = os.path.join(ROOT_PATH, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, "data", "processed")
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
FIGURES_PATH = os.path.join(ROOT_PATH, "results", "figures")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
BASELINE_MODELS_PATH = os.path.join(ROOT_PATH, "baseline_models")
LOG_PATH = os.path.join(ROOT_PATH, "logs")
