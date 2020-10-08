import os

ROOT_PATH = os.path.dirname(__file__)


DATA_PATH = os.path.join(ROOT_PATH, "data")
RAW_DATA_PATH = os.path.join(ROOT_PATH, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, "data", "processed")
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
FIGURES_PATH = os.path.join(ROOT_PATH, "results", "figures")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
BASELINE_MODELS_PATH = os.path.join(ROOT_PATH, "baseline_models")
LOG_PATH = os.path.join(ROOT_PATH, "logs")
FIG_PATH = os.path.join(ROOT_PATH, "figures")
EXAMPLE_PATH = os.path.join(DATA_PATH, "examples")
