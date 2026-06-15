import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ephishLLM.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data Splitting Configurations
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Hyperparameters
LR_MAX_ITER = 1000
LR_RANDOM_STATE = 42
SVM_RANDOM_STATE = 42
HGB_RANDOM_STATE = 42

# Engineered Numeric Features
NUMERIC_FEATURES = [
    "subject_len",
    "body_len",
    "subject_word_count",
    "body_word_count",
    "body_digit_count",
    "body_url_count",
    "body_special_char_count"
]

TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5

