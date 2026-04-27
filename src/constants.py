"""
Project configuration constants.
"""

# ---- Numerical stability ----
LOG_EPSILON = 1e-8

# ---- Data generation ----
DATA_TRAIN_SIZE = 100000
DATA_TEST_SIZE = 10000
DATA_EVAL_FRACTION = 0.1

S_RANGE = (20, 140)
LOG_MONEYNESS_RANGE = (-2, 2)
T_RANGE = (0.01, 2.0)
R_DEFAULT = 0.02
SIGMA_RANGE = (0.1, 0.5)
SIGMA_CLIP = (0.05, 0.6)

# ---- Model architecture ----
MODEL_LAYERS = [4, 128, 128, 64, 1]

# ---- Training Hyperparameters ----
# NN
TRAIN_EPOCHS = 400
TRAIN_LR = 1e-3
TRAIN_PATIENCE = 25
TRAIN_MIN_DELTA = 5e-4
TRAIN_LOSS_ALPHA = 0.5  # weights both rel and abs equally
# Elastic Net
# TRAIN_ALPHA = 0.5
# TRAIN_L1_RATIO = 0.7 # weights L1 slightly more

# ---- Directories ----
NN_MODEL_PATH = "artifacts/models/bs_model.pth"
ELASTIC_MODEL_PATH = "artifacts/models/elastic_model.joblib"
XGBOOST_MODEL_PATH = "artifacts/models/xgboost_model.ubj"
TRAIN_DATA_PATH = "artifacts/data/bs_dataset.npz"
EVAL_DATA_PATH = "artifacts/data/bs_dataset_eval.npz"
TEST_DATA_PATH = "artifacts/data/test_dataset.npz"
PLOTS_DIR = "artifacts/plots"

# ---- Seeds ----
TRAIN_DATA_SEED = 42
TEST_DATA_SEED = 123
TRAINING_SEED = 42