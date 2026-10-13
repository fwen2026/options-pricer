import sys
sys.path.insert(0, '.')

from src.data.synthetic_iv_dataset import generate_full_dataset
from src.data.dataset_io import save_dataset
from src.util.synthetic_iv_config import load_synthetic_iv_config

cfg = load_synthetic_iv_config("config/synthetic_iv_datasets/synthetic_iv.yaml")

dataset = generate_full_dataset(cfg=cfg)
path = cfg["save_path"]
save_dataset(dataset, path, compressed=True)
