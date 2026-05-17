import sys
sys.path.insert(0, '.')

from src.data.iv_dataset_generator import generate_full_dataset, save_dataset
from src.util.synthetic_iv_config import load_synthetic_iv_config

cfg = load_synthetic_iv_config("config/synthetic_iv_datasets/synthetic_iv.yaml")
dataset = generate_full_dataset(cfg=cfg)
save_dataset(dataset, cfg)
