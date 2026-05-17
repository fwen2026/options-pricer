import numpy as np
from pathlib import Path


def save_dataset(dataset : dict, path : str | Path, compressed : bool):
    save_path = path
    if compressed:
        np.savez_compressed(save_path, **dataset)
    else:
        np.savez(save_path, **dataset)
    print(f"Dataset saved to {save_path}")
    

def load_dataset(path : str | Path) -> dict:
    load_path = path
    with np.load(load_path) as data:
        return {key: data[key] for key in data.files}
    

def load_single_datapoint(path : str | Path, index: int, label: str) -> dict:
    with np.load(path) as data:
        dataset_filter = data[label] == index
        return {key: arr[dataset_filter] for key, arr in data.items()}
    
    
def train_eval_test_split(cfg, dataset: dict):
    """Since we don't train the model on BS, split by surface."""
    cfg_dataset = cfg["dataset"]

    train_id_threshold = int(cfg_dataset["num_samples"] * cfg_dataset["split"]["train"])
    val_id_threshold = int(train_id_threshold + cfg_dataset["num_samples"] * cfg_dataset["split"]["val"])

    train_mask = dataset["surface_id"] < train_id_threshold
    val_mask = (dataset["surface_id"] >= train_id_threshold) & (dataset["surface_id"] < val_id_threshold)
    test_mask = dataset["surface_id"] >= val_id_threshold

    train_data = {key: arr[train_mask] for key, arr in dataset.items()}
    eval_data = {key: arr[val_mask] for key, arr in dataset.items()}
    test_data = {key: arr[test_mask] for key, arr in dataset.items()}

    return train_data, eval_data, test_data