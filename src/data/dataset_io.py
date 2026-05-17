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
    load_path = path
    
    dataset_filter = data[label] == index
    with np.load(load_path) as data:
        return {
            key: arr[dataset_filter]
            for key, arr in data.items()
        }