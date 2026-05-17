import sys
sys.path.insert(0, '.')

import argparse
import numpy as np

from src.data.visualization import visualize_surface, save_figure
from src.util.synthetic_iv_config import load_synthetic_iv_config

if __name__ == "__main__":
    cfg = load_synthetic_iv_config("config/synthetic_iv_datasets/synthetic_iv.yaml")

    parser = argparse.ArgumentParser(description="Visualize a single IV surface from the dataset")
    parser.add_argument("--path", type=str, default=cfg["save_path"],
                        help="Path to the .npz dataset (default: config save_path)")
    parser.add_argument("--surface-id", type=int, default=None,
                        help="Surface ID to plot (default: random)")
    parser.add_argument("--total-variance", action="store_true",
                        help="Plot total variance instead of IV")
    parser.add_argument("--heatmap", action="store_true",
                        help="Use heatmap instead of 3D surface")
    parser.add_argument("--output", type=str, default=None,
                        help="Save to file (.html or .png) instead of opening in browser")
    args = parser.parse_args()

    with np.load(args.path) as data:
        ids = np.unique(data["surface_id"])

    sid = args.surface_id if args.surface_id is not None else int(np.random.choice(ids))
    print(f"Plotting surface_id={sid}  total_variance={args.total_variance}  heatmap={args.heatmap}")

    fig = visualize_surface({"dataset_path": args.path}, sid, args.total_variance, args.heatmap)

    if args.output:
        ext = args.output.rsplit(".", 1)[-1]
        save_figure(fig, args.output, ext)
        print(f"Saved to {args.output}")
    else:
        fig.show()
