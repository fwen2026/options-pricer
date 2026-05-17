

# import plotly.express as px
# import plotly.graph_objects as go
# import numpy as np

# from src.data.dataset_io import load_single_datapoint

# def visualize_surface_3d(cfg, surface_id: int, use_total_variance: bool, use_heatmap: bool):
#     surface_data = load_single_datapoint(cfg["dataset_path"], surface_id, "surface_id")

#     x_raw = surface_data["log_moneyness"]
#     y_raw = surface_data["T"]
#     z_raw = surface_data["total_variance"] if use_total_variance else surface_data["iv"]

#     x_unique = np.unique(x_raw)
#     y_unique = np.unique(y_raw) # uniqueness check. probably unnecessary but still good to have

#     z_grid = np.reshape(z_raw, (len(y_unique), len(x_unique)))
#     plot_style = go.Surface if not use_heatmap else go.Heatmap
#     fig = go.Figure(data=[plot_style(
#         x=x_unique,
#         y=y_unique,
#         z=z_grid,
#         colorscale='Viridis',
#         colorbar=dict(
#             title='Total Variance' if use_total_variance else 'Implied Volatility'
#             ),
#     )])
    
#     return fig

# def savez

# if __name__ == "__main__":
#     visualize_surface_3d(cfg, surface_id, use_total_variance, use_heatmap)