import plotly.express as px
import plotly.graph_objects as go

from src.data.dataset_io import load_dataset

def visualize_surface(surface_id: int):
    dataset = load_dataset("path/to/dataset.npz")
    surface_data = dataset["surfaces"][surface_id]
    fig = go.Figure(data=go.Surface(z=surface_data))
    fig.show()