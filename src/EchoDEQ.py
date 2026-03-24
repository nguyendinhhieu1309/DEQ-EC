import numpy as np
import networkx as nx
import pandas as pd


# PyThorch
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

# Typing
from typing import Tuple, List, Dict, Any, Optional
from pandas import DataFrame
from networkx import Graph


from .DEQ import run, run_dynamic


def EchoGAE_algorithm(
    G,
    user_embeddings=None,
    show_progress=True,
    epochs=300,
    hidden_channels=100,
    out_channels=50,
) -> np.ndarray:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create node features
    if user_embeddings is None:
        X = torch.eye(len(G.nodes), dtype=torch.float32, device=DEVICE)
    else:
        X = []
        for node in G.nodes:
            X.append(user_embeddings[node])
        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    # Create edge list
    edge_list = np.array(G.edges).T
    edge_list = torch.tensor(edge_list, dtype=torch.int64).to(DEVICE)

    # Data object
    data = Data(x=X, edge_index=edge_list)
    data = train_test_split_edges(data)

    # Run the model
    model, x, train_pos_edge_index = run(
        data,
        show_progress=show_progress,
        epochs=epochs,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    # Embedding
    GAE_embedding = model.encode(x, train_pos_edge_index).detach().cpu().numpy()

    return GAE_embedding


def _build_snapshot_data(
    G: Graph,
    user_embeddings: Optional[Dict[Any, np.ndarray]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_ids = list(G.nodes())

    if user_embeddings is None:
        X = torch.eye(len(node_ids), dtype=torch.float32, device=device)
    else:
        X = np.array([user_embeddings[node] for node in node_ids], dtype=np.float32)
        X = torch.tensor(X, dtype=torch.float32, device=device)

    # Keep graph directed if present. For undirected networkx graphs, .edges is enough.
    edge_list = np.array(list(G.edges())).T
    if edge_list.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device)

    return {"x": X, "edge_index": edge_index, "node_ids": node_ids}


def EchoDEQ_algorithm(
    temporal_graphs: List[Graph],
    temporal_user_embeddings: Optional[List[Dict[Any, np.ndarray]]] = None,
    show_progress: bool = True,
    epochs: int = 300,
    hidden_channels: int = 100,
    out_channels: int = 50,
    gamma: float = 0.1,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """
    Dynamic Exposure-Aware GAE embedding training across temporal snapshots.
    Returns per-window embeddings and matching node ids to support DEQ-EC scoring.
    """
    if len(temporal_graphs) == 0:
        raise ValueError("temporal_graphs must contain at least one snapshot.")

    if temporal_user_embeddings is not None and len(temporal_user_embeddings) != len(
        temporal_graphs
    ):
        raise ValueError(
            "temporal_user_embeddings must have same length as temporal_graphs."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshots = []
    for idx, graph in enumerate(temporal_graphs):
        emb_dict = None if temporal_user_embeddings is None else temporal_user_embeddings[idx]
        snapshots.append(_build_snapshot_data(graph, emb_dict, device=device))

    out = run_dynamic(
        snapshots=snapshots,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        epochs=epochs,
        lr=lr,
        gamma=gamma,
        show_progress=show_progress,
    )

    return {
        "model": out["model"],
        "embeddings": out["embeddings"],
        "node_ids": [snap["node_ids"] for snap in snapshots],
    }
