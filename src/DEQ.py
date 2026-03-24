import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()

        # Layer 1:
        # cached only for transductive learning
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)

        # Layer 2:
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def __train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, train_pos_edge_index)

    # Compute loss
    loss = model.recon_loss(z, train_pos_edge_index)

    loss.backward()

    # Update parameters
    optimizer.step()

    return float(loss)


def __test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def run(data, hidden_channels, out_channels=10, epochs=300, show_progress=True):
    # set the seed
    torch.manual_seed(42)

    num_features = data.num_features

    # model
    model = GAE(GCNEncoder(num_features, hidden_channels, out_channels))

    # move to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)

    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        loss = __train(model, optimizer, x, train_pos_edge_index)

        auc, ap = __test(
            model,
            x,
            train_pos_edge_index,
            data.test_pos_edge_index,
            data.test_neg_edge_index,
        )
        if show_progress:
            print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))

    return model, x, train_pos_edge_index


def _build_node_id_index(node_ids: List[int]) -> Dict[int, int]:
    return {int(node_id): idx for idx, node_id in enumerate(node_ids)}


def _temporal_smoothness_loss(
    z_curr: torch.Tensor,
    z_prev: torch.Tensor,
    node_ids_curr: List[int],
    node_ids_prev: List[int],
) -> torch.Tensor:
    idx_curr = _build_node_id_index(node_ids_curr)
    idx_prev = _build_node_id_index(node_ids_prev)

    common_nodes = set(idx_curr.keys()).intersection(idx_prev.keys())
    if not common_nodes:
        return torch.tensor(0.0, device=z_curr.device)

    curr_rows = torch.tensor(
        [idx_curr[node_id] for node_id in common_nodes],
        dtype=torch.long,
        device=z_curr.device,
    )
    prev_rows = torch.tensor(
        [idx_prev[node_id] for node_id in common_nodes],
        dtype=torch.long,
        device=z_prev.device,
    )

    return F.mse_loss(z_curr[curr_rows], z_prev[prev_rows], reduction="sum")


def run_dynamic(
    snapshots: List[Dict[str, Any]],
    hidden_channels: int = 100,
    out_channels: int = 50,
    epochs: int = 300,
    lr: float = 1e-3,
    gamma: float = 0.1,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Train a shared GAE encoder across temporal snapshots.

    Expected snapshot format:
      {
        "x": torch.Tensor [n_t, d],
        "edge_index": torch.LongTensor [2, m_t],
        "node_ids": List[int]  # global user ids for alignment
      }
    """
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one window.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = snapshots[0]["x"].shape[1]
    model = GAE(GCNEncoder(in_channels, hidden_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    proc = []
    for snapshot in snapshots:
        x = snapshot["x"].to(device)
        edge_index = snapshot["edge_index"].to(device)
        node_ids = list(snapshot["node_ids"])
        proc.append({"x": x, "edge_index": edge_index, "node_ids": node_ids})

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z_by_tau = []
        recon_loss = torch.tensor(0.0, device=device)
        for snap in proc:
            z_t = model.encode(snap["x"], snap["edge_index"])
            z_by_tau.append(z_t)
            recon_loss = recon_loss + model.recon_loss(z_t, snap["edge_index"])

        temporal_loss = torch.tensor(0.0, device=device)
        for tau in range(1, len(proc)):
            temporal_loss = temporal_loss + _temporal_smoothness_loss(
                z_curr=z_by_tau[tau],
                z_prev=z_by_tau[tau - 1],
                node_ids_curr=proc[tau]["node_ids"],
                node_ids_prev=proc[tau - 1]["node_ids"],
            )

        loss = recon_loss + gamma * temporal_loss
        loss.backward()
        optimizer.step()

        if show_progress and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(
                "Epoch: {:03d}, Loss: {:.6f}, Recon: {:.6f}, Temporal: {:.6f}".format(
                    epoch,
                    float(loss.detach().cpu()),
                    float(recon_loss.detach().cpu()),
                    float(temporal_loss.detach().cpu()),
                )
            )

    model.eval()
    with torch.no_grad():
        embeddings = []
        for snap in proc:
            z_t = model.encode(snap["x"], snap["edge_index"])
            embeddings.append(z_t.detach().cpu().numpy())

    return {
        "model": model,
        "embeddings": embeddings,
    }
