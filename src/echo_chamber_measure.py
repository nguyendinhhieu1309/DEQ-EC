from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np
from typing import List, Dict, Optional, Any


class EchoChamberMeasure:
    def __init__(
        self,
        users_representations: np.ndarray,
        labels: np.ndarray,
        metric: str = "euclidean",
    ):
        if metric == "euclidean":
            self.distances = euclidean_distances(users_representations)
        elif metric == "cosine":
            self.distances = cosine_distances(users_representations)
        self.labels = labels

    def cohesion_node(self, idx: int) -> float:
        node_label = self.labels[idx]

        node_distances = self.distances[idx, self.labels == node_label]

        return np.mean(node_distances)

    def separation_node(self, idx: int) -> float:
        node_label = self.labels[idx]

        dist = []
        for l in np.unique(self.labels):
            if l == node_label:
                continue
            dist.append(np.mean(self.distances[idx, self.labels == l]))

        return np.min(dist)

    def metric(self, idx: int) -> float:
        a = self.cohesion_node(idx)
        b = self.separation_node(idx)

        return (-a + b + max(a, b)) / (2 * max(a, b))

    def echo_chamber_index(self) -> float:
        nodes_metric = []
        for i in range(self.distances.shape[0]):
            nodes_metric.append(self.metric(i))
        return np.mean(nodes_metric)

    def community_echo_chamber_index(self, community_label: int) -> float:
        com_eci = []

        for i in range(self.distances.shape[0]):
            if self.labels[i] == community_label:
                com_eci.append(self.metric(i))

        return np.mean(com_eci)


class DEQECMeasure:
    """
    Dynamic Exposure-Aware Echo Chamber quantification.
    Implements Eq. (4) - Eq. (10) in the DEQ-EC pipeline.
    """

    def __init__(self, metric: str = "euclidean"):
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be either 'euclidean' or 'cosine'.")
        self.metric = metric

    def _pairwise_distances(self, users_representations: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return euclidean_distances(users_representations)
        return cosine_distances(users_representations)

    def _cohesion_node(self, idx: int, labels: np.ndarray, distances: np.ndarray) -> float:
        node_label = labels[idx]
        node_distances = distances[idx, labels == node_label]
        return float(np.mean(node_distances))

    def _separation_node(
        self,
        idx: int,
        labels: np.ndarray,
        distances: np.ndarray,
    ) -> float:
        node_label = labels[idx]
        dist = []
        for l in np.unique(labels):
            if l == node_label:
                continue
            dist.append(np.mean(distances[idx, labels == l]))
        return float(np.min(dist)) if len(dist) > 0 else 0.0

    def _exposure_ratio_node(
        self,
        idx: int,
        labels: np.ndarray,
        edges: np.ndarray,
    ) -> float:
        # edges: shape [2, m] with directed interactions u -> v in local node indices.
        u = idx
        out_mask = edges[0] == u
        neighbors = edges[1][out_mask]
        if neighbors.size == 0:
            return 0.0
        cross = np.sum(labels[neighbors] != labels[u])
        return float(cross / neighbors.size)

    def snapshot_score(
        self,
        users_representations: np.ndarray,
        labels: np.ndarray,
        edges: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute DEQ-EC^(tau) for one temporal snapshot.
        """
        distances = self._pairwise_distances(users_representations)

        s_values = []
        s_tilde_values = []
        exposure_values = []
        cohesion_values = []
        separation_values = []

        for i in range(users_representations.shape[0]):
            c_i = self._cohesion_node(i, labels, distances)
            d_i = self._separation_node(i, labels, distances)

            denom = max(c_i, d_i)
            s_i = 0.0 if denom == 0 else (d_i - c_i) / denom

            phi_i = self._exposure_ratio_node(i, labels, edges)
            s_tilde_i = s_i * (1.0 - phi_i)

            cohesion_values.append(c_i)
            separation_values.append(d_i)
            s_values.append(s_i)
            exposure_values.append(phi_i)
            s_tilde_values.append(s_tilde_i)

        return {
            "deq_ec_tau": float(np.mean(s_tilde_values)),
            "s": np.array(s_values),
            "s_tilde": np.array(s_tilde_values),
            "cohesion": np.array(cohesion_values),
            "separation": np.array(separation_values),
            "exposure_ratio": np.array(exposure_values),
        }

    def dynamic_score(
        self,
        temporal_embeddings: List[np.ndarray],
        temporal_labels: List[np.ndarray],
        temporal_edges: List[np.ndarray],
    ) -> Dict[str, Any]:
        if not (
            len(temporal_embeddings)
            == len(temporal_labels)
            == len(temporal_edges)
        ):
            raise ValueError("All temporal inputs must have the same length.")

        out_per_tau = []
        for emb, labels, edges in zip(
            temporal_embeddings, temporal_labels, temporal_edges
        ):
            out_per_tau.append(self.snapshot_score(emb, labels, edges))

        deq_ec_t = np.array([x["deq_ec_tau"] for x in out_per_tau], dtype=float)
        return {
            "deq_ec_t": deq_ec_t,
            "deq_ec": float(np.mean(deq_ec_t)),
            "per_snapshot": out_per_tau,
        }
