"""
Contrastive loss functions for geographic image representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from datasets import CoordinateNormalizer


def compute_haversine_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Haversine distance matrix for a batch of coordinates.

    Args:
        coords: [N, 2] tensor with [lon, lat] in degrees.

    Returns:
        [N, N] tensor of distances in kilometers.
    """
    normalizer = CoordinateNormalizer()
    N = coords.shape[0]

    # Expand for pairwise computation
    coords_i = coords.unsqueeze(1).expand(-1, N, -1)  # [N, N, 2]
    coords_j = coords.unsqueeze(0).expand(N, -1, -1)   # [N, N, 2]

    # Flatten to [N*N, 2] for CoordinateNormalizer
    flat_i = coords_i.reshape(-1, 2)
    flat_j = coords_j.reshape(-1, 2)

    distances = normalizer.compute_haversine_distance(flat_i, flat_j)
    return distances.reshape(N, N)


def build_geographic_labels(
    coords: torch.Tensor,
    pos_threshold_km: float = 500.0,
    neg_threshold_km: float = 5000.0,
) -> torch.Tensor:
    """Build a label mask from pairwise Haversine distances.

    Each sample gets a label equal to its own index (so the diagonal is
    always positive). Additionally, any sample within pos_threshold_km
    gets the same label, so they are treated as positives in SupConLoss.

    Returns:
        labels: [N] tensor where same value = positive pair.
    """
    dist_matrix = compute_haversine_matrix(coords)
    N = dist_matrix.shape[0]

    # Start with each sample as its own class
    labels = torch.arange(N, device=coords.device, dtype=torch.long)

    # For each pair within the positive threshold, merge their labels
    # to the minimum index (simple connected-components via union)
    pos_mask = dist_matrix < pos_threshold_km

    # Iteratively propagate labels: samples within threshold share the
    # label of the lowest-indexed sample in their positive set.
    for i in range(N):
        positives = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(positives) > 0:
            min_label = labels[positives].min()
            labels[positives] = min_label

    return labels


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Uses geographic proximity labels to pull positives together and push
    negatives apart in a normalized embedding space.

    Args:
        temperature: Softmax temperature scaling (default: 0.07).
        base_temperature: Base temperature for loss scaling (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            features: [N, D] L2-normalized feature vectors.
            labels: [N] integer labels where same value = same class.
            mask: [N, N] optional boolean mask (True = valid pair).

        Returns:
            Scalar loss.
        """
        device = features.device
        N = features.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Build positive mask: samples with same label AND not self
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        # If a sample has no positives, skip it by zeroing its loss later
        has_positive = pos_mask.sum(dim=1) > 0

        # Compute logits: similarity / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        # Numerical stability: subtract max per row
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Denominator: sum over all except self
        exp_logits = torch.exp(logits)
        # Zero out self-contribution
        logits_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
        exp_logits = exp_logits * logits_mask
        denom = exp_logits.sum(dim=1, keepdim=True)

        # Log-probabilities
        log_prob = logits - torch.log(denom + 1e-8)

        # Mean over positives per anchor
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
            pos_mask.sum(dim=1) + 1e-8
        )

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[has_positive].mean()

        if torch.isnan(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


def compute_embedding_distances(
    embeddings: torch.Tensor,
    coords: torch.Tensor,
    pos_threshold_km: float = 500.0,
    neg_threshold_km: float = 5000.0,
):
    """Compute mean embedding distances for positive and negative geographic pairs.

    Useful for TensorBoard monitoring during pre-training.

    Returns:
        (mean_pos_dist, mean_neg_dist, ratio) — L2 distances between embeddings.
    """
    N = embeddings.shape[0]
    if N < 2:
        return 0.0, 0.0, 0.0

    dist_matrix = compute_haversine_matrix(coords)
    emb_distances = torch.cdist(embeddings, embeddings, p=2)

    pos_mask = (dist_matrix < pos_threshold_km).fill_diagonal_(False)
    neg_mask = (dist_matrix > neg_threshold_km)

    pos_dist = emb_distances[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_dist = emb_distances[neg_mask].mean().item() if neg_mask.any() else 0.0
    ratio = neg_dist / (pos_dist + 1e-8)

    return pos_dist, neg_dist, ratio
