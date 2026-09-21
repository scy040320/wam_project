"""A small dependency-aware counterfactual attribution network.

The module intentionally has no dependency on Cosmos internals.  Visual features
are precomputed by a frozen DINOv2-S encoder; this model receives only vectors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TypedGraphAttention(nn.Module):
    def __init__(self, width: int, edge_type_count: int):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.edge_bias = nn.Embedding(edge_type_count, 1)
        self.relation_value = nn.Embedding(edge_type_count, width)
        self.output = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.LayerNorm(width))

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_type: Tensor, edge_mask: Tensor | None = None) -> Tensor:
        # nodes [batch, nodes, width]; edge_index [2, edges]
        batch, count, width = nodes.shape
        query, key, value = self.query(nodes), self.key(nodes), self.value(nodes)
        aggregate = torch.zeros_like(nodes)
        normalizer = torch.zeros((batch, count, 1), device=nodes.device, dtype=nodes.dtype)
        for index in range(edge_index.shape[1]):
            source, target = edge_index[:, index]
            score = (query[:, target] * key[:, source]).sum(-1, keepdim=True) / math.sqrt(width)
            score = score + self.edge_bias(edge_type[index])
            weight = score.sigmoid()
            if edge_mask is not None:
                weight = weight * edge_mask[:, index].reshape(batch, 1)
            message = value[:, source] + self.relation_value(edge_type[index])
            aggregate[:, target] += weight * message
            normalizer[:, target] += weight
        aggregate = aggregate / normalizer.clamp_min(1.0)
        return self.output(nodes + aggregate)


class CounterfactualAttributor(nn.Module):
    """Two-layer GAT with cause, affected-node and hypothesis-decoder heads."""

    def __init__(self, residual_dim: int, node_feature_dim: int, edge_type_count: int, width: int = 128, cause_count: int = 5):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(residual_dim, width), nn.ReLU(), nn.LayerNorm(width))
        self.node_input = nn.Sequential(nn.Linear(node_feature_dim + width, width), nn.ReLU(), nn.LayerNorm(width))
        self.gat1 = TypedGraphAttention(width, edge_type_count)
        self.gat2 = TypedGraphAttention(width, edge_type_count)
        self.cause_head = nn.Linear(width, cause_count)
        self.mask_head = nn.Linear(width, 1)
        self.hypothesis_head = nn.Linear(width, cause_count * residual_dim)
        self.cause_count = cause_count
        self.residual_dim = residual_dim

    def forward(
        self, residual: Tensor, node_features: Tensor, edge_index: Tensor, edge_type: Tensor,
        edge_mask: Tensor | None = None, node_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded_residual = self.residual(residual)
        broadcast = encoded_residual.unsqueeze(1).expand(-1, node_features.shape[1], -1)
        nodes = self.node_input(torch.cat((node_features, broadcast), dim=-1))
        nodes = self.gat1(nodes, edge_index, edge_type, edge_mask)
        nodes = self.gat2(nodes, edge_index, edge_type, edge_mask)
        if node_valid is None:
            pooled = nodes.mean(dim=1)
        else:
            valid = node_valid.to(nodes.dtype).unsqueeze(-1)
            pooled = (nodes * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return {
            "cause_logits": self.cause_head(pooled),
            "mask_logits": self.mask_head(nodes).squeeze(-1),
            "hypothesis_residuals": self.hypothesis_head(pooled).view(-1, self.cause_count, self.residual_dim),
        }
