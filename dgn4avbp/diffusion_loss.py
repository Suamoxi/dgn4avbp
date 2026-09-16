from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from .dgn_model import DiffusionModel
from .improved_ddpm import hybrid_loss_terms


class CanonicalHybridLoss(nn.Module):
    """Improved-DDPM hybrid objective with the paper's full-VLB scaling."""

    def __init__(self, lambda_vlb: float = 0.001) -> None:
        super().__init__()
        if lambda_vlb < 0.0:
            raise ValueError(f"lambda_vlb must be non-negative, got {lambda_vlb}.")
        self.lambda_vlb = float(lambda_vlb)

    def forward(self, model: DiffusionModel, graph: Data) -> torch.Tensor:
        if not model.learnable_variance:
            raise ValueError("CanonicalHybridLoss requires learnable_variance=True.")
        model_epsilon, model_v = model(graph)
        terms = hybrid_loss_terms(
            model.diffusion_process,
            field_start=graph.field_start,
            field_r=graph.field_r,
            noise=graph.noise,
            model_epsilon=model_epsilon,
            model_v=model_v,
            batch=graph.batch,
            r=graph.r,
            lambda_vlb=self.lambda_vlb,
        )
        return terms["loss"]
