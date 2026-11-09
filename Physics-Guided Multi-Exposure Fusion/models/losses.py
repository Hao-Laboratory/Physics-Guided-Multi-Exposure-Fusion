"""Training losses for nine-channel HDR measurement fusion."""

import math

import torch
import torch.nn as nn


def range_compressor(image: torch.Tensor, mu: float = 5000.0) -> torch.Tensor:
    return torch.log1p(mu * image) / math.log1p(mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu: float = 5000.0) -> None:
        super().__init__()
        self.mu = float(mu)
        self.l1 = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(
            range_compressor(prediction, self.mu),
            range_compressor(target, self.mu),
        )


class MeasurementSAMLoss(nn.Module):
    """Mean angle (radians) between per-pixel nine-dimensional vectors."""

    def __init__(self, eps: float = 1e-8, cosine_margin: float = 1e-7) -> None:
        super().__init__()
        self.eps = float(eps)
        self.cosine_margin = float(cosine_margin)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes")
        if prediction.ndim != 4 or prediction.shape[1] != 9:
            raise ValueError(f"expected [B,9,H,W], got {tuple(prediction.shape)}")
        pred = prediction.float().permute(0, 2, 3, 1).reshape(-1, 9)
        truth = target.float().permute(0, 2, 3, 1).reshape(-1, 9)
        numerator = torch.sum(pred * truth, dim=1)
        denominator = torch.linalg.vector_norm(
            pred, dim=1
        ) * torch.linalg.vector_norm(truth, dim=1)
        cosine = (numerator / denominator.clamp_min(self.eps)).clamp(
            -1.0 + self.cosine_margin, 1.0 - self.cosine_margin
        )
        return torch.acos(cosine).mean()


class ReconstructionLoss(nn.Module):
    """Mu-law L1 plus weighted measurement-domain SAM."""

    def __init__(
        self, mu: float = 5000.0, sam_weight: float = 0.05, eps: float = 1e-8
    ) -> None:
        super().__init__()
        if sam_weight < 0:
            raise ValueError("sam_weight must be non-negative")
        self.mu = float(mu)
        self.sam_weight = float(sam_weight)
        self.l1mu = L1MuLoss(mu=self.mu)
        self.sam = MeasurementSAMLoss(eps=eps)

    def components(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1mu = self.l1mu(prediction, target)
        sam = self.sam(prediction, target)
        return l1mu + self.sam_weight * sam, l1mu, sam

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(prediction, target)[0]
