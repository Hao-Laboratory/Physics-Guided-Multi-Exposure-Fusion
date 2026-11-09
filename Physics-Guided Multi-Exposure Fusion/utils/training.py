"""Small training and evaluation utilities."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def measurement_metrics(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    difference = prediction - target
    rmse = torch.sqrt(torch.mean(difference.square()))
    pred = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.shape[1])
    truth = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    numerator = torch.sum(pred * truth, dim=1)
    denominator = torch.linalg.vector_norm(pred, dim=1) * torch.linalg.vector_norm(
        truth, dim=1
    )
    cosine = (numerator / denominator.clamp_min(eps)).clamp(
        -1.0 + 1e-7, 1.0 - 1e-7
    )
    return {"rmse": rmse, "sam_rad": torch.acos(cosine).mean()}


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    initial_lr: float,
    epoch_index: int,
    decay_interval: int,
) -> float:
    learning_rate = float(initial_lr) * (0.5 ** (int(epoch_index) // int(decay_interval)))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def create_run_directory(output_root: str | Path, run_name: str | None = None) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / name
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


class EqualCombinationAccumulator:
    """Average scenes within a combination, then average combinations equally."""

    def __init__(self, combinations: list[tuple[float, ...]]) -> None:
        self.combinations = combinations
        self.metric_names = (
            "loss",
            "l1mu",
            "sam_rad",
            "weighted_sam",
            "rmse",
        )
        self.sums = {
            name: [0.0] * len(combinations) for name in self.metric_names
        }
        self.counts = [0] * len(combinations)

    def update(self, combo_id: int, **metrics: float) -> None:
        if set(metrics) != set(self.metric_names):
            raise KeyError("validation metric set does not match")
        for name, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite validation metric: {name}")
            self.sums[name][combo_id] += value
        self.counts[combo_id] += 1

    def finalize(self) -> dict:
        if any(count == 0 for count in self.counts):
            raise RuntimeError("one or more validation combinations have no samples")
        per_combination = []
        for combo_id, evs in enumerate(self.combinations):
            row = {
                "combo_id": combo_id,
                "evs": list(evs),
                "count": self.counts[combo_id],
            }
            for name in self.metric_names:
                row[name] = self.sums[name][combo_id] / self.counts[combo_id]
            per_combination.append(row)
        mean = {
            name: sum(row[name] for row in per_combination) / len(per_combination)
            for name in self.metric_names
        }
        return {"mean": mean, "per_combination": per_combination}


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint:
        return {"state_dict": checkpoint}
    return checkpoint


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, content: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(content), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
