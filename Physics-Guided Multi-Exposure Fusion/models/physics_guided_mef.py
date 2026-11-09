"""Physics-guided multi-exposure fusion network for nine-channel measurements."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockNoBN(nn.Module):
    """Two-convolution residual block without batch normalization."""

    def __init__(self, channels: int, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.body(x)


class ResidualStage(nn.Sequential):
    def __init__(
        self, channels: int, num_blocks: int, residual_scale: float = 0.1
    ) -> None:
        super().__init__(
            *[
                ResidualBlockNoBN(channels, residual_scale=residual_scale)
                for _ in range(int(num_blocks))
            ]
        )


class ResidualUNet(nn.Module):
    """Three-scale residual U-Net with bilinear upsampling and skip connections."""

    def __init__(
        self,
        channels: tuple[int, int, int] = (32, 64, 96),
        blocks: tuple[int, int, int] = (2, 2, 4),
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3 = (int(value) for value in channels)
        b1, b2, b3 = (int(value) for value in blocks)

        self.encoder1 = ResidualStage(c1, b1, residual_scale)
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.encoder2 = ResidualStage(c2, b2, residual_scale)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.bottleneck = ResidualStage(c3, b3, residual_scale)

        self.up2_projection = nn.Sequential(
            nn.Conv2d(c3, c2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.decoder2_fusion = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.decoder2 = ResidualStage(c2, b2, residual_scale)
        self.up1_projection = nn.Sequential(
            nn.Conv2d(c2, c1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.decoder1_fusion = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.decoder1 = ResidualStage(c1, b1, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.down1(skip1))
        deep = self.bottleneck(self.down2(skip2))

        up2 = F.interpolate(
            deep, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        up2 = self.up2_projection(up2)
        decoded2 = self.decoder2(
            self.decoder2_fusion(torch.cat((up2, skip2), dim=1))
        )
        up1 = F.interpolate(
            decoded2,
            size=skip1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up1 = self.up1_projection(up1)
        return self.decoder1(
            self.decoder1_fusion(torch.cat((up1, skip1), dim=1))
        )


class SharedReliabilityEstimator(nn.Module):
    """Estimate per-channel reliability corrections with shared exposure weights."""

    def __init__(self, num_measurements: int = 9, hidden_channels: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_measurements, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_channels, num_measurements, 3, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.body(raw)


class PhysicsGuidedMEF(nn.Module):
    """Fuse an exposure set and refine the physical estimate with a residual U-Net.

    Inputs ``raw`` and ``normalized`` have shape ``[B, E, 9, H, W]``.
    ``ev`` and ``mask`` have shape ``[B, E]``. EV values are retained in the
    public interface; exposure compensation is performed during preprocessing.
    """

    def __init__(
        self,
        num_measurements: int = 9,
        reliability_channels: int = 16,
        unet_channels: tuple[int, int, int] = (32, 64, 96),
        unet_blocks: tuple[int, int, int] = (2, 2, 4),
        residual_scale: float = 0.1,
        reliability_floor: float = 0.01,
        reliability_eps: float = 1e-6,
        output_init_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if len(unet_channels) != 3 or len(unet_blocks) != 3:
            raise ValueError("unet_channels and unet_blocks must have length three")
        if reliability_floor <= 0 or reliability_eps <= 0:
            raise ValueError("reliability_floor and reliability_eps must be positive")
        if output_init_scale <= 0:
            raise ValueError("output_init_scale must be positive")

        self.num_measurements = int(num_measurements)
        self.reliability_floor = float(reliability_floor)
        self.reliability_eps = float(reliability_eps)
        self.output_init_scale = float(output_init_scale)
        # Keep module attribute names stable so the released checkpoint is
        # directly compatible with the original trained parameters.
        self.reliability_head = SharedReliabilityEstimator(
            self.num_measurements, reliability_channels
        )
        first_channels = int(unet_channels[0])
        self.fusion_stem = nn.Sequential(
            nn.Conv2d(self.num_measurements, first_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(first_channels, first_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.resunet = ResidualUNet(
            channels=unet_channels,
            blocks=unet_blocks,
            residual_scale=residual_scale,
        )
        self.residual_head = nn.Conv2d(
            first_channels, self.num_measurements, 3, 1, 1
        )
        self._small_kaiming_init(self.residual_head)

    def _small_kaiming_init(self, layer: nn.Conv2d) -> None:
        nn.init.kaiming_normal_(
            layer.weight,
            a=0.1,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        with torch.no_grad():
            layer.weight.mul_(self.output_init_scale)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _validate_inputs(
        self,
        raw: torch.Tensor,
        normalized: torch.Tensor,
        ev: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if raw.ndim != 5:
            raise ValueError(f"raw must be [B,E,9,H,W], got {tuple(raw.shape)}")
        if normalized.shape != raw.shape:
            raise ValueError("normalized and raw must have identical shapes")
        if raw.shape[2] != self.num_measurements:
            raise ValueError(f"expected {self.num_measurements} channels")
        if ev.shape != raw.shape[:2] or mask.shape != raw.shape[:2]:
            raise ValueError("ev and mask must have shape [B,E]")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every sample must contain at least one exposure")

    def forward(
        self,
        raw: torch.Tensor,
        normalized: torch.Tensor,
        ev: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | bool]:
        if mask is None:
            mask = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
        else:
            mask = mask.to(device=raw.device, dtype=torch.bool)
        ev = ev.to(device=raw.device, dtype=raw.dtype)
        self._validate_inputs(raw, normalized, ev, mask)

        batch, exposure_count, _, height, width = raw.shape
        raw = raw.clamp(0.0, 1.0)
        hat_prior = 4.0 * raw * (1.0 - raw)
        raw_flat = raw.reshape(
            batch * exposure_count, self.num_measurements, height, width
        )
        learned_logits = self.reliability_head(raw_flat).reshape_as(raw)
        reliability_logits = (
            torch.log(hat_prior + self.reliability_floor) + learned_logits
        )

        valid = mask[:, :, None, None, None]
        reliability_logits = reliability_logits.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(reliability_logits, dim=1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            self.reliability_eps
        )
        y_fuse = torch.sum(weights * normalized, dim=1).clamp(0.0, 1.0)

        fused_features = self.fusion_stem(y_fuse)
        decoded = self.resunet(fused_features)
        residual = self.residual_head(decoded)
        prediction = torch.clamp(y_fuse + residual, 0.0, 1.0)

        if return_details:
            return {
                "prediction": prediction,
                "y_fuse": y_fuse,
                "weights": weights,
                "hat_prior": hat_prior,
                "learned_logits": learned_logits,
                "fused_features": fused_features,
                "residual": residual,
                "mask": mask,
                "ev_used_by_model": False,
                "hat_used_as_head_input": False,
                "explicit_hat_logits_enabled": True,
            }
        return prediction


DEFAULT_MODEL_CONFIG = {
    "num_measurements": 9,
    "reliability_channels": 16,
    "unet_channels": (32, 64, 96),
    "unet_blocks": (2, 2, 4),
    "residual_scale": 0.1,
    "reliability_floor": 0.01,
    "reliability_eps": 1e-6,
    "output_init_scale": 0.10,
}


def build_model(model_config: dict | None = None) -> PhysicsGuidedMEF:
    config = dict(DEFAULT_MODEL_CONFIG)
    if model_config:
        config.update(model_config)
    if isinstance(config.get("unet_channels"), list):
        config["unet_channels"] = tuple(config["unet_channels"])
    if isinstance(config.get("unet_blocks"), list):
        config["unet_blocks"] = tuple(config["unet_blocks"])
    return PhysicsGuidedMEF(**config)
