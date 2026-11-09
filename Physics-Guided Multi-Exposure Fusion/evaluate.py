"""Fuse full-resolution multi-exposure measurements into linear HDR output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from datasets import ExposureProtocol
from models import build_model
from utils import load_checkpoint, measurement_metrics, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "pretrained" / "model_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "fusion"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Scene name. Repeat this option to select multiple scenes; omit for all scenes.",
    )
    parser.add_argument(
        "--combination",
        default=None,
        help='Comma-separated EV values, for example "-2,0,2".',
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-context", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_combination(value: str | None, protocol: ExposureProtocol) -> tuple[float, ...]:
    if value is None:
        values = protocol.exposure_config["main_test_combination"]
    else:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    evs = tuple(sorted(float(item) for item in values))
    if not evs:
        raise ValueError("the exposure combination cannot be empty")
    missing = [ev for ev in evs if ev not in protocol.exposure_by_ev]
    if missing:
        raise ValueError(f"no input filename is configured for EV values {missing}")
    return evs


def load_scene_inputs(
    scene_dir: Path, protocol: ExposureProtocol, evs: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray | None]:
    raw_arrays = []
    shape = None
    for ev in evs:
        exposure = protocol.exposure_by_ev[ev]
        path = scene_dir / exposure["filename"]
        raw = np.asarray(np.load(path), dtype=np.float32)
        if raw.ndim != 3 or raw.shape[-1] != 9:
            raise ValueError(f"expected HxWx9 data at {path}, got {raw.shape}")
        if shape is not None and raw.shape != shape:
            raise ValueError(f"exposure shape mismatch in {scene_dir}")
        shape = raw.shape
        raw_arrays.append(raw / protocol.max_dn)

    raw = np.stack(raw_arrays, axis=0)
    ratios = np.asarray([2.0**ev for ev in evs], dtype=np.float32)[:, None, None, None]
    normalized = raw / ratios / protocol.hdr_scale
    raw_tensor = torch.from_numpy(raw.transpose(0, 3, 1, 2)).unsqueeze(0)
    normalized_tensor = torch.from_numpy(
        normalized.transpose(0, 3, 1, 2)
    ).unsqueeze(0)
    ev_tensor = torch.tensor(evs, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones((1, len(evs)), dtype=torch.bool)

    gt_path = scene_dir / protocol.gt_filename
    ground_truth = None
    if gt_path.is_file():
        ground_truth = np.asarray(np.load(gt_path), dtype=np.float32)
    return raw_tensor, normalized_tensor, ev_tensor, mask, ground_truth


def predict_with_context_tiles(
    model,
    raw: torch.Tensor,
    normalized: torch.Tensor,
    ev: torch.Tensor,
    mask: torch.Tensor,
    tile_size: int,
    context: int,
) -> torch.Tensor:
    if tile_size <= 0 or context < 0:
        raise ValueError("tile_size must be positive and tile_context non-negative")
    _, _, channels, height, width = raw.shape
    prediction = torch.empty(
        (1, channels, height, width), device=raw.device, dtype=raw.dtype
    )
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            outer_top = max(0, top - context)
            outer_left = max(0, left - context)
            outer_bottom = min(height, bottom + context)
            outer_right = min(width, right + context)
            tile = model(
                raw[:, :, :, outer_top:outer_bottom, outer_left:outer_right],
                normalized[:, :, :, outer_top:outer_bottom, outer_left:outer_right],
                ev,
                mask,
            )
            crop_top = top - outer_top
            crop_left = left - outer_left
            prediction[:, :, top:bottom, left:right] = tile[
                :,
                :,
                crop_top : crop_top + bottom - top,
                crop_left : crop_left + right - left,
            ]
    return prediction


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    protocol = ExposureProtocol(args.config, data_root=args.data_root)
    evs = parse_combination(args.combination, protocol)
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_config = checkpoint.get("model_config", protocol.config["model"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    hdr_scale = float(checkpoint.get("hdr_scale", protocol.hdr_scale))

    split_dir = protocol.config["data"]["splits"][args.split]["directory"]
    split_root = protocol.data_root / split_dir
    available = protocol.split_scene_names(args.split)
    scenes = args.scene or available
    unknown = sorted(set(scenes) - set(available))
    if unknown:
        raise ValueError(f"unknown {args.split} scenes: {unknown}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Device: {device}")
    print(f"Exposure combination: {list(evs)}")
    with torch.no_grad():
        for scene in scenes:
            scene_output = output_root / scene
            scene_output.mkdir(parents=True, exist_ok=True)
            output_path = scene_output / "fused_hdr_9ch.npy"
            if args.skip_existing and output_path.is_file():
                fused = np.asarray(np.load(output_path), dtype=np.float32)
                ground_truth_path = split_root / scene / protocol.gt_filename
                ground_truth = (
                    np.asarray(np.load(ground_truth_path), dtype=np.float32)
                    if ground_truth_path.is_file()
                    else None
                )
            else:
                raw, normalized, ev, mask, ground_truth = load_scene_inputs(
                    split_root / scene, protocol, evs
                )
                prediction = predict_with_context_tiles(
                    model,
                    raw.to(device),
                    normalized.to(device),
                    ev.to(device),
                    mask.to(device),
                    args.tile_size,
                    args.tile_context,
                )
                fused = (
                    prediction.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    * hdr_scale
                ).astype(np.float32, copy=False)
                atomic_save_npy(output_path, fused)

            row = {
                "scene": scene,
                "output": str(output_path),
                "height": int(fused.shape[0]),
                "width": int(fused.shape[1]),
                "channels": int(fused.shape[2]),
                "minimum": float(fused.min()),
                "maximum": float(fused.max()),
                "rmse_linear": None,
                "sam_rad": None,
            }
            if ground_truth is not None:
                pred_tensor = torch.from_numpy(fused.transpose(2, 0, 1)).unsqueeze(0)
                target_tensor = torch.from_numpy(
                    ground_truth.transpose(2, 0, 1)
                ).unsqueeze(0)
                metrics = measurement_metrics(pred_tensor, target_tensor)
                row.update(
                    rmse_linear=float(metrics["rmse"].item()),
                    sam_rad=float(metrics["sam_rad"].item()),
                )
            rows.append(row)
            print(f"{scene}: saved {fused.shape} to {output_path}")

    with open(output_root / "fusion_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_root / "fusion_summary.json",
        {
            "project": "Physics-Guided Multi-Exposure Fusion",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_root": str(protocol.data_root),
            "split": args.split,
            "exposure_combination": list(evs),
            "hdr_scale": hdr_scale,
            "output_scale": "linear HDR measurement",
            "tile_size": args.tile_size,
            "tile_context": args.tile_context,
            "scenes": rows,
        },
    )


if __name__ == "__main__":
    main()
