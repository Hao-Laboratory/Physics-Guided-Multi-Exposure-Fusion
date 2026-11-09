"""Train Physics-Guided Multi-Exposure Fusion on ICVL measurements."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from datasets import (
    ExposureProtocol,
    ScheduledEpochSampler,
    ScheduledTrainingDataset,
    ValidationDataset,
)
from models import ReconstructionLoss, build_model
from utils import (
    EqualCombinationAccumulator,
    adjust_learning_rate,
    count_parameters,
    create_run_directory,
    load_checkpoint,
    measurement_metrics,
    set_random_seed,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_batch(model, batch: dict, device: torch.device):
    target = batch["label"].to(device, non_blocking=True)
    prediction = model(
        batch["raw"].to(device, non_blocking=True),
        batch["normalized"].to(device, non_blocking=True),
        batch["ev"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
    )
    return prediction, target


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled: bool,
    grad_clip: float,
    epoch: int,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "l1mu": 0.0,
        "sam_rad": 0.0,
        "weighted_sam": 0.0,
        "grad_norm": 0.0,
    }
    max_grad_norm = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            prediction, target = forward_batch(model, batch, device)
            loss, l1mu, sam = criterion.components(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {batch_index}")

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        batch_size = int(target.shape[0])
        grad_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
        totals["loss"] += float(loss.item()) * batch_size
        totals["l1mu"] += float(l1mu.item()) * batch_size
        totals["sam_rad"] += float(sam.item()) * batch_size
        totals["weighted_sam"] += float(criterion.sam_weight * sam.item()) * batch_size
        totals["grad_norm"] += grad_value * batch_size
        max_grad_norm = max(max_grad_norm, grad_value)
        sample_count += batch_size
        if batch_index % log_interval == 0 or batch_index == len(loader):
            print(
                f"epoch {epoch:03d} batch {batch_index:04d}/{len(loader):04d} "
                f"loss {loss.item():.6f} l1mu {l1mu.item():.6f} "
                f"sam {sam.item():.6f} grad_norm {grad_value:.6f}"
            )
    return {
        "loss": totals["loss"] / sample_count,
        "l1mu": totals["l1mu"] / sample_count,
        "sam_rad": totals["sam_rad"] / sample_count,
        "weighted_sam": totals["weighted_sam"] / sample_count,
        "grad_norm_mean": totals["grad_norm"] / sample_count,
        "grad_norm_max": max_grad_norm,
    }


def validate(model, loader, criterion, device, amp_enabled, combinations) -> dict:
    model.eval()
    accumulator = EqualCombinationAccumulator(combinations)
    with torch.no_grad():
        for batch in loader:
            with autocast(device_type=device.type, enabled=amp_enabled):
                prediction, target = forward_batch(model, batch, device)
                loss, l1mu, sam = criterion.components(prediction, target)
            metrics = measurement_metrics(prediction.float(), target.float())
            accumulator.update(
                int(batch["combo_id"].item()),
                loss=float(loss.item()),
                l1mu=float(l1mu.item()),
                sam_rad=float(sam.item()),
                weighted_sam=float(criterion.sam_weight * sam.item()),
                rmse=float(metrics["rmse"].item()),
            )
    return accumulator.finalize()


def write_metrics(path: Path, rows: list[dict]) -> None:
    evaluated_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("val_loss", math.nan)))
    ]
    if not evaluated_rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated_rows[0]))
        writer.writeheader()
        writer.writerows(evaluated_rows)


def write_curve(path: Path, rows: list[dict]) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, [row["train_loss"] for row in rows], label="train")
    axis.plot(epochs, [row["val_loss"] for row in rows], label="validation")
    axis.set(xlabel="Epoch", ylabel="Reconstruction loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    protocol = ExposureProtocol(args.config, data_root=args.data_root)
    config = protocol.config
    training = config["training"]
    optimization = config["optimization"]
    loss_config = config["loss"]

    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    epochs = int(args.epochs or training["epochs"])
    schedule_epochs = int(protocol.schedule["epochs"][0])
    if not 1 <= epochs <= schedule_epochs:
        raise ValueError(f"epochs must be in 1..{schedule_epochs}")
    batch_size = int(args.batch_size or training["batch_size"])
    num_workers = int(
        training["num_workers"] if args.num_workers is None else args.num_workers
    )
    device = select_device(args.device)
    amp_requested = training["amp"] if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")
    set_random_seed(int(training["seed"]))

    output_root = Path(args.output_root or config["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = create_run_directory(output_root, args.run_name)

    train_dataset = ScheduledTrainingDataset(protocol)
    train_sampler = ScheduledEpochSampler(train_dataset, limit=args.limit_train)
    val_dataset = ValidationDataset(protocol, limit_scenes=args.limit_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config["model"]).to(device)
    criterion = ReconstructionLoss(**loss_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimization["initial_lr"]),
        betas=tuple(optimization["betas"]),
        eps=float(optimization["eps"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = None
    rows = []
    if args.init_checkpoint:
        checkpoint = load_checkpoint(args.init_checkpoint, device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = checkpoint.get("best_epoch")
        rows = list(checkpoint.get("metrics", []))

    print(f"Project: Physics-Guided Multi-Exposure Fusion")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data root: {protocol.data_root}")
    print(f"Run directory: {run_dir}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Training samples per epoch: {len(train_sampler)}")
    print(f"Validation samples: {len(val_dataset)}")

    write_json(
        run_dir / "resolved_config.json",
        {
            "project": "Physics-Guided Multi-Exposure Fusion",
            "training_config": config,
            "data_root": str(protocol.data_root),
            "epochs_this_run": epochs,
            "parameter_count": count_parameters(model),
            "device": str(device),
            "amp_enabled": amp_enabled,
        },
    )

    validation_interval = int(training["validation_interval"])
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        learning_rate = adjust_learning_rate(
            optimizer,
            float(optimization["initial_lr"]),
            epoch - 1,
            int(optimization["lr_decay_interval"]),
        )
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            float(optimization["gradient_clip_norm"]),
            epoch,
            args.log_interval,
        )
        should_validate = epoch % validation_interval == 0 or epoch == epochs
        validation = None
        if should_validate:
            validation = validate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled,
                protocol.validation_combinations,
            )
            validation_mean = validation["mean"]
        else:
            validation_mean = {
                name: math.nan
                for name in ("loss", "l1mu", "sam_rad", "weighted_sam", "rmse")
            }

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_l1mu": train_metrics["l1mu"],
            "train_sam_rad": train_metrics["sam_rad"],
            "train_weighted_sam": train_metrics["weighted_sam"],
            "train_grad_norm_mean": train_metrics["grad_norm_mean"],
            "train_grad_norm_max": train_metrics["grad_norm_max"],
            "val_loss": validation_mean["loss"],
            "val_l1mu": validation_mean["l1mu"],
            "val_sam_rad": validation_mean["sam_rad"],
            "val_weighted_sam": validation_mean["weighted_sam"],
            "val_rmse": validation_mean["rmse"],
            "lr": learning_rate,
        }
        rows.append(row)
        print(
            f"epoch {epoch:03d} train {train_metrics['loss']:.6f} "
            f"val {validation_mean['loss']:.6f} lr {learning_rate:.8f}"
        )

        improved = validation is not None and validation_mean["loss"] < best_val_loss
        if improved:
            best_val_loss = validation_mean["loss"]
            best_epoch = epoch
        checkpoint = {
            "format_version": 1,
            "project": "Physics-Guided Multi-Exposure Fusion",
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": rows,
            "validation": validation,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "hdr_scale": protocol.hdr_scale,
            "model_config": config["model"],
            "loss_config": loss_config,
        }
        torch.save(checkpoint, run_dir / "latest_checkpoint.pth")
        if improved:
            torch.save(checkpoint, run_dir / "best_checkpoint.pth")
        write_metrics(run_dir / "metrics.csv", rows)
        write_curve(run_dir / "training_curve.png", rows)

    print(f"Best validation loss: {best_val_loss:.9f} at epoch {best_epoch}")
    print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    main()
