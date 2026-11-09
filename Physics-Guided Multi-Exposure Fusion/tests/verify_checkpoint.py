"""Check that the released checkpoint loads strictly and produces finite output."""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model


def main() -> None:
    checkpoint = torch.load(
        PROJECT_ROOT / "pretrained" / "model_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    raw = torch.rand(1, 3, 9, 32, 32)
    ev = torch.tensor([[-2.0, 0.0, 2.0]])
    normalized = raw / (2.0**ev[:, :, None, None, None]) / checkpoint["hdr_scale"]
    with torch.no_grad():
        output = model(raw, normalized, ev)
    if output.shape != (1, 9, 32, 32) or not torch.isfinite(output).all():
        raise RuntimeError("released checkpoint verification failed")
    print("Checkpoint verification passed.")


if __name__ == "__main__":
    main()
