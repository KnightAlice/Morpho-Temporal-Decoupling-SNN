"""Strict checkpoint loader for the inference-only patent package."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .model import MovingMNISTInferenceSNN, PROFILE_SCALES


def load_inference_bundle(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[dict[str, Any], MovingMNISTInferenceSNN, nn.Linear]:
    """Load one recovered model and its online linear probe with strict keys."""
    path = Path(checkpoint_path)
    # weights_only=False is required because these trusted project checkpoints
    # contain their nested experiment dictionaries in addition to tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("train_mode") != "supcon_probe":
        raise ValueError(f"Unsupported checkpoint train_mode in {path}")
    if checkpoint.get("probe") is None:
        raise ValueError(f"Checkpoint has no online linear probe: {path}")
    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    profile = str(model_cfg["profile"])
    if profile not in PROFILE_SCALES:
        raise ValueError(f"Checkpoint uses an unsupported profile: {profile!r}")

    model = MovingMNISTInferenceSNN(
        profile=profile,
        time_steps=int(data_cfg["T"]),
        tau0=float(model_cfg["tau0"]),
        tau_min=float(model_cfg["tau_min"]),
        tau_max=float(model_cfg["tau_max"]),
        tau_u_variance_floor=float(model_cfg["tau_u_variance_floor"]),
    )
    # Strict loading is the architecture/version gate: expired model variants
    # cannot produce results if any required key or tensor shape differs.
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    probe = nn.Linear(model.fc.in_features, 10, bias=True)
    probe.load_state_dict(checkpoint["probe"], strict=True)
    probe.to(device).eval()
    return checkpoint, model, probe
