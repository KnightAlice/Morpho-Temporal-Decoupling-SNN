#!/usr/bin/env python3
"""Strict-load all four checkpoints and execute one saved stimulus per model."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from patent_snn.checkpoint import load_inference_bundle  # noqa: E402


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    replay = np.load(REPO / "data" / "samples" / "paired_replay_120.npz")
    # One saved standard-condition video is sufficient to exercise every layer
    # without downloading MNIST; quantitative replay uses all 120 digits.
    video = torch.from_numpy(replay["example_videos"][0]).unsqueeze(0).unsqueeze(2)
    video = video.to(device)
    for profile in ("bio", "equal", "reverse", "fixed"):
        checkpoint, model, probe = load_inference_bundle(
            REPO / "checkpoints" / profile / "best.pt", device
        )
        with torch.inference_mode():
            representation, spike_rate = model.forward_spike_repr(
                video, int(checkpoint["last_k_steps"])
            )
            prediction = int(probe(representation).argmax(dim=1).item())
        if tuple(representation.shape) != (1, 128):
            raise AssertionError(f"Unexpected representation shape for {profile}")
        print(
            f"{profile}: strict-load PASS, prediction={prediction}, "
            f"mean_spike_rate={float(spike_rate):.6f}"
        )


if __name__ == "__main__":
    main()
