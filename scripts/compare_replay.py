#!/usr/bin/env python3
"""Compare a regenerated 120-sample replay with the bundled reference."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "generated",
        type=Path,
        nargs="?",
        default=repo / "outputs" / "replay" / "paired_replay_120.npz",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=repo / "data" / "samples" / "paired_replay_120.npz",
    )
    args = parser.parse_args()
    reference = np.load(args.reference)
    generated = np.load(args.generated)
    if reference.files != generated.files:
        raise AssertionError("Replay array keys differ")

    exact_keys = ("conditions", "labels", "digit_indices", "predictions", "example_videos")
    for key in exact_keys:
        if not np.array_equal(reference[key], generated[key]):
            raise AssertionError(f"Exact replay mismatch: {key}")
        print(f"{key}: exact")

    # CUDA convolution order can move a borderline binary spike.  One top-layer
    # spike changes a last-four-step 8x8 pooled feature by exactly 1/(4*64).
    tolerances = {"features": 1.0 / 256.0, "response_mean": 1e-5}
    for key, tolerance in tolerances.items():
        maximum = float(
            np.max(
                np.abs(
                    reference[key].astype(np.float64)
                    - generated[key].astype(np.float64)
                )
            )
        )
        if maximum > tolerance:
            raise AssertionError(
                f"Replay mismatch for {key}: {maximum} > tolerance {tolerance}"
            )
        print(f"{key}: max_abs={maximum:.8g} <= {tolerance:.8g}")
    print("Replay comparison PASS")


if __name__ == "__main__":
    main()
