#!/usr/bin/env python3
"""Download the public MNIST files required for checkpoint inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo / ".data")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    # Both files are downloaded so the repository-local MNIST directory is a
    # complete standard copy; all supplied inference paths use the test split.
    torchvision_root = args.root / "mnist_raw"
    datasets.MNIST(root=str(torchvision_root), train=True, download=True)
    datasets.MNIST(root=str(torchvision_root), train=False, download=True)
    raw = torchvision_root / "MNIST" / "raw"
    required = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    missing = [name for name in required if not (raw / name).is_file()]
    if missing:
        raise RuntimeError(f"MNIST download incomplete: {missing}")
    print(f"MNIST ready: {raw}")


if __name__ == "__main__":
    main()
