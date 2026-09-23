#!/usr/bin/env python3
"""Recompute the five principal accuracies from one or all supplied checkpoints."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from patent_snn.checkpoint import load_inference_bundle  # noqa: E402
from patent_snn.data import EvaluationMovingMNIST  # noqa: E402


PROFILES = ("bio", "equal", "reverse", "fixed")
SPLITS = (
    "test_id",
    "test_ood_speed",
    "test_ood_flicker",
    "test_ood_noise",
    "test_combined_ood",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILES,
        default=list(PROFILES),
        help="Checkpoint profiles to evaluate; default: all four.",
    )
    parser.add_argument("--data-root", type=Path, default=REPO / ".data")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Regenerate all evaluation caches from raw MNIST.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs" / "inference" / "main_results.csv",
    )
    return parser.parse_args()


def evaluate_split(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    last_k: int,
) -> float:
    """Run the frozen SNN and probe over one deterministic split."""
    correct = 0
    total = 0
    with torch.inference_mode():
        for videos, labels in loader:
            # Moving only the current batch limits GPU memory while the full
            # deterministic split remains in the CPU evidence cache.
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            representation, _ = model.forward_spike_repr(videos, last_k)
            predictions = probe(representation).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    device = torch.device(args.device)
    bundles = {}
    for profile in args.profiles:
        checkpoint_path = REPO / "checkpoints" / profile / "best.pt"
        checkpoint, model, probe = load_inference_bundle(checkpoint_path, device)
        if checkpoint["cfg"]["model"]["profile"] != profile:
            raise AssertionError(f"Checkpoint/profile mismatch: {profile}")
        bundles[profile] = (checkpoint, model, probe)

    # All four runs share one data protocol; checking every embedded config
    # prevents a result from being paired with a different test distribution.
    reference_cfg = bundles[args.profiles[0]][0]["cfg"]["data"]
    comparable_keys = (
        "dataset_version",
        "cache_gen_seed",
        "T",
        "test_size",
        "sparse_digit_mask",
        "sparse_fg_retain_frac",
        "eval_noise_stds",
    )
    for profile, (checkpoint, _, _) in bundles.items():
        cfg = checkpoint["cfg"]["data"]
        if any(cfg[key] != reference_cfg[key] for key in comparable_keys):
            raise AssertionError(f"Evaluation protocol mismatch: {profile}")

    results = {profile: {"profile": profile} for profile in args.profiles}
    for split in SPLITS:
        dataset = EvaluationMovingMNIST(
            args.data_root,
            split=split,
            length=int(reference_cfg["test_size"]),
            time_steps=int(reference_cfg["T"]),
            noise_std=float(reference_cfg["eval_noise_stds"][split]),
            seed=int(reference_cfg["cache_gen_seed"]),
            dataset_version=str(reference_cfg["dataset_version"]),
            sparse_digit_mask=bool(reference_cfg["sparse_digit_mask"]),
            sparse_fg_retain_frac=float(reference_cfg["sparse_fg_retain_frac"]),
            rebuild_cache=args.rebuild_cache,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for profile, (checkpoint, model, probe) in bundles.items():
            accuracy = evaluate_split(
                model,
                probe,
                loader,
                device,
                int(checkpoint["last_k_steps"]),
            )
            recorded = float(checkpoint["test_accs"][split])
            results[profile][split] = accuracy
            results[profile][f"{split}_checkpoint"] = recorded
            print(
                f"{profile:7s} {split:18s} inferred={accuracy:.4%} "
                f"checkpoint={recorded:.4%} delta={accuracy-recorded:+.6f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[args.profiles[0]])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results[profile] for profile in args.profiles)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
