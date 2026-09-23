#!/usr/bin/env python3
"""Audit the four patent checkpoints and rebuild their tabular evidence.

This script is deliberately independent of the SNN runtime.  It reads the
saved parameters directly, recomputes channel-wise membrane constants, and
checks the same-epoch TensorBoard scalars.  That keeps the primary patent
evidence auditable on a CPU without regenerating MNIST or retraining a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing import event_accumulator


PROFILES = ("bio", "equal", "reverse", "fixed")
SPLITS = (
    "test_id",
    "test_ood_speed",
    "test_ood_noise",
    "test_ood_flicker",
    "test_combined_ood",
)
SCALES = {
    "bio": [0.80, 0.55, 0.35, 0.15],
    "equal": [0.522] * 4,
    "reverse": [0.15, 0.35, 0.55, 0.80],
    "fixed": [0.0] * 4,
}


def parse_args() -> argparse.Namespace:
    """Resolve all default paths from the repository instead of a server path."""
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=repo / "checkpoints")
    parser.add_argument("--tensorboard-root", type=Path, default=repo / "data" / "tensorboard")
    parser.add_argument("--output-dir", type=Path, default=repo / "outputs" / "audit")
    parser.add_argument(
        "--skip-tensorboard",
        action="store_true",
        help="Rebuild checkpoint tables without validating the bundled event records.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Use explicit columns so a later plot cannot silently reorder test splits."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Fingerprint each evidence file so reviewers can detect substitutions."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_tau(
    raw_u: torch.Tensor,
    scale: float,
    tau0: float,
    variance_floor: float,
    tau_min: float,
    tau_max: float,
) -> torch.Tensor:
    """Apply the exact mapping used by ``src/patent_snn/model.py``."""
    eps = 1e-5
    effective_variance = raw_u.var(unbiased=False).clamp(min=variance_floor)
    normalized = (raw_u - raw_u.mean()) / (effective_variance.sqrt() + eps)
    exp_scaled = torch.exp(scale * normalized)
    return (tau0 * exp_scaled / (exp_scaled.mean() + eps)).clamp(tau_min, tau_max)


def load_events(path: Path) -> event_accumulator.EventAccumulator:
    """Load every scalar because resumed runs contain multiple event files."""
    reader = event_accumulator.EventAccumulator(
        str(path), size_guidance={"scalars": 0, "histograms": 0}
    )
    reader.Reload()
    return reader


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    peaks: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    tau_channels: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    common_config: dict[str, Any] | None = None

    for profile in PROFILES:
        checkpoint_path = args.checkpoint_root / profile / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        # Loading on CPU makes the audit usable without a CUDA device.  The
        # file is a trusted project artifact and contains optimizer metadata,
        # so ``weights_only=False`` is intentional and must not be used on an
        # untrusted checkpoint downloaded from elsewhere.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        cfg = checkpoint["cfg"]
        if checkpoint["train_mode"] != "supcon_probe":
            raise AssertionError(f"Unexpected train mode for {profile}")
        if cfg["model"]["profile"] != profile or cfg["train"]["seed"] != 0:
            raise AssertionError(f"Profile/seed mismatch for {profile}")
        if checkpoint["probe"] is None or checkpoint["projection_head"] is None:
            raise AssertionError(f"Missing SupCon heads for {profile}")
        if int(checkpoint["last_k_steps"]) != 4:
            raise AssertionError(f"Unexpected temporal readout for {profile}")

        # Removing only the intended name/profile fields proves the four runs
        # share the same architecture, data and optimizer configuration.
        comparable = json.loads(json.dumps(cfg))
        comparable["experiment"].pop("name", None)
        comparable["model"].pop("profile", None)
        if common_config is None:
            common_config = comparable
        elif comparable != common_config:
            raise AssertionError(f"Non-profile configuration mismatch: {profile}")

        epoch = int(checkpoint["epoch"])
        scores = checkpoint["test_accs"]
        if set(scores) != set(SPLITS):
            raise AssertionError(f"Unexpected score keys for {profile}: {sorted(scores)}")
        metrics.append(
            {
                "profile": profile,
                "epoch_zero_based": epoch,
                "epoch_one_based": epoch + 1,
                "train_mode": checkpoint["train_mode"],
                **{split: float(scores[split]) for split in SPLITS},
            }
        )

        reader = None
        scalar_tags: set[str] = set()
        if not args.skip_tensorboard:
            reader = load_events(args.tensorboard_root / profile)
            scalar_tags = set(reader.Tags()["scalars"])
            for split in SPLITS:
                tag = f"{split}/accuracy_probe"
                if tag not in scalar_tags:
                    raise AssertionError((profile, tag))
                values = reader.Scalars(tag)
                # This call checks that every table entry exists at the exact
                # checkpoint step, rather than accepting a split-wise peak.
                if not any(
                    event.step == epoch
                    and math.isclose(event.value, float(scores[split]), abs_tol=1e-5)
                    for event in values
                ):
                    raise AssertionError((profile, split, epoch))
                peak = max(values, key=lambda event: event.value)
                peaks.append(
                    {
                        "profile": profile,
                        "split": split,
                        "peak_accuracy": float(peak.value),
                        "peak_step_zero_based": int(peak.step),
                        "checkpoint_accuracy": float(scores[split]),
                    }
                )

        settings = cfg["model"]
        for layer, scale in enumerate(SCALES[profile], start=1):
            raw_u = checkpoint["model"][f"plif{layer}.u"]
            tau = compute_tau(
                raw_u,
                scale,
                float(settings["tau0"]),
                float(settings["tau_u_variance_floor"]),
                float(settings["tau_min"]),
                float(settings["tau_max"]),
            )
            tau_mean = float(tau.mean())
            tau_std = float(tau.std(unbiased=False))
            if reader is not None:
                for statistic, calculated in (("mean", tau_mean), ("std", tau_std)):
                    tag = f"tau_supcon/layer_{layer}_{statistic}"
                    if tag not in scalar_tags or not any(
                        event.step == epoch
                        and math.isclose(event.value, calculated, abs_tol=1e-4)
                        for event in reader.Scalars(tag)
                    ):
                        raise AssertionError((profile, layer, statistic))

            quantiles = torch.quantile(
                tau, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            ).tolist()
            clipped = int(
                (
                    (tau <= float(settings["tau_min"]) + 1e-6)
                    | (tau >= float(settings["tau_max"]) - 1e-6)
                ).sum()
            )
            tau_rows.append(
                {
                    "profile": profile,
                    "layer": layer,
                    "scale": scale,
                    "n_channels": raw_u.numel(),
                    "tau_mean": tau_mean,
                    "tau_std_population": tau_std,
                    "log_tau_std_population": float(tau.log().std(unbiased=False)),
                    "min": quantiles[0],
                    "q1": quantiles[1],
                    "median": quantiles[2],
                    "q3": quantiles[3],
                    "max": quantiles[4],
                    "clipped_channels": clipped,
                    "variance_floor_active": bool(
                        raw_u.var(unbiased=False)
                        < float(settings["tau_u_variance_floor"])
                    ),
                }
            )
            for channel, value in enumerate(tau.tolist(), start=1):
                tau_channels.append(
                    {
                        "profile": profile,
                        "layer": layer,
                        "channel": channel,
                        "tau": value,
                    }
                )

        manifest.append(
            {
                "profile": profile,
                "checkpoint": checkpoint_path.relative_to(checkpoint_path.parents[2]).as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256(checkpoint_path),
            }
        )

    write_csv(args.output_dir / "best_checkpoint_metrics.csv", list(metrics[0]), metrics)
    if peaks:
        write_csv(args.output_dir / "training_peak_metrics.csv", list(peaks[0]), peaks)
    write_csv(args.output_dir / "tau_summary.csv", list(tau_rows[0]), tau_rows)
    write_csv(args.output_dir / "tau_channels.csv", list(tau_channels[0]), tau_channels)
    write_csv(args.output_dir / "checkpoint_manifest.csv", list(manifest[0]), manifest)

    summary = {
        "status": "PASS",
        "profiles": list(PROFILES),
        "tensorboard_checked": not args.skip_tensorboard,
        "same_checkpoint_result_rule": True,
        "bio_budget": sum(value * value for value in SCALES["bio"]) / 4,
        "equal_budget": sum(value * value for value in SCALES["equal"]) / 4,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Evidence rebuilt in: {args.output_dir}")


if __name__ == "__main__":
    main()
