#!/usr/bin/env python3
"""Rebuild portable English evidence plots from the bundled numeric files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

# A non-interactive backend makes plotting work on review servers without X11.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROFILE_LABELS = {
    "bio": "Decreasing",
    "equal": "Uniform",
    "reverse": "Reverse",
    "fixed": "Fixed",
}
SPLITS = [
    ("test_id", "Standard"),
    ("test_ood_speed", "Unseen speed"),
    ("test_ood_flicker", "Unseen flicker"),
    ("test_ood_noise", "Unseen noise"),
    ("test_combined_ood", "Combined"),
]


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir", type=Path, default=repo / "outputs" / "audit"
    )
    parser.add_argument(
        "--sample", type=Path, default=repo / "data" / "samples" / "paired_replay_120.npz"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo / "outputs" / "figures"
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read named columns so plots remain stable if a CSV column is moved."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Use fixed metadata and a tight canvas for review-ready PNG output."""
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accuracy(rows: list[dict[str, str]], output: Path) -> None:
    """Compare all five scores from each single selected checkpoint."""
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(SPLITS))
    offsets = np.linspace(-0.27, 0.27, len(rows))
    hatches = ("", "///", "xxx", "...")
    for index, (row, offset) in enumerate(zip(rows, offsets)):
        values = [100.0 * float(row[key]) for key, _ in SPLITS]
        ax.bar(
            x + offset,
            values,
            width=0.18,
            label=PROFILE_LABELS[row["profile"]],
            color="white",
            edgecolor="black",
            linewidth=1.0,
            hatch=hatches[index],
        )
    ax.set_xticks(x, [label for _, label in SPLITS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 100)
    ax.set_title("Same-checkpoint classification accuracy")
    ax.grid(axis="y", color="0.85", linewidth=0.7)
    ax.legend(ncol=4, frameon=False, loc="lower center")
    # Fixed margins keep labels stable across headless Matplotlib backends.
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.88)
    save_figure(fig, output)


def plot_tau(rows: list[dict[str, str]], output: Path) -> None:
    """Show the measured channel distribution for the decreasing profile."""
    selected = [row for row in rows if row["profile"] == "bio"]
    selected.sort(key=lambda row: int(row["layer"]))
    stats = []
    for row in selected:
        # ``bxp`` uses supplied measured quartiles and does not synthesize raw data.
        stats.append(
            {
                "label": f"Layer {row['layer']}",
                "whislo": float(row["min"]),
                "q1": float(row["q1"]),
                "med": float(row["median"]),
                "q3": float(row["q3"]),
                "whishi": float(row["max"]),
                "fliers": [],
            }
        )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    artists = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.55)
    for box in artists["boxes"]:
        box.set(facecolor="white", edgecolor="black", linewidth=1.2)
    for group in ("whiskers", "caps", "medians"):
        for artist in artists[group]:
            artist.set(color="black", linewidth=1.2)
    ax.set_ylabel("Membrane time constant, tau")
    ax.set_title("Measured tau distributions: decreasing profile")
    ax.grid(axis="y", color="0.87", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, output)


def plot_replay(sample_path: Path, output: Path) -> None:
    """Plot actual paired inputs and measured network activity from the replay."""
    bundle = np.load(sample_path)
    conditions = [str(value) for value in bundle["conditions"]]
    videos = bundle["example_videos"]
    responses = bundle["response_mean"]
    labels = bundle["labels"]
    display_index = int(np.flatnonzero(labels == 8)[0])
    predictions = bundle["predictions"][:, display_index]

    fig = plt.figure(figsize=(13.0, 6.6))
    grid = fig.add_gridspec(2, 5, height_ratios=(1.0, 1.15), hspace=0.42)
    representative_frame = videos.shape[1] // 2
    for column, condition in enumerate(conditions):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(videos[column, representative_frame], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{condition}\npred={int(predictions[column])}", fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    time = np.arange(1, responses.shape[-1] + 1)
    linestyles = ("-", "--", ":", "-.", (0, (5, 2)))
    for index, condition in enumerate(conditions):
        # Layers 1 and 4 are plotted as a compact observation of early/deep
        # activity.  They are spike rates, not filter impulse responses.
        ax.plot(
            time,
            responses[index, 0],
            color="black",
            linestyle=linestyles[index],
            linewidth=1.25,
            label=f"{condition}: layer 1",
        )
        ax.plot(
            time,
            responses[index, 3],
            color="0.50",
            linestyle=linestyles[index],
            linewidth=1.25,
            label=f"{condition}: layer 4",
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean spike rate")
    ax.set_title("120-sample paired replay")
    ax.grid(color="0.88", linewidth=0.7)
    ax.legend(ncol=5, fontsize=7.3, frameon=False, loc="upper center")
    fig.suptitle("Controlled input conditions and measured SNN activity", fontsize=14)
    # The lower axis spans all columns; explicit margins are more predictable
    # than ``tight_layout`` for this mixed GridSpec arrangement.
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.88)
    save_figure(fig, output)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(args.audit_dir / "best_checkpoint_metrics.csv")
    tau_summary = read_csv(args.audit_dir / "tau_summary.csv")
    plot_accuracy(metrics, args.output_dir / "accuracy_same_checkpoint.png")
    plot_tau(tau_summary, args.output_dir / "tau_distribution_bio.png")
    plot_replay(args.sample, args.output_dir / "paired_replay_summary.png")
    print(f"Plots rebuilt in: {args.output_dir}")


if __name__ == "__main__":
    main()
