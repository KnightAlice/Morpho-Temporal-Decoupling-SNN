#!/usr/bin/env python3
"""Replay controlled test digits through the recovered bio SupCon checkpoint.

The saved checkpoint contains weights and split accuracies, but no per-frame
responses or sample embeddings. This script extracts those two measurements
without training and labels them as a small, paired replay rather than as the
original 10,000-sample evaluation splits.
"""

from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import gettempdir

import numpy as np
import torch
from torchvision import datasets, transforms


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "checkpoints" / "bio" / "best.pt"
DEFAULT_MNIST_RAW = REPO / ".data" / "mnist_raw" / "MNIST" / "raw"
CONDITIONS = ("standard", "fast", "flicker", "noise", "combined")
SAMPLES_PER_CLASS = 12

# This import path selects the bundled copy of the newest verified model rather
# than an older checkout elsewhere on the machine.  The exact legacy
# SpikingJelly runtime is supplied by PYTHONPATH or ``vendor/spikingjelly``.
sys.path.insert(0, str(REPO / "src"))
from patent_snn.checkpoint import load_inference_bundle  # noqa: E402
from patent_snn.data import make_moving_digit  # noqa: E402


def stage_mnist(raw_dir: Path) -> Path:
    """Expose existing MNIST bytes in the layout expected by the data code."""
    required = (
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
    )
    missing = [name for name in required if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MNIST raw directory misses {missing}: {raw_dir}")
    root = Path(gettempdir()) / "patent_figure5_mnist"
    link = root / "mnist_raw" / "MNIST" / "raw"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() != raw_dir.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(raw_dir.resolve(), target_is_directory=True)
    return root


def load_checkpoint(path: Path, device: torch.device):
    """Load the exact profile and probe whose results appear in patent Table 1."""
    # The shared loader performs strict model and probe key checks before any
    # measurement, preventing expired architectures from being replayed.
    checkpoint, model, probe = load_inference_bundle(path, device)
    if checkpoint["cfg"]["model"].get("profile") != "bio":
        raise ValueError("Figure 5 replay must use the descending bio profile")
    return checkpoint, model, probe


def selected_test_digits(root: Path):
    """Choose the first 12 test digits of each class before viewing features."""
    dataset = datasets.MNIST(
        root=str(root / "mnist_raw"), train=False, download=False,
        transform=transforms.ToTensor(),
    )
    counts = [0] * 10
    selected = []
    for index in range(len(dataset)):
        digit, label = dataset[index]
        if counts[label] < SAMPLES_PER_CLASS:
            selected.append((index, digit, int(label)))
            counts[label] += 1
        if all(count == SAMPLES_PER_CLASS for count in counts):
            break
    if len(selected) != 10 * SAMPLES_PER_CLASS:
        raise ValueError("Could not select a balanced set of MNIST test digits")
    return selected


def make_video(digit: torch.Tensor, item_index: int, condition: str, cfg: dict):
    """Change one nuisance factor at a time for a paired stimulus replay."""
    # Sharing the spatial seed preserves initial placement and sparse mask
    # draws between variants; only the named condition changes the input.
    parameter_rng = torch.Generator().manual_seed(100_000 + item_index)
    theta = (torch.rand((), generator=parameter_rng).item() * 2.0 - 1.0) * np.pi
    phase = torch.rand((), generator=parameter_rng).item() * 2.0 * np.pi
    spatial_rng = torch.Generator().manual_seed(200_000 + item_index)
    fast = condition in ("fast", "combined")
    flicker = condition in ("flicker", "combined")
    video = make_moving_digit(
        digit,
        time_steps=int(cfg["T"]),
        height=64,
        width=64,
        speed=6.0 if fast else 2.0,
        theta=theta,
        flicker=(1.0 / 32.0) if flicker else None,
        flicker_phase=phase,
        noise_std=0.0,
        generator=spatial_rng,
        sparse_digit_mask=bool(cfg.get("sparse_digit_mask", True)),
        sparse_fg_retain_frac=float(cfg.get("sparse_fg_retain_frac", 0.25)),
    )
    if condition in ("noise", "combined"):
        # Add the evaluation noise after drawing the same masked trajectory so
        # the paired contrast is attributable to the named perturbation.
        noise_rng = torch.Generator().manual_seed(300_000 + item_index)
        video = (video + 0.06 * torch.randn(video.shape, generator=noise_rng)).clamp(0, 1)
    return video


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mnist-raw", type=Path, default=DEFAULT_MNIST_RAW)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs" / "replay" / "paired_replay_120.npz",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    root = stage_mnist(args.mnist_raw)
    checkpoint, model, probe = load_checkpoint(args.checkpoint, device)
    digits = selected_test_digits(root)
    labels = np.asarray([label for _, _, label in digits], dtype=np.int16)
    digit_indices = np.asarray([index for index, _, _ in digits], dtype=np.int32)
    data_cfg = checkpoint["cfg"]["data"]
    time_steps = int(data_cfg["T"])
    count = len(digits)
    # Selection order follows the MNIST test indices, so find the first digit
    # 8 explicitly rather than assuming samples are grouped by label.
    display_index = next(i for i, (_, _, label) in enumerate(digits) if label == 8)
    features = np.zeros((len(CONDITIONS), count, model.fc.in_features), np.float32)
    predictions = np.zeros((len(CONDITIONS), count), np.int16)
    responses = np.zeros((len(CONDITIONS), 4, time_steps), np.float32)
    example_videos = np.zeros((len(CONDITIONS), time_steps, 64, 64), np.float32)

    for condition_index, condition in enumerate(CONDITIONS):
        response_sum = np.zeros((4, time_steps), np.float64)
        for start in range(0, count, args.batch_size):
            batch_items = digits[start : start + args.batch_size]
            videos = [
                make_video(digit, index, condition, data_cfg)
                for index, digit, _ in batch_items
            ]
            if start <= display_index < start + len(videos):
                # A fixed class-8 exemplar has more visible strokes under the
                # training-time sparse mask; all quantitative panels use all 120 digits.
                example_videos[condition_index] = videos[display_index - start][:, 0].numpy()
            batch = torch.stack(videos).to(device)
            rate_steps = [[] for _ in range(4)]
            handles = []
            for layer_index, layer in enumerate(model.plif_layers):
                # The SNN calls each layer once per video frame. A forward hook
                # records measured spike rates without altering its computation.
                def record(_module, _inputs, output, layer_index=layer_index):
                    rate_steps[layer_index].append(
                        output.detach().float().mean(dim=(1, 2, 3)).cpu().numpy()
                    )
                handles.append(layer.register_forward_hook(record))
            try:
                with torch.inference_mode():
                    representation, _ = model.forward_spike_repr(
                        batch, int(checkpoint["last_k_steps"])
                    )
                    logits = probe(representation)
            finally:
                for handle in handles:
                    handle.remove()
            features[condition_index, start : start + len(videos)] = (
                representation.detach().cpu().numpy()
            )
            predictions[condition_index, start : start + len(videos)] = (
                logits.argmax(-1).detach().cpu().numpy()
            )
            for layer_index, steps in enumerate(rate_steps):
                if len(steps) != time_steps:
                    raise RuntimeError(f"Layer {layer_index + 1} emitted {len(steps)} frames")
                response_sum[layer_index] += np.stack(steps).sum(axis=1)
        responses[condition_index] = (response_sum / count).astype(np.float32)
        print(f"{condition}: paired-replay accuracy={(predictions[condition_index] == labels).mean():.3f}", flush=True)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    # Retain only measurements and one display video per condition; the full
    # 600-video sample set is reproducible from indices, seeds, and data code.
    np.savez_compressed(
        output, conditions=np.asarray(CONDITIONS), labels=labels,
        digit_indices=digit_indices, features=features, predictions=predictions,
        response_mean=responses, example_videos=example_videos,
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest(),
        "mnist_raw_supplied": str(args.mnist_raw),
        "samples_per_class": SAMPLES_PER_CLASS,
        "display_digit": 8,
        "conditions": list(CONDITIONS),
        "note": "Paired replay over five controlled input conditions.",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
