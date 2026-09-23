"""Deterministic Moving-MNIST stimulus generation for checkpoint inference.

Only the five evaluation conditions and the fixed paired replay are retained.
Online training augmentation, balanced training samplers and epoch state are
deliberately absent from this evidence package.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress display is optional.
    def tqdm(iterator: Any, **_: Any) -> Any:
        return iterator


TRAIN_SPEED_CORRIDOR = (1, 2, 3, 4, 8)
TRAIN_FLICKER_CORRIDOR = (None, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4)
SPLIT_CONDITIONS: dict[str, dict[str, Sequence[float | None]]] = {
    "test_id": {
        "speed_set": (1, 2, 4),
        "flicker_set": (None, 1 / 16, 1 / 8, 1 / 4),
    },
    "test_ood_speed": {
        "speed_set": (6,),
        "flicker_set": TRAIN_FLICKER_CORRIDOR,
    },
    "test_ood_noise": {
        "speed_set": TRAIN_SPEED_CORRIDOR,
        "flicker_set": TRAIN_FLICKER_CORRIDOR,
    },
    "test_ood_flicker": {
        "speed_set": TRAIN_SPEED_CORRIDOR,
        "flicker_set": (1 / 32,),
    },
    "test_combined_ood": {
        "speed_set": (6,),
        "flicker_set": (1 / 32,),
    },
}

SPLIT_SEED_OFFSET = {
    "test_id": 203,
    "test_ood_speed": 307,
    "test_ood_noise": 401,
    "test_ood_flicker": 503,
    "test_combined_ood": 607,
}


def _rand_uniform(generator: torch.Generator, low: float, high: float) -> float:
    value = torch.empty(1, dtype=torch.float32)
    value.uniform_(low, high, generator=generator)
    return float(value.item())


def _pick(generator: torch.Generator, choices: Sequence[Any]) -> Any:
    index = int(torch.randint(len(choices), (1,), generator=generator).item())
    return choices[index]


def _sparse_foreground(
    frame: torch.Tensor,
    y: int,
    x: int,
    height: int,
    width: int,
    generator: torch.Generator,
    retain_fraction: float,
) -> None:
    """Retain a seeded fraction of nonzero digit pixels in-place."""
    patch = frame[y : y + height, x : x + width]
    foreground = patch > 1e-3
    if not bool(foreground.any().item()):
        return
    draws = torch.rand((height, width), dtype=torch.float32, generator=generator)
    kept = foreground & (draws < retain_fraction)
    frame[y : y + height, x : x + width] = torch.where(
        kept, patch, torch.zeros_like(patch)
    )


def make_moving_digit(
    digit: torch.Tensor,
    *,
    time_steps: int,
    height: int,
    width: int,
    speed: float,
    theta: float,
    flicker: float | None,
    flicker_phase: float,
    noise_std: float,
    generator: torch.Generator,
    sparse_digit_mask: bool,
    sparse_fg_retain_frac: float,
) -> torch.Tensor:
    """Generate the exact 64x64 moving stimulus used by the saved evaluations."""
    video = torch.zeros(time_steps, 1, height, width, dtype=torch.float32)
    digit_height, digit_width = digit.shape[-2:]
    velocity_x = speed * math.cos(theta)
    velocity_y = speed * math.sin(theta)
    x = _rand_uniform(generator, 0, width - digit_width)
    y = _rand_uniform(generator, 0, height - digit_height)

    for time_index in range(time_steps):
        if x <= 0 and velocity_x < 0 or x >= width - digit_width and velocity_x > 0:
            velocity_x = -velocity_x
        if y <= 0 and velocity_y < 0 or y >= height - digit_height and velocity_y > 0:
            velocity_y = -velocity_y
        x = max(0.0, min(float(width - digit_width), x + velocity_x))
        y = max(0.0, min(float(height - digit_height), y + velocity_y))
        x_index = int(round(x))
        y_index = int(round(y))
        frame = video[time_index, 0]
        frame[y_index : y_index + digit_height, x_index : x_index + digit_width] = digit[0]
        if flicker is not None:
            angle = 2.0 * math.pi * flicker * float(time_index) + flicker_phase
            frame.mul_(0.75 + 0.25 * math.sin(angle))
        if sparse_digit_mask:
            # The foreground draw occurs before noise, matching the experiment
            # and ensuring background noise is never interpreted as digit mask.
            _sparse_foreground(
                frame,
                y_index,
                x_index,
                digit_height,
                digit_width,
                generator,
                sparse_fg_retain_frac,
            )
        if noise_std > 0:
            frame.add_(noise_std * torch.randn(frame.shape, generator=generator))
        frame.clamp_(0.0, 1.0)
    return video


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    raise TypeError(f"Cannot include {type(value)} in cache fingerprint")


def evaluation_cache_path(
    data_root: Path,
    *,
    split: str,
    length: int,
    time_steps: int,
    noise_std: float,
    seed: int,
    dataset_version: str,
    sparse_fg_retain_frac: float,
) -> Path:
    """Reproduce the original cache filename so existing caches remain usable."""
    # The historical fingerprint also contained zero-occlusion placeholders;
    # include them here even though occlusion code is absent from inference.
    specs = {
        **SPLIT_CONDITIONS[split],
        "occlusion_set": [0],
        "occ_patch_set": [20],
    }
    payload = {
        "split": split,
        "T": time_steps,
        "noise_std": float(f"{noise_std:.12g}"),
        "sparse_fg_retain_frac": float(f"{sparse_fg_retain_frac:.12g}"),
        "specs": _canonical(specs),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    safe_version = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in dataset_version
    )
    filename = (
        f"{split}_{length}_T{time_steps}_noise{noise_std:g}_g{seed}_"
        f"{safe_version}_{fingerprint}.pt"
    )
    return data_root / "moving_mnist_cache" / filename


class EvaluationMovingMNIST(Dataset[tuple[torch.Tensor, int]]):
    """One deterministic saved-protocol evaluation split."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        split: str,
        length: int,
        time_steps: int,
        noise_std: float,
        seed: int,
        dataset_version: str,
        sparse_digit_mask: bool,
        sparse_fg_retain_frac: float,
        rebuild_cache: bool = False,
    ) -> None:
        if split not in SPLIT_CONDITIONS:
            raise ValueError(f"Unknown evaluation split: {split}")
        self.data_root = Path(data_root)
        self.split = split
        self.length = int(length)
        self.time_steps = int(time_steps)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.sparse_digit_mask = bool(sparse_digit_mask)
        self.sparse_fg_retain_frac = float(sparse_fg_retain_frac)
        self.cache_path = evaluation_cache_path(
            self.data_root,
            split=split,
            length=length,
            time_steps=time_steps,
            noise_std=noise_std,
            seed=seed,
            dataset_version=dataset_version,
            sparse_fg_retain_frac=sparse_fg_retain_frac,
        )
        if rebuild_cache or not self.cache_path.is_file():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Building once preserves the original single-generator draw order;
            # per-item lazy generation would produce different test stimuli.
            torch.save(self._generate_all(), self.cache_path)
        self.cached = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        if int(self.cached["videos"].shape[0]) != self.length:
            raise ValueError(f"Cache length mismatch: {self.cache_path}")

    def _generate_all(self) -> dict[str, Any]:
        dataset = datasets.MNIST(
            root=str(self.data_root / "mnist_raw"),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        generator = torch.Generator().manual_seed(
            self.seed + SPLIT_SEED_OFFSET[self.split]
        )
        specs = SPLIT_CONDITIONS[self.split]
        videos = torch.empty(
            self.length, self.time_steps, 1, 64, 64, dtype=torch.float32
        )
        labels = torch.empty(self.length, dtype=torch.long)
        for index in tqdm(range(self.length), desc=f"build {self.split}"):
            mnist_index = int(
                torch.randint(0, len(dataset), (1,), generator=generator).item()
            )
            image, label = dataset[mnist_index]
            speed = float(_pick(generator, specs["speed_set"]))
            flicker = _pick(generator, specs["flicker_set"])
            theta = _rand_uniform(generator, -math.pi, math.pi)
            phase = _rand_uniform(generator, 0, 2 * math.pi)
            # All calls share one generator because this is the exact cache
            # construction sequence used by the recorded 10,000-sample scores.
            videos[index] = make_moving_digit(
                image,
                time_steps=self.time_steps,
                height=64,
                width=64,
                speed=speed,
                theta=theta,
                flicker=flicker,
                flicker_phase=phase,
                noise_std=self.noise_std,
                generator=generator,
                sparse_digit_mask=self.sparse_digit_mask,
                sparse_fg_retain_frac=self.sparse_fg_retain_frac,
            )
            labels[index] = int(label)
        return {"videos": videos, "labels": labels}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Cloning protects the memory-backed evidence cache from transforms or
        # accidental in-place device preparation in downstream code.
        return self.cached["videos"][index].clone(), int(self.cached["labels"][index])
