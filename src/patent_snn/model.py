"""Minimal four-layer ProfilePLIF network used by the four supplied checkpoints.

This module intentionally contains inference code only.  Training losses,
projection heads, optimizers and later dendritic/three-compartment experiments
are excluded because none of them is called to reproduce the patent results.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, surrogate
from spikingjelly.clock_driven.neuron import BaseNode


PROFILE_SCALES = {
    "bio": (0.80, 0.55, 0.35, 0.15),
    "equal": (0.522, 0.522, 0.522, 0.522),
    "reverse": (0.15, 0.35, 0.55, 0.80),
    "fixed": (0.0, 0.0, 0.0, 0.0),
}


class ProfilePLIFNode(BaseNode):
    """Per-channel PLIF node with the checkpoint's layer profile mapping."""

    def __init__(
        self,
        channels: int,
        profile_scale: float,
        tau0: float,
        tau_min: float,
        tau_max: float,
        tau_u_variance_floor: float,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
    ) -> None:
        # ATan is the historical surrogate stored in the source experiment;
        # inference outputs are binary, while this function preserves the same
        # forward threshold and checkpoint-compatible module construction.
        super().__init__(
            v_threshold,
            v_reset,
            surrogate.ATan(),
            True,
            monitor_state=False,
        )
        self.channels = int(channels)
        self.profile_scale = float(profile_scale)
        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_u_variance_floor = float(tau_u_variance_floor)
        # The state key must remain ``u`` so strict checkpoint loading detects
        # any architecture mismatch instead of silently dropping parameters.
        self.u = nn.Parameter(torch.zeros(channels, dtype=torch.float32))
        self.last_h: torch.Tensor | None = None
        self.last_spike: torch.Tensor | None = None

    def compute_tau(self) -> torch.Tensor:
        """Map learned channel parameters to bounded, mean-normalized tau."""
        eps = 1e-5
        variance = self.u.var(unbiased=False).clamp(min=self.tau_u_variance_floor)
        normalized = (self.u - self.u.mean()) / (variance.sqrt() + eps)
        scaled = torch.exp(self.profile_scale * normalized)
        tau = self.tau0 * scaled / (scaled.mean() + eps)
        return tau.clamp(self.tau_min, self.tau_max)

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        """Advance one stateful PLIF step with channel-wise tau."""
        if drive.ndim != 4 or drive.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(drive.shape)}"
            )
        previous = self.v
        if not isinstance(previous, torch.Tensor):
            # BaseNode starts from a scalar reset value.  Materializing it with
            # the input shape makes the first step identical on CPU and CUDA.
            previous = torch.full_like(drive, float(previous))
        tau = self.compute_tau().view(1, self.channels, 1, 1)
        reset_value = 0.0 if self.v_reset is None else float(self.v_reset)
        pre_reset = previous + (drive - (previous - reset_value)) / tau
        # Calling BaseNode's configured surrogate preserves the original hard
        # threshold in the forward pass and its exact reset rule.
        spike = self.surrogate_function(pre_reset - self.v_threshold)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            next_v = pre_reset - reset_spike * self.v_threshold
        else:
            next_v = pre_reset * (1.0 - reset_spike) + reset_value * reset_spike
        self.v = next_v
        self.last_h = pre_reset.clone()
        self.last_spike = spike
        return spike

    def tau_vector_detach(self) -> torch.Tensor:
        """Return learned physical time constants for evidence plotting."""
        return self.compute_tau().detach()


class MovingMNISTInferenceSNN(nn.Module):
    """Exact inference path used by the four recovered SupCon probe runs."""

    def __init__(
        self,
        *,
        profile: str,
        time_steps: int,
        tau0: float,
        tau_min: float,
        tau_max: float,
        tau_u_variance_floor: float,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if profile not in PROFILE_SCALES:
            raise ValueError(f"Unsupported checkpoint profile: {profile!r}")
        self.profile = profile
        self.T = int(time_steps)
        scales = PROFILE_SCALES[profile]
        channels = (32, 64, 128, 128)

        def make_node(channel_count: int, scale: float) -> ProfilePLIFNode:
            # Keeping construction in one place ensures all four layers use the
            # same physical bounds recorded inside each checkpoint.
            return ProfilePLIFNode(
                channel_count,
                scale,
                tau0,
                tau_min,
                tau_max,
                tau_u_variance_floor,
            )

        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.plif1 = make_node(channels[0], scales[0])
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.plif2 = make_node(channels[1], scales[1])
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.plif3 = make_node(channels[2], scales[2])
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels[3])
        self.plif4 = make_node(channels[3], scales[3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        # ``fc`` is not used by the SupCon probe prediction, but it belongs to
        # the saved model state and is retained for strict integrity checking.
        self.fc = nn.Linear(channels[3], num_classes)
        self._plif_layers = [self.plif1, self.plif2, self.plif3, self.plif4]

    @property
    def plif_layers(self) -> list[ProfilePLIFNode]:
        """Expose the four layers for measured response hooks."""
        return self._plif_layers

    def forward_spike_repr(
        self, videos: torch.Tensor, last_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the last-k top-layer spike representation and mean spike rate."""
        if videos.ndim != 5 or videos.shape[2] != 1:
            raise ValueError(f"Expected videos [B,T,1,H,W], got {tuple(videos.shape)}")
        if videos.shape[1] != self.T:
            raise ValueError(f"Checkpoint expects T={self.T}, got T={videos.shape[1]}")
        # Resetting before every video batch prevents membrane state leakage
        # between batches and is required by the original evaluation path.
        functional.reset_net(self)
        batch_size, time_steps = videos.shape[:2]
        k_effective = max(1, min(int(last_k), time_steps))
        start = time_steps - k_effective
        representation_sum = torch.zeros(
            batch_size,
            self.fc.in_features,
            device=videos.device,
            dtype=videos.dtype,
        )
        spike_sums = torch.zeros(4, device=videos.device, dtype=videos.dtype)

        for time_index in range(time_steps):
            out = self.plif1(self.bn1(self.conv1(videos[:, time_index])))
            spike_sums[0] += out.float().mean()
            out = self.pool1(out)
            out = self.plif2(self.bn2(self.conv2(out)))
            spike_sums[1] += out.float().mean()
            out = self.pool2(out)
            out = self.plif3(self.bn3(self.conv3(out)))
            spike_sums[2] += out.float().mean()
            out = self.pool3(out)
            out = self.plif4(self.bn4(self.conv4(out)))
            spike_sums[3] += out.float().mean()
            if time_index >= start:
                # SupCon inference uses top-layer spikes, averaged spatially and
                # over the last four frames, before applying the linear probe.
                representation_sum += self.gap(out.float()).flatten(1)

        representation = representation_sum / float(k_effective)
        mean_spike_rate = (spike_sums / float(time_steps)).mean()
        return representation, mean_spike_rate

    def forward(self, videos: torch.Tensor, last_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias the only supported inference representation path."""
        return self.forward_spike_repr(videos, last_k)

    def tau_stats(self) -> list[tuple[float, float, float, float]]:
        """Return mean, population std, minimum and maximum tau per layer."""
        rows = []
        for layer in self.plif_layers:
            values = layer.tau_vector_detach()
            rows.append(
                (
                    values.mean().item(),
                    values.std(unbiased=False).item(),
                    values.min().item(),
                    values.max().item(),
                )
            )
        return rows
