#!/usr/bin/env python3
"""Create a short MP4/GIF of the bundled paired SNN replay.

The animation is generated from the saved input frames and measured mean spike
rates.  It documents an actual inference replay; it is not a recording of a new
training run and does not add synthetic performance results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

# Rendering through Agg keeps video creation reproducible on a headless server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=Path, default=repo / "data" / "samples" / "paired_replay_120.npz"
    )
    parser.add_argument("--output-dir", type=Path, default=repo / "media")
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def render_frame(
    videos: np.ndarray,
    responses: np.ndarray,
    conditions: list[str],
    predictions: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    """Render one video frame with inputs above and measured layer rates below."""
    fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.4), dpi=100)
    for column, condition in enumerate(conditions):
        image_ax = axes[0, column]
        image_ax.imshow(videos[column, frame_index], cmap="gray", vmin=0, vmax=1)
        image_ax.set_title(
            f"{condition}\nprediction: {int(predictions[column])}", fontsize=9
        )
        image_ax.axis("off")

        rate_ax = axes[1, column]
        rates = responses[column, :, frame_index]
        rate_ax.bar(
            np.arange(1, 5),
            rates,
            color=("white", "0.75", "0.50", "0.25"),
            edgecolor="black",
            linewidth=0.8,
        )
        rate_ax.set_xticks((1, 2, 3, 4))
        rate_ax.set_xlabel("SNN layer", fontsize=8)
        if column == 0:
            rate_ax.set_ylabel("Mean spike rate", fontsize=8)
        rate_ax.set_ylim(0, max(0.12, float(responses.max()) * 1.08))
        rate_ax.grid(axis="y", color="0.88", linewidth=0.6)
        rate_ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Hierarchical SNN paired replay | true digit: 8 | "
        f"frame {frame_index + 1:02d}/{videos.shape[1]:02d}",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "Inputs and spike rates are measurements saved by the replay script; no retraining.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    # Reading the Agg canvas returns the exact RGB pixels encoded into both media files.
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(fig)
    return rgb


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Encode H.264 with the pinned backend or a locally available PyAV fallback."""
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        # Some project environments already provide PyAV but not the optional
        # imageio plugin.  Calling PyAV directly avoids imageio's incompatible
        # plugin keyword translation while producing the same review format.
        import av

        height, width = frames[0].shape[:2]
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        # Flushing emits delayed H.264 packets before closing the container.
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return

    # Selecting FFMPEG explicitly prevents an optional PyAV installation from
    # intercepting the .mp4 suffix with an incompatible keyword interface.
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            # Appending each frame keeps peak memory below a duplicated video
            # batch while using the pinned imageio-ffmpeg encoder.
            writer.append_data(frame)


def main() -> None:
    args = parse_args()
    if args.fps < 1:
        raise ValueError("fps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = np.load(args.sample)
    conditions = [str(value) for value in bundle["conditions"]]
    videos = bundle["example_videos"]
    responses = bundle["response_mean"]
    labels = bundle["labels"]
    display_index = int(np.flatnonzero(labels == 8)[0])
    predictions = bundle["predictions"][:, display_index]

    frames = [
        render_frame(videos, responses, conditions, predictions, frame_index)
        for frame_index in range(videos.shape[1])
    ]
    mp4_path = args.output_dir / "software_run_demo.mp4"
    gif_path = args.output_dir / "software_run_demo.gif"
    # imageio-ffmpeg supplies its own encoder, so MP4 creation does not depend
    # on a system ffmpeg command being installed on the review machine.
    write_mp4(mp4_path, frames, args.fps)
    # GIF is retained as an inline preview for GitHub and document reviewers.
    imageio.mimsave(gif_path, frames, duration=1000 / args.fps, loop=0)
    print(f"Created: {mp4_path}")
    print(f"Created: {gif_path}")


if __name__ == "__main__":
    main()
