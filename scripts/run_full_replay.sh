#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MNIST_RAW="${MNIST_RAW:-$REPO_ROOT/.data/mnist_raw/MNIST/raw}"
SPIKINGJELLY_DIR="$REPO_ROOT/vendor/spikingjelly"

if [[ ! -d "$SPIKINGJELLY_DIR/spikingjelly/clock_driven" ]]; then
  echo "Missing legacy SpikingJelly. Run scripts/setup_full_runtime.sh first." >&2
  exit 2
fi
if [[ ! -f "$MNIST_RAW/t10k-images-idx3-ubyte" ]]; then
  echo "Missing MNIST raw files. Run scripts/download_mnist.py first." >&2
  exit 2
fi

# The local src path is placed first to prevent an expired Moving-MNIST checkout
# from being imported. The second path supplies only the historical SNN runtime.
PYTHONPATH="$REPO_ROOT/src:$SPIKINGJELLY_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$REPO_ROOT/scripts/replay_checkpoint.py" \
  --checkpoint "$REPO_ROOT/checkpoints/bio/best.pt" \
  --mnist-raw "$MNIST_RAW" \
  --output "$REPO_ROOT/outputs/replay/paired_replay_120.npz" \
  "$@"

# Exact labels/predictions and bounded floating-point measurements demonstrate
# that the compact inference path reproduces the bundled reference replay.
"$PYTHON_BIN" "$REPO_ROOT/scripts/compare_replay.py" \
  "$REPO_ROOT/outputs/replay/paired_replay_120.npz"

echo "Full replay complete: $REPO_ROOT/outputs/replay/paired_replay_120.npz"
