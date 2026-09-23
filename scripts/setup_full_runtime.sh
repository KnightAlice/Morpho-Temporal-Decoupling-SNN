#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository from this script so the command works from any cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPIKINGJELLY_DIR="$REPO_ROOT/vendor/spikingjelly"
SPIKINGJELLY_COMMIT="73f94ab983d0167623015537f7d4460b064cfca1"

# Installing pinned packages supplies the numerical, plotting and MP4 runtime
# used to validate this package. A virtual environment is strongly recommended.
"$PYTHON_BIN" -m pip install --disable-pip-version-check -r "$REPO_ROOT/requirements.txt"

if [[ ! -d "$SPIKINGJELLY_DIR/.git" ]]; then
  # The historical SNN implementation imports ``spikingjelly.clock_driven``;
  # cloning its exact 2020 commit avoids API drift in newer releases.
  git clone https://github.com/fangwei123456/spikingjelly.git "$SPIKINGJELLY_DIR"
fi
git -C "$SPIKINGJELLY_DIR" fetch --quiet origin "$SPIKINGJELLY_COMMIT"
git -C "$SPIKINGJELLY_DIR" checkout --quiet "$SPIKINGJELLY_COMMIT"

echo "Inference runtime ready. Add the following before direct Python calls:"
echo "export PYTHONPATH=$REPO_ROOT/src:$SPIKINGJELLY_DIR"
