#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# The audit reconstructs all table values from the four checkpoints and then
# checks those values against the bundled same-step TensorBoard scalars.
"$PYTHON_BIN" "$REPO_ROOT/scripts/audit_checkpoints.py"

# Plotting consumes only audited CSVs and the bundled paired replay sample.
"$PYTHON_BIN" "$REPO_ROOT/scripts/plot_evidence.py"

# The media step documents the saved replay as both MP4 and GitHub-preview GIF.
"$PYTHON_BIN" "$REPO_ROOT/scripts/create_demo_media.py"

# The final check detects missing or modified primary evidence files.
"$PYTHON_BIN" "$REPO_ROOT/scripts/verify_evidence.py"

echo "Quick demo PASS: inspect $REPO_ROOT/outputs and $REPO_ROOT/media"
