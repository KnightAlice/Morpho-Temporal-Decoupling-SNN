#!/usr/bin/env python3
"""Verify SHA-256 fingerprints of primary checkpoints, records and samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash in blocks so checkpoint verification has bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / "EVIDENCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["files"]:
        path = repo / record["path"]
        if not path.is_file():
            failures.append(f"missing: {record['path']}")
            continue
        actual = sha256(path)
        if actual != record["sha256"]:
            failures.append(f"hash mismatch: {record['path']}")
        if path.stat().st_size != int(record["bytes"]):
            failures.append(f"size mismatch: {record['path']}")
    if failures:
        raise SystemExit("Evidence verification FAILED\n" + "\n".join(failures))
    print(f"Evidence verification PASS ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()
