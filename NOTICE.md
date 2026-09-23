# Notice

This package contains an inference-only technical snapshot assembled for reproducibility and patent examination. Training entry points and architectures unused by the supplied checkpoints are outside its scope. The patent draft and personal applicant/inventor information are intentionally excluded from the public package.

MNIST is not redistributed. `scripts/download_mnist.py` obtains it through `torchvision`. The legacy SpikingJelly runtime is not vendored; `scripts/setup_full_runtime.sh` checks out upstream commit `73f94ab983d0167623015537f7d4460b064cfca1`. Use of those dependencies remains subject to their respective upstream terms.

The included checkpoint and event files are evidence artifacts. Loading a PyTorch checkpoint may execute pickle deserialization; use the bundled files or otherwise trusted files only.
