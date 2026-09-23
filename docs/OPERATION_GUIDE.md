# Operation Guide

## 1. Scope

This package reproduces the principal results through inference only. It has no training entry point. Inference settings come from each `best.pt` embedded `cfg`; model and 128→10 probe states load with `strict=True`.

| Level | Purpose | Requirements |
| --- | --- | --- |
| Quick audit | Checkpoints, same-step TensorBoard, tau, plots, media, hashes | CPU; no MNIST/SpikingJelly |
| Runtime smoke test | Strict-load four checkpoints and run one saved stimulus each | CPU/GPU; legacy SpikingJelly |
| 120-sample replay | Predictions, features, and layer spike rates under five conditions | MNIST; CPU/GPU |
| Full main results | Four profiles × five 10,000-sample splits | MNIST; GPU recommended; about 13 GiB cache |

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For network inference, install the pinned historical runtime:

```bash
bash scripts/setup_full_runtime.sh
```

It checks out SpikingJelly commit `73f94ab983d0167623015537f7d4460b064cfca1`, which provides the checkpoint-era `spikingjelly.clock_driven` API.

## 3. Quick audit

```bash
bash scripts/run_quick_demo.sh
```

This reconstructs the five same-checkpoint scores and channel taus, validates same-step TensorBoard scalars, rebuilds evidence plots/media, and checks SHA-256 records. It should end with `Evidence verification PASS` and `Quick demo PASS`.

## 4. Inference runtime smoke test

```bash
export PYTHONPATH="$PWD/src:$PWD/vendor/spikingjelly"
python scripts/check_inference_runtime.py
```

All four checkpoints are strictly loaded and run through one saved 16-frame standard-condition stimulus.

## 5. Paired replay

```bash
python scripts/download_mnist.py
bash scripts/run_full_replay.sh --device cuda:0
```

CPU alternative:

```bash
bash scripts/run_full_replay.sh --device cpu --batch-size 8
```

Outputs are written to `outputs/replay/`. The comparison requires exact inputs, labels, indices, and predictions. It permits `1/256` for representations and `1e-5` for mean spike rates to cover rare CUDA threshold-boundary differences.

## 6. Recompute the principal results

```bash
export PYTHONPATH="$PWD/src:$PWD/vendor/spikingjelly"
python scripts/evaluate_main_results.py \
  --data-root "$PWD/.data" \
  --device cuda:0
```

Use `--profiles bio` for only the decreasing checkpoint. The first run builds five deterministic caches in the original single-generator order. `--rebuild-cache` forces regeneration. `outputs/inference/main_results.csv` includes both inferred and checkpoint-recorded accuracy for each split.

## 7. Accounting and limits

1. Patent Table 1 uses five values from one `best.pt` per profile, never a separate peak for each split.
2. `training_peak_metrics.csv` documents an expired plot convention and is not Table 1 evidence.
3. All available checkpoints use seed 0. The decreasing profile leads the five current comparisons, without a multi-seed significance test.
4. Uniform B is 0.272484 and only approximately matches 0.271875; fixed B is zero.
5. Paired-replay spike rates and PCA are descriptive, not standalone causal evidence.

## 8. Before upload

```bash
bash scripts/run_quick_demo.sh
export PYTHONPATH="$PWD/src:$PWD/vendor/spikingjelly"
python scripts/check_inference_runtime.py
git status --short
```

`.data/`, `vendor/spikingjelly/`, virtual environments, and generated caches are Git-ignored.
