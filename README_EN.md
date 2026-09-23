# Morpho-Temporal Decoupling SNN

[中文 README](README.md) · [中文操作说明](docs/操作说明.md) · [English operation guide](docs/OPERATION_GUIDE.md)

This repository contains only the **inference code and evidence artifacts** required to reproduce the patent's principal results. It includes four best checkpoints, deterministic generation for five test conditions, the four-layer ProfilePLIF network, trained linear probes, full-test inference, a 120-sample paired replay, evidence audits, figures, and demo media. Training loops, optimizers, SupCon loss, training samplers, training YAML files, and architectures unused by these checkpoints are excluded.

![Software replay demo](media/software_run_demo.gif)

## Three runnable paths

| Entry point | Purpose | MNIST / SNN runtime | Main output |
| --- | --- | --- | --- |
| `bash scripts/run_quick_demo.sh` | Rebuild tables, tau values, figures, and media from checkpoints and TensorBoard | Not required | `outputs/audit/`, `outputs/figures/` |
| `bash scripts/run_full_replay.sh` | Re-infer 120 fixed digits and compare with the saved sample | Required | `outputs/replay/` |
| `python scripts/evaluate_main_results.py` | Re-run five 10,000-sample tests for all four checkpoints | Required | `outputs/inference/main_results.csv` |

CPU quick audit:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_quick_demo.sh
```

A successful run ends with `Quick demo PASS`.

## Inference architecture

```text
[B,16,1,64,64]
  → Conv/BN/ProfilePLIF(32)  → AvgPool
  → Conv/BN/ProfilePLIF(64)  → AvgPool
  → Conv/BN/ProfilePLIF(128) → AvgPool
  → Conv/BN/ProfilePLIF(128)
  → spatial and temporal mean of the last four spike frames → 128-D representation
  → trained 128→10 linear probe → digit class
```

The minimal implementation is under `src/patent_snn/`: `model.py` defines the four-layer network and tau mapping, `checkpoint.py` strictly loads the model and probe, and `data.py` implements deterministic evaluation stimuli.

The checkpoint still contains its training-time projection head and optimizer state as original records. Inference neither instantiates nor calls them. Runtime settings come from the embedded checkpoint `cfg`, avoiding drift from copied training YAML files.

## Profiles and principal results

| Profile | Layer coefficients | Nominal B | Standard | Speed | Flicker | Noise | Combined |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decreasing `bio` | `[.80,.55,.35,.15]` | 0.271875 | 96.81% | 94.59% | 95.84% | 95.70% | 93.97% |
| Uniform `equal` | `[.522,.522,.522,.522]` | 0.272484 | 95.20% | 92.87% | 93.65% | 94.14% | 91.84% |
| Reverse `reverse` | `[.15,.35,.55,.80]` | 0.271875 | 94.41% | 90.06% | 91.24% | 93.82% | 90.55% |
| Fixed `fixed` | `[0,0,0,0]` | 0 | 89.73% | 77.82% | 84.13% | 87.87% | 80.20% |

Each row uses all five scores from one selected checkpoint. The decreasing profile leads all five test results. The uniform profile is an approximate budget match.

## Recompute the full main results

```bash
bash scripts/setup_full_runtime.sh
python scripts/download_mnist.py
export PYTHONPATH="$PWD/src:$PWD/vendor/spikingjelly"
python scripts/evaluate_main_results.py --device cuda:0
```

The script strictly loads all four checkpoints, builds each deterministic test cache, and writes inferred and checkpoint-recorded accuracies. Five caches of 10,000 16-frame 64×64 videos require substantial runtime and about 13 GiB of free disk. Use `--profiles bio` to run only the decreasing checkpoint.

For the smaller paired replay:

```bash
bash scripts/run_full_replay.sh --device cuda:0
```

The comparison requires exact labels, indices, predictions, and input frames, with explicit tolerances for rare GPU boundary-spike differences.

## Evidence layout and scope

| Path | Purpose |
| --- | --- |
| `src/patent_snn/` | Minimal principal-result inference implementation |
| `checkpoints/` | Best checkpoints for the four profiles |
| `data/tensorboard/` | Original run records used for same-step verification |
| `data/derived/` | Table 1, tau, and historical peak-accounting data |
| `data/samples/` | Fixed 120-sample paired replay |
| `scripts/` | Audit, full evaluation, replay, plotting, and setup |
| `figures/`, `media/` | Patent figures and software-run media |
| `docs/` | Bilingual guides and patent evidence maps |

The inference math and stimulus generation were extracted from the newest verified compatible implementation updated in July 2026. The older workspace copy was excluded. `training_peak_metrics.csv` remains solely to explain an expired split-wise-peak plot; Patent Table 1 always uses same-checkpoint values. The patent draft itself is excluded because it contains applicant and inventor information.

## Change log

- 2026-09-23: assembled and verified the standalone evidence package.
- 2026-09-23: reduced the package to principal-result inference, added four-profile full evaluation, removed training and unused extension code, and refreshed bilingual documentation and hashes.
- 2026-09-23: adopted the standalone project name `Morpho-Temporal-Decoupling-SNN` for its dedicated GitHub repository.
