# Patent Technical Content and Inference Evidence Map

This map links paragraph numbers in the reviewed technical draft to public code and data. The patent document is excluded because it contains applicant and inventor information. As requested, this package supplies principal-result inference rather than training source.

| Patent location | Technical subject | Inference code | Data or runnable evidence |
| --- | --- | --- | --- |
| Claim 1, [0013]–[0018] | Within-layer normalization, exponential mean normalization, clipping, profile coefficients, budget | `src/patent_snn/model.py`: `PROFILE_SCALES`, `compute_tau()` | checkpoint audit and tau CSVs |
| Claim 2, [0020]–[0021] | PLIF membrane update, threshold, reset | `ProfilePLIFNode.forward()` | checkpoints, strict load, replay |
| [0019], [0037]–[0040] | Last-K spike aggregation and online linear classification | `forward_spike_repr()`, checkpoint loader | `last_k_steps`, probe, TensorBoard |
| [0034]–[0036], [0041] | Four convolutional layers, 16 frames, channels, pooling | `src/patent_snn/model.py` | embedded checkpoint `cfg` and state |
| [0042]–[0047], Fig. 3 | Trained tau distributions and budget controls | audit and plot scripts | Figure 3 and tau CSVs |
| [0049]–[0051], Fig. 4 | Last-four-step 128-D representation and 128→10 probe | model and strict loader | checkpoint probe and `last_k_steps` |
| [0052] | Motion, flicker, noise, sparse visibility | `src/patent_snn/data.py` | replay NPZ and software demo |
| [0053]–[0055], Table 1 | Four profiles under five conditions | full evaluator and audit | checkpoints, events, metrics CSV |
| Fig. 5 | Inputs, measured tau, spike response, final representation | replay and plotting scripts | Figure 5 and replay NPZ |
| [0056]–[0058] | Fixed channel tau during inference | model and loader | general CPU/GPU inference; no hardware measurement |

Primary evidence is the four `best.pt` files and their TensorBoard records. Rebuildable evidence includes same-checkpoint metrics, tau tables, the paired replay, figures, and media. Checkpoints retain their training configuration, projection head, and optimizer state for identity inspection; inference strictly loads only `model` and `probe`. SupCon loss, BPTT, optimizers, and training samplers are outside this package's executable scope.

The inference math and stimulus generator were extracted from the newest compatible implementation reviewed in July 2026. The older workspace checkout, ordinary cross-entropy runs, ten-epoch last-membrane runs, and later dendritic, tri-compartment, and task-conditioned architectures are excluded. `training_peak_metrics.csv` is retained only to document an expired split-wise-peak plot convention and is never used for Patent Table 1.

All five values in a table row come from one checkpoint selected by standard-test accuracy. Only seed 0 is available. Decreasing and reverse exactly share the nominal budget; uniform is approximate; fixed has zero budget. The paired replay is descriptive and does not establish causal layer function. Multi-seed statistics, selection-independent holdout evaluation, causal intervention, and neuromorphic-hardware measurements remain absent.
