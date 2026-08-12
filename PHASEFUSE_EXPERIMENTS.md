# PhaseFuse ICASSP experiment

PhaseFuse is a training-free, query-conditioned video tokenizer for a fixed
frame budget. TI-DWT/SWT is one internal saliency encoder; it is not the main
contribution. The method targets a failure that remains after transform-level
shift stabilization: changing the physical sampling origin can expose a
different set of RGB frames and therefore different evidence.

## Method under test

For every independently evaluated outer origin, PhaseFuse:

1. decodes one interleaved 4-fps lattice containing four physical 1-fps phases;
2. applies the temporal transform independently to each phase;
3. aligns evidence in absolute time and fuses it with a phase-permutation-
   invariant median, scaled-MAD uncertainty, and phase vote;
4. chooses event boundaries with a capacity-aware dynamic program constrained
   by the downstream budget; and
5. allocates and selects exactly 16 unique source frames with no empty segment.

With `K=16` and two frames reserved per segment, the method can retain at most
seven boundaries. This prevents the `B+1>K` allocation pressure observed in
the earlier QVHighlights policy study. Finite four-phase marginalization is not
claimed to provide perfect continuous shift invariance.

The five outer origins are evaluation perturbations and are never fused. The
four inner phases are method inputs and are rebuilt independently inside every
outer-origin cell. Mixing those two roles invalidates the experiment.

## Compute-matched arms

All arms receive the same dense decoded frames and cached query-conditioned
features, return exactly 16 unique source frames, and are evaluated on the same
five outer origins.

| Method | Purpose |
|---|---|
| `uniform_dense` | Dense-pool uniform baseline |
| `dense_topk_mmr` | Dense relevance plus MMR baseline |
| `single_dwt` | Legacy 1-fps DWT-WFS |
| `single_swt` | Legacy 1-fps TI-DWT/SWT-WFS |
| `multiphase_dwt` | Physical multi-phase acquisition with DWT |
| `multiphase_swt_mean` | Multi-phase SWT with mean fusion and no uncertainty penalty |
| `dense_swt` | Direct dense single-stream SWT compute control |
| `phasefuse` | Median/MAD/vote fusion plus budget-coupled DP and allocation |

The main novelty gate is `phasefuse` versus `dense_swt`. If PhaseFuse cannot
beat this arm, any gain can be explained by denser sampling rather than phase
fusion. `single_swt` isolates the value of TI-DWT alone, while
`multiphase_swt_mean` isolates the uncertainty-aware fusion component.

## Frozen protocol

- dense acquisition: 4 fps = four interleaved 1-fps phases;
- outer origins: 5 stratified deterministic offsets;
- frame budget: 16 unique source frames;
- feature model: BLIP-2 ITM;
- MLLM: Qwen2.5-VL-7B-Instruct, greedy decoding, BF16, SDPA;
- Qwen visual cap: `max_pixels=200704` for a 40-GiB A100 MIG slice;
- inference unit: video cluster, not question or origin;
- default bootstrap: 10,000 video-cluster replicates.

The development run uses VideoMME indices 0–19 and is not confirmatory. Do not
choose hyperparameters, arms, or endpoints using a held-out run after looking
at its answers.

## One-command development run

The existing server environment can run:

```bash
cd ~/WFS-SB
git pull --ff-only origin phase-stable-icassp
bash scripts/run_phasefuse_experiment.sh 2>&1 | tee phasefuse_dev20.log
```

On a fresh A100 host, first prepare the pinned environment and patched
`lmms-eval` checkout without exposing the token on the command line:

```bash
cd ~/WFS-SB
read -rsp 'HF token: ' HF_TOKEN && echo
export HF_TOKEN
bash scripts/bootstrap_a100.sh --datasets videomme --dataset-check full
unset HF_TOKEN
bash scripts/run_phasefuse_experiment.sh 2>&1 | tee phasefuse_dev20.log
```

The development grid contains 8 methods × 5 origins = 40 Qwen cells. Based on
the earlier 20-video timing, expect roughly 8–12 hours including dense feature
preprocessing on one A100 40-GiB allocation. The launcher is foreground and
resume-safe; re-running the same command validates artifacts and skips valid
cells.

For a cheap selector-only check:

```bash
bash scripts/run_phasefuse_experiment.sh \
  --run-dir artifacts/phasefuse_videomme_selector_smoke \
  --video-indices '0' --skip-mllm
```

For a Qwen pipeline smoke, use an isolated directory. `--limit` intentionally
skips the final full-cohort strict join:

```bash
bash scripts/run_phasefuse_experiment.sh \
  --run-dir artifacts/phasefuse_videomme_qwen_smoke \
  --video-indices '0' --limit 1
```

## Confirmatory run

Freeze a held-out, duration-stratified VideoMME list that excludes every video
used in development. Keep the list under version control before inference.
For example:

```bash
HELDOUT_INDICES="$(tr '\n,' '  ' < protocols/videomme_phasefuse_heldout_indices.txt)"
bash scripts/run_phasefuse_experiment.sh \
  --run-dir artifacts/phasefuse_videomme_confirmatory \
  --video-indices "${HELDOUT_INDICES}" \
  2>&1 | tee phasefuse_confirmatory.log
```

Recommended primary decision rule:

- PhaseFuse has lower pairwise answer disagreement than the strongest
  compute-matched baseline (`dense_swt`); and
- mean answer accuracy satisfies a predeclared non-inferiority margin, e.g.
  lower confidence bound greater than −2 percentage points.

Report robust accuracy, worst-origin accuracy, stable-correct, stable-wrong,
selected-set consistency, and QV evidence fidelity as secondary or diagnostic
endpoints. A lower answer disagreement alone is not success because a method
can be stably wrong.

## QVHighlights fidelity audit

QVHighlights has no configured `lmms-eval` QA task. Run selector/fidelity
analysis only, always in its own benchmark-specific artifact root:

```bash
bash scripts/run_phasefuse_experiment.sh \
  --benchmark qvhighlights \
  --questions-file datasets/qvhighlights/highlight_val_release.jsonl \
  --dataset-root datasets/qvhighlights \
  --run-dir artifacts/phasefuse_qvhighlights_dev \
  --video-indices '0 1 2 3 4' \
  --skip-mllm
```

The QV metrics are sparse selected-frame evidence diagnostics, not official
highlight mAP or moment-retrieval R@IoU unless those official evaluators are
run separately.

## Artifact contract and recovery

The run root contains:

```text
preprocess/
  dense_signals.jsonl
  multiphase_manifests.jsonl
  source_video_bundle.jsonl
  feature_bundle.jsonl
  visual_features/*.npy
selection/
  traces.jsonl
  trace_array_bundle.jsonl
  trace_arrays/*.npz
  analysis_summary.json
keyframes/videomme_<method>_originNN.json
mllm/videomme/<method>/originNN/.complete
predictions.jsonl
phasefuse_downstream_summary.json
mllm_runtime_provenance.json
```

Stage markers validate exact SHA-256 values. Feature and trace bundles validate
every material array, and MLLM cell markers validate keyframes, result/sample
logs, code/runtime signature, package versions, and the resolved Hugging Face
snapshot revision. A failed cell receives no valid marker. Never use
`--force-preprocess` or `--force-selection` merely to resume an interrupted
run; re-run the same command first.

## Claims allowed by this experiment

If the confirmatory gates pass, the defensible claim is that marginalizing
multiple physical sampling phases, using phase disagreement as uncertainty,
and coupling event cardinality to a fixed token budget improves
sampling-origin robustness while preserving task accuracy.

Do not claim that SWT universally creates more boundaries, universally detects
better boundaries, or by itself improves video understanding. Earlier
VideoMME and QVHighlights results already show that the direction of raw peak
and boundary-count changes is dataset and policy dependent.
