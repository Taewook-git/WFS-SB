# PhaseFuse-RC12 frozen method and cached selector gate

Status: frozen primary method for the next dev20 exact-decode MLLM gate.
Base revision: 3ec7dc57a21bcacec4eb40be6138bb2d5142c01a.
Freeze date: 2026-08-13 Asia/Seoul.

## Scientific claim

PhaseFuse-RC means sampling-phase-robust canonical resampling. It does not mean
wavelet fusion. The main contribution is a two-stage scout -> canonical
decision -> fresh resample pipeline with a risk-controlled adaptive token
budget. TI-DWT is diagnostic-only and has zero weight in the frozen decision.

## Frozen RC12 decision

Inputs are one query-conditioned scout record (strictly increasing requested
timestamps and aligned BLIP2 query-relevance scores) plus an origin-independent
shared support interval for the item.

1. Shared support:
   - VideoMME dev20: the manifest common_valid_support_sec shared by all five
     outer origins for that video.
   - QVHighlights cached gate: [max_o first_scout_time_o,
     min_o last_scout_time_o] per video/question. The resulting origin-zero
     integer ticks are 1,...,149 seconds for the 150-second records.
2. Canonical physical-time lattice:
   - fixed origin 0.0 seconds;
   - scout cadence = median adjacent requested-scout timestamp difference;
   - decision step = max(scout cadence, 0.5 seconds);
   - take every origin-zero tick within the closed shared support.
   - Therefore VideoMME uses 0.5-second ticks and QVHighlights uses 1-second
     ticks. No scout-origin offset appears in the decision coordinates.
3. Score:
   - linearly interpolate raw query relevance from the scout timestamps to the
     canonical lattice;
   - apply tie-aware empirical percentile ranks;
   - Gaussian smooth in physical time with sigma=4.0 seconds and nearest-edge
     padding;
   - quantize downward as floor(score / 0.1) * 0.1.
   - No visual features, MMR, TI-DWT, phase consensus, or uncertainty enter the
     frozen score.
4. Coverage scaffold:
   - K=16 total and A=12 anchors;
   - place targets at the 12 equal-width temporal-bin centers spanning the
     first and last canonical lattice ticks;
   - snap each center to the nearest still-unused canonical tick; exact
     midpoint ties choose the earlier tick.
5. Four adaptive residuals:
   - candidates are all non-selected canonical ticks;
   - greedily require at least 2.0 seconds from every already selected token;
   - maximize lexicographically: quantized score, then the nearest-selected
     distance capped at 8.0 seconds, then earlier timestamp;
   - if no candidate satisfies 2.0 seconds, relax distance only for that token
     and keep the same score/coverage/earlier ordering;
   - exact K=16 is mandatory.
6. Canonical uniform control:
   - same support, lattice, exact decode, and K=16;
   - A=16 and no residual tokens.

## Mandatory second-stage decode contract

The selector returns 16 canonical target timestamps, never scout indices or
scout actual PTS. The source video must be decoded again against those exact
targets with the sequential nearest-PTS decoder. At an exact midpoint the
earlier PTS wins. The confirmatory trace must record both target timestamp and
fresh actual PTS.

It is prohibited to map a target to the nearest already-decoded scout frame.
That would reintroduce the outer-origin perturbation and invalidate the cached
selector estimand.

Fresh matches are deduplicated by decoded_frame_index. If a duplicate occurs,
keep the target with smaller absolute decode error (then earlier target on a
tie), take the next not-yet-selected canonical candidate under the frozen
residual priority, and fresh-decode that backup. Repeat until 16 unique frames;
abort the cell with provenance if the lattice is exhausted. This rare repair
must be counted and reported. No origin-dependent scout frame may be used as a
repair.

## Cached selector results

All intervals below are 10,000-draw dataset/video cluster bootstraps with seed
20260813. QV has 100 video clusters; VideoMME dev20 has 20 video clusters.

VideoMME dev20 selector timestamp F1 at 0.5 seconds:

| Method | Mean | Worst |
| --- | ---: | ---: |
| Canonical uniform | 1.00000 | 1.00000 |
| PhaseFuse-RC12 relevance-only | 0.97219 | 0.94375 |
| Matched dense-global SWT | 0.88458 | 0.81354 |
| PhaseFuse-v2 | 0.88042 | 0.80833 |

RC12 minus matched dense-global:
- mean F1 +0.08760, 95% CI [0.07552, 0.10021];
- worst F1 +0.13021, 95% CI [0.11250, 0.14792].

PhaseFuse-v2 minus matched dense-global:
- mean F1 -0.00417, 95% CI [-0.00990, 0.00167];
- worst F1 -0.00521, 95% CI [-0.01771, 0.00625].

QVHighlights 100-video cached gate:

| Method | Mean F1 | Worst F1 | Relevant fraction | Clip recall |
| --- | ---: | ---: | ---: | ---: |
| Canonical uniform | 1.00000 | 1.00000 | 0.16063 | 0.21588 |
| PhaseFuse-RC12 relevance-only | 0.90513 | 0.84750 | 0.29525 | 0.41771 |

RC12 minus canonical uniform:
- selected relevant fraction +0.13463, 95% CI [0.11600, 0.15287];
- relevant clip recall +0.20182, 95% CI [0.17272, 0.23133];
- mean saliency vote +0.46508, 95% CI [0.40120, 0.52721];
- GT nearest-token distance -0.63323 seconds, 95% CI
  [-0.76804, -0.49357].

PhaseFuse-v2 and matched dense-global are unavailable on cached QV100 because
those records contain 1 Hz single-stream scouts, not the required four
interleaved physical phases. No value is imputed.

## Falsification and TI-DWT negative diagnostics

A score blend of 0.9 smoothed relevance + 0.1 TI-DWT saliency was dominated by
relevance-only RC12:
- dev mean/worst F1: 0.95469/0.92188 versus 0.97219/0.94375;
- QV relevant fraction: 0.29275 versus 0.29525;
- QV clip recall: 0.41505 versus 0.41771.
It is not a method arm.

Exactly one cross-phase diagnostic gate was evaluated on dev20. Four
interleaved phase streams received SWT saliency, phase-wise percentile scaling,
alignment to the canonical lattice, median consensus, and scaled-MAD
dispersion. The exploratory gate required consensus >= 0.5 and dispersion <=
the pooled dev20 q90 (=0.2299773813). It reduced mean/worst F1 from
0.97219/0.94375 to 0.94063/0.89271. Distance relaxation and gate-off fallback
were both 0/300 scout records. QV cannot honestly compute this four-phase gate.
The gate is rejected and TI-DWT remains diagnostic-only.

## Downstream dev20 gate

Run at most two exact-decode arms: PhaseFuse-RC12 and canonical uniform. Do not
run heldout before this gate.

Primary safety rules:
- accuracy non-inferiority: lower 95% video-cluster-bootstrap bound for
  RC12-minus-uniform must exceed -0.03;
- PAD safety: upper 95% bound for the RC12-minus-uniform pairwise answer
  disagreement increase must be below +0.03.

Report accuracy, robust accuracy, stable-correct fraction, pairwise answer
disagreement, origin accuracies, exact decode repair count, and per-video
cluster intervals. If either safety rule fails, there is no defensible
utility-safe ICASSP tokenizer claim from RC12.

Existing uniform_dense MLLM results (accuracy 0.5700, PAD 0.08333, robust
accuracy 0.5000) are context only. They are not a substitute for the new
canonical second-stage exact-decode uniform control.


## Cached input provenance

- phasefuse_videomme_dev20/preprocess/dense_signals.jsonl:
  cca842810d063f714ce3c6655baf672e22255b6c8714f2b363a56a7a4b6dce66
- phasefuse_videomme_dev20/preprocess/multiphase_manifests.jsonl:
  d493f6763ef0196196a8014d1006a219426bc2e013c1c5b687410a0d4e2378c3
- v2_selector_ablation/selection/dense_vs_phasefuse_v2_analysis.json:
  3aceeeb24748aa7c998d524995025ae344c6c51cba454ce4242bdb9d5ce082bc
- qvhighlights_policy_val100/origin_signals.jsonl:
  c00c44f081b4e37f608e722677ead777ddd7b5bb5beab997b3c5258657b59e78

