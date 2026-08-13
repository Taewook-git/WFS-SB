# PhaseFuse nested-R2 frozen scientific contract

## Claim and scope

Nested-R2 tests one narrow hypothesis: a query-adaptive selector can recover
local relevance while remaining a small, auditable perturbation of the exact
canonical-uniform K=16 control.  It is a canonical resampling method, not a
TI-DWT or multi-phase fusion claim.  TI-DWT, visual features, MMR, uncertainty,
phase consensus, model answers, and annotation outcomes are absent from the
decision rule.

The method was designed from code contracts only.  No RC14 outcome was read or
used to choose this rule.  The commit and the SHA-256 of
`configs/nested_r2_frozen.yaml` must be recorded before opening any future
confirmation outcome.

## Exact decision rule

1. Recreate the frozen RC12 origin-zero canonical lattice and its relevance
   signal exactly: interpolation, tie-aware percentile rank, 4 s Gaussian
   smoothing, and downward 0.1 quantisation are shared code, not a second
   implementation.
2. Construct the exact canonical-uniform A16 target indices.  These are the
   sole starting decision.
3. Assign every lattice tick to its nearest A16 timestamp.  A midpoint tie is
   owned by the earlier A16 slot.  These 16 disjoint cells are the fixed
   coverage regions.
4. Only interior slots 1 through 14 may donate a target.  A candidate must:
   stay in its donor's region; differ from the donor by at least 2 s; remain at
   least 2 s from every other A16 target; and have strictly higher quantised
   relevance than its donor.  There is no distance relaxation.
5. The best candidate per donor uses the frozen RC12 ordering: quantised
   relevance, distance to retained A16 capped at 8 s, then earlier time.
6. Enumerate the empty set, every singleton, and every non-adjacent pair of
   donor proposals.  Choose the set with maximum summed quantised gain.  An
   exact gain tie chooses fewer changes, then summed relevance, summed capped
   coverage, earlier donor slots, and earlier candidates.
7. Fresh-decode the 16 physical-time targets from the source video using the
   unchanged canonical nearest-PTS decoder and dynamic duplicate repair.

"Preserve 14 slots" is a hard floor: the final decision shares at least 14
exact timestamps with A16.  When two safe positive-gain moves exist it shares
exactly 14; with only one or no safe move it deliberately shares 15 or 16.
Forcing two weak moves would contradict the `at most 2` safety contract.

## Machine-checked invariants

- exactly 16 distinct, increasing canonical targets;
- symmetric difference from A16 contains no more than two donor/residual pairs;
- at least 14 exact A16 timestamps remain;
- first and last A16 timestamps always remain;
- every original A16 Voronoi region contains exactly one final primary target;
- two donor slots are never adjacent;
- every actual relocation has strictly positive quantised gain and is at least
  2 s from its donor and all other original anchors;
- nested-R2 score arrays exactly equal frozen RC12 score arrays;
- constant relevance is an exact A16 no-op;
- provenance exposes base, donor, retained, and residual indices separately;
- exact source decoding accepts a truthful `nested_r2` role without changing
  its PTS, collision, repair, or batching policy.

## Leakage-free untouched validation

Before mapping cohort identifiers to examples, publish the code commit, config
hash, cohort-construction script/hash, exclusion manifest, endpoint order,
bootstrap seed, and stopping rule.  Exclude every VideoMME/QV video used in
RC12/RC14 development, selector diagnostics, prompt debugging, or manual
inspection.  Select the remainder by a fixed SHA-256 ordering over opaque video
IDs; never replace failed or inconvenient examples after outcomes are visible.

For VideoMME, run a paired two-arm grid on the same videos, questions, five
scout origins, source files, fresh decode, Qwen snapshot, prompt, and generation
arguments.  The baseline must be the exact A16 targets embedded in each
nested-R2 decision artifact, not an older uniform run unless pixel hashes prove
exact reuse.  Keep the existing strict safety gate unchanged:

- accuracy non-inferiority: video-cluster 95% CI lower bound for
  `nested_r2 - canonical_uniform` mean accuracy is greater than -0.03;
- PAD safety: video-cluster 95% CI upper bound for the pairwise answer
  disagreement increase is less than +0.03.

Report mean and robust accuracy, stable-correct rate, PAD, every origin's
accuracy, relocation rate, and accuracy stratified by 0/1/2 relocations.  Do not
claim utility safety if either gate fails.

For QVHighlights, freeze an outcome-unseen cohort with the same exclusion and
hashing procedure.  Use the same paired A16/nested targets and five origins.
The ordered selector endpoints are relevant-frame fraction first and clip
recall second, with video-cluster bootstrap CIs; test the second only if the
first has a lower 95% CI above zero.  Report saliency-vote delta, nearest-event
distance, relocation rate, and all origin-wise effects as secondary outcomes.
The deterministic coverage invariants are audited on every row but are not
substitutes for the statistical endpoints.

No parameter, threshold, cohort, endpoint, or decoding rule may change after
either dataset's confirmation outcomes are opened.  Null and negative results
remain part of the final report.
