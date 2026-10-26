# Bench 954 — kv960 T0b + T1: real-forward KV-sensitivity capture, ranking, and failure-mode classification

**Status:** COMPLETE — T0b + T1 landed; the T3-shaped structural-correlation finding + the T1 mode classification recorded.
**Issue:** [Issue 960](../HISTORY.md) (closed 2026-10-24, record in HISTORY.md; T0b) · Research 633 (arXiv:2606.09864, the KV-quant alignment-collapse law) · instrument: Bench from T0a (`ec7d3e2af`).

## T1 — failure-mode classification (same harness, second test)

`kv960_t1_failure_mode_classification` classifies each head's sensitivity
profile into the paper's three PCR modes (translated to non-safety
vocabulary) via the sign of ρ(damage, channel-range) at τ = 0.4, plus the
model-level layer-mass delocalization index `(PR − 1)/(n − 1)` (PR =
(Σm)²/Σm² ∈ [1, n]; 1 = flat = diluted, n⁻¹-normalized 0 = one hot layer).

| layer | kvh | ρ(damage,range) | mode (full) |
|---|---|---|---|
| 0 | 0 | 0.532 | behavior-in-outliers |
| 0 | 1 | 0.341 | diffuse |
| 0 | 2 | 0.450 | behavior-in-outliers |
| 1 | 0 | 0.329 | diffuse |
| 1 | 1 | 0.362 | diffuse |
| 1 | 2 | 0.571 | behavior-in-outliers |
| 2 | 0 | 0.553 | behavior-in-outliers |
| 2 | 1 | 0.415 | behavior-in-outliers |
| 2 | 2 | 0.697 | behavior-in-outliers |
| 3 | 0 | 0.415 | behavior-in-outliers |
| 3 | 1 | 0.400 | behavior-in-outliers |
| 3 | 2 | 0.709 | behavior-in-outliers |

**Verdicts: modes 9 behavior-in-outliers / 0 quiet-channels / 3 diffuse of 12
heads; layer masses [92.3, 94.2, 95.6, 98.8] → delocalization index 0.999 =
DILUTED (mode 3); mean ρ(damage,range) 0.481 (pinned > 0.3); ρ half-split
rank stability 0.804 (pinned > 0.5); sub-budget (10-prompt) mode agreement
6/12 — reported, not gated: borderline heads at τ flip under a half corpus,
the paper's budget classifies ONCE at full corpus.**

Reading (honest):

- **behavior-in-outliers 9/12 with positive mean ρ is the EXPECTED mechanical
  outcome on random weights** — the per-channel uniform quantizer's step is
  range/(2^bits − 1), so damage tracks range whenever score sensitivity is
  roughly channel-uniform. This head-count is the negative control behaving
  as expected; the classifier's discriminating power (the quiet-channels
  mode, T2's justifying case) needs trained models where damage and range
  dissociate. 0/12 quiet-channels here says nothing about trained models.
- **Dilution (mode 3) fires: the damage mass is spread flat across layers**
  (92–99 per layer, ~7% spread) — per-layer fixes fail; allocation must
  spread cross-layer. This is a property of the mass SHAPE on this model
  class; re-read on trained models.
- **outlier-crushes-behavior (paper mode 1) is EXCLUDED BY CONSTRUCTION on
  the reduction instrument**: the per-channel quantizer gives every channel
  its own min–max grid, so no shared scale exists for outliers to stretch.
  That mode needs the attribution path over a real backend store (T2/T3) —
  and a per-channel-scale backend IS the mitigation for it, which the
  reduction instrument already bakes in. Stated in the classifier output,
  not silently dropped.
- The mitigation map emitted per mode: behavior-in-outliers → finer
  granularity on the damaged channels; behavior-in-quiet-channels →
  range-blind behavioral-damage weighting (T2's lever); diffuse →
  allocation-spread axes; outlier-crushes → per-channel scales
  (attribution-mode lane).

## What ran

~20 seeded prompts (seq 40) through the REAL forward (`forward_base`, the
f16-reference half of the quant path) with the new `kv_capture` sink installed;
per (layer, kv head) the captured Q/K/V feed the T0a probe (reduction mode,
bits 2, top_k 1, GQA query groups packed per kv head); channels ranked by
accumulated behavioral damage; the two T0b questions answered:

1. **Rank stability (G1 analog)** — Spearman between the two disjoint
   10-prompt halves' accumulated-damage ranks, per (layer, kv head).
2. **Structural correlation (the T3 finding)** — Spearman between the K-side
   behavioral damage rank and the shipped structural weighting projected per
   raw channel (`Σ_i bits_i · R[i][c]²`, the calibrated eigenbasis'
   bit-weighted column energy; two-regime ladder b_high/b_low from the shipped
   `BitAllocator::allocate` at d_eff, avg 4 bits; calibration on the head's
   full 800-sample captured key population — the shipped `from_keys` input
   semantics). Reported, never gated: "correlate or not" IS the finding.

Harness: `tests/kv960_t0b_capture.rs` (required-features
`kv_capture,kv_sensitivity,domain_latent`). Config: 4 layers × 6 query heads
over 3 kv heads (2:1 GQA), head_dim 16, n_embd 96, vocab 128, block 48.
Weights seeded (`Rng::new(960)`), prompts seeded (base 1000) — deterministic.

## Results (--release; debug BYTE-IDENTICAL — the fixed-order-fold determinism holds across profiles)

| layer | kvh | K stab | V stab | K↔structural |
|---|---|---|---|---|
| 0 | 0 | 0.832 | 0.918 | 0.268 |
| 0 | 1 | 0.900 | 0.632 | -0.274 |
| 0 | 2 | 0.726 | 0.676 | -0.082 |
| 1 | 0 | 0.703 | 0.462 | 0.303 |
| 1 | 1 | 0.803 | 0.594 | -0.082 |
| 1 | 2 | 0.844 | 0.874 | -0.141 |
| 2 | 0 | 0.815 | 0.709 | 0.438 |
| 2 | 1 | 0.809 | 0.894 | -0.579 |
| 2 | 2 | 0.924 | 0.471 | 0.421 |
| 3 | 0 | 0.756 | 0.671 | -0.376 |
| 3 | 1 | 0.888 | 0.388 | -0.153 |
| 3 | 2 | 0.935 | 0.759 | -0.200 |

**Verdicts: mean K stability 0.828 · mean V stability 0.671 · mean K↔structural correlation −0.038.**

## Reading

- **K-side rank stability 0.828** (per-head range 0.703–0.935): two disjoint
  prompt halves rank the same channels fragile — the ~20-prompt budget is
  already rank-stable at this shape, the paper's calibration-budget premise
  reproduces. Pinned in the harness at `> 0.5` (margin 0.33).
- **V-side stability 0.671** (range 0.388–0.918): stable on average, noisier
  per head — the context-distortion metric is a mean over query weights
  (uniform here) and concentrates less than the top-k overlap damage.
  Pinned at `> 0.5` (margin 0.17).
- **K↔structural −0.038, straddling zero** (per-head −0.579..+0.438): on this
  model the behavioral damage ranking is UNCORRELATED with the shipped
  structural (eigenbasis bit-weighted) weighting. This is the T3-shaped
  dissociation in miniature: structural/eigen-energy concentration does NOT
  predict behavioral fragility — which is exactly the precondition that makes
  T2's behavioral-damage weight input informative (if structural already
  captured behavior, T2 would be redundant). Recorded as a finding about THIS
  model class; the trained-model leg is the honest caveat below.

## Honest caveats

- **Synthetic weights.** The runner is the issue's named candidate (the
  forward the quant backends' tests exercise) with seeded random
  `TransformerWeights` — this architecture has no trained-checkpoint loader
  in-tree. The stability result (probe ranks are prompt-stable) is a property
  of the instrument + the forward; the STRUCTURAL-DISSOCIATION result is a
  property of this model instance and must be re-read on a trained model
  before it is quoted as evidence about the law (T3's real eval leg). The
  per-(model, backend, bits) ranking cache the issue names starts here:
  re-running the harness with a different seed/config IS the per-model run.
- **Uniform V weights.** The V probe ran with `weights: None` (uniform
  1/seq) — the forward does not expose its softmax attention weights (and
  must not: the sigmoid-not-softmax law keeps the instrument pre-softmax).
  A caller with real attention weights can supply them through
  `ValueTensors::weights`; the harness does not.
- **Probe bits = 2 only.** The paper's collapse regime; the bit ladder
  (T2/T3's phase-transition localization) is future work on the same harness.

## Box state

M3 Max (16 cores), AC power, ~25% CPU at start with sibling cargo lanes
light-active (no compute-saturation — Plan-337 not triggered; this is a
calibration diagnostic, not a perf claim; issue 960 G4 exempts the probe).
Runtime 0.11 s (--release), 1.45 s (debug). Isolated target dir
`/tmp/kv960b-target`, removed after the run.

## Gates

- clippy `-D warnings`: `katgpt-forward` at BOTH postures (default =
  capture code compiles to nothing; `kv_capture` on) + the root test target
  at `kv_capture,kv_sensitivity` — all clean.
- Test green in both profiles, byte-identical verdict lines (determinism).
- Docs gates: count_features ✓ (713 = unique-name union; the root
  `kv_sensitivity` forwarding row shares katgpt-spectral's T0a name and
  counts once — the catalog gained a `kv_capture` §68 row),
  cargo_comment_audit ✓ (468 comments, 0 mismatches), markdown_fence ✓,
  required_features_static_gate ✓ (912 rows).
- Default posture: `cargo check -p katgpt-forward` clean (gated field
  compiles out; `None` costs one branch per layer when on).
