# Bench 954 — kv960 T0b: real-forward KV-sensitivity capture + ranking

**Status:** COMPLETE — T0b landed; the T3-shaped structural-correlation finding recorded.
**Issue:** [`.issues/960`](../.issues/960_kv_quant_behavioral_subspace_poc.md) (T0b) · Research 633 (arXiv:2606.09864, the KV-quant alignment-collapse law) · instrument: Bench from T0a (`ec7d3e2af`).

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
