# Bench 955 — kv960 T2: behavioral-damage-weighted bit allocation GOAT

**Status:** COMPLETE — GOAT PASS (G1 + G2 + G3-compile + G4); stays opt-in (random-weight caveat; the trained-model read is T3).
**Issue:** [`.issues/960`](../.issues/960_kv_quant_behavioral_subspace_poc.md) (T2) · Research 633 (arXiv:2606.09864) · instrument: Bench 954 (T0b + T1 at `3b5bc8faf`), T0a (`ec7d3e2af`).

## What shipped

- **`crates/katgpt-spectral/src/kv_sensitivity_alloc.rs`** (opt-in
  `kv_sensitivity_alloc`, implies `kv_sensitivity`):
  - `behavioral_waterfill_weights(damage, probe_bits)` — the weight vector
    `D_c = damage_c · 4^{b₀}`: the probe damage extrapolated to the bit-0
    scale under the uniform min–max error model (`damage_c(b) ≈ D_c/4^b`),
    so the SHIPPED water-fill (`waterfill_bits`, unchanged) greedy-equalizes
    predicted damage `Σ D_c/4^{b_c}` — one weight input, no new search, no
    new quantizer. Scale-invariant (only the ranking carries information;
    unit-pinned).
  - `allocate_bits(weights, avg_bits, min_bits, max_bits) -> BitTable` —
    the budget arithmetic + contract asserts (finite non-negative weights,
    all-zero refuses loud, `min ≤ avg ≤ max`, `max ≤ 24`) + the equal-bits
    law asserted (`Σb_c == round(avg·d)` exactly — the GOAT premise).
  - `BitTable` — the per-channel `u8` decode-path surface; `bits(c)` is an
    O(1) index read (G4).
- **`joint_key_read`** in `kv_sensitivity` (the GOAT read; T3 reuses it) —
  the AGGREGATE behavioral damage when ALL key channels are degraded AT
  ONCE at per-channel widths (the complement of the per-channel marginal
  probe: score-space deltas interfere, joint ≠ sum). Shares the probe's
  reference pass (extracted `key_reference_pass`, one computation, both
  reads) and its exact tie laws; per-query delta accumulates in f64 over
  ascending channels (Issue-940 LOCAL-ONLY note at the site). 6 unit tests
  incl. the provable bimodal fixture at k=1 (damage 0.5 exactly) and k=2
  (0.25 exactly — one lost slot of two on the flipping half).
- Root forwarding feature `kv_sensitivity_alloc` + `[[test]]`
  `kv960_t2_damage_alloc` (required-features `kv_capture,kv_sensitivity_alloc,domain_latent`)
  + the G4 single-test binary `kv960_t2_bit_table_alloc` (spectral crate —
  the counting allocator is per-process, so the pin needs a binary with one
  test).

## The GOAT protocol

Three arms at EQUAL average bits across the low-bit ladder
[1.25, 1.5, 2.0, 2.5, 3.0] (head_dim 16 → every rung an exact integer
budget; min 1 / max 8):

1. **behavioral** — weights from the T0a probe's damage ranking over the
   20-prompt calibration corpus (the T0b seed world verbatim: weights
   `Rng::new(960)`, prompts base 1000, 4 layers × 6 q heads over 3 kv heads,
   head_dim 16, seq 40).
2. **structural** — the shipped weighting projected per raw channel
   (`Σ_i bits_i · R[i][c]²`, the T0b projection) over the SAME calibration
   population.
3. **uniform** — flat weights (the water-fill round-robins
   channel-ascending).

Retention read = `joint_key_read` over 10 HELD-OUT eval prompts (seeds
1500+ — calibration never doubles as the eval population; the paper's
protocol shape). Report-the-floor: every behavioral number prints beside
BOTH controls.

## Results (--release, M3 Max, battery 71% charging, ~82% CPU with sibling lanes — calibration diagnostic, not a perf claim)

G1 alloc-rank stability (Spearman): nested first-half vs full **0.840**
(floor 0.5, the T0b nested shape — 0.828 there) · disjoint half-vs-half
**0.490** (disclosed, not gated: a 10-prompt disjoint resample really
carries this much rank noise; the paper's budget classifies ONCE at full
corpus — T1's sub-budget disclosure precedent).

| avg_bits | total | behavioral | structural | uniform | flip(behav) |
|---|---|---|---|---|---|
| 1.25 | 20 | **0.6619** | 0.6721 | 0.6654 | 0.6619 |
| 1.5 | 24 | **0.6116** | 0.6305 | 0.6252 | 0.6116 |
| 2.0 | 32 | 0.4166 | 0.4166 | 0.4166 | 0.4166 |
| 2.5 | 40 | **0.3159** | 0.3374 | 0.3401 | 0.3159 |
| 3.0 | 48 | 0.2001 | 0.2001 | 0.2001 | 0.2001 |

**Verdict: G2 PASS — behavioral ≤ BOTH controls at EVERY rung (dominance),
max per-rung win 0.0215 damage (2.15 pp of top-1 target overlap) over the
better control at rung 2.5 (margin floor 0.02, deterministic seeds — no
flake surface). Bits-equivalence is DEGENERATE at the integer rungs and
that is the honest finding: with a moderate weight spread the water-fill
round-robins `+1` to everyone before doubling anyone, so at 2.0/3.0 all
three arms converge to the same `[2;16]`/`[3;16]` allocation and equal
bits-equals equal damage. The win lives exactly where allocation
granularity exists — the fractional rungs — and it favors behavioral at
every one of them.**

`flip(behav) == damage` at every rung: with `top_k = 1` the damage metric
IS the pick-flip rate (consistency check, not a coincidence).

## Gates

- **G1** — joint read repeat-identical (unit-pinned + harness-pinned);
  nested resample stability 0.840 ≥ 0.5; equal-bits across arms asserted
  at every rung (`allocate_bits` asserts the total, the harness asserts
  cross-arm equality).
- **G2** — dominance at every rung + the strict win 0.0215 ≥ 0.02. A
  negative here would have been a finding about the model, never the
  instrument (the full curve prints either way — it did not need to).
- **G3** — feature-off compiles clean (`katgpt-spectral` default posture
  clippy `-D warnings` ✓; the feature gates NEW code only — no existing
  path changes, the lossless path is untouched); T0b/T1 re-run green after
  the `key_reference_pass` extraction (byte-identical behavior).
- **G4** — 0 allocs over 10 000 `BitTable::bits` lookups (own single-test
  binary, counting allocator).
- Docs gates: count_features ✓ (714 = the new unique leaf name), catalog
  §`kv_sensitivity_alloc` row, cargo_comment_audit, markdown_fence,
  required_features_static_gate — all green (run below).

## Honest caveats

- **Random weights — the T1 caveat binds verbatim.** The behavioral
  ranking on this model co-locates with channel range (9/12
  behavior-in-outliers, Bench 954's negative control), and the structural
  arm is near-flat (isotropic-ish covariance) — so this GOAT validates the
  PROCEDURE (allocator + weight input + joint read), not the model-level
  law. The trained-model read (where damage and range dissociate, and the
  quiet-channels case can exist at all) is T3.
- **Reduction-posture read.** The joint read degrades channels with the
  per-channel uniform min–max quantizer — the per-channel-scale posture
  that IS the mitigation for the paper's outlier-crushes mode (excluded by
  construction here, Bench 954 §T1). A shared-scale backend's joint
  behavior is a T3 attribution-lane read over a real backend store.
- **K side only.** The three arms are well-defined on K (the shipped
  structural weighting is K-side); a V-side allocation has no structural
  control to compare against — deferred with reason, not silently dropped.
- **Calibration-class.** The probe + allocation are one-shot per allocation
  decision (G4-exempt by the issue's own wording); the decode-path surface
  is the `BitTable` lookup, which is the pinned 0-alloc leg.

## Box state

M3 Max (16 cores), battery 71% (charging), CPU ~82% with sibling cargo
lanes light-active (calibration diagnostic, not a perf claim; issue 960 G4
exempts the probe). Runtime 2.06 s (--release), 39.95 s (debug). Isolated
target dir `/tmp/kv960t2-target`, removed after the run.
