# Plan 151: NITP Representation Geometry Diagnostics — Modelless

**Date:** 2026-05-26
**Status:** ✅ COMPLETE
**Research:** 113 (NITP — Next Implicit Token Prediction)
**Classification:** 🔧 **Utility — no feature gate needed**
**Related:** Plan 061 (entropy anomaly), Plan 102 (TileRT), Plan 055 (MTP drafter)
**Verdict:** Modelless diagnostics to measure representation health of our LoRA models. Pure utility functions, no inference impact.

---

## Task Index

- [x] T1: `effective_rank()` — Representation Dimensionality Metric
- [x] T2: `avg_cosine_similarity()` — Anisotropy Metric
- [x] T3: `representation_geometry_report()` — Combined Diagnostic
- [x] T4: GOAT Proof — Baseline Measurement
- [x] T5: Integration with `DataProbe` (Plan 141)

## Why This Plan Exists

Research 113 (NITP) reveals that representation degeneration (low effective rank, high cosine similarity) is a measurable pathology of NTP training. Even without implementing NITP's training loss, we can monitor representation geometry as a **diagnostic** — if our LoRA-trained models show degeneration, that explains poor downstream quality and justifies adding NITP loss (riir-ai Plan 148).

This is the modelless first step: measure the problem before committing to the fix.

---

## Tasks

- [x] ### T1: `effective_rank()` — Representation Dimensionality Metric

```rust
/// Compute the effective rank of a set of hidden state vectors.
/// Uses entropy-based effective rank (Roy & Vetterli, 2007) from the
/// eigenvalue spectrum of the empirical covariance matrix.
///
/// High effective rank = healthy, diverse representations.
/// Low effective rank = degenerate, collapsed representations.
pub fn effective_rank(hidden_states: &[Vec<f32>]) -> f32
```

- Input: `&[Vec<f32>]` — batch of hidden state vectors from any layer
- Process: mean-center → covariance → eigenvalues → entropy-based rank
- Reuse: eigenvalue decomposition from `entropy_anomaly_detection` (Plan 061)
- Location: `src/data_probe/` (already has diagnostics infrastructure)

- [x] ### T2: `avg_cosine_similarity()` — Anisotropy Metric

```rust
/// Compute average pairwise cosine similarity between hidden states.
/// High similarity = anisotropic (degenerate), Low = isotropic (healthy).
pub fn avg_cosine_similarity(hidden_states: &[Vec<f32>]) -> f32
```

- Input: `&[Vec<f32>]` — batch of hidden state vectors
- Process: normalize → pairwise dot products → average
- Location: `src/data_probe/`

- [x] ### T3: `representation_geometry_report()` — Combined Diagnostic

```rust
/// Combined representation geometry report for a set of hidden states.
pub struct GeometryReport {
    pub effective_rank: f32,
    pub avg_cosine_sim: f32,
    pub layer_index: usize,
    pub n_tokens: usize,
    pub hidden_dim: usize,
}

pub fn representation_geometry_report(
    hidden_states: &[Vec<f32>],
    layer_index: usize,
) -> GeometryReport
```

- [x] ### T4: GOAT Proof — Baseline Measurement

Run `effective_rank()` and `avg_cosine_similarity()` on:
1. Random weights (before any training) — establish isotropic baseline
2. After NTP-only training (if we have a checkpoint) — measure degeneration
3. After LoRA training — measure if LoRA helps or hurts geometry

**GOAT threshold:** Effective rank > 0.5 * hidden_dim AND avg_cosine_sim < 0.7 for healthy representations.

- [x] ### T5: Integration with `DataProbe` (Plan 141)

Wire geometry metrics into the existing `DataProbe` infrastructure so they can be collected during benchmark runs without separate tooling.

---

## GOAT Proof Structure

```
G1: effective_rank() matches numpy reference on synthetic data (±0.01)
G2: avg_cosine_similarity() matches numpy reference on synthetic data (±0.001)
G3: Random init → effective_rank > 0.8 * hidden_dim (isotropic)
G4: After training → effective_rank still > 0.3 * hidden_dim (not collapsed)
G5: GeometryReport integrates with DataProbe pipeline
```

---

## Scope

| Item | Included | Notes |
|------|----------|-------|
| Feature gate | ❌ Not needed | Pure utility, no side effects |
| New dependencies | ❌ None | Eigenvalue decomposition already in tree |
| Inference impact | ❌ None | Diagnostics only, not called in hot path |
| Training impact | ❌ None | Modelless analysis |
| Game AI impact | ❌ None | Generic diagnostics |

---

## What This Enables

1. **Immediate:** Measure whether our LoRA models have representation degeneration
2. **If degeneration found:** Justifies riir-ai Plan 148 (NITP training loss)
3. **If no degeneration:** Validates our current training recipe, no NITP loss needed
4. **Ongoing:** Monitor representation geometry across training runs

---

## Cross-References

- **Research 113:** Full NITP distillation with verdict
- **riir-ai Plan 148:** NITP LoRA training auxiliary loss (model-based, feature-gated)
- **Plan 061:** Entropy anomaly detection (reuses eigenvalue decomposition)
- **Plan 141:** Data probes (integration target)
- **27_mmo_goat_pillars_decision_matrix.md:** NITP is a LoRA bet (Secret A)
