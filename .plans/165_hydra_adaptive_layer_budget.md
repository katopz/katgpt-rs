# Plan 165: Hydra-Aware Adaptive Layer Budget

> **Research:** 148 (The Hydra Effect: Emergent Self-Repair)
> **Source:** [arXiv:2307.15771](https://arxiv.org/pdf/2307.15771) — McGrath et al. (DeepMind)
> **Feature Gate:** `hydra_budget` (default-OFF until GOAT proves gain)
> **Priority:** Medium — composable enhancement over existing early_exit system
> **Date:** 2026-06-01

---

## Summary

Distill the Hydra Effect (emergent self-repair in transformers) into an adaptive layer budget that skips non-contributing layers during the forward pass. The paper shows that transformer layers are loosely coupled: most layers have negligible direct effect on output logits, and ablations are compensated by specific backup layers. This means we can skip non-critical layers without quality loss.

Two modes:
- **Modelless**: Pre-computed layer importance profiles (lookup table, zero overhead)
- **Model-based**: Per-layer logit lens scoring during forward pass (one matmul per layer)

---

## Architecture

```text
Forward Pass
  │
  ├── For each layer l:
  │     ├── Compute residual z^l = z^{l-1} + a^l + m^l
  │     ├── [model-based] Logit lens: score_l = RMSNorm(z^l) @ W_U  (centered, top token)
  │     ├── [modelless] Lookup: score_l = profile[l].importance
  │     └── Skip decision:
  │           if |score_l| < threshold && !is_backup[l]:
  │             skip (z^l = z^{l-1}, zero-out a^l + m^l)
  │
  ├── Adaptive depth gate:
  │     if cumulative |DE| > 0.95 * total |DE|:
  │       early-terminate (remaining layers contribute < 5%)
  │
  └── Final logits = RMSNorm(z^L) @ W_U
```

### Types

```rust
/// Per-layer Hydra profile entry (modelless mode).
/// Pre-computed from calibration data, stored in config.
#[derive(Clone, Debug)]
pub struct HydraLayerProfile {
    /// Mean absolute direct effect on top-token logit.
    pub mean_de: f32,
    /// Fraction of prompts where this layer is a Hydra backup.
    pub backup_frequency: f32,
    /// Whether this layer acts as erasure (mean DE < 0 for MLP).
    pub is_erasure: bool,
}

/// Hydra budget configuration.
pub struct HydraBudgetConfig {
    /// Skip layers with |DE| below this threshold.
    pub skip_threshold: f32,
    /// Use modelless mode (lookup) vs model-based (logit lens).
    pub modelless: bool,
    /// Skip erasure MLPs during draft stage.
    pub skip_erasure_draft: bool,
    /// Early-terminate when cumulative DE reaches this fraction of total.
    pub cumulative_threshold: f32,
}
```

---

## Tasks

### Phase 1: Infrastructure

- [x] T1: Add `HydraLayerProfile` and `HydraBudgetConfig` to `katgpt-core/src/types.rs`
- [x] T2: Add `hydra_budget` feature gate to `Cargo.toml`
- [x] T3: Add `HydraBudgetConfig` fields to `Config` and `InferenceOverrides`
- [x] T4: Add `Vec<HydraLayerProfile>` to `Config` (populated from calibration data or defaults)

### Phase 2: Modelless Layer Skip

- [x] T5: Implement `hydra_layer_skip()` function — given profiles and threshold, return set of layers to skip
- [x] T6: Integrate layer skip into `transformer.rs` forward pass — conditionally zero-out skipped layers
- [x] T7: Add profile calibration tool — run logit lens on calibration data, output `HydraLayerProfile` per layer
- [x] T8: GOAT proof P4 — modelless profile stability test (profiles are consistent across seeds)

### Phase 3: Model-Based Logit Lens

- [x] T9: Implement per-layer logit lens scoring — `score_l = centered_logits(RMSNorm(z^l) @ W_U)` for top token
- [x] T10: Implement adaptive depth gate — cumulative DE convergence detection
- [x] T11: GOAT proof P1 — layer skip correctness (cosine sim > 0.99 vs baseline)
- [x] T12: GOAT proof P3 — adaptive budget speedup (throughput gain > 0%)

### Phase 4: Erasure-Aware Draft

- [x] T13: Implement erasure detection — identify MLP layers with negative mean DE
- [x] T14: Integrate erasure skip into `DecodeStage::Draft` — skip erasure MLPs during draft only
- [x] T15: GOAT proof P2 — erasure skip improves draft acceptance rate
- [x] T16: End-to-end GOAT proof — all 4 proofs pass, acceptance rate within 2% of baseline

### Phase 5: Default-On Decision

- [x] T17: Run full benchmark suite with `hydra_budget` enabled
- [x] T18: If all GOAT proofs pass AND no perf regression → flip to default-on (GOAT passed: P1-P4 all ✓, now default-on)
- [x] T19: Update plan with completed tasks

---

## Optimization Constraints (per optimization.md)

1. **No allocation in hot path** — `HydraLayerProfile` is a fixed-size array indexed by layer number. No Vec lookup in the forward pass.
2. **Pre-compute profile once** — profiles are computed offline during calibration, stored in config struct. O(1) reads.
3. **Logit lens is one matmul** — `z^l @ W_U` is already the same pattern as the final logit computation. Reuse the same kernel.
4. **Modelless mode has zero overhead** — layer skip decision is a single `if` branch on a pre-computed f32 threshold.
5. **No Rayon for per-layer decisions** — layer processing is sequential in the forward pass. Each decision is ~1ns (compare f32).

---

## GOAT Proof Targets

| Proof | Metric | Target |
|-------|--------|--------|
| P1: Skip Correctness | Cosine similarity (skipped vs full) | > 0.99 |
| P2: Erasure Skip | Draft acceptance rate Δ | > 0% (improvement) |
| P3: Speedup | Throughput (tokens/sec) | > 0% (improvement) |
| P4: Profile Stability | Top-k overlap across seeds | > 80% |

---

## Dependencies

- `katgpt-rs-core` types (Config, InferenceOverrides)
- `transformer.rs` forward pass (layer loop)
- `data_probe` module (logit lens utilities)
- Existing `early_exit_patience` / `early_exit_gap` system (complements, doesn't replace)
