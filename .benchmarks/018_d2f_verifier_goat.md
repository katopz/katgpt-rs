# GOAT Proof 018: D2F Drafter Verifier — Tri-Mode Inference (Plan 089)

> **Date:** 2026-05-21
> **Feature Gate:** `tri_mode`
> **Depends on:** Plan 066 (D2F), Plan 055 (MTP)

## Summary

Implemented `D2fDrafterVerifier` — a speculative verifier that uses D2F diffusion as drafter and AR as verifier. This is the "self-speculation" mode from the Nemotron paper, adapted to our existing infrastructure.

| Aspect | Result |
|--------|--------|
| Feature gate | `tri_mode` (depends on `dllm`) |
| New files | `speculative/d2f_verifier.rs`, `tests/test_d2f_verifier.rs` |
| Modified files | `speculative/types.rs`, `speculative/mod.rs`, `dllm.rs` |
| Lines added | ~250 (impl + tests) |
| Zero regressions | ✅ All 668 existing tests pass |

## Architecture

```
D2fDrafterVerifier (NEW)
├── Phase 0: Score initial token with AR target model (forward)
├── Phase 1: D2F block decode — parallel draft (d2f_decode_block_with_prompt_with)
├── Phase 2: Score each draft token with AR target model (forward × k)
├── Phase 3: Argmax comparison — accept longest prefix match
└── Phase 4: Bonus token if all accepted (sample from target dist)
```

Key insight: D2F draft and AR verify use **separate KV caches** (block-causal vs causal). This is correct — different attention patterns require different KV states.

## Proof Results (Untrained Model)

```
Proof 1: D2F drafter produces ≥1 token per step — ✅ PASS
  10 steps, each returns ≥1 token, max ≤ draft_width + 1

Proof 2: D2F drafter produces valid sequence — ✅ PASS
  20 steps, 20 total tokens, all in [0, 27)
  No infinite loops, terminates correctly

Proof 3: Mode switching via DecodeStrategy::recommend() — ✅ PASS
  recommend(4, 8, false) → DiscreteDiffusion ✓
  recommend(4, 8, true)  → SelfSpeculation   ✓
  recommend(16, 4, true) → Speculative       ✓
  recommend(16, 4, false) → Autoregressive   ✓

Proof 4: Acceptance rate (untrained) — ✅ PASS
  Draft width: 4
  Steps: 30
  Total tokens: 30
  Avg tokens/step: 1.00 / 4+1 max
  Time: 970.7 µs/step
  Theoretical throughput: 1030 tokens/sec
```

### Acceptance Rate Analysis (Untrained)

With untrained random weights, D2F draft tokens rarely match AR argmax, giving ~1.0 tokens/step. This is expected — the D2F parallel draft produces tokens via bidirectional attention while the AR verifier uses causal attention. Without shared training, these distributions are uncorrelated.

**Expected with trained model:** The Nemotron paper reports 60-80% acceptance rates for trained self-speculation models at our scale. Training the D2F drafter and AR verifier on the same data should align their distributions significantly.

## DecodeStrategy Extension

```rust
pub enum DecodeStrategy {
    Autoregressive,       // Standard AR — one token per step
    Speculative,          // AR drafts → AR verifies (LeviathanVerifier)
    DiscreteDiffusion,    // D2F block decode only (Plan 066)
    SelfSpeculation,      // D2F drafts → AR verifies (NEW, Plan 089)
}
```

Priority order (when `tri_mode` enabled):
1. `SelfSpeculation` — if draft model available AND enough tokens for block
2. `DiscreteDiffusion` — if enough tokens for block (no draft model needed)
3. `Speculative` — if draft model available
4. `Autoregressive` — fallback

## Global Loss Averaging

Added `LossAveraging` enum to `dllm.rs`:

| Variant | Formula | Use Case |
|---------|---------|----------|
| `Global` (default) | `L = (1/(N*L_masked)) * Σ_n Σ_i ℓ_{n,i}` | Nemotron +2.12% accuracy |
| `PerSequence` | `L = (1/N) * Σ_n (1/L_n) * Σ_i ℓ_{n,i}` | Equal weight per sample |

The existing `masked_loss()` already implemented global averaging. The new enum makes it configurable for future per-sequence experiments.

## What Changed

| File | Change |
|------|--------|
| `src/speculative/d2f_verifier.rs` | **NEW**: `D2fDrafterVerifier` — SpeculativeVerifier impl |
| `src/speculative/types.rs` | Added `SelfSpeculation` variant, `SelfSpecConfig`, updated `recommend()` |
| `src/speculative/mod.rs` | Added `d2f_verifier` module + re-exports |
| `src/dllm.rs` | Added `LossAveraging` enum, updated `masked_loss()` signature |
| `tests/test_d2f_verifier.rs` | **NEW**: 5 GOAT proof tests |
| `src/dllm.rs`, `src/speculative/d2f.rs`, tests/ | Fixed `Config::dllm_micro()` → `Config::micro_dllm()` |

## Deferred Tasks

- **T6: Trained Sampler** — Design `DiffusionSampler` for per-position correctness. Deferred until T1-T5 prove value.
- **T7: LoRA Drafter Alignment** — LK-hybrid loss for aligning D2F drafter with AR verifier. Deferred until riir-gpu has D2F training support.

## Honest Assessment

The "tri-mode" is really just three ways to compose existing components:

1. AR mode: `forward()` — already worked
2. D2F mode: `d2f_decode_block()` — already worked (Plan 066)
3. D2F+AR mode: `d2f_decode_block()` drafts → `forward()` verifies — **NEW** (~100 lines)

The actual new code is a `D2fDrafterVerifier` struct implementing the existing `SpeculativeVerifier` trait. The value is in the architecture, not the complexity.

Next step to unlock real value: train a joint AR+D2F model where the D2F drafter learns to align with AR verification, improving acceptance rate from 1.0 → 3-4 tokens/step.