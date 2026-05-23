# Plan 089: Tri-Mode Inference — D2F Drafter Verifier + Mode Switching

> **Research:** `.research/055_Nemotron_TriMode_Diffusion.md`
> **Paper:** Nemotron-Labs-Diffusion (NVIDIA 2026)
> **Depends on:** Plan 066 (D2F, ✅ complete), Plan 055 (MTP, ✅ complete)
> **Feature Gate:** `tri_mode` (opt-in, proofs required)

## Objective

Add a **D2F-backed `SpeculativeVerifier`** variant that uses our existing D2F diffusion as drafter and existing AR as verifier. This is NOT a new architecture — it's a new verifier strategy in our existing `SpeculativeVerifier` trait, swapping DFlash drafter for D2F drafter.

The "tri-mode" is just three ways to use what we already have:
1. **AR mode**: `forward()` causal — already works ✅
2. **Diffusion mode**: `d2f_decode_block()` — already works ✅ (Plan 066)
3. **D2F+AR mode**: `d2f_decode_block()` drafts → `forward()` verifies → prefix accept — NEW (this variant)

## Honest Assessment: What's Actually New

The Nemotron paper calls this "self-speculation" and presents it as a major contribution. But looking at our code:

| Component | What We Already Have | Delta |
|---|---|---|
| Draft→Verify→Accept pattern | `speculative/step.rs` `speculative_step_rollback()` | ✅ Same pattern |
| `SpeculativeVerifier` trait | `verifier.rs` with `SimulatedVerifier`, `LeviathanVerifier` | ✅ Abstraction ready |
| AR verification | `forward()` with causal attention | ✅ Already there |
| Prefix acceptance | `speculate()` in `LeviathanVerifier` | ✅ Already there |
| DDTree path extraction | `dd_tree.rs` | ✅ Already there |
| KV cache snapshot/rollback | `MultiLayerKVCache::snapshot()/restore()` | ✅ Already there |
| D2F block decode (drafter) | `d2f_decode_block()` in `speculative/d2f.rs` | ✅ Already there |
| D2F context (zero-alloc) | `D2fContext` in `dllm.rs` | ✅ Already there |
| **D2F drafter verifier** | **MISSING** | ❌ ~100 lines, new `SpeculativeVerifier` impl |

The actual new code is a `D2fDrafterVerifier` struct that:
1. Calls `d2f_decode_block()` instead of `dflash_predict()` for drafting
2. Calls existing `forward()` for verification (same as `LeviathanVerifier`)
3. Uses existing prefix acceptance logic (same as `LeviathanVerifier`)

This is a **variant**, not a new system.

## Feature Gate

All new code behind `tri_mode` feature flag (already added to `Cargo.toml`):
```toml
tri_mode = ["dllm"]  # depends on dllm for D2F drafter
```

---

## Tasks

### T1: D2F Drafter Verifier (microgpt-rs) — The Core Delta ✅
- [x] Create `src/speculative/d2f_verifier.rs` (feature-gated `tri_mode`)
- [x] Define `D2fDrafterVerifier`:
  ```rust
  /// Speculative verifier that uses D2F diffusion as drafter, AR as verifier.
  ///
  /// This is the Nemotron "self-speculation" mode — same draft→verify→accept
  /// pattern as LeviathanVerifier, but D2F drafts in parallel instead of
  /// DFlash drafting sequentially.
  pub struct D2fDrafterVerifier<'a> {
      d2f_ctx: &'a mut D2fContext,
      d2f_config: D2fDecodeConfig,
      target_ctx: &'a mut ForwardContext,
      target_cache: &'a mut MultiLayerKVCache,
      target_weights: &'a TransformerWeights,
      target_config: &'a Config,
  }
  ```
- [x] Implement `SpeculativeVerifier` for `D2fDrafterVerifier`:
  1. **Draft**: Call `d2f_decode_block()` on mask tokens for positions [pos..pos+draft_width]
     - Uses block-causal attention (bidirectional within block = parallel draft)
     - Returns k draft tokens
  2. **Verify**: Run `forward()` on draft tokens with causal attention
     - Reuse existing target model KV cache
     - Get AR logits at each draft position
  3. **Accept**: Compare draft[i] vs AR argmax[i], accept longest prefix
     - Same logic as `LeviathanVerifier::speculate()` prefix matching
     - Bonus token at first rejection (AR provides one extra)
- [x] Test: D2F drafter accepts ≥1 token per step
- [x] Test: D2F drafter terminates, produces valid token sequence
- [x] Test: acceptance rate measurement on pattern data vs LeviathanVerifier (AR drafter)

### T2: DecodeStrategy Extension (microgpt-rs) ✅
- [x] Extend `DecodeStrategy` enum in `speculative/types.rs`:
  ```rust
  pub enum DecodeStrategy {
      Autoregressive,
      Speculative,       // AR drafts → AR verifies (existing LeviathanVerifier)
      DiscreteDiffusion, // D2F block decode only (existing D2F pipeline)
      SelfSpeculation,   // D2F drafts → AR verifies (NEW D2fDrafterVerifier)
  }
  ```
- [x] Update `DecodeStrategy::recommend()` heuristic
- [x] Feature-gate `SelfSpeculation` variant with `#[cfg(feature = "tri_mode")]`
- [x] Test: recommend() returns correct strategy per config

### T3: Wire Into Existing Pipeline (microgpt-rs) ✅
- [x] Add `pub mod d2f_verifier;` to `speculative/mod.rs` behind `tri_mode` feature
- [x] Ensure `D2fDrafterVerifier` integrates with existing `ConstraintPruner`
  - D2F draft already calls pruner at each denoising step ✅
  - AR verify already prunes via `is_valid()` ✅
  - No new integration needed
- [x] Ensure KV cache flows correctly:
  - D2F draft: uses `D2fContext` KV (block-causal)
  - AR verify: uses `MultiLayerKVCache` KV (causal)
  - These are separate caches — draft KV is NOT reused for verify
  - This is correct: different attention patterns need different KV states
- [x] Add `SelfSpecConfig` to `speculative/types.rs`:
  ```rust
  /// Config for D2F-drafter self-speculation mode.
  /// Wraps D2F decode config + draft width for speculative step.
  pub struct SelfSpecConfig {
      pub draft_width: usize,       // k tokens per draft (default: 8)
      pub d2f_config: D2fDecodeConfig, // D2F decode parameters
  }
  ```
- [x] Test: end-to-end D2F+AR decode on pattern data
- [x] Test: D2F+AR output matches AR-only ground truth (quality check)

### T4: Global Loss Averaging (microgpt-rs dllm.rs) ✅
- [x] Update `masked_loss()` in `src/dllm.rs`:
  ```rust
  // BEFORE (per-sequence):
  // L = (1/N) * Σ_n (1/L) * Σ_i ℓ_{n,i}
  //
  // AFTER (global — Nemotron validates +2.12% accuracy):
  // L = (1/(N*L_masked)) * Σ_n Σ_i ℓ_{n,i}
  //    where L_masked = total masked positions across batch
  ```
- [x] Add `LossAveraging` enum: `PerSequence`, `Global` (default `Global`)
- [x] Test: global averaging produces different loss when masking varies per sample
- [x] Test: training with global averaging converges (re-train mini dLLM)
- [x] Benchmark: convergence speed global vs per-sequence on pattern data

### T5: GOAT Proof — D2F Drafter vs AR Drafter (microgpt-rs) ✅
- [x] Create `tests/test_d2f_verifier.rs` (feature-gated `tri_mode`)
- [x] Proof 1: D2F drafter acceptance rate ≥ AR drafter acceptance rate
  - Train mini model with D2F + AR capability (reuse Plan 066 training)
  - Run 1000 steps `D2fDrafterVerifier` vs 1000 steps `LeviathanVerifier`
  - Measure: avg tokens accepted per step
  - Hypothesis: D2F parallel draft may have lower per-token accuracy but
    higher throughput (more tokens per forward pass)
- [x] Proof 2: D2F+AR produces valid output
  - Decode 100 tokens with `D2fDrafterVerifier`
  - Verify: all tokens in valid range, no infinite loops, terminates
- [x] Proof 3: Mode switching works correctly
  - Start AR, switch to SelfSpeculation, switch to DiscreteDiffusion
  - Verify: seamless transition
- [x] Benchmark: throughput comparison AR vs Speculative vs SelfSpeculation vs D2F
- [x] Record results in `.benchmarks/018_d2f_verifier_goat.md`

### T6: Trained Sampler Research → Consolidated into Plan 116
- [ ] **MOVED to `.plans/116_consolidated_diffusion_sampler_goat.md` T1-T4**
- `diffusion_sampler.rs` created (43K, ~30 tests), needs wiring into `mod.rs`
- Plan 116 T1: wire module, T2: run tests, T3: integrate into D2F loop, T4: GOAT benchmark
- Deferred until T1-T5 prove self-speculation has value at our scale ✅ (proved)

### T7: LoRA Drafter Alignment → Consolidated into Plan 116
- [ ] **MOVED to `.plans/116_consolidated_diffusion_sampler_goat.md` T6**
- Blocked on riir-gpu D2F training support
- Plan 116 tracks as deferred task for visibility

---

## Architecture After

```
speculative/
├── mod.rs                # pub mod d2f_verifier (feature-gated tri_mode)
├── types.rs              # DecodeStrategy + SelfSpecConfig + SelfSpeculation variant
├── step.rs               # AR speculative step (unchanged)
├── d2f.rs                # D2F block decode (unchanged — used as drafter)
├── d2f_verifier.rs       # NEW: D2fDrafterVerifier — SpeculativeVerifier impl
│   └── D2fDrafterVerifier — uses d2f_decode_block() as drafter
│                          — uses forward() as verifier
│                          — same prefix acceptance as LeviathanVerifier
├── verifier.rs           # SpeculativeVerifier trait (unchanged)
├── dd_tree.rs            # DDTree (unchanged)
├── dflash.rs             # AR drafter (unchanged — used by LeviathanVerifier)
└── ...
```

## Key Difference: D2F Drafter vs AR Drafter

| Aspect | AR Drafter (DFlash/Leviathan) | D2F Drafter (D2fDrafterVerifier) |
|---|---|---|
| Draft method | `dflash_predict()` — sequential AR | `d2f_decode_block()` — parallel diffusion |
| Draft quality | High per-token accuracy (causal) | Lower per-token, but parallel (bidirectional) |
| Draft cost | O(k) sequential forwards | O(1) parallel forward + denoising steps |
| KV reuse | Same KV for draft + verify | Separate KV (block-causal vs causal) |
| Best for | Low latency per token | High throughput (batch-size-1) |

## Estimated Effort

| Task | Lines | Effort | Depends On |
|------|-------|--------|-----------|
| T1: D2F drafter verifier | ~100 | 1 day | D2F (done), SpeculativeVerifier (done) |
| T2: DecodeStrategy extension | ~30 | 0.5 day | T1 |
| T3: Pipeline wiring | ~50 | 0.5 day | T1, T2 |
| T4: Global loss averaging | ~30 | 0.5 day | None |
| T5: GOAT proof | ~200 (tests) | 2 days | T1-T4 |
| T6: Trained sampler | deferred | — | T5 |
| T7: LoRA alignment | deferred | — | T5, riir-gpu |

**Total: ~4-5 days for T1-T5**

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| D2F draft quality too low for verification | Low acceptance rate | Increase denoising steps; fall back to AR mode |
| Separate KV caches waste memory | 2× KV memory | D2F context is temporary, freed after accept |
| No real tri-mode model to test | Can't validate at scale | Test with mini dLLM (Plan 066 proved this works) |
| Feature gate conflicts | Build failures | tri_mode → dllm dependency, CI tested |

## What This Does NOT Do

- ❌ Does NOT train a joint AR-diffusion model (need 1T-token pretraining)
- ❌ Does NOT implement dual-stream attention (training-only optimization)
- ❌ Does NOT add VLM support (no vision encoder)
- ❌ Does NOT implement quadratic self-speculation (kernel complexity)
- ❌ Does NOT change existing AR, D2F, or LeviathanVerifier code (feature-gated only)

## Success Criteria

1. ✅ `D2fDrafterVerifier` implements `SpeculativeVerifier` trait
2. ✅ D2F+AR mode produces valid token sequences
3. ✅ Mode switching works via `DecodeStrategy` enum
4. ✅ Global loss averaging improves D2F training convergence
5. ✅ All new code behind `tri_mode` feature gate
6. ✅ Zero regression in existing AR/D2F/speculative benchmarks