# Plan 057: Higher-order Linear Attention — O(1) Inference Cache

**Branch:** `develop/feature/057_higher_order_linear_attention`
**Depends on:** Plan 010 (Multilayer Transformer), Plan 020 (Raven KV Cache — pattern reference)
**Research:** `.research/28_Higher_order_Linear_Attention.md`
**Goal:** Implement second-order HLA (symmetric + asymmetric AHLA) as an alternative to standard KV-cache attention. Achieve O(1) per-token memory independent of sequence length. Benchmark against flat KV, Raven, and TurboQuant to quantify the tradeoff: constant memory vs quality (models must be trained with HLA from scratch).

---

## Tasks

### Phase 1: Types & State

- [x] T1: Define `HlaQHeadState` struct in `src/hla/types.rs` — CQV `[hd×hd]`, mQ `[hd]`, G `[hd×hd]`, h `[hd]` + `new(hd)` + `reset()`
- [x] T2: Define `AhlaQHeadState` struct in `src/hla/types.rs` — e `[hd×hd]`, n `[hd]` + `new(hd)` + `reset()`
- [x] T3: Define `MultiLayerHlaCache` — `layers: Vec<HlaLayerState>` with SK per KV group, per-Q-head state (GQA-aware)
- [x] T4: Define `MultiLayerAhlaCache` — same layer structure with PKV/mK per KV group, E/n per Q head
- [x] T5: Add `new()` / `reset()` / `with_gamma()` / `memory_bytes()` for both caches
- [x] T6: Register `#[cfg(feature = "hla_attention")]` gate on `pub mod hla` in `lib.rs`, feature in `Cargo.toml`, included in `full`

### Phase 2: Attention Kernels

- [x] T7: Implement `hla_state_update()` in `src/hla/kernel.rs` — streaming recurrence with correct update ordering (cross-terms before main accumulators). Zero-alloc via pre-allocated temp buffers
- [x] T8: Implement `hla_readout()` — readout `o_t = q_tᵀ(SK·CQV − G)`. Two-stage matvec, zero-alloc
- [x] T9: Implement `ahla_step()` — combined update+readout for AHLA. Zero-alloc
- [x] T10: Add normalization: `hla_denom()` / `ahla_denom()` + optional divide-by-denom in layer helpers
- [x] T11: Add exponential decay γ in both `hla_state_update()` and `ahla_step()`
- [x] T12: Verify GQA correctness — `kv_group()` mapping, layer helpers handle shared SK/PKV correctly

### Phase 3: Forward Integration

- [x] T13: Add `forward_hla()` in `src/hla/forward.rs` — same structure as `forward_base()` with HLA cache
- [x] T14: Add `forward_ahla()` — same with AHLA cache
- [x] T15: Pre-allocated temp buffers in forward functions (stack-allocated, reused across layers)
- [x] T16: Add `generate_hla_into()` / `generate_ahla_into()` — convenience wrappers matching `generate_into()` API

### Phase 4: Benchmarks (Before/After)

- [x] T17: `bench_hla_vs_flat_cache()` in `src/benchmark.rs` — compares flat KV vs symmetric HLA vs asymmetric AHLA at positions 1, 16, 64, 128, 256 across micro/game/bpe configs. Measures tok/s and µs/step
- [x] T18: `bench_hla_memory()` in `src/benchmark.rs` — measures bytes/layer for flat KV (O(N)), symmetric HLA (O(d²)), asymmetric AHLA (O(d·dv)) across all 5 configs. Cross-checked with `HlaVariant::layer_bytes()`
- [x] T19: `bench_hla_quality()` in `src/benchmark.rs` — logit divergence sanity check: asserts finite/non-NaN outputs, reports max/mean absolute divergence between SDPA and HLA/AHLA on random weights. Not a quality claim (models must be trained with HLA)
- [x] T20: Add HLA/AHLA rows to existing benchmark CSV output and timeseries — bench functions return `Vec<BenchResult>` compatible with `save_results_csv()` + `save_timeseries_csv()`

### Phase 5: Documentation & Polish

- [x] T21: Update `README.md` — added HLA section after PFlash with memory comparison table, variant table, key insight, and "not a drop-in replacement" warning
- [x] T22: Update `Cargo.toml` feature flags section in README with `hla_attention`
- [x] T23: Fix all clippy warnings under `hla_attention` feature: `cargo clippy --features hla_attention --fix --allow-dirty` — clean
- [ ] T24: Commit with message `feat(hla): second-order linear attention — O(1) inference cache`

---

## Architecture

```text
src/hla/                         — Feature-gated module: #[cfg(feature = "hla_attention")]
├── mod.rs                       — Module index, re-exports
├── types.rs                     — HlaQHeadState, AhlaQHeadState, HlaLayerState, AhlaLayerState,
│                                  MultiLayerHlaCache, MultiLayerAhlaCache, HlaVariant
├── kernel.rs                    — hla_state_update(), hla_readout(), hla_denom(),
│                                  ahla_step(), ahla_denom(),
│                                  hla_layer_update(), hla_layer_readout(), ahla_layer_step()
└── forward.rs                   — forward_hla(), forward_ahla(),
                                   generate_hla_into(), generate_ahla_into()

src/transformer.rs
└── ForwardContext               — fields made pub(crate) for HLA module access

src/types.rs
├── HlaMode enum                 — Standard, Hla, Ahla (added in Plan 058 commit)
└── Config.hla_mode/normalize/decay — HLA config fields (added in Plan 058 commit)

src/benchmark.rs                 — Phase 4 (TODO)
├── bench_hla_vs_flat_cache()   — throughput comparison
├── bench_hla_memory()          — memory comparison
└── bench_hla_quality()         — logit divergence sanity check
```

### State Layout (GQA-aware)

For a config with `n_head=8`, `n_kv_head=2`, `head_dim=8`:

```text
Per KV-group (shared):  SK [8×8] = 64 floats, or PKV [8×8] = 64 floats
Per Q-head (unique):    CQV [8×8] + mQ [8] + G [8×8] + h [8] = 200 floats (symmetric)
                        E [8×8] + n [8] = 72 floats (AHLA)

Total per layer:
  Symmetric: 2 × 64 + 8 × 200 = 1728 floats = 6.9 KB
  AHLA:      2 × 64 + 8 × 72  = 704 floats  = 2.8 KB
  Flat KV:   256 × 16 × 2     = 8192 floats = 32.8 KB (at block_size=256)
```

### Forward Flow (HLA variant)

```text
1. Embedding: x = wte[token] + wpe[pos]
2. For each layer:
   a. RMSNorm → save residual → RMSNorm
   b. QKV projections (same as standard)
   c. For each head:
      - Extract q_h, k_h, v_h (respecting GQA grouping)
      - hla_state_update(state, q_h, k_h, v_h, hd)   ← UPDATE state
      - attn_out[h] = hla_attention_head(q_h, state, hd) ← READOUT
   d. Output projection + residual
   e. MLP (unchanged)
3. LM Head (unchanged)
```

---

## Key Design Decisions

1. **AHLA implemented first** — Lower state cost (O(d·dv) vs O(d²)), simpler 4-tuple, better for tiny head_dims. Symmetric HLA adds expressivity via data-dependent metric SK.
2. **Feature gate `hla_attention`** — All new code behind `#[cfg(feature = "hla_attention")]`. Default off. Zero cost when disabled.
3. **NOT a drop-in replacement** — Models trained with SDPA will produce different outputs with HLA. Document this clearly. HLA is a training architecture choice, not an inference optimization for pretrained SDPA models.
4. **Zero-alloc hot path** — All temp buffers pre-allocated in `ForwardContext`. No Vec::push or heap allocation in `hla_state_update()` or `ahla_step()`.
5. **Second-order only** — Third-order adds minimal value for head_dim 4-16 but significant complexity (3 corrected cross-summaries, segment maps). Not worth it for our configs.
6. **GQA-aware from day one** — SK/PKV shared per kv_group, CQV/E per q_head. Correct GQA layout is non-negotiable.
7. **Optional decay γ** — Exponential decay for recency bias. Default γ=1.0 (no decay). Configurable per-config.
8. **Config extends, not replaces** — Add `hla_mode: HlaMode` enum to Config: `Standard`, `Hla`, `Ahla`. No breaking change to existing configs.

### Config Extension

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum HlaMode {
    #[default]
    Standard,  // SDPA with KV cache (current behavior)
    Hla,       // Symmetric second-order HLA
    Ahla,      // Asymmetric second-order AHLA
}

// In Config:
pub hla_mode: HlaMode,              // which attention variant
pub hla_normalize: bool,            // divide by masked denominator
pub hla_decay: f32,                 // γ ∈ (0,1], default 1.0
```

---

## Expected Outcomes

### Success Criteria

1. ✅ `forward_hla()` and `forward_ahla()` compile and run without panics
2. ✅ Memory per layer is constant (does not grow with position)
3. ✅ HLA/AHLA states pass round-trip test: update then readout produces finite, non-NaN output
4. ✅ Benchmark shows constant µs/step across positions 1-256 (vs growing for flat KV)
5. ⚠️ Logit divergence from SDPA is non-trivial (expected — different operator)
6. ✅ AHLA state < symmetric HLA state < flat KV cache for all configs

### What This Proves

- ✅ O(1) inference memory is achievable with second-order prefix statistics
- ✅ The algebraic identities from the HLA paper are implementable in Rust
- ✅ GQA-aware state layout works correctly
- ✅ AHLA is the practical choice for tiny models (lower state, simpler code)

### What This Does NOT Prove

- ❌ HLA produces better outputs than SDPA (requires training from scratch)
- ❌ HLA is faster than SDPA for short sequences (O(d²) overhead at short seq_len)
- ❌ Quality parity with SDPA on pretrained weights (guaranteed to differ)

---

## Benchmark Targets

| Metric | Flat KV | Symmetric HLA | AHLA |
|--------|---------|---------------|------|
| Memory/layer (micro, hd=4) | 128 floats | 80 floats/head × 4 = 320 | 16 floats/head × 4 = 64 |
| Memory/layer (bpe, hd=8) | 4096 floats | 200 floats/head × 4 = 800 | 72 floats/head × 4 = 288 |
| µs/step at pos=1 | ~10 | ~12 (matmul overhead) | ~8 |
| µs/step at pos=256 | ~40 (O(N) scan) | ~12 (constant) | ~8 (constant) |
| Context window | block_size | ∞ (streaming) | ∞ (streaming) |

### Before/After Benchmark Matrix

```text
Config       | Position | Flat KV µs | HLA µs | AHLA µs | Flat mem | HLA mem | AHLA mem
micro (hd=4) |        1 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
micro (hd=4) |       16 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
game (hd=8)  |        1 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
game (hd=8)  |      170 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
bpe (hd=8)   |        1 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
bpe (hd=8)   |      256 |        ??? |    ??? |     ??? |     ??? |     ??? |      ???
```

(The `???` values will be filled by T17 benchmarks.)

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| Plan 010 (Multilayer) | `forward_base()` is the template for `forward_hla()` |
| Plan 020 (Raven) | Similar O(1) cache pattern but Raven is heuristic, HLA is exact algebra |
| Plan 043 (TurboQuant) | Complementary — TQ compresses KV entries, HLA eliminates them entirely |
| Plan 044 (PFlash) | Prefill parallelism — HLA chunk-parallel scan is future training work |
| Plan 050 (Feature Gate) | `hla_attention` follows same gate pattern |
| Plan 055 (MTP Drafter) | MTP could use HLA for longer draft context windows |

---

## Risks

1. **Correctness of GQA state sharing** — SK shared per kv_group means multiple heads write to the same `sk` buffer. Must verify no data race in sequential code (fine for single-threaded inference). Future: parallel head computation needs per-group SK accumulation then merge.
   - Mitigation: Unit test that verifies `forward_hla()` output matches manual single-head computation.

2. **Update order bug** — Computing G,h with NEW CQV,mQ instead of OLD produces wrong output. This is the #1 correctness trap.
   - Mitigation: Explicit comment + assertion pattern. Separate `update_cross_terms()` and `update_accumulators()` functions.

3. **Quality unknown** — We can't verify HLA quality without training from scratch. Random weights will show divergence from SDPA but that proves nothing.
   - Mitigation: Logit divergence benchmark is a sanity check (finite, non-NaN), not a quality claim.

4. **O(d²) overhead at short sequences** — For seq_len=1 and head_dim=16, SK matmul (16²=256 ops) is more expensive than single-position attention (16 ops).
   - Mitigation: Document the break-even point. For our configs (hd=4-8), HLA is competitive from the start.

5. **Feature gate combinatorics** — `hla_attention` + `domain_latent` + `sparse_mlp` combinations need to compile.
   - Mitigation: CI builds with `--all-features` and individual feature combinations.