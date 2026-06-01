# Plan 166: FlashAR Consensus Tri-Mode with Ternary Thermal Paths

**Source:** Research 149 — FlashAR Diagonal-Step Parallel Decoding
**Status:** In Progress
**Feature Gate:** `flashar_consensus` (requires `tri_mode` + `dllm` + `plasma_path`)
**Default:** Off until GOAT proof

---

## Goal

Replace tri_mode's prefix-match acceptance with FlashAR-inspired **dual-path consensus draft + ternary thermal path routing**. When MTP and D2F agree at a position (ternary = 0), accept without AR verification (plasma path). Disputed positions route to hot/warm/cold paths based on confidence. The ternary fusion gate reuses `simd_ternary_matvec` — zero multiplication.

## Architecture

```text
Current D2fDrafterVerifier.speculate():
  D2F draft → AR sequential verify → prefix-match accept

FlashAR Consensus with Ternary Thermal Paths:
  Path H: AR/MTP draft      → per-position tokens + DiffusionSampler features
  Path V: D2F block draft   → per-position tokens + DiffusionSampler features

  Ternary consensus per position:
    +1 → H wins (MTP confident, D2F uncertain)
     0 → AGREE (both same token) → PLASMA PATH (skip verify, zero compute)
    -1 → V wins (D2F confident, MTP uncertain)

  Thermal routing (ternary magnitude + confidence):
    PLASMA  (ternary=0, high conf)   → accept immediately
    HOT     (ternary=±1, high conf)  → accept winner, no verify
    WARM    (ternary=±1, mid conf)   → AR spot-check this position only
    COLD    (both low conf)          → fallback prefix-match

  Fusion gate: simd_ternary_matvec(gate_weights, sampler_features) → path scores
  → Reuses plasma_path SIMD kernels (Neon/AVX2/scalar) — zero multiplication
```

### Key Insight

FlashAR's fusion gate σ(MLP) per position is replaced by our ternary thermal path router:
1. The consensus is *naturally* ternary {-1, 0, +1}
2. `TernaryWeights` + `simd_ternary_matvec` already provide SIMD-accelerated ternary operations
3. The fusion decision becomes a multiplication-free matvec on bitmasks
4. Thermal paths stratify verification cost: plasma=0, hot=~0, warm=1 AR step, cold=N AR steps

---

## Tasks

### Phase 1: Ternary Consensus Core (Modelless)

- [x] T1: Create `src/speculative/flashar_consensus.rs` behind `flashar_consensus` feature gate
  - `ThermalPath` enum: Plasma, Hot, Warm, Cold
  - `ConsensusConfig`: plasma_threshold, hot_threshold, warm_threshold (confidence boundaries)
  - `ConsensusResult`: per-position `ThermalPath` + accepted token + ternary code
  - Stack-allocated: `[ThermalPath; 64]` thermal buffer (bounded by draft_width)
  - Stack-allocated: `[i8; 64]` ternary consensus codes

- [x] T2: Implement `dual_path_draft()` — run both MTP and D2F drafting
  - Reuse `dflash_predict_ar_with` for Path H (AR draft)
  - Reuse `d2f_decode_block_with_prompt_with` for Path V (D2F draft)
  - Return both token arrays + per-position `SamplerFeatures` for each path
  - Zero-alloc: reuse `SpeculativeContext` and `D2fContext`

- [x] T3: Implement `compute_ternary_consensus()` — per-position ternary encoding
  - For each position i in [0, draft_width):
    - `ternary[i] = 0` if `h_i == v_i` (consensus)
    - `ternary[i] = +1` if `h_i != v_i` AND `conf_H > conf_V`
    - `ternary[i] = -1` if `h_i != v_i` AND `conf_V >= conf_H`
  - O(draft_width) — linear scan, fixed-size stack arrays
  - Branch-free where possible: use `signum(h_conf - v_conf)` pattern

- [x] T4: Implement `route_thermal_paths()` — ternary → thermal path assignment
  - `plasma_threshold`: if consensus AND both top1_prob > threshold → Plasma
  - `hot_threshold`: if disputed AND winner top1_prob > threshold → Hot
  - `warm_threshold`: if disputed AND winner top1_prob > lower threshold → Warm
  - Default: Cold (both low confidence → fallback to prefix-match)
  - Returns `[ThermalPath; 64]` — one thermal path per position

- [x] T5: Implement ternary SIMD fusion gate (optional, modelless heuristic fallback)
  - When `ConsensusConfig::use_ternary_gate == true`:
    - Concatenate DiffusionSampler features from BOTH paths: `[h_features_6d || v_features_6d]` = 12-dim
    - Mirrors FlashAR's `σ(MLP([h_feat || v_feat]))` pattern but with ternary weights
    - Use `simd_ternary_matvec(&gate_weights, &dual_features_12d, &mut scores)`
    - `gate_weights`: `TernaryWeights` with rows=4 (4 thermal paths), cols=12 (dual-path features)
    - Output: 4 thermal path scores per position → argmax selects path
  - When `false`: use heuristic thresholds from T4 (no SIMD call)
  - Both paths reuse `plasma_path` SIMD infrastructure — zero new kernels

### Phase 2: Thermal Verification Engine

- [x] T6: Implement `FlashARConsensusVerifier` — new `SpeculativeVerifier` impl
  - `speculate()`: dual_path_draft → ternary_consensus → thermal_route → selective verify
  - Plasma positions: accept immediately (no AR forward pass)
  - Hot positions: accept winner (no AR forward pass)
  - Warm positions: single AR forward pass for just that position
  - Cold positions: fallback to prefix-match for contiguous prefix
  - Assemble final accepted token list from plasma + hot + warm + verified

- [x] T7: Implement non-contiguous KV cache commit
  - Current: accepts sequential prefix → contiguous KV write
  - New: accepts sparse set → must handle gaps in KV cache
  - Strategy: commit accepted tokens contiguously (drop rejected positions)
  - Track position mapping: draft_pos → kv_pos for correct cache indexing

### Phase 3: GOAT Proof

- [x] T8: Create `tests/bench_166_flashar_consensus_goat.rs`
  - Test 1: `dual_path_draft` produces both MTP and D2F token sets
  - Test 2: `compute_ternary_consensus` correctly encodes {-1, 0, +1}
  - Test 3: `route_thermal_paths` assigns correct thermal paths by confidence
  - Test 4: `FlashARConsensusVerifier` accepts ≥ 1 token always (safety)
  - Test 5: Consensus acceptance rate ≥ prefix-match rate (never worse)
  - Test 6: Plasma path skips AR verification (zero forward passes for consensus)
  - Test 7: Ternary gate produces same routing as heuristic (validation)

- [x] T9: Benchmark: FlashAR Consensus vs current tri_mode prefix-match
  - Metric: average tokens accepted per speculate() call
  - Metric: wall-clock time per accepted token
  - Metric: plasma path hit rate (% of positions that skip verification)
  - Must show ≥ 1.2× improvement in tokens-accepted-per-cycle to justify

### Phase 4: Default-On Promotion (if GOAT passes)

- [x] T10: T9 shows gain with no perf hurt, promoted to default-on
  - Added `flashar_consensus` to default features in `Cargo.toml`
  - Update README with GOAT proof results + plasma path hit rate (pending)
  - Update `.contexts/optimization.md` with ternary thermal path pattern (pending)

### Phase 5: Strided Anchor D2F (Stretch)

- [ ] T11: Implement strided anchor-then-fill pattern under `flashar_anchor` feature gate
  - Round 1: predict every S-th position via AR (diagonal analog)
  - Round 2: D2F decode remaining positions with anchor positions pre-filled
  - Measure: denoising iterations reduction with vs without anchors

---

## Implementation Notes

### Alignment with optimization.md

- **No allocation in hot path**: `ConsensusResult` uses stack `[ThermalPath; 64]` + `[i8; 64]`
- **Pre-allocated buffers**: reuse `SpeculativeContext` for Path H, `D2fContext` for Path V
- **Fixed-size arrays**: all arrays bounded by `draft_width` (default 8, max 64)
- **Benchmark before/after**: T9 compares vs current tri_mode baseline
- **SIMD reuse**: ternary fusion gate uses existing `simd_ternary_matvec` — zero new kernels

### Ternary Thermal Path Details

```text
Dual-path feature vector (12-dim, mirrors FlashAR [h_feat || v_feat]):
  [h_top1, h_margin, h_top3, h_entropy, h_step_norm, h_pos_norm,
   v_top1, v_margin, v_top3, v_entropy, v_step_norm, v_pos_norm]

Ternary gate: TernaryWeights(rows=4, cols=12) → 4 thermal path scores
  Row 0: Plasma path weights (favors agreement + high confidence)
  Row 1: Hot path weights (favors strong winner)
  Row 2: Warm path weights (favors moderate disagreement)
  Row 3: Cold path weights (favors low confidence)

Thermal path = argmax(plasma_score, hot_score, warm_score, cold_score)
All via ternary SIMD: acc += (pos_bits & x) - (neg_bits & x) — zero multiplication

Default thresholds (heuristic fallback when ternary gate not trained):
  τ_plasma=0.7, τ_hot=0.5, τ_warm=0.3
```

### Fallback Safety

- `FlashARConsensusVerifier` falls back to prefix-match if all positions are Cold
- Always returns ≥ 1 token (same guarantee as current verifier)
- Thermal path routing is additive — if it doesn't help, prefix-match path still works

### Feature Gate Structure

```toml
[features]
tri_mode = ["dllm"]
flashar_consensus = ["tri_mode", "plasma_path"]  # dual-path consensus + ternary thermal paths
flashar_anchor = ["dllm"]                         # strided anchor-then-fill (stretch)
```
