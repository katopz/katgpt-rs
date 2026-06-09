# Plan 173: Wall Attention — Diagonal Forget Gates Replacing RoPE

This plan implements Wall Attention (diagonal forget gates replacing RoPE) in katgpt-rs. It must align with optimization.md principles (profile first, pre-allocate, SIMD-friendly, no allocation in hot loops).

Plan number: **173**
Feature gate: `wall_attention`

---

## Key Design Decisions

1. Wall replaces RoPE entirely when enabled — not additive (paper confirms Wall(NoPE) > Wall+RoPE)
2. Key-projected gate variant (derive gate from K) is preferred for zero KV cache overhead
3. Gate bias initialized to 6.0 (open-gate init matching vanilla attention)
4. KV-head gate tying by default in GQA configs (one gate per KV head)
5. The factorized form means Wall is algorithmically identical to standard attention after Q/K rescaling — no attention kernel changes needed
6. Per-layer prefix sums: each layer maintains independent prefix sums `[n_layer × n_kv_head × head_dim]`

---

## Architecture Integration

The factorized form `q̃_i = exp(P_i) ⊙ q_i`, `k̃_j = exp(-P_j) ⊙ k_j` means:

- Replace `apply_rope_with_freq(q, k, pos, ...)` with `apply_wall_rescale(q, k, prefix_sum_buf, ...)`
- The attention kernels (`attention_head`, `attention_head_softcap`, etc.) are UNCHANGED — they just receive pre-rescaled Q and K
- For decode: maintain running `P_t` prefix sum (O(1) update per token), rescale only the current query
- For prefill: compute prefix sum once over all positions, rescale Q and K in one pass
- Per-layer isolation: prefix sums indexed by `[layer_idx * n_kv_head * head_dim + kv_head * head_dim + d]`

---

## Task 1: WallAttention types and config

- [x] Define `WallConfig` struct with:
  - `gate_bias: f32` (default 6.0)
  - `gate_max: f32` (default 0.87)
  - `use_key_projected: bool` (default true)
- [x] Add `wall_config: Option<WallConfig>` to `Config` (`None` = use RoPE/fallback)
- [x] Add `wall_enabled()` convenience method to `Config`
- [x] Add gate weight `W_g` projection matrix to `TransformerWeights` (`attn_wg: Vec<f32>` per layer, `[kv_dim]` elements)

## Task 2: Wall gate computation kernel

- [x] Implement `wall_gate_project(hidden: &[f32], w_g: &[f32], bias: f32, gate_max: f32, out: &mut [f32])`
  - Projects hidden state → raw gate logits → log-sigmoid → soft-clamp to (-gate_max, 0]
  - SIMD-friendly: elementwise operations, no branches
- [x] Implement key-projected variant: `wall_gate_from_key(key: &[f32], w_g: &[f32], bias: f32, ...)`
  - `compute_gate_from_key()` in `WallPrefixState`

## Task 3: Wall prefix sum and rescale

- [x] Define `WallPrefixState` struct with `prefix_sums: Vec<f32>` `[n_layer × n_kv_head × head_dim]`
- [x] Implement `update_prefix(layer_idx, kv_head, gate)` — O(head_dim) per token
- [x] Implement `rescale_query(layer_idx, q, kv_group_lut, n_head)` — `q *= exp(prefix)`
- [x] Implement `rescale_key(layer_idx, k)` — `k *= exp(-prefix)`
- [x] Per-layer isolation: each layer reads/writes its own slice of prefix_sums

## Task 4: Forward pass integration

- [x] Modify existing forward paths (not a separate function):
  - `forward_base`: Wall gate + prefix sum + Q/K rescale after QKV projection
  - `forward_coda`: Same integration (CODA-fused path)
  - `forward_prefill`: Wall gate in Phase A (K/V computation), Q rescale before Phase B
  - `forward_paged`: Wall gate + Q/K rescale after QKV projection
- [x] Where RoPE is applied: replace with Wall gate projection + prefix sum update + Q/K rescaling
- [x] Attention kernels unchanged — they receive wall-rescaled Q/K
- [x] KV cache stores wall-rescaled keys (same as RoPE stores rotated keys)
- [x] Prefix sum reset at sequence start (pos=0 in decode, prefill start)

---

## Task 5: Wall + GDN2 unified gate infrastructure

- [x] Extract shared `DiagonalGate` trait from GDN2's channel-wise decay and Wall's diagonal gate
- [x] Both use the same `Diag(g_t)` primitive — just applied differently:
  - GDN2: gate applied to recurrent state decay
  - Wall: gate applied to softmax attention via factorized Q/K rescale

## Task 6: Wall + DashAttention integration

- [ ] Use gate-derived "forgetfulness scores" for block-level routing decisions
- [ ] When all channels of a key have decayed below threshold → skip block in sparse attention
- [ ] Compute per-block min-retention from prefix sums at block boundaries

## Task 7: Wall + RTPurbo gate-aware retrieval

- [ ] Analyze gate variance per head/channel to identify "always-on" (retrieval-critical) vs "dynamic" (recency) dimensions
- [ ] Weight RTPurbo's low-dim projection toward high-variance channels (dynamic = content-dependent)
- [ ] Gate statistics as additional features for retrieval head scoring

## Task 8: GOAT proof — Wall correctness

- [x] Unit test: known gate values → expected Q/K rescaling → expected attention scores
- [x] Unit test: `gate_bias=6.0` → retention≈1.0 (vanilla attention behavior)
- [x] Unit test: `gate_bias=0` → retention≈0.62 (active forgetting)
- [x] Numerical stability: prefix sum doesn't overflow at `seq_len=8192` with `gate_max=0.87`
- [x] End-to-end: multi-layer Wall forward produces finite logits
- [x] End-to-end: prefill→decode with Wall produces finite logits
- [x] `wall_enabled()` convenience method test
- [x] Gate weights properly initialized (kv_dim elements, non-zero)

## Task 9: GOAT proof — Wall performance

- [x] Benchmark: Wall rescale overhead vs RoPE rotation overhead (expect: same or less — elementwise multiply vs paired rotation)
- [x] Benchmark: decode throughput with Wall vs RoPE (expect: identical — attention kernels unchanged)
- [x] Profile: gate projection matmul is the only new cost — should be < 1μs for typical head_dim

## Task 10: Feature gate and Config integration

- [x] Add `wall_attention` feature flag
- [x] When enabled + `wall_config` is `Some`: use Wall (ignore `use_rope`)
- [x] When enabled + `wall_config` is `None`: fall back to RoPE if `use_rope=true`
- [x] Update `active_features` benchmark output

---

## Alignment with optimization.md

- **Gate computation**: elementwise SIMD, no allocation (pre-allocated `prefix_sum` in context)
- **Q/K rescale**: elementwise multiply, same cost as RoPE rotation
- **Prefix sum**: O(head_dim) per token, incremental update (no full recomputation)
- **No new allocations in hot path** — all buffers pre-allocated in ForwardContext
- **Per-layer prefix sums**: `n_layer × n_kv_head × head_dim` total, no per-call allocation

---

## Files Changed

- `crates/katgpt-core/src/types.rs`: `WallConfig`, `Config.wall_config`, `Config::wall_enabled()`
- `crates/katgpt-core/Cargo.toml`: `wall_attention` feature flag
- `src/transformer.rs`: `WallPrefixState`, `LayerWeights.attn_wg`, forward path integration
- `Cargo.toml`: `wall_attention` feature flag, `full` feature
- `tests/goat_172_173_rim_wall.rs`: 12 GOAT proof tests

---

## Risks

- Requires model weights with `W_g` gate projection — only works with Wall-trained models
- For existing RoPE models (Gemma 2): Wall not applicable without retraining
- Soft-clamp parameter sensitivity — need to verify `gate_max=0.87` is correct for our block sizes
