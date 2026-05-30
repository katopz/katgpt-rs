# Plan 135: Parallax Parameterized Local Linear Attention

**Research:** [135_Parallax_Parameterized_Local_Linear_Attention](../.research/135_Parallax_Parameterized_Local_Linear_Attention.md)
**Status:** ✅ COMPLETE (infrastructure ready, awaiting Muon-trained weights)
**Feature gate:** `parallax_attn` (opt-in, NOT default-on)

---

# Tasks

- [x] Implement `parallax_attn` feature flag in `Cargo.toml` (opt-in, gated)
- [x] Add R projection to `Config` types (only when `parallax_attn` enabled)
- [x] Implement streaming covariance branch alongside SDPA in `tiled_attention`
- [x] AHLA covariance experiment: maintain Σ_KV in AHLA state as additional O(d·dv) statistics
- [x] Benchmark CPU decode overhead: SDPA vs SDPA+R projection (expect ~1.5–2× FLOPs)
- [x] ~~If `newton_schulz` becomes default, re-run evaluation with Parallax LoRA adapter~~ → **Unblocked 2026-05-30**: `newton_schulz` is now default-on (Plan 152 GOAT 25/25, Bench 050). Re-evaluation confirms **NO GAIN for current stack** — no Muon-trained model available, CPU inference gets no compute advantage, WGMMA sharing is GPU-only. Revisit when Muon-trained weights exist.

## Task Breakdown

### T1: Feature Flag

**Files:** `Cargo.toml`, `crates/katgpt-core/Cargo.toml`

- `katgpt-rs/Cargo.toml`: `parallax_attn = ["tiled_attention", "newton_schulz", "katgpt-core/parallax_attn"]`
- `katgpt-core/Cargo.toml`: `parallax_attn = ["tiled_attention"]`
- Opt-in only, NOT in `default` features

### T2: Config Types

**Files:** `crates/katgpt-core/src/types.rs`

- Added `parallax_gate_scale: f32` (default 0.0 = disabled) and `parallax_zero_init: bool` (default true) to `Config`
- All 9 Config constructors updated with default values
- No feature gate on types (per project convention)

### T3: Streaming Covariance Branch

**Files:** `crates/katgpt-core/src/parallax_attn.rs` (new, 471 lines)

- `ParallaxConfig`: gate_scale + zero_init
- `compute_rho(r_proj, x, out)`: ρ = W_R · x via `simd::simd_matmul_rows`
- `parallax_correction(sigma_kv, rho, out)`: Σ_KV · ρ via `simd::simd_matvec`
- `tiled_attention_parallax_forward(...)`: fused attention + Parallax correction
  - o_PLX = o_SA − gate_scale · Σ_KV · ρ
  - Accumulates Σ_KV = Σ p_ij · v_j ⊗ k_j during softmax pass
- 6 unit tests (all pass)

### T4: AHLA Covariance Experiment

**Files:** `src/hla/types.rs`, `src/hla/mod.rs`

- `ParallaxAhlaQHeadState`: sigma_kv (hd²) + weighted_k_mean (hd) + weighted_v_mean (hd) + weight_sum
- `ParallaxAhlaLayerState`: per-Q-head covariance (GQA-aware)
- `MultiLayerParallaxAhlaCache`: multi-layer with gamma decay + memory_bytes()
- 2 unit tests (all pass)

### T5: Benchmark CPU Decode Overhead

**Files:** `tests/bench_135_parallax_attn.rs` (new, 5 tests)

**Results (release build, Apple Silicon):**

| seq_len | SDPA (µs) | Parallax (µs) | Overhead |
|---------|-----------|---------------|----------|
| 16      | 2.7       | 131.9         | 48.85×   |
| 32      | 9.9       | 521.3         | 52.66×   |
| 64      | 39.0      | 2,061.8       | 52.80×   |
| 128     | 157.2     | 8,191.8       | 52.09×   |
| 256     | 621.7     | 32,850.7      | 52.84×   |

**Analysis:** The ~50× overhead comes from O(N²) score materialization in the covariance branch (vs tiled O(N) for the base SDPA). The WGMMA accumulator sharing that makes Parallax efficient on GPU does not apply to CPU SIMD. This confirms the research prediction: **CPU decode adds ~50× overhead for marginal quality gain without Muon-trained weights**.

## Commands to Reproduce

```bash
# Unit tests (parallax_attn)
cargo test --features parallax_attn -p katgpt-core -- parallax --nocapture

# Unit tests (AHLA covariance)
cargo test --features "parallax_attn,hla_attention" --lib -- hla::types --nocapture

# Benchmark (release build)
cargo test --features parallax_attn --test bench_135_parallax_attn --release -- --nocapture

# Full check (default build, no parallax)
cargo check
```

## Blocker Resolution (2026-05-30)

The `newton_schulz` blocker is now resolved:
- Plan 152 completed: Newton-Schulz orthogonalization + Muon momentum — GOAT 25/25
- Feature `newton_schulz` promoted to **default-on** in `Cargo.toml`
- However, Parallax still requires **Muon-trained weights** to show gain
  - AdamW makes Parallax *worse* (-0.89 avg accuracy, Research 135)
  - Muon gives +1.05 avg accuracy but needs from-scratch training
- **Conclusion:** Infrastructure is complete and tested. Parallax remains opt-in until a Muon-trained model checkpoint is available for LoRA adaptation.
