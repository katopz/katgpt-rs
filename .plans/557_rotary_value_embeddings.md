# Plan 557: RoVE — Rotary Value Embeddings Attention (Modelless)

**Date:** 2026-07-22
**Research:** [katgpt-rs/.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md](../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md)
**Source paper:** [arXiv:2606.11275](https://arxiv.org/abs/2606.11275) — García-Castellanos, Weiler, Bekkers, Jul 2026 (RoVE)
**Source code:** [github.com/AGarciaCast/RoVE](https://github.com/AGarciaCast/RoVE)
**Target:** `katgpt-rs/crates/katgpt-core/src/rotary_value_embedding.rs` (new module, sibling to `position_group_action.rs`) + Cargo feature `rotary_value_embedding` (re-exported from root `katgpt-rs/Cargo.toml` as `rotary_value_embedding = ["katgpt-core/rotary_value_embedding"]`). Wiring in `katgpt-rs/crates/katgpt-attn/src/` (Phase 3) + `katgpt-rs/crates/katgpt-attn-match/src/chunked.rs` (Phase 4).
**Status:** All phases DONE. Substrate (Phases 1-4) + wiring (T3.B) + retrofit PoC (Phase 5 partial, Option 2). **Verdict: RoVE retrofit HURTS perplexity (+12.5% loss, +48.2% ppl) on RoPE-trained checkpoints.** Feature stays opt-in permanently. The C control (RoVE-trained-from-scratch, ~~riir-train Issue 379~~ closed 2026-07-26) remains a non-blocking validation follow-up.

> **Numbering note.** Research note 452 + Plan 557 deliberately use *different* numbers because `.research/` and `.plans/` are independent namespaces with independent highwater markers — `.research/` was at 451 (next free = 452), `.plans/` was at 556 (next free = 557). The number collision in `.plans/452` (`452_simd_lut_dequant.md` already exists) is what forces the plan to 557. The research note at `.research/452_*.md` is the design doc; this plan is the execution tracker.

---

## Goal

Distill Research 452 into a generic, modelless, MIT-licensed Rust module that applies the RoPE rotation family to attention **values** (in addition to the existing Q/K rotations) and inverse-rotates the aggregated output back into the query's frame. Concretely, given:
- the V projection `V[j] = attn_wv · x[j]` (already computed by every attention variant),
- the existing RoPE rotation family `{R_t}` (already computed via `RopeFreqs` / `MixedRopeSummarizer` / `RopeAction`),

RoVE produces:

```
# Per-position value rotation (before softmax aggregation):
V_rot[j] = R_j · V[j]                              ← rotate_values_into(j, V[j], &mut V_rot[j])

# Post-aggregation inverse rotation (after softmax(A) · V_rot):
ỹ[i] = R_{−i} · Σ_j softmax(A)_ij · V_rot[j]       ← inverse_rotate_output_into(i, aggregated, &mut ỹ[i])
```

The two-step composition collapses to `ỹ[i] = Σ_j A_ij · (R_{j−i} · W_V) · x[j] = Σ_j A_ij · ψ_{j−i} · x[j]` — an attentive convolution with offset-indexed block-Toeplitz kernel `ψ_δ = R_δ · W_V` (paper Eq. 3). Standard RoPE is recovered when both calls are no-ops (the feature-off path).

**Sibling to GRAPE Plan 446 Issue 160** (`PositionGroupAction` trait), not a replacement. RoVE is the **first concrete hot-path consumer** of that trait — turning GRAPE's documented "vocabulary bridge for cold-path tools" into a real attention variant. The trait provides `apply_at(n, x, out)` and `apply_inverse_at(n, x, out)`; RoVE adds the wiring that calls these on the V projection and the post-softmax output.

**Parameter-free, FlashAttention-compatible, `O(nd)` linear overhead.** The paper reports consistent gains over RoPE on in-context learning, OOD perplexity (64% reduction at 16k tokens), and long-context retrieval (RULER mean NLL 6.62 → 4.33 at 354M with YaRN) at both 124M and 354M scales.

**Honest scope caveat (Research 452 §3 "Honest caveat").** The paper validates RoVE as a training-time architectural choice (model trained from scratch with V rotation). It does NOT validate RoVE as an inference-time retrofit onto RoPE-trained checkpoints. Plan 557 ships the primitive for forward-compat (scenario 1: RoVE-trained upstream checkpoints) + benchmarks the retrofit (scenario 2) as a Phase 5 honest PoC. Promotion to default-on requires Phase 5 to show no regression.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

Goal: a compiling, tested, feature-gated module that implements the two RoVE primitives (`rotate_values_into`, `inverse_rotate_output_into`) as zero-allocation wrappers around `RopeAction::apply_at` / `apply_inverse_at`. No attention forward path wiring yet (Phase 3).

### Tasks

- [x] **T1.1** Add feature flag `rotary_value_embedding = ["position_group_action"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml` features section (near `position_group_action`). Add a root-level alias `rotary_value_embedding = ["katgpt-core/rotary_value_embedding"]` to `katgpt-rs/Cargo.toml` (mirror the GRAPE Issue 160 root-facade pattern). **CORRECTION (implementation-time):** the original plan claimed RoVE does NOT imply `position_group_action` — this was wrong. `RopeAction` lives inside the `position_group_action` module which is `#[cfg(feature = "position_group_action")]`-gated at the MODULE level (`pub mod position_group_action`), not at the re-export level. If the feature is off, `RopeAction` does not exist at all, so RoVE cannot compile without it. The feature therefore implies `position_group_action` (which transitively implies `grapem_rodrigues`). This was verified during implementation — the corrected dependency is in the Cargo.toml comment.
- [x] **T1.2** Add `#[cfg(feature = "rotary_value_embedding")] pub mod rotary_value_embedding;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (near `position_group_action`).
- [x] **T1.3** Implement `RoVeConfig` struct in `rotary_value_embedding.rs`:
  - `theta: f32` (=10000.0 default — matches paper's RoPE base and our existing `RopeFreqs` / `MixedRopeSummarizer`).
  - No other config — RoVE is parameter-free. The config exists only to thread `theta` (for YaRN-style rescaling in future work, see paper Appendix C).
- [x] **T1.4** Implement `pub fn rotate_values_into(action: &RopeAction, pos: usize, values: &[f32], out: &mut [f32])`:
  - Wraps `action.apply_at(pos as f32, values, out)`.
  - **Zero allocation.** Caller-owned `out` buffer (length `d`).
  - **Semantics:** rotates the V projection at position `pos` into the global frame. Mathematically `V_rot[pos] = R_pos · V[pos]`.
- [x] **T1.5** Implement `pub fn inverse_rotate_output_into(action: &RopeAction, pos: usize, aggregated: &[f32], out: &mut [f32])`:
  - Wraps `action.apply_inverse_at(pos as f32, aggregated, out)`.
  - **Zero allocation.** Caller-owned `out` buffer (length `d`).
  - **Semantics:** rotates the softmax-aggregated output at query position `pos` from the global frame back into the query's local frame. Mathematically `ỹ[pos] = R_{−pos} · aggregated[pos]`.
- [x] **T1.6** Implement `pub fn batch_rotate_values_into(action: &RopeAction, positions: &[usize], values: &[f32], out: &mut [f32], dim: usize)`:
  - Loops over `positions.len()` tokens, calling `rotate_values_into` per token.
  - `values` and `out` are flat `[n * d]` row-major; per-token slice = `[token_idx * dim .. (token_idx + 1) * dim]`.
  - **Zero allocation** in the loop (per-token slice borrows only).
  - This is the API the attention forward path calls once per layer (Phase 3 wiring).
- [x] **T1.7** Implement `pub fn batch_inverse_rotate_output_into(action: &RopeAction, positions: &[usize], aggregated: &[f32], out: &mut [f32], dim: usize)`:
  - Same as T1.6 for the inverse direction.
- [x] **T1.8** Write unit tests in `rotary_value_embedding.rs` `mod tests`:
  - [x] **G1 mechanics (identity at pos 0):** `rotate_values_into(action, 0, v, out)` writes `v` to `out` (rotation by angle 0 is identity).
  - [x] **G2 mechanics (round-trip):** `rotate_values_into(action, p, v, tmp)` followed by `inverse_rotate_output_into(action, p, tmp, recovered)` recovers `v` to f32 precision.
  - [x] **G3 relativity check:** `rotate_values_into(action, j, v, v_at_j)` then `inverse_rotate_output_into(action, i, v_at_j, v_at_i)` produces `R_{j−i} · v` — equivalent to a single `action.apply_at((j − i) as f32, v, v_at_i)`. Verifies the offset-indexed kernel `ψ_{j−i} = R_{j−i} · W_V` claim from the paper (Eq. 3).
  - [-] **G4 zero-degradation when feature off:** architecturally unverifiable from within the feature-gated module (when the feature is off, the module doesn't compile, so no test can run). The guarantee is structural (the `#[cfg]` on the module declaration) and is verified by the Exit Criterion `cargo build` (default features) unchanged. Deferred — not a test, a structural property.
  - [-] **G5 zero-alloc:** `rotate_values_into` and `inverse_rotate_output_into` perform zero heap allocations. Unit test uses the code-inspection pattern (matching `phase_rotation.rs` — can't use `#[global_allocator]` in lib unit tests due to parallel test harness collisions). The empirical `CountingAllocator` audit is deferred to the Phase 2 bench (`benches/rotary_value_embedding_goat.rs`).
  - [x] **G6 batch correctness:** `batch_rotate_values_into` produces identical results to per-token `rotate_values_into` for every token.
  - [x] **G7 odd-dim safety:** RoPE requires even `dim`; `RoVeConfig::build_rope_action` delegates to `RopeAction::with_theta`, which panics on odd `dim`.
- [x] **T1.9** Document module in `rotary_value_embedding.rs` header with:
  - Paper reference (arXiv:2606.11275 + GitHub link).
  - Three-lens summary (convolution / matrix mixer / local frame) from Research 452 §1.2.
  - Sibling-relationship note with `position_group_action` (GRAPE Issue 160) — RoVE is the first hot-path consumer of `RopeAction::apply_at` / `apply_inverse_at`.
  - Inference-only caveat (paper validates training-time only; Phase 5 PoC settles retrofit).
  - FlashAttention-compat note: rotations act on V (pre-kernel) and aggregated output (post-kernel), never on the `n×n` score matrix.

### Phase 1 Exit Criteria
- [x] `cargo build --features rotary_value_embedding -p katgpt-core` compiles clean.
- [x] `cargo test --features rotary_value_embedding -p katgpt-core --lib rotary_value_embedding` passes (G1–G7: 9/9 tests).
- [x] `cargo clippy --features rotary_value_embedding -p katgpt-core --lib` zero warnings (pre-existing test warnings in `bench_453_*` are unrelated).
- [x] `cargo build` (default features) unchanged — RoVE is opt-in, no impact on existing paths.

---

## Phase 2 — GOAT Gate (CORE — required before any wiring)

Goal: prove the primitive is correct, fast, alloc-free, and FlashAttention-compatible. Promotion to default-on DEFERRED until Phase 5 (retrofit PoC) settles the open question.

### Tasks

- [x] **T2.1** Write `benches/bench_557_rotary_value_embedding_goat.rs` with GOAT benchmarks (named per repo convention `bench_NNN_*`, not `rotary_value_embedding_goat.rs`):
  - **G1 bit-identical to RoPE-when-disabled:** pos=0 identity is exact (cos=1, sin=0 in IEEE); round-trip at nonzero pos holds to f32 precision (1 ULP from library cosf/sinf, budget 1e-6). Feature surgical scope verified structurally (module cfg-gated in lib.rs). **PASS** (worst 1.79e-7).
  - **G2 latency overhead:** `batch_rotate_values_into` + `batch_inverse_rotate_output_into` per-layer cost at `n=1024, d=768`. Target: `< 5%` of `O(nd²)` V projection via `types::math::matmul`. **FAIL** (6.45%) — honest: scalar rotation (~0.7 GFLOP/s) vs SIMD matmul (~17 GFLOP/s) is a ~24× throughput gap that inflates the 0.13% FLOP ratio to 6.45% wall-clock. SIMD RoVE (Phase 3) is the unblock path. Gate NOT relaxed.
  - **G3 no-regression:** opt-in + additive; default build clean; 9/9 Phase 1 tests pass with feature on. **PASS**.
  - **G4 zero steady-state alloc:** `CountingAllocator` over 1000 calls on batch hot path — **PASS** (0/0).
  - **G5 FlashAttention output-equivalence:** two-path comparison (RoVE: rotate V → aggregate → inverse-rotate; reference: per-(i,j) R_{j−i} rotation) on n=16, d=32 random fixture. **PASS** (rel err 2.69e-8 < 1e-4 budget). Proves `(R_{−i} · Σ_j A_ij · R_j · V_j) = (Σ_j A_ij · R_{j−i} · V_j)`.
- [x] **T2.2** Run the GOAT gate. Record results in `.benchmarks/557_rotary_value_embedding_goat.md`. Honest reporting: G2 FAIL recorded as ❌ with the actual numbers (6.45% vs 5%) + documented root cause (scalar-vs-SIMD throughput gap). No target relaxation.

### Phase 2 Exit Criteria
- [x] G1, G3, G4, G5 PASS. G2 honest ❌ (6.45% vs 5%) with documented reason (scalar throughput gap) + deferral to Phase 3 SIMD work.
- [x] `.benchmarks/557_rotary_value_embedding_goat.md` written.
- [x] **Promotion deferred** — two independent blockers: G2 FAIL + Phase 5 retrofit PoC not done.

---

## Phase 3 — Hot-Path Wiring in `katgpt-attn` (CORE)

Goal: add an opt-in forward path that calls `rotate_values_into` and `inverse_rotate_output_into` from a real attention variant. Mirror the existing RoPE-on-QK call site in `dash_attn/forward.rs`.

### Tasks

- [-] **T3.1** ~~Identify the RoPE-on-QK call site in `katgpt-attn/src/dash_attn/forward.rs`.~~ **MOOT** — see Phase 3 ACTUAL OUTCOME below. The target (`dash_attn/forward.rs`) is an MVP stub that never applies RoPE to Q/K and doesn't do real attention. The real RoPE-on-QK call site is in `riir-engine/src/transformer/gemma2.rs` (riir-ai, cross-repo). Replaced by T3.A (substrate G2 unblock, DONE) + T3.B (cross-repo wiring to riir-ai, DEFERRED pending Phase 5).
- [-] **T3.2** ~~Add an opt-in RoVE branch in the same forward path.~~ **MOOT** — the target forward path doesn't exist as a real attention implementation. See T3.B for the real wiring target.
- [-] **T3.3** ~~Repeat T3.2 for `forward_dash_attn_decode`.~~ **MOOT** — same reason as T3.2.
- [-] **T3.4** ~~Add a feature-gated `RoVeToggle` to the dash_attn `Config`.~~ **MOOT** — the toggle belongs in the riir-ai consumer, not the katgpt-attn stub. See T3.B.
- [-] **T3.5** ~~Write integration tests in `katgpt-attn`.~~ **MOOT** — the substrate-level G8 tests (forward bit-identical to scalar, inverse ≤1 ULP) already landed in T3.A. Integration tests for the real wiring belong in riir-ai (T3.B).

### Phase 3 Exit Criteria
- [-] ~~`cargo build --features rotary_value_embedding,dash_attn -p katgpt-attn` compiles clean.~~ **MOOT** — the dash_attn forward path is a stub; the real wiring target is riir-ai's gemma2.rs (T3.B).
- [-] ~~`cargo test --features rotary_value_embedding,dash_attn -p katgpt-attn` passes.~~ **MOOT** — substrate tests already pass (14/14 in katgpt-core); integration tests belong in riir-ai.
- [-] ~~The dash_attn forward path supports RoVE as an opt-in toggle.~~ **MOOT** — replaced by the substrate-level fast path (`batch_rotate_values_into_fast` / `batch_inverse_rotate_output_into_fast`) which is the API the real consumer (riir-ai gemma2.rs) will call.

### Phase 3 ACTUAL OUTCOME (2026-07-22 — reframed)

**T3.1 PREMISE WAS WRONG.** The plan assumed `katgpt-attn/src/dash_attn/forward.rs` applies RoPE to Q/K and only needs a V-side extension. Investigation found:

1. `dash_attn/forward.rs` is an **MVP stub** — it computes QKV projections but never applies RoPE to Q/K, and it doesn't do real attention (no QK^T → softmax → V aggregation). The "attention" is just `attn_out = W_o · q` (a pass-through).
2. The real RoPE-on-QK call site is in **riir-ai** (`riir-engine/src/transformer/gemma2.rs` line ~330: `crate::rope::apply_rope_with_freq(&mut ctx.q, &mut ctx.k, pos, hd, ...)`), NOT in katgpt-rs.
3. RoPE appears in katgpt-rs only in: (a) KV cache compaction (`katgpt-kv/src/shard_kv` — undo/reapply to strip position before PCA), (b) attention matching (`katgpt-attn-match/src/chunked.rs` — `apply_rope_phase_shift` on keys during compaction), (c) a standalone fused kernel (`simd_matmul_rmsnorm_rope`) that is never called from any production forward path.

**The Research 452 §2.2 claim "✅ in every attention variant (dash_attn, gdn2, ega, hga)" was overconfident** — it confused the existence of RoPE utilities with their use in production forward paths. The reality: katgpt-rs is the **public substrate** repo; the **production attention runtime with RoPE lives in riir-ai**.

**What Phase 3 ACTUALLY delivered (the G2 unblock):**

Instead of wiring into a non-existent attention path, Phase 3 delivered the **G2 unblock at the substrate layer**:

- [x] **T3.A (substrate — DONE):** Added `RoVeRotationTable` + `batch_rotate_values_into_fast` + `batch_inverse_rotate_output_into_fast` to `katgpt-core/src/rotary_value_embedding.rs`. The table precomputes cos/sin for all `(position, pair)` combinations once, eliminating the per-call transcendental cost (the dominant bottleneck). The fast path inner loop is pure `mul_add` arithmetic.
  - **G2 result:** 6.62% → **2.29%** (2.88× speedup). **PASS** (< 5% target).
  - **G8 tests:** forward direction bit-identical to scalar (tol 0.0); inverse direction ≤1 ULP (tol 1e-6, the same ULP floor as Phase 2 G1's round-trip budget).
  - **Memory cost:** 3 MB table for n=1024, d=768 (amortized across all layers).

- [x] **T3.B (wiring — DONE, riir-ai commit `cd62e6cd6` on develop 2026-07-22):** The wiring target is `riir-engine/src/transformer/gemma2.rs` (the real production attention path with RoPE-on-QK). This is a **cross-repo task** — katgpt-rs provides the substrate (Phase 1 + 2 + 3.A), riir-ai consumes it. **Critical finding:** the katgpt-core `RopeAction` uses adjacent-pair rotations `(2i, 2i+1)` while riir-engine's RoPE uses rotate-half `(i, i+half)` — different rotation subgroups. The forward-path V rotation MUST use the same convention as Q/K for the paper's equivalence claim to hold. **Resolution:** riir-engine ships its own `apply_rope_values` + `apply_inverse_rope_output` (rotate-half, convention-consistent), NOT katgpt-core's `RopeAction`. The katgpt-core substrate remains valuable for attention-matching compaction tests (Phase 4 G9/G10) + benchmarking. Wired across 4 forward paths (per-token decode + block-causal prefill Phase A/B + instrumented forward). 3111/3111 tests pass with feature on; 3103/3103 with feature off. Full wiring details are documented inline in `riir-engine/src/rope.rs` (the "RoVE convention note" doc-comment block on `apply_rope_values`).

**Net Phase 3 outcome:**
- G2 **UNBLOCKED** (6.45% → 2.29%).
- The substrate now ships both scalar (reference, bit-identical contract) + fast (production, precomputed table) paths.
- The original T3.1-T3.5 tasks (wire into dash_attn) are **moot** — the target was a stub. The real wiring is cross-repo.
- **One blocker remains for default-on promotion:** Phase 5 retrofit PoC.

---

## Phase 4 — Attention Matching Fusion (modelless, optional)

Goal: when RoVE + Attention Matching both active, fit `C_V` in position-free V space. Mirror the existing key-side `apply_rope_phase_shift` pattern in `katgpt-attn-match/src/chunked.rs`.

### Tasks

- [-] **T4.1** In `ChunkedCompactor::compact_text_based` (`katgpt-attn-match/src/chunked.rs`), the existing key-side path is:
  ```rust
  let pf = PositionFreeBridge::new(ROPE_THETA, d);
  let pos_free_keys = pf.un_rotate_f32(chunk_keys, chunk.start_pos);
  // ... compact pos_free_keys ...
  // ... re-rotate at compacted position ...
  ```
  Add a parallel value-side path gated by `#[cfg(feature = "rotary_value_embedding")]`:
  ```rust
  let pos_free_values = pf.un_rotate_f32(chunk_values, chunk.start_pos);
  // ... compact pos_free_values together with pos_free_keys ...
  // ... re-rotate at compacted position ...
  ```
- [-] **T4.2** Update the `ChunkedCompactor` API to accept an optional `rove_active: bool` flag (or a `RoVeToggle`). When true, the compactor un-rotates V before compaction and re-rotates after; when false, the path is unchanged.
- [x] **T4.3** Write integration tests (reframed as verification tests — see Phase 4 ACTUAL OUTCOME):
  - **G9 compaction fidelity under RoVE:** PASS — cosine 0.999925.
  - **G10 position-consistency:** PASS — cosine ≥ 0.991.

### Phase 4 Exit Criteria
- [x] `cargo test --features attn_match,rotary_value_embedding -p katgpt-attn-match` passes.
- [x] The two features compose cleanly with no fidelity regression.

### Phase 4 ACTUAL OUTCOME (2026-07-22 — REFRAMED)

**The plan's original T4.1 approach (un-rotate values before compaction, compact in
position-free space, re-rotate) was found to be MATHEMATICALLY INCORRECT during
implementation. The tasks T4.1 + T4.2 are marked `[-]` (deferred/abandoned) and
NO un-rotate/re-rotate code ships.**

**The mathematical proof (why position-free value compaction is wrong):**

The value fitting (least-squares) minimizes `||A_sel · Cv - A · V||²` where A is
the attention weight matrix and V is the values. When we un-rotate values first,
the fit optimizes `||A_sel · Cv_plain - A · V_plain||²` — but the actual attention
output uses **rotated** values: `out_i = Σ_j A_ij · R_j · V_plain[j]`. Since `R_j`
varies per position (each token has a different rotation angle), the position-free
objective ≠ the rotated-space objective.

Concretely: `Σ_j A_ij · R_j · Cv_plain[j] ≠ R_k · Σ_j A_ij · Cv_plain[j]` for any
single k, because R_j is different for each j. The position-free fit can't account
for the per-position rotation mixing.

**Measured evidence:** G9 with the un-rotate approach measured cosine **0.17** (vs
0.991 target) — a catastrophic FAIL, confirming the mathematical analysis.

**The CORRECT finding:** the existing `compact_text_based` already handles
RoVE-rotated values correctly — NO special code is needed. The value fitting
(least-squares) operates in whatever space the values are in, so RoVE-rotated
values are fitted correctly by the existing compaction. G9 verified this: compacting
RoVE-rotated values AS-IS gives cosine **0.999925**.

**What shipped (Plan 557 Phase 4):**

1. **`rotary_value_embedding` feature in `katgpt-attn-match`** — pulls
   `katgpt-core/rotary_value_embedding` for test access to RoVE primitives.
2. **Root `Cargo.toml` feature forwarding** — the root `rotary_value_embedding`
   feature now also forwards to `katgpt-attn-match/rotary_value_embedding`.
3. **G9 + G10 verification tests** — confirm the existing compaction is
   RoVE-transparent (cosine ≥ 0.991).
4. **Documentation** on `compact_text_based` noting it handles RoVE-rotated
   values correctly as-is.

**What did NOT ship (and why):**

- No `RoVeToggle` enum, no `compact_text_based_with_rove` method, no un-rotate/
  re-rotate helpers. These were implemented and then reverted after the
  mathematical analysis proved them incorrect.
- No `RoveFeatureOff` error variant.

**Future work (if token relocation is ever needed):** The only case where RoVE-
aware value handling matters is RELOCATION (moving compacted tokens to different
positions). The current code uses `new_pos = start_pos` (no relocation), so the
issue doesn't arise. If relocation is needed in the future, a correct approach
would require fitting in the ROTATED domain at the TARGET positions, not
position-free fitting — a non-trivial extension that's out of scope for Plan 557.

---

## Phase 5 — Honest Retrofit PoC (HONEST CAVEAT — settles the open question)

Goal: settle whether RoVE as an inference-time retrofit onto RoPE-trained checkpoints helps, hurts, or is neutral. This is the question the paper does NOT answer (paper trains from scratch with RoVE; our engine serves upstream checkpoints).

**This phase is honest research, not a feature gate.** The output is a `.benchmarks/557_rove_retrofit_poc.md` document recording the result. No matter the outcome, Phase 5 informs the promotion decision.

### Phase 5 ACTUAL OUTCOME (2026-07-22 — Partial Phase 5 via Option 2)

**The retrofit question is ANSWERED: RoVE retrofit HURTS perplexity. The feature
stays opt-in permanently for RoPE-trained checkpoints.**

Per the plan's recommended path (§Blocker option 2), we used the existing
gemma-2-2b-it checkpoint instead of training a from-scratch toy GPT-2. This
tested A (RoPE-only) vs B (RoVE retrofit) — the core retrofit question. The C
control (RoVE-trained-from-scratch) requires GPU training (riir-train Issue
379) and remains a non-blocking follow-up.

**Measured results** (gemma-2-2b-it, CPU):

| Text | Predictions | Config | Avg Loss | Perplexity | Δ Loss |
|---|---|---|---|---|---|
| Short | 65 | A) RoPE-only | 3.143 | 23.17 | — |
| Short | 65 | B) RoVE retrofit | 3.536 | 34.34 | **+12.5%** |
| Longer (Wizard of Oz) | 162 | A) RoPE-only | 2.364 | 10.64 | — |
| Longer (Wizard of Oz) | 162 | B) RoVE retrofit | 3.292 | 26.89 | **+39.2%** (+153% ppl) |

**Verdict: B > A in loss (worse) on both measurements.** The V rotation
perturbs the OV circuit in a way the RoPE-trained model has not learned to
compensate for. This is the expected result — the paper's equivalence is a
training-time claim, not an inference-time retrofit claim. See
`.benchmarks/557_rove_retrofit_poc.md` for the full analysis + honest caveats.

### Tasks (updated)

- [-] **T5.1** Train a toy GPT-2 on FineWebEdu-10B without RoVE. **DEFERRED** —
  requires GPU training. Option 2 (existing gemma-2-2b-it checkpoint) was used
  instead, which answers the core retrofit question without needing from-scratch
  training. The C control (RoVE-trained-from-scratch) remains a non-blocking
  validation follow-up; ~~riir-train Issue 379~~ closed 2026-07-26 (promotion
  question settled — stay opt-in per A vs B comparison; C control re-file as a
  fresh issue if/when paper-fidelity validation becomes load-bearing).
- [x] **T5.2** Benchmark A (RoPE-only) vs B (RoVE retrofit) using gemma-2-2b-it.
  Harness: `riir-ai/crates/riir-engine/tests/bench_557_rove_retrofit.rs`.
- [-] **T5.3** Measure on Core ICL, OOD perplexity, RULER. **PARTIALLY DONE** —
  perplexity on fixed English text measured (the headline metric). Core ICL +
  RULER require additional dataset infrastructure (math-500, RULER harness);
  the perplexity signal (+12.5% loss) is strong enough to settle the promotion
  question without them.
- [x] **T5.4** Write `.benchmarks/557_rove_retrofit_poc.md` — DONE. A vs B
  comparison recorded with honest verdict.
- [-] **T5.5** (N/A — verdict is "retrofit hurts", not "helps". The promotion path is T5.6, not T5.5.)
- [x] **T5.6** Keep `rotary_value_embedding` opt-in. Documented in
  `.benchmarks/557_rove_retrofit_poc.md`.

### Phase 5 Exit Criteria
- [x] `.benchmarks/557_rove_retrofit_poc.md` written.
- [x] Promotion decision recorded: **stay opt-in** (retrofit hurts, +12.5% loss).

### Blocker (RESOLVED 2026-07-22 via Option 2)

**Phase 5 is blocked on riir-train from-scratch pretraining infrastructure.**

The current `riir-train-gpu::Trainer` is a **LoRA adapter trainer** — its
`Trainer::new()` takes pre-existing `TransformerWeights` and trains LoRA
adapters on top. It does NOT support from-scratch pretraining (random weight
initialization → gradient descent on the full parameter set). T5.1 requires
the latter.

**Options to unblock:**
1. **Extend riir-train-gpu with a from-scratch pretraining loop** — the GPU
   kernels (forward/backward) exist for the transformer layers; what's missing
   is the weight initialization + full-parameter optimizer (AdamW) + data
   pipeline for FineWebEdu tokenization. This is a multi-day infrastructure
   task.
2. **Use an existing RoPE-trained checkpoint** (e.g. `gemma-2-2b-it-f16.gguf`
   in `riir-train/data/`) for a **partial Phase 5** (A vs B only, no C
   control). This would answer the retrofit question (does applying V
   rotation at inference help or hurt?) but cannot validate our RoVE impl
   against the paper (that needs the C control: RoVE-trained from scratch).
   Requires T3.B (RoVE wiring into riir-ai's gemma2.rs forward path) to be
   landed first.
3. **Outsource the training** — use an external pretraining run (e.g. OLMo,
   Pythia) and load the checkpoint. Not aligned with the repo's Rust-native
   training stack, but would give a proper RoPE-trained baseline.

**Recommended path:** option 2 (partial Phase 5 with existing checkpoint) as
an interim measure, with option 1 (riir-train from-scratch loop) tracked as
a separate riir-train issue for the full A/B/C comparison.

**Update (2026-07-22):** T3.B (RoVE wiring into gemma2.rs) is now DONE
(sibling agent, riir-ai commit `cd62e6cd6`). Option 2 is unblocked —
the perplexity comparison harness is the next step.

---

## Constraints

- **Modelless only.** No new parameters; no gradient descent; no training in the primitive itself. (Phase 5 *uses* a trained checkpoint but only to benchmark the inference-time retrofit question — the primitive is still modelless.)
- **Zero allocation in hot paths.** `rotate_values_into` and `inverse_rotate_output_into` are pure float arithmetic over caller-owned buffers. The `ForwardContext` scratch buffers (`ctx.v_rot`, `ctx.attn_out_final`) are pre-allocated once per forward pass, reused across tokens.
- **FlashAttention-compat.** The rotations act on V (before the kernel call) and on the aggregated output (after the kernel call). Never on the `n×n` score matrix. Phase 2 G5 verifies this via the output-equivalence test.
- **Sigmoid, not softmax** (AGENTS.md global rule). RoVE does not introduce any new activation — the softmax is the existing attention softmax. No new scoring function; the value rotation is a linear operation.
- **Even dim only.** RoPE requires even `dim` (per-pair rotation). `RoVeConfig` panics on odd `dim`, mirroring `RopeAction::with_theta`.
- **No YaRN in this plan.** The paper composes RoVE with YaRN frequency interpolation for OOD contexts. YaRN is not shipped in katgpt-rs today (fixed `θ_0 = 10000`). Adding YaRN is a separate future plan; RoVE is fully functional without it (the paper's "RoVE (ours)" row in Table 1 is without YaRN).

---

## Honest caveats (carried from Research 452 §3)

1. **Retrofit is unvalidated by the paper.** Phase 5 is mandatory before any default-on promotion. The structural argument cuts both ways: RoVE makes the OV circuit offset-aware, but the model's `W_V` was trained under the offset-blind assumption. We do not know whether the retrofit helps or hurts until we measure.
2. **Substrate dependency on `position_group_action` feature gate.** Phase 1 T1.1 asserts RoVE does NOT imply `position_group_action`. This is because `RopeAction` is a concrete struct with inherent methods `apply_at` / `apply_inverse_at` (from `impl PositionGroupAction for RopeAction`); the trait dispatch is static. Verify this with `cargo build --features rotary_value_embedding --no-default-features` — it should compile with `position_group_action` off but `RopeAction` reachable.
3. **No new pillar.** RoVE touches transformer attention substrate only. It does not connect to HLA / latent_functor / cgsp_runtime / neuron-shard / LatCal. The verdict is GOAT (engine completeness), not Super-GOAT (game-AI moat).
4. **Phase 5 requires GPU training coordination.** T5.1 is a riir-train task (train the toy GPT-2 baseline). Block Plan 557 Phase 5 on riir-train availability; do not implement a CPU-only toy that wouldn't match the paper's setup.

---

## References

- **Research note:** [`.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md`](../.research/452_RoVE_Rotary_Value_Embeddings_Attentive_Convolution.md)
- **Source paper:** [arXiv:2606.11275](https://arxiv.org/abs/2606.11275) — García-Castellanos, Weiler, Bekkers.
- **Source code:** [github.com/AGarciaCast/RoVE](https://github.com/AGarciaCast/RoVE)
- **Closest cousin plans:**
  - ``.plans/446_GRAPE_Group_Representational_Position_Encoding.md`` — provides the `PositionGroupAction` trait + `RopeAction` that RoVE consumes. **Plan 446 does not exist as a single file** — it landed as Issues 159/160/161/163 (all removed per noise rule; verdicts in `.benchmarks/457`/`458`/`459`/`460`). See Research 446 §4 for the full follow-up record.
  - [`.plans/271_attention_matching_compaction.md`](271_attention_matching_compaction.md) — KV compaction that preserves RoPE on keys; Phase 4 of this plan extends it to RoVE-aware V space.
  - [`.plans/173_wall_attention_diagonal_gate.md`](173_wall_attention_diagonal_gate.md) — Wall Attention (orthogonal axis; Wall replaces RoPE on QK, RoVE extends RoPE to OV).
- **Canonical format example:** [`.plans/271_attention_matching_compaction.md`](271_attention_matching_compaction.md)
