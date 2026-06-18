# Plan 286: Functional Attention — Spectral Transport Operator (Open Primitive)

**Date:** 2026-06-17
**Research:** [257_Functional_Attention_Spectral_Transport_Operator](../.research/257_Functional_Attention_Spectral_Transport_Operator.md)
**Source paper:** [arxiv 2605.31559](https://arxiv.org/pdf/2605.31559) — Functional Attention: From Pairwise Affinities to Functional Correspondences (Xiao et al., ICML 2026)
**Target:** `crates/katgpt-core/src/funcattn.rs` (new module) + Cargo feature `funcattn`
**Status:** Active — Phase 1 done (T1.1–T1.5 ✅), Phase 2 done (T2.1–T2.3 ✅ G1+G4+G5 PASS), Phase 3 partial (T3.1 ✅ G3 PASS — sigmoid 33% BETTER than softmax; T3.2 deferred)
**Tier:** Gain (open primitive; await GOAT proof before opt-in promotion; **do not promote to default** until LLM-domain evidence exists)

---

## Goal

Ship Functional Attention (FUNCATTN) as a new attention operator in katgpt-rs. **The paper's math, not the paper's softmax basis** — per AGENTS.md we use sigmoid-normalized basis (partition-of-unity property holds for any row-normalized non-negative kernel, see Research 257 §4).

This is **Gain-tier** because:
- The paper itself has not verified FUNCATTN on NLP / token prediction (§6: "investigating functional attention in domains with less direct function-space interpretations, such as natural language processing, remains a promising future task").
- All math pieces (ridge solve, eigenbasis, sigmoid partition-of-unity) are already in our stack.
- Parallax (Plan 135) is the closest shipped cousin; its audit (2026-05-30) found **NO GAIN** without Muon-trained weights. FUNCATTN may share the same fate.

**Ship the primitive, run an honest GOAT gate, demote if it loses.**

**GOAT gate (must pass before opt-in promotion to default-features list):**
- G1: FUNCATTN with random-init weights produces finite, bounded output for any input ‖X‖≤B (mechanics — Prop 4.5 Lipschitz check)
- G2: FUNCATTN approximates SDPA on a synthetic regression task better than Parallax at fixed parameter budget (Research 257 §2.4 F2 hypothesis)
- G3: sigmoid-basis FUNCATTN ≈ softmax-basis FUNCATTN on PDE-style proxy (no accuracy loss from sigmoid swap)
- G4: linear-in-n scaling verified at n ∈ {512, 2048, 8192} (per paper Fig 5)
- G5: zero-alloc hot path — single forward pass reuses pre-allocated scratch, no per-call Vec allocation

**Out of scope (this plan):**
- LLM-domain token-prediction GOAT gate (await Research 257 §5 Q2 — needs real LM weights, deferred until evidence exists)
- riir-ai integration (that's Plan 318 — rank-k latent functor upgrade, primary value path)
- Sink-aware attention wiring (closed as not-applicable — see [Research 261](../.research/261_FuncAttn_Sink_Semantics_Verdict.md); FuncAttn's `Φ · C · Ṽ` structure has no `n×n` attention matrix for the sink classifier to scan)

---

## Phase 1 — Skeleton (CORE)

Minimal module, behind feature flag, not in default features.

### Tasks

- [x] **T1.1** Add `funcattn` feature to `katgpt-rs/Cargo.toml` and `katgpt-rs/crates/katgpt-core/Cargo.toml`. **Not in default features.** Add to `full` feature aggregation.
- [x] **T1.2** Create `crates/katgpt-core/src/funcattn.rs` with the core types.
  ```rust
  pub enum FuncAttnBasis {
      /// Paper Eq. 9 / code L60: Φ = Softmax(Linear(X) / τ) along k-dim.
      /// τ is learnable per-head, clamped [0.1, 5.0] (code L13, L61).
      Softmax,
      /// AGENTS.md compliance: Φ = Sigmoid(Linear(X) · β) then row-normalize.
      /// Partition-of-unity still holds (any row-normalized non-negative kernel).
      /// β plays the role of 1/τ.
      Sigmoid,
  }

  pub struct FuncAttnConfig {
      pub k: usize,                  // basis dimension, paper default 64
      pub basis: FuncAttnBasis,      // default Sigmoid
      /// Convex-combo regularization coefficient α = sigmoid(alpha_param).
      /// Code uses (1-α)·K̃ᵀK̃ + α·I — NOT paper Eq. 7's additive K̃K̃ᵀ+λI.
      /// See Research 257 §6 Discrepancy 1. Bounded spectrum for α∈(0,1).
      pub alpha: f32,                // default 0.5 (sigmoid(0)); range (0.01, 0.99)
      /// Per-head learnable temperature τ ∈ [0.1, 5.0] (code L13, L61).
      /// For Sigmoid basis, reinterpreted as inverse slope β = 1/τ.
      pub temperature: f32,          // default 0.5 (matches code init)
      pub transpose_proj: bool,      // paper Rem 4.1: use Φᵀ not Φᵀ⁺. Default true.
  }

  pub struct FuncAttnScratch {
      // Pre-allocated scratch buffers for zero-alloc hot path:
      // phi (n×k), psi (n×k), q_tilde (k×d), k_tilde (k×d), v_tilde (k×d),
      // ktk (d×d DUAL FORM — not ktkt k×k primal!), reg (d×d), z_op (d×k),
      // c_op (k×k), pv (n×d), scores (n×k)
      // Note: code uses d×d dual form because d ≤ k typically; see Research 257 §6.
  }
  ```
- [x] **T1.3** Implement `compute_basis_into(x, w, bias, n, d, k, kind, temperature, out)` — writes row-normalized basis to `out: &mut [f32]` of length `n*k`. Zero-alloc.
- [x] **T1.4** Implement `funcattn_forward(x_basis, x_value, w_basis, w_q, w_k, w_v, cfg, scratch, out)`. Follows reference code (Functional_attention.py L50-89):
  - **Basis computation**: `Φ = softmax_or_sigmoid(w_basis(x_proj) / τ)` where `x_proj = w_phi(x)` (NOT the same as the value projection — see code L17-18, two separate Conv2d layers `in_project_x` and `in_project_fx`).
  - **Slice tokens** (code L62-64): `slice_token[g] = Σ_n Φ[n,g] · fx_mid[n] / (Σ_n Φ[n,g] + ε)` — this is a weighted average, not just a projection. The code normalizes by column sum.
  - **Project**: `Q̃ = slice_token_q`, `K̃ = slice_token_k`, `Ṽ = slice_token_v` after applying `to_q`, `to_k`, `to_v` linear layers (these are separate from the basis projection).
  - **Operator solve (DUAL FORM — code L71-76)**:
    - `kH = K̃ᵀ` (d×k)
    - `K̃ᵀK̃ = kH · K̃` (d×d — DUAL, not k×k primal)
    - `reg = (1-α)·K̃ᵀK̃ + α·I_d`  ← **convex combo, not additive**
    - `Z = solve(reg, kH)` solving `reg · Z = K̃ᵀ` (d×k)
    - `C = Q̃ · Z` (k×k)
  - **Apply**: `out_slice = C · Ṽ` (k×d), then `out = Φ · out_slice` (n×d) — inverse projection via the SAME basis weights Φ used in forward slice.
  - All in `scratch`, output to caller-owned `out: &mut [f32]`
  - **Orthogonal init** for `w_basis` (code L20-21: `torch.nn.init.orthogonal_`) — document in module doc, applied by caller (we don't init weights in inference paths).
- [x] **T1.5** Reuse `crates/katgpt-core/src/simd.rs` for matmuls. Added `pub fn solve_convex_combo_dual(...)` helper to `funcattn.rs` — forms `(1-α)·K̃ᵀK̃ + α·I_d` and solves via in-place Cholesky (vendored, MIT).

---

## Phase 2 — Mechanics Gate (no accuracy claim yet)

### Tasks

- [x] **T2.1 (G1)** `g1_lipschitz_bounded`: implemented as `g1_finite_output_random_inputs`, `g1_sweep_input_norm_and_alpha`, and `g1_lipschitz_bounded`. Finite output for B ∈ {1, 10, 100} and α ∈ {0.01, 0.5, 0.99}. **Caveat:** Prop 4.5 is stated for the additive-λ primal form; the convex-combo dual form's Lipschitz bound is a function of α/(1-α). We check finiteness, not the exact C₁/λ + C₂/λ² scaling.
- [x] **T2.2 (G4)** `g4_linear_in_n_scaling`: **DONE 2026-06-17.** Bench `benches/funcattn_scaling_bench.rs` (std::time::Instant, harness=false). Slope of `log(time) vs log(n)` over n ∈ {2048, 8192, 32768} = **0.9407** (target [0.85, 1.15]) → **PASS**. At n=8192 FUNCATTN is **66.56×** faster than `tiled_attention` (17.9ms vs 1191ms). Full table in `.benchmarks/058_funcattn_goat.md` “G4 Results”.
  - n ∈ {512, 2048, 8192, 32768}, d=128, k=64.
  - Measure forward time. Assert linear scaling (R² > 0.95 on log-log fit, slope ≈ 1.0).
  - Compare against `tiled_attention` baseline — at n=32768, FUNCATTN should be >10× faster.
- [x] **T2.3 (G5)** `g5_zero_alloc`: **DONE 2026-06-17.** Test `tests/funcattn_g5_zero_alloc.rs` mirrors the `bench_275_swir_goat.rs::g7_step_zero_allocation_debug` pattern (debug-only `TrackingAllocator`). After 50 warmup calls, **0 allocations / 0 bytes** over 100 measured `funcattn_forward` calls (d=128, k=64, n=512). Release path exercises the same hot path with a timing sanity check.
  - Run `cargo test --features funcattn` with allocator counting (or `cargo bench` with `--bench allocator_count` if available).
  - Assert 0 allocations per forward call after warmup.

---

## Phase 3 — Accuracy Gate (the actual GOAT decision)

### Tasks

- [x] **T3.1 (G3 — sigmoid vs softmax)** `g3_sigmoid_matches_softmax`: **DONE 2026-06-18.** Test `tests/funcattn_g3_sigmoid_vs_softmax.rs`. Tiny model (n=32, d=8, k=4) trained 1000 steps via central-FD SGD on a synthetic Burgers-like regression. Sigmoid **outperforms** softmax at matched hyperparameters (rel-L2 0.087 vs 0.130, ratio 0.67). See `.benchmarks/058_funcattn_goat.md` G3 Results. Key finding: sigmoid needs τ=0.1 (sharp slope, lower bound of reference clamp [0.1,5.0]) to produce non-uniform row distributions at small input scales. At τ=0.5 (reference default), sigmoid fails to learn because sigmoid(2·s) on unit-scale inputs produces near-uniform Φ after row-norm. This is a temperature-scale mismatch, NOT a fundamental sigmoid deficiency — documented in the benchmark doc with implications for callers.
  - Synthetic PDE proxy: Burgers-equation-style dataset (paper §5.6 setup).
  - Train two FUNCATTN models (softmax basis vs sigmoid basis) for 1000 steps with identical seeds.
  - Assert sigmoid model's relative L2 error ≤ softmax model's + 5%.
  - **If sigmoid is >10% worse**: we have a problem (AGENTS.md says sigmoid, but if it doesn't work here, escalate as issue).
- [ ] **T3.2 (G2 — vs Parallax)** `g2_beats_parallax_on_regression`:
  - Sinusoidal few-shot regression (paper §5.1 setup, Fig 2).
  - Compare FUNCATTN vs Parallax (sigmoid) vs SDPA at matched parameter count.
  - Assert FUNCATTN MSE ≤ Parallax MSE × 0.5 AND FUNCATTN MSE ≤ SDPA MSE × 0.1.
  - This is the **paper's headline result** — we should reproduce it.
  - **Reference implementation**: `.raw/FUNCATTN/Few-Shot-Regression/models.py::FuncAttn` (L123-176). Port this exact architecture including the convex-combo regularization (`reg = (1-self.ridge)*kkH + self.ridge*I`, L173) — NOT the paper Eq. 7 additive form. The few-shot code uses fixed `ridge=1e-4` (not learnable α); match that for direct reproduction.

---

## Phase 4 — Verdict

### Tasks

- [ ] **T4.1** Write `katgpt-rs/.benchmarks/046_funcattn_goat.md` with G1–G5 results.
- [ ] **T4.2** If G1, G3, G4, G5 pass AND G2 shows FUNCATTN beats Parallax → **promote `funcattn` to opt-in (in `full` aggregation, NOT in default features)**. Document in `.docs/01_overview.md` Feature Flags table.
- [ ] **T4.3** If G2 fails (FUNCATTN does not beat Parallax on regression) → keep feature flag, document null result, **do not promote**. Note that the paper's gain is PDE-specific and may not transfer to our domains.
- [ ] **T4.4** **Do NOT promote to default until LLM-domain token-prediction evidence exists.** This is a separate gate (deferred per Research 257 §5 Q2).

---

## Phase 5 — Composition (post-GOAT only)

If Phase 4 promotes, wire composability. Each opt-in.

### Tasks

- [ ] **T5.1** Compose with SpectralQuant: pre-rotate basis weights via `calibrate_eigenbasis`. Hypothesis: eigenbasis-aligned FUNCATTN basis is more expressive per parameter.
- [ ] **T5.2** Compose with CHIAR (Plan 269): route between FUNCATTN and Parallax by per-token spectral entropy. FUNCATTN for low-entropy (structured) tokens, Parallax for high-entropy (chaotic) tokens.
- [ ] **T5.3** Compose with freeze/thaw: version basis snapshots `W_Φ, W_Ψ` as atomic Arc-swapped, BLAKE3-committed. Per-domain basis hot-swap. (This is the bridge to riir-ai Plan 318.)

---

## Files

- `crates/katgpt-core/Cargo.toml` — `funcattn` feature
- `crates/katgpt-core/src/funcattn.rs` — new module
- `crates/katgpt-core/src/lib.rs` — `#[cfg(feature = "funcattn")] pub mod funcattn;`
- `Cargo.toml` — top-level `funcattn = ["katgpt-core/funcattn"]`
- `benches/funcattn_scaling_bench.rs` — G4 linear-in-n scaling bench (T2.2)
- `tests/funcattn_g5_zero_alloc.rs` — G5 zero-allocation gate (T2.3)
- `tests/funcattn_g3_sigmoid_vs_softmax.rs` — G3 sigmoid-vs-softmax basis gate (T3.1)
- `.docs/01_overview.md` — Feature Flags table entry (Phase 4 if promoted)

## Open Questions

1. **Cholesky source.** Vendor minimal k×k Cholesky (clean, MIT-compatible) or reuse `riir-gpu/schur.rs` (Apache-2.0, requires attribution header)? Vendor is simpler for the public engine. ~30 lines.
2. **PDE proxy data.** Do we have a Burgers-equation dataset, or do we generate one synthetically? Paper uses Kovachki et al. 2023 benchmark — we'd need to either download or generate. For G2/G3, synthetic sinusoidal regression (paper §5.1) is sufficient and self-contained.
3. **Training loop for G2/G3.** The basis matrices `W_Φ, W_Ψ` need to be trained. This is technically "training" but it's standard transformer training (AdamW on a small model), not a new training method. Acceptable per skill constraint §1 ("no LLM training" refers to fine-tuning base LLMs, not training small diagnostic models for GOAT gates).

## Constraints Check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ C solve is closed-form given trained W_Φ,W_Ψ |
| Latent-to-latent preferred | ✅ All in spectral space; only final `Φ·C·Ṽ` returns to raw |
| Sigmoid not softmax | ✅ `FuncAttnBasis::Sigmoid` is the default |
| Freeze/thaw over fine-tuning | ✅ W_Φ,W_Ψ are swappable snapshots (Phase 5.3) |
| 3-repo discipline | ✅ Open primitive in katgpt-rs; no game IP; no training know-how |
| Zero-alloc hot path | ✅ `FuncAttnScratch` pre-allocated; all `_into` APIs |
| CPU/SIMD first | ✅ All matmuls via `simd_matmul_rows`; Cholesky is k×k (L1-resident for k=64) |
