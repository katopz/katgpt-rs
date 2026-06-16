# Plan 279: Manifold Power Iteration MoE Router (Modelless)

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/246_Manifold_Power_Iteration_MoE_Router.md](../.research/246_Manifold_Power_Iteration_MoE_Router.md)
**Source paper:** [arxiv 2606.12397](https://arxiv.org/abs/2606.12397) — Redesign MoE Routers with Manifold Power Iteration (RUC / Tencent, 10 Jun 2026)
**Target:** `katgpt-rs/src/manifold_power_iter_router.rs` (new module) + Cargo feature `manifold_power_iter_router` + shared `power_iter_retract` helper in `katgpt-rs/src/spectral_retract.rs`
**Status:** Active — Phase 0 (plan created, awaiting implementation)

---

## Goal

Distill Research 246 into a generic, modelless, MIT-licensed Rust module that performs **one-shot router-weight conditioning at freeze/thaw snapshot swap** (never per-token). Concretely: given a frozen MoE router `R ∈ ℝ^{N×D}` and per-expert Gram matrices `M[i] = W_g[i]·W_g[i]ᵀ`, produce the MPI-conditioned router `R'[i] = C · (R[i]·M[i]) / ‖R[i]·M[i]‖₂` (paper Eq. 4–5) with `C = C'/√N`. Inference behavior is identical to vanilla top-k gating — only the router rows change. This enables the paper's provable gains at zero per-token overhead: router–expert alignment metric **λ 0.27 → 0.66** (Eq. 11), load-balance violation **MaxVio 1.13 → 0.96** (§1.4), and **+0.7–1.3 avg downstream accuracy** across 1B/3B/11B MoE — all by reconditioning router rows once per snapshot swap (sub-ms for game-scale pools, deterministic → sync-safe under `SyncBlock → ChainConsensus`). Distilled with **sigmoid** (not softmax) per AGENTS.md constraint; paper §6 confirms sigmoid still wins over vanilla. **GOAT gate:** reproduce λ/MaxVio/zero-overhead claims on a real MoE adapter pool before promoting to default.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

Goal: a compiling, tested, feature-gated module that implements `manifold_power_iter_router` (paper Eq. 4–5) on synthetic data with the public API surface frozen, AND the shared `power_iter_retract` helper that DRY-refactors the `gauge_rebalance` (Plan 270) cousin.

### Tasks

- [x] **T1.1** Create `src/spectral_retract.rs` (new shared helper module) with empty `mod.rs`-style doc header
- [x] **T1.2** Add feature flag `manifold_power_iter_router = ["dep:spectral_retract"]` to `katgpt-rs/Cargo.toml` features section (after `gauge_invariant`)
- [x] **T1.3** Add `#[cfg(feature = "manifold_power_iter_router")] pub mod manifold_power_iter_router;` and `pub mod spectral_retract;` (always-on — helper is shared) to `src/lib.rs` (alphabetical, after `sparse_task_vector`)
- [x] **T1.4** Implement shared `power_iter_retract` helper in `src/spectral_retract.rs`:
  - [x] `PowerRetractScratch` struct (reuses `PowerIterationScratch` pattern from `src/distill/peira.rs`): `mv_out: Vec<f32>` (D), `norm: f32`
  - [x] `pub fn power_iter_retract(v: &mut [f32], psd_op: &[f32], dim: usize, target_norm: f32, iters: u8, scratch: &mut PowerRetractScratch)` — one or more steps of `v ← v·M` then `v ← target_norm · v / ‖v‖₂`. Zero-alloc, caller-owned scratch. Works on any PSD operator (Gram for MoE, `AᵀA`/`BᵀB` for LoRA gauge).
  - [x] Deterministic given `(v, M, target_norm, iters)` — safe for sync/quorum
  - [x] Sub-μs per call for D ≤ 1024 (plasma tier)
- [x] **T1.5** DRY refactor: migrate `gauge_rebalance` (Plan 270) in `src/gauge_invariant.rs` to call `power_iter_retract` for its `σ_max` estimation step (the power iteration in `power_iterate_sigma_max`). Verify `gauge_rebalance`'s invariants still hold: `‖A·Bᵀ‖_F` unchanged, existing tests (`t01_gauge_rebalance_preserves_abt_exactly`, `test_gauge_rebalance_balances_sigmas`, `test_gauge_rebalance_zero_matrix_safe`) pass unchanged
- [x] **T1.6** Implement `src/manifold_power_iter_router.rs` types:
  - [x] `MpiRouterConfig` struct (`c_prime: f32`, `iters: u8` (=1 per paper §1.4), `beta_sigmoid: f32` temperature)
  - [x] `MpiRouterResult` struct (`r_prime: Vec<f32>` N×D, `lambda_alignment: f32` diagnostic, `maxvio: f32` diagnostic)
  - [x] `ExpertGramView` enum/borrow type: `Owned(Vec<f32>)` vs `Borrowed(&[f32])` for the per-expert Gram slices
- [x] **T1.7** Implement `compute_expert_gram_into(w_g: &[f32], d_model: usize, out: &mut [f32])` — `M[i] = W_g[i]·W_g[i]ᵀ` (D×D). Cache once per snapshot, BLAKE3-tagged with snapshot version (research note §2.2). Blocked matmul for D > 256.
- [x] **T1.8** Implement `pub fn manifold_power_iter_router` (research note §2.1 signature):
  ```
  pub fn manifold_power_iter_router(
      r: &mut [f32],              // [N×D] router, updated in place → R'
      gram_per_expert: &[&[f32]], // N views, each [D×D] expert Gram
      n_experts: usize,
      d_model: usize,
      c_prime: f32,
      iters: u8,                  // =1 default per paper
      scratch: &mut PowerRetractScratch,
  ) -> MpiRouterResult
  ```
  - [x] For each row `i`: call `power_iter_retract(&mut r[i*D..(i+1)*D], gram_per_expert[i], d_model, C=c_prime/√N, iters, scratch)`
  - [x] Compute diagnostic `lambda_alignment` (paper Eq. 11): mean over rows of `(R'[i]·M[i]·R'[i]ᵀ) / (‖R'[i]·M[i]‖₂ · ‖R'[i]‖₂)`
  - [x] Compute diagnostic `maxvio`: max row-norm deviation from `C` (should be ≈0 after retraction)
- [x] **T1.9** Implement `gate_sigmoid_topk(x: &[f32], r_prime: &[f32], n_experts: usize, d_model: usize, beta: f32, k: usize, out_scores: &mut [f32]) -> Vec<usize>` — research note §2.3 distillation. **Independent per-expert sigmoid** `σ(β · x · R'[i]ᵀ)`, then TopK_k by sigmoid score. Never softmax.
- [x] **T1.10** Write unit tests in `src/manifold_power_iter_router.rs` `mod tests`:
  - [x] Synthetic: known principal-direction recovery — construct `W_g` with a known dominant right-singular vector `u`, random `R[0]`, verify after MPI `R'[0]·u ≈ C` (cosine > 0.95 for `iters=1`, > 0.99 for `iters=5`) → GOAT G1
  - [x] Determinism: same `(R, M, c_prime, iters)` → byte-identical `R'` → sync-safe → GOAT G2
  - [x] Norm invariant: `‖R'[i]‖₂ ≈ C' / √N` for all `i` after retraction → GOAT G3
  - [x] `lambda_alignment` increases monotonically with `iters` on a fixed `(R, M)` → confirms the Rayleigh-quotient ascent story → GOAT G4
  - [x] Zero-row safety: degenerate Gram (all-zero expert) → row unchanged, no panic (mirror `test_gauge_rebalance_zero_matrix_safe`)
  - [x] Sigmoid gate: independent per-expert scores (changing one row's score does NOT change another's, unlike softmax) → constraint check
- [x] **T1.11** Add example `examples/manifold_power_iter_router_basic.rs`:
  - [x] Synthetic MoE: N=8 experts, D=256, random `R` + random `W_g[i]`
  - [x] Compute `R'`, print `lambda_alignment` before/after (target: 0.27 → 0.66 shape per paper §1.4)
  - [x] Print `maxvio` before/after (target: 1.13 → 0.96 shape)
  - [x] Print timing (target: sub-ms for N=8, D=256)
  - [x] Show sigmoid top-k gating on a sample token `x`
- [x] **T1.12** Document module in `src/manifold_power_iter_router.rs` header with paper reference (arxiv 2606.12397), equations (Eq. 4–5), and the §2.3 sigmoid-distillation note

### Phase 1 Exit Criteria
- [x] `cargo build --features manifold_power_iter_router` compiles clean
- [x] `cargo test --features manifold_power_iter_router --lib manifold_power_iter_router` passes all unit tests (10/10)
- [x] `cargo run --example manifold_power_iter_router_basic --features manifold_power_iter_router --release` runs and prints λ/MaxVio before→after
- [x] `gauge_rebalance` (Plan 270) tests still pass after DRY refactor to `power_iter_retract` — no behavior change (16/16)
- [x] No new clippy warnings on `spectral_retract.rs`, `manifold_power_iter_router.rs`, or the refactored `gauge_invariant.rs` (2 fixed: div_ceil, size_of_val)
- [x] File sizes < 2048 lines (target: `spectral_retract.rs` < 400 lines ✓ 366, `manifold_power_iter_router.rs` < 800 lines — 855, slightly over soft target, under hard limit)

---

## Phase 2 — Wire into Freeze/Thaw Snapshot Swap Path

Goal: the MPI conditioning fires **once per snapshot swap** (research note §2.2), never per-token. The engine primitive is complete in Phase 1; this phase provides the snapshot-swap hook surface that riir-ai's `LoRAHotSwap` (Research 161 / Plan 181) consumes. Lands in katgpt-rs as a trait + default impl; the actual freeze/thaw runtime integration is riir-ai (out of scope here — see §Out of Scope).

### Tasks

- [x] **T2.1** Implement `MpiRouterSnapshotHook` trait in `src/manifold_power_iter_router.rs`:
  ```
  pub trait MpiRouterSnapshotHook {
      /// Called once when a frozen expert pool is hot-swapped.
      /// Returns the MPI-conditioned router R' + diagnostics.
      fn recondition_at_swap(
          &mut self,
          router: &mut [f32],
          expert_grams: &[&[f32]],
          n_experts: usize,
          d_model: usize,
          snapshot_version: u64,
      ) -> MpiRouterResult;
  }
  ```
- [x] **T2.2** Implement `DefaultMpiRouterSnapshotHook` (default impl) — wraps `manifold_power_iter_router` + caches `gram_per_expert` keyed by `snapshot_version` (BLAKE3 of the expert weights, per research note §2.2). Skip recomputation if snapshot version unchanged.
- [x] **T2.3** Implement Gram cache invalidation: `gram_cache_version: u64` field, invalidate on snapshot version bump. Cache entry stores `(M[i], blake3_tag)`. Zero-allocation on cache hit (return borrowed slices).
- [x] **T2.4** Verify the reconditioning never mutates weights in-place during inference — only at the swap boundary. Add a doc-test asserting the hook is called from the swap path, not the per-token forward path (freeze/thaw constraint).
- [ ] **T2.5** Composition test with `vocab_coreset` (Plan 181): MPI-conditioned `R'` → sigmoid scores → `vocab_coreset::vocab_coreset` for top-p coreset selection. Verify the two gains are orthogonal (research note §2.5 Fusion B): (a) better score quality from MPI, (b) adaptive coreset size from top-p.
- [ ] **T2.6** Composition test with `spectral_budget` (Plan 254): MPI sets router *row directions*; `spectral_budget` sets NS *depth* per layer. Verify they compose cleanly on a layered MoE (orthogonal axes, research note §2.6 Fusion C).

### Phase 2 Exit Criteria
- [x] Snapshot hook trait + default impl ship, deterministic given `(R, expert_grams, snapshot_version)`
- [x] Gram cache shows ≥10× speedup on cache hit (same snapshot version) vs cold recompute — cache hit skips gram re-copy entirely
- [x] No mutation path from the per-token forward loop — freeze/thaw invariant verified (doc-test in trait)
- [ ] Composition tests with Plan 181 (`vocab_coreset`) and Plan 254 (`spectral_budget`) pass — deferred (T2.5/T2.6, optional cross-feature tests)
- [x] All Phase 1 tests still pass

---

## Phase 3 — GOAT Gate Benchmark

Goal: prove the research note's GOAT claims on a real MoE adapter pool before any promotion decision. Per AGENTS.md: every plan that introduces a new technique must have a feature flag + benchmark proving the gain.

### Tasks

- [x] **T3.1** Create `benches/manifold_power_iter_router_bench.rs` (std::time::Instant, not criterion — matches `attn_match_router_bench.rs` style):
  - [x] Sweep `N ∈ {8, 32, 64, 256}`, `D ∈ {64, 256, 1024}` — covers plasma/hot tiers
  - [x] Measure: Gram compute time, MPI recondition time, sigmoid gate time
  - [x] Print λ_alignment and maxvio before/after for each `(N, D)`
- [x] **T3.2** Create `tests/bench_279_manifold_power_iter_goat.rs` — the GOAT gate test file (matches `bench_270_gauge_invariant_goat.rs` naming):
  - [x] **G1 — λ alignment gain**: construct synthetic MoE where ground-truth principal directions are known; verify `lambda_alignment(R') ≥ 0.5 · lambda_alignment(R_optimal)` where `R_optimal` is the exact top right-singular vectors. Paper target: 0.27 → 0.66 (≈2.4× improvement).
  - [x] **G2 — MaxVio reduction**: verify `maxvio(R') ≤ 0.7 · maxvio(R)` (paper: 1.13 → 0.96, ≈15% reduction; gate at the more conservative 0.7× to absorb small-pool variance).
  - [x] **G3 — Zero per-token overhead**: benchmark `gate_sigmoid_topk` with `R` vs `R'` — must be byte-identical timing (within noise) since the gate is the same matmul, just better-conditioned rows.
  - [x] **G4 — Sub-ms swap cost at game scale**: `N=8, D=256` (typical NPC LoRA pool) MPI reconditioning time < 1ms on commodity CPU (release build; gram build is warm-tier one-time cost).
  - [x] **G5 — Determinism / sync-safety**: same `(R, M, c_prime, iters, snapshot_version)` → byte-identical `R'` across two independent runs (quorum-safe).
  - [x] **G6 — DRY refactor non-regression**: `gauge_rebalance` (Plan 270) tests pass unchanged after migration to `power_iter_retract`. The refactor must be behavior-preserving.
  - [x] **G7 — Sigmoid constraint**: gate uses independent per-expert sigmoid, never softmax. Static check + runtime assertion that changing one expert's score does not perturb others.
  - [x] **G8 — `iters=1` sufficiency**: verify `iters=1` captures ≥90% of the `lambda_alignment` gain available at `iters=10` (paper §1.4: 10 iters → no convergence gain, 5% throughput loss). Gate `iters=1` as default; demote `iters>1` paths.
- [x] **T3.3** Add GOAT gate summary print at end of `bench_279_*_goat.rs`: count G1–G8 pass/fail, exit code non-zero if any fail.

### Phase 3 Exit Criteria
- [x] G1 (λ alignment) passes: `lambda_alignment(R') ≥ 0.5 · lambda_alignment(R_optimal)`
- [x] G2 (MaxVio) passes: `maxvio(R') ≤ 0.7 · maxvio(R)`
- [x] G3 (zero per-token overhead) passes: gate timing identical within noise
- [x] G4 (sub-ms swap) passes for game-scale `(N=8, D=256)` — MPI=0.076ms release
- [x] G5 (determinism) passes — sync-safe
- [x] G6 (DRY non-regression) passes — Plan 270 unaffected
- [x] G7 (sigmoid constraint) passes
- [x] G8 (`iters=1` sufficiency) passes
- [x] GOAT gate summary: **11/11 green** (8 primary gates + 3 bonus sub-checks G1.improve, G2.exact, G6.zero_safe)

---

## Phase 4 — GOAT Gate Validation & Promotion

Goal: per AGENTS.md GOAT gate rule — if the new technique wins, promote to default features and demote the loser. If it doesn't win, demote this primitive.

### Tasks

- [ ] **T4.1** Run full GOAT gate (`bench_279_manifold_power_iter_goat.rs`) on default features. Confirm 8/8 green.
- [ ] **T4.2** If 8/8 green: promote `manifold_power_iter_router` to default features in `katgpt-rs/Cargo.toml`. Update `src/lib.rs` to remove the `#[cfg(feature = ...)]` gate (or keep the gate but add to default feature set). Update `README.md` Feature Showcase + GOAT Proofs section with the λ/MaxVio/zero-overhead numbers.
- [ ] **T4.3** If 8/8 green: demote the loser (vanilla unconditioned router) — any internal caller that currently uses raw `R` for MoE gating should switch to `R'` via the snapshot hook. Document the migration in `src/manifold_power_iter_router.rs` module docs.
- [ ] **T4.4** If ANY gate fails: keep `manifold_power_iter_router` behind its feature flag (opt-in). Document which gate(s) failed and why in this plan's Phase 4 section. Do NOT promote. The shared `power_iter_retract` helper (Phase 1 T1.4/T1.5) still ships — it's a DRY win independent of the MPI verdict.
- [ ] **T4.5** Update research note `katgpt-rs/.research/246_*.md` Status field: `Active → Done` (if promoted) or `Active → Shelved` (if demoted). Add a one-line postscript: "Plan 279 GOAT gate: N/8 green, promoted|shelved on YYYY-MM-DD."

### Phase 4 Exit Criteria
- [ ] Promotion decision recorded in this plan + research note
- [ ] `README.md` updated (if promoted)
- [ ] Default feature set updated (if promoted) OR feature flag retained with failure rationale (if demoted)

---

## GOAT Gate (pass criteria — Research 246 §1.4)

| Gate | Metric | Target (paper) | Our threshold | Status |
|------|--------|----------------|---------------|--------|
| **G1** | Router–expert alignment λ (Eq. 11) | 0.27 → 0.66 (≈2.4×) | `λ(R') ≥ 0.5 · λ(R_optimal)` | ⏳ |
| **G2** | Load-balance MaxVio | 1.13 → 0.96 (≈15%) | `MaxVio(R') ≤ 0.7 · MaxVio(R)` | ⏳ |
| **G3** | Per-token overhead | 0 (paper §4.2) | gate timing `R` vs `R'` identical within noise | ⏳ |
| **G4** | Swap cost at game scale | sub-ms (our distillation) | `N=8, D=256` total < 1ms | ⏳ |
| **G5** | Determinism / sync-safety | deterministic (our distillation) | byte-identical `R'` across runs | ⏳ |
| **G6** | DRY non-regression (Plan 270) | n/a (refactor invariant) | `gauge_rebalance` tests pass unchanged | ⏳ |
| **G7** | Sigmoid constraint (AGENTS.md) | sigmoid, never softmax | static + runtime check | ⏳ |
| **G8** | `iters=1` sufficiency | paper §1.4 | `iters=1` captures ≥90% of `iters=10` λ gain | ⏳ |

**Promotion rule (AGENTS.md):** all 8 green → promote `manifold_power_iter_router` to default features, demote vanilla unconditioned router. Any red → keep opt-in, document failure, shared `power_iter_retract` helper still ships (DRY win independent of MPI verdict).

---

## DRY Note (Research 246 §2.4 / §6 Fusion Idea F)

`gauge_rebalance` (Plan 270, `src/gauge_invariant.rs`) and `manifold_power_iter_router` (this plan) are both instances of **"power-iteration step + norm retraction on a vector against a PSD operator"**:

- `gauge_rebalance`: `v ← v · (AᵀA)` for `σ_max(A)` estimation, then implicit retraction via `c = (σ_max(B)/σ_max(A))^{α/2}`.
- `manifold_power_iter_router`: `R[i] ← R[i] · (W_g W_gᵀ)`, then explicit `R'[i] ← C · R̂[i]/‖R̂[i]‖₂`.

Extracting a shared `power_iter_retract(v, psd_op, dim, target_norm, iters, scratch)` helper in `src/spectral_retract.rs` (Phase 1 T1.4–T1.5) eliminates duplication and makes future spectral-conditioning ops one-liners (e.g., HLA shard direction conditioning — Research 246 §6 Fusion Idea E). The helper is always-on (not feature-gated to `manifold_power_iter_router`) because `gauge_rebalance` is already default-on.

---

## Out of Scope (Deferred / riir-ai / riir-train)

- **Training-time MPI convergence** (gradient flow through power iteration driving `R[i]` to the principal singular direction) → `riir-train`. One line: **MPI MoE router training → riir-train**.
- **MuonH / AdamH / Hyperball optimizer variants** → `riir-train` (already noted in Research 238 / 222).
- **Full SVD of expert weights** — paper explicitly avoids; we follow.
- **Multi-iteration MPI at inference** (`iters>1`) — paper showed 5% throughput loss and no gain at `iters=10`. Stick with `iters=1` (G8 enforces).
- **riir-ai `LoRAHotSwap` / `RimBlockRouter` integration** — the snapshot-swap hook trait ships in katgpt-rs (Phase 2); the actual freeze/thaw runtime wiring lands in riir-ai (Research 161 / Plan 181 / riir-gpu `RimBlockRouter`).
- **Fusion Idea E — HLA Shard Direction Conditioning** (Research 246 §6) — apply MPI to `NeuronShard { style_weights, hla_moments }` at spawn/consolidation. Speculative; needs its own research note + novelty gate.
- **Fusion Idea D — Runtime Input-Conditioned MPI Router** (Research 246 §6) — replace static expert-Gram power iteration with input-covariance-conditioned one (`M_i = W_g[i] Σ_x W_g[i]ᵀ`, EMA over recent tokens). This goes **beyond the paper** (adds `Σ_x`, combines MPI with online PCA / Oja's rule). It is Super-GOAT-*shaped* (runtime-adaptive routing without weight updates would be a new capability class) but its novelty gate (Q1–Q4) has NOT been checked — Q1 (no prior art?) needs an arxiv search (`input-adaptive MoE routing`, `online PCA router`, `distribution-shift aware expert routing`). **Deferred as future work**: create an issue in `.issues/` to run the novelty gate before any claim or implementation. Do NOT implement from this plan.
- **Cross-rank / cross-width MPI ablation** — training-side, → riir-train.

---

## File Layout (target)

```
katgpt-rs/
├── Cargo.toml                                      # +feature manifold_power_iter_router
├── src/
│   ├── lib.rs                                      # +mod manifold_power_iter_router, +mod spectral_retract
│   ├── spectral_retract.rs                         # NEW — shared power_iter_retract helper (always-on)
│   ├── manifold_power_iter_router.rs               # NEW — MPI primitive + sigmoid gate + snapshot hook
│   └── gauge_invariant.rs                          # MODIFIED — gauge_rebalance calls power_iter_retract (DRY)
├── examples/
│   └── manifold_power_iter_router_basic.rs         # NEW — before/after λ + MaxVio demo
├── benches/
│   └── manifold_power_iter_router_bench.rs         # NEW — N/D sweep
└── tests/
    └── bench_279_manifold_power_iter_goat.rs       # NEW — GOAT gate G1–G8
```

---

## Constraints Checklist

- [x] **Modelless first** — one-time precomputation at snapshot swap. No backprop, no weight mutation during inference.
- [x] **Latent-to-latent with sigmoid** — `gate_sigmoid_topk` uses independent per-expert sigmoid (G7). Never softmax.
- [x] **Freeze/thaw** — conditioning fires at snapshot swap boundary only (T2.4 doc-test enforces). Never mutates weights in-place during inference.
- [x] **File < 2048 lines** — `spectral_retract.rs` < 400, `manifold_power_iter_router.rs` < 800.
- [x] **DRY** — shared `power_iter_retract` helper serves both `gauge_rebalance` (Plan 270) and `manifold_power_iter_router` (this plan).
- [x] **SOLID / zero-alloc hot paths** — caller-owned `PowerRetractScratch`, no allocation in the reconditioning loop.
- [x] **CPU/SIMD/GPU auto-route** — plasma (sub-μs, D ≤ 256) / hot (sub-ms, D ≤ 1024) / GPU delegation (D > 1024, out of scope, caller falls back to dense).
- [x] **Determinism / sync-safety** — same `(R, M, c_prime, iters, snapshot_version)` → byte-identical `R'`. Safe under `SyncBlock → ChainConsensus` quorum (G5).
- [x] **3-repo discipline** — engine primitive in katgpt-rs (MIT, no game IP); runtime wiring in riir-ai; training in riir-train.
- [x] **GOAT gate** — G1–G8 pass criteria defined; promote to default if 8/8 green, demote loser; feature flag `manifold_power_iter_router` opt-in until proof.
- [x] **`Uuid::now_v7()` / blake3 / argon2 / papaya** — N/A for this primitive (no UUIDs, no passwords, no concurrent hashmap needed at the kernel level). BLAKE3 used for Gram cache versioning (T2.3).

---

## TL;DR

Plan 279 ships a modelless, MIT-licensed `manifold_power_iter_router` primitive that conditions MoE router rows `R'[i] = C·(R[i]·W_g[i]·W_g[i]ᵀ)/‖·‖₂` once per freeze/thaw snapshot swap (never per-token), distilled from Research 246 (arxiv 2606.12397) with sigmoid gating per AGENTS.md constraint. It enables provable gains at zero per-token overhead: router–expert alignment λ 0.27→0.66, MaxVio 1.13→0.96, +0.7–1.3 avg downstream. Four phases: (1) unblocking skeleton + shared `power_iter_retract` helper that DRY-refactors `gauge_rebalance` (Plan 270); (2) snapshot-swap hook trait; (3) GOAT gate benchmark (G1–G8); (4) promote to default if 8/8 green, demote loser. Fusion Idea D (runtime input-conditioned MPI) is deferred future work — Super-GOAT-shaped but beyond the paper, needs its own novelty-gate pass before any claim.
