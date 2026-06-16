# Plan 281: BoMSampler — Best-of-Many Single-Pass K-Hypothesis Belief Sampling

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md](../.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md)
**Source paper:** [arXiv:2604.04913](https://arxiv.org/abs/2604.04913) — Kerssies et al., "A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens", Apr 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/micro_belief/` (extend `MicroRecurrentBeliefState` with an opt-in stochastic variant) + Cargo feature `bom_sampling`
**Status:** Phase 0–1 + G1/G3 landed (2026-06-16). `bom_sampling` opt-in feature ships `BoMSampler` trait + impls for `AttractorKernel` + `LeakyIntegrator`. **G1.1/G1.2/G1.3 PASS** (17 tests). **G3 borderline-FAIL** (K=8 attractor at 2.54×, target ≤2× — Issue 025, shared root cause with Issue 024; K=4 passes at 1.6×). **G2 (arena) deferred to riir-ai** (T2.3). **Verdict: Gain** (not GOAT, not Super-GOAT — see Research 248 §3). Stays opt-in until G2 passes.

---

## Goal

Add a `BoMSampler` extension to `MicroRecurrentBeliefState` (Plan 276) that produces **K diverse plausible next-belief-states per tick in a single batched kernel evaluation**, by injecting K Gaussian noise queries at the kernel input site. This is the only novel inference primitive distilled from DeltaTok/DeltaWorld (Research 248) — the delta-token compression itself is already shipped via `evolve_hla` / `MicroRecurrentBeliefState` / NextLat residual.

The GOAT-gate question (G2): **does planning against K diverse belief hypotheses improve arena win rate / HL score over planning against 1 deterministic belief + K diverse DDTree actions?** If no → demote to experimental, keep the trait method but never promote to default.

**Out of scope (stays in riir-ai/.plans if G2 passes):** NPC tick dispatch changes, minimax-over-K-beliefs planner, ANE batch dispatch for K-query evaluation. This plan ships *only* the generic `BoMSampler` trait + the `MicroRecurrentBeliefState` impl + the G1–G3 benchmarks.

---

## Phase 0 — Pre-flight (this plan)

### Tasks

- [x] **T0.1** Research note `katgpt-rs/.research/248_*.md` created.
- [x] **T0.2** This plan created.
- [x] **T0.3** Audit `MicroRecurrentBeliefState` trait (`micro_belief/types.rs`) — **DONE.** `step(&self, state: &mut [f32], input: &[f32])` is the deterministic path. Plan 281 adds a *new* `BoMSampler` trait with a *new* method `sample_k_states` rather than extending `step()` — zero existing callers affected, `step()` stays deterministic-by-default. ✅
- [x] **T0.4** Audit SIMD matvec infra (`crate::simd`) — **DONE.** `simd_dot_f32(a, b, len)` + `fast_sigmoid(x)` suffice. BoM's "K-row batched matvec" is really **1 matvec** (base activation `act[i] = W_s[i]·s + W_x[i]·x + b[i]`, D dot products reusing `simd_dot_f32`) **+ K × (D elementwise adds + D sigmoids)**. The elementwise K-loop auto-vectorizes. No new SIMD helper needed. ✅
- [x] **T0.5** Audit `MicroRecurrentKernelSnapshot` (`micro_belief/snapshot.rs`) — **DONE.** Snapshot commits BLAKE3 over `(family_byte, dim_le, weights_blob)`. Adding a field would bump `SNAPSHOT_VERSION` (currently 1) and break Plan 276's G1.5 atomicity tests. **Decision:** give `NoiseQueryConfig` its OWN `commit()` method (separate BLAKE3 over `sigma_le || k_le || seed_strategy_byte`), treat it as a *companion artifact* to the kernel snapshot (caller embeds both commitments in the hot-swap audit event). `MicroRecurrentKernelSnapshot` is unchanged. ✅

---

## Phase 1 — Core Skeleton (BoMSampler trait + impl)

**Unblocks:** G1.1, G1.2, G1.3. This is the correctness phase.

### Architecture

```rust
// micro_belief/bom.rs (new, behind `bom_sampling` feature)

/// K-hypothesis belief sampling (Research 248, Plan 281).
///
/// Injects K Gaussian noise queries at the kernel input site and evaluates
/// the kernel K times in a single batched matvec. Returns K diverse
/// next-belief-states. The deterministic `step()` path is unchanged.
pub trait BoMSampler: MicroRecurrentBeliefState {
    /// Sample K diverse next-states from (s_prev, x) in one batched call.
    ///
    /// `queries` is a `[K][D]` slice where D = kernel input dim. Each row is
    /// a noise vector `q_k ~ N(0, σ²I)`; σ comes from `NoiseQueryConfig`.
    /// Writes K next-states into `out` (caller-allocated `[K][D]` scratch).
    fn sample_k_states(
        &self,
        s_prev: &[f32],
        x: &[f32],
        queries: &[f32],   // [K * D_q], row-major
        out: &mut [f32],   // [K * D_state], row-major
        cfg: &NoiseQueryConfig,
    );

    /// Select the best hypothesis by a caller-provided scorer (e.g. minimax
    /// over threat, or max dot-product against a target direction). Returns
    /// the index of the best hypothesis in `out`.
    fn select_best(
        &self,
        hypotheses: &[f32], // [K * D_state]
        scorer: impl Fn(&[f32]) -> f32,
        k: usize,
    ) -> usize;
}

/// Noise query distribution config. Versioned via `MicroRecurrentKernelSnapshot`.
#[derive(Clone, Copy, Debug, blake3::Hashable)]
pub struct NoiseQueryConfig {
    pub sigma: f32,       // paper default 0.02; needs calibration for [-1,1] HLA space (R3)
    pub k: usize,         // paper trains K=256, evals K=20; we default K=8 (plasma-tier budget)
    pub seed_strategy: SeedStrategy,  // Uuid::now_v7()-derived per-NPC, or shared per-class
}
```

**Implementation for `AttractorKernel` (Family A):** the K noise queries are added to the `W_x · x` term before the sigmoid: `state_k[i] = clamp(2·σ(W_s·s + W_x·x + q_k + b) − 1, ±clamp)`. The K-row matvec over `W_s·s + W_x·x` is computed once; the K noise additions + K sigmoids are SIMD-batched.

**Implementation for `LeakyIntegrator` (Family C / `evolve_hla`):** the K noise queries perturb the delta: `delta_k = clamp(lr·(normalized − half_total)·scale + q_k, max_delta)`. K additions + K clamps, SIMD-batched.

### Tasks

- [x] **T1.1** Create `micro_belief/bom.rs` with `BoMSampler` trait + `NoiseQueryConfig` + `SeedStrategy` (behind `bom_sampling` feature).
- [x] **T1.2** Implement `BoMSampler` for `AttractorKernel`. Zero-alloc: base activation computed once (chunked-4 loop mirroring `step()` for bit-identical σ=0 degeneracy), K elementwise perturbations write directly into `out`.
- [x] **T1.3** Implement `BoMSampler` for `LeakyIntegrator` (the `evolve_hla` family). Shared normalization computed once, K elementwise delta perturbations; zero-total guard copies `s_prev` into every row.
- [x] **T1.4** `select_best()` with a generic scorer closure, factored through `select_best_generic` helper (DRY). Default scorer factory `dot_product_scorer` reuses `simd_dot_f32`.
- [x] **T1.5** Unit tests (17 total): (a) `bom_determinism_fixed_queries` G1.1 PASS; (b) `bom_distinct_hypotheses` G1.2 PASS (cosine sim < 0.99 at σ=0.1); (c) `bom_sigma_zero_matches_step_attractor` + `_leaky` + `_leaky_zero_total` G1.3 PASS. Plus boundedness, coherence 1000-tick, select_best (max/ties/leaky), commit roundtrip.
- [x] **T1.6** `NoiseQueryConfig::commit()` BLAKE3 over `(sigma_le || k_le || seed_strategy_byte)` as a *companion artifact* to `MicroRecurrentKernelSnapshot` (see T0.5 decision — kernel snapshot unchanged, no SNAPSHOT_VERSION bump).

---

## Phase 2 — GOAT Gate (G1 mechanics + G2 quality + G3 latency)

**The actual GOAT decision.** If G2 fails, demote to experimental; keep the trait, never promote to default.

### GOAT Proofs Required

| # | Metric | Threshold | Measurement |
|---|--------|-----------|-------------|
| **G1.1** | Determinism | bit-identical `out` for fixed `queries` + fixed kernel | Unit test (T1.5a) |
| **G1.2** | Distinctness | K hypotheses pairwise distinct (cosine sim < 0.99) when queries are distinct | Unit test (T1.5b) |
| **G1.3** | σ=0 degeneracy | BoM with σ=0 reproduces deterministic `step()` | Unit test (T1.5c) |
| **G2** | **Planning quality (the GOAT gate)** | K-hypothesis belief planning (minimax over K beliefs) ≥ deterministic-belief planning + DDTree action diversity, on a bomber/go arena benchmark, by ≥ +5pp win rate or HL score | Arena benchmark (deferred to riir-ai if needed — but the primitive must be usable from a test harness) |
| **G3** | Latency | `sample_k_states(K=8)` ≤ 2× the cost of a single `step()` call (batched matvec should be near-1×, the K noise additions + sigmoids add ≤ 2×). Measured on CPU SIMD plasma-tier path. | `micro_belief_bench` extension |

### Tasks

- [x] **T2.1** Added `sample_k_states` bench to `micro_belief_bench.rs` (K ∈ {1, 4, 8, 16}). **G3 result:** K=1 0.89× PASS, K=4 1.60× PASS, **K=8 2.54× FAIL** (target ≤2×), K=16 4.52× FAIL. Root cause: K×D scalar `fast_sigmoid`/`exp()` calls — **Issue 025** (shared with Issue 024). K=4 is the practical plasma-tier ceiling until SIMD-sigmoid lands.
- [x] **T2.2** Synthetic coherence tests: `bom_coherence_1000_ticks_bounded_attractor` + `_leaky` — 1000 ticks with random queries, all K trajectories stay bounded. PASS for both families.
- [ ] **T2.3** G2 arena harness: **DEFERRED to riir-ai** per plan §Phase 3. The primitive is usable from a test harness at K=8 regardless of the 2.54× G3 ratio (G2 is a quality question, not a latency question). If G2 fails by > −5pp → demote to experimental, document why, stop.
- [ ] **T2.4** If G2 passes: promote `bom_sampling` from opt-in to default-on for the trait method (NOT for the planner — the planner stays in riir-ai). Demote the loser (deterministic-only planning) if BoM strictly wins. **BLOCKED on T2.3 (G2 result) + Issue 025 (G3 fix).**

---

## Phase 3 — (Deferred to riir-ai if G2 passes)

Only if G2 passes. These tasks belong in `riir-ai/.plans/`, not here:

- [ ] NPC tick dispatch: batch K-query evaluation across N NPCs (one ANE batch = N × K noise queries).
- [ ] Minimax-over-K-beliefs planner: plan against the most threatening hypothesis.
- [ ] Per-NPC-class σ calibration (R3): bandit-tune σ per class, store in `NoiseQueryConfig`.
- [ ] Sync boundary rule (R4): only the selected belief (or mean of K) projects to synced scalars. Never sync the K-vector distribution.

---

## Notes

- **The delta-token compression (DeltaTok's encoder) is NOT part of this plan.** It is already shipped via `evolve_hla` / `MicroRecurrentBeliefState` (Research 248 §2.2). This plan is ONLY the BoM sampling primitive.
- **The ECHO training fix (delta-token obs head) is NOT part of this plan.** That is riir-train territory (`riir-train/.plans/272` T1 redesign, benchmark 288). This paper is the literature backup for that fix — cross-ref only.
- **σ calibration (R3) is critical.** The paper's `σ=0.02` is tuned for DINOv3 features. Our HLA space is `[-1, 1]` (8-dim). σ=0.02 may produce near-identical hypotheses (cosine sim ≈ 1.0). The G1.2 distinctness test will catch this; if it fails, σ needs to be ~0.1–0.5 for our space.
- **K budget.** Paper trains K=256, evals K=20. For plasma-tier (µs budget, 1000 NPCs × 20Hz), K=8 is the practical ceiling per NPC. ANE batching could raise this, but that's Phase 3 (riir-ai).

---

## TL;DR

Plan 281 adds `BoMSampler` — a `MicroRecurrentBeliefState` extension that injects K Gaussian noise queries and evaluates K diverse next-belief-states in one batched matvec (the only novel inference primitive from DeltaTok/DeltaWorld, Research 248). The delta-token compression itself is already shipped. GOAT gate G2: does K-hypothesis belief planning beat deterministic-belief + DDTree-action-diversity planning on an arena by ≥ +5pp? If no → demote to experimental. Opt-in behind `bom_sampling` feature until G1–G3 pass. The ECHO training fix (delta-token obs head) is a riir-train cross-ref, not this plan.
