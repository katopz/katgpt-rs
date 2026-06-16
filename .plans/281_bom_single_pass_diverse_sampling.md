# Plan 281: BoMSampler — Best-of-Many Single-Pass K-Hypothesis Belief Sampling

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md](../.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md)
**Source paper:** [arXiv:2604.04913](https://arxiv.org/abs/2604.04913) — Kerssies et al., "A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens", Apr 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/micro_belief/` (extend `MicroRecurrentBeliefState` with an opt-in stochastic variant) + Cargo feature `bom_sampling`
**Status:** Active — Phase 0 (planning). **Verdict: Gain** (not GOAT, not Super-GOAT — see Research 248 §3). Opt-in until G1–G3 pass; demote if G2 fails.

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
- [ ] **T0.3** Audit `MicroRecurrentBeliefState` trait (`micro_belief/types.rs`) — confirm `step()` signature can be extended with an optional `queries` parameter without breaking existing callers. The deterministic `step()` MUST remain the default path; BoM is opt-in.
- [ ] **T0.4** Audit SIMD matvec infra (`crate::simd`) — confirm a K-row batched matvec (K noise queries × W_s/W_x) fits the existing SIMD helpers, or identify the minimal addition needed.
- [ ] **T0.5** Audit `KernelHotSwap` (Plan 276 T0.4) — confirm the noise query distribution `N(0, σ²I)` can be versioned as part of `MicroRecurrentKernelSnapshot` (so σ is per-NPC-class, freeze/thaw-able). If not, add a `NoiseQueryConfig` to the snapshot.

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

- [ ] **T1.1** Create `micro_belief/bom.rs` with `BoMSampler` trait + `NoiseQueryConfig` (behind `bom_sampling` feature).
- [ ] **T1.2** Implement `BoMSampler` for `AttractorKernel`. Zero-alloc: caller passes `[K * D]` scratch, writes in-place.
- [ ] **T1.3** Implement `BoMSampler` for `LeakyIntegrator` (the `evolve_hla` family).
- [ ] **T1.4** `select_best()` with a generic scorer closure. Default scorer: max dot-product against a caller-provided direction vector (reuses `simd_dot_f32`).
- [ ] **T1.5** Unit tests: (a) determinism for fixed queries (same `queries` → same `out` bit-identical); (b) K hypotheses are distinct when queries are distinct; (c) zero-sigma reproduces deterministic `step()` output (BoM degenerates to deterministic when σ=0).
- [ ] **T1.6** `NoiseQueryConfig` BLAKE3 commitment + integration into `MicroRecurrentKernelSnapshot` (version the σ and K, freeze/thaw-able).

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

- [ ] **T2.1** Add `sample_k_states` bench to `micro_belief_bench.rs`: K ∈ {1, 4, 8, 16}, measure ns/call, confirm G3 threshold.
- [ ] **T2.2** Synthetic coherence test: run `sample_k_states` for 1000 ticks with random queries, confirm all K trajectories stay bounded (`‖s_t‖` ≤ clamp bound). Catches Family A divergence (Research 242 R1).
- [ ] **T2.3** G2 arena harness: wire `BoMSampler` into a minimal planning loop (select_best by max-threat, then plan action against the selected belief). Run on bomber self-play. **If G2 fails by > −5pp → demote to experimental, document why, stop.**
- [ ] **T2.4** If G2 passes: promote `bom_sampling` from opt-in to default-on for the trait method (NOT for the planner — the planner stays in riir-ai). Demote the loser (deterministic-only planning) if BoM strictly wins.

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
