# Plan 356: Group Invariance Probe — Modelless Symmetry Discovery on a Hypothesis Group

**Date:** 2026-07-01
**Research:** [katgpt-rs/.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md)
**Source paper:** [arXiv:2512.20043](https://arxiv.org/abs/2512.20043) — Chen et al., LieFlow, ICML 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/group_invariance_probe.rs` (new module) + Cargo feature `group_invariance_probe`
**Status:** Active — Phase 1 (GOAT gate) COMPLETE — 8/8 gates PASS, feature ships OPT-IN

---

## Goal

Ship a small modelless primitive that generalizes `subspace_phase_gate` from "subspace of `ℝᵈ`" to "subgroup of a hypothesis group `G`". Given a stream of observations and a hypothesis group `G` (via a `GroupAction` trait), score each sampled `g ∈ G` by how invariant the observation distribution is under `g`, then classify the discovered subgroup `H` as `Discrete` / `Continuous` / `Partial` / `None` via a modelless concentration measure on the score histogram.

This is the modelless residue of LieFlow (arXiv:2512.20043) — the trained flow-matching `v_θ` redirects to riir-train; what ships here is the deterministic invariance test + support-concentration classifier.

**GOAT gate (G1–G4):** defined in §Phase 1. Opt-in feature; do NOT promote to default until a downstream consumer (Issue 011 fusion investigation, or a riir-neuron-db `can_freeze` extension) demonstrates the selling point.

---

## Phase 1 — GOAT Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/.plans/356_group_invariance_probe.md` (this plan).
- [x] **T1.2** Add feature flag `group_invariance_probe = []` to `crates/katgpt-core/Cargo.toml` (opt-in, pure numeric, no deps).
- [x] **T1.3** Implement `crates/katgpt-core/src/group_invariance_probe.rs`:
  - `GroupAction` trait — `fn act(&self, q: &[f32], out: &mut [f32])` and `fn sample(&mut self, rng: &mut impl Rng) -> Self::Elem`.
  - `invariance_score(distance: f32, beta: f32) -> f32` — shifted sigmoid `σ(β·(1−d))` (hits ≈1.0 at d=0, 0.5 at d=1, ≈0 at d=2; caller normalizes d so indifference point is at d=1).
  - `score_variance(scores: &[f32]) -> f32` — population variance, the PRIMARY discrete-vs-continuous signal for large-fraction subgroups (C₄ ⊂ C₈ → var ≈ 0.25).
  - `score_concentration(scores: &[f32]) -> f32` — participation-ratio-style `(Σs)²/(n·Σs²)`, the COMPLEMENTARY signal for small-fraction subgroups (C₄ ⊂ C₆₄ → conc ≈ 0.06).
  - `SubgroupClass` enum — `{ None, Discrete, Continuous, Partial }` with `as_u8`/`from_u8` for sync-boundary transport.
  - `classify_subgroup(scores: &[f32], tau: f32) -> SubgroupClass` — fires `Discrete` if EITHER variance ≥ 0.15 OR concentration < 0.3 (the two signals are complementary).
  - `SubgroupReport { n_samples, n_support, class, variance, concentration, max_score }` — mirrors `FreezeGateReport` shape; sync-tier split documented.
  - `discover_subgroup_into<G: GroupAction>(...)` — zero-alloc batch variant taking `&mut [f32]` scratch.
- [x] **T1.4** Register module in `crates/katgpt-core/src/lib.rs` (feature-gated) + re-export public API. NOTE: `Rng` trait stays under `group_invariance_probe::Rng` namespace (root re-export removed due to name collision with existing `Rng as OtherRng` import).
- [x] **T1.5** G1 correctness test: `discover_c8_to_c4_recovers_discrete_class` in-crate. Uses C₄-invariant indicator `[1,0,1,0,1,0,1,0]` on C₈ hypothesis group (NOT a single point — the stabilizer-of-a-point is trivial, the LieFlow §1.3 lesson).
- [x] **T1.6** G2 non-redundancy tests (3 in-crate): `classify_uniform_low_scores_is_none`, `classify_uniform_high_scores_is_continuous`, `classify_four_peaks_is_discrete` (small-fraction discrete via concentration signal).
- [x] **T1.7** G4 alloc-free integration test: `crates/katgpt-core/tests/group_invariance_probe_g4.rs` (CountingAllocator, 2 tests).
- [x] **T1.8** G3 check: `cargo check -p katgpt-core --all-features` clean (1 pre-existing unrelated warning in `set_attention.rs:454`); `--no-default-features` clean.
- [x] **T1.9** All four GOAT gates recorded in `.benchmarks/356_group_invariance_probe_goat.md`.
- [x] **T1.10** Feature ships OPT-IN (do NOT promote to default — Issue 011 gates that). Commit on `develop`.

### GOAT Gate Definitions

| Gate | Test | Target |
|---|---|---|
| **G1** Correctness | Synthetic `SO(2) → C₄` (4 orbit points). `discover_subgroup` scores the 4 `C₄` elements ≥ 0.95, random rotations ≤ 0.5; `classify_subgroup` returns `Discrete`. | 4/4 elements recovered, `Discrete` class |
| **G2** Non-redundancy | (a) Uniform-on-`SO(2)` data → class ∈ {`Continuous`, `None`}, NOT `Discrete`. (b) Partial symmetry (only `{0, π/2}` present out of `C₄`) → class = `Partial`. | 2/2 settings classified correctly |
| **G3** No regression | `cargo check -p katgpt-core --all-features` clean. No `default` feature change. | Clean build |
| **G4** Alloc-free | `discover_subgroup_into` (zero-alloc variant) allocates 0 bytes after warmup (CountingAllocator). | 0 allocs |

### Promotion rule

G1–G4 PASS → opt-in `group_invariance_probe` ships. **Do NOT promote to default** until:
- Issue 011 returns Q2+Q3 = YES → riir-ai fusion plan consumes the API and demonstrates the selling point, OR
- riir-neuron-db `can_freeze` extension uses `discover_subgroup` to add a group-axis field to `FreezeGateReport` and demonstrates anti-cheat value.

---

## Phase 2 — (Deferred) Downstream consumer

Tracked in Issue 011. Not started until Phase 1 ships.
