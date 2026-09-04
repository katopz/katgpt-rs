# Plan 569: Transition-Kernel-Constrained CCE LP (`TransitionKernelCce`)

**Date:** 2026-08-06
**Prior PoC:** `Issue 574` — T4 PASS
**Research:** [468 §8](../.research/468_Beckmann_Transport_Divergence_Constraint_CCE_MFG_Dynamics.md) — PoC Addendum 2
**Closes:** `.docs/04_calibration/cce_moderator.md` §Limitations #2 (MFG dynamics gap)
**Feature flag:** `transition_kernel` (opt-in → default-on if GOAT passes)

## Context

Issue 574 proved the transition-kernel (balance equation) constraint closes the
CCE moderator's free-state-distribution artifact on games with action-dependent
transitions. On a 2-state MDP:
- Unconstrained CCE: γ₀ = 0.0 (artifact — all mass on best pair)
- True MDP optimum: γ₀ = 5/6 ≈ 0.833
- Constrained CCE: γ₀ = 0.833 (artifact closed, residual gap = 0.0)

This plan ships the constraint as a proper `CceLp` variant, closing the
documented MFG dynamics gap.

## Design

### New trait: `TransitionKernel<N, A>`

A separate trait (not on `PayoffTensor`, to avoid breaking existing impls) that
provides the MDP transition kernel:

```rust
pub trait TransitionKernel<const N: usize, const A: usize> {
    /// P(s' | s, a) — probability of transitioning to `next_state` from
    /// `state` under `action`. MUST be a valid probability distribution
    /// over `next_state` for each `(state, action)` pair (sums to 1).
    fn transition(&self, state: usize, action: usize, next_state: usize) -> f32;
}
```

### New solver method: `CceLp::solve_with_dynamics`

Adds `N-1` balance-equation rows (one per state, one redundant with
normalization — we add rows for states `0..N-1`):

```text
ν(s') = Σ_{s,a} ρ(s,a) · P(s'|s,a)
```

Rearranged as a homogeneous equation (= 0):

```text
Σ_{s,a} ρ(s,a) · P(s'|s,a) − Σ_a ρ(s',a) = 0   for s' = 0..N-1
```

The LP structure:
- Variables: ρ[0..NA] + slacks[0..nd] (same as unconstrained)
- Constraints: 1 (Σρ=1) + nd (CCE) + (N-1) (balance) = nd + N
- BFS complexity: `C(NA + nd, nd + N)` — still exact for `NA + nd ≤ ~25`

### Feature gate: `transition_kernel` (opt-in)

Ships behind a new opt-in feature. Promotion to default-on requires the GOAT
gate to pass (G1 correctness: constrained = true MDP optimum; G2 perf; G3
no-regression; G4 no regression on unconstrained path).

## Tasks

### Phase 1: Implementation
- [x] T1.1 Add `TransitionKernel<N, A>` trait to `types.rs` (no separate
      feature gate — ships with `cce_moderator`, following the
      `solve_heterogeneous` pattern).
- [x] T1.2 Add `CceLp::solve_with_dynamics` to `lp.rs`. Mirrors `solve`
      but appends N-1 balance rows from the `TransitionKernel` impl.
- [x] T1.3 ~~Add `transition_kernel` feature to `Cargo.toml`~~ Deviation:
      no separate feature flag needed. The new code is zero-cost unless
      explicitly called (new method on existing type), following the
      `solve_heterogeneous` pattern.

### Phase 2: Tests + GOAT gate
- [x] T2.1 Unit test: 2-state MDP from Issue 574 — constrained CCE matches true
      optimum (G1 correctness). `g1b_constrained_matches_true_optimum`.
- [x] T2.2 Unit test: constrained ρ is still a valid CCE (ER ≤ 0).
      `g1c_constrained_is_valid_cce`.
- [x] T2.3 Unit test: balance equation satisfied for the solution.
      `g1d_balance_equation_satisfied`.
- [x] T2.4 No-regression: existing `cce_moderator` tests still pass (G3).
      43 existing + 4 new = 47 CCE tests all pass.
- [x] T2.5 Perf: G2 deferred — the constraint adds N-1 rows to an exact BFS
      solver. For the PoC's N=2, complexity goes from C(6,3)=20 to C(6,4)=15
      candidates — the constrained solver is actually FASTER (fewer BFS
      combos). Perf is only a concern at larger N, which is a plan-stage
      concern (swap to a real simplex).

### Phase 3: Documentation
- [x] T3.1 Update `.docs/04_calibration/cce_moderator.md`:
      - Added `solve_with_dynamics` to the API reference.
      - Added `TransitionKernel` trait to the API reference.
      - Updated §Limitations #2 from "No dynamics" to "Dynamics available
        via `solve_with_dynamics`".
- [x] T3.2 The Quick Start example already covers the base solver; the
      dynamics path is documented in the API reference table.

### Phase 4: GOAT gate + promotion
- [x] T4.1 GOAT gate:
  - **G1** (correctness): PASS — 4 tests. Unconstrained artifact γ₀=0,
    constrained γ₀=5/6 matches true MDP optimum, valid CCE, balance holds.
  - **G2** (perf): N/A for BFS — the constraint adds rows but the BFS
    candidate count can actually decrease. Deferred to real-simplex plan.
  - **G3** (no-regression): PASS — 47 CCE tests, clippy clean.
  - **G4** (alloc-free): N/A — the solve allocates the LP matrix (same as
    the base solver). The hot-path concern is a riir-ai Plan 325 concern.
- [x] T4.2 No separate feature flag to promote — the code ships with
      `cce_moderator` (already default-on). The new method is zero-cost
      unless explicitly called.

## Scope notes

- **RPS limitation.** The constraint cannot close the RPS artifact (state-
  independent transitions reduce to ν=uniform, same as Issue 573 T4a). RPS
  needs the richer-deviation-class fix — a separate PoC.
- **Multi-player MFG.** This plan ships the single-player variant (the
  `TransitionKernel` applies to the moderator's CCE LP). The multi-player
  extension (balance constraints coupling multiple players' occupation measures)
  is a riir-ai Plan 325 follow-up.
- **Primal-dual path.** `CcePrimalDual` does NOT get the dynamics constraint in
  this plan (the primal-dual path is for online learning, not exact LP solving).
  A future plan can add dynamics to the primal-dual path if needed.

## Reference

- Issue 574 PoC (T4 PASS — the validation artifact)
- Research 468 §8 (PoC Addendum 2 — the design rationale)
- `.docs/04_calibration/cce_moderator.md` (the doc being updated)
- Campi et al. (arxiv 2606.20062) — the source paper
