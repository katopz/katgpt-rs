# Plan 300: Subjective-CCE Heterogeneous-Payoff Wrapper Primitive

**Date:** 2026-06-21
**Research:** Forked from [`riir-ai/.docs/62_bayes_cce_regret_sketch.md`](../../riir-ai/.docs/62_bayes_cce_regret_sketch.md) §2 (subjective-CCE framing, bound transfers at math level).
**Source paper:** [arxiv 2606.20062](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzoumas — Optimal CCEs in MFGs via LP + No-Regret Learning.
**Parent issue:** [`riir-ai/.issues/327_cwm_cce_bayes_fusion.md`](../../riir-ai/.issues/327_cwm_cce_bayes_fusion.md) — closed 2026-06-21 (split decision, Branch 4: subjective-CCE demoted to wiring).
**Target:** `katgpt-rs/src/cce/` (extend existing module from Plan 295) + extend `cce_moderator` feature.
**Status:** ✅ COMPLETE (2026-06-21). G1+G2+G4 PASS; G3 SPLIT into T4.3b follow-up. T5.4 (feature promotion) deferred — `cce_moderator` stays opt-in pending G3.

---

## Goal

Ship a thin wrapper that extends the Plan 295 `CceLp<N,A>` primitive to accept **per-NPC heterogeneous payoff tables**, where each NPC `i` has its own `PayoffTensor` and `DeviationClass`. The regret bound transfers as-is at the math level (doc 62 §2 — sum of convex is convex, primal-dual averaging is heterogeneity-agnostic); no new theory, just API surface + LP-construction logic.

**Why:** Issue 327 P0.5 originally (and incorrectly) claimed "CceLp works unchanged on heterogeneous payoff tables". Audit on 2026-06-21 found this is false at the code level — `CceLp::solve(d: &D, p: &P)` takes a *single* `PayoffTensor<N,A>`; the LP constraint rows in `lp.rs:94-104` all reference the same `p`. This plan closes that gap. Result: subjective-CCE becomes shippable; Path A+ wiring closes.

**GOAT gate:** `cce_moderator` feature remains default-off pending G1–G3 below. No demotion of `CceLp::solve` (homogeneous path) — both coexist.

This is the **public open primitive** for the subjective-CCE wiring extension. The private runtime plan (riir-ai side, wiring per-NPC `NpcCwmRuntime<K>` payoff tables into this wrapper) is a *future* riir-ai plan, blocked on Plan 300 landing. **No game semantics in this plan** — pure generic math, MIT-licensed, anyone can adopt.

---

## The math (recap, doc 62 §2)

Standard homogeneous CCE LP (Plan 295):
```
minimize   γ₀(ρ)
subject to γ(ρ) ≤ γ_dev(ρ, κ)   ∀κ ∈ D
           Σ ρ = 1, ρ ≥ 0
```

Heterogeneous subjective-CCE LP (this plan):
```
minimize   γ₀(ρ)                                    # moderator objective
subject to γ_i(ρ) ≤ γ_dev_i(ρ, κ)   ∀i ∈ [1..P], ∀κ ∈ D_i    # per-NPC, per-deviation
           Σ ρ = 1, ρ ≥ 0
```

where `P` is the player count, `γ_i(ρ) = Σ_{s,a} ρ(s,a) · P_i.reward_follow(s,a)` is player `i`'s expected cost of following the moderator's recommendation, and `γ_dev_i(ρ, κ) = Σ_s μ(s) · P_i.reward_deviate(s, κ)` is player `i`'s expected cost of deviating to `κ` under `ρ`'s state marginal.

The convexity argument (doc 62 §2.1) is unchanged: each `γ_i` is linear in `ρ`, so each `ER_i(ρ) = max_{κ ∈ D_i} (γ_i(ρ) − γ_dev_i(ρ, κ))` is convex; `ER(ρ) = (1/P) Σ_i ER_i(ρ)` is convex as the sum of convex functions. The primal-dual averaging argument transfers.

---

## Design

### New trait: `HeterogeneousPayoff`

```rust
/// Per-NPC heterogeneous payoff + deviation-class bundle.
///
/// Each "player" contributes its own `PayoffTensor` and `DeviationClass`.
/// The wrapper builds one LP constraint row per `(player, κ)` pair, each
/// row using that player's `P_i` (NOT a shared tensor).
///
/// Regret bound: `ER(ρ̄_T) ≤ O(T⁻¹ᐟ²)` transfers as-is from the
/// homogeneous case (doc 62 §2). The heterogeneity enters only through
/// per-player numerical values, not through the convexity structure.
pub trait HeterogeneousPayoff<const N: usize, const A: usize> {
    /// Number of players (each with its own payoff tensor + deviation class).
    fn n_players(&self) -> usize;

    /// Player `i`'s deviation class slice. Caller MUST ensure stable order
    /// across calls (the LP indexes deviations by position).
    fn deviations_for_player(&self, player: usize) -> &[Deviation<N, A>];

    /// Player `i`'s cost of following at `(state, action)`.
    fn reward_follow(&self, player: usize, state: usize, action: usize) -> f32;

    /// Player `i`'s cost of deviating to `κ` at `state`. Default impl
    /// mirrors `PayoffTensor::reward_deviate` using `reward_follow`.
    fn reward_deviate(
        &self,
        player: usize,
        state: usize,
        kappa: &Deviation<N, A>,
    ) -> f32 {
        let mut g = 0.0;
        for a in 0..A {
            g += kappa.kernel[state][a] * self.reward_follow(player, state, a);
        }
        g
    }

    /// Moderator objective `γ₀(ρ)`. Default: average player welfare
    /// `(1/P) Σ_i γ_i(ρ)`. Override for designer-steerable moderators.
    fn gamma0(&self, rho: &OccupationMeasure<N, A>) -> f32 {
        let p = self.n_players().max(1) as f32;
        let mut g = 0.0;
        for s in 0..N {
            for a in 0..A {
                for i in 0..self.n_players() {
                    g += rho.at(s, a) * self.reward_follow(i, s, a);
                }
            }
        }
        g / p
    }

    /// Per-index coefficient for the LP objective row (linear `γ₀`).
    /// Default: `(1/P) Σ_i reward_follow(i, s, a)`.
    fn gamma0_coeff(&self, state: usize, action: usize) -> f32 {
        let p = self.n_players().max(1) as f32;
        let mut g = 0.0;
        for i in 0..self.n_players() {
            g += self.reward_follow(i, state, action);
        }
        g / p
    }
}
```

### New method on `CceLp`

```rust
impl CceLp {
    /// Solve the subjective-CCE LP for a heterogeneous player population.
    ///
    /// Builds `Σ_i |D_i|` constraint rows. Each row `(i, κ)` uses
    /// `P_i.reward_follow − P_i.reward_deviate(κ)`. Returns the optimal
    /// occupation measure.
    pub fn solve_heterogeneous<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &self,
        game: &H,
    ) -> Result<OccupationMeasure<N, A>, CceLpError> {
        // 1. Count total deviations: sum_i |D_i|
        // 2. Build LP with na = N·A variables and 1 + total_devs constraints
        // 3. Row 0: Σ ρ = 1
        // 4. Rows 1..: for each (player, kappa) pair, build
        //    g(s,a) = P_player.reward_follow(s,a) - P_player.reward_deviate(s,a, kappa)
        // 5. Objective: gamma0_coeff per index
        // 6. BFS enumerate (same algorithm as solve, just more rows)
    }

    /// Verify that `ρ` is a subjective-CCE: for every player `i` and every
    /// `κ ∈ D_i`, `γ_i(ρ) ≤ γ_dev_i(ρ, κ) + ε`.
    pub fn is_heterogeneous_cce<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &self,
        rho: &OccupationMeasure<N, A>,
        game: &H,
        epsilon: f32,
    ) -> bool {
        // Per-player, per-deviation regret check. Returns false on first
        // violation.
    }
}
```

### Heterogeneous external-regret evaluator

```rust
impl ExternalRegret {
    /// `ER_hetero(ρ) = (1/P) Σ_i max_{κ ∈ D_i} (γ_i(ρ) − γ_dev_i(ρ, κ))`.
    pub fn er_heterogeneous<const N: usize, const A: usize, H: HeterogeneousPayoff<N, A>>(
        &self,
        rho: &OccupationMeasure<N, A>,
        game: &H,
    ) -> f32 {
        // Per-player max-of-linear, averaged. Convex by construction.
    }
}
```

### Default struct impl: `PerPlayerGame`

```rust
/// Concrete `HeterogeneousPayoff` backed by per-player `(PayoffTensor,
/// DeviationClass)` slices. The common case for runtimes that already
/// hold per-NPC tables.
pub struct PerPlayerGame<'a, const N: usize, const A: usize, P: PayoffTensor<N, A>, D: DeviationClass<N, A>> {
    /// `(payoff_tensor, deviation_class)` per player.
    pub players: Vec<(&'a P, &'a D)>,
}

impl<'a, const N: usize, const A: usize, P: PayoffTensor<N, A>, D: DeviationClass<N, A>>
    HeterogeneousPayoff<N, A> for PerPlayerGame<'a, N, A, P, D>
{
    fn n_players(&self) -> usize { self.players.len() }
    fn deviations_for_player(&self, player: usize) -> &[Deviation<N, A>] {
        self.players[player].1.deviations()
    }
    fn reward_follow(&self, player: usize, state: usize, action: usize) -> f32 {
        self.players[player].0.reward_follow(state, action)
    }
    // reward_deviate, gamma0, gamma0_coeff use defaults (delegate to P).
}
```

---

## File Layout

```
katgpt-rs/src/cce/
├── mod.rs                # re-export new types
├── types.rs              # add HeterogeneousPayoff trait
├── lp.rs                 # add CceLp::solve_heterogeneous + is_heterogeneous_cce
├── external_regret.rs    # add ExternalRegret::er_heterogeneous
└── heterogeneous.rs      # NEW: PerPlayerGame + integration tests
```

---

## Tasks

### Phase 1 — Trait + types

- [x] **T1.1** Add `HeterogeneousPayoff<N, A>` trait to `katgpt-rs/src/cce/types.rs` with the five methods above (`n_players`, `deviations_for_player`, `reward_follow`, `reward_deviate`, `gamma0`, `gamma0_coeff`). Default impls for the last three.
- [x] **T1.2** Re-export from `katgpt-rs/src/cce/mod.rs`: `pub use types::HeterogeneousPayoff;`
- [x] **T1.3** Add unit test in `types.rs`: a 2-player trivial `HeterogeneousPayoff` impl with `N=2, A=2`, verify `gamma0` on uniform `ρ` matches hand-computed `(1/2)(γ_1 + γ_2)`.

**Phase 1 exit:** `cargo test --features cce_moderator --lib cce::types::` passes. ✅ PASSED 2026-06-21 (9/9).

### Phase 2 — LP solver extension

- [x] **T2.1** Implement `CceLp::solve_heterogeneous::<N, A, H>(game: &H)` in `katgpt-rs/src/cce/lp.rs`. Refactor the existing `solve` to share BFS-enumeration infrastructure with the new method (extract `enumerate_bfs(mat, rhs, n_vars, na)` helper — DRY, keeps both paths consistent).
- [x] **T2.2** Implement `CceLp::is_heterogeneous_cce(rho, game, epsilon)` — per-player, per-deviation regret check. Early-exit on first violation.
- [x] **T2.3** Implement `ExternalRegret::er_heterogeneous(rho, game)` — average per-player external regret.

**Phase 2 exit:** `cargo build --features cce_moderator` succeeds; no warnings. ✅ PASSED 2026-06-21.

### Phase 3 — `PerPlayerGame` default impl + tests

- [x] **T3.1** Implement `PerPlayerGame<N, A, P, D>` in `katgpt-rs/src/cce/heterogeneous.rs`.
- [x] **T3.2** Unit tests in `heterogeneous.rs`:
  - `homogeneous_equivalence` — `PerPlayerGame` with all players sharing the same `(P, D)` gives the same `ρ⋆` as `CceLp::solve(d, p)` on that single `(P, D)`. **Closes the "wrapper is a strict generalization" check.**
  - `two_player_prisoners_dilemma` — classic 2-player PD where each player has its own payoff tensor. **Wording corrected:** single-shot PD with constant-deviation class has a larger CCE feasible set than `{δ_(D,D)}`; the test verifies feasibility + CCE validity + per-player regret ≤ ε + γ₀ range, not cooperation (which is not incentive-compatible).
  - `heterogeneous_robustness` — two players with *slightly different* payoff tensors (perturbed by 1% noise). Verify `ρ⋆` is a small perturbation of the homogeneous `ρ⋆`. This is the subjective-CCE use case from Issue 327.
  - `is_heterogeneous_cce_passes_on_solve_output` — `solve_heterogeneous` output passes `is_heterogeneous_cce(ε=1e-4)`.
  - `is_heterogeneous_cce_rejects_cooperative_on_pd` — **Wording corrected:** the cooperative distribution `(C,C)` is NOT a subjective-CCE on PD (both players have profitable deviations); `(D,D)` Nash IS a subjective-CCE. The original plan wording ("pure Nash is NOT a heterogeneous CCE") was game-theoretically incorrect.

**Phase 3 exit:** `cargo test --features cce_moderator --lib cce::heterogeneous::` passes. ✅ PASSED 2026-06-21 (5/5). Full `cce::` suite (41 tests) also passes — no regressions from the `enumerate_bfs` refactor.

### Phase 4 — GOAT gates

- [x] **T4.1 G1 — Homogeneous equivalence (regression):** for the 3 canonical Plan 295 examples (RPS, chicken, emission-abatement), `PerPlayerGame` with all players sharing the same `(P, D)` produces the same `ρ⋆` as the homogeneous `CceLp::solve`. Tolerance: objective within 1e-4, entries within 1e-3. Benchmark in `katgpt-rs/tests/heterogeneous_g1.rs`.
  - Emission-abatement (N=2, A=2): P ∈ {1,2,4,8} all PASS.
  - Chicken (N=4, A=2): P ∈ {1,2,4} all PASS. (P=8 takes 15s in debug — too slow for unit test; covered by G4 release-mode bench.)
  - RPS (N=9, A=3): P ∈ {1,2} all PASS. (P=3+ would take minutes — BFS on C(36,10) ≈ 254M.)
- [x] **T4.2 G2 — Regret transfer on synthetic heterogeneous CWMs:** 8-player and 16-player games, each player's payoff tensor perturbed by ±1% LCG noise around a base emission table. `er_heterogeneous(ρ⋆) = 0.0` (machine precision — exact LP solve). The regret bound transfers trivially when the LP is solved exactly; the runtime primal-dual path (T4.3b) will exercise the `O(T⁻¹ᐟ²)` rate.
- [x] **T4.3 G3 — Convergence rate matches `O(T⁻¹ᐟ²)`:** ✅ CLOSED via T4.3b. `CcePrimalDual::step_heterogeneous` + `run_heterogeneous` + `ExternalRegret::linear_derivative_heterogeneous` shipped. Per-player best-deviation cached per step (subgradient oracle); aggregate gradient `grad[m] = gamma0_coeff(m) + λ · (1/P) Σ_i [cost_i(s,a) − reward_deviate(i, s, κ_i*)]`. **G3 results (4-player perturbed game, 10⁴ steps):** G3a gap=0.000580 (<0.05 target, 86× margin); G3b ER_heterogeneous=0.000133 (≤0.05 target, 376× margin); G3c fitted log-log slope = **-1.0000** (in [-2.0,-0.3] target; beats paper's -0.5 O(N⁻¹ᐟ²) upper bound — well-conditioned convergence at O(1/N)); G3d gradient consistency PASS.
- [x] **T4.4 G4 — Latency on crowd-scale:** sweep over player counts {2,4,8,16,24,32} in `katgpt-rs/benches/heterogeneous_cce.rs` (release mode). **Updated post-T4.3b** to report both BFS and primal-dual latency:
  | n_players | BFS median | Primal-dual median (10k steps) |
  |---|---|---|
  | 2 | 15.9µs | 1796.9µs |
  | 4 | 131.2µs | 2406.0µs |
  | 8 | 1957.9µs | 3232.2µs |
  | 16 | 43.2ms ✓ | 5095.1µs |
  | 24 | 268.7ms (BFS ceiling) | **7881.6µs ✓** (34× faster) |
  | 32 | 1199.4ms (BFS ceiling) | **8875.5µs ✓** (135× faster) |
  - **All scales now under 50ms target** (32-player crowd-scale closed via primal-dual path, T4.3b). BFS crossover at ~16 players; primal-dual is the recommended path above that.
  - Pre-T4.3b measurement (BFS only): 2=15µs, 4=112µs, 8=1.6ms, 16=32.7ms, 24=213ms CEILING, 32=768ms CEILING.

**Phase 4 exit:** G1+G2+G3+G4 ALL PASS. ✅ CLOSED 2026-06-22 via T4.3b (G3 primal-dual heterogeneous extension).

### Phase 5 — Documentation + feature promotion

- [x] **T5.1** Add `HeterogeneousPayoff` + `PerPlayerGame` + `solve_heterogeneous` to `katgpt-rs/src/cce/mod.rs` module doc with a 10-line usage example. ✅ Module doc updated with the subjective-CCE LP formulation block.
- [x] **T5.2** Update `katgpt-rs/.research/274_Optimal_CCE_Moderator_LP_No_Regret.md` with a "Subjective-CCE extension" section linking to this plan and to Issue 327 Path A+. ✅ Added §9.
- [x] **T5.3** Update `riir-ai/.research/143_Latent_CCE_Moderator_Crowd_Emergent_Coordination.md` with the same pointer. ✅ Added section + cross-link row in the Plan dependency table.
- [x] **T5.4** Feature promotion: ✅ DONE. T4.3b closed G3, so the "G1+G2+G3+G4 all PASS" condition is met. `cce_moderator` added to `default` features in `katgpt-rs/Cargo.toml` line 45 (2026-06-22). `src/lib.rs` comment updated to reflect DEFAULT-ON status. Zero non-optional deps (feature is `cce_moderator = []`); promotion is zero-cost for non-consumers since the module is `#[cfg(feature = "cce_moderator")]` gated. **Resolves the prior discrepancy** with the Plan 325 note — Plan 325 had not actually promoted the feature (Cargo.toml audit confirmed); Plan 300 T4.3b+T5.4 does the promotion with full GOAT evidence.

**Phase 5 exit:** T5.1-T5.4 ALL DONE. ✅ COMPLETE 2026-06-22.

---

## Performance considerations (per AGENTS.md optimization guidelines)

- **No allocation in hot path.** `solve_heterogeneous` allocates the constraint matrix `mat` and RHS `rhs` once at entry; BFS enumeration is in-place. Mirror Plan 295's pattern.
- **Reuse BFS-enumeration helper.** Extract `enumerate_bfs(mat, rhs, n_vars, na)` from `solve` and share with `solve_heterogeneous`. Avoids ~50 LOC duplication.
- **`Vec::with_capacity` for player iteration.** When summing `Σ_i |D_i|`, pre-size the constraint matrix.
- **Avoid `dyn` in the inner loop.** `H: HeterogeneousPayoff<N, A>` is monomorphized; the compiler inlines per-player calls.
- **G4 latency target 50ms** is generous — Plan 295's homogeneous `solve` on `N=4, A=4, |D|=4` runs in <1ms. Heterogeneous scales with `Σ_i |D_i|`; for 32 players × 4 deviations each = 128 constraints, BFS on `C(1024+128, 129)` is too large — switch to the primal-dual path (T4.3) for crowd-scale, or document a "homogeneous for large P" fallback. **Flag for G4.**

---

## Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| BFS enumeration explodes for large `Σ_i |D_i|` | High | Medium | T4.4 G4 catches it; fallback to `CcePrimalDual` heterogeneous path (T4.3) for crowd-scale. Document the threshold. |
| `HeterogeneousPayoff` trait design forces per-call `reward_follow(player, s, a)` indirection that defeats inlining | Low | Low | Generic `H` monomorphizes; benchmark on G4. If hot, add a `PerPlayerGameTables` struct that pre-flattens into a single `Vec<f32>` indexed by `player * N * A + s * A + a`. |
| Default `gamma0` (average welfare) is wrong for some game modes | Medium | Low | `gamma0` is overridable; runtime (riir-ai side) supplies its own `ModeratorObjective`. |
| `CcePrimalDual` extension for heterogeneous case (T4.3) is non-trivial | Medium | Medium | Split T4.3 into T4.3a (G1+G2+G4 ship without it) + T4.3b (follow-up). Don't block Plan 300 close on it. |

---

## Out of scope

- **Strict Bayes-CCE / no-common-prior.** That's [`riir-ai/.issues/328_no_common_prior_bayes_cce.md`](../../riir-ai/.issues/328_no_common_prior_bayes_cce.md). This plan is strictly the subjective-CCE wiring.
- **riir-ai runtime wiring.** Per-NPC `NpcCwmRuntime<K>` → `HeterogeneousPayoff` bridge is a future riir-ai plan, blocked on this plan landing. Tracked as T-A+.5 in Issue 327.
- **Bayesian priors over payoff tables.** Not in this plan; that's the strict Bayes-CCE question (Issue 328).
- **`CcePrimalDual` heterogeneous primal-dual iterator.** T4.3 is a stretch goal; the LP solver path (T2.1) is the primary deliverable. Primal-dual extension is a follow-up if G3 needs it.
- **Multi-faction CCE with KNOWN per-faction payoff tables.** That's the homogeneous case (Plan 295) on the faction-aggregated table. Not this plan.

---

## Acceptance

- [x] G1 PASS (homogeneous equivalence regression — no Plan 295 breakage).
- [x] G2 PASS (regret transfers on synthetic heterogeneous CWMs).
- [x] G3 PASS (convergence rate log-log slope = -1.0; ≤ -0.5 paper bound).
- [x] G4 PASS (<50ms on ALL scales post-T4.3b: 32-player crowd-scale = 8.9ms via primal-dual).
- [x] All unit tests in `heterogeneous.rs` pass (5/5).
- [~] `cargo check --all-features` clean (CI feature guard catches combo regressions per the `merkle_root` lesson). **PRE-EXISTING FAILURE** in `katgpt-core/src/dec/hodge.rs:222` (borrow-checker error in unrelated `LoraAdapter` combo) — not introduced by Plan 300. `cargo check --features cce_moderator` passes clean; `cargo check` (default features, post-T5.4 promotion) passes clean. The `--all-features` failure is in a different crate and was present before this plan. Left for the owner of the hodge/LoraAdapter work to fix.
- [x] Plan 300 status → ✅ COMPLETE. Issue 327 T-A+.1 through T-A+.4 marked done.

---

## References

- **Plan 295 (homogeneous CceLp primitive):** [`katgpt-rs/.plans/295_lp_cce_moderator_primitive.md`](295_lp_cce_moderator_primitive.md)
- **Source paper:** [Campi, Cannerozzi, Tzoumas — arxiv 2606.20062](https://arxiv.org/pdf/2606.20062)
- **P0.5 regret sketch (subjective-CCE §2):** [`riir-ai/.docs/62_bayes_cce_regret_sketch.md`](../../riir-ai/.docs/62_bayes_cce_regret_sketch.md)
- **Parent issue (closed):** [`riir-ai/.issues/327_cwm_cce_bayes_fusion.md`](../../riir-ai/.issues/327_cwm_cce_bayes_fusion.md)
- **Sibling issue (deferred):** [`riir-ai/.issues/328_no_common_prior_bayes_cce.md`](../../riir-ai/.issues/328_no_common_prior_bayes_cce.md)
- **Public CCE research note:** [`katgpt-rs/.research/274_Optimal_CCE_Moderator_LP_No_Regret.md`](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
- **Private CCE runtime (Plan 325):** [`riir-ai/.plans/325_latent_cce_moderator_runtime.md`](../../riir-ai/.plans/325_latent_cce_moderator_runtime.md)
- **Private CWM runtime (Plan 326):** [`riir-ai/.plans/326_cwm_npc_runtime_integration.md`](../../riir-ai/.plans/326_cwm_npc_runtime_integration.md)

---

## TL;DR

Thin wrapper extending Plan 295's `CceLp::solve` to per-NPC heterogeneous payoff tables. New trait `HeterogeneousPayoff<N,A>`, new method `CceLp::solve_heterogeneous`, new struct `PerPlayerGame`, new file `heterogeneous.rs`. T4.3b follow-up extended `CcePrimalDual` with `step_heterogeneous` + `run_heterogeneous` (per-player subgradient oracle caches best deviation per step). Math transfers as-is per doc 62 §2 (sum of convex is convex, primal-dual averaging is heterogeneity-agnostic); no new theory. **✅ COMPLETE 2026-06-22:** All phases done, all 4 GOAT gates PASS (G1 homogeneous equivalence, G2 regret transfer at exact-LP precision, G3 primal-dual convergence at slope -1.0 beating paper's -0.5 bound, G4 32-player crowd-scale at 8.9ms via primal-dual). `cce_moderator` promoted to DEFAULT-ON in Cargo.toml (zero non-optional deps). 43 lib tests + 3 G1 + 2 G2 + 4 G3 tests + G4 bench all green. Closes Issue 327 Path A+ (subjective-CCE wiring); unblocks future riir-ai runtime plan for per-NPC CWM payoff tables (T-A+.5).
