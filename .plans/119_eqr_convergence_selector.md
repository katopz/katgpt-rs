# Plan 119: EqR Convergence-Based Selection for Breadth Scaling

> **Status:** ⬜ Not Started
> **Branch:** `develop/feature/119_eqr_convergence_selector`
> **Depends on:** Plan 079 (ELF SDE ✅), Plan 083 (PTRM width scaling ✅), Plan 030 (BanditPruner ✅)
> **Research:** `.research/079_EqR_Equilibrium_Reasoners.md` (pending creation)
> **Source:** arXiv:2605.21488 — Equilibrium Reasoners (EqR)
> **Feature gate:** `eqr_convergence` (opt-in, depends on `elf_sde`)
> **Goal:** Implement the ONE actionable insight from EqR that we don't already have: **convergence-based selection for breadth scaling**. After landscape shaping (RI + NI training), the fixed-point residual ∥fθ(z;x)−z∥ becomes a reliable proxy for answer correctness. Selecting the rollout with smallest residual (Top-1 Converged) beats majority voting and matches oracle selection.

## Summary

EqR proves that after **landscape shaping** (RI + NI training), the fixed-point residual
∥fθ(z;x) − z∥ becomes a reliable proxy for answer correctness. The key insight:

| Condition | Residual Reliability | Selection Quality |
|-----------|---------------------|-------------------|
| Before landscape shaping | ❌ UNRELIABLE — converges to spurious attractors | Worse than random |
| After landscape shaping | ✅ RELIABLE — residual correlates with correctness | Matches oracle |

**What this means for us:** Our `best_of_k_rollouts` already has `WidthSelectionMode::BestQ` and `MostFrequent`. EqR adds a third selection mode: **Top1Converged** — pick the rollout whose latent trajectory has the smallest final residual ∥z_{k+1} − z_k∥. This is only valid after the model has been trained with RI + NI (our `elf_sde` + loop training).

### What We Already Have (DO NOT reimplement)

| Component | Location | Role |
|-----------|----------|------|
| `best_of_k_rollouts()` | `src/speculative/dd_tree.rs` | Width scaling — K parallel trees |
| `inject_sde_noise()` | `src/speculative/dd_tree.rs` | Noise injection (EqR NI analog) |
| `BanditPruner<P>` with UCB1 | `src/pruners/bandit.rs` | Q-value trajectory selection |
| `DDTreeBranchCache` with `max_branches` | `src/speculative/types.rs` | Breadth scaling |
| `width_rollouts` in Config | `crates/microgpt-core/src/types.rs` | Rollout count configuration |
| `LoopMode::WeightShared` | `crates/microgpt-core/src/types.rs` | Weight-shared iteration |
| `ResidualGate` | `crates/microgpt-core/src/types.rs` | Per-loop residual gate |
| `SdpaOutputGate` | `crates/microgpt-core/src/types.rs` | Attention sink suppression |
| HLA/AHLA linear attention | `src/attention/` | Constant-state latent recursion |
| `WidthSelectionMode` enum | `src/speculative/dd_tree.rs` | BestQ, MostFrequent selection |

### What We NEED to Implement

| Addition | Purpose |
|----------|---------|
| `ConvergenceSelector` enum | Selection strategy enum (BestQ, MajorityVote, Top1Converged, BtRank) |
| `ResidualTracker` struct | Track ∥z_{k+1} − z_k∥ across loop iterations |
| `best_of_k_rollouts` integration | Support Top1Converged selection mode |
| `convergence_selector` Config field | Default: BestQ (no behavior change) |
| GOAT proof test | Validate residual predicts correctness on synthetic task |
| Benchmark comparison | BestQ vs Top1Converged vs MajorityVote at K=[1,4,8,16,32] |

---

## Tasks

- [ ] **T1: Add `ConvergenceSelector` enum** — Selection strategy taxonomy
  - Location: `crates/microgpt-core/src/types.rs` (after `ResidualGate`)
  - Feature gate: `#[cfg(feature = "eqr_convergence")]`
  - Variants:
    - `BestQ` — Highest cumulative relevance (current default, PTRM)
    - `MajorityVote` — Most common path across rollouts (mode@K)
    - `Top1Converged` — Smallest final residual ∥z_{k+1} − z_k∥ (EqR)
    - `BtRank` — Pairwise Bradley-Terry ranking (if `bt_rank` feature)
  - `#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]`
  - Default: `BestQ` (no behavior change)
  - ~15 lines of code

- [ ] **T2: Add `ResidualTracker` struct** — Per-rollout residual tracking
  - Location: `src/speculative/dd_tree.rs` (after `WidthScaleConfig`)
  - Feature gate: `#[cfg(feature = "eqr_convergence")]`
  - Struct fields:
    - `residuals: Vec<f32>` — ∥z_{k+1} − z_k∥ at each loop step
    - `max_steps: usize` — Capacity hint
  - Methods:
    - `new(max_steps: usize) -> Self` — Pre-allocate
    - `record_step(&mut self, z_prev: &[f32], z_curr: &[f32])` — Compute and store ∥z_curr − z_prev∥₂
    - `final_residual(&self) -> f32` — Last recorded residual (0.0 if empty)
    - `mean_residual(&self) -> f32` — Average across all steps
    - `is_converged(&self, threshold: f32) -> bool` — `final_residual() < threshold`
  - Implementation:
    ```rust
    // Using blake3-style simple L2 norm (no external deps)
    pub fn record_step(&mut self, z_prev: &[f32], z_curr: &[f32]) {
        let diff: f32 = z_prev.iter().zip(z_curr.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        self.residuals.push(diff.sqrt());
    }
    ```
  - ~40 lines of code

- [ ] **T3: Integrate into `best_of_k_rollouts()`** — Top1Converged selection
  - Location: `src/speculative/dd_tree.rs`
  - Feature gate: `#[cfg(feature = "eqr_convergence")]` on new match arms
  - Changes to `WidthSelectionMode` (extend existing enum):
    - Add `Top1Converged` variant to `WidthSelectionMode`
  - Changes to `best_of_k_rollouts()`:
    - When `selection == Top1Converged`: compute `ResidualTracker` per rollout
    - For discrete DDTree: approximate residual via marginal change ∥p_{k+1} − p_k∥₁
      between successive expansion depths (no latent state, use marginals as proxy)
    - Select rollout with smallest `final_residual()`
  - Fallback: if no residual data available (e.g., single depth), fall back to `BestQ`
  - ~30 lines of code
  - **Important**: This is an approximation. True EqR residual requires latent state z
    from loop iterations. Our DDTree is depth-first on marginals, not iterative on latents.
    The marginal-change proxy ∥p_{depth+1} − p_{depth}∥ is a reasonable discrete analog.

- [ ] **T4: Add `convergence_selector` to Config** — Configuration wiring
  - Location: `crates/microgpt-core/src/types.rs`
  - Feature gate: `#[cfg(feature = "eqr_convergence")]`
  - Add field: `pub convergence_selector: ConvergenceSelector` (default: `BestQ`)
  - Add to `InferenceOverrides`: `pub convergence_selector: Option<ConvergenceSelector>`
  - Wire in `with_overrides()` method
  - Add to all Config constructors (`micro`, `game`, `game_go`, `draft`, `small_target`, etc.)
  - Update `test_with_overrides_all_fields` test
  - ~20 lines of code across multiple constructors

- [ ] **T5: GOAT proof test** — Top1Converged validates residual predicts correctness
  - Location: `tests/test_eqr_convergence_selector.rs`
  - Feature gate: `#[cfg(all(feature = "eqr_convergence", feature = "elf_sde"))]`
  - Test: `test_top1_converged_beats_majority_vote`
    - Create synthetic marginals with known "correct" path
    - Run K=16 rollouts with SDE noise
    - Assert Top1Converged selects correct path more often than MajorityVote
    - Rationale: if residual correlates with correctness, Top1Converged should win
  - Test: `test_residual_tracker_l2_norm`
    - Unit test: record known vectors, verify L2 norm computation
  - Test: `test_residual_decreases_with_convergence`
    - Simulate converging marginals (p_{k+1} → p_k)
    - Verify residuals decrease monotonically
  - ~80 lines of code

- [ ] **T6: Benchmark comparison** — BestQ vs Top1Converged vs MajorityVote
  - Location: `tests/bench_eqr_convergence.rs`
  - Feature gate: `#[cfg(all(feature = "eqr_convergence", feature = "elf_sde"))]`
  - Parameters:
    - K (rollouts): [1, 4, 8, 16, 32]
    - Selection modes: BestQ, MajorityVote, Top1Converged
    - Noise γ: [0.5, 1.0] (SDE scale)
    - Trials: 20 per config (fixed seeds)
  - Metrics:
    - Path quality (cumulative relevance)
    - Top-1 agreement with greedy baseline
    - Path diversity (unique paths / total rollouts)
    - Latency per selection
  - Config: `Config::draft()` with `draft_lookahead = 4`
  - Output: `.benchmarks/020_eqr_convergence_selector.md`
  - Expected: Top1Converged ≥ BestQ on path quality when SDE is active
  - ~120 lines of code

- [ ] **T7: Update docs and references**
  - Create `.research/079_EqR_Equilibrium_Reasoners.md` — research note
  - Update `README.md` — add EqR convergence section under "🧪 Tech Stack"
  - Update `src/speculative/dd_tree.rs` module doc — reference EqR selection
  - Mark task complete in this plan

---

## Design Decisions

### Why `ConvergenceSelector` as separate enum (not extending `WidthSelectionMode`)

The user spec requests `ConvergenceSelector` as a new enum. However, architecturally this
duplicates `WidthSelectionMode` which already has `BestQ` and `MostFrequent`. The cleanest
approach is to **extend `WidthSelectionMode`** with the `Top1Converged` variant, making
`ConvergenceSelector` a type alias or removing it. This avoids two parallel enums with
overlapping variants.

**Decision:** Extend `WidthSelectionMode` with `Top1Converged`. Keep `ConvergenceSelector`
as a type alias for backward compatibility if user code references it.

### Why marginal-change proxy instead of true latent residual

EqR's fixed-point residual ∥fθ(z;x) − z∥ operates in latent space. Our `best_of_k_rollouts`
operates on **discrete marginals** (token probability distributions). We approximate:

```
EqR:   ∥z_{k+1} − z_k∥₂     (latent space)
Ours:  ∥p_{d+1} − p_d∥₁     (marginal space, depth d in DDTree)
```

This is a reasonable proxy because:
1. Marginals are the output of the model at each depth — they reflect the latent state
2. A path whose marginals stop changing (low ∥p_{d+1} − p_d∥) has "converged" in output space
3. EqR's insight (convergence = correctness proxy) should transfer to output space

**Caveat:** This is NOT a true fixed-point residual. If the GOAT proof fails, the marginal
proxy may be insufficient and we'd need true latent state access (requires `LoopMode` integration).

### Why feature gate `eqr_convergence` instead of just `elf_sde`

EqR's convergence-based selection is a **research-grade** feature with specific preconditions:
1. Model must be trained with RI + NI (landscape shaping)
2. Residual is UNRELIABLE without landscape shaping
3. Our marginal-change proxy is an approximation, not the real thing

Making it a separate opt-in feature (`eqr_convergence` depends on `elf_sde`) ensures users
don't accidentally enable it on untrained models where it would degrade performance.

### Why default is `BestQ` not `Top1Converged`

`BestQ` (highest cumulative relevance) is our proven default from Plan 083 (PTRM). Until
the GOAT proof validates that Top1Converged actually beats BestQ on our stack, we keep
BestQ as default. If T5/T6 prove Top1Converged superior, we switch default in a follow-up.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    best_of_k_rollouts()                          │
│                  (src/speculative/dd_tree.rs)                    │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Rollout 0 │    │ Rollout 1    │    │ Rollout K-1         │   │
│  │ seed+0   │    │ seed+1       │    │ seed+K-1            │   │
│  └────┬─────┘    └──────┬───────┘    └──────────┬──────────┘   │
│       │                 │                       │               │
│  ┌────▼─────┐    ┌──────▼───────┐    ┌──────────▼──────────┐   │
│  │ SDE noise│    │ SDE noise    │    │ SDE noise           │   │
│  │ + DDTree │    │ + DDTree     │    │ + DDTree            │   │
│  └────┬─────┘    └──────┬───────┘    └──────────┬──────────┘   │
│       │                 │                       │               │
│  ┌────▼─────┐    ┌──────▼───────┐    ┌──────────▼──────────┐   │
│  │ path_0   │    │ path_1       │    │ path_{K-1}          │   │
│  │ score_0  │    │ score_1      │    │ score_{K-1}         │   │
│  │ resid_0  │    │ resid_1      │    │ resid_{K-1}         │   │
│  └────┬─────┘    └──────┬───────┘    └──────────┬──────────┘   │
│       │                 │                       │               │
│       └─────────────────┼───────────────────────┘               │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │ WidthSelectionMode  │                            │
│              ├─────────────────────┤                            │
│              │ BestQ    → max(score)                           │
│              │ MostFreq → mode(paths)                          │
│              │ Top1Conv → min(residual)  ← NEW (EqR)          │
│              └──────────┬──────────┘                            │
│                         ▼                                       │
│                  selected path                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ResidualTracker                                │
│              (src/speculative/dd_tree.rs)                        │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │ residuals: Vec<f32>                  │                       │
│  │ max_steps: usize                     │                       │
│  ├──────────────────────────────────────┤                       │
│  │ record_step(z_prev, z_curr)          │  ∥z_curr - z_prev∥₂ │
│  │ final_residual() -> f32              │  Last recorded        │
│  │ mean_residual() -> f32               │  Average              │
│  │ is_converged(threshold) -> bool      │  final < threshold    │
│  └──────────────────────────────────────┘                       │
│                                                                  │
│  Discrete approximation:                                          │
│  record_step(marginals[d], marginals[d+1])                       │
│  → ∥p_{d+1} - p_d∥₂ as convergence proxy                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Gate Summary

| Addition | Gate | Depends On | Reason |
|----------|------|------------|--------|
| `ConvergenceSelector` / `Top1Converged` variant | `eqr_convergence` | `elf_sde` | Requires SDE noise for diverse rollouts |
| `ResidualTracker` | `eqr_convergence` | — | Pure math, no model deps |
| `best_of_k_rollouts` Top1Converged arm | `eqr_convergence` | `elf_sde` | Integration point |
| `convergence_selector` in Config | `eqr_convergence` | — | Configuration field |
| GOAT proof tests | `eqr_convergence` + `elf_sde` | — | Validation |
| Benchmark | `eqr_convergence` + `elf_sde` | — | Performance comparison |

**Cargo.toml addition:**
```toml
[features]
eqr_convergence = ["elf_sde"]
```

---

## Success Criteria

| # | Criterion | Pass If |
|---|-----------|---------|
| G1 | Top1Converged ≥ BestQ on path quality | Mean quality within 5% or better |
| G2 | Residual correlates with correctness | Pearson r ≥ 0.3 on synthetic task |
| G3 | No regression on existing tests | All `elf_sde` tests pass |
| G4 | Zero-cost when disabled | No overhead when `eqr_convergence` off |

**GOAT PROVED** = G1 + G2 pass → EqR convergence selection validated on our stack.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Marginal proxy insufficient | Medium | High — Top1Converged worse than BestQ | Fallback to BestQ; document as negative result |
| Residual unreliable without RI+NI training | High (on untrained models) | Medium | Feature gate opt-in; docs warn about precondition |
| Feature gate dependency chain too deep | Low | Low | `eqr_convergence` → `elf_sde` only, no deeper |
| Benchmark noise obscures signal | Medium | Low | 20 trials per config, fixed seeds |
| Adds complexity for marginal gain | Medium | Medium | Keep code minimal (~100 LOC production); remove if GOAT fails |

---

## Files to Create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `tests/test_eqr_convergence_selector.rs` | GOAT proof tests | ~80 |
| `tests/bench_eqr_convergence.rs` | Benchmark comparison | ~120 |
| `.benchmarks/020_eqr_convergence_selector.md` | Results (auto-generated by T6) | — |
| `.research/079_EqR_Equilibrium_Reasoners.md` | Research note | — |

## Files to Modify

| File | Change | Lines (est.) |
|------|--------|-------------|
| `crates/microgpt-core/src/types.rs` | Add `ConvergenceSelector` + Config field | ~25 |
| `crates/microgpt-core/src/lib.rs` | Export `ConvergenceSelector` | ~1 |
| `src/speculative/dd_tree.rs` | Add `ResidualTracker` + Top1Converged selection | ~70 |
| `src/speculative/mod.rs` | Export `ResidualTracker` | ~1 |
| `Cargo.toml` (workspace + crate) | Add `eqr_convergence` feature | ~3 |
| `README.md` | Add EqR section | ~5 |

**Total production code:** ~100 LOC (excluding tests/benchmarks)

---

## References

- **EqR Paper:** arXiv:2605.21488 — Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning (CMU, May 2026)
- **Research 079:** `.research/079_EqR_Equilibrium_Reasoners.md` (pending)
- **Research 044:** `.research/044_ELF_Embedded_Language_Flows.md` (SDE noise, Plan 079)
- **Research 049:** `.research/049_PTRM_Probabilistic_Tiny_Recursive_Model.md` (width scaling, Plan 083)
- **Research 035:** `.research/035_Attractor_Models_Fixed_Point_Refinement.md` (fixed-point theory)
- **Plan 079:** `.plans/079_elf_embedded_language_flows_modelless.md` (SDE GOAT proof)
- **Plan 083:** `.plans/083_ptrm_width_scaling_goat.md` (width scaling, `best_of_k_rollouts`)
- **Plan 030:** `.plans/030_multi_armed_bandit.md` (BanditPruner, UCB1)

---

## Key Principle

> **Residual is only reliable AFTER landscape shaping.** Before RI+NI training, fixed-point
> iteration converges to spurious attractors and the residual is meaningless. This feature
> is only useful for models trained with our `elf_sde` + loop training pipeline. The feature
> gate and default-off configuration reflect this precondition.