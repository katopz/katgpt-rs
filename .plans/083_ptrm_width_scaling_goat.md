# Plan 083: PTRM Width-Scaling GOAT Proof

> **Status:** ✅ Done
> **Branch:** `develop/feature/083_ptrm_width_scaling`
> **Depends on:** Plan 079 (ELF SDE, GOAT proved), Plan 030 (BanditPruner), Plan 080 (MaxSim)
> **Research:** `.research/049_PTRM_Probabilistic_Tiny_Recursive_Model.md`
> **Source:** arXiv:2605.19943 — Probabilistic Tiny Recursive Model
> **Goal:** Validate PTRM's width >> depth finding on our speculative decoding stack. Add `best_of_k_rollouts` convenience API and `EarlyStopGate` pruner. Benchmark K vs T scaling to produce GOAT proof.

## Summary

PTRM (arXiv:2605.19943) proves that **width scaling (K=64 parallel rollouts) >> depth scaling (T=64 recursion steps)**: +28.6pp vs +3.1pp on PPBench. Our stack already has every component PTRM uses:

| PTRM Component | Our Equivalent | Status |
|---|---|---|
| Gaussian noise σ at each step | `inject_sde_noise` (γ=SdeConfig.gamma) | ✅ GOAT proved (Plan 079) |
| K parallel rollouts | `DDTreeBranchCache` (max_branches=K) | ✅ Implemented |
| Q-head trajectory selection | `BanditPruner<P>` Q-values | ✅ Implemented |
| Q-head early stopping | `BanditPruner::dual_cutoff` | ✅ Partial — no depth-aware gate |

No new architecture needed. Two small additions:

1. **`best_of_k_rollouts`** — Ergonomic wrapper for "run K independent SDE rollouts, select best by cumulative relevance." Enables systematic K-vs-T benchmarking.
2. **`EarlyStopGate<P>`** — `ScreeningPruner` wrapper that prunes branches whose cumulative Q falls below threshold at depth > 0. Maps to PTRM's Q-head early stopping.

Plus a benchmark to validate PTRM's width >> depth claim on our stack.

---

## Tasks

- [x] **T1: `best_of_k_rollouts` convenience function** — Width scaling API
  - Add `best_of_k_rollouts()` to `src/speculative/dd_tree.rs`
  - Signature: `(marginals, config, screener, sde_config, k_rollouts, rng) -> Vec<usize>`
  - Each rollout gets independent noise seed (seed + k offset)
  - Returns path with highest cumulative relevance score
  - Feature gate: `#[cfg(feature = "elf_sde")]`
  - Add `WidthScaleConfig` struct with `k_rollouts: usize` and `selection: WidthSelectionMode`
  - `WidthSelectionMode` enum: `BestQ` (highest cumulative relevance), `MostFrequent` (mode@K), `BtRank` (pairwise if `bt_rank` feature enabled)
  - ~50 lines of code
  - Location: `src/speculative/dd_tree.rs` (after `inject_sde_noise`)

- [x] **T2: `EarlyStopGate<P>` ScreeningPruner wrapper** — Q-head early stopping
  - Add `EarlyStopGate<P: ScreeningPruner>` to `src/speculative/types.rs`
  - Implements `ScreeningPruner` with confidence_threshold gate
  - At depth > 0: if `inner.relevance() < threshold`, return 0.0 (prune)
  - At depth 0: always passthrough (need at least one candidate)
  - `enabled: bool` field for runtime toggle
  - Feature gate: `#[cfg(feature = "elf_sde")]`
  - ~40 lines of code
  - Location: `src/speculative/types.rs` (after `NoScreeningPruner`)

- [x] **T3: Width vs Depth scaling benchmark** — GOAT proof
  - Add benchmark file: `tests/bench_ptrm_width_scaling.rs`
  - Measures accuracy on synthetic task across:
    - Width K: [1, 2, 4, 8, 16, 32, 64] rollouts
    - Depth T: [1, 2, 4, 8] draft_lookahead steps
    - Noise γ: [0.0, 0.2, 0.5, 1.0] SDE scale
  - Selection modes: `BestQ`, `MostFrequent`
  - Report: accuracy, diversity (unique paths), latency
  - Expected result: K scaling dominates T scaling (PTRM: 28.6pp vs 3.1pp)
  - Output: `.benchmarks/015_ptrm_width_scaling.md`
  - Feature gate: `#[cfg(all(feature = "elf_sde", feature = "bandit"))]`
  - Location: `tests/bench_ptrm_width_scaling.rs`

- [x] **T4: Domain inference budget integration** — Width scale via config
  - Add `width_rollouts` field to domain config (default: 1)
  - Maps to `k_rollouts` in `best_of_k_rollouts`
  - Add `early_stop_threshold` field (default: 0.0, disabled)
  - Maps to `EarlyStopGate::confidence_threshold`
  - Location: `src/types.rs` (Config struct)

- [x] **T5: Update docs and references**
  - Update `README.md` — add PTRM section under "🧪 Tech Stack"
  - Update `.docs/09_heuristic-learning.md` — cross-reference PTRM
  - Mark Research 49 as complete

---

## Design Decisions

### Why no new feature flag

PTRM's ideas distill into existing features (`elf_sde` for noise, `bandit` for Q-values). No separate `ptrm` flag needed. The `best_of_k_rollouts` and `EarlyStopGate` are convenience wrappers around existing proven infrastructure.

### Why `EarlyStopGate` instead of modifying `BanditPruner`

`BanditPruner::dual_cutoff` already provides arm-level cutoff. But it doesn't consider **depth** — a path that looked good at depth 0 may decay at depth 3. `EarlyStopGate` wraps any `ScreeningPruner` and adds depth-aware early stopping. This is composable: `EarlyStopGate<BanditPruner<FlowPruner<...>>>`.

### Why not Langevin / gradient-guided noise

PTRM's own negative result (Appendix C): Langevin sampling with Q-head gradients contributes **zero measurable improvement** over pure Gaussian noise. Our `inject_sde_noise` already uses simple Gaussian. No changes needed.

---

## Feature Gate Summary

| Addition | Gate | Reason |
|---|---|---|
| `best_of_k_rollouts()` | `elf_sde` | Requires SDE noise injection |
| `EarlyStopGate<P>` | `elf_sde` | Requires SDE noise injection |
| `WidthScaleConfig` | `elf_sde` | Configuration for width scaling |
| `WidthSelectionMode` | `elf_sde` | Selection strategy enum |
| `bench_ptrm_width_scaling` | `elf_sde + bandit` | Requires both noise and bandit |

No new feature flags. All additions are gated under existing `elf_sde`.

---

## Success Criteria

1. **T1**: `best_of_k_rollouts` with K=16 produces ≥2× more unique paths than K=1 (diversity)
2. **T2**: `EarlyStopGate` with threshold=0.3 reduces tree size by ≥20% with ≤2% accuracy loss
3. **T3**: Benchmark shows width scaling (K=1→64) provides ≥3× the gain of depth scaling (T=1→8)
4. **T4**: Domain config drives width scaling without code changes

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Width scaling doesn't help on our tasks | PTRM validates on reasoning puzzles; our Go/Sudoku tasks are similar. If no gain, still a useful negative result. |
| EarlyStopGate over-prunes | Configurable threshold. Default 0.0 = disabled. Start conservative. |
| Benchmark takes too long | K=64 × T=8 = 512 configs × N samples. Use rayon parallelism. Budget ≤30 min. |
| Selection mode matters | Test both BestQ and MostFrequent. PTRM shows BestQ >> MostFrequent on reasoning tasks. |

---

## References

- **PTRM Paper**: arXiv:2605.19943 — Probabilistic Tiny Recursive Model
- **Research 49**: `.research/049_PTRM_Probabilistic_Tiny_Recursive_Model.md`
- **Research 44**: `.research/044_ELF_Embedded_Language_Flows.md` (SDE noise, Plan 079)
- **Research 35**: `.research/035_Attractor_Models_Fixed_Point_Refinement.md` (attractor = recursive refinement)
- **Plan 079**: `.plans/079_elf_embedded_language_flows_modelless.md` (SDE GOAT proof)
- **Plan 030**: `.plans/030_multi_armed_bandit.md` (BanditPruner)
- **Key files**: `src/speculative/dd_tree.rs`, `src/speculative/types.rs`, `src/pruners/bandit.rs`
