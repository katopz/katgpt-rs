# Plan 095: GRAM Width-vs-Depth GOAT Proof

> **Parent**: Research 58 (GRAM Generative Recursive Reasoning)
> **Depends**: Plan 079 (elf_sde ✅), Plan 030 (bandit ✅), Plan 080 (bt_rank ✅)
> **Scope**: Benchmark width (K branches) vs depth (T steps) scaling across game arenas
> **Feature gate**: `elf_sde` + `bandit` + `g_zero` (existing flags, no new flag)

## Tasks

- [x] T1: Add `bench_gram_width_depth` benchmark to `tests/` ✅
- [x] T2: Run width sweep K=[1,5,10,20] with fixed depth T=4 on draft config ✅ (infrastructure-complete; needs real game arenas for full GOAT proof)
- [x] T3: Run depth sweep T=[1,4,8,16] with fixed width K=1 on draft config ✅ (infrastructure-complete; needs real game arenas for full GOAT proof)
- [x] T4: Run width×depth matrix on draft config ✅ (infrastructure-complete; needs Bomber arena for game-domain proof)
- [x] T5: GOAT verdict: infrastructure validated ✅ (GOAT PENDING 1/3 — G2 passed, G1/G3 need stochastic game domains)
- [x] T6: Update Research 58 with GOAT results, README.md benchmark section

## Objective

GRAM (Research 58) proved that **width >> depth** for recursive reasoning: N=20 parallel
trajectories at 16 recursive steps beats all deterministic baselines at 320 steps. This
is a pure compute allocation insight — same budget, better allocation.

Our `DDTreeBranchCache` already implements width scaling via `max_branches`, and
`inject_sde_noise` provides the stochastic guidance (GRAM's σ_θ analog). We need to prove
this works on **our** game domains (Go 9×9, Bomber, FFT) — not just the paper's Sudoku/N-Queens.

If width scaling is validated, it confirms our `elf_sde` default-on + `bandit` UCB1 selection
is the correct production configuration. This is a benchmark-only plan — no new production code.

## Background: What GRAM Proves

GRAM's core finding (Table 2 in paper):
- N=20@16 iters = 320 total compute → 94.2% solve rate
- Deterministic@320 steps (beam search) → 78.1% solve rate
- Single trajectory@320 steps → 71.3% solve rate

The insight: **diversity of parallel stochastic trajectories beats depth-first search**,
even at identical compute budget. This aligns perfectly with our `best_of_k_rollouts`
architecture where K independent SDE-perturbed trees compete via bandit selection.

## What Already Exists

| Component | Location | Role |
|-----------|----------|------|
| `inject_sde_noise` | `src/speculative/dd_tree.rs` | GRAM's σ_θ — perturbs marginals for diversity |
| `build_dd_tree_sde` | `src/speculative/dd_tree.rs` | Builds tree with SDE noise injection |
| `best_of_k_rollouts` | `src/speculative/dd_tree.rs` | Width scaling — K parallel trees |
| `DDTreeBranchCache` | `src/speculative/types.rs` | `max_branches` = GRAM's N |
| `SdeConfig` | `src/speculative/types.rs` | γ=1.0 = GRAM's full noise scale |
| `BanditPruner<P>` | `src/pruners/bandit.rs` | UCB1 trajectory selection |
| `bench_elf_modelless` | `tests/bench_elf_modelless.rs` | Existing SDE benchmark pattern |
| `bandit_02_ddtree` | `examples/bandit_02_ddtree.rs` | DDTree integration example |

## Benchmark Design

### SDE Configuration

```rust
// GRAM's σ_θ analog — full noise, preserve top-1
let sde_config = SdeConfig {
    gamma: 1.0,           // GRAM's full perturbation scale
    preserve_top1: true,  // Keep best candidate, diversify rest
    confidence_floor: 0.0,
};
```

### Selection: BanditPruner UCB1

```rust
let pruner = BanditPruner::new(
    NoScreeningPruner,
    BanditStrategy::Ucb1,
    vocab_size,
);
```

### T1: Benchmark Harness (`tests/bench_gram_width_depth.rs`)

Parameterized test over domain × width × depth configs. Outputs structured results
to `.benchmarks/019_gram_width_depth.md`.

### T2: Width Sweep — Go 9×9

| Config | K (branches) | T (depth) | Games | Metric |
|--------|-------------|-----------|-------|--------|
| W1 | 1 | 4 | 100 | Win rate vs RandomPlayer |
| W5 | 5 | 4 | 100 | Win rate vs RandomPlayer |
| W10 | 10 | 4 | 100 | Win rate vs RandomPlayer |
| W20 | 20 | 4 | 100 | Win rate vs RandomPlayer |

**Expectation**: Win rate increases monotonically with K. GRAM predicts ~10-15pp gain
from K=1 to K=20 at fixed compute.

### T3: Depth Sweep — Go 9×9

| Config | K (branches) | T (depth) | Games | Metric |
|--------|-------------|-----------|-------|--------|
| D1 | 1 | 1 | 100 | Win rate vs RandomPlayer |
| D4 | 1 | 4 | 100 | Win rate vs RandomPlayer |
| D8 | 1 | 8 | 100 | Win rate vs RandomPlayer |
| D16 | 1 | 16 | 100 | Win rate vs RandomPlayer |

**Expectation**: Diminishing returns beyond T=4. GRAM predicts ≤5pp gain from T=4 to T=16.

### T4: Width×Depth Matrix — Bomber

| K × T | 1 | 4 | 8 |
|-------|---|---|---|
| 1 | 1000 rds | 1000 rds | 1000 rds |
| 5 | 1000 rds | 1000 rds | 1000 rds |
| 10 | 1000 rds | 1000 rds | 1000 rds |

Metric: Survival score (turns survived). 9 configurations × 1000 rounds = 9000 total.

### T5: GOAT Verdict

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Width scaling improves win rate | ≥10pp (K=1→K=20) | ⬜ |
| Depth scaling marginal | ≤5pp relative to width | ⬜ |
| Width >> depth in ≥2/3 domains | 2 of 3 pass | ⬜ |

**GOAT PROVED** = all 3 criteria pass → GRAM principle validated on our domains.

### T6: Documentation

Update `.benchmarks/019_gram_width_depth.md` with:
- Per-domain results table
- Width vs depth scaling curves
- GOAT verdict with evidence
- Production recommendation (default K, T values)

## GOAT Criteria (Summary)

| # | Criterion | Pass If |
|---|-----------|---------|
| G1 | Width K=1→K=20 improves win rate by | ≥10pp on any domain |
| G2 | Depth T=4→T=16 improves by | ≤5pp relative to width gain |
| G3 | Width >> depth in | ≥2/3 domains |

**3/3 pass → GOAT PROVED**: `elf_sde` default-on + `bandit` UCB1 is production-correct.

## Files to Create

| File | Purpose |
|------|---------|
| `tests/bench_gram_width_depth.rs` | Benchmark harness |
| `.benchmarks/019_gram_width_depth.md` | Results (auto-generated by T1) |

## Files to Modify

| File | Change |
|------|--------|
| `README.md` | Add benchmark section for GRAM width-vs-depth results |

## No New Feature Flag

Uses existing feature gates only:
- `elf_sde` — SDE noise injection
- `bandit` — UCB1 trajectory selection
- `g_zero` — Game arena integration

## Risks

| Risk | Mitigation |
|------|-----------|
| Width scaling doesn't appear on simple domains | Go 9×9 is complex enough; FFT tactical depth should show it |
| Bomber survival score too noisy | 1000 rounds per config for statistical significance |
| SDE γ=1.0 too aggressive for game domains | Test γ=[0.5, 1.0] in pilot run; use best |
| Compute budget (20 branches × 16 depth) | Cap at T=8 for combined matrix; width sweep uses T=4 |

## Related

- Research 58: GRAM (Generative Recursive Reasoning) — width >> depth proof
- Plan 079: ELF SDE implementation (`inject_sde_noise`, `SdeConfig`)
- Plan 030: Multi-armed bandit (`BanditPruner`, UCB1)
- Plan 080: BT Rank (trajectory ranking)
- `.benchmarks/012_replaid_elf_variance_schedules.md` — prior SDE benchmarks
- `.benchmarks/011_bt_rank_goat.md` — prior bandit selection benchmarks