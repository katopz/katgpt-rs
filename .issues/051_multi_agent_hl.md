# Issue 051: Multi-Agent Heuristic Learning — Shared Bandit Coordination

> **Plan:** 032 (Heuristic Learning Infrastructure)
> **Status:** Partial — Tasks T1-T3 complete, T4-T6 open
> **Depends on:** Plan 033 ✅ (Bomberman arena, HLPlayer proven +177)

## Problem

`HLPlayer` in the Bomberman arena is a single-agent bandit — each player maintains independent Q-values, visits, and absorb-compress state. With 4 players, the team runs 4 separate learning processes that never share experience. A team of 4 HLPlayers sharing one `BanditStats` would converge **4× faster** (4× more samples per round).

The pattern is proven: Plan 033's tournament shows HL (+177) > Greedy (+131) > Validator (-30) > Random (-55). Multi-agent HL would strengthen this further by eliminating redundant exploration.

## Current State

### HLPlayer (hand-rolled bandit in `src/pruners/bomber/players.rs`)

```
HLPlayer {
    q_values: [f32; 6],        // 6 BomberAction variants
    visits: [u32; 6],
    total_pulls: u32,
    compressed: [bool; 6],     // absorb-compress
}
```

- `select_action()`: heuristic score + ε-greedy (10%) over safe actions
- `update_outcome(survived, killed, powerups)`: shaped reward with exponential recency weighting
- `compress_cycle()`: absorb-compress (min_visits=20, threshold=0.1)
- Each `HLPlayer::new(id)` creates **independent** state — no shared reference

### Existing Reusable Infrastructure

| Component | Location | Reusability |
|-----------|----------|:-----------:|
| `BanditStats` | `src/pruners/bandit.rs` | High — pure data, wrap in `Arc<Mutex<_>>` |
| `TrialLog` | `src/pruners/trial_log.rs` | High — append-only, add `player_id` field |
| `AbsorbCompressLayer` | `src/pruners/absorb_compress.rs` | High — domain-independent |
| `ReviewMetrics` | `src/pruners/review_metrics.rs` | High — already `Arc`-based |

### Instantiation Pattern (examples)

```rust
let mut players: Vec<Box<dyn BomberPlayer>> = vec![
    Box::new(RandomPlayer::new(0)),
    Box::new(GreedyPlayer::new(1)),
    Box::new(ValidatorPlayer::new(2)),
    Box::new(HLPlayer::new(3)),  // independent bandit
];
```

## Tasks

- [x] **T1: Shared `BanditStats` abstraction** ✅
  - Created `SharedBanditStats` wrapping `Mutex<BanditStatsInner>` in `src/pruners/bandit.rs`
  - Methods: `update()`, `ucb1_score()`, `best_arm()`, `is_compressed()`, `compress_arm()`, `total_pulls()`, `visits()`, `q_value()`
  - Gated behind `#[cfg(feature = "bandit")]`
  - Optimistic init (Q=1.0) matching HLPlayer pattern
  - Unit test: `test_shared_bandit_stats_convergence` — 4 threads, 200 updates each, verifies convergence

- [x] **T2: Refactor `HLPlayer` to use shared stats** ✅
  - Added `HLPlayer::with_shared_stats(id, stats: Arc<SharedBanditStats>)` constructor (feature-gated)
  - Added accessor methods (`arm_compressed`, `arm_visits`, `arm_q`, `update_arm_q`, `mark_compressed`) with dual implementation
  - `select_action()` and `update_outcome()` use accessors that delegate to shared stats when present
  - `HLPlayer::new(id)` works exactly as before (shared_stats=None)

- [x] **T3: Shared absorb-compress** ✅
  - `SharedBanditStats` holds shared `compressed: Vec<bool>` inside mutex
  - `compress_arm()` and `is_compressed()` are shared — one agent's compression propagates to all
  - HLPlayer's `compress_cycle()` delegates to shared stats when present

- [x] **T4: `TrialLog` multi-writer support** ✅
  - Added `player_id: u32` to `TrialRecord` with `#[serde(default)]` for backward compat
  - Created `SharedTrialLog` wrapper (`Arc<Mutex<TrialLog>>`) behind `#[cfg(feature = "bandit")]`
  - `HLPlayer::with_shared_stats()` now accepts optional `SharedTrialLog`
  - `update_outcome()` writes `TrialRecord` with `player_id = self._id as u32` when shared log present
  - Re-exported `SharedTrialLog` from `pruners/mod.rs`
  - Test: `test_shared_trial_log_multi_writer` — 4 threads × 50 records, verifies 200 total + per-player counts
  - Test: `test_player_id_backward_compat` — legacy JSONL without `player_id` parses with default `0`

- [ ] **T5: Tournament benchmark**
  - Run 1000-game tournament: 4× shared HL vs 4× independent HL
  - Metrics: convergence speed (Q-value stability), win rate, survival rate
  - Record in `.benchmarks/` with sequential numbering

- [x] **T6: Generalize to `BanditPruner<P>`** ✅
  - Added `BanditPruner::with_shared_stats(inner, strategy, num_arms, stats: Arc<SharedBanditStats>)` constructor (feature-gated)
  - Added accessor methods (`arm_visits`, `arm_q`, `arm_total_pulls`, `arm_ucb1`, `arm_thompson`, `update_arm`, `arm_best`) with dual implementation
  - `relevance()`, `prepare_episode()`, `update()`, `best_arm()`, `total_pulls()` use accessors that delegate to shared stats when present
  - `BanditPruner::new()` works exactly as before (shared_stats=None)
  - Test: `test_bandit_pruner_shared_stats` — 2 pruners sharing one `SharedBanditStats`, verifies cross-pruner visibility

## Design Decisions

### Shared Q-values (not shared policy)

Share the **learning** (Q-table, visits) but keep **action selection** per-agent. Each agent still:
- Computes its own heuristic scores based on local game state
- Uses ε-greedy with its own RNG
- Has its own position/health/powerup context

The shared bandit provides the **priors** (which actions tend to be good), not the **policy** (what to do right now).

### Arc<Mutex> first, papaya later

Start with `Arc<Mutex<SharedBanditStats>>` for correctness. Profile before optimizing — bomber rounds are ~200 ticks with 1 update per round, so contention is minimal. If contention becomes measurable, swap to `papaya::HashMap`-based lock-free.

### Backward compatible

`HLPlayer::new(id)` keeps working as independent. `HLPlayer::with_shared_stats()` is additive. No breaking changes to existing examples.

## Scope

- **In scope:** Shared bandit learning for cooperative team play
- **Out of scope:** Opponent modeling (separate issue), per-phase bandits, negotiation/communication protocols

## References

- Plan 032: Heuristic Learning Infrastructure
- Plan 033: Bomberman Arena (HLPlayer implementation)
- `src/pruners/bandit.rs` — `BanditStats`, `BanditPruner<P>`
- `src/pruners/bomber/players.rs` — `HLPlayer` (lines 1103-1427)