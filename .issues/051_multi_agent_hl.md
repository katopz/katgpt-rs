# Issue 051: Multi-Agent Heuristic Learning — Shared Bandit Coordination

> **Plan:** 032 (Heuristic Learning Infrastructure)
> **Status:** Open
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

- [ ] **T1: Shared `BanditStats` abstraction**
  - Create `SharedBanditStats` wrapping `BanditStats` in `Arc<Mutex<_>>` (or `papaya` lock-free)
  - Methods: `update(arm, reward)`, `ucb1_score(arm)`, `thompson_sample(arm, rng)`, `best_arm()`
  - Gate behind `#[cfg(feature = "bandit")]`
  - Unit tests: concurrent updates from 4 threads, convergence correctness

- [ ] **T2: Refactor `HLPlayer` to use shared stats**
  - Add `HLPlayer::with_shared_stats(id, stats: Arc<SharedBanditStats>)` constructor
  - `select_action()` reads from shared stats
  - `update_outcome()` writes to shared stats
  - Keep `HLPlayer::new(id)` as independent (backward compat)
  - Integration test: 4 shared HLPlayers converge faster than 4 independent

- [ ] **T3: Shared absorb-compress**
  - When any shared agent compresses an arm, all agents respect the block
  - `SharedBanditStats` holds shared `compressed: Vec<bool>`
  - Test: one agent's compression propagates to all

- [ ] **T4: `TrialLog` multi-writer support**
  - Add `player_id: u32` to `TrialRecord`
  - All shared HLPlayers write to same JSONL with their ID
  - Enables post-hoc analysis of which agent contributed which experience

- [ ] **T5: Tournament benchmark**
  - Run 1000-game tournament: 4× shared HL vs 4× independent HL
  - Metrics: convergence speed (Q-value stability), win rate, survival rate
  - Record in `.benchmarks/` with sequential numbering

- [ ] **T6: Generalize to `BanditPruner<P>`**
  - After bomber proof, add `BanditPruner::with_shared_stats(stats: Arc<SharedBanditStats>)`
  - Enables multi-agent HL in any domain (not just bomber)
  - Document in `.docs/` as reusable pattern

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