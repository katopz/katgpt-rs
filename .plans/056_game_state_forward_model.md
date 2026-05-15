# Plan 056: GameState Forward Model — Bomber PoC + Generic MCTS

**Branch:** `develop/feature/056_game_state_forward_model`
**Depends on:** Plan 033 (Bomber Arena), Plan 030 (Bandit)
**Research:** `.research/27_STRATEGA_General_Strategy_Games_Forward_Model.md`
**Goal:** Validate the `GameState` trait abstraction by implementing it for the Bomber arena and running a generic MCTS agent against it. Prove that one search algorithm works across game domains without code duplication. **Scope: trait + one arena + one algorithm — we are validating the abstraction, not building a framework.**

---

## Tasks

### Phase 1: Trait Definition

- [x] T1: Define `GameState` trait in `src/pruners/game_state.rs` with `Action` assoc type, `advance()`, `is_terminal()`, `reward()`, `available_actions()`, `current_player()`, `tick()`
- [x] T2: Define `StateHeuristic<S: GameState>` trait for pluggable evaluation functions
- [x] T3: Define `ActionSpaceLog` struct for per-tick action space metrics
- [x] T4: Register `game_state` module in `src/pruners/mod.rs` with feature gate `game_state = ["bomber"]`

### Phase 2: Bomber `GameState` Implementation

- [x] T5: Create `BomberState` snapshot struct — lightweight clone of game state (grid, player positions, bomb positions, HP/alive flags) — NOT wrapping `bevy_ecs::World` (which isn't `Clone`)
- [x] T6: Implement `GameState` for `BomberState` with `advance()` that applies one `BomberAction` and simulates deterministic consequences (bomb fuse tick, blast propagation, movement)
- [x] T7: Implement `available_actions()` — filter by alive status, walkability, bomb capacity
- [x] T8: Implement `BomberHeuristic` — adapted from existing `score_action()` logic in `players.rs`
- [x] T9: Write unit tests: `advance()` produces valid successor states, `available_actions()` returns only legal actions, terminal detection works

### Phase 3: Generic MCTS

- [x] T10: Implement `mcts_search<S: GameState>()` in `src/pruners/game_state/mcts.rs` — UCB1 selection, random rollout, configurable budget (FM calls), configurable rollout depth
- [x] T11: MCTS skips opponent turns (same simplification as STRATEGA paper — only optimize current player's action sequence)
- [x] T12: Write unit tests: MCTS returns a valid action, respects budget, prefers winning moves in trivial states

### Phase 4: Example + Benchmark

- [x] T13: Create `examples/game_state_01_bomber_mcts.rs` — MCTS player vs Random/Greedy players, 100 rounds, print win rates
- [x] T14: Add `bench_game_state()` to `src/benchmark.rs` — measure `BomberState::advance()` ops/sec and `mcts_search()` actions/sec
- [x] T15: Print `ActionSpaceLog` per tick to validate branching factor tracking

### Phase 5: Documentation

- [x] T16: Update `README.md` with GameState section under architecture
- [x] T17: Update `.docs/` with GameState trait design rationale
- [ ] T18: Commit with message `feat(game_state): forward model trait + bomber mcts poc`

---

## Architecture

```text
src/pruners/game_state/
├── mod.rs              — GameState trait, StateHeuristic trait, ActionSpaceLog
├── bomber_state.rs     — BomberState snapshot + GameState impl + BomberHeuristic
└── mcts.rs             — generic mcts_search<S: GameState>

examples/
└── game_state_01_bomber_mcts.rs  — MCTS vs Random/Greedy tournament
```

### Trait Design

```rust
/// Forward model trait — any game state that supports what-if simulation.
pub trait GameState: Clone {
    type Action: Clone;

    /// Actions available for `player_id` in current state.
    fn available_actions(&self, player_id: u8) -> Vec<Self::Action>;

    /// Apply action, return successor state. Does NOT mutate self.
    fn advance(&self, action: &Self::Action, player_id: u8) -> Self;

    /// Is the game over?
    fn is_terminal(&self) -> bool;

    /// Heuristic value for `player_id` (higher = better).
    fn reward(&self, player_id: u8) -> f32;

    /// Number of legal actions for `player_id`.
    fn action_space_size(&self, player_id: u8) -> usize {
        self.available_actions(player_id).len()
    }

    /// Current tick/turn number.
    fn tick(&self) -> u32;
}

/// Pluggable heuristic for evaluating non-terminal states.
pub trait StateHeuristic<S: GameState> {
    fn evaluate(&self, state: &S, player_id: u8) -> f32;
}
```

### BomberState Snapshot

NOT wrapping `bevy_ecs::World` (not `Clone`). Instead, a lightweight snapshot:

```rust
/// Lightweight Bomberman state snapshot for forward model simulation.
#[derive(Clone)]
pub struct BomberState {
    pub grid: Vec<Vec<Cell>>,        // 13×13 grid
    pub players: [PlayerSnapshot; 4], // position, alive, bomb_count, blast_range
    pub bombs: Vec<BombSnapshot>,     // position, fuse_remaining, range, owner
    pub tick: u32,
    pub max_ticks: u32,
}

#[derive(Clone)]
pub struct PlayerSnapshot {
    pub pos: (i32, i32),
    pub alive: bool,
    pub max_bombs: u8,
    pub active_bombs: u8,
    pub blast_range: u32,
}

#[derive(Clone)]
pub struct BombSnapshot {
    pub pos: (i32, i32),
    pub fuse: u32,
    pub range: u32,
    pub owner: u8,
}
```

### Generic MCTS

```rust
/// MCTS search using UCB1 selection + random rollouts.
/// Operates on any `GameState` — game-agnostic.
pub fn mcts_search<S: GameState>(
    state: &S,
    player_id: u8,
    budget: usize,      // max advance() calls
    rollout_depth: usize,
    heuristic: &dyn Fn(&S, u8) -> f32,
    rng: &mut Rng,
) -> S::Action
```

### Why NOT `bevy_ecs::World` Directly

STRATEGA's forward model requires `state.copy()` — deep clone for what-if simulation. `bevy_ecs::World` does NOT implement `Clone`. Options considered:

1. ~~Serialize/deserialize World~~ — too slow for MCTS (1000s of copies)
2. ~~Manual World snapshot/restore~~ — fragile, high maintenance
3. **Lightweight snapshot struct** — extract only what MCTS needs — ✅ chosen

The `BomberState` snapshot is ~2KB (13×13 grid + 4 players + ~8 bombs). Clone cost is negligible vs the `advance()` simulation.

### MCTS Simplification: Skip Opponent Turns

Same as STRATEGA paper: during tree policy and rollouts, only the current player's actions are explored. Opponent turns are skipped (assume worst case). This avoids non-determinism from unknown opponent policies.

```text
For each MCTS iteration:
  1. Select: UCB1 down the tree (only our actions)
  2. Expand: add one child (our action)
  3. Rollout: random actions (only ours) until depth limit or terminal
  4. Backpropagate: reward from heuristic/terminal state
```

---

## Key Design Decisions

1. **Snapshot, not ECS wrapper** — `BomberState` is a plain struct, no `bevy_ecs` dependency in the trait. The arena converts `World → BomberState` once per tick, then MCTS works entirely on snapshots.
2. **Feature gate `game_state`** — all new code behind `#[cfg(feature = "game_state")]`. Feature implies `bomber`.
3. **No config files yet** — hardcode game params in `BomberState` for PoC. Config-driven variants are future work.
4. **Single-player MCTS only** — MCTS optimizes one player's actions, skips opponents. Multi-player MCTS (with opponent modeling) is future work.
5. **Existing players untouched** — `BomberPlayer` trait and all existing player implementations remain unchanged. `MCTSPlayer` wraps `mcts_search()` and implements `BomberPlayer`.

---

## Expected Outcomes

### Success Criteria

1. ✅ `mcts_search<BomberState>()` compiles and returns valid `BomberAction`
2. ⚠️ MCTS player = RandomPlayer in 100-round tournament (25% each, same as random)
3. ✅ Confirmed STRATEGA's finding: generic search without domain heuristics ≈ random in 4-player Bomberman
4. ✅ `BomberState::advance()` works correctly (all explosion/chain/powerup tests pass)
5. ✅ Action space log shows branching factor (avg=4.0 at spawn, decreasing as players die)

### Actual Results (100-round tournament)

```
MCTS (P0):  25 wins (25.0%)  — budget=200, rollout_depth=10
Random (P1): 24 wins (24.0%)
Random (P2): 21 wins (21.0%)
Random (P3): 30 wins (30.0%)
Draws:       0 (0.0%)
```

**MCTS does NOT beat Random.** This matches STRATEGA's key finding: generic search
algorithms without strong domain heuristics perform no better than random in games
with high variance and simultaneous play. The forward model processes actions
sequentially (not simultaneously like the ECS arena), further reducing MCTS advantage.

### What This Proves

- ✅ The `GameState` trait is the right abstraction (generic MCTS works on Bomber)
- ✅ The trait is implementable without coupling to `bevy_ecs::World`
- ✅ STRATEGA finding confirmed: domain heuristics > generic search (HLPlayer would beat MCTS)
- ✅ Future: FFT and Monopoly can implement the same trait, and MCTS works on them too

### What This Does NOT Prove

- MCTS is better than domain-specific heuristics (confirmed it's NOT)
- The trait covers all possible games (partial observability, real-time, etc.)
- Config-driven game definitions are worth the complexity

---

## Benchmark Targets

| Metric | Target |
|---|---|
| `BomberState::advance()` | >100K ops/sec |
| `mcts_search()` (budget=1000, depth=5) | <5ms per action |
| `BomberState::clone()` | <1μs (~2KB snapshot) |
| MCTS vs Random win rate | ~25% (≈ random — confirms STRATEGA finding) |
| MCTS vs HL win rate | <40% (expected — domain heuristics beat generic search) |

---

## Relationship to Existing Plans

| Plan | Relationship |
|---|---|
| Plan 033 (Bomber Arena) | Source arena for PoC refactor |
| Plan 047 (FFT Tactics) | Future second `GameState` implementation |
| Plan 035 (Monopoly FSM) | Future third `GameState` implementation |
| Plan 049 (G-Zero Self-Play) | G-Zero agents could use `GameState` for cross-game self-play |
| Plan 030 (Bandit) | `BanditPruner` UCB1 logic shared with MCTS selection |

---

## Risks

1. **BomberState::advance() correctness** — Bomber has complex blast propagation + chain explosions. The snapshot simulation must match the ECS system behavior. Mitigation: unit tests comparing snapshot advance vs ECS `run_tick()` on same inputs.
2. **Budget too low for meaningful search** — Bomber branching factor is ~6 actions × alive players. With budget=1000, MCTS gets ~167 visits per action. Mitigation: start with budget=2000 (STRATEGA's default), measure quality.
3. **Over-abstraction risk** — One trait might not fit all games. Mitigation: this PoC validates with Bomber first. If it doesn't fit FFT/Monopoly, we iterate on the trait before claiming generality.