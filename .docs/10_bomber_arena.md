# katgpt-rs: Bomberman HL Arena — 4-Player Heuristic Learning Proof

## Overview

A headless Bomberman arena using `bevy_ecs` standalone (not the full Bevy engine) for deterministic, tick-based simulation. AI players compete across 6 core and 6 research technology levels, proving that adaptive intelligence outperforms static rules.

The arena serves as the integration test bed for the HL thesis: **bandit-driven action selection + deterministic safety validation > pure heuristics or random baselines**.

## Architecture

### Tick Loop

All systems operate on `&mut World` directly — no ECS schedule, no real-time delta, no plugins.

```text
init_world(seed)
  ├─ ArenaGrid::generate(seed)          → 13×13 procedural grid
  ├─ GameRng, TickCounter, ScoreBoard   → resources
  └─ Events<GameEvent>                  → event bus

init_world_with_arena(arena)            → alternative: pre-built ArenaGrid (fixed template)

spawn_players(world)
  └─ 4 entities at corner spawns with Player, GridPos, BombCount, BombRange, Speed, Alive

run_tick(world, actions) → bool         // returns false when round ends
  ├─ tick_bomb_fuses()                  → countdown (skips Remote/Landmine), collect expired
  ├─ detonate_remote_bombs()            → player-triggered Detonate action
  ├─ process_explosions()               → blast propagation (cardinal, wall-blocking, piercing)
  ├─ apply_movement()                   → move players, wall/bomb collision
  ├─ trigger_landmines()                → proximity-based landmine detonation (after movement)
  ├─ place_bombs()                      → spawn bomb entities if action=Bomb
  ├─ collect_powerups()                 → walk over revealed power-ups
  └─ cleanup_and_check()                → kill players in blast, check round end
```

### Event Scoping

Events must be **tick-scoped** for AI decisions and **accumulated** only for end-of-round scoring. The examples use two separate buffers:

```text
tick_events = drain from ECS (this tick only) → passed to select_action()
round_events += tick_events.clone()           → used for final score calculation
```

Accumulating all events and passing them to `select_action` every tick causes `update_bombs()` to replay stale `BombExploded`/`BombPlaced` events, resetting bomb fuses in the AI's model and creating phantom blast zones.

### Grid Layout (13×13)

Standard Bomberman layout generated from seed:
- **Border walls** — fixed perimeter
- **Interior pillars** — fixed walls at even (x, y) intersections
- **Destructible walls** — ~40% fill with hidden power-ups (`BombUp`, `FireUp`, `SpeedUp`)
- **Spawn zones** — 3×3 corners kept clear at (1,1), (11,1), (1,11), (11,11)

## ECS Components & Resources

| Component | Purpose |
|-----------|--------|
| `Player { id }` | Player identity (0–3) |
| `GridPos { x, y }` | Position on grid |
| `Bomb { bomb_type: BombType }` | Bomb entity with type (Timed, Piercing, Remote, Landmine) |
| `BombFuse { owner, ticks_remaining }` | Fuse countdown; Remote/Landmine skip fuse tick |
| `BombRange { cells }` | Blast radius |
| `BombCount { max, active }` | Per-player bomb limit tracking |
| `Speed { cells_per_tick }` | Movement speed |
| `Alive` | Marker component (removed on death) |
| `DestructibleWall` | Destructible wall entity |
| `PowerUp { kind }` | Collectible power-up |
| `Blast` | Visual blast marker (1-tick lifetime) |

**BombType** enum (4 variants):

| Variant | Behavior |
|---------|----------|
| `Timed` (default) | Fuse-based, stops at walls |
| `Piercing` | Blast continues through destructible walls |
| `Remote` | Detonates on player `Detonate` action, ignores fuse |
| `Landmine` | Invisible until stepped on, 1-range instant blast |

| Resource | Purpose |
|----------|---------|
| `ArenaGrid` | 13×13 grid of `Cell` enum |
| `GameRng` | Deterministic seed |
| `TickCounter` | Current tick number |
| `ScoreBoard` | Per-player scores |
| `PlayerEntities` | 4 player `Entity` ids |

## Events

```rust
enum GameEvent {
    PlayerMoved { player, from, to },
    BombPlaced { player, pos },
    BombExploded { pos, range },
    PlayerKilled { victim, killer },
    PowerUpCollected { player, kind, pos },
    PowerUpRevealed { pos, kind },
    WallDestroyed { pos },
    RoundEnd { survivors },
}
```

## Player Types (6 Core + 6 Research Players)

### P1 🐰 RandomPlayer — Baseline

- **Tech:** None. Random selection from all 7 actions with equal probability.
- **Safety:** Only avoids walking into walls (up to 3 re-rolls, then Wait). No blast zone avoidance.
- **No learning, no memory, no model.** Pure baseline for comparison.

### P2 🐱 GreedyPlayer — Heuristic

- **Tech:** Heuristic scoring of all 7 actions (Up/Down/Left/Right/Bomb/Wait/Detonate).
- **Selection:** Scores by proximity to power-ups (+3.0 step on, +2.0 toward), wall density (+0.3 per wall in range 3), adjacent wall bonus (+1.0), center bias (+0.2).
- **Safety:** Penalizes blast zones. 20% ε-greedy safe exploration.
- **No opponent tracking, no safety validation.**

### P3 🐶 ValidatorPlayer — Heuristic + Safety Rules

- **Tech:** Same heuristic as P2, plus hard safety validation.
- **Validation rules:**
  - Hard-blocks walking into blast zones (wall-aware blast calculation)
  - Hard-blocks placing bomb with no escape route (BFS checks reachable safe cell)
  - Escape mode when in danger zone (scored by `escape_distance`)
  - Safe mode when clear (full heuristic + safety filter)
- **Tracks:** Known bombs with fuse countdown, revealed power-ups.
- **Limitation:** Static rules prevent suicides but also prevent kills. Too conservative.

### P4 🐵 HLPlayer — Full HL (Heuristic + Attack Tactics + Bandit)

- **Tech:** P3 base + opponent tracking + attack tactics + bandit Q-values + absorb-compress.
- **Tracks:** Known bombs, revealed power-ups, opponent positions with trajectory history.
- **Persists across rounds:** Q-values, visits, compressed arms (bandit memory).

#### FSM Decision Priority (per tick)

| Priority | State | Trigger | Action |
|----------|-------|---------|--------|
| 1 | **Evade** | `in_blast_zone(pos)` is true | BFS `escape_distance()` to find safe tile, score movement toward safety (+10.0) |
| 2 | **Wait** | Safe tile, no goals nearby | `BomberAction::Wait` (-1.0 score, hard-blocked if in blast zone) |
| 3 | **Collect** | Revealed power-up visible | Move toward nearest power-up (+3.0 step on, +2.0 toward) |
| 4 | **Attack** | Opponent within range | Intercept predicted path, trap scoring, bomb placement |
| 5 | **Explore** | No threats, no loot, no enemies | Move toward wall-dense areas, center bias, bomb adjacent walls |

#### Attack Tactics

HLPlayer implements four attack functions:

| Function | Purpose | Bonus |
|----------|---------|-------|
| `predict_direction(current, prev)` | Extrapolates opponent heading from position history | feeds into intercept |
| `intercept_score(target, opponent, predicted)` | Move toward opponent's predicted next position | +1.0 toward predicted |
| `count_escape_routes(pos, grid)` | Count walkable neighbors (fewer = better trap) | feeds into trap + chokepoint |
| `trap_score(bomb_pos, opponent, grid, range)` | Score bomb by how trapped opponent would be | +4.0 blast hit, +3.0 dead-end, +2.0 corridor, +1.0 close |
| chokepoint (inline) | Prefer moving where opponent has ≤1 escape route | +1.0 |

#### Opponent Tracking

```rust
type KnownOpponent = (u8, (i32, i32), Option<(i32, i32)>);
//                    id   current_pos   prev_pos (for trajectory)
```

`update_opponents()` stores previous position on each `PlayerMoved` event, enabling `predict_direction()` to extrapolate the opponent's heading.

#### Bandit Layer

- **Blended scoring:** heuristic + strategy bonus (bandit Q-values currently disabled — too sparse at this scale)
- **ε-greedy:** 10% explore, 90% exploit (safe moves only, filtered by blast zone)
- **Absorb-Compress:** Every 100 rounds, arms with `visits ≥ 20 && Q < 0.1` get hard-blocked
- **Reward shaping:** `+1.0 survive, -1.0 die, +0.5 kill, +0.2/powerup`
- **Shared stats:** `with_shared_stats()` enables multi-agent cooperative learning via `SharedBanditStats` + `SharedTrialLog` (feature: `bandit`)
- **Freeze/Thaw:** `freeze()` → `BomberFrozenBandit` for disk persistence; `thaw()` restores Q-values/visits/compressed across sessions (Plan 092)

### P5 🤖 GZeroPlayer — G-Zero Self-Play (Plan 052)

- **Tech:** Weak heuristic + template hints + Hint-δ + DeltaBanditPruner + DeltaGatedAbsorbCompress + HL safety filter.
- **Feature gate:** `--features g_zero` (implies `bandit`, `bomber`)
- **Tracks:** Known bombs, powerups, opponents + template distribution + δ history + bandit Q-values.

#### Decision Flow (per tick)

| Step | Component | What it does |
|------|-----------|-------------|
| 1 | Weak heuristic | Simple walkability + mild powerup/bomb/center scoring (no BFS escape) |
| 2 | `BomberTemplateProposer` | UCB1 selects from 8 strategy templates (FleeBlast, ChaseNearest, BombWall, etc.) |
| 3 | `hint_score_override` | Template adds ±1-3 to each action score |
| 4 | `HintDelta::compute` | δ = how much template shifted scores vs baseline |
| 5 | Safety filter | When in blast zone → override with `score_action` BFS escape; block moves into blast zones |
| 6 | ε-greedy | 15% random safe exploration |

#### Templates (8 strategy archetypes)

| Template | Hint Effect | When useful |
|----------|------------|-------------|
| FleeBlast | +3.0 away from bombs, -2.0 Bomb/Wait | Bomb-heavy situations |
| ChaseNearest | +2.0 toward closest opponent | Aggressive hunting |
| BombWall | +2.0 near destructible walls | Wall clearing |
| CampCorner | +1.5 corner positions with escape | Defensive play |
| PowerUpHunt | +2.5 toward powerups | Resource collection |
| CutoffOpponent | +2.0 blocking opponent escape | Trapping |
| CenterControl | +1.0 center grid positions | Map control |
| WaitTrap | +1.0 Wait near opponents | Baiting opponents |

#### Key Insight: Safety Filter Dominance

The safety filter (step 5) overrides G-Zero's template decisions in all blast-zone situations. The G-Zero components (Hint-δ, bandit, absorb-compress) influence only **safe, normal moves** — they contribute ±1-3 points on top of the weak baseline. The 64.1% survival rate comes primarily from the HL safety filter + 15% random exploration, not from template intelligence.

### P6 🦊 TftPlayer — Tit-for-Tat (Issue 056)

Game theory's Tit-for-Tat applied to bomberman. 2-state FSM with wall-aware provocation detection.

- **Strategy:** Nice by default (score-based like Greedy). Retaliates when in blast zone + opponent nearby.
- **Retaliation:** Hunt bonus (+1.5), intercept (+1.0), chokepoint (+1.0) — targets nearest opponent.
- **Forgiving:** 10-tick auto-reset timer + 10% generous forgiveness chance.
- **Safety:** Always flees when in blast zone, even in Retaliatory mode.
- **Feature gate:** `--features g_zero` (same as GZero).
- **Mixed tournament result:** 58.4% survival, 3.1 avg score, 0.32 kills/rnd (highest kills, 2nd best score).

### P7 🏷️ RubricPlayer — ROPD Rubric-Vector Bomber

Rubric-pattern bandit player using multi-criteria rubric vectors instead of scalar rewards.

- **Tech:** GZero base + `RubricBanditPruner` + `RubricGatedAbsorbCompress`.
- **Bandit:** Multi-criteria rubric vectors (survival, kills, powerups, aggression) with reference gap scoring.
- **Absorb:** Rubric-gated compression promotes arms when weighted rubric score approaches reference.
- **Feature gate:** `--features ropd_rubric`.

### P8 📊 SdarPlayer — SDAR Sigmoid-Gated Bomber

SDAR (Sigmoid-gated Delta-Aware Reward) bandit player with smooth gating.

- **Tech:** GZero base + `SdarBanditPruner` + `SdarGatedAbsorbCompress`.
- **Bandit:** Sigmoid-gated δ-reward filtering — smoothly transitions between exploring and exploiting based on delta magnitude.
- **Absorb:** SDAR-gated compression with smooth promotion threshold.
- **Feature gate:** `--features sdar_gate`.

### P9 🎛️ Sr2amPlayer — SR²AM Configurator Bomber

SR²AM configurator-driven player with FeedbackBandit for planning decisions.

- **Tech:** GZero base + `DeltaBanditPruner` + `DeltaGatedAbsorbCompress` + planning + FeedbackBandit.
- **Planning:** Decides between plan_new, plan_extend, plan_skip per tick with FeedbackBandit.
- **Freeze/Thaw:** Full `freeze()` → `BomberFrozenBandit` / `thaw()` cycle for cross-session persistence.
- **Feature gate:** `--features sr2am_configurator`.

### P10 🧪 VpdPlayer — VPD-EM Distillation Bomber

Variational Posterior Distillation player using EM-cycle bandit updates.

- **Tech:** GZero base + `SdarGatedAbsorbCompress` + `VpdEmCycle`.
- **Bandit:** E-step/M-step cycle — periodic E-step reclusters templates, M-step updates per-template rewards from outcome EM.
- **Configurable:** `VpdConfig` for E-step frequency and reward shaping.
- **Feature gate:** `--features vpd_em_distill`.

### P11 📐 RmsdPlayer — RMSD Relevance Distillation Bomber (❌ NO GOAT)

RMSD (Root Mean Square Deviation) filtering player that concentrates updates on most informative actions. **Negative arena result** — no improvement over SDAR in tournament play. Kept for infrastructure reuse.

- **Tech:** GZero base + `SdarBanditPruner` + `SdarGatedAbsorbCompress` + RMSD delta filtering.
- **Bandit:** SDAR sigmoid-gated with RMSD-based delta concentration — filters high-variance updates for faster convergence.
- **Feature gate:** `--features rmsd_distill`.

### NN Players (in `players.rs`)

Three additional players for neural network-based action selection:

- **LoraPlayer** — LoRA adapter on top of heuristic base, applies trained LoRA weights to action scoring.
- **LoraWasmPlayer** — LoRA + WASM batch validation, combines learned weights with WASM pruner.
- **NNPlayer** — Pure WASM-based neural network player, delegates all scoring to WASM module.

## Shared AI Functions (`players.rs`)

These utility functions are used by multiple player types:

| Function | Purpose | Used By |
|----------|---------|---------|
| `in_blast_zone(pos, grid, bombs)` | Check if position is in any bomb's blast (wall-blocking) | All |
| `is_in_single_blast(pos, grid, bomb_pos, range)` | Single bomb blast check with wall blocking | All |
| `escape_distance(pos, grid, bombs, blocked)` | BFS distance to nearest safe cell | Greedy, Validator, HL |
| `has_escape_route(grid, pos, new_bomb, range, bombs)` | Can player flee after placing bomb? | Validator, HL |
| `is_safe_action(action, grid, pos, bombs)` | Is action safe given bomb state? | Validator, HL |
| `should_place_bomb(grid, pos, bombs)` | Has adjacent wall + escape route? | Greedy, Validator, HL |
| `score_action(action, grid, pos, bombs, powerups, last_dir)` | Base heuristic scoring | Greedy, Validator, HL |

## Key Files

| File | Lines | Purpose |
|------|-------|--------|
| `src/pruners/bomber/mod.rs` | 498 | Module index: enums, components, resources, events, constants, `BombType`, `BomberFrozenBandit` |
| `src/pruners/bomber/arena.rs` | 457 | Procedural 13×13 grid generation with `ArenaGrid::generate(seed)`, fixed templates |
| `src/pruners/bomber/systems.rs` | 1389 | World-based ECS systems: `init_world`, `spawn_players`, `run_tick` (with remote/landmine/piercing) |
| `src/pruners/bomber/players.rs` | 2388 | `BomberPlayer` trait + 7 implementations (Random, Greedy, Validator, HL, LoraPlayer, LoraWasmPlayer, NNPlayer) + shared AI functions |
| `src/pruners/bomber/arena_runner.rs` | 482 | `BomberArenaConfig`, `BomberRoundResult`, `run_bomber_game`, `run_bomber_matchup` |
| `src/pruners/bomber/replay.rs` | 598 | `ReplaySample`, `ReplayWriter`, board/bomb/powerup serialization (Plan 039) |
| `src/pruners/bomber/replay_backward.rs` | 386 | `BackwardSample`, `ReplayBackwardWalker` — GFlowNet-inspired backward policy extraction |
| `src/pruners/bomber/g_zero_player.rs` | 826 | `GZeroPlayer` — G-Zero self-play with template hints + Hint-δ (Plan 052) |
| `src/pruners/bomber/tft_player.rs` | 657 | `TftPlayer` — game theory Tit-for-Tat bomber (Issue 056) |
| `src/pruners/bomber/rubric_player.rs` | 824 | `RubricPlayer` — ROPD rubric-pattern bomber (feature: `ropd_rubric`) |
| `src/pruners/bomber/sdar_player.rs` | 788 | `SdarPlayer` — SDAR-gated bomber (feature: `sdar_gate`) |
| `src/pruners/bomber/sr2am_player.rs` | 1014 | `Sr2amPlayer` — SR2AM configurator bomber (feature: `sr2am_configurator`) |
| `src/pruners/bomber/vpd_player.rs` | 872 | `VpdPlayer` — VPD-EM distillation bomber (feature: `vpd_em_distill`) |
| `src/pruners/bomber/rmsd_player.rs` | 811 | `RmsdPlayer` — RMSD relevance distillation bomber (feature: `rmsd_distill`) |
| `src/pruners/bomber/validator_agent.rs` | 1144 | `AgentLoop`, `ValidatorCandidate`, `RulePlayer` — auto-eval validator search (feature: `bomber-agent`) |
| `src/pruners/bomber/event_log_player.rs` | 320 | `BomberEventLog`, `BomberForkDiff` — event-log-backed replay player (feature: `event_log`) |
| `src/pruners/bomber/wasm_pruner.rs` | 950 | `BomberWasmPruner` — WASM-based batch validation with `BatchResult` (feature: `bomber-wasm`) |
| `src/pruners/bomber/wasm_state.rs` | 697 | `serialize_into_buffer`, `ZeroCopyStateBuffer` — efficient ECS→WASM state transfer (feature: `bomber-wasm`) |
| `src/pruners/g_zero/bomber_templates.rs` | — | `BomberTemplate` + `BomberTemplateProposer` — 8 strategy archetypes |
| `examples/bomber_01_arena.rs` | 443 | Headless 100-round tournament runner + `--replay-dir` dump |
| `examples/bomber_02_tui.rs` | 576 | Animated ratatui TUI replay with emoji rendering |
| `examples/bomber_03_hl_proof.rs` | 632 | 1000-round HL proof with golden traces + `--replay-dir` filtered dump |
| `examples/bomber_04_nn.rs` | 430 | NN player (LoRA/WASM) tournament runner |
| `examples/bomber_05_replay_gen.rs` | 357 | Dedicated replay generator for training data (Plan 039) |
| `examples/bomber_06_replay_gen_v2.rs` | 499 | V2 replay generator with backward walker |
| `examples/bomber_07_bomb_types.rs` | 197 | Bomb types showcase (Piercing, Remote, Landmine) |
| `examples/bomber_08_agent_loop.rs` | 83 | Agent loop runner (feature: `bomber-agent`) |
| `examples/bomber_09_rubric_tournament.rs` | 353 | Rubric player tournament |
| `examples/bomber_10_sdar_tournament.rs` | 370 | SDAR player tournament |
| `examples/bomber_11_bt_rank_tournament.rs` | 11 | Bradley-Terry ranking tournament |
| `examples/bomber_12_self_play_freeze.rs` | 475 | Self-play with freeze/thaw cycle |
| `examples/bomber_13_reflection_qa.rs` | 76 | Memo reflection QA runner |
| `examples/bomber_14_sr2am_tournament.rs` | 373 | SR2AM player tournament |
| `examples/bomber_15_vpd_tournament.rs` | 389 | VPD player tournament |
| `examples/bomber_16_rmsd_tournament.rs` | 394 | RMSD player tournament |
| `tests/bench_bomber_arena.rs` | 744 | Benchmark tests |

## Results

### 100-Round Arena (seed=42)

```text
#1 🐱 Greedy     Score= +171  Wins=5   Deaths=41
#2 🐵 HL         Score= +146  Wins=13  Deaths=43
#3 🐶 Validator  Score=  -23  Wins=1   Deaths=61
#4 🐰 Random     Score=  -43  Wins=12  Deaths=38
```

### 1000-Round Proof (seed=42)

```text
#1 🐵 HL         Survival=7.8%  Score=-0.1  Kills=0.03/rnd
#2 🐰 Random     Survival=4.7%  Score=-0.5  Kills=0.00/rnd
#3 🐱 Greedy     Survival=3.9%  Score=+2.6  Kills=0.39/rnd
#4 🐶 Validator  Survival=0.7%  Score=-0.2  Kills=0.25/rnd
```

**Key Proof:** P4 (HL) survival 7.8% vs P3 (Validator) 0.7% = **+7.1pp** (✅ proven, threshold 5pp).

### Player A/B Benchmark — Isolated Performance + Latency (Plan 054)

Each player type runs as all 4 slots (same type) for 1000 rounds. Measures survival, score, kills, and per-action latency in release mode.

```
Config        │ Survival │ Avg Score │ Avg Kills │ P50 (μs) │ P95 (μs) │ P99 (μs) │ Wins
🐱 Greedy     │   72.1%  │       3.1 │      0.12 │      1.3 │      2.1 │      5.0 │  108
🐶 Validator  │   58.6%  │       1.7 │      0.05 │      1.2 │      1.6 │      2.0 │  292
🐵 HL         │   57.0%  │       2.1 │      0.24 │      0.6 │      1.2 │      2.1 │  222
🤖 GZero      │   64.1%  │       1.8 │      0.06 │      0.5 │      1.0 │      1.5 │  167
```

#### Key Findings

1. **Greedy wins survival** (72.1%) — simplest heuristic is most robust in mirror matches.
2. **GZero > HL** (+7.1pp survival, 0.73× latency) — better at surviving AND faster.
3. **GZero is fastest** (0.5μs P50, 94.6% sub-microsecond) — optimizer inlines template+δ well.
4. **HL most aggressive** (0.24 kills/round) but kills ≠ survival.
5. **Validator worst** (58.6%) — over-conservative safety filter loses positioning.

#### Production Recommendation

| Use Case | Player | Why |
|----------|--------|-----|
| Real-time MMO (lowest latency) | GZero | 0.5μs P50, 94.6% sub-μs, 64.1% survival |
| Survival-focused | Greedy | 72.1% survival, robust heuristic |
| Aggressive/hunt | HL | 0.24 kills/round (but lower survival) |
| Avoid | Validator | Worst survival, no latency advantage |

#### Run the Benchmark

```bash
cargo run -p riir-examples --example g_zero_04_player_ab_benchmark --features g_zero --release
```

### Observations

1. **HL wins most rounds (13/100)** — attack tactics + survival balance makes it the deadliest player.
2. **Greedy has highest score (+171)** — farms power-ups aggressively (3.2/round) but dies more.
3. **Validator is too conservative** — static safety rules prevent suicides but also prevent kills and trap the player in corners.
4. **Random wins via survival** — doesn't hunt or bomb, avoids dangerous situations, outlives aggressive players in chaotic rounds.
5. **Score ≠ Survival** — Greedy optimizes score (power-ups), HL optimizes survival (wins), Random gets lucky.
6. **GZero beats HL in isolation** — weaker baseline + template hints create more robust policy than HL's strategy bonus, AND faster latency.

### TFT Mixed Tournament (Issue 056)

1000-round mixed tournament, release build:

```
Player    │ Survival │ Avg Score │ Avg Kills │ Game Theory Analog
🐱 Greedy │   64.5%  │      3.6  │     0.26  │ Pure Cooperator
🐵 HL     │   60.6%  │      1.7  │     0.00  │ Grim Trigger
🤖 GZero  │   70.5%  │      1.9  │     0.04  │ Noisy Cooperator
🦊 TFT    │   58.4%  │      3.1  │     0.32  │ Tit-for-Tat
```

**Key findings:**
1. **TFT highest kills** (0.32/rnd) — retaliatory mode punishes aggressors.
2. **TFT 2nd best score** (3.1) — Nice mode collects powerups like Greedy.
3. **TFT survival below target** (58.4% < 68% hypothesis) — retaliation bonus pulls toward danger.
4. **FFT is better TFT domain** — `GameEvent::DamageDealt` is crystal-clear signal vs bomber's ambiguous proximity.

Run: `cargo run -p riir-examples --example g_zero_05_tft_mixed --features g_zero --release`

## How to Run

```bash
# Headless 100-round tournament
cargo run --example bomber_01_arena --features bomber

# Animated TUI replay (keyboard controls: ←/→/Space/Q)
cargo run --example bomber_02_tui --features bomber

# 1000-round HL proof experiment with stats
cargo run --example bomber_03_hl_proof --features bomber

# Benchmarks
cargo test --features bomber bench_bomber_arena -- --nocapture

# Tests
cargo test --features bomber
```

## Replay Training Data Pipeline (Plan 039)

The arena can dump tick-level training data as JSONL for downstream LoRA training in `riir-gpu`.

### Data Flow

```text
bomber_05_replay_gen (1000 rounds)
  │  At each tick, for P3/P4 alive players:
  │    serialize(board_state, action_taken, player_type)
  │
  ▼
output/replays/bomber_replay_{timestamp}.jsonl
  │  Filter: quality > 0.5 (survived/won only)
  │
  ▼
riir-gpu/examples/train_bomber.rs
  │  Loads JSONL → GameSample → wgpu LoRA training
  │
  ▼
output/game_lora.bin → NNPlayer (trained policy adapter)
```

### ReplaySample Format (JSONL)

Each line is one `(board_state, action, quality)` sample:

```json
{
  "board": [0,0,1,2,0,...],
  "player_pos": [3, 5],
  "player_id": 3,
  "bombs": [[3,5,3,8],[7,2,3,4]],
  "powerups": [[10,10],[5,8]],
  "action": 4,
  "quality": 0.85,
  "tick": 42,
  "round": 7,
  "player_type": "HL"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `board` | `Vec<u8>` | 13×13 grid flattened. Floor=0, FixedWall=1, DestructibleWall=2, PowerUpHidden=3 |
| `player_pos` | `[u8; 2]` | Player position (x, y) |
| `player_id` | `u8` | Player index (0-3) |
| `bombs` | `Vec<[u8; 4]>` | Active bombs: (x, y, blast_range, fuse_ticks) |
| `powerups` | `Vec<[u8; 2]>` | Active powerup positions: (x, y) |
| `action` | `u8` | 0=Up, 1=Down, 2=Left, 3=Right, 4=Bomb, 5=Wait, 6=Detonate |
| `quality` | `f32` | 0.0 (death) → 0.5 (survived) → 1.0 (winner) + bonuses |
| `player_type` | `String` | "Random", "Greedy", "Validator", "HL" |

### Quality Scoring

| Outcome | Base | Bonus |
|---------|------|-------|
| Death | 0.0 | — |
| Survived | 0.5 | +0.05/powerup (cap +0.2), +0.1/kill (cap +0.3) |
| Winner | 1.0 | +0.05/powerup (cap +0.2), +0.1/kill (cap +0.3) |

### Key Files

| File | Purpose |
|------|--------|
| `src/pruners/bomber/replay.rs` | `ReplaySample`, `ReplayWriter`, board/bomb/powerup serialization |
| `examples/bomber_05_replay_gen.rs` | Dedicated replay generator (1000 rounds, filtered P3/P4) |
| `examples/bomber_06_replay_gen_v2.rs` | V2 replay generator with backward walker |
| `examples/bomber_01_arena.rs` | `--replay-dir` flag for optional replay dump |
| `examples/bomber_03_hl_proof.rs` | `--replay-dir` flag with P3/P4 quality filtering |

### Commands

```bash
# Generate replay data (1000 rounds, filtered P3/P4 winning episodes)
cargo run --example bomber_05_replay_gen --features bomber

# Generate with custom output dir
cargo run --example bomber_05_replay_gen --features bomber -- output/my_replays

# V2 replay generator with backward walker
cargo run --example bomber_06_replay_gen_v2 --features bomber

# Arena with optional replay dump (all players, all samples)
cargo run --example bomber_01_arena --features bomber -- --replay-dir output/replays

# HL proof with filtered replay dump
cargo run --example bomber_03_hl_proof --features bomber -- --replay-dir output/replays
```

## Design Lessons

1. **Event scoping matters** — accumulating events across ticks poisons AI state; tick-scoped events for decisions, accumulated only for scoring.
2. **ConstraintPruner is domain-agnostic** — same `is_safe_action` pattern serves both Bomberman blast zones and Sudoku rule validation.
3. **Wall-aware blast calculation is essential** — naive range checks without wall blocking create phantom danger zones.
4. **Trajectory prediction > reactive tracking** — extrapolating opponent heading from position history enables interception.
5. **Static safety can be counterproductive** — Validator's hard blocks prevent all suicides but also prevent strategic risk-taking that wins games.
6. **Attack tactics are additive** — hunt, intercept, chokepoint, and trap scoring compose cleanly on top of the base heuristic.