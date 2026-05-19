# microgpt-rs: Go Arena — AI vs AI Auto-Play Engine (Plan 065)

## Overview

A full Go (Baduk/Weiqi) game engine in pure Rust with 6 AI player strategies, Tromp-Taylor area scoring, and headless tournament infrastructure. Supports 9×9, 13×13, and 19×19 boards with komi, ko rule, and suicide prevention.

The arena serves as a fourth integration test bed for the HL thesis: **bandit-driven action selection + template-guided exploration > static heuristics or random baselines** in a deterministic perfect-information game.

Feature flag: `go = ["bandit", "dep:reqwest"]` (reqwest for AutoGo API bridge).

## Architecture

### Game Engine (`src/pruners/go/`)

| Module | Purpose |
|--------|---------|
| `types.rs` | `GoState`, `GoAction`, `GoCell`, `GoScore` |
| `state.rs` | Board state, legal moves, advance, scoring |
| `players/` | 6 AI player implementations |
| `autogo.rs` | REST API client for external Go engines |

### Core Types

```text
GoState
  ├─ board: Vec<GoCell>        // Black, White, Empty
  ├─ size: usize               // 9, 13, or 19
  ├─ to_play: GoCell           // current player
  ├─ ko_point: Option<usize>   // simple ko rule
  ├─ captures: [u32; 2]        // prisoners per color
  └─ move_history: Vec<GoAction>

GoAction
  ├─ Place(coord)              // stone placement
  └─ Pass                      // pass turn

GoScore (Tromp-Taylor)
  ├─ black_area: f64           // stones + territory
  ├─ white_area: f64           // stones + territory
  ├─ komi: f64                 // compensation (default 7.5)
  └─ winner: GoCell            // Black or White
```

### Scoring: Tromp-Taylor Area

Tromp-Taylor area scoring counts stones on the board plus empty points surrounded entirely by one color. This is the standard for computer Go — simple, deterministic, and no need for life/death judgment.

```text
score = own_stones + empty_points_reachable_only_by_own_color
winner = if black_score > white_score + komi then Black else White
```

## Player Strategies (6 Tiers)

### T1: Random

Randomly selects from legal moves. Baseline only.

```text
Strategy: uniform random from legal_moves()
Win rate vs self: ~30% (first-player disadvantage from komi)
```

### T2: Greedy

Scores every legal move by captures + liberties + positional heuristics.

```text
Scoring weights (Plan 073 — corner priority heuristic):
  capture_value:      10.0 per captured stone
  liberty_value:       1.0 per liberty after placement
  corner_side_bonus:   3.0 for 3rd line (territory), 2.0 for 4th line (influence)
  edge_penalty:       -2.0 for 1st line (too close to edge)
  connect_bonus:       1.0 per adjacent own stone, 0.5 per diagonal (bamboo joint)
  isolation_penalty:  -1.0 if isolated in enemy territory (≥2 adjacent opponent)
  random_noise:        0.1 for tie-breaking
Win rate vs Random: 70% (14/20 games) — more positional, less greedy on captures
```

### T3: Validator

Greedy + deterministic safety rules to avoid obviously bad moves.

```text
Safety rules:
  - No self-atari (placing into own group with 1 liberty)
  - No filling own eyes (surrounded by 4+ own stones)
  - Avoid 1-point jumps into enemy territory
Win rate vs Random: 100% (10/10 games)
```

### T4: HL (Heuristic Learning)

Bandit Q-learning over 8 move categories with AbsorbCompress.

```text
Categories: Capture, Defend, Extend, Invade, Atari, Connect, Tenuki, Pass
Strategy: UCB1 over category Q-values → pick best legal move in category
Adaptation: Q-values update per game outcome, cross-game learning
Win rate vs Random: 100% (10/10 games)
```

### T5: GZero (Template UCB1)

Template-based action proposal + bandit selection + delta-gating absorb-compress.

```text
Templates: CornerStar, Capture, Defend, Tenuki
Selection: UCB1 over template bandit arms
Delta-gating: only promote templates with positive δ (outcome improvement)
Self-play: 500 episodes, learns template ranking over time
```

### T6: MCTS (Monte Carlo Tree Search)

Generic MCTS with `GoState::advance()` as forward model.

```text
Configurable budget: 50–1000 simulations per move
Rollout: random playout to terminal, then Tromp-Taylor scoring
Selection: UCB1 tree policy
Weakness: budget=200 insufficient for Go's ~80 branching factor
Win rate vs Random: 65% (13/20 games, improved from 55% via territorial heuristic)
```

## Examples & Results

### go_00_api_bridge

REST API client for playing against an external AutoGo server.

**Status:** Requires running AutoGo server (`scripts/autogo_server.sh`). Plays random games against server agents via HTTP.

```bash
# Start AutoGo server first
./scripts/autogo_server.sh

# Run bridge
cargo run --features go --example go_00_api_bridge
```

---

### go_01_mcts — MCTS vs Random Benchmark

MCTS (budget=200) vs Random, 20 games on 9×9, komi=7.5.

**Results:**

| Metric | Value |
|--------|-------|
| MCTS Win Rate | 65% (13W / 7L) |
| As Black | 60% |
| As White | 70% |
| Avg Moves/Game | ~185 |
| Avg Time/Game | ~0.18s |
| Moves/sec | ~1018 |

**Verdict:** MCTS beats Random at budget=200 after Plan 073 territorial heuristic fix (65%, up from 55%). Previous 55% was caused by backwards center-preference weight — Go rewards corner/edge territory, not center control. Budget=200 still insufficient for Go's ~80 branching factor; higher budgets needed for stronger play.

```bash
cargo run --features go --example go_01_mcts
```

---

### go_02_tournament — All Players vs Random

Round-robin tournament: each player vs Random, 10 games, 9×9.

**Results:**

| Player | vs Random Win% | Avg Moves | Avg Time |
|--------|---------------|-----------|----------|
| Random | 35% | ~291 | <0.1s |
| Greedy | 70% | ~302 | 0.2s |
| Validator | **100%** | ~302 | 0.2s |
| HL | **100%** | ~302 | 0.5s |
| MCTS | 80% | ~196 | 0.2s |

**Verdict:**
- Validator/HL dominate Random with 100% win rate
- Greedy dropped to 70% after Plan 073 — more positional play, sometimes misses tactical captures
- MCTS improved from 55% → 80% via territorial heuristic in tournament format
- Random baseline wins ~35% due to first-player advantage from komi (Black)

```bash
cargo run --features go --example go_02_tournament
```

---

### go_03_head_to_head — AutoGo Server Tournament

Head-to-head matchups against external Go engines (e.g., GNU Go) via AutoGo REST API.

**Status:** Requires running AutoGo server. Tests Random, Greedy, HL, GZero, MCTS against server agents.

```bash
# Start AutoGo server
cd autogo && python play.py

# Run with minimal games for quick test
GO_GAMES=2 cargo run --features go --example go_03_head_to_head
```

---

### go_04_gzero — G-Zero Self-Play

GZero template-based self-play with delta-gating absorb-compress. 500 episodes on 9×9.

**Results:**

| Metric | Value |
|--------|-------|
| Total Episodes | 500 |
| Duration | 3.3 min |
| Episodes/sec | 2.5 |
| Black Wins | 493 (98.6%) |
| White Wins | 7 (1.4%) |
| Avg Moves/Game | 243 |

**Template δ Ranking:**

| Rank | Template | δ |
|------|----------|---|
| 🥇 | Capture | +0.0000 |
| 🥈 | CornerStar | -7.25 |
| 🥉 | Tenuki | -9.50 |
| 4 | Defend | -50.22 |

**Verdict:**
- Massive first-move (Black) advantage in self-play — Black wins 98.6%
- Capture is the only neutral-δ template (safe to play)
- Defend has worst δ — over-defending loses territory in self-play
- Absorb-compress: no templates promoted (δ below threshold)

```bash
cargo run --features go --example go_04_gzero
```

---

### go_05_autoresearch — AutoResearch Hyperparameter Scan

Bandit-driven hyperparameter search. 10 arms (configs), 50 evaluations, 10 games/eval, Greedy player vs Random baseline.

**Config Space:**

| Param | Range |
|-------|-------|
| MCTS Budget | 0 (Greedy only) |
| Rollout Depth | 10–50 |
| Exploration C | 0.9–1.7 |
| Bandit ε | 0.11–0.47 |
| Templates | 2–4 |

**Results:**

| Metric | Value |
|--------|-------|
| Best Config | M0:D30:C1.7:E0.11:T4 |
| Best Win Rate | 100% |
| Total Arms | 10 (all active) |
| Total Games | 500 |
| Duration | 62.9s (8 games/s) |
| Convergence | STABLE (-1.4pp Q1→Q4) |

**Top 5 Arms:**

| Rank | Config | Win Rate |
|------|--------|----------|
| 1 | M0:D20:C1.7:E0.32:T3 | 100% |
| 2 | M0:D20:C1.0:E0.14:T4 | 100% |
| 3 | M0:D50:C0.9:E0.21:T4 | 100% |
| 4 | M0:D50:C0.9:E0.15:T3 | 100% |
| 5 | M0:D50:C1.4:E0.41:T2 | 100% |

**Verdict:** All configs beat Random since they use Greedy baseline. AutoResearch shows the bandit correctly identifies that Random is an easy opponent — no meaningful config differentiation yet. Needs harder opponents (Validator, HL) to find meaningful hyperparameter differences.

```bash
cargo run --features go --example go_05_autoresearch
```

---

### go_06_bench — Go Benchmark Suite

Comprehensive benchmark: advance performance, MCTS throughput, player scaling laws.

#### T43: GoState::advance() Performance

| Config | Legal Moves | ops/sec | µs/advance | µs/clone |
|--------|-------------|---------|------------|----------|
| 9×9 opening | 82 | 486,773 | 2.05 | 1.87 |
| 9×9 midgame | 53 | 432,855 | 2.31 | 1.66 |
| 9×9 endgame | 11 | 390,964 | 2.56 | 1.81 |
| 19×19 opening | 362 | 129,442 | 7.73 | 8.18 |
| 19×19 midgame | 312 | 132,774 | 7.53 | 11.57 |
| 19×19 endgame | 169 | 122,326 | 8.17 | 7.42 |

#### T44: MCTS Search Throughput (9×9, ~10 moves played)

| Budget | µs/search | actions/sec | nodes/sec |
|--------|-----------|-------------|-----------|
| 50 | 316 | 3,168 | 158,408 |
| 200 | 1,439 | 695 | 138,982 |
| 500 | 3,291 | 304 | 151,915 |
| 1000 | 6,920 | 145 | 144,504 |

#### T46: Player Scaling Laws (9×9, 20 games each, Plan 073)

| Player | Wins | Losses | Win Rate |
|--------|------|--------|----------|
| Random | 7 | 13 | 35.0% |
| Greedy | 14 | 6 | 70.0% |
| Validator | 20 | 0 | **100.0%** |
| HL | 20 | 0 | **100.0%** |
| MCTS(200) | 17 | 3 | **85.0%** |

**Verdict:**
- `advance()` is fast: 2–3µs on 9×9, 7–8µs on 19×19 (2.5× faster than Plan 075 baseline)
- Clone cost ≈ advance cost (both copy the board vector)
- MCTS throughput scales linearly with budget (~140K nodes/sec, 5.6× faster than baseline)
- MCTS win rate improved from 60% → 85% via territorial heuristic
- Greedy more positional at 70% — trades some tactical wins for better shape

```bash
cargo run --features go --example go_06_bench
```

---

### go_07_tui — AI vs AI Auto-Play TUI

Animated ratatui TUI replay with unicode stone rendering. Two-panel layout: board grid + scoreboard.

```bash
# Default: Greedy (Black) vs Validator (White) on 9×9
cargo run --features go --example go_07_tui

# Custom players and board
cargo run --features go --example go_07_tui -- --black hl --white gzero --size 9

# Custom seed
cargo run --features go --example go_07_tui -- --seed 99
```

**Controls:** ←/→ step, Space auto-play (300ms), R new game, Home/End jump, Q quit.

**Rendering:** 1-char-wide symbols: `●` (black), `○` (white), `·` (empty), `+` (star/hoshi), `x` (ko). Last move highlighted green+bold.

## Cross-Domain Comparison

| Domain | Engine | ECS | Best AI | vs Random | Key Metric |
|--------|--------|-----|---------|-----------|------------|
| Bomberman | Tick-based | bevy_ecs | HL (bandit) | ~4:1 score | Survival 4% |
| Monopoly | FSM events | bevy_ecs | HL (bandit) | 56.5% win | Survival 93.7% |
| FFT Tactics | ATB queue | — | HL (bandit) | Enemy 93% | Unit MVP: Knight-HL 176 kills |
| **Go** | **Turn-based** | — | **Greedy/HL** | **100%** | **MCTS needs 1000+ budget** |

## Key Findings

1. **Territorial heuristic fixed (Plan 073)** — Go rewards corner/edge territory, not center control. Flipping `center_preference()` to `territorial_preference()` with phase-aware evaluation (Early: corners, Late: influence) improved MCTS from 55% → 65% vs Random.

2. **MCTS modest at budget=200** — Go's branching factor (~80 on 9×9, ~250 on 19×19) exhausts budget. The corrected heuristic helps (55% → 65%), but budget=200 remains insufficient for reliable dominance.

3. **Greedy trades tactics for position** — After Plan 073, Greedy dropped from 100% → 70% vs Random. The corner/side bonus makes it play more positionally, sometimes missing tactical captures. Validator and HL (which layer on top of Greedy) remain at 100%.

4. **Black advantage is massive in self-play** — GZero self-play shows 98.6% Black wins. First move + komi creates asymmetric advantage that template learning alone cannot overcome.

5. **HL adapts but doesn't surpass Greedy** — HL's bandit learning reaches 100% vs Random (matching Validator), but hasn't been tested head-to-head against Greedy yet. The TUI makes this easy to observe.

6. **advance() is production-ready** — 2µs on 9×9, 8µs on 19×19 (2.5× faster than initial). MCTS at budget=1000 processes ~144K nodes/sec, enabling real-time play on 9×9.

## Bug Fixes Discovered

### bomber_05_replay_gen.rs & bomber_06_replay_gen_v2.rs — Index Out of Bounds

`BomberAction` enum has 7 variants (Up, Down, Left, Right, Bomb, Wait, Detonate) but `ACTION_NAMES` and `action_counts` arrays had only 6 elements. When `Detonate` action (index 6) was selected, `action_counts[6]` panicked with "index out of bounds: the len is 6 but the index is 6".

**Fix:** Changed arrays from `[T; 6]` to `[T; 7]` and added `"Detonate"` to `ACTION_NAMES`.