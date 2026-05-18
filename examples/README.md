# Examples

All examples run with `cargo run --example <name>`. Some require feature flags.

## Quick Reference

| # | Group | Examples | Feature |
|---|-------|----------|---------|
| 1 | Bandit (RL) | 7 examples | `bandit` |
| 2 | Heuristic Learning | 2 examples | `bandit` |
| 3 | Bomberman Arena | 8 examples | `bomber`, `bomber-wasm`, `bomber-agent` |
| 4 | Monopoly FSM | 4 examples | `monopoly` |
| 5 | FFT Tactics Arena | 1 example | — |
| 6 | GameState Forward Model | 2 examples | `game_state` |
| 7 | Blue Bear | 2 examples | — |
| 8 | Core | 4 examples | varies |
| 9 | Dungeon | 2 examples | — |
| 10 | Sudoku | 4 examples | `sudoku` |
| 11 | Tactical AI | 6 examples | — |
| 12 | Review | 1 example | `bandit` |
| 13 | Percepta Comparison | 1 example | — |
| 14 | Getting Started | 1 example | — |
| 15 | Stepwise Reward Shaping | 1 example | `stepcode` |
| 16 | Go (AutoGo) | 8 examples | `go` |

---

## 1. Bandit (RL / Game Theory)

Multi-armed bandit strategies for adaptive decision-making under uncertainty.

### bandit_01_basic

Basic bandit strategy comparison: UCB1, ε-greedy with decay, Thompson Sampling on a 5-armed Bernoulli bandit with ASCII regret plot.

```bash
cargo run --example bandit_01_basic --features bandit
```

### bandit_02_ddtree

DDTree-based speculative decoding with bandit arm selection for token pruning.

```bash
cargo run --example bandit_02_ddtree --features bandit
```

### bandit_03_slot

Slot machine simulation demonstrating bandit reward optimization.

```bash
cargo run --example bandit_03_slot --features bandit
```

### bandit_04_combat

Combat scenario with bandit-driven action selection and opponent modeling.

```bash
cargo run --example bandit_04_combat --features bandit
```

### bandit_05_rps

Rock-Paper-Scissors tournament with bandit strategy adaptation.

```bash
cargo run --example bandit_05_rps --features bandit
```

### bandit_06_resolver

Conflict resolver using constrained bandit with action masking — blocked arms get relevance 0.0, never explored even if highest reward.

```bash
cargo run --example bandit_06_resolver --features bandit
```

### bandit_07_director

Director pattern — meta-bandit that selects which sub-bandit strategy to activate per round.

```bash
cargo run --example bandit_07_director --features bandit
```

---

## 2. Heuristic Learning (HL)

Trial logging and hot-swapping for the HL (Heuristic Learning) infrastructure.

### hl_01_trial_log

Trial logging with structured outcome tracking — logs strategy selection, reward, and Q-value updates per trial.

```bash
cargo run --example hl_01_trial_log --features bandit
```

### hl_02_hotswap

Hot-swap demo — dynamically switches between pruner strategies at runtime based on performance feedback.

```bash
cargo run --example hl_02_hotswap --features bandit
```

---

## 3. Bomberman HL Arena

4-player Bomberman with `bevy_ecs` standalone. Tick-based priority FSM with 8 states, 4 AI tiers, WASM-validated NN player, and replay data generation.

### bomber_01_arena

Headless 100-game tournament. Per-game results, cumulative standings, and HL thesis check.

```bash
cargo run --example bomber_01_arena --features bomber
```

### bomber_02_tui

Animated ratatui TUI replay with board rendering, player stats, and event log. Controls: Space/←/→/F/A/Q.

```bash
cargo run --example bomber_02_tui --features bomber
```

### bomber_03_hl_proof

1000-game HL proof experiment. Measures win rate, survival rate, and bandit learning Q-values.

```bash
cargo run --example bomber_03_hl_proof --features bomber
```

### bomber_04_nn

NNPlayer demo with WASM validator safety checks. Loads `bomber_validator.wasm` at runtime for A/B comparison vs native safety rules. Falls back gracefully if WASM unavailable.

```bash
# With WASM validator:
cargo run --example bomber_04_nn --features bomber-wasm -- /path/to/bomber_validator.wasm

# Without WASM (native fallback):
cargo run --example bomber_04_nn --features bomber-wasm
```

### bomber_05_replay_gen

Dedicated replay generator for training data. 1000 rounds, filters P3 (Validator) and P4 (HL) winning episodes with quality > 0.5. Outputs JSONL with board state, bombs, powerups, and action labels.

```bash
cargo run --example bomber_05_replay_gen --features bomber
```

### bomber_06_replay_gen_v2

Enhanced replay generator with richer per-sample metrics: `danger_level`, `nearest_opponent_dist`, `escape_routes`. Enables ML training with context-aware quality scoring.

```bash
cargo run --example bomber_06_replay_gen_v2 --features bomber
```

### bomber_07_bomb_types

Bomb type demo — showcases 4 bomb variants: Timed (standard fuse), Piercing (blast passes through walls), Remote (detonate on demand), and Landmine (proximity trigger). Each type demonstrated with tick-by-tick output.

```bash
cargo run --example bomber_07_bomb_types --features bomber
```

### bomber_08_agent_loop

Agent validator optimization loop — evolves bomber safety rule sets using population-based search. Discovers optimal rule combinations (BlockOpponent, DistanceFromBomb, AvoidBlast, AvoidDeadEnd, SeekPowerUp) across generations with stagnation detection.

```bash
cargo run --example bomber_08_agent_loop --features bomber-agent
```

---

## 4. Monopoly FSM Arena

4-player Monopoly with `bevy_ecs` standalone. Turn-based event-driven FSM with 8 phases, 40-square board, and 4 AI tiers.

### monopoly_01_arena

Headless 100-game tournament with cumulative standings and HL thesis check.

```bash
cargo run --example monopoly_01_arena --features monopoly
```

### monopoly_02_tui

Animated ratatui TUI replay with colored property groups, player stats, and scrollable event log. Controls: Space/←/→/F/A/Home/End/Q.

```bash
cargo run --example monopoly_02_tui --features monopoly
```

### monopoly_03_hl_proof

1000-game HL proof experiment. Survival rates, win rates, bandit Q-values, and statistical significance check.

**Results:** HL 56.5% win rate, 93.7% survival, +41.3pp over Validator. ✅ HL Thesis PROVEN.

```bash
cargo run --example monopoly_03_hl_proof --features monopoly
```

### monopoly_04_bench

Performance benchmark — measures game throughput, per-turn latency, and latency distribution (p50/p90/p99).

**Performance:** 87 games/sec, 41µs/turn (25× under 1ms target).

```bash
cargo run --example monopoly_04_bench --features monopoly
```

---

## 5. FFT Tactics Arena

Final Fantasy Tactics-inspired 4v4 turn-based battle. Pure data-driven (no ECS), speed-based turn queue, 4 classes with HP/MP, and 4 AI tiers.

### fft_01_arena

Headless 100-round tournament. 8 units (4v4) with classes: Knight, Archer, Black Mage, White Mage. AI strategies: Random, Greedy, Validator, HL (bandit Q-learning). Outputs per-round kills, final standings, unit stats, and HL Q-value convergence.

```bash
cargo run --example fft_01_arena
```

---

## 6. GameState Forward Model (STRATEGA)

Generic forward model trait + MCTS search across game domains. Validates the `GameState` abstraction from STRATEGA research.

### game_state_01_bomber_mcts

MCTS vs Random 4-player FFA tournament (100 rounds). Confirms STRATEGA finding: generic MCTS ≈ Random (25%) in high-variance FFA without domain heuristics.

```bash
cargo run --example game_state_01_bomber_mcts --features game_state
```

### game_state_02_bomber_gvg

2v2 GvG (Guild vs Guild) MCTS showcase. Team Alpha (P0,P1) uses MCTS with team-aware heuristic vs Team Beta (P2,P3) Random/Greedy. Demonstrates MCTS beats Random (62%) with clear team objectives, but Greedy (OSLA) still dominates (100%). Budget scaling shows diminishing returns after 500.

```bash
cargo run --example game_state_02_bomber_gvg --features game_state
```

## 7. Blue Bear

Experimental tools and TUI demos.

### bear_01_demo

Blue Bear tactical puzzle solver — uses DDTree with `TacticalPruner` as a heavily constrained state-space search. Solves a 3×3 map (BXT/#MG) in 7 steps with step-by-step verification.

```bash
cargo run --example bear_01_demo
```

### bear_02_tui

Ratatui TUI with animated step-through, auto-play, solution replay, and emoji grid rendering. Navigate with ←/→/Home/End, toggle auto-play with A.

```bash
cargo run --example bear_02_tui
```

---

## 8. Core

Core library features — validation, inference, sampling.

### core_01_validator

Syntax-aware token pruning with `SynPruner`. BPE tokenizes Rust source, validates partial syntax with `syn`, only explores syntactically valid branches.

```bash
cargo run --example core_01_validator --features validator
```

### core_02_raven

Raven RSM (Routing Slot Memory) demo — 3 parts: (1) frozen-slot memory preservation under noise, (2) O(1) per-step scaling vs O(N) flat attention, (3) memory footprint comparison.

```bash
cargo run --example core_02_raven
```

### core_03_ppot

PPoT logit-parameterized CPU resampling.

```bash
cargo run --example core_03_ppot --features ppot
```

### core_04_prefill

Prefill/prompt processing demo.

```bash
cargo run --example core_04_prefill
```

---

## 9. Dungeon

Roguelike dungeon generation and TUI.

### dungeon_01_tui

Ratatui TUI dungeon explorer with procedural generation.

```bash
cargo run --example dungeon_01_tui
```

### dungeon_02_multifloor

Multi-floor dungeon generation with stairs and floor transitions.

```bash
cargo run --example dungeon_02_multifloor
```

---

## 10. Sudoku

Streaming "Thinking" Sudoku solver with deterministic validation.

### sudoku_01_9x9

Standard 9×9 solver demonstrating deterministic rules engine, O(log N) attention, and streaming step-by-step constraint satisfaction.

```bash
cargo run --example sudoku_01_9x9 --features sudoku
```

### sudoku_02_speculative

DDTree + Deterministic Validator pruning with 3-level comparison: Unpruned vs Static-Only vs Path-Aware. Shows path-aware pruning catches cross-depth conflicts.

```bash
cargo run --example sudoku_02_speculative --features sudoku
```

### sudoku_03_tui

Ratatui TUI with color-coded grid, step/trace panels, and speculative mode comparison side-by-side.

```bash
cargo run --example sudoku_03_tui --features sudoku
```

---

## 11. Tactical AI

Grid-based tactical AI with terrain, procedural maps, and parallel simulation.

### tactical_01_ai

Basic tactical AI decision-making on a grid.

```bash
cargo run --example tactical_01_ai
```

### tactical_02_terrain

Terrain-aware pathfinding and movement cost calculation.

```bash
cargo run --example tactical_02_terrain
```

### tactical_03_procedural

Procedural map generation for tactical scenarios.

```bash
cargo run --example tactical_03_procedural
```

### tactical_04_parallel

Parallel tactical simulation with rayon — runs multiple scenarios concurrently.

```bash
cargo run --example tactical_04_parallel
```

### tactical_05_bench

Performance benchmark for tactical AI systems.

```bash
cargo run --example tactical_05_bench
```

### tactical_06_tui

Ratatui TUI tactical map viewer with unit positions and terrain rendering.

```bash
cargo run --example tactical_06_tui
```

---

## 12. Review

Inference-time review metrics based on arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents".

### review_01_metrics

Tracks how often the bandit reviewer *fixes* a wrong random pick (helpful) vs *breaks* a correct pick (harmful). Computes benefit-to-risk ratio and AbsorbCompress gating. Compares UCB1, Thompson Sampling, and ε-greedy strategies.

```bash
cargo run --example review_01_metrics --features bandit
```

---

## 13. Getting Started

### sudoku_04_percepta_vs

Percepta head-to-head comparison: Rust hull attention (O(log N) Graham Scan) vs Python+C++ transformer (WASM bytecodes). Benchmarks two Sudoku puzzles with unfair-but-informative speed comparison.

**Results:** Rust backtracking ~350K steps/sec, 2500× faster than Percepta's 30K tok/s — mostly algorithmic advantage, not language.

```bash
cargo run --example sudoku_04_percepta_vs
```

---

## 14. Getting Started

### hello_py2rs

Python-to-Rust migration primer — demonstrates idiomatic Rust patterns for Python developers.

```bash
cargo run --example hello_py2rs
```

---

## 15. Stepwise Reward Shaping (StepCodeReasoner Plan 054)

Intra-trajectory reward shaping distilled from StepCodeReasoner (ICML 2026). Rewards bandit arms proportionally to how many downstream arms they enable.

### stepcode_01_shaped_bandit

Demonstrates flat vs shaped reward convergence, path consistency metrics, and the effect of λ (shaping coefficient).

```bash
cargo run --example stepcode_01_shaped_bandit --features stepcode
```

---

## 16. Go (AutoGo)

Go game AI with 6 player strategies: Random, Greedy, Validator, HL, GZero, MCTS. Tromp-Taylor area scoring on 9×9, 13×13, or 19×19 boards. Full docs: [`.docs/15_go_arena.md`](../.docs/15_go_arena.md).

### go_00_api_bridge

REST API client for playing against an external AutoGo server. Requires running AutoGo server (`scripts/autogo_server.sh`). Plays random games against server agents via HTTP.

```bash
# Start AutoGo server first
./scripts/autogo_server.sh
cargo run --features go --example go_00_api_bridge
```

### go_01_mcts

MCTS (budget=200) vs Random benchmark, 20 games on 9×9. **Result:** MCTS wins 55% — barely above random due to insufficient budget for Go's ~80 branching factor.

```bash
cargo run --features go --example go_01_mcts
```

### go_02_tournament

Round-robin tournament: each player vs Random, 10 games. **Results:** Greedy/Validator/HL all 100%, MCTS 70%.

```bash
cargo run --features go --example go_02_tournament
```

### go_03_head_to_head

Head-to-head matchups against external Go engines (e.g., GNU Go) via AutoGo REST API. Requires running server.

```bash
GO_GAMES=2 cargo run --features go --example go_03_head_to_head
```

### go_04_gzero

GZero template-based self-play with delta-gating absorb-compress. 500 episodes. **Result:** Black wins 98.6% — massive first-move advantage. Template ranking: Capture (+0.0) > CornerStar (-7.25) > Tenuki (-9.50) > Defend (-50.22).

```bash
cargo run --features go --example go_04_gzero
```

### go_05_autoresearch

Bandit-driven hyperparameter search. 10 arms, 50 evaluations, Greedy vs Random. **Result:** All configs win 100% (Random too easy for meaningful differentiation).

```bash
cargo run --features go --example go_05_autoresearch
```

### go_06_bench

Comprehensive benchmark: advance performance (5µs 9×9, 23µs 19×19), MCTS throughput (~25K nodes/sec), player scaling laws.

```bash
cargo run --features go --example go_06_bench
```

### go_07_tui

Animated ratatui TUI replay — AI vs AI auto-play on a Go board. Two-panel layout: unicode stone grid + scoreboard with captures, score estimate, and last move. Supports 6 player types and configurable board size.

```bash
# Default: Greedy (Black) vs Validator (White) on 9×9
cargo run --features go --example go_07_tui

# Custom players and board
cargo run --features go --example go_07_tui -- --black hl --white gzero --size 9

# Custom seed
cargo run --features go --example go_07_tui -- --seed 99
```

Controls: ←/→ step, Space auto-play, R new game, Q quit.

---

## Feature Flags

| Flag | Gates | Dependencies |
|------|-------|-------------|
| `sudoku` | SudokuPruner, sudoku examples | — |
| `validator` | SynPruner, syntax validation | `syn`, `proc-macro2` |
| `ppot` | PPoT resampling | — |
| `bandit` | BanditPruner, bandit/HL examples | — |
| `bomber` | Bomberman arena (Plan 033) | `bevy_ecs`, `bandit` |
| `bomber-wasm` | Bomberman NNPlayer with WASM validator | `wasmtime`, `bomber` |
| `monopoly` | Monopoly FSM arena (Plan 035) | `bevy_ecs`, `bandit` |
| `go` | AutoGo API bridge + Go GameState (Plan 065) | `bandit`, `reqwest` |
| `stepcode` | StepCode shaped bandit rewards | — |
| — | FFT Tactics Arena (Plan 047) | `fastrand` |
| `rest` | REST API client | `reqwest`, `tokio` |
| `embedding_router` | Semantic embedding retrieval | — |
| `gpu` | GPU compute | `wgpu`, `safetensors` |
| `leviathan` | LeviathanVerifier rejection sampling | — |
| `full` | All of the above | — |

```bash
# Run with specific feature
cargo run --example monopoly_01_arena --features monopoly

# Run with all features
cargo run --example sudoku_01_9x9 --features full