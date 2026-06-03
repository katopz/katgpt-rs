# Heuristic Learning & Arena Detail

> Arena proofs validate the HL thesis: adaptive intelligence > static rules > random.
> See main README for TLDR results. This document contains full benchmarks, architecture, and API detail.

## 1. Heuristic Learning Infrastructure

HL = software systems evolve through **code updates** not weight updates. A coding agent reads feedback and directly edits policies, validators, tests.

```
Episode N:   BanditPruner selects arm → environment runs → reward → TrialLog.append()
Episode N+k: AbsorbCompress promotes stable low-Q arms to hard blocks
Round N+m:   Agent writes new validator.rs → compile .wasm → HotSwapPruner.reload() → RegressionSuite
```

📖 See [`.docs/09_heuristic-learning.md`](../.docs/09_heuristic-learning.md).

### Inference-Time Review Metrics

Based on arXiv:2604.27233 — tracks whether reviewer intervention is net-positive via **Helpfulness/Harmfulness** metrics and a **benefit-to-risk ratio** (paper found 3.1:1 for o3-mini). Gates `AbsorbCompress` when ratio drops below threshold.

| Ratio | Interpretation |
|:-----:|:---------------|
| > 3.0 | Excellent reviewer (paper quality) |
| 2.0–3.0 | Acceptable (default threshold) |
| < 1.0 | Net-negative — stop reviewing |

Run: `cargo run --example review_01_metrics --features bandit`

### Emotion Vector Inference (Plan 162, Research 144)

Zero-cost behavioral early-warning via linear emotion projections from mid-layer residual-stream activations. Based on Anthropic Transformer Circuits research showing `desperation +0.1` → 14× reward-hacking increase; `calm +0.05` → 0% blackmail.

Each decode step: one O(d) dot product per emotion axis (valence, arousal, desperation, calm). No extra forward passes. `ReviewMetrics` accumulates emotion readings; `is_desperate_session(threshold)` fires when mean desperation exceeds threshold — enabling SR²AM to switch to safer planning mode before a DDTree commits to a high-risk path.

| Direction | Causal Effect |
|-----------|--------------|
| `desperation +0.05` | 22% → 72% reward-hacking (+50pp) |
| `desperation +0.1` | 5% → 70% reward-hacking (**14× increase**) |
| `calm +0.05` | baseline → **0% blackmail** |

`src/pruners/emotion_vector.rs` — `EmotionDirections`, `EmotionReading`. Phase 1 ✅ complete; Phase 2 ⏳ GOAT proof in progress.

### Entropy Anomaly Detection (Plan 061)

Session-level Out-Of-Distribution (OOD) monitoring using signals already in the pipeline:

| Signal | Source | Meaning |
|:-------|:-------|:--------|
| Mean entropy | `PPoT` Shannon entropy | Model confused by user inputs |
| Max entropy spike | Per-position `token_entropy()` | Single-position uncertainty peak |
| Prediction error | `DeltaMemoryState` error history | Inputs drifting from learned patterns |

`ReviewMetrics` now tracks `entropy_mean`, `entropy_max`, `entropy_n` per session. High mean entropy indicates the model cannot predict the user's intent — potential OOD or adversarial input.

```rust
// Wire into existing session
let metrics = Arc::new(ReviewMetrics::new());
metrics.record_entropy(token_entropy(&marginals)); // per decoding step

// Check anomaly
if metrics.is_high_entropy_session(threshold) {
    // Session is statistically abnormal
}
```

`DeltaMemoryState::mean_prediction_error()` exposes the running average prediction error as a drift signal — no new storage, data already tracked internally.

### ⚠️ Stepwise Reward Shaping (Plan 054) — NO GAIN

Distilled from [StepCodeReasoner](https://arxiv.org/pdf/2605.11922) (ICML 2026). **Benchmarked, no measurable improvement over flat rewards.** Feature-gated off by default, not in `full`.

| Method | Nodes | PathLen | Goal% | Time |
|--------|-------|---------|-------|------|
| Baseline (BinaryScreen) | 256 | 7 | 100% | 297ms |
| Flat rewards (λ=0) | 256 | 7 | 100% | 356ms |
| **Shaped rewards (λ=0.3)** | **256** | **7** | **100%** | **475ms** |

Same tree, same path, same goal rate — shaped rewards only add +33% latency. The paper's +7-14% gains come from GRPO gradient updates on a 7B model, not from post-hoc reward shaping on a bandit Q-value.

Infrastructure kept for future GRPO integration (G-Zero Phase 2). `stepcode` feature must be explicitly enabled.

Run: `cargo test --features "stepcode" --test bench_stepcode_modelless -- --nocapture`

## 2. Bomberman HL Arena — ✅ HL Thesis Proven

4-player Bomberman arena with `bevy_ecs` standalone. **Result: HL (+177) > Greedy (+131) > Validator (-30) > Random (-55)**.

| Player | Tech | Score | Wins |
|--------|------|-------|------|
| **HL** 🐵 | Opponent tracking + strategy + bandit | **+177** | **8** |
| Greedy 🐱 | Heuristic + 20% safe exploration | +131 | 5 |
| Validator 🐶 | Static safety rules | -30 | 1 |
| Random 🐰 | Blast-zone avoidance only | -55 | 9 |
| Rubric 🎯 | Multi-criteria rubric reward + template hints + Q-learning (`ropd_rubric`+`g_zero`+`bomber`) | — | 8 (8.0%)* |

*\*Plan 076 tournament: Rubric ≈ GZero (8W each), confirming single-axis hypothesis. High FFA draw rate (~80%) limits decisive outcomes. See `.benchmarks/009_arena_integration.md`.*

📖 See [`.docs/10_bomber_arena.md`](../.docs/10_bomber_arena.md). Tournament infrastructure: `bomber_09_rubric_tournament` example.

## 3. GameState Forward Model — STRATEGA Distillation

Generic `GameState` trait for what-if simulation, distilled from [STRATEGA framework](https://www.tnt.uni-hannover.de/papers/data/1606/2020__AIIDE_SGW__STRATEGA__A_General_Strategy_Games_Framework.pdf). Snapshot-based design: lightweight `Clone` structs (~2KB), no `bevy_ecs::World` dependency in the trait.

**Key finding confirmed: generic MCTS ≈ random (25% each) in 4-player Bomberman.** Domain heuristics (HLPlayer) beat generic search — exactly what STRATEGA reported.

| Component | Description |
|-----------|-------------|
| `GameState` trait | `advance()`, `available_actions()`, `is_terminal()`, `reward()`, `tick()` |
| `StateHeuristic<S>` trait | Pluggable evaluation for non-terminal states |
| `BomberState` snapshot | 13×13 grid + 4 players + bombs + power-ups, fully deterministic `advance()` |
| `mcts_search<S>()` | UCB1 tree selection + random rollouts, configurable budget/depth |
| `ActionSpaceLog` | Per-tick branching factor metrics |

100-round tournament (budget=200, rollout_depth=10):

| Player | Win Rate | Note |
|--------|----------|------|
| MCTS (P0) | 25.0% | ≈ random — generic search needs domain heuristics |
| Random (P1) | 24.0% | Baseline |
| Random (P2) | 21.0% | Baseline |
| Random (P3) | 30.0% | Baseline |

Feature gate: `game_state` (implies `bomber`). 50 unit tests covering explosions, chain reactions, power-ups, MCTS correctness.

Run: `cargo run --features game_state --example game_state_01_bomber_mcts`

📖 See [`.plans/056_game_state_forward_model.md`](../.plans/056_game_state_forward_model.md), [`.research/027_STRATEGA_General_Strategy_Games_Forward_Model.md`](../.research/027_STRATEGA_General_Strategy_Games_Forward_Model.md).

### 🔄 NFSP/MCTS Duality (Plan 067)

Both methods find a better action at state `s` for a student policy to imitate. They differ only in where the better action comes from:

```text
              Past                    Future
         ┌──────────────────┬──────────────────────┐
  Real   │ ReplayBackward  │  MCTS rollouts        │
         │ (BanditPruner)  │  (mcts_search)        │
         ├──────────────────┼──────────────────────┤
  Counter│ Bandit Q-update  │  Hint-δ              │
 factual │ (what worked)   │  (what model doesn't  │
         │                  │   know)               │
         └──────────────────┴──────────────────────┘
  Student: AbsorbCompress (doesn't know which teacher spoke)
```

**Why generic MCTS failed**: `mcts_search<S>()` uses random rollouts with no backward signal. Every game starts from scratch. Meanwhile `BanditPruner` carries Q-values across episodes — that's why HL (+177) dominates MCTS (25%, ≈ random). The fix: wire bandit Q-values into MCTS rollouts (AlphaZero pattern, but modelless).

| Teacher | Direction | Component | Signal |
|---------|-----------|-----------|--------|
| A (NFSP) | ← Backward | `BanditPruner` Q-values | Q(s,a) from past episodes |
| B (MCTS) | → Forward | `mcts_search<S>()` | Simulated rollouts |
| A+B | Both | `BanditRolloutPolicy` (Plan 067) | Bandit-informed rollouts |
| Neither | Counterfactual | `HintDelta` | Distribution shift at one state |

The inference pipeline (DDTree + BanditPruner) already embodies this duality at the token level — backward Q-values inform forward best-first search.


**Benchmark results (100-round tournament, release build):**

| Player | Wins | Win Rate | Note |
|--------|------|----------|------|
| **BanditMCTS (P0)** | **75** | **75.0%** | Bandit Q-values + domain heuristic |
| MCTS (P1) | 8 | 8.0% | Random rollouts, no memory |
| Random (P2) | 11 | 11.0% | Baseline |
| Random (P3) | 6 | 6.0% | Baseline |

**Δ BanditMCTS vs MCTS: +67.0pp** — confirms the duality hypothesis. Wiring backward signal (bandit Q-values) into forward search (MCTS rollouts) transforms MCTS from ≈random (Plan 056) to dominant. The AlphaZero pattern works even modelless (no neural net, just bandit statistics).

Feature gate: `bandit_mcts` (implies `game_state`). Run: `cargo test --release --features bandit_mcts --test bench_067_bandit_mcts -- --nocapture`

📖 See [`.plans/067_nfsp_mcts_duality.md`](../.plans/067_nfsp_mcts_duality.md).

## 4. Monopoly FSM Arena

4-player Monopoly with `bevy_ecs` standalone. Turn-based event-driven FSM with 8 phases, 40-square board, and 4 AI tiers.

| Player | Tech | Strategy |
|--------|------|----------|
| **HL** 🧠 | Bandit + opponent modeling + phase adaptation | Adaptive (Development preferred, Q=0.71) |
| Greedy 💰 | Heuristic scoring + set-completing trades | Aggressive acquisition + building |
| Validator 🛡️ | Safety rules ($200 reserve, no opponent monopolies) | Strategic buys + efficient building |
| Random 🎲 | Square-parity pseudo-random | Baseline |

**1000-game proof:** HL 56.5% win rate, 93.7% survival, +41.3pp over Validator. ✅ HL Thesis PROVEN (threshold: ≥5pp). Bandit explores all 5 strategies. Performance: 84.5 games/sec, 41µs/turn (24.4× under target).

4 examples (headless arena, TUI replay, 1000-game proof, benchmark).

📖 See [`.docs/11_monopoly_fsm.md`](../.docs/11_monopoly_fsm.md).

## 5. FFT Tactics Arena — TFT Party AI

Final Fantasy Tactics-inspired 4v4 ATB (Active Time Battle) arena with status effects, 6 classes, and 5 AI strategies. **TFT (Tit-for-Tat) dominates with 99% win rate** — game theory's optimal strategy applied to MMORPG party combat.

| Player | Tech | Win% | Survival | Kills/rnd |
|--------|------|------|----------|-----------|
| **TFT** 🦊 | Provocation FSM + role-based response | **99.0** | **95.7%** | **1.10** |
| HL 🐵 | Bandit Q-learning over 9 action types | 91.5 | 85.9% | 0.88 |
| Greedy 🐱 | Weakest-target + heal + potion | 56.1 | 35.7% | 0.83 |
| GZero 🤖 | Template hints + δ bandit + heuristics | 60.0* | 61.9% | 0.16 |
| Rubric 🎯 | Multi-criteria rubric reward + template hints + Q-learning (`ropd_rubric`+`g_zero`+`fft`) | 60.0* | — | — |
| Validator 🐶 | Safety-first + debuff cure + retreat | 5.0* | — | — |

*\*Plan 076 tournament (600 battles): Rubric ≡ GZero (identical 60% win rate, 100% draws head-to-head). The 3-criterion rubric collapses to scalar-equivalent signal. See `.benchmarks/009_arena_integration.md`.*

**TFT game theory:** Nice (role default) → Retaliatory (on provoke from `GameEvent::DamageDealt`) → Forgiving (10% generous TFT + 5-tick timer). Each class retaliates differently: Knight intercepts, WhiteMage heals first then attacks, BlackMage bursts.

**GvG Round-Robin** (250 rounds × 6 matchups): TFT 92.5% > HL 73.0% > Greedy 61.6%. Nash analysis confirms TFT is a dominant strategy.

4 examples (arena, rubric tournament, GvG tournament, A/B benchmark).
📖 See [`.docs/09_heuristic-learning.md`](../.docs/09_heuristic-learning.md) for full benchmark results.

## 6. Go: AutoGo Distillation (Plan 065)

Go GameState with full game logic (simple ko, Tromp-Taylor scoring), REST API bridge to AutoGo, 6 AI player strategies, G-Zero self-play, and AutoResearch loop for automated hyperparameter search. Port from `alpha_go/go.py:FastGoBoard` + `go_game.h:GoBoard`.

### GoState Performance (release build)

| Config | Legal Moves | advance() ops/sec | µs/advance | µs/clone |
|--------|-------------|-------------------|------------|----------|
| 9×9 opening | 82 | 619,009 | 1.62 | 1.70 |
| 9×9 midgame | 53 | 571,287 | 1.75 | 1.54 |
| 9×9 endgame | 11 | 436,576 | 2.29 | 1.55 |
| 19×19 opening | 362 | 145,737 | 6.86 | 6.66 |
| 19×19 midgame | 312 | 142,680 | 7.01 | 6.74 |
| 19×19 endgame | 169 | 135,793 | 7.36 | 6.70 |

### MCTS Throughput (9×9, ~10 moves played)

| Budget | µs/search | actions/sec | nodes/sec |
|--------|-----------|-------------|-----------|
| 50 | 305 | 3,274 | 163,680 |
| 200 | 1,330 | 752 | 150,329 |
| 500 | 3,123 | 320 | 160,120 |
| 1000 | 6,455 | 155 | 154,912 |

### Player Scaling Laws (9×9, 20 games vs Random)

| Player | Tech | Win% |
|--------|------|------|
| Greedy 🐱 | Capture + liberty + positional scoring | **100%** |
| Validator 🐶 | Safety-first rules on greedy | **100%** |
| HL 🐵 | Bandit Q-learning over 8 move categories | **100%** |
| MCTS (budget=200) | UCB1 tree + heuristic rollout | 60% |
| Random 🎲 | Uniform random legal move | 35% |

**Key finding**: Greedy/Validator/HL dominate random play. MCTS with random rollouts underperforms heuristic players — confirms STRATEGA result that generic search needs domain heuristics.

### Module Structure

| Component | Description |
|-----------|-------------|
| `GoState` | Flat array board, simple ko, Tromp-Taylor scoring, `GameState` trait |
| `GoHeuristic` | Weighted: liberty (40%) + capture (30%) + influence (20%) + center (10%) |
| `AutoGoClient` | REST API bridge to AutoGo `play.py` server |
| `GoPlayer` trait | `select_move()` — 6 implementations (Random, Greedy, Validator, HL, GZero, MCTS) |
| `GoReplay` | Game recording + deterministic playback |
| `GoTournament` | Head-to-head against AutoGo agents via API |
| `GoGZeroSelfPlay` | G-Zero self-play with HintDelta + absorb-compress |
| `AutoResearchLoop` | UCB1 bandit over config arms, early stopping, evolution |

Feature gate: `go` (implies `bandit`, `reqwest`). 693 tests pass. 7 examples.

Run: `cargo run --features go --example go_06_bench --release`

📖 See [`.plans/065_autogo_distillation.md`](../.plans/065_autogo_distillation.md).

## 7. Freeze/Thaw Knowledge Pipeline (Plan 092)

Zero-dependency `repr(C)` binary persistence for bandit knowledge. Play → learn → freeze to disk → reload → replay same rounds → measure improvement.

| Struct | Game | Size | Fields |
|--------|------|------|--------|
| `BomberFrozenBandit` | Bomber HL + GZero | ~92 bytes | Q-values (7), visits (7), compressed flags (7), total pulls |
| `GoFrozenBandit` | Go HL | ~88 bytes | Q-values (8), visits (8), epsilon, total pulls |
| `GoFrozenTemplates` | Go GZero | ~60 bytes | Q-values (4), visits (4), total pulls |

### Architecture

```text
┌────────────┐    freeze()    ┌──────────────┐   save_frozen()   ┌─────────────┐
│ HLPlayer   │──────────────▸│ repr(C)      │─────────────────▸│ .bin file   │
│ GZeroPlayer│               │ FrozenBandit │                   │ (raw bytes) │
│ GoHLPlayer │    thaw()     │ magic+ver+Q  │   load_frozen()   │ zero-dep    │
│ GoGZero    │◂──────────────│              │◂─────────────────│             │
└────────────┘               └──────────────┘                   └─────────────┘
```

- **Zero dependencies** — raw `std::fs::write`/`read` on `repr(C)` struct, no serde/bincode
- **Magic bytes + version** — `BDTB`/`GODT`/`GOTM` + version 1 for format validation
- **Deterministic replay** — same seed per round in both phases; frozen knowledge changes action selection but game engine is deterministic

### Example Results (100 rounds × 3 phases)

```sh
cargo run --example bomber_12_self_play_freeze --features bomber
cargo run --example go_08_self_play_freeze --features go
```

#### Go: GoHL vs Validator (α=1.0 per-move reward fix)

| Metric | Frozen | Baseline | Δ |
|--------|--------|----------|---|
| Win Rate | 25% | 14% | **+11pp ✅** |
| Avg Score | -13.3 | -16.8 | **+3.5 ✅** |

Q-values after learning (real differentiation vs old flat ~0.25):
```
Corner:0.80 Side:0.64 Center:0.74 Cap:0.75 Def:0.40 Ext:0.48 Inf:0.59 Pass:0.00
```

**Key fix:** α=1.0 (pure per-move reward) + 10× delta amplification. Old α=0.3 with game-end blending caused all Q-values to converge to ~0.25 when losing 86% of games — binary win/loss drowned the per-move heuristic signal.

- **Learning vs Random verified:** Q-values differentiate with spread > 0.1 (old bug: spread ~0.0), confirming per-move reward works against both strong and weak opponents. Test: `hl_learning_vs_random_q_values_differentiate`.

Feature gate: `bomber` or `go` (both imply `bandit`). 19 round-trip tests pass (includes `hl_learning_vs_random_q_values_differentiate`).

📖 See [`.plans/092_self_play_freeze_thaw.md`](../.plans/092_self_play_freeze_thaw.md).

## 8. Event Log — Game Trace Fork-Diff (Plan 124)

Append-only event-sourced game traces with fork-and-diff for counterfactual strategy exploration.

- **Deterministic replay** — any game byte-reproducible from event log
- **Cheap forking** — branch at move N without re-executing prefix
- **Structural diff** — compare two game traces event-by-event
- **Eval cache** — content-addressed evaluation with blake3 hashing

Feature gate: `event_log`

```rust
use katgpt_rs::pruners::event_log::*;

let mut log: EventLog<String> = EventLog::new();
log.push(EventType::GameStart, "start".into(), Actor::Runtime, None);

// Fork at event 3 for counterfactual
let forked = log.fork(EventId(3));
let diff = log.diff(&forked);
```

### GOAT Proofs (22/22 ✅)

| # | Proof | Status |
|---|-------|--------|
| 1 | Push/get/iter/len monotonic IDs | ✅ |
| 2 | Deterministic replay (100 games) | ✅ |
| 3 | Fork shares exact prefix events | ✅ |
| 4 | Structural diff identifies divergence | ✅ |
| 5 | Identical logs diff to empty | ✅ |
| 6 | Different length diff | ✅ |
| 7 | Causal chain via `caused_by` | ✅ |
| 8 | EvalCache insert/get/hit_rate | ✅ |
| 9 | Boundary fork (at end, past end) | ✅ |

### Game-Specific Wrappers

| Game | Wrapper | Actions |
|------|---------|---------|
| Bomber | `BomberEventLog` | `record_move`, `record_bomb`, `record_eval`, `record_game_start/end` |
| Go | `GoEventLog` | `record_place_stone`, `record_pass`, `record_resign`, `record_eval` |

📖 See [`.plans/124_event_log_game_trace_fork_diff.md`](../.plans/124_event_log_game_trace_fork_diff.md).

## 9. MeMo Reflection QA Pipeline (Plan 094)

Five-step data synthesis for generating compositional training data from game replays. Distilled from [MeMo: Memory as a Model](https://arxiv.org/abs/2605.15156).

| Step | Function | Output |
|------|----------|--------|
| 1. Extract | `(state, action, outcome) → QA` | Direct + indirect facts |
| 2. Consolidate | Merge related facts | Multi-fact questions |
| 3. Verify | Self-containment check | Verified QA pairs |
| 4. Surface | Entity-from-pattern | Reverse lookup QA |
| 5. Cross-Game | Converging clues | Cross-domain QA |

Feature gate: `memo_reflections`. Consumed by `BanditPruner` and `AbsorbCompress` — modelless path.

```sh
cargo run --example bomber_13_reflection_qa --features memo_reflections --release
cargo run --example go_09_reflection_qa --features memo_reflections --release
cargo test --features memo_reflections --test test_memo_reflections -- --nocapture
```

## 10. Self-Improving Loop (Plan 048)

The system closes the feedback → retrain → hot-swap cycle for continuous improvement:

```text
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌───────────┐
│  Inference   │────▸│  anyrag Cache     │────▸│  LoRA Retrain │────▸│  Hot-Swap  │
│  + Feedback  │     │  episodic memory  │     │  (wgpu GPU)   │     │  zero-downtime │
└─────────────┘     └──────────────────┘     └──────────────┘     └───────────┘
```

- **FeedbackConsumer** polls anyrag episodic cache for new feedback samples
- **Retrain** triggers LoRA fine-tuning on accumulated samples via wgpu GPU pipeline
- **Hot-Swap** signals inference layer to swap adapters without downtime
- Feature-gated: `cargo build -p riir-gpu --features feedback-consumer`

See [riir-ai `.docs/13_research_audit_results.md`](../../riir-ai/.docs/13_research_audit_results.md) for the full research audit.

## 11. G-Zero: Verifier-Free Self-Play (Plan 049)

Distilled from [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/pdf/2605.09959) (Huang et al., 2026). Makes our existing **modelless HL smarter** with the Hint-δ signal, then optionally adds gradient-based self-play on top.

### Core Innovation: Hint-δ

An intrinsic reward measuring how much a hint shifts the Generator's predictive distribution — **no external verifier or LLM judge needed**:

```text
δ(q, h, a_hard) = (1/T) Σ [log πG(at | q, h, a<t) − log πG(at | q, a<t)]
```

δ is large only when the query is challenging AND the hint carries information the Generator lacks. Two objectives in one scalar — and it's architecture-agnostic.

### Two Phases: Modelless First, Model-Based Second

| Phase | Mechanism | Updates | Cost | Strength |
|-------|-----------|---------|------|----------|
| **Phase 1 (Modelless)** | δ → `AbsorbCompress` + `BanditPruner` | Heuristics/rules | Low | Safe, fast, proven HL loop |
| **Phase 2 (Model-Based)** | δ → GRPO + DPO | LoRA weights | High | Stronger for open-ended domains |

Phase 1 makes the existing modelless path **smarter** — δ is a denser, more informative reward than raw environment feedback. Phase 2 adds neural self-play only when needed.

### Phase 1: Smarter Modelless (T1–T5)

```text
TemplateProposer ──(query, hint)──▸ Generator (frozen, inference only)
       │                                    │
       │                             log-probs with/without hint
       │                                    │
       │                               HintDelta
       │                                    │
       │                    ┌───────────────┴──────────────┐
       │                    ▼                              ▼
       │          DeltaGatedAbsorbCompress      DeltaBanditPruner
       │          (promote high-δ arms          (δ as dense reward
       │           to hard constraints)          for arm selection)
       │                    │                              │
       │                    └──────────┬───────────────────┘
       │                               ▼
       │                     TrialLog (JSONL)
       │                               │
       └─── next episode ◂─────────────┘
```

**No gradient updates.** The model generates log-probs for inference only. All learning happens through heuristic promotion and bandit Q-values, same as existing HL — but with a better reward signal.

| New Component | What | Why Smarter |
|---------------|------|-------------|
| `HintDelta` | Log-prob shift computation | Shared foundation for both phases |
| `DeltaGatedAbsorbCompress` | Absorb only when δ reveals blind spot | Promotes heuristics the model doesn't already know |
| `DeltaBanditPruner` | δ as dense reward for arm selection | No need to wait for episode completion |
| `TemplateProposer` | Rule-based query-hint generation | 0 GPU cost, targets blind spots from bandit history |

### Phase 2: Model-Based Self-Play (T6–T9) — ✅ Complete (Plan 059)

Implemented in `riir-gpu` (3,369 lines, 76 tests). Builds on Phase 1's δ computation — adds gradient-based training via GRPO (Proposer) and length-normalized DPO (Generator):

```text
Phase 2a — Proposer Training (GRPO):
  NeuralProposer πP generates {(qi, hi)} → Generator answers unassisted
  → δ reward + length/BLEU penalties → GRPO gradient update

Phase 2b — Generator Training (Length-Normalized DPO):
  Frozen πP generates query-hints → Generator answers with/without hint
  → lower-half δ filter → DPO update (hint-assisted=chosen, unassisted=rejected)
  → HotSwapPruner reloads adapter (zero-downtime)
```

| Module | Lines | Key Components | Tests |
|--------|-------|---------------|-------|
| `loss_dpo.rs` | 774 | `LengthNormalizedDpo`, `PreferencePair`, `DpoMetrics`, GPU DPO pipeline | CPU parity + GPU tests |
| `loss_grpo.rs` | 565 | `GrpoConfig`, `group_advantage`, `grpo_loss`, `cispo_loss` (default), `GrpoLossVariant`, `grpo_reward`, `length_penalty` | Advantage + loss + CISPO GOAT tests |
| `proposer.rs` | 413 | `Proposer` trait, `NeuralProposer`, `TemplateProposerAdapter`, `QueryTemplate` | Template tests |
| `delta_filter.rs` | 794 | 6-stage filter (δ percentile → length → ratio → zlib → echo → role markers) | 24 filter tests |
| `gzero_loop.rs` | 823 | `GZeroLoop`, `GZeroRound`, `RoundMetrics`, `GZeroCheckpoint` (crash recovery) | 5 checkpoint tests |
| GPU kernels | — | `dpo_log_ratio.wgsl` + `dpo_reduce.wgsl` (per-pair log-ratio + tree reduction) | GPU parity tests |

### Three Training Paths

```text
SelfImprovingCycle {
  Collecting → ReadyToSynthesize → ...
    ├── Path A (existing):  Export JSONL → riir-burner LoRA SFT          (modelless HL)
    ├── Path B (Phase 1):   δ → DeltaGatedAbsorbCompress + DeltaBanditPruner (smarter modelless)
    └── Path C (Phase 2):   Proposer↔Generator self-play → DPO LoRA      (model-based G-Zero)
}
```

Path A → B is **incremental** (same architecture, better signal). Path B → C is **opt-in** (add gradient training when modelless plateaus). All three feed into `HotSwapPruner`.

### Key Design Decisions (from paper)

| Decision | Rationale |
|----------|-----------|
| **Modelless first** | δ is architecture-agnostic — use it without DPO/GRPO before adding complexity |
| Lower-half δ filter `[0, 50th %ile]` | Low-δ = hard-to-distinguish pairs = fine-grained DPO signal; high-δ = answer leakage |
| Length-normalized DPO | Neutralizes vanilla DPO's length bias via per-token mean log-ratio |
| Length penalty `λ·max(0, |h|-200)/100` | Prevents verbose hint reward hacking |
| BLEU duplication penalty `|Ci|/|B|` | Prevents Proposer collapse into repetitive pairs |

### Critical Finding

>70% of DPO training pool is **non-verifiable tasks** (advice, writing, explanation), yet reasoning **transfers** to verifiable math domains. Structural depth is internalized, not memorized.

| Model | Chat (AlpLC) | IFEval-pS | AIME25 | Average |
|-------|-------------|-----------|--------|---------|
| Qwen3-8B base → G-Zero R2 | 8.47 | 43.81 | **12.40** | **35.43** (+1.48) |
| Llama-3.1-8B → G-Zero R2 | **27.86** | 59.52 | 0.63 | **43.90** (+1.13) |

### Phase 1 Benchmark Results (Plan 049 T5)

Run: `cargo test --features "g_zero,bomber" --test bench_gzero_modelless -- --nocapture`

| Metric | GZero | HL | Greedy | Random |
|--------|-------|----|--------|--------|
| Survival (500r) | 3.8% | 4.6% | 4.4% | 5.6% |
| Total Score | 10 | 927 | 835 | -359 |
| δ mean | +1.77 | — | — | — |
| Templates explored | 8/8 | — | — | — |
| select_action | 1.8µs | 5.2µs | 10.9µs | 0.4µs |

**Key findings:**
- δ signal is meaningful: mean +1.77, 100% positive, variance σ²=3.30
- GZero is 65% faster than HL on `select_action` (no BFS escape in hot path)
- Template exploration covers all 8 archetypes (>5% weight each)
- Phase 2 (GRPO + DPO) blocked on `riir-gpu` training infrastructure

📖 See [`.plans/049_g_zero_self_play.md`](../.plans/049_g_zero_self_play.md) for full implementation plan, types, hyperparameters, and risk assessment.

## 12. Emotion Vector Inference Control (Plan 162)

Modelless emotion reading from mid-layer activations during decode — zero extra forward pass, O(d) dot product per step. Based on [Anthropic Transformer Circuits Thread 2026](https://transformer-circuits.pub/) finding that emotion vectors causally drive behavior (desperation → 14× reward hacking increase).

| Signal | What It Measures | Integration |
|--------|-----------------|-------------|
| **Valence** (PC1) | Happy/calm vs desperate/angry | `ReviewMetrics.emotion_profile_summary().valence` |
| **Arousal** (PC2) | High vs low activation | `ReviewMetrics.emotion_profile_summary().arousal` |
| **Desperation** | Reward-hacking-prone regimes | `ReviewMetrics.is_desperate_session(0.3)` |
| **Calm** | Stable, confident regimes | `ReviewMetrics.emotion_profile_summary().calm` |

### GOAT Proof Results

Run: `cargo test --features bandit --test bench_emotion_vector_goat -- --nocapture`

| Proof | Description | Verdict |
|-------|-------------|--------|
| G1 | Throughput: 4×O(d) dot products < 20% overhead at d=64 debug, <0.1% at production scale | ✅ |
| G2 | Binary size: EmotionReading=16B, EmotionProfileSummary=40B, zero heap alloc | ✅ |
| G3 | Information gain: desperation vs entropy r=-0.45, R²=0.20, 80% unexplained variance | ✅ |
| G4 | Desperation predicts failure: r=0.99, `is_desperate_session()` correctly flags | ✅ |

### Key API

```rust,ignore
use katgpt_rs::pruners::emotion_vector::EmotionDirections;
use katgpt_rs::pruners::ReviewMetrics;

// Load calibrated directions (once at model init)
let dirs = EmotionDirections::zeros(d_model);
let reading = dirs.read_emotions(&mid_layer_activation);

// Record into review metrics (thread-safe, atomic)
metrics.record_emotion(reading.valence, reading.arousal, reading.desperation, reading.calm);

// Check desperation flag
if metrics.is_desperate_session(0.3) {
    // Trigger cautionary planning via SR²AM configurator
}

// Get full profile summary
let profile = metrics.emotion_profile_summary();
println!("valence={:.3} arousal={:.3} desperation={:.3} calm={:.3}",
    profile.valence, profile.arousal, profile.desperation, profile.calm);
```

### SR²AM Integration

`ConfiguratorContext` now includes `desperation_bin` (Plan 162 T11), allowing the SR²AM configurator bandit to learn different planning strategies for desperate vs calm sessions.

**Default-on** — no feature gate needed. The `Config.emotion_desperation_threshold` defaults to `0.5`.
