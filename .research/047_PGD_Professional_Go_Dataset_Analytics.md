# Research: PGD — Professional Go Dataset for Data-driven Analytics (47)

> Source: [PGD: A Large-scale Professional Go Dataset for Data-driven Analytics](https://arxiv.org/abs/2205.00254) by Yifan Gao (USTC), IEEE CoG 2022
> Dataset: https://github.com/Gifanan/Professional-Go-Dataset
> Date: 2022-04 (paper), distilled 2026-06
> **Verdict: MODERATE VALUE — Feature engineering framework for Go game analytics directly maps to our `GoState`/`GoReplay` pipeline. In-game features (Garbage Moves, Unstable Rounds, Coincidence Rate, MLWR) are computable modelless from existing primitives. Outcome prediction requires more game data than we currently collect. Biggest win: early termination via Garbage Move detection for G-Zero self-play compute savings.**

## TL;DR

PGD is the first large-scale professional Go dataset: **98,043 games** by **2,148 players** (1950–2021) with three feature pillars that achieve **75.30%** game outcome prediction (CatBoost) vs 65.67% for WHR alone:

| Feature Category | Description | Accuracy (ablation) |
|---|---|---|
| Meta-information | Player age/gender/rank/WHR, tournament type/region/importance | 67.19% |
| Contextual | Recent win/loss streaks, matchup history, cross-region counts | 70.99% |
| In-game | KataGo win rate/score traces, Garbage Moves, Unstable Rounds, Coincidence Rate, MLWR | 68.83% |
| **All combined** | | **75.30%** |

Key finding: post-AlphaGo (2016+), professional Coincidence Rate (agreement with AI moves) spiked in opening, and Mean Loss Win Rate/Score dropped — pros improved by imitating AI.

## Why This Matters for Our Stack

### 1. We Already Have a "KataGo Lite" — `GoHeuristic`

Our `GoHeuristic` (in `src/pruners/go/state.rs`) evaluates Go states using a weighted combination:

| Component | Weight | PGD Equivalent |
|---|---|---|
| Liberty advantage | 35% | Proxy for KataGo win rate |
| Capture delta | 30% | Proxy for score differential |
| Influence (BFS closest-stone) | 20% | Territory control |
| Territorial preference (phase-aware) | 15% | Positional quality |

**Gap:** We compute heuristic at single states but don't **trace** it over time. PGD's power comes from temporal features computed across all moves in a game.

### 2. In-Game Features Are Modelless — No Neural Network Needed

PGD's most transferable insight: the in-game features are **engineered from board state + game history**, not neural embeddings:

| PGD Feature | Definition | Our Equivalent | Gap |
|---|---|---|---|
| Win Rate Trace | KataGo eval at each move | `GoHeuristic::evaluate()` at each move | Not traced |
| Score Trace | KataGo score diff at each move | `GoState::score()` at each move | Not traced |
| Coincidence Rate (CR) | % moves matching KataGo recommendation | % moves matching `GoGreedyPlayer` selection | Not computed |
| Mean Loss Win Rate (MLWR) | Avg win rate lost per move | Avg heuristic delta per move | Not computed |
| Mean Loss Score (MLS) | Avg score lost per move | Avg territory delta per move | Not computed |
| Garbage Moves (GM) | Moves after game is decided (win rate >90% for 4+ moves) | Heuristic stable > threshold for N moves | Not detected |
| Unstable Rounds (UR) | Consecutive moves with large heuristic swings | Not implemented | Not detected |
| Advantage Rounds (AR) | Rounds with 5%+ win rate advantage | Not implemented | Not detected |

### 3. Contextual Features Are Pure Statistics

Win/loss records, matchup history, and streak detection are countable from tournament results:

| PGD Feature | Our Data Source | Status |
|---|---|---|
| Match Results (last 10/20) | `GoTournamentResult` games | Partial (recorded per-tournament, not aggregated) |
| Match Results by Region | N/A (no region concept) | Not applicable |
| Matchup Results (H2H) | Tournament game pairs | Could compute from `GoTournamentResult` |
| Tournament Results (per-event) | Per-run results | Recorded but not indexed by player |
| Cross-region Counts | N/A | Not applicable |

## Distillation: Model-Based vs Modelless

### Modelless Path (microgpt-rs)

**All PGD in-game features are computable without a neural model.** This is the highest-value, lowest-risk distillation:

```
GoGameAnalytics (new module: src/pruners/go/analytics.rs)
├─ WinRateTrace: Vec<f32>            // GoHeuristic at each move
├─ ScoreTrace: Vec<f32>              // territory scoring at each move
├─ GarbageMoveRatio: f32             // % moves after game decided
├─ UnstableRoundCount: usize         // consecutive volatile swings
├─ MeanLossWinRate: f32              // avg heuristic drop per move
├─ MeanLossScore: f32                // avg territory drop per move
├─ CoincidenceRate: f32              // % agreement with GreedyPlayer
├─ AdvantageRoundCount: usize        // rounds with clear advantage
├─ StrongAdvantageRoundCount: usize  // rounds with dominant advantage
└─ CategoryDistribution: [f32; 8]    // HL category histogram (player style)
```

**Primitives already exist:**

- `GoHeuristic::evaluate()` → win rate proxy
- `GoState::score()` → score trace
- `GoGreedyPlayer::select_move()` → "recommended" move for CR
- `GoHLPlayer` category trace → style distribution
- `GoReplay::moves` → temporal data structure

**What's missing:** temporal aggregation loop — iterate through `GoReplay` moves, evaluate state at each step, compute deltas and thresholds.

#### Key Algo: Garbage Move Detection (Modelless)

PGD defines GM as: if after move X, under moving average (window=4), the leading player's win rate is always >90% or score difference is always >3 points, then all moves after X are garbage.

Our adaptation (no KataGo, use `GoHeuristic`):

```rust
fn detect_garbage_moves(trace: &[f32], threshold: f32, window: usize) -> usize {
    // threshold: heuristic > 0.85 means "clearly winning"
    // window: 4 consecutive stable evaluations
    if trace.len() < window { return 0; }
    let moving_avg: Vec<f32> = trace.windows(window)
        .map(|w| w.iter().sum::<f32>() / window as f32)
        .collect();
    for (i, &avg) in moving_avg.iter().enumerate() {
        if avg.abs() > threshold {
            // All moves from index i onward are garbage
            return trace.len() - i;
        }
    }
    0
}
```

#### Key Algo: Coincidence Rate (Modelless)

PGD uses KataGo's top-1 recommendation. We use `GoGreedyPlayer`:

```rust
fn coincidence_rate(replay: &GoReplay, state_seq: &[GoState]) -> f32 {
    let mut matches = 0usize;
    let mut greedy = GoGreedyPlayer;
    let mut rng = Rng::with_seed(0); // deterministic for comparison
    for (i, record) in replay.moves.iter().enumerate() {
        let state = &state_seq[i];
        let legal = state.legal_moves();
        let recommended = greedy.select_move(state, &legal, &mut rng);
        if let (GoAction::Place(r1, c1), GoActionSer::Place { row: r2, col: c2 }) =
            (&recommended, &record.action) {
            if r1 == *r2 && c1 == *c2 { matches += 1; }
        }
    }
    matches as f32 / replay.moves.len().max(1) as f32
}
```

### Model-Based Path (riir-ai)

**Requires a trained prediction model.** Lower priority until we have enough game data:

1. **GoOutcomePredictor** — LoRA or tree model predicting P(Black wins) from game features
   - Input: meta + contextual + in-game features from Phase 1
   - Output: win probability ∈ [0, 1]
   - Training data: collected from `go_03_tournament`, `go_09_lora_arena` runs
   - Risk: paper used 98K real pro games with KataGo; our simulated games with `GoHeuristic` have much noisier signal

2. **GoStyleEncoder** — embed player style from category distribution + CR + MLWR
   - Input: `[f32; 8]` category histogram + coincidence rate + MLWR
   - Output: style embedding for opponent modeling
   - Could condition LoRA training (already have `riir-gpu` LoRA stack)
   - Maps to PGD Section VI.B: "behavior and style modeling"

3. **Analytics Training Pipeline** — bridge from microgpt-rs features to riir-ai training
   - Collect features from `go_*` examples during tournament runs
   - Serialize to training format (already have `GoReplay::to_json()`)
   - Train predictor on historical data

## Actionable Plan

### Phase 1: Modelless Analytics (microgpt-rs, High Value, Low Risk)

**File:** `src/pruners/go/analytics.rs` (new, ~300 lines)

| Task | Description | Dependencies |
|---|---|---|
| T1 | `GoGameAnalytics` struct with all PGD-style features | `GoReplay`, `GoState`, `GoHeuristic` |
| T2 | `compute_analytics()` — trace `GoHeuristic` through replay moves | T1 |
| T3 | Garbage Move detection — early termination signal for G-Zero | T2 |
| T4 | Coincidence Rate — compare with `GoGreedyPlayer` recommendations | T2, `GoGreedyPlayer` |
| T5 | MLWR/MLS — per-move heuristic delta aggregation | T2 |
| T6 | Unstable Round detection — volatile swing counting | T2 |
| T7 | Player style histogram from `GoHLPlayer` category traces | `GoMoveCategory` |
| T8 | Integration into `go_07_tui` — show analytics overlay | T1–T7 |
| T9 | Integration into G-Zero self-play — early termination via GM | T3 |

**Highest ROI:** T3 (Garbage Moves) + T9 (early termination in G-Zero). Currently G-Zero plays ~243 moves/game, many after the outcome is decided. Cutting games short saves significant compute in self-play loops.

### Phase 2: Model-Based Prediction (riir-ai, Speculative)

**Files:** `riir-engine/src/go_analytics.rs` or `riir-gpu/src/game/go.rs` extension

| Task | Description | Dependencies |
|---|---|---|
| T10 | Feature serialization format (JSON → training tensor) | Phase 1 complete, game data collected |
| T11 | `GoOutcomePredictor` trait + LoRA implementation | T10, `riir-gpu` LoRA stack |
| T12 | `GoStyleEncoder` — embedding from category/CR/MLWR | T10 |
| T13 | Training pipeline — collect → serialize → train loop | T11 or T12 |
| T14 | Integration into tournament — pre-game prediction display | T11 |

**Risk:** We need ~1000+ games with features before training is meaningful. Paper had 98K. Start collecting data during Phase 1.

### Phase 3: Feedback Loop (Both Projects)

| Task | Description |
|---|---|
| T15 | G-Zero reward shaping — use MLWR as per-move signal (instead of only game outcome) |
| T16 | Opponent modeling — style embeddings condition `GoGZeroPlayer` template selection |
| T17 | Tournament analytics dashboard — aggregate features across tournament runs |
| T18 | AutoResearch integration — add in-game features to `ResearchArm` evaluation |

## Key Findings from PGD That Apply to Us

### 1. Post-AlphaGo Imitation Effect (Section V, Figure 5)

PGD shows CR spiked after 2016 in opening moves. **Our analogy:** as our G-Zero self-play converges, CR should increase — the AI learns to play "correctly" (matching greedy recommendations). Tracking CR over training episodes is a **modelless training progress metric**.

### 2. WHR Beats ELO (Table VII)

WHR (Whole-History Rating) outperforms ELO because it exploits long-term temporal dependencies. Our `arena::types::Ranking` uses ELO. **If we track player ratings across tournaments, WHR would improve our ranking quality.** However, WHR is expensive (Bayesian inference) and we have few repeated matchups currently.

### 3. Each Feature Category Is Additive (Table VIII)

Metadata=67.2%, Contextual=71.0%, In-game=68.8%, All=75.3%. **The features are complementary, not redundant.** This means even if our heuristic is noisy, adding contextual features (win streaks, H2H records) on top should still improve prediction.

### 4. Black Advantage Is Real (Table VI)

Black win rate drops from 55.5% (komi=4.5) to 46.9% (komi=7.5). **Our G-Zero self-play saw 98.6% Black wins** — this is partly a komi calibration issue and partly a first-mover advantage in self-play with weak players. PGD's data suggests our komi=7.5 is correct for 9×9, and the extreme Black advantage is a symptom of weak play, not rules.

## What NOT to Distill

| PGD Concept | Reason to Skip |
|---|---|
| Cross-region features | Our AI players have no "region" — this is a human sports analytics concept |
| Gender/age features | Not applicable to AI players |
| Tournament importance labeling | Our tournaments don't have prestige tiers |
| CatBoost/XGBoost/LightGBM training | We don't have enough data; would need external dependency |
| Live commentary enhancement (Section VI.D) | Out of scope — we're a training/research system, not a broadcast platform |
| 75.3% accuracy target | Unrealistic with simulated games + `GoHeuristic` vs 98K real games + KataGo |

## Existing Code Mapping

```
microgpt-rs/src/pruners/go/
├── state.rs          → GoHeuristic (our "KataGo lite")
├── replay.rs         → GoReplay (temporal data source)
├── players.rs        → GoGreedyPlayer (CR reference), GoHLPlayer (style trace)
├── g_zero_player.rs  → Integration point for early termination
├── tournament.rs     → Contextual feature data source
└── analytics.rs      → NEW: PGD feature extraction

riir-ai/crates/riir-gpu/src/game/
└── go.rs             → Model-based prediction (future)

riir-ai/crates/riir-examples/examples/
├── go_09_lora_arena.rs  → Training data collection
└── go_10_move_accuracy.rs  → Coincidence Rate validation
```

## Confidence Assessment

| Aspect | Confidence | Reason |
|---|---|---|
| Feature engineering maps to our codebase | **95%** | Direct structural match — `GoHeuristic` is our "KataGo lite" |
| Modelless analytics is implementable | **95%** | All features computable from existing primitives |
| Garbage Move early termination saves compute | **85%** | Simple threshold on heuristic stability |
| Coincidence Rate as training progress metric | **80%** | Our greedy player is ~70% win rate — noisy but directional |
| MLWR as per-move reward signal | **70%** | Heuristic deltas are noisy but carry signal |
| 75% prediction accuracy achievable | **40%** | Paper used 98K real games with KataGo; our signal is noisier |
| WHR rating improvement over ELO | **30%** | Need many more repeated matchups to justify WHR complexity |
| Style classification useful for G-Zero | **70%** | `GoHLPlayer` categories already capture style; aggregating is straightforward |
| Model-based prediction (LoRA) | **50%** | Architecture exists but untested for tabular/game prediction |

## References

- Paper: https://arxiv.org/abs/2205.00254
- Dataset: https://github.com/Gifanan/Professional-Go-Dataset
- Our Go engine: `.docs/14_go_arena.md`, Plan 065
- Our G-Zero self-play: Plan 049, `.plans/065_autogo_distillation.md`
- Our heuristic: `src/pruners/go/state.rs:GoHeuristic` (L560–790)