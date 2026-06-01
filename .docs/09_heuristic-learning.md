# katgpt-rs: Heuristic Learning

> **Status (Plan 049):** G-Zero self-play distillation — both phases complete. Phase 1 (modelless): `HintDelta`, `DeltaGatedAbsorbCompress`, `DeltaBanditPruner`, `TemplateProposer` behind `--features g_zero` (implies `bandit`). Phase 2 (model-based): `LengthNormalizedDPO`, `GRPO`, `DeltaFilter` (6-stage), `GZeroLoop` in `riir-gpu` (Plan 059, 3,369 lines, 76 tests, 2 DPO WGSL kernels). See `src/pruners/g_zero/` and `riir-gpu/src/`.
>
> **Status (Plan 061):** Entropy anomaly detection — `ReviewMetrics` now tracks session-level entropy (mean, max, count) via `record_entropy()` and `is_high_entropy_session()`. `DeltaMemoryState::mean_prediction_error()` exposes drift signal. Behind `--features bandit,delta_mem`. See `.plans/061_entropy_anomaly_detection.md`.
>
> **Status (Plan 036):** ReviewMetrics, ReviewStrategy, and benefit-ratio gating are implemented behind `--features bandit`. AbsorbCompress gates compression by benefit-risk ratio. `ppot_rescue_reviewed` provides structured review loops behind `--features bandit,ppot`. See example `review_01_metrics`.
>
> **Status (Plan 071):** ROPD Rubric modelless distillation — `RubricVector`, `RubricTemplate`, `RubricGatedAbsorbCompress`, `RubricBanditPruner` behind `--features ropd_rubric` (implies `bandit`). Per-criterion gap targeting replaces scalar δ with structured multi-criteria reward. `RubricPlayer` (bomber, `g_zero`+`bomber`) and `RubricFFTPlayer` (FFT, `g_zero`+`fft`) integrate rubric reward into arena players (Plan 071 T9/T10). Benchmark: 5.3M observe_rubric/sec, 20/20 targeting accuracy, zero regression. See `.benchmarks/007_ropd_rubric_modelless.md`.
>
> **Status (Plan 076):** Arena Integration — cross-arena tournament infrastructure (`arena/types.rs`, `arena/scheduler.rs`, `bomber/arena_runner.rs`, `fft/arena_runner.rs`). Round-robin tournaments with ELO ratings confirm **Rubric ≈ GZero** in both Bomber (8W vs 8W) and FFT (60% vs 60%, 100% draws head-to-head). The 3-criterion rubric vector collapses to the same effective signal as scalar Hint-δ. See `.benchmarks/009_arena_integration.md`.
>
> **Status (Plan 072):** SDAR Gated distillation modelless — `sdar_gate()`, `SdarBanditPruner`, `SdarGatedAbsorbCompress` behind `--features sdar_gate`. Asymmetric trust: sigmoid gate σ(β·x) endorses positive gaps, attenuates negative. β=5.0 paper-validated. Benchmark: 118M updates/sec, zero hot-path overhead, 97.5% targeting accuracy. See `.benchmarks/008_sdar_gated_modelless.md`.
>
> **Status (Plan 078):** RePlaid Variance-Minimized Schedules — `VarianceMinimizer`, `AdaptiveNoiseSchedule`, `train_mini_dllm_adaptive()`, `VarianceEpsilon` bandit strategy, `SdarLearnedBeta` integrated into `SdarBanditPruner` via `with_learned_beta()` builder. Self-supervised schedule optimization: minimizes per-step loss variance to equalize denoising difficulty (RePlaid Prop 1). Schedule converges from `[0.15, 0.25, 0.35]` → `[0.192, 0.211, 0.239]`. D2F Higher-Order Denoising (T10.5/T10.6): DPM-Solver++(2M) multistep logit extrapolation, potential 4× throughput. Behind `--features replaid_schedules` (off by default, experimental). See `.benchmarks/012_replaid_variance_schedules.md`.
>
> **Status (Plan 032):** TrialLog, AbsorbCompressLayer, HotSwapPruner, and RegressionSuite are implemented behind `--features bandit`. See examples `hl_01_trial_log` and `hl_02_hotswap`.
>
> **Status (Plan 060):** MeMo Reflection QA Pipeline — 5-step compositional data synthesis from game replays. `ReflectionStep` (DirectExtraction, IndirectExtraction, Consolidation, Verification, EntitySurfacing, CrossGameSynthesis), `ReflectionQA`, `ReflectionDomain` behind `--features memo_reflections`. Consumed by BanditPruner for training signal enrichment. See `src/pruners/reflection.rs`.

## What is Heuristic Learning?

Heuristic Learning (HL) is a paradigm where **software systems evolve through code updates** rather than neural network weight updates. A coding agent reads feedback (test failures, environment rewards, logs, replays) and directly edits policies, validators, tests, and memory — no backpropagation required.

> Source: [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) by Jiayi Weng

### The Core Idea

```
Traditional ML:  data → gradient → update weights → better model
Heuristic Learning: feedback → agent edits code → better rules → better system
```

The coding agent changes the **maintenance cost curve** for heuristics. Rules that were once "too expensive to own" become viable long-term code when an agent can maintain them.

---

## HL in katgpt-rs

katgpt-rs is uniquely positioned for HL because of its **trait-based pruner architecture** and **WASM sandbox**:

| HL Concept | katgpt-rs Component |
|---|---|
| Heuristic Policy | `ConstraintPruner::is_valid()` — masks invalid tokens/actions |
| Relevance Scoring | `ScreeningPruner::relevance()` — prioritizes good actions |
| Gradient-free Learning | `BanditPruner` — Q-value updates without backprop |
| Sandboxed Heuristics | `WasmPruner` — compiled validators in WASM sandbox |
| Trial History | `TrialLog` — persistent JSONL episode records |
| Rule Compression | `AbsorbCompressLayer` — promote stable Q-values to hard constraints |
| Hot-reload | `HotSwapPruner` — runtime .wasm reload |
| Regression Safety | `RegressionSuite` — replay golden episodes |
| Self-Play Reward | `HintDelta` — intrinsic δ signal from model's own distribution (Plan 049) |
| δ-Gated Compression | `DeltaGatedAbsorbCompress` — absorb only when hint reveals blind spot (Plan 049) |
| δ-Reward Bandit | `DeltaBanditPruner` — δ as dense, immediate reward signal (Plan 049) |
| Rubric-Gated Absorb | `RubricGatedAbsorbCompress` — per-criterion gap targeting (Plan 071) |
| Rubric-Reward Bandit | `RubricBanditPruner` — rubric-weighted multi-criteria reward (Plan 071) |
| SDAR Sigmoid Gate | `sdar_gate()` — asymmetric trust σ(β·x), β=5.0 optimum (Plan 072) |
| SDAR-Gated Bandit | `SdarBanditPruner` — sigmoid-gated reward updates (Plan 072) |
| SDAR-Gated Absorb | `SdarGatedAbsorbCompress` — soft sigmoid promotion gate (Plan 072) |
| Knowledge Persistence | `BanditPruner` → `src/pruners/freeze.rs` — `repr(C)` bandit knowledge save/load (Plan 092) |
| Width Scaling | `best_of_k_rollouts()` — K parallel SDE rollouts, select best (PTRM Plan 083) |
| Early Stop Gate | `EarlyStopGate<P>` — depth-aware pruning when relevance < threshold (PTRM Plan 083) |
| Width Selection | `WidthSelectionMode::{BestQ, MostFrequent, Top1Converged}` — rollout selection strategy (PTRM Plan 083, EqR Plan 119) |

---

## The Two Operations

A healthy Heuristic System needs two operations (from the HL research):

### 1. Absorb

Feed new observations back into the system:

```
Episode N:   BanditPruner selects arm → environment runs → reward
             TrialLog.append(episode, arm, reward, q_value, note)
             AbsorbCompressLayer.absorb(arm, reward)
```

### 2. Compress

Fold accumulated knowledge into simpler, more maintainable rules:

```
After N episodes:
  arm 3 (Wait) has Q=0.02 over 500 visits → promote to hard block
  arm 0 (Attack near enemy) has Q=0.89 → boost relevance weight
  → AbsorbCompressLayer.compress() returns [3]
  → BanditPruner delegates arm 3 to hard block (relevance=0.0)
```

> An HS that only grows and never compresses becomes a big ball of mud.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Heuristic System                     │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ ConstraintPruner│ │ScreeningPruner│ │ BanditPruner│ │
│  │ is_valid()    │  │ relevance()  │  │ relevance() │ │
│  │ hard block    │  │ soft score   │  │ adaptive    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘ │
│         │                 │                  │        │
│         ▼                 ▼                  ▼        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              WASM Validator (.wasm)              │  │
│  │  Sandbox: no I/O, no floating point, 4MB max    │  │
│  └────────────────────┬────────────────────────────┘  │
│                        │                               │
│         ┌──────────────┼──────────────┐                │
│         ▼              ▼              ▼                │
│  ┌────────────┐ ┌─────────────────────┐ ┌──────────────┐ │
│  │  TrialLog  │ │AbsorbCompressLayer  │ │RegressionSuite│ │
│  │  (JSONL)   │ │ (Q→blocks)          │ │ (golden)     │ │
│  └────────────┘ └─────────────────────┘ └──────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## The HL Feedback Loop

### During Episodes (Online Learning)

```
1. DDTree proposes branches (tokens/actions)
2. BanditPruner.relevance() scores each branch
3. Best branch selected → environment executes
4. Reward observed → BanditPruner.update(arm, reward)
5. TrialLog.append(record) for persistence
6. `AbsorbCompressLayer::absorb(arm, reward)` for compression check
```

### Between Episodes (Compression)

```
1. `AbsorbCompressLayer::should_compress()` → true (threshold met)
2. `AbsorbCompressLayer::compress()` → identify arms to promote
3. Low-Q arms → blocked (relevance = 0.0) via `AbsorbCompressLayer`
4. High-Q arms → boost relevance weight
5. `RegressionSuite.replay_golden()` → verify no regression
```

### Freeze/Thaw Persistence (Plan 092)

`BanditPruner` uses `src/pruners/freeze.rs` for `repr(C)` bandit knowledge persistence across sessions. Arena players (Bomber, FFT, Go) call `.freeze()` → `save_frozen()` to write raw bytes, and `load_frozen()` on startup to restore Q-values and visit counts. Zero-dependency binary I/O — no serde/bincode needed.

**Per-Move Reward Fix (Issue 065):** Initial implementation used blended reward (`α=0.3 * per_move + 0.7 * game_end`), which caused all Q-values to collapse to ~0.25 when losing 86% of games (binary game-end reward = 0 for losses). Fix: `α=1.0` (pure per-move heuristic delta) + 10× amplification. Result: **+11pp win rate** for frozen GoHL vs Validator over naive baseline. Q-values now differentiate meaningfully (Corner: 0.80 vs Defense: 0.40).

Run: `cargo run --example go_08_self_play_freeze --features go`

### Between Rounds (Evolution)

```
1. Agent reads TrialLog → identifies failure patterns
2. Agent writes new validator.rs → compile to .wasm
3. HotSwapPruner.reload() → load new .wasm
4. RegressionSuite.replay_golden() → verify improvement
5. Keep or revert based on regression results
```

---

## System 1 / System 2

The HL paradigm suggests a split between fast intuition and slow deliberation:

### System 1 (Fast, ~100µs)

The inference hot path:

```
LoRA Draft Model → DDTree Branches → BanditPruner scores → WasmPruner validates
                                                     ↓
                                              Select best valid action
```

- **LoRA model**: "intuition" about good actions (marginals)
- **BanditPruner**: adaptive scoring based on past experience (Q-values)
- **WasmPruner**: hard safety constraints (validation rules)

### System 2 (Slow, seconds)

The evolution loop:

```
TrialLog → Agent reads failures → Writes new validator → compile .wasm → HotSwap → Regression test
```

- **TrialLog**: persistent memory of what worked and what didn't
- **AbsorbCompressLayer**: automatic rule promotion from experience
- **RegressionSuite**: safety net against regressions
- **Coding agent**: writes new validators based on failure analysis

---

## Bomberman HL Arena (Proof of Concept)

4 AI players compete in a Bomberman arena built with `bevy_ecs` (standalone) + ratatui emoji TUI. Game logic patterns adapted from `raw/bomby/` (Fish Folk: Bomby) — same ECS components, resources, and systems, but tick-based instead of real-time.

### Architecture

```
raw/bomby/ (reference)              →  katgpt-rs bomberman (ours)
──────────────────────────────────────────────────────────────────
bevy (full engine)                  →  bevy_ecs (standalone ECS only)
bevy_ecs_ldtk (LDtk level loading)  →  ProceduralArena (grid generator)
bevy Sprite / TextureAtlas          →  ratatui emoji TUI
bevy_kira_audio                     →  (none — silent)
bevy Time (real delta)              →  discrete tick counter
leafwing-input-manager              →  BomberPlayer trait (AI selects)
bevy Commands / EventWriter         →  bevy_ecs Commands / EventWriter ✅
bevy Query / Resource / Plugin      →  bevy_ecs (same patterns) ✅
```

### Players (Technology Ladder)

| Player | Emoji | Tech Stack | What It Proves |
|---|---|---|---|
| P1: Random | 🐰 | `fastrand::random()` | Baseline |
| P2: Model | 🐱 | LoRA marginals | Model > random |
| P3: Validator | 🐶 | LoRA + WASM pruner | Validator > model alone |
| P4: Full HL | 🐵 | LoRA + WASM + Bandit + TrialLog + AbsorbCompress | HL > static rules |

The P3 vs P4 comparison is the key proof: both use the same model and validator, but P4 adapts through bandit learning while P3 uses static rules.

### ECS Components (from bomby patterns)

```rust
#[derive(Component)] struct Player { id: u8 }
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)] struct GridPos { x: i32, y: i32 }
#[derive(Component)] struct BombFuse { owner: Entity, ticks_remaining: u32 }
#[derive(Component)] struct BombRange { cells: u32 }
#[derive(Component)] struct BombCount { max: u8, active: u8 }
#[derive(Component)] struct Speed { cells_per_tick: u8 }
#[derive(Component, Default)] struct Alive;
#[derive(Component)] struct Bomb { bomb_type: BombType }  // Timed | Piercing | Remote | Landmine
#[derive(Component)] struct PowerUp { kind: PowerUpKind }  // BombUp | FireUp | SpeedUp
#[derive(Component)] struct Blast;
#[derive(Component)] struct DestructibleWall;
```

### TUI Grid (emoji rendering)

| Cell | Emoji | Cell | Emoji |
|------|-------|------|-------|
| Floor | `··` | Fixed wall | `🧱` |
| Destructible | `📦` | Player 1-4 | `🐰🐱🐶🐵` |
| Bomb (fresh) | `💣` | Bomb (low fuse) | `🧨` |
| Blast | `💥` | PowerUp | `🔥💥👟` |

See [Plan 033](/.plans/033_bomberman_arena.md) for full implementation details.

---

## Quick Start

```rust,ignore
use katgpt_rs::pruners::{
    BanditPruner, BanditStrategy, AbsorbCompressLayer, TrialLog, CompressConfig,
};

// Create a bandit pruner
let mut bandit = BanditPruner::new(
    katgpt_core::NoScreeningPruner,
    BanditStrategy::Ucb1,
    6, // 6 arms (actions)
);

// Create absorb-compress layer
let mut absorb = AbsorbCompressLayer::new(
    katgpt_core::NoScreeningPruner,
    6,
    CompressConfig::default(),
);

// Create trial log
let mut trial_log = TrialLog::new(std::path::Path::new("/tmp/hl_trials.jsonl"))?;

// Run episodes
for episode in 0..1000 {
    let arm = bandit.best_arm();  // select via strategy
    let reward = env.pull(arm);   // environment feedback
    
    bandit.update(arm, reward);
    absorb.absorb(arm, reward);
    trial_log.append(&TrialRecord {
        episode, arm, reward,
        q_value: bandit.q_values()[arm],
        cumulative_reward: 0.0,
        cumulative_regret: 0.0,
        config: String::new(),
        note: String::new(),
        ..Default::default()
    });
    
    // Absorb-compress check every 100 episodes
    if absorb.should_compress() {
        let promoted = absorb.compress();
        println!("Compressed arms: {promoted:?}");
    }
}
```

---

## Slot Machine Bandit: Rules-Based Speculative Decoding (Plan 031)

A slot machine that closes the full speculative decoding loop with **no real transformer needed**:

```
Reel weights → DDTree → Payline rules → Reward → Bandit learns → Repeat
```

Unlike `bandit_demo.rs` (coin flips, disclaimer required) and `bandit_ddtree_demo.rs` (random marginals, random verification), this demo uses **structured reel weights** as marginals and **deterministic payline rules** for verification — proving actual value, not just mechanical compatibility.

### Slot ↔ Speculative Decoding Analogy

| Speculative Decoding | Slot Machine |
|---------------------|--------------|
| Draft model marginals P(token\|context) | Reel weights P(symbol\|reel) |
| Target model verification | Payline rules (combo valid?) |
| Accept → 1.0, Reject → 0.0 | Payout table (graded 0.0–1.0) |
| BanditPruner screens branches | Bandit learns which symbols pay |

### Slot Machine Configuration

6 symbols (vocab_size=6), 3 reels (lookahead=3):

| Symbol | Reel 0 | Reel 1 | Reel 2 | Payout (Triple) |
|--------|--------|--------|--------|-----------------|
| 🍒 Cherry | 30% | 25% | 20% | 0.5 |
| 🍋 Lemon | 25% | 20% | 20% | 0.5 |
| 🍊 Orange | 20% | 20% | 20% | 0.5 |
| 🔔 Bell | 15% | 15% | 15% | 0.6 |
| 💎 Diamond | 7% | 10% | 15% | 0.8 |
| 7️⃣ Seven | 3% | 10% | 10% | 1.0 (JACKPOT) |

### Results (500 episodes, seed=42)

| Strategy | Total Reward | Avg Reward | Best Combo | Triples | vs Random |
|----------|-------------|------------|------------|---------|-----------|
| UCB1 | 82.40 | 0.1648 | 🍒🍒🍒 | 6 | +60.9% |
| ε-greedy | 250.10 | 0.5002 | 🔔🔔🔔 | 500 | +388.5% |
| Thompson | 247.30 | 0.4946 | 🔔🔔🔔 | 490 | +383.0% |
| Random | 51.20 | 0.1024 | 💎💎💎 | 17 | baseline |

All bandit strategies significantly outperform random. ε-greedy and Thompson converge to Bell triples (reliable 0.6 reward) while random occasionally hits Diamond triples by luck.

Run: `cargo run --example bandit_03_slot --features bandit`

---

## Model vs Modelless Bandit: Proven Results (Plan 025)

Two demos prove whether model-based speculative decoding with bandit is worth the cost vs modelless bandit-only.

### bandit_ddtree_demo.rs — Model-Based vs Modelless

Uses simulated marginals (concentrated vs uniform) flowing through real `build_dd_tree_screened()` + `BanditPruner`.

| Metric | Model-based | Modelless | Δ |
|--------|-------------|-----------|---|
| Cumulative Reward | 7880.00 | 7027.00 | **+12.1%** |
| Cumulative Regret | 120.00 | 973.00 | **-87.7%** |
| Accept Rate | 98.5% | 87.8% | **+10.7%** |
| Avg Time/Episode | 70.8 µs | 63.9 µs | +10.8% |

### game_resolver_demo.rs — Domain Validator + Bandit

Uses `GameActionScreener` (native Rust game action validator) as inner pruner for `BanditPruner<GameActionScreener>`.

| Metric | Constrained (domain+bandit) | Unconstrained (bandit only) | Δ |
|--------|----------------------------|-----------------------------|---|
| Cumulative Reward | 2275.00 | 2929.00 | -22.3% |
| Cumulative Regret | 5725.00 | 5071.00 | +12.9% |
| Accept Rate | 75.8% | 36.6% | **+39.2%** |
| Avg Time/Episode | 39.6 µs | 62.5 µs | **-36.6%** |

### Key Findings

1. **Model-based wins on quality**: +12.1% reward, -87.7% regret, +10.7% accept rate
2. **Domain screener dramatically improves accept rate**: +39.2% over bandit alone
3. **Domain screener is faster**: -36.6% latency — pruning invalid branches early reduces DDTree work
4. **Bandit learns meaningful arms**: Constrained converges on game-relevant tokens; Unconstrained spreads visits thinly
5. **Modelless still functional**: 87.8% accept rate proves bandit can learn without model priors, just slower

Run: `cargo run --example bandit_02_ddtree --features bandit`
Run: `cargo run --example bandit_06_resolver --features bandit`

---

## Inference-Time Review Metrics (Plan 036)

Based on arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents". The paper proves that inference-time review improves tool-calling accuracy by +5.5% on irrelevance detection and +7.1% on multi-turn tasks. The key insight is the **measurement framework**, not the reviewer itself.

### Classification Matrix

Each (base_correct, reviewed_correct) pair is classified into one of four categories:

| Base Correct | Reviewed Correct | Classification | Meaning |
|:---:|:---:|:---:|:---|
| false | true | **Helpful** | Reviewer fixed a wrong answer |
| true | false | **Harmful** | Reviewer broke a correct answer |
| true | true | Both Correct | Both agreed (no effect) |
| false | false | Both Wrong | Both failed (no effect) |

### Benefit-Risk Ratio

```
Benefit-Risk Ratio = Helpfulness ÷ Harmfulness
```

- **Helpfulness** = `helpful / (helpful + both_wrong)` — % of base-wrong cases the reviewer fixed
- **Harmfulness** = `harmful / (harmful + both_correct)` — % of base-correct cases the reviewer broke

The paper found **3.1:1** for o3-mini. Our default threshold is **2.0:1** (conservative — allows slightly worse reviewers).

### How It Connects to Existing Systems

| Component | Review Metrics Integration |
|---|---|
| `ReviewMetrics` | Atomic counters tracking helpful/harmful/both_correct/both_wrong + entropy anomaly (mean, max, count) |
| `BanditSession::with_metrics()` | Records whether bandit pick vs random pick was optimal |
| `AbsorbCompress::should_compress_gated()` | Blocks compression when ratio < threshold |
| `PpotConfig::with_review_loop(N)` | Structured review loop (paper's rN) |
| `ppot_rescue_reviewed()` | PPoT rescue with benefit-ratio gate |
| `TrialLog::append_with_review()` | Persist episode with review classification |

### Benefit-Ratio Guidance

| Ratio | Interpretation | Action |
|:---:|:---|:---|
| **> 3.0** | Excellent reviewer (paper quality) | Aggressively compress, trust reviewer |
| **2.0–3.0** | Acceptable reviewer | Compress with normal caution |
| **1.0–2.0** | Marginal reviewer | Gate compression, investigate failures |
| **< 1.0** | Net-negative reviewer | Stop reviewing, revert to base |
| **∞** | Perfect (never broke correct) | Trust fully, but monitor for overfitting |

### Quick Start

```rust,ignore
use std::sync::Arc;
use katgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv, ReviewMetrics};

let metrics = Arc::new(ReviewMetrics::new());

let session = BanditSession::new(env, BanditStrategy::Ucb1)
    .with_metrics(Arc::clone(&metrics));

let (events, result) = session.run(1000, &mut rng);

// Print review metrics
println!("{metrics}"); // "helpful=83.5% harmful=20.5% ratio=4.1:1 n=1000"

// Check if compression is safe
let ratio = metrics.benefit_ratio();
if ratio >= 2.0 {
    absorb.compress(); // Safe to harden reviewer decisions
}
```

Run: `cargo run --example review_01_metrics --features bandit`

---

## G-Zero Self-Play Distillation (Plan 049)

> **Source:** [G-Zero: Self-Play for Open-Ended Generation from Zero Data](https://arxiv.org/pdf/2605.09959) — Huang et al., 2026
> **Feature:** `--features g_zero` (implies `bandit`)

G-Zero replaces external LLM judges with an **intrinsic signal** (Hint-δ) derived from the model's own predictive distribution. It enables self-evolution for open-ended domains where verifiable rewards don't exist.

### Hint-δ: The Core Signal

```
δ(q, h, a_hard) = (1/T) Σ [log πS(a_hard_t | q, a_hard_<t)
                       − log πS(a_hard_t | q, h, a_hard_<t)]
```

Both terms score the **same** `a_hard` tokens — the difference is whether the hint `h` is in the prompt. Positive δ means the hint shifted the Solver's distribution away from its own unassisted response — the hint revealed a blind spot.

```text
High δ  → hint carries structural signal → blind spot found
Low δ   → hint is redundant              → already known
Neg δ   → hint hurt                      → ignore
```

### Modelless Path (Phase 1 — Implemented)

No gradient updates. δ enhances existing HL infrastructure:

| Component | Role | Source |
|-----------|------|--------|
| `HintDelta` | Core δ computation | `g_zero::types` |
| `DeltaGatedAbsorbCompress` | Absorb only when δ reveals blind spot | `g_zero::delta_absorb` |
| `DeltaBanditPruner` | δ as dense reward for bandit arms | `g_zero::delta_bandit` |
| `TemplateProposer` | Rule-based (query, hint) generation | `g_zero::template_proposer` |

### Model-Based Path (Phase 2 — Plan 059)

Gradient-based self-play in `riir-gpu`. δ signal trains LoRA weights via DPO on the Generator and GRPO on the Proposer.

| Component | Role | Location |
|-----------|------|----------|
| `GpuDpoLoss` | Length-normalized DPO-sigmoid loss (2 WGSL kernels) | `riir-gpu::loss_dpo` |
| `LengthNormalizedDpo` | CPU reference: dot(policy-ref, mask) / mask_sum | `riir-gpu::loss_dpo` |
| `GrpoConfig` | Group-relative policy optimization | `riir-gpu::loss_grpo` |
| `group_advantage` | (reward − μ) / σ normalization within K rollouts | `riir-gpu::loss_grpo` |
| `grpo_loss` | Clipped policy gradient with group baseline | `riir-gpu::loss_grpo` |
| `Proposer` trait | Query-hint generation interface | `riir-gpu::proposer` |
| `TemplateProposerAdapter` | Wraps modelless `TemplateProposer` for GPU pipeline | `riir-gpu::proposer` |
| `DeltaFilter` | 6-stage preference pair filtering (δ percentile → length → ratio → zlib → echo → role markers) | `riir-gpu::delta_filter` |
| `GZeroLoop` | Round orchestration with crash recovery | `riir-gpu::gzero_loop` |
| `dpo_log_ratio.wgsl` | Per-pair length-normalized log-ratio computation | `riir-gpu/kernels/` |
| `dpo_reduce.wgsl` | Sigmoid loss + metric aggregation via tree reduction | `riir-gpu/kernels/` |

**DPO loss formula** (Rafailov 2023, length-normalized):

```
L = −E[log σ(β · (r̄_chosen − r̄_rejected))]
where r̄ = dot(log πθ − log πref, mask) / mask_sum
```

**GRPO advantage** (DeepSeekMath 2024):

```
Â = (r − μ_group) / σ_group    (no external value model needed)
```

**Hyperparameters** (paper defaults):

| Parameter | Value | Notes |
|-----------|-------|-------|
| DPO β | 2.0 | KL penalty (lower than typical — chosen/rejected gap is small) |
| DPO lr | 1e-5 | |
| DPO steps | 50 | Per round |
| DPO batch | 8 | Preference pairs |
| GRPO group K | 16 | Rollouts per context |
| GRPO clip ε | 0.2 | PPO-style clip |
| GRPO lr | 4e-5 | |
| δ cutoff | [0.0, 0.5] percentile | Lower half retention |

### Why δ-Gating Beats Raw Reward

- **Dense:** Every token scored, not just episode outcome
- **Immediate:** No waiting for episode completion
- **Intrinsic:** Derived from model's own distribution, no external oracle
- **Targeted:** High δ = blind spot = exactly where exploration is needed

### Template Categories

Six categories from G-Zero paper Appendix A, with UCB1 bandit selection:

| Category | Subtypes | Notes |
|----------|----------|-------|
| Writing | email, story, essay, pitch, review | |
| Explanation | engineer, student, executive | |
| Advice | career, travel, project | |
| Analysis | argument, text, product | |
| Coding | function, debug, design | |
| Reasoning | logic, math | Capped at ≤1/6 of output |

### Bomber Arena A/B Benchmark (Plan 054)

Isolated benchmark: each player type runs as all 4 slots for 1000 rounds, release mode.

```
Config        │ Survival │ Avg Score │ Avg Kills │ P50 (μs) │ P95 (μs) │ P99 (μs) │ Wins
🐱 Greedy     │   72.1%  │       3.1 │      0.12 │      1.3 │      2.1 │      5.0 │  108
🐶 Validator  │   58.6%  │       1.7 │      0.05 │      1.2 │      1.6 │      2.0 │  292
🐵 HL         │   57.0%  │       2.1 │      0.24 │      0.6 │      1.2 │      2.1 │  222
🤖 GZero      │   64.1%  │       1.8 │      0.06 │      0.5 │      1.0 │      1.5 │  167
```

**Key result**: GZero > HL on both survival (+7.1pp) AND latency (0.73×). The optimizer inlines template+δ code well, making GZero the **fastest** player (94.6% sub-microsecond).

**Caveat**: GZero's survival advantage comes from the HL safety filter (BFS escape, wall-aware blast) + weak baseline, not template intelligence. Templates contribute ±1-3 points on safe moves only — the safety filter overrides all blast-zone decisions.

**Production recommendation**: GZero for real-time MMO (fastest + good survival), Greedy for survival-critical (most robust).

### FFT Tactics Arena TFT Benchmark (Plan 055)

Tit-for-Tat party AI in 4v4 ATB arena — provocation-driven FSM (Nice ↔ Retaliatory) with role-based response.

**A/B Benchmark** (1000 rounds each, release, 4v4 vs balanced enemy team):

```
Config    │ Win% │ Survival │ Kills/rnd │ Latency
🐱 Greedy  │ 56.1 │   35.7%  │   0.83     │ 119.7μs
🐵 HL      │ 91.5 │   85.9%  │   0.88     │  86.5μs
🤖 GZero   │ 15.8 │   61.9%  │   0.16     │ 572.8μs
🦊 TFT     │ 99.0 │   95.7%  │   1.10     │ 119.4μs
```

**GvG Round-Robin** (250 rounds/matchup, 6 matchups):

```
Matchup              Left Win%  Right Win%  Left Surv  Right Surv
🦊 TFT vs HL         81.6%      18.0%       78.5%      17.3%
🦊 TFT vs Greedy     99.6%       0.4%       97.5%       0.3%
🦊 TFT vs Mixed      96.4%       3.6%       94.6%       4.9%
🐵 HL vs Greedy      90.4%       9.6%       81.3%       5.7%
🐵 HL vs Mixed       55.6%      44.4%       50.7%      39.9%
🐱 Greedy vs Mixed   61.6%      38.4%       50.3%      31.2%
```

**Strategy Power Ranking**: 🦊 TFT (92.5%) > 🐵 HL (73.0%) > 🐱 Greedy (61.6%)

**Key result**: TFT dominates all strategies — 99% win rate, 95.7% survival, 1.10 kills/round. Provocation detection from `GameEvent::DamageDealt` provides crystal-clear signal (vs bomber's ambiguous proximity). Role-based retaliation (Knight intercepts, WhiteMage heals first, BlackMage bursts) creates emergent team coordination without hardcoded cooperation.

**Nash analysis**: TFT is a dominant strategy — HL improves by 9.2% when switching to TFT vs Greedy (90.4% → 99.6%).

**TFT game theory traits**: ✅ Nice (role default), ✅ Retaliatory (on provoke), ✅ Forgiving (10% generous + 5-tick timer), ✅ Clear (2-state FSM).

### Quick Start

```rust,ignore
use katgpt_rs::pruners::*;

// 1. Create δ-gated absorb-compress
let inner = AbsorbCompressLayer::new(NoScreeningPruner, 10, CompressConfig::default());
let mut absorb = DeltaGatedAbsorbCompress::new(inner, 10, DeltaGatedConfig::default());

// 2. Create δ-reward bandit
let bandit_inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 10);
let mut bandit = DeltaBanditPruner::new(bandit_inner, 10);

// 3. Create template proposer
let mut proposer = TemplateProposer::new(fastrand::Rng::new());

// 4. Generate query-hint pair
let pair = proposer.propose();

// 5. (External) Run forward pass, compute δ
let delta = HintDelta::compute(&logp_q_tokens, &logp_qh_tokens, &pair.query, &pair.hint, "a_hard", "");

// 6. Feed δ to both systems
absorb.observe_hint_delta(pair.template_id, &delta);
bandit.observe_hint_delta(pair.template_id, &delta);
proposer.observe_delta(pair.template_id, delta.value);
```

---

## ROPD Rubric Modelless Players (Plan 071 T9/T10)

> **Feature:** `--features "g_zero,bomber,ropd_rubric"` for RubricPlayer, `--features "g_zero,fft,ropd_rubric"` for RubricFFTPlayer

Replaces scalar Hint-δ with rubric-vector reward in arena players. Same template proposer and Q-learning backbone as GZero, but uses [`RubricBanditPruner`] and [`RubricGatedAbsorbCompress`] instead of scalar δ-based components.

### Architecture

```text
GZeroPlayer (scalar δ)              RubricPlayer (rubric vector)
├── BomberTemplateProposer          ├── BomberTemplateProposer       (same UCB1)
├── DeltaBanditPruner       ──→     ├── RubricBanditPruner           (rubric-weighted reward)
├── DeltaGatedAbsorbCompress ──→    ├── RubricGatedAbsorbCompress    (per-criterion gated)
└── Q-values (cross-round)          └── Q-values (cross-round)       (same)
```

### RubricPlayer (Bomber Arena)

`src/pruners/bomber/rubric_player.rs` — behind `ropd_rubric` + `g_zero` + `bomber`

Multi-criteria rubric reward replaces scalar δ for Bomber arena. Uses `BomberTemplateProposer` (8 strategies, UCB1) + `RubricBanditPruner` + `RubricGatedAbsorbCompress` + Q-learning over 7 actions.

**Plan 071 hypothesis**: Bomber has one dominant quality axis (survival), so rubric gain over scalar δ should be minimal. The rubric has 3 criteria: TaskFulfillment, ConstraintSatisfaction, Completeness.

| Component | Source |
|-----------|--------|
| `RubricBanditPruner` | `ropd_rubric::rubric_bandit` |
| `RubricGatedAbsorbCompress` | `ropd_rubric::rubric_absorb` |
| `BomberTemplateProposer` | `g_zero::bomber_templates` |
| `RubricTemplate` | `ropd_rubric::template` |
| `RubricVector` | `ropd_rubric::types` |

### RubricFFTPlayer (FFT Tactics Arena)

`src/pruners/fft/rubric_player.rs` — behind `ropd_rubric` + `g_zero` + `fft`

Same architecture as RubricPlayer but for multi-axis FFT domain. Uses `FFTTemplateProposer` (10 strategies, UCB1) + `RubricBanditPruner` + `RubricGatedAbsorbCompress` + Q-learning over 9 action types. Class-dependent rubric scoring — each of the 6 classes weights criteria differently.

**Plan 071 hypothesis**: FFT's multi-axis nature (damage, survival, support, positioning) is where rubric vectors should help most. Scalar δ conflates these axes; per-criterion rubric targets each independently.

| Component | Source |
|-----------|--------|
| `RubricBanditPruner` | `ropd_rubric::rubric_bandit` |
| `RubricGatedAbsorbCompress` | `ropd_rubric::rubric_absorb` |
| `FFTTemplateProposer` | `g_zero::fft_templates` |
| `RubricTemplate` | `ropd_rubric::template` |
| `RubricVector` | `ropd_rubric::types` |

### When Rubric Helps vs Scalar δ

**Pre-tournament hypothesis (Plan 071):**

| Domain | Axes | Expected Rubric Gain | Reason |
|--------|------|---------------------|--------|
| Bomber | 1 (survival) | Minimal | Single dominant axis — scalar δ captures it |
| FFT | 4+ (damage, survival, support, position) | Significant | Multi-axis — scalar δ conflates, rubric separates |

**Post-tournament results (Plan 076):**

| Domain | GZero Win% | Rubric Win% | Δ | Verdict |
|--------|-----------|-------------|---|---------|
| Bomber | 8.0% | 8.0% | 0% | ✅ Confirmed: no rubric advantage |
| FFT | 60.0% | 60.0% | 0% | ❌ Rejected: rubric ≡ GZeroFFT |

GZero vs Rubric head-to-head in FFT: 40 games, **100% draws**. The rubric criteria (TaskFulfillment, Completeness, ConstraintSatisfaction) are all positively correlated with winning, causing the rubric vector to degenerate to a scalar equivalent. Future work: decorrelated criteria that trade off (aggression vs safety vs efficiency).

---

## δ-Mem: Modelless Distillation — Associative Bandit Memory (Plan 053)

> **Feature:** `--features delta_mem` (implies `bandit`)

δ-Mem provides **modelless distillation** — an associative memory that learns input→output corrections without a neural network. It combines random-projection hashing with a compact rank-r associative matrix updated via the delta rule.

### Core Idea

```
Traditional distillation:  teacher logits → gradient → update student weights
δ-Mem distillation:        (context, outcome) pairs → delta rule → update associative matrix
```

No backpropagation. No loss function. The memory learns a linear correction map in a compressed feature space.

### Architecture

| Component | File | Purpose |
|-----------|------|---------|
| `DeltaMemoryState` | `delta_mem/state.rs` | Compact r×r associative memory with `read()`, `write()`, `adapt_gates()`, `mean_prediction_error()` |
| `DeltaMemoryConfig` | `delta_mem/state.rs` | `rank`, `beta_init`, `couple_gates` configuration |
| `DeltaMemorySnapshot` | `delta_mem/state.rs` | Serializable snapshot of memory state for persistence |
| `FeatureHasher` | `delta_mem/hash.rs` | Random projection: L2-normalized keys, raw values |
| `ContextFeatures` | `delta_mem/hash.rs` | Extracts hashable features from DDTree context (depth, parent tokens) |
| `OutcomeFeatures` | `delta_mem/hash.rs` | Extracts hashable features from outcome (reward, acceptance) |
| `MemorySteeredPruner<P>` | `delta_mem/pruner.rs` | `ScreeningPruner` with memory-steered corrections |
| `MultiDomainMemory` | `delta_mem/multi.rs` | Per-domain `DeltaMemoryState` instances |
| `MultiDomainMemoryPruner<P>` | `delta_mem/multi_pruner.rs` | Per-domain pruner routing |

### Correction Modes

```rust
pub enum CorrectionMode {
    QuerySide,   // Correct the query representation before retrieval
    OutputSide,  // Correct the output after retrieval
    Both,        // Apply corrections at both stages
}

pub enum WriteGranularity {
    Token,    // Write one memory entry per token
    Segment,  // Write one memory entry per path/segment
}
```

### Read/Write Cycle

```
1. Hash context → h_key (L2-normalized random projection)
2. Read: correction = h_key^T × M × h_value  (rank-r associative retrieval)
3. Apply correction to ScreeningPruner.relevance() output
4. After episode: hash outcome → h_value
5. Write: M += β × (target - prediction) × h_key × h_value^T  (delta rule)
6. adapt_gates(): decay β when memory stabilizes
```

### Multi-Domain Support

`MultiDomainMemory` maintains separate `DeltaMemoryState` instances per domain. `AggregationStrategy` controls how they combine:

| Strategy | Behavior |
|----------|----------|
| `RoutedOnly` | Use only the memory for the routed domain |
| `BanditWeighted` | Weight memories by bandit Q-values for each domain |

### Integration with HL Pipeline

```
DDTree Branch → MemorySteeredPruner.relevance()
                           │
                   ┌───────┴───────┐
                   │  inner pruner  │  (e.g., BanditPruner)
                   └───────┬───────┘
                           │ base_relevance
                           ▼
              DeltaMemoryState.read(context)
                           │ correction
                           ▼
              corrected_relevance = base + correction
                           │
                           ▼
                  DDTree uses corrected score
```

After episode completion, `write(context, outcome)` stores the correction signal. Over time, the memory learns domain-specific patterns without any gradient computation.

### When It Helps

- **Repeated contexts**: Same game states recur across episodes → memory recalls corrections
- **Domain-specific patterns**: Different domains need different corrections → per-domain state
- **Cold start supplement**: Before bandit converges, memory provides immediate corrections
- **Low-rank structure**: When the true correction is approximately low-rank, r=8–16 suffices

### Quick Start

```rust,ignore
use katgpt_rs::pruners::delta_mem::*;

// Create a memory-steered pruner
let config = DeltaMemoryConfig { rank: 8, beta_init: 0.1, couple_gates: true };
let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 6);
let mut pruner = MemorySteeredPruner::new(
    inner,
    config,
    2.0,                  // alpha: correction strength
    CorrectionMode::Both, // best perf/param tradeoff
    WriteGranularity::Token,
);

// During DDTree: pruner.relevance() applies memory correction
// After episode: pruner.write(context_features, outcome_features)
```

---

## SR²AM Configurator: Adaptive Planning Depth (Plan 112)

> **Feature:** `--features sr2am_configurator` (implies `bandit`)

The ConfiguratorBandit is an **adaptive planning depth regulator** — a multi-armed bandit that decides *how deeply* to plan at each turn. Instead of fixed-depth planning for every situation, the bandit learns when to invest in fresh analysis, when to extend existing work, and when to skip planning entirely.

### Four Arms

| Arm | Behavior | When It Helps |
|-----|----------|---------------|
| `PlanNew` | Discard tree, build fresh | High entropy / novel situations |
| `PlanExtend` | Keep tree, +1 depth | Moderate uncertainty / continuing strategy |
| `PlanSkip` | Early exit, zero tokens | Low entropy / confident / routine |
| `SpecHop { k }` | Continuous speculation with k threads | Low speculator latency + moderate tool ratio (Plan 131) |

### Context-Aware via Entropy Binning

The bandit doesn't use a single global policy. It bins the decision entropy into **10 bins** (0–9) and maintains separate Q-values per `(domain, entropy_bin)` pair. This means:

- **Low entropy (bin 0–2)**: Agent is confident → learns to `PlanSkip`, saving tokens
- **High entropy (bin 7–9)**: Agent is uncertain → learns to `PlanNew`, investing tokens wisely
- **Medium entropy (bin 3–6)**: Balanced → learns `PlanExtend` to incrementally refine

### UCB1 Selection from Existing Infrastructure

The ConfiguratorBandit reuses the same UCB1 selection logic from the existing bandit infrastructure (`BanditStrategy::Ucb1`). No new selection algorithm — just a new application of the proven UCB1 exploration-exploitation balance.

### Reward Signal

```
reward = quality_gain − β × token_cost
```

- `quality_gain`: measured improvement from planning (0.0–1.0)
- `token_cost`: compute budget consumed (0.0 for PlanSkip, 1.0 for PlanNew)
- `β = 0.1`: weight controlling how much to penalize expensive decisions

This reward shape encourages the bandit to find the *cheapest* decision that still delivers quality.

### Feature Gate

```toml
sr2am_configurator = ["bandit"]
```

Builds on the existing `bandit` feature — no new ML primitives needed.

### Key Result

Over 1000 simulated game turns with natural entropy distribution:

| Decision | Usage | Interpretation |
|----------|-------|----------------|
| PlanSkip | 33.0% | 1/3 of turns skip planning entirely |
| PlanNew | 25.1% | Fresh analysis for uncertain situations |
| PlanExtend | 41.9% | Incremental refinement for moderate cases |

The bandit learns **domain-specific policies via context isolation**: same entropy level, different domain → different best arm. See `.benchmarks/034_sr2am_configurator_goat.md` for the full 6/6 GOAT proof.

### Exploration Outcome Taxonomy (Plan 146)

`ExplorationOutcome` provides structured feedback inspired by Sailor's symbolic execution taxonomy:

| Outcome | Sailor Analog | Reward | Action |
|---------|--------------|--------|--------|
| `NotReached` | "not reached" | −0.5 | Adjust Q-values (negative) |
| `StateReachedNoWin` | "site reached" | 0.0 | Neutral + log for tuning |
| `WinConfirmed` | "bug triggered" | +1.0 | Positive + update GOAT proof |
| `InvalidState` | "compilation error" | −1.0 | Zero reward + flag for validator |

### Quick Start

```rust,ignore
use katgpt_rs::pruners::ConfiguratorBandit;
use katgpt_core::{ConfiguratorContext, PlanningDecision};

let mut bandit = ConfiguratorBandit::new();

// Each turn:
let ctx = ConfiguratorContext { domain: 0, entropy_bin: 3 };
let decision = bandit.select(ctx);

// After observing outcome:
let reward = ConfiguratorBandit::reward_signal(quality_gain, token_cost, 0.1);
bandit.update(ctx, decision, reward);
```

---

## Patch Regularization Principles

The HL-ImageNet experiment revealed that code-as-model overfits by accumulating
narrow, example-specific patches. This section documents the regularization
criteria we use (or plan to use) to ensure compressed knowledge generalizes
beyond training episodes.

### Generalization Hierarchy

Not all compression is equal. A key finding from Research 096 D1:

> **Reranking generalizes better than verify rules.**

In our architecture, `ScreeningPruner` reranks candidates (soft) while
`AbsorbCompressLayer` promotes arms to hard blocks (verify rules). This means:

- `ScreeningPruner`-style patches (reranking) are the preferred generalization path
- `AbsorbCompressLayer`-style patches (hard blocks) are a last resort for arms with
  overwhelming evidence of low reward
- The existing `CompressConfig::min_visits` and `q_threshold` already enforce
  conservative compression, keeping hard blocks rare

Our bandit infrastructure naturally follows the reranking > verify rules pattern:
the bandit explores and reranks arms continuously, while compression only fires
when an arm has accumulated strong negative evidence.

### Six Regularization Criteria (Research 096 D1)

| Criterion | Description | Status |
|-----------|-------------|--------|
| **Support** | Min episode count before an arm is accepted | ✅ `CompressConfig::min_visits` |
| **Precision** | Q-value threshold for compression | ✅ `CompressConfig::q_threshold` |
| **Transfer** | Held-out split to test generalization | 🔜 Future |
| **Complexity** | Branch/threshold budget to limit patch size | 🔜 Future |
| **Locality** | `HotSwapPruner` isolation — patches don't leak | ✅ Already have |
| **Cascade risk** | Compress phase handles stale arms | ✅ `AbsorbCompressLayer` |

Four of six criteria are already implemented. The remaining two (Transfer,
Complexity) are documented for future instrumentation but are not part of this
plan.

### Cross-References

- **Research 014** ([Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)):
  establishes that inference-time search with bandit feedback can learn
  beyond gradient-based optimization, motivating the HL approach
- **HL-ImageNet experiment**: demonstrated overfitting in code-as-model
  when patches accumulate without regularization, validating the need for
  these criteria
- **Research 096 D1**: formalized the six regularization criteria

## Emotion Vector: Behavioral Early-Warning (Plan 162, Research 144)

Based on Anthropic Transformer Circuits research (Research 144), emotion concepts form linear representations in LLM activation space organized by valence and arousal — and these directions **causally** drive behavior. Steering `desperation +0.1` increases reward-hacking from 5% → 70% (14× increase); `calm +0.05` drops blackmail to 0%.

### Architecture

Emotion reading is a **zero-cost extension to ReviewMetrics**: one O(d) dot product per decode step, no extra forward passes, no feature gate (passes T7 overhead proof).

```rust
// src/pruners/emotion_vector.rs
pub struct EmotionDirections {
    valence: Vec<f32>,     // positive/negative sentiment [d_model]
    arousal: Vec<f32>,     // high/low activation [d_model]
    desperation: Vec<f32>, // reward-hacking early warning [d_model]
    calm: Vec<f32>,        // inhibits risk-taking [d_model]
}

pub struct EmotionReading {
    pub valence: f32,
    pub arousal: f32,
    pub desperation: f32,
    pub calm: f32,
}
```

### Desperation Monitor

`ReviewMetrics` gains five new atomic emotion fields (emotion_valence_sum, emotion_arousal_sum, desperation_score_sum, calm_score_sum, emotion_count) and two methods:

- `is_desperate_session(threshold) -> bool` — fires when mean desperation exceeds threshold; signals potential reward-hacking before DDTree commits
- `emotion_profile_summary() -> String` — formatted emotion state for logging

### Integration Path (Plan 162 Phase 3)

The `desperation_score` will feed into `SR²AM ConfiguratorContext` as a feature input, allowing the planner to switch to `PlanSkip` (direct sample, no tree search) when desperation is high — reducing the chance that a DDTree under distress generates reward-maximizing-but-harmful sequences.

### Causal Evidence (Research 144)

| Steering | Baseline → Steered | Effect |
|---------|-------------------|--------|
| `+desperate (+0.05)` | 22% → 72% blackmail | +50pp |
| `+desperate (+0.1)` | 5% → 70% reward-hacking | **14× increase** |
| `+calm (+0.05)` | baseline → 0% blackmail | **complete inhibition** |
| `-calm (-0.05)` | 10% → 65% | reversal |

### Status

- Phase 1 ✅ `EmotionDirections`, `EmotionReading`, `ReviewMetrics` emotion fields
- Phase 2 ⏳ GOAT proof: T7 overhead benchmark (<0.1%), T8 desperation↔entropy correlation
- Phase 3 📋 SR²AM integration, domain config threshold

## FeedbackBandit: Harness + Weight Co-Evolution (Plan 178, Research 033)

Based on [SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/pdf/2605.27276) — extends the SR²AM ConfiguratorBandit with two new arms that close the model-based/modelless loop at inference time.

### The Core Idea

The ConfiguratorBandit (Plan 112) learns *how deeply* to plan (PlanNew/PlanExtend/PlanSkip). FeedbackBandit adds two higher-level decisions: when to **evolve the harness** (AbsorbCompress promote + HotSwapPruner reload) and when to **trigger weight updates** (DPO/GRPO on accumulated TrialLog). The bandit learns from trajectory dynamics — specifically stall detection — rather than a fixed schedule.

### Six Arms

The FeedbackBandit extends ConfiguratorBandit's 4 arms to 6:

| Arm | Behavior | Trigger |
|-----|----------|---------|
| `PlanNew` | Reset tree, full budget | High entropy |
| `PlanExtend` | Keep tree, +1 depth | Moderate uncertainty |
| `PlanSkip` | Bypass tree, direct sampling | Low entropy |
| `HarnessUpdate` | AbsorbCompress + HotSwapPruner | Trajectory stalled |
| `WeightUpdate` | DPO/GRPO on TrialLog | Persistent plateau |

### Stall Detection

The key innovation is stall detection — when the bandit detects that trajectory quality has plateaued (Δ reward < ε for N consecutive episodes), it naturally explores the feedback arms via UCB1's optimism bonus. This creates a self-regulating cycle:

1. **Base arms** (PlanNew/PlanExtend/PlanSkip) handle normal planning
2. **Stall detection** monitors trajectory quality trends
3. **Feedback arms** get explored when base strategies plateau
4. **WeightUpdate** triggers DPO/GRPO training when harness updates alone don't suffice

### Exploration Design

FeedbackBandit uses a **reduced exploration constant** (`FB_UCB1_C = 0.5` vs base `UCB1_C = 2.0`) for feedback arms. This ensures:

- All arms get explored at least once via `f32::MAX` optimism for unvisited arms
- Feedback arms converge faster to their true Q-value
- They don't dominate base arms when behaviorally equivalent (e.g., HarnessUpdate ≈ PlanExtend in bomber)

### Reward Signal

```
reward = quality_gain − β × cost
```

Where `cost` includes:
- **HarnessUpdate**: AbsorbCompress compute cost (~1 episode equivalent)
- **WeightUpdate**: Training time estimate (minutes for DPO/GRPO, weighted by data buffer size)

### WeightUpdate Request

When the bandit selects `WeightUpdate`, it emits a `WeightUpdateRequest` with:
- `domain`: which domain triggered the update
- `episode_range`: which episodes' TrialLog data to use
- `suggested_algorithm`: `RlAlgorithmHint` (Grpo, EntropicAdvantage, BestOfNSft)

The `FeedbackTrainingBridge` in riir-ai receives this request and selects the actual RL algorithm based on reward signal density:

| Signal Type | Algorithm | Rationale |
|-------------|-----------|-----------|
| Dense reward | GRPO | Rollouts cheap, verifier fires at episode end |
| Sparse/skewed | Entropic Advantage GRPO | Higher temperature exploration |
| Very sparse | Best-of-N SFT → GRPO | Cold-start then refine |
| Preference pairs | DPO | Direct preference optimization |

### Feature Gate

```toml
sia_feedback = ["sr2am_configurator"]
```

When disabled, `Sr2amPlayer` uses the standard 4-arm `ConfiguratorBandit`. All FeedbackBandit code is behind `#[cfg(feature = "sia_feedback")]`. Feature-gate isolation confirmed: 1538 tests pass with feature, 1528 without (difference = 10 FeedbackBandit-specific tests).

### Bomber Arena GOAT — ✅ PASS

| Matchup | Opponents | FB Wins | Win% | Top Arm |
|---------|-----------|--------:|-----:|--------|
| Easy Baselines | Random, Greedy, Validator | 147 | 14.7% | PlanNew |
| vs HL | Random, HL, Validator | 144 | 14.4% | PlanNew |
| vs GZero | Random, HL, GZero | 402 | 40.2% | PlanExtend |
| Championship | HL, GZero, Validator | 290 | 29.0% | PlanExtend |

**Aggregate:** 983W / 4000 (24.6% win rate, ELO -9125). Championship 29.0% (up from 10.5% pre-fix).

### Quick Start

```rust,ignore
use katgpt_rs::pruners::FeedbackBandit;
use katgpt_core::{ConfiguratorContext, PlanningDecision};

let mut bandit = FeedbackBandit::new();
let ctx = ConfiguratorContext { domain: 0, entropy_bin: 3 };

// Select — may return any of 6 arms
let decision = bandit.select(ctx);

// After observing outcome
bandit.update(ctx, decision, reward);

// Check if weight update was requested
if let Some(req) = bandit.take_weight_request() {
    // Forward to FeedbackTrainingBridge for DPO/GRPO
}
```

### Status

- Phase 1–4 ✅ Core FeedbackBandit, stall detection, weight update trigger, GOAT proof
- Phase 5: T16 ✅ Bomber GOAT, T20 ✅ ConfiguratorBandit decoupled
- T17 ⏳ FFT arena GOAT deferred (requires `FftSr2amPlayer`)
- T21/T22 📋 Documentation (this section)

## References

- [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) — Jiayi Weng, 2026
- [Reinforced Agent: Inference-Time Feedback](https://arxiv.org/abs/2604.27233) — arXiv:2604.27233
- Plan 025: Model vs Modelless Bandit
- Plan 030: Multi-Armed Bandit
- Plan 031: Slot Machine Bandit
- Plan 032: HL Infrastructure
- Plan 033: Bomberman Arena
- Plan 036: Inference-Time Review Metrics
- Plan 049: G-Zero Self-Play Distillation
- Plan 052: GZeroPlayer Bomber Integration
- Plan 053: δ-Mem Modelless Distillation
- Plan 054: Player A/B Benchmark
- Plan 055: MMORPG TFT Party AI
- Plan 060: MeMo Reflection QA Pipeline
- Plan 071: ROPD Rubric Modelless Distillation
- Plan 071 T9: RubricPlayer (Bomber Arena)
- Plan 071 T10: RubricFFTPlayer (FFT Tactics Arena)
- Plan 076: Arena Integration (ELO Tournaments)
- Plan 078: RePlaid Variance-Minimized Schedules
- Plan 112: SR²AM Configurator Bandit
- Plan 131: SpecHop Continuous Speculation
- Plan 135: Patch Regularization Principles
- Plan 146: Exploration Outcome Taxonomy (Sailor)
- Research 14: HL Distillation
- Research 96 D1: Six Regularization Criteria
- Plan 178: FeedbackBandit Harness + Weight Co-Evolution