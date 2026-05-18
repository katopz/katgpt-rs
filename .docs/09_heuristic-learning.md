# microgpt-rs: Heuristic Learning

> **Status (Plan 049):** G-Zero self-play distillation — both phases complete. Phase 1 (modelless): `HintDelta`, `DeltaGatedAbsorbCompress`, `DeltaBanditPruner`, `TemplateProposer` behind `--features g_zero` (implies `bandit`). Phase 2 (model-based): `LengthNormalizedDPO`, `GRPO`, `DeltaFilter` (6-stage), `GZeroLoop` in `riir-gpu` (Plan 059, 3,369 lines, 76 tests, 2 DPO WGSL kernels). See `src/pruners/g_zero/` and `riir-gpu/src/`.
>
> **Status (Plan 061):** Entropy anomaly detection — `ReviewMetrics` now tracks session-level entropy (mean, max, count) via `record_entropy()` and `is_high_entropy_session()`. `DeltaMemoryState::mean_prediction_error()` exposes drift signal. Behind `--features bandit,delta_mem`. See `.plans/061_entropy_anomaly_detection.md`.
>
> **Status (Plan 036):** ReviewMetrics, ReviewStrategy, and benefit-ratio gating are implemented behind `--features bandit`. AbsorbCompress gates compression by benefit-risk ratio. `ppot_rescue_reviewed` provides structured review loops behind `--features bandit,ppot`. See example `review_01_metrics`.
>
> **Status (Plan 071):** ROPD Rubric modelless distillation — `RubricVector`, `RubricTemplate`, `RubricGatedAbsorbCompress`, `RubricBanditPruner` behind `--features ropd_rubric` (implies `bandit`). Per-criterion gap targeting replaces scalar δ with structured multi-criteria reward. `RubricPlayer` (bomber, `g_zero`+`bomber`) and `RubricFFTPlayer` (FFT, `g_zero`+`fft`) integrate rubric reward into arena players (Plan 071 T9/T10). Benchmark: 5.3M observe_rubric/sec, 20/20 targeting accuracy, zero regression. See `.benchmarks/007_ropd_rubric_modelless.md`.
>
> **Status (Plan 072):** SDAR Gated distillation modelless — `sdar_gate()`, `SdarBanditPruner`, `SdarGatedAbsorbCompress` behind `--features sdar_gate`. Asymmetric trust: sigmoid gate σ(β·x) endorses positive gaps, attenuates negative. β=5.0 paper-validated. Benchmark: 118M updates/sec, zero hot-path overhead, 97.5% targeting accuracy. See `.benchmarks/008_sdar_gated_modelless.md`.
>
> **Status (Plan 032):** TrialLog, AbsorbCompress, HotSwapPruner, and RegressionSuite are implemented behind `--features bandit`. See examples `hl_01_trial_log` and `hl_02_hotswap`.

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

## HL in microgpt-rs

microgpt-rs is uniquely positioned for HL because of its **trait-based pruner architecture** and **WASM sandbox**:

| HL Concept | microgpt-rs Component |
|---|---|
| Heuristic Policy | `ConstraintPruner::is_valid()` — masks invalid tokens/actions |
| Relevance Scoring | `ScreeningPruner::relevance()` — prioritizes good actions |
| Gradient-free Learning | `BanditPruner` — Q-value updates without backprop |
| Sandboxed Heuristics | `WasmPruner` — compiled validators in WASM sandbox |
| Trial History | `TrialLog` — persistent JSONL episode records |
| Rule Compression | `AbsorbCompress` — promote stable Q-values to hard constraints |
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

---

## The Two Operations

A healthy Heuristic System needs two operations (from the HL research):

### 1. Absorb

Feed new observations back into the system:

```
Episode N:   BanditPruner selects arm → environment runs → reward
             TrialLog.append(episode, arm, reward, q_value, note)
             AbsorbCompress.absorb(arm, reward)
```

### 2. Compress

Fold accumulated knowledge into simpler, more maintainable rules:

```
After N episodes:
  arm 3 (Wait) has Q=0.02 over 500 visits → promote to hard block
  arm 0 (Attack near enemy) has Q=0.89 → boost relevance weight
  → AbsorbCompress.compress() returns [3]
  → BanditPruner delegates arm 3 to BlockedArmPruner
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
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐      │
│  │  TrialLog  │ │AbsorbCompress│ │RegressionSuite│    │
│  │  (JSONL)   │ │ (Q→blocks)  │ │ (golden)     │      │
│  └────────────┘ └────────────┘ └──────────────┘      │
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
6. AbsorbCompress.absorb(arm, reward) for compression check
```

### Between Episodes (Compression)

```
1. AbsorbCompress.should_compress() → true (threshold met)
2. AbsorbCompress.compress() → identify arms to promote
3. Low-Q arms → BlockedArmPruner (hard constraints)
4. High-Q arms → boost relevance weight
5. RegressionSuite.replay_golden() → verify no regression
```

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
- **AbsorbCompress**: automatic rule promotion from experience
- **RegressionSuite**: safety net against regressions
- **Coding agent**: writes new validators based on failure analysis

---

## Bomberman HL Arena (Proof of Concept)

4 AI players compete in a Bomberman arena built with `bevy_ecs` (standalone) + ratatui emoji TUI. Game logic patterns adapted from `raw/bomby/` (Fish Folk: Bomby) — same ECS components, resources, and systems, but tick-based instead of real-time.

### Architecture

```
raw/bomby/ (reference)              →  microgpt-rs bomberman (ours)
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
#[derive(Component)] struct GridPos { x: i32, y: i32 }
#[derive(Component)] struct BombFuse { owner: Entity, ticks_remaining: u32 }
#[derive(Component)] struct BombRange { cells: u32 }
#[derive(Component)] struct BombCount { max: u8, active: u8 }
#[derive(Component)] struct Speed { cells_per_tick: u8 }
#[derive(Component)] struct Alive;
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

```rust
use microgpt_rs::pruners::{
    BanditPruner, BanditStrategy, AbsorbCompress, TrialLog, CompressConfig,
};

// Create a bandit pruner with absorb-compress
let mut bandit = BanditPruner::new(
    domain_screener,
    BanditStrategy::Ucb1,
    6, // 6 arms (actions)
);

// Create trial log
let mut trial_log = TrialLog::new("/tmp/hl_trials.jsonl")?;

// Run episodes
for episode in 0..1000 {
    let arm = bandit.best_arm();  // select via strategy
    let reward = env.pull(arm);   // environment feedback
    
    bandit.update(arm, reward);
    trial_log.append(TrialRecord {
        episode, arm, reward,
        q_value: bandit.q_value(arm),
        cumulative_reward: bandit.total_pulls() as f32 * reward,
        cumulative_regret: 0.0,
        config: String::new(),
        note: String::new(),
    });
    
    // Absorb-compress check every 100 episodes
    if episode % 100 == 0 && bandit.should_compress() {
        let promoted = bandit.compress();
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
use microgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv, ReviewMetrics};

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
use microgpt_rs::pruners::*;

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

| Domain | Axes | Expected Rubric Gain | Reason |
|--------|------|---------------------|--------|
| Bomber | 1 (survival) | Minimal | Single dominant axis — scalar δ captures it |
| FFT | 4+ (damage, survival, support, position) | Significant | Multi-axis — scalar δ conflates, rubric separates |

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
use microgpt_rs::pruners::delta_mem::*;

// Create a memory-steered pruner
let config = DeltaMemoryConfig { rank: 8, beta_init: 0.1, couple_gates: true };
let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 6);
let mut pruner = MemorySteeredPruner::new(inner, config, 6);

// During DDTree: pruner.relevance() applies memory correction
// After episode: pruner.write(context_features, outcome_features)
```

---

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
- Plan 071: ROPD Rubric Modelless Distillation
- Plan 071 T9: RubricPlayer (Bomber Arena)
- Plan 071 T10: RubricFFTPlayer (FFT Tactics Arena)
- Research 14: HL Distillation