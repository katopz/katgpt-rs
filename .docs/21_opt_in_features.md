# Opt-In & Gated Features — Full Detail

> These features are proven and tested but opt-in (not in default feature set).
> See main README for the default GOAT stack. Each feature is behind a feature flag.

## 1. D2F: Discrete Diffusion Forcing (Plan 066)

Block-parallel decoding via iterative denoising — a third decode strategy alongside autoregressive and speculative. Feature-gated behind `dllm`.

- **Block-causal attention**: bidirectional within block, causal across blocks → existing KV cache works
- **`D2fContext`**: pre-allocated flat buffers, zero `Vec<Vec<f32>>` per denoising step
- **`D2fPipeline`**: multi-block sequential decode with KV cache commit across blocks
- **`DecodeStrategy::DiscreteDiffusion`**: config-driven auto-switch heuristic (AR → Speculative → D2F)

📖 See [`.docs/03_speculative_decoding.md`](03_speculative_decoding.md) for D2F API details and [`.research/034_D2F_Discrete_Diffusion_Forcing.md`](../.research/034_D2F_Discrete_Diffusion_Forcing.md) for experimental results.

### Tri-Mode: D2F+AR Self-Speculation (Plan 089)

D2F drafts in parallel → AR verifies causally → accept longest prefix match. Feature-gated behind `tri_mode` (requires `dllm`).

- **`D2fDrafterVerifier`**: `d2f_decode_block()` drafts → `forward()` verifies → prefix accept + bonus token
- **`DecodeStrategy::SelfSpeculation`**: D2F+AR mode, auto-selected by `recommend()` when draft model available
- **Global Loss Averaging**: `LossAveraging::Global` (Nemotron +2.12% accuracy vs per-sequence)
- **`DiffusionSampler`**: per-position correctness predictor replaces fixed confidence threshold — Logistic (AUC 0.765) / MLP (AUC 0.781) vs fixed baseline 0.343 (Plan 116, Bench 019)
- **GOAT 9/9 passed**: Tri-Mode 4/4 (Bench 018) + DiffusionSampler 5/5 (Bench 019) + Natsukaze validation 100.0% accuracy

📖 See [`.benchmarks/018_d2f_verifier_goat.md`](../.benchmarks/018_d2f_verifier_goat.md) and [`.benchmarks/019_diffusion_sampler_goat.md`](../.benchmarks/019_diffusion_sampler_goat.md) for full GOAT proof results.

## 2. SR²AM Configurator Bandit (Plan 112)

Distilled from [SR²AM: Self-Regulated Simulative Reasoning](https://arxiv.org/pdf/2605.22138) (Deng, Hou, Sá Neves et al., 2026). Bandit-based per-turn planning regulation — learns when to plan deep, extend, or skip entirely.

### Adaptive Planning Decisions

| Decision | When | Effect |
|----------|------|--------|
| `PlanNew` | High uncertainty, new sub-problem | Reset tree, full budget allocation |
| `PlanExtend` | Moderate uncertainty, continuing | Keep tree, +1 depth level |
| `PlanSkip` | Low uncertainty, confident | Bypass tree, direct token sampling |

### Context-Aware UCB1 Selection

```text
Context: (domain, entropy_bin)
  → ConfiguratorBandit selects arm via UCB1
  → Reward: quality_gain − β × token_cost
```

Entropy binning (10 bins via `floor(entropy * 10.0)`) provides coarse context — low entropy → `PlanSkip`, high → `PlanNew`.

### Uncertainty-Aware Horizon Truncation

High-uncertainty states cap `draft_lookahead` at 2 (SR²AM finding: web tasks benefit from short horizons). Configurable via `max_plan_horizon` override.

### Feature Gate

`sr2am_configurator = ["bandit"]` — default-on. All new code behind feature flag. `InferenceResult` extended with `planning_decision` and `plan_horizon_used` metrics.

🧪 `tests/test_sr2am_configurator_goat.rs` — 29 integration tests (arm selection, context isolation, entropy truncation, pipeline wiring)

📖 See [`.plans/112_sr2am_configurator_bandit.md`](../.plans/112_sr2am_configurator_bandit.md) for full plan.

## 3. FeedbackBandit — Harness + Weight Co-Evolution (Plan 178)

Distilled from [SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/pdf/2605.27276). Extends the SR²AM ConfiguratorBandit (4 arms) with 2 new arms that close the model-based/modelless loop. The bandit learns when to trigger harness hot-swaps and weight updates based on trajectory dynamics, not a fixed schedule.

### Six Arms

| Arm | Behavior | When It Helps |
|-----|----------|---------------|
| `PlanNew` | Discard tree, build fresh | High entropy / novel situations |
| `PlanExtend` | Keep tree, +1 depth | Moderate uncertainty / continuing |
| `PlanSkip` | Early exit, zero tokens | Low entropy / confident |
| `SpecHop { k }` | Continuous speculation, k threads | Fast speculator + tool-bound workload |
| `HarnessUpdate` | AbsorbCompress promote + HotSwapPruner reload | Trajectory stalled, new heuristic needed |
| `WeightUpdate` | Trigger DPO/GRPO on TrialLog buffer | Persistent plateau, model refinement needed |

### Architecture

```text
FeedbackBandit extends ConfiguratorBandit:
  Base arms (SR²AM):      PlanNew, PlanExtend, PlanSkip, SpecHop
  New arms (SIA):         HarnessUpdate, WeightUpdate
  Selection:              UCB1 over (domain, entropy_bin) context
  Exploration:            FB_UCB1_C = 0.5 (reduced) for faster feedback arm convergence
  Reward:                 quality_gain − β × cost
  Stall detection:        Δ reward < ε for N episodes → triggers feedback arm exploration
```

### Bomber Arena GOAT — ✅ PASS

**Setup:** 4 matchups × 1000 games = 4000 total, `Sr2amPlayer` with `sia_feedback` (6 arms) vs baselines.

| Matchup | Opponents | FB Wins | Win% | Top Arm |
|---------|-----------|--------:|-----:|--------|
| Easy Baselines | Random, Greedy, Validator | 147 | 14.7% | PlanNew |
| vs HL | Random, HL, Validator | 144 | 14.4% | PlanNew |
| vs GZero | Random, HL, GZero | 402 | 40.2% | PlanExtend |
| Championship | HL, GZero, Validator | 290 | 29.0% | PlanExtend |

**Aggregate:** 983W / 4000 games (24.6% win rate, ELO -9125). FB arms explored: 20 (HarnessUpdate=16, WeightUpdate=4).

### Feature Gate

`sia_feedback = ["sr2am_configurator"]` — **opt-in**. FeedbackBandit manages own 6-arm UCB1; ConfiguratorBandit remains unchanged at 4 arms when feature is off. All new code behind feature flag. 10 FeedbackBandit tests + 15 ConfiguratorBandit tests pass independently.

🧪 `examples/bomber_17_feedback_goat.rs` — 4000-game arena GOAT regression proof

📖 See [`riir-ai/.plans/178_sia_feedback_bandit.md`](../../riir-ai/.plans/178_sia_feedback_bandit.md) for full plan.

## 4. SpecHop — Continuous Multi-Hop Speculation (Plan 131)

Hop-level speculative execution for multi-step tool-use agents. Based on [arXiv:2605.21965](https://arxiv.org/pdf/2605.21965) — continuous speculation at trajectory granularity (not token level).

### How It Works

```text
Agent trajectory:  [hop₁] → [hop₂] → [hop₃] → [hop₄]
                        ↘ spec    ↘ spec    ↘ spec
                     Thread k=1   k=2       k=3       k=4
                        ↓          ↓          ↓          ↓
                  Verify earliest pending → Commit ✓ or Rollback ✗
```

The pipeline maintains **k speculative threads** that predict tool-call observations ahead of actual tool responses. When the target tool returns, a verifier checks equivalence → commit correct branch, rollback incorrect ones.

### Theoretical Cost Model

| Parameter | Meaning | Formula |
|-----------|---------|---------|
| α | Speculator latency ratio | `E[T_spec] / E[T_target]` |
| β | Decode-to-tool ratio | `E[T_seg] / E[T_target]` |
| p | Speculator hit rate | Fraction of correct predictions |
| k* | Optimal threads | `⌈(1+β)/(α+β)⌉` (Theorem 2) |
| RelLat* | Oracle latency | `1 − p(1−α)/(1+β)` (Theorem 3) |

Example: α=0.2, β=0.15, p=0.7 → k*=4, RelLat*=0.513 (1.95× speedup).

### SR²AM Integration

`PlanningDecision::SpecHop { k }` arm added to the configurator bandit (Plan 112). Auto-activated when:
- α < 0.3 (fast speculator)
- β < 0.5 (tool-bound workload)
- `reward = latency_reduction / α > 1.0`

### Hop-Level DDTree

`build_hop_dd_tree()` extends the token-level DDTree concept to hop granularity. Each node is an (action, observation) pair scored by speculator confidence. `verify_hop_tree()` wires `ObservationVerifier` for branch accept/reject.

### Module Structure

```text
src/spechop/
├── mod.rs              # Module index, re-exports, feature gate
├── types.rs            # SpecHopConfig, HopObservation, SpecOutcome, HopState
├── cost_model.rs       # α/β/p → k*, RelLat, starvation probability
├── verifier.rs         # ObservationVerifier trait + RuleBasedVerifier
├── speculator.rs       # HopSpeculator trait + CacheSpeculator + BanditSpeculator
├── window.rs           # SpecWindow k-bounded thread manager
├── pipeline.rs         # SpecHopPipeline continuous loop (Algorithm 1)
├── hop_tree.rs         # Hop-level DDTree integration
└── segment_match.rs    # Rolling hash sub-sequence matching (Plan 140 T19, behind cache_prune+spechop)
```

### Examples

```bash
cargo run --example spechop_01_pipeline --features spechop   # 4-hop continuous speculation
cargo run --example spechop_02_cost_model --features spechop  # α/β/p → k* and RelLat
```

🔧 Feature flag: `spechop = ["bandit"]` (**opt-in** — requires GOAT proof before default-on promotion)

📖 See [`.plans/131_spechop_continuous_spec_pipeline.md`](../.plans/131_spechop_continuous_spec_pipeline.md) for full plan (T1–T32, T40–T41 complete).

## 5. Parallel-Probe 2D (Plan 133)

Training-free 2D probing controller for N parallel reasoning branches. Based on [arXiv:2602.03845](https://arxiv.org/pdf/2602.03845) — monitors branches via periodic answer extraction, uses **consensus-based early stopping** + **deviation-based branch pruning** to reduce sequential tokens by ~30%.

The key insight: **answer-level consensus across parallel branches is O(N) per probe step** — uniquely cheap compared to EqR distribution residuals (O(N×V)) or trajectory bandit scores (requires reward signal).

```text
Parallel Branch 0: ...think...think... → "42"
Parallel Branch 1: ...think...think... → "42"  ← consensus!
Parallel Branch 2: ...think...think... → "17"  ← deviant, prune after k steps
                     ↑
              Probe every Δ tokens
              → majority vote → stop if stable for u steps
              → prune branches that disagree for k steps
```

### Components

| Component | Purpose |
|-----------|----------|
| `ParallelProbeController<A>` | Generic controller: probe(), majority_vote(), should_stop(), should_prune() |
| `ProbeDecision` | Continue / Stop / Prune / StopAndPrune |
| `AnswerExtractor` trait | Pluggable answer extraction (regex, think-token, game actions) |
| `RegexAnswerExtractor` | `\boxed{...}`, "The answer is ...", numeric patterns |
| `ThinkTokenExtractor` | `</think`> boundary detection |
| `DiscreteActionExtractor` | Game domain actions (Bomber, Go moves) |
| `ParallelProbeVerifier<V>` | Wraps any `SpeculativeVerifier` with probe control |

26 unit tests covering: consensus detection, deviation pruning, warmup suppression, all answer formats, integer/generic answer types.

🔧 Feature flag: `parallel_probe` (**default-on**)

📖 See [`.plans/133_parallel_probe_2d_probing.md`](../.plans/133_parallel_probe_2d_probing.md) for full plan.

## 6. GFlowNet Modelless Distillation (Plan 052)

Distills the GFlowNet shortest-path theorem — **minimize flow = shortest paths** — into the existing ScreeningPruner + BanditPruner + DDTree stack **without any neural network training**.

**Core insight:** The paper proves that minimizing expected trajectory length `E[nτ]` forces the backward policy `P_B` to assign zero probability to all non-shortest paths. Our stack already computes forward marginals (LoRA logits = P_F), backward relevance (WASM validator = P_B), and flow proxy (BanditPruner Q-values = F(s)). We harmonize these signals.

### Four Additive Distillations

| Distillation | Component | What It Does |
|-------------|-----------|-------------|
| **D1: FlowPruner** | `FlowPruner<P: ScreeningPruner>` | Wraps any screener, adds `λ × (1 - stop_prob[depth])` flow bonus |
| **D2: Balanced DDTree** | `build_dd_tree_balanced()` | Scores beams with `ln(P_llm) + w × ln(R) + λ × flow_bonus` |
| **D3: Flow-weighted bandit** | `observe_delta_with_flow()` | Adds `λ_length / prefix_len` trajectory length bonus to δ reward |
| **D4: Backward replay** | `ReplayBackwardWalker` | Walks winning replays backward, finds safe alternatives = P_B data |

### Benchmark Results (NoScreeningPruner baseline)

| Metric | Result |
|--------|--------|
| FlowPruner node delta | **+0.0%** ✅ |
| Balanced DDTree backward compat | **Identical to `build_screened`** ✅ |
| Flow-weighted bandit reward delta | **+0.0%** ✅ |
| Backward replay alternatives | **4.0 avg/tick** (target: ≥2) ✅ |

Run: `cargo test --features "bandit,g_zero,bomber" --test bench_gflownet_modelless -- --nocapture`

📖 See [`.plans/052_gflownet_modelless_distillation.md`](../.plans/052_gflownet_modelless_distillation.md) for full plan, [`.research/023_GFlowNet_Shortest_Paths.md`](../.research/023_GFlowNet_Shortest_Paths.md) for paper analysis.

## 7. ROPD Rubric Modelless Distillation (Plan 071)

Distills ROPD's rubric-based scoring into our modelless stack. Replaces scalar [`HintDelta`](#-g-zero-verifier-free-self-play-plan-049) with structured [`RubricVector`] — multi-criteria reward without LLM judges. Template rubrics + pattern scorers provide per-criterion scoring at inference speed (~µs).

### Key Innovation: Per-Criterion Gap Targeting

- **Scalar δ**: `gate = mean_delta > threshold` (blind — *why* did it trigger?)
- **Rubric**: `gate = any(high_weight_criterion_gap > threshold)` (targeted — "constraint #2 failed")

### Multi-Reference Requirement

ROPD ablation (Table 6): m=4→m=1 costs **−17.94 pts** — the single biggest impact. Single reference over-anchors rubric to one trajectory. Always use M ≥ 2 references.

### Benchmark Results (`.benchmarks/007_ropd_rubric_modelless.md`)

| Method | Throughput | Hot-path overhead |
|--------|-----------|-------------------|
| `observe_rubric()` (bomber) | 4.9M/sec | — |
| `observe_rubric()` (generic) | 5.3M/sec | — |
| `RubricBanditPruner::observe_rubric()` | 14.1M/sec | — |
| `relevance()` (absorb) | — | ~0% (inlined) |
| `relevance()` (bandit) | — | -2.7% (inlined) |

| Targeting | Detected | Expected |
|-----------|----------|----------|
| High-weight gaps (w=4.0) | 20/20 | ✅ All |
| Low-weight gaps (w=1.0) | 0/10 | ✅ Filtered |
| No-gap arms | 0/55 | ✅ Excluded |

**Feature gate:** `ropd_rubric = ["bandit"]` — off by default.

## 8. VPD — Variational Policy Distillation

EM-style co-evolutionary teacher-student distillation that actively trains the feedback-conditioned teacher via BCO (Binary Cross-Entropy Optimization).

- **E-step (every F=5 rounds)**: BCO refines teacher Q-values from unpaired outcome preferences
- **M-step (every round)**: KL-gated distillation of teacher → student with dynamic prior
- **Dynamic prior**: Student Q tracks teacher Q via soft update (η=0.2), breaking SDAR plateau
- **+6.3% win rate over SDAR** in fixed-seed bomber tournament (38.0% vs 31.7%)
- **Non-degrading** in varied-seed arena (within 2.3% of SDAR over 1000 games)

Feature gate: `vpd_em_distill` (requires `sdar_gate`, `bandit`)

```rust
use katgpt_rs::pruners::vpd_em::{VpdConfig, VpdEmCycle};
use katgpt_rs::pruners::bomber::VpdPlayer;

// Create VPD player with paper defaults
let player = VpdPlayer::new(0);

// Or customize: F=5, β=0.1, λ=0.1, dynamic prior
let config = VpdConfig::default();
let player = VpdPlayer::with_config(0, config);
```

Paper: arXiv:2605.15113 — Variational Policy Distillation (Salesforce AI Research, 2026)

## 9. Committee Boost (Plan 132)

Four diagnostics from the [boosting committee paper](https://arxiv.org/pdf/2605.14163) that our DDTree + BtRank + ScreeningPruner stack already supports conceptually but lacked as measurable metrics:

| Diagnostic | What It Measures | Our Stack Mapping |
|------------|-----------------|-------------------|
| **Oracle-gap recovery** `Rec = (p_system - p1) / (p_oracle - p1)` | How much latent capability the selector recovers | `ConstraintPruner` measures selection vs coverage failure |
| **Position-swap debiasing** | Eliminates lead-position bias in BtRank | `DebiasedComparator` wraps pairwise comparison |
| **Budget sizing** (Theorem 3) | Given (α₀, β₀, σ₀, L, δ) → optimal (k, m, r) | Sizes DDTree width, ScreeningPruner depth, BtRank votes |
| **Blind-spot floor** `B = 1 - lim_{k→∞} p_oracle(k)` | Proposer diversity ceiling | CoverageDiagnostic recommends action |

The paper proves our stack IS the committee protocol Π_{k,m,r}. These additions make the theoretical guarantees **measurable and actionable**.

### GOAT Proof Results (`.benchmarks/020_committee_boost_goat.md`)

Run: `cargo test --features committee_boost --test bench_committee_boost_goat -- --nocapture`

| Proof | Description | Verdict |
|-------|-------------|--------|
| G1 | Oracle-gap recovery: Rec within ±0.01 for 6 known cases | ✅ |
| G2 | Debiased comparison: 100% Tie rate for biased comparator | ✅ |
| G2b | Debiasing catches lead-position bias (false rankings eliminated) | ✅ |
| G3 | Budget sizing: Theorem 3 monotonicity + determinism | ✅ |
| G3b | Budget rejects all invalid parameters | ✅ |
| G4 | Blind-spot floor: 8 cases verified (B estimation, convergence, diagnostics) | ✅ |
| G5 | End-to-end: committee improves ≥5% over single-shot | ✅ |

### Key API

```rust,ignore
use katgpt_rs::pruners::committee_boost::{
    OracleGapRecovery, FailureMode, DebiasedComparator, CommitteeBudget,
    committee_budget, estimate_blind_spot_floor, coverage_diagnostic,
};

// Oracle-gap recovery
let r = OracleGapRecovery::new(0.5, 0.8, 0.74);
let rec = r.recovery(); // Some(0.8)
let mode = r.failure_mode(); // CoverageLimited
let diag = r.diagnostic(); // "Recovery=80.0% (coverage-limited); ..."

// Debiased BtRank comparison
let comparator = DebiasedComparator::new(|i, j| biased_compare(i, j));
let comparisons = comparator.tournament(4); // Vec<BtComparison>

// Budget sizing (Theorem 3)
let budget = committee_budget(10, 0.05, 0.3, 0.2, 0.4, 2)?;
println!("k={}, m={}, r={}", budget.k, budget.m, budget.r);

// Blind-spot floor
let rates = vec![(1, 0.5), (2, 0.65), (4, 0.75), (8, 0.8)];
let b = estimate_blind_spot_floor(&rates); // 0.2
let diag = coverage_diagnostic(&rates);
println!("B={:.3}, action={}", diag.blind_spot_floor, diag.action);
```

### Module Structure

```
src/pruners/committee_boost/
    mod.rs               ← Module index, re-exports
    types.rs             ← OracleGapRecovery, FailureMode
    debiased_compare.rs  ← DebiasedComparator, debiased_compare
    budget.rs            ← CommitteeBudget, committee_budget
    blind_spot.rs        ← BlindSpotEstimate, coverage_diagnostic
tests/
    bench_committee_boost_goat.rs  ← 7-proof GOAT benchmark
```

**Feature gate:** `committee_boost = ["bt_rank", "bandit"]` — **opt-in**.

📖 See [`.research/093_Boosting_Weak_Reasoning_Committee_Search.md`](../.research/093_Boosting_Weak_Reasoning_Committee_Search.md) for the paper distillation.
