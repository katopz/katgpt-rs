# Opt-In & Gated Features — Full Detail

> These features are proven and tested but opt-in (not in default feature set).
> See main README for the default GOAT stack. Each feature is behind a feature flag.

## 1. D2F: Discrete Diffusion Forcing (Plan 066)

Block-parallel decoding via iterative denoising — a third decode strategy alongside autoregressive and speculative. Feature-gated behind `dllm`.

- **Block-causal attention**: bidirectional within block, causal across blocks → existing KV cache works
- **`D2fContext`**: pre-allocated flat buffers, zero `Vec<Vec<f32>>` per denoising step
- **`D2fPipeline`**: multi-block sequential decode with KV cache commit across blocks
- **`DecodeStrategy::DiscreteDiffusion`**: config-driven auto-switch heuristic (AR → Speculative → D2F)

📖 See [`.docs/02_inference/speculative_decoding.md`](../02_inference/speculative_decoding.md) for D2F API details and [`.research/034_D2F_Discrete_Diffusion_Forcing.md`](../../.research/034_D2F_Discrete_Diffusion_Forcing.md) for experimental results.

### Tri-Mode: D2F+AR Self-Speculation (Plan 089)

D2F drafts in parallel → AR verifies causally → accept longest prefix match. Feature-gated behind `tri_mode` (requires `dllm`).

- **`D2fDrafterVerifier`**: `d2f_decode_block()` drafts → `forward()` verifies → prefix accept + bonus token
- **`DecodeStrategy::SelfSpeculation`**: D2F+AR mode, auto-selected by `recommend()` when draft model available
- **Global Loss Averaging**: `LossAveraging::Global` (Nemotron +2.12% accuracy vs per-sequence)
- **`DiffusionSampler`**: per-position correctness predictor replaces fixed confidence threshold — Logistic (AUC 0.765) / MLP (AUC 0.781) vs fixed baseline 0.343 (Plan 116, Bench 019)
- **GOAT 9/9 passed**: Tri-Mode 4/4 (Bench 018) + DiffusionSampler 5/5 (Bench 019) + Natsukaze validation 100.0% accuracy

📖 See [`.benchmarks/018_d2f_verifier_goat.md`](../../.benchmarks/018_d2f_verifier_goat.md) and [`.benchmarks/019_diffusion_sampler_goat.md`](../../.benchmarks/019_diffusion_sampler_goat.md) for full GOAT proof results.

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

📖 See [`.plans/112_sr2am_configurator_bandit.md`](../../.plans/112_sr2am_configurator_bandit.md) for full plan.

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

📖 See [`riir-ai/.plans/178_sia_feedback_bandit.md`](../../../riir-ai/.plans/178_sia_feedback_bandit.md) for full plan.

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

📖 See [`.plans/131_spechop_continuous_spec_pipeline.md`](../../.plans/131_spechop_continuous_spec_pipeline.md) for full plan (T1–T32, T40–T41 complete).

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

📖 See [`.plans/133_parallel_probe_2d_probing.md`](../../.plans/133_parallel_probe_2d_probing.md) for full plan.

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

📖 See [`.plans/052_gflownet_modelless_distillation.md`](../../.plans/052_gflownet_modelless_distillation.md) for full plan, [`.research/023_GFlowNet_Shortest_Paths.md`](../../.research/023_GFlowNet_Shortest_Paths.md) for paper analysis.

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

📖 See [`.research/093_Boosting_Weak_Reasoning_Committee_Search.md`](../../.research/093_Boosting_Weak_Reasoning_Committee_Search.md) for the paper distillation.

## 10. Induced CWM (Plan 296)

Open half of the Code World Models Super-GOAT, distilled from [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) (Lehrach et al., DeepMind Oct 2025). A generic, IP-free trait surface for LLM-induced forward-model implementations that are **verifiable** (transition unit tests), **committable** (BLAKE3 over canonical bytes), and **hot-swappable** (atomic slot). The kernel primitive is shipped open in `katgpt-core`; the LLM-induction pipeline itself is private (riir-ai Plan 326).

The primitive exists to let downstream consumers (Bomber, Go, NPC domains, custom IIGs) plug in induced forward models behind a stable trait boundary — `InducedCwmKernel: GameState` — without coupling to any specific induction recipe.

- **`induced_cwm`** — `InducedCwmKernel: GameState` marker + `CwmCommitment` (BLAKE3) + `BeliefInferenceFn<S>` + `TransitionUnitTest` + `verify_transition` (Plan 296 Phase 1).
- **`induced_cwm_ismcts`** (requires `induced_cwm`) — Information-Set MCTS over an induced CWM + belief fn: `ismcts_search_with_inference<S, B>` + `InformationSet` + `NodeStats` (Plan 296 Phase 2).
- **`induced_cwm_tournament`** (requires `induced_cwm`) — Value Function Tournament: round-robin arena-play selector over `StateHeuristic` candidates, `ValueFnTournament<S, V>` + `PlayerStats` + `TournamentWinner` (Plan 296 Phase 3).

Phase 4 ships `InducedCwmSlot<K>` — lock-free atomic hot-swap slot for live kernel replacement (under the `induced_cwm` feature).

**GOAT 4/4 PASS** (all gates green, see [`.benchmarks/296_induced_cwm_primitive_goat.md`](../../.benchmarks/296_induced_cwm_primitive_goat.md)):

| Gate | Target | Verdict |
|------|--------|--------|
| **G1** Verifiability | 100% pass on known-correct transitions; correct diff on mutation | ✅ PASS |
| **G2** Play strength | ISMCTS picks non-fold ≥ 70% when P(strong) ≥ 0.6 | ✅ PASS |
| **G3** Latency | `advance()` ≤ 10 µs/call on mock CWM | ✅ PASS (~1–5 ns, ~3 orders of magnitude under budget) |
| **G4** Commitment integrity | Same logical kernel → identical BLAKE3 across 10 re-runs | ✅ PASS |

The primitive stays **opt-in by design** — it's a primitive surface, not a default-on capability; downstream pipelines opt in by enabling the feature. **Ready for downstream consumption** (riir-ai Plan 326).

### Examples

```bash
cargo run --example induced_cwm_01_mock_iig            --features induced_cwm_ismcts        # Phase 2: mock Leduc IIG + ISMCTS
cargo run --example induced_cwm_02_value_tournament    --features induced_cwm_tournament     # Phase 3: value-fn tournament arena
```

🔧 Feature flags: `induced_cwm`, `induced_cwm_ismcts` (deps `induced_cwm`), `induced_cwm_tournament` (deps `induced_cwm`) — all **opt-in**.

📖 See [`.plans/296_induced_cwm_kernel_primitive.md`](../../.plans/296_induced_cwm_kernel_primitive.md) for the plan, [`.research/275_Code_World_Model_Induced_Forward_Model.md`](../../.research/275_Code_World_Model_Induced_Forward_Model.md) for the paper distillation, [`.benchmarks/296_induced_cwm_primitive_goat.md`](../../.benchmarks/296_induced_cwm_primitive_goat.md) for the GOAT proof (G1–G4 all PASS).

## 11. HLA Windowed Eigenbasis Recovery (Issue 001)

Per-NPC eigenbasis recovery from a windowed HLA activation matrix — **modelless**, no LAPACK, no training. Power iteration with deflation on the D×D Gram `W^T W` (D = HLA dim, 8 today) recovers the top-`k` orthogonal principal directions of a single NPC's recent affective trajectory. Those eigenvectors are the right singular vectors `V` of `W`; their eigenvalues are `σ²`. The recovered basis is a per-NPC rotation/projection matrix usable for emotion routing, zone attention, or adapter selection — every NPC currently shares the same hand-tuned universal basis (Research 032); this exposes individualized affective geometry from the NPC's *own* experience.

The deterministic seed is `1/sqrt(D)` (no RNG), mirroring `stable_rank_update_into` — the same cross-platform determinism surface.

Three entry points serve three operating points:

| Entry point | Path | When to use |
|------------|------|-------------|
| `recover_eigenbasis_from_window` | cold (BLAKE3 + `Uuid::now_v7` provenance) | freeze/thaw cache validation |
| `recover_eigenbasis_from_window_fast` | cold-start (no provenance, rebuilds Gram) | first-time recovery from a stored window |
| `EigenbasisTracker` | plasma-tier hot path (incremental Gram, O(D²)/tick) | live NPC, one push + one recover per tick |

**GOAT gate PASS (synthetic, 2026-06-30)** — see [`.benchmarks/001_hla_eigenbasis_recovery_goat.md`](../../.benchmarks/001_hla_eigenbasis_recovery_goat.md):

| Gate | Target | Verdict |
|------|--------|--------|
| **G1** Latency (`EigenbasisTracker` hot path) | ≤ 2 µs/tick, T=512 D=8 k=4 | ✅ PASS (613.9 ns/tick, 3.25× margin) |
| **G2** Determinism (same-binary) | 0 bit diffs | ✅ PASS (cross-platform protocol in `tests/hla_eigenbasis_determinism.rs`) |
| **G3** Quality (reconstruction error) | < 0.10, k=4, rank-3 ground truth | ✅ PASS (0.0003, 333× margin) |
| **G4** Behavioral divergence | > 50% of 1000-NPC pairs cos < 0.7 | ✅ PASS (87.8%) |
| **G5** Memory (per-NPC) | ≤ 256 bytes at D=8, k=4 | ✅ PASS (144 bytes, 1.78× margin) |

**Opt-in by design.** The issue's GOAT outcome requires a head-to-head against Research 032's hand-tuned axes + a private `riir-ai` architectural guide before promotion to default — both cross the repo boundary and are tracked as `riir-ai` follow-ups. The stateless path (~9 µs) and full provenance path (~17 µs) are reported for transparency; only the `EigenbasisTracker` hot path is the G1 budget path.

**Sync-boundary compliant** (per AGENTS.md): the recovered eigenbasis stays local to the NPC — never synced, never crosses `LatCalFixed`/`SyncBlock`, never used for anti-cheat. `EigenbasisProvenance.window_hash` is a cache key, not a synced value.

🔧 Feature flag: `hla_eigenbasis_recovery` — **opt-in**.

📖 See [`.benchmarks/001_hla_eigenbasis_recovery_goat.md`](../../.benchmarks/001_hla_eigenbasis_recovery_goat.md) for the full GOAT proof and the G1 three-path latency breakdown.

## 12. Canvas Schema Compiler (Plan 419)

A typed `CanvasSchema` compiler that lowers a declared region layout + directed topology into a sparse `AttentionMaskSpec` (consumable by AC-Prefix / VortexFlow / any sparse-attention path), a per-position `LossWeightMask`, and a **reachability** primitive proving **exact marginal independence for binary masks** — absent edge ⟹ no influence, by construction. Plus a `transfer_distance` semantic-type compatibility scalar (`1 − cosine` of frozen embeddings, schema-ABI check from paper §2.4 Table 1).

**Modelless by construction** (Plan 419, Research 398, Valdez *Canvas Engineering* July 2026): every primitive is a pure function over index sets + graphs. Zero backprop, zero weight mutation. The compiler ships on **structural / correctness** merits — the reachability guarantee is provable by construction (like the DEC `d∘d=0` identity, Plan 251), NOT on the paper's behavioral headline (1.73× parameter efficiency, cortical R²=0.825), which is **training-dependent** (`.issues/043` fusion PoC resolved-and-removed 2026-07-09, inconclusive; see Research 398 §7–8).

**Direction convention (paper §2.2):** `Connection(src, dst)` licenses `src` to query `dst` keys/values; information flows `dst → src`; the information-flow graph `G` has arc `dst → src`; `can_reach(from, to)` therefore reads as "`from` influences `to`". `causal_chain([A,B,C])` emits each region querying its predecessor → info arcs `A → B → C` → `can_reach(A, C, 2) == true` (Plan 419 T3.6).

**GOAT gate — ✅ PASS (all G1–G6)**, see [`.benchmarks/419_canvas_schema_goat.md`](../../.benchmarks/419_canvas_schema_goat.md):

| Gate | Target | Verdict |
|------|--------|--------|
| **G1** Reachability soundness (LOAD-BEARING) | absent edge ⟹ `can_reach == false` ∀ horizons | ✅ PASS (exact marginal independence by construction) |
| **G2** Horizon bound (T3.6) | `can_reach(A,C,1)=false`, `can_reach(A,C,2)=true` | ✅ PASS |
| **G3** No-regression | `--all-features` + `--no-default-features` clean | ✅ PASS |
| **G4** Alloc-free hot path | `TransitiveClosure::reaches` + `reachability_horizon` = 0 allocs/call | ✅ PASS (0/1000 reaches, 0/1000 horizon) |
| **G5** Perf | `compile_schema` (199-region ICU schema) < 10 ms; `reaches` p50 < 100 ns | ✅ PASS (compile = **1515 ns** (6600× under); reaches p50 = **0 ns**) |
| **G6** Feature isolation | `canvas_schema` gates all symbols; 0 bytes when disabled | ✅ PASS |

**What the GOAT does NOT claim** (the honesty): behavioral parity with the paper's training-dependent results. Applying a declared-topology mask to a frozen untrained-for-it backbone is a documented 19% loss (paper §5 calibration #2). The modelless primitive ships the *compilation* + the *guarantee*; the *behavioral gain* requires riir-train (`.issues/043` fusion PoC resolved-and-removed 2026-07-09, inconclusive).

Module split (AGENTS.md `< 2048` line rule): `canvas/{mod,types,mask,reachability,transfer}.rs`.

🔧 Feature flag: `canvas_schema` — **opt-in** (promotion deferred; `.issues/043` fusion PoC resolved inconclusively, constituents already default-on with runtime consumers — see Research 398 §8).

📖 See [`.benchmarks/419_canvas_schema_goat.md`](../../.benchmarks/419_canvas_schema_goat.md) for the full GOAT proof + the direction-convention derivation.

## 13. Multi-scale V-cycle on Cell Complexes (Plan 413)

Fills the multi-scale composition gap in the shipped single-complex DEC operators (`exterior_derivative`, `codifferential`, `hodge_laplacian`, `hodge_decompose`). Those handle one resolution level; `htno_v_cycle` composes two (fine → coarse → fine): restrict a fine vertex cochain to a coarse complex, apply a caller-supplied coarse operator, prolongate back — the classic multigrid V-cycle on DEC cochains.

**GOAT gate:** G1 (commutativity) `dₖKc ∘ Rₖ = Rₖ₊₁ ∘ dₖK` verified on induced sub-complexes; G2 (perf) restrict/prolongate cheaper than rebuilding the complex; G3 (no-regression) clean with/without feature; G4 (alloc-free) `htno_v_cycle_into` zero bytes beyond pre-allocated scratch.

The 2×2 aggregation coarsening is documented as **non-commuting** (its coarse edges are long-range, not fine edges) — the V-cycle still provides coarse smoothing, but it is a smoother, not a d-commuting transfer.

🔧 Feature flag: `htno_v_cycle` — **opt-in** (in `katgpt-dec`). Forwarded through `katgpt_core::dec::htno_v_cycle`.

📖 Plan: [`.plans/413_multiscale_v_cycle_primitive.md`](../../.plans/413_multiscale_v_cycle_primitive.md)

## 14. HLA Committed-Belief π-Sensitivity Probe (Plan 414)

A modelless diagnostic that perturbs the committed `π` weights of a `CommittedFieldBlend`, re-evaluates the blend map, and measures output drift against an on-the-fly theoretical **π-sensitivity Lipschitz bound** (`L_π = max_j (1/τ)·σ_j·(1−σ_j)·‖f_j(z)‖`). A bound violation flags a numerics bug in the committed blend.

**Key design correction:** the cached `CommittedFieldBlend::lipschitz_bound` computes the **z-sensitivity** bound, not the π-sensitivity bound. The F4 probe computes its own on-the-fly π-bound using the actual `‖f_j(z)‖` — so it catches bugs even when a field under-reports its Lipschitz constant.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Lipschitz bound holds | 1000/1000 random configs, 0 violations | ✅ PASS |
| **G2** | Bug detection (NaN → reject) | NaN in π → `accepted=false` | ✅ PASS |
| **G3** | No regression | 13/13 existing tests pass | ✅ PASS |
| **G4** | Zero-alloc hot path | 0 allocs/1000 calls | ✅ PASS |
| **G5** | Latency | p50 = 3.042µs (target <5µs) | ✅ PASS |

DRY refactor extracts `apply_blended_with_pi` free function shared by production + probe.

🔧 Feature flag: `hla_committed_belief_probe` — **opt-in** (diagnostic/self-verifier, no runtime consumer yet). F4 fusion follow-up from Plan 406 (renoise-CE).

📖 Plan: [`.plans/414_hla_committed_belief_lipschitz_probe.md`](../../.plans/414_hla_committed_belief_lipschitz_probe.md)

## 15. Within-Class Effective Rank (Plan 415)

Class-conditioned collapse diagnostic: the entropy-based effective rank of the **within-class residual** covariance matrix (arXiv:2412.19419 §5.3.1). Fusion of two shipped halves never combined: `effective_rank` (class-agnostic) + `within_class_adjacency` / `between_class_adjacency` (class-conditioning from `riir-ai/crates/riir-engine/src/latent_functor/quality_gate.rs`).

Fills the gap where the class-agnostic `effective_rank` cannot distinguish "between-class variance dominates, within-class collapsed" from "all variance is healthy and isotropic". The existing Dirichlet-energy quality gate measures *separation* (between > within) but not *within-class subspace health*.

**Key insight:** effective rank is scale-invariant — tiny-but-isotropic within-class variance still gives high rank; the low-rank signal requires rank-deficient within-class structure, not just small-magnitude variance.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | `r_WC ∈ [1, min(d, n−C)]`, monotone | 3 tests pass | ✅ PASS |
| **G2** | Non-redundancy vs global `effective_rank` (load-bearing) | within ≈ 0, global ≈ 3 | ✅ PASS |
| **G3** | No regression | 1385 tests pass | ✅ PASS |
| **G4** | Latency | within-class 232µs (0.485× of global) | ✅ PASS |

Not UQ-bearing, not Super-GOAT (Q2 fails — better diagnostic for existing class, not new class).

🔧 Feature flag: inherits `sink_aware_attn` (same gate as sibling `effective_rank`). **Opt-in** — stays alongside its sibling.

📖 Plan: [`.plans/415_within_class_effective_rank.md`](../../.plans/415_within_class_effective_rank.md)

## 16. Cochain Point Sampler (Plan 422)

Continuous intra-primitive cochain field sampler that answers "what is the cochain value at continuous point `p` inside cell `Ω`?" with local-coordinate conditioning. The modelless LPPN *input* computation — Whitney/de-Rham reconstruction turning a discrete `CochainField` into a continuously-queryable field.

Ships quad (2D grid, bilinear λ-weights) and triangle (mesh, barycentric sort + CDF remap) samplers with local-coordinate augmentation (`sin/cos` harmonics for quad, barycentric sort-CDF for tri). The barycentric sort enforces C⁰ continuity across triangle edges (vertices listed in arbitrary order per face).

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Linear-precision exactness | 1250 points, all < 1e-5 | ✅ PASS |
| **G2** | Partition-of-unity (Σλ = 1, λ ≥ 0) | both quad + tri | ✅ PASS |
| **G3** | C⁰ continuity across boundaries | 0 diff | ✅ PASS |
| **G4** | Zero-alloc steady state | 0 allocs on `*_into` paths | ✅ PASS |
| **G5** | Latency | 11.2 ns/call on 64×64 grid | ✅ PASS |

🔧 Feature flag: `cochain_point_sampler` — **opt-in** (in `katgpt-dec`). Gain-tier — substrate-completeness primitive, not a default-path improvement.

📖 Plan: [`.plans/422_cochain_point_sampler_primitive.md`](../../.plans/422_cochain_point_sampler_primitive.md), Research: [`.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md`](../../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md), Paper: [arXiv:2506.22899](https://arxiv.org/abs/2506.22899)

## 17. Spectral Rewiring (Plan 423)

The modelless SAR kernel: project a weight delta onto the base matrix's SVD subspace, extract the compact rewiring matrix M, reconstruct the purified on-manifold delta ΔW*. Reuses `thin_svd_into` from `subspace_phase_gate` (Plan 301).

**Stays opt-in** because the spectral concentration assumption (G1b) is unvalidated without real training deltas — a generic delta is NOT concentrated (0.12–0.18). Promotion to default is blocked on Issue 123 (real-delta test). The SVD 64-col cap (Issue 124) blocks 128×128/512×512. The cached-index path (`SpectralRewireIndex`) is the recommended hot-loop API.

**LLM-scale escape hatch CLOSED (Issue 151, 2026-07-15).** The SAR concentration phenomenon was the foundational assumption for the SAR × QuasiMoTTo Pass@k fusion (Issue 151) — the hypothesis was that concentration holds at LLM scale (4096×4096 weight matrices) even though it failed at NPC scale (Issue 123). A 1.5B-scale Phase 1 PoC **refuted** concentration: 0/196 layers exceeded the 0.8 `on_manifold_fraction` threshold. The LLM-scale regime does NOT rescue the concentration assumption. See negative_results §10.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1a** | SVD recovery | ~8e-6 | ✅ PASS |
| **G1b** | Spectral concentration at NPC-scale | 0.12–0.18 (NOT concentrated) | ❌ UNVALIDATED |
| **G3** | Determinism | bit-identical | ✅ PASS |
| **G4** | Zero-alloc | 0 allocs | ✅ PASS |
| **G5** | Latency | 0.41µs NPC-scale (cached-index) | ✅ PASS |

Cross-repo applications (freeze/thaw purification, spectral LoRA, spectral TIES) are noted as follow-ups but NOT implemented in this plan.

🔧 Feature flag: `spectral_rewire` — **opt-in** (blocked on Issue 123 real-delta validation).

📖 Plan: [`.plans/423_spectral_rewire_primitive.md`](../../.plans/423_spectral_rewire_primitive.md)

## 18. GDN Rollback-Free Tree Verification (Plan 424)

Verifies speculative draft trees against GDN (Gated DeltaNet) recurrent layers **without rolling back the recurrent state**. The algorithm (arXiv:2607.06763 §3.4) extends the chunked delta-rule recurrence to tree-structured drafts via a partial order (ancestor relation), reducing verification to a masked triangular solve `(I + X)U = βV` followed by an ancestor-masked output read.

Fills a confirmed gap: katgpt-rs ships GDN2 (Plan 105, default-on) and KV-cache snapshot/rollback tree verification for attention models (Plan 012), but has **no tree verification for GDN/delta-rule recurrent layers**. Includes multi-head batching + QwenDeltaNet hybrid integration (attention layers use per-branch sequential KV-rollback; DeltaNet layers use tree verify).

**Chain tree speedup matches paper's B200 GPU numbers on CPU SIMD**:

| Tree size T | Speedup | Paper B200 |
|---|---|---|
| T=16 | **1.93×** | 1.5× |
| T=32 | **2.79×** | 2.7× |
| T=64 | **4.66×** | 4.6× |
| T=128 | **7.09×** | 7.1× |

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Bit-exact vs per-branch sequential verify | within 1e-3 (f32 accumulation) | ✅ PASS |
| **G2** | ≥2× faster at T=32, ≥4× at T=64 | 2.79× / 4.66× / 7.09× | ✅ PASS |
| **G3** | No regression | 1429 tests pass | ✅ PASS |
| **G4** | Alloc-free hot path | 0 allocs steady-state | ✅ PASS |

Phase 6 (DDTree argmax-of-marginal tuning) produced a **negative result** — the paper's §3.5 insight does not transfer to best-first tree building (best-first search already prioritizes the argmax path naturally).

🔧 Feature flag: `gdn_tree_verify` — **opt-in** (complement to Plan 012's attention verify; only relevant for `QwenDeltaNet` / GDN-layer configs).

📖 Plan: [`.plans/424_gdn_tree_verification_primitive.md`](../../.plans/424_gdn_tree_verification_primitive.md), Research: [`.research/407_Trees_from_Marginals_GDN_Tree_Verify.md`](../../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md), Benchmark: [`.benchmarks/424_gdn_tree_verify_goat.md`](../../.benchmarks/424_gdn_tree_verify_goat.md), Paper: [arXiv:2607.06763](https://arxiv.org/abs/2607.06763)

## 19. Interpolation Geometry — iMAUVE + Intervention Battery (Research 445)

Modelless evaluation methodology for committed latent substrates — answers "does the *midpoint* of two committed latents decode to a coherent intermediate behavior?" Distilled from Prabhudesai & Geng, *Latent Thought Flows with Text Compression* (Jun 2026). The paper's headline metric **iMAUVE** (nearest-neighbor midpoint interpolation quality) predicts downstream generation quality with Pearson r=0.99; the **5-way intervention probe** (matched/shuffled/zero/mean/noise) extends Plan 278's binary FaithfulnessProbe to per-entity committed state.

Generic `LatentSpace` trait abstracts over the **six committed-latent substrates** cataloged in Research 445 §2.6: HLA `[f32;8]`, `NeuronShard::style_weights[64]`, `ArchetypeBlendShard.pi`, `KarcShard.wout`, `ZoneGeometryPod`, `MerkleFrozenEnvelope`-versioned states. Pure evaluation methodology — NOT a training primitive.

**Three-pressure audit (all six substrates PASS):**
- **Q1 (summarize-vs-route)** — does the latent summarize the underlying trajectory, or is it a lookup key? Subsample the trajectory, recompute the latent, measure divergence. A summarizing latent is stable under subsampling.
- **Q2 (runtime-depends-on-latent)** — does runtime behavior actually use the committed latent, or bypass via raw state? Zero/shuffle the latent (intervention battery), measure behavior delta.
- **Q3 (local-context-vs-bypass)** — does the runtime's attention to the latent stay local? Structural code audit (decode-path purity + consumer-input audit + locality-mechanism inventory). SpKv (Plan 070) + RTPurbo (Plan 126) enforce locality at the transformer-attention layer.

🔧 Feature flag: `interpolation_geometry` (in `katgpt-core`) — **opt-in**. Pure evaluation methodology, no runtime consumer.

📖 Research: [`.research/445_Latent_Thought_Flows_Text_Compression.md`](../../.research/445_Latent_Thought_Flows_Text_Compression.md), Benchmark: [`.benchmarks/456_interpolation_geometry_goat.md`](../../.benchmarks/456_interpolation_geometry_goat.md), Doc: [`.docs/04_calibration/interpolation_geometry.md`](../04_calibration/interpolation_geometry.md), Source: [latent-thought.vercel.app](https://latent-thought.vercel.app) (Prabhudesai & Geng 2026 blog + MeanFlow arXiv:2601.22158)

## 20. GRAPE-M Rank-2 Rodrigues Exponential (Research 446)

Closed-form application of `exp(n·ω·L)` for an arbitrary rank-2 skew-symmetric generator `L = abᵀ − baᵀ ∈ so(d)` (arXiv:2512.07805 §2.3). Uses the Rodrigues formula `I + (sin s / s)·L + ((1 − cos s) / s²)·L²` with `s = ω·‖a∧b‖`, evaluated as `O(d)` work via two inner products `⟨a,x⟩`, `⟨b,x⟩` (no materialized `L` or `L²`).

Generalizes `phase_rotation`'s scalar-broadcast 2D rotation (canonical basis special case where `a = e_i`, `b = e_{i+D/2}`). Pure modelless float arithmetic on a user-supplied plane `(a, b)`.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Bit-identical to materialized `expm(L)` on random `(a,b,ω)` | 0.0 diff | ✅ PASS |
| **G2** | Latency `< 2× phase_rotation_gate_into` | within bound | ✅ PASS |
| **G4** | Alloc-free | 0 allocs | ✅ PASS |

🔧 Feature flag: `grapem_rodrigues` (in `katgpt-core`) — **opt-in**. `Rank2Plane` retains `a, b` as `Box<[f32]>` (not just the 4 scalars) — mathematically necessary for the projections.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/457_grapem_rodrigues_goat.md`](../../.benchmarks/457_grapem_rodrigues_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 21. Unified PositionGroupAction Trait (Research 446)

Abstract trait unifying five position-encoding families under one `G(n) = exp(n·ω·L)` interface (arXiv:2512.07805 §2.2 + §4.1):

- **RoPE** (`SO(d)` multiplicative action) — wraps `PositionFreeCompactor`'s math
- **ALiBi / FoX / Wall** (`GL(d+2)` unipotent lift — additive bias family)
- **NoPE** (trivial `L = 0`)
- **GRAPE-M** (rotary generalization — wraps `Rank2Plane`)

All obey the exact relative law `G(t−s) = G(s)^T·G(t)`, enabling position-encoding-agnostic tooling (KV compaction, attention matching). Hot-path code keeps using `PositionFreeCompactor` / `WallDiagonalGate` directly; the trait is for cold-path interop.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G3** | No-regression — existing RoPE/Wall paths unchanged when feature off | bit-identical | ✅ PASS |

🔧 Feature flag: `position_group_action` (in `katgpt-core`, implies `grapem_rodrigues`) — **opt-in**. 19 unit tests in-crate.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/458_position_group_action_goat.md`](../../.benchmarks/458_position_group_action_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 22. GRAPE-AP Vector-Similarity Gates (Research 446)

Content-aware extension of Wall Attention's scalar prefix-sum gates (arXiv:2512.07805 §5). For each head `h` and decoding step `t`, the bias from key position `j` to query `t` is a path integral of edge potentials:

```
b_h(t, j) = Σ_{ℓ=j+1}^{t} ψ_h(t, ℓ)
ψ_h(t, ℓ) = α · g(⟨p_t, R_ℓ · p_ℓ⟩ / d)
```

with `g = log_sigmoid` (the paper's choice) and `R_ℓ = exp(ℓ·J)` a cached rotation schedule. Tokens whose positional embedding matches the query's decay slower. **Wall is the scalar special case** (endpoint-independent embeddings).

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G2** | Latency `< 1.5×` Wall's scalar path | within bound | ✅ PASS |
| **G4** | Alloc-free after scratch init | 0 steady-state allocs | ✅ PASS |
| **G5** | Direction-check per paper's 1/d normalization | verified | ✅ PASS |

🔧 Feature flag: `grape_ap_vector` (in `katgpt-core`) — **opt-in**. 15 unit tests in-crate.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/459_grape_ap_vector_goat.md`](../../.benchmarks/459_grape_ap_vector_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 23. GRAPE Joint Lift — GL(d+2) Block-Diagonal Composition (Research 446)

Composes rotary (GRAPE-M) + additive (GRAPE-A) into a single block-diagonal group action per Appendix E of arXiv:2512.07805. One-pass `score_into`:

```
score(q, k) = q^T · exp(m · ω_rot · L) · k / √d  +  m · ω_add · (softplus(v · q / √d) + softplus(u · k / √d))
```

Closes the GRAPE composition story: today Wall *replaces* RoPE in our stack; this primitive proves they *compose* into a single one-parameter subgroup of `GL(d+2)` while preserving the exact relative law. The decoupled `omega_rot` / `omega_add` is a strict generalization of the paper's shared `ω`.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Bit-identical to manual composition + relativity law | 0.0 diff | ✅ PASS |
| **G2** | Latency smoke | within bound | ✅ PASS |
| **G4** | Alloc-free after `new` | 0 steady-state allocs | ✅ PASS |

🔧 Feature flag: `grape_joint_lift` (in `katgpt-core`, implies `grapem_rodrigues`) — **opt-in**.

📖 Research: [`.research/446_GRAPE_Group_Representational_Position_Encoding.md`](../../.research/446_GRAPE_Group_Representational_Position_Encoding.md), Benchmark: [`.benchmarks/460_grape_joint_lift_goat.md`](../../.benchmarks/460_grape_joint_lift_goat.md), Paper: [arXiv:2512.07805](https://arxiv.org/abs/2512.07805)

## 24. KARC Family — Delay-Basis Ridge Forecaster + Mitigations + Eigensolver (Plan 308 + Plan 556 + Issues 186/187)

The KARC family is the open-primitive surface for Kolmogorov-Arnold Reservoir Computing (Huang/Kurths/Tang 2026, arXiv:2606.19984). One core forecaster + one large-d_h eigensolver path (Issues 186/187) + three Plan 556 mitigations (regime gate, batched matvec, LOD tier) that address KARC's structural cons (periodic-blindness, crowd-scale per-NPC cost, tiered compute) without changing the core algorithm. All six features are opt-in; the core forecaster's promotion is blocked on the G1 threshold leg (10% short at K=8/M=8/R=2 d_h=18_720).

### 24.1 Core Forecaster — `karc_forecaster` (Plan 308)

`KarcForecaster<D, M, K>` × sealed `KarcBasis` trait (Fourier/Chebyshev/BSpline shipped) × closed-form ridge readout `Wout = YH^T(HH^T + λI)^{-1}`. Phase 2 ships higher-order R=2 (pair-product features, paper Eq. 32) + chunked Gram + ALS low-rank `Wout ≈ A·B` (the form that persists into a `KarcShard` in riir-neuron-db).

| Gate | Target | Result (Phase 5.1, 2026-07-20) | Verdict |
|------|--------|------------------------------|---------|
| **G1 NRMSE** | double-scroll Table I ≤ 1.0×10⁻³ (paper: 5.3×10⁻⁴) | **9.43e-4** (K=8/M=8/R=2 d_h=18_720, λ=5e-2 λ-sweep) | ✅ PASS |
| **G1 threshold** | ≥ 8 Lyapunov times | 2.85 LT (K=4) / **7.23 LT** (K=8/M=8/R=2, 10% short) / 8.16 LT (Phase 1 K=8/M=24 first-order) | ❌ FAIL |
| **G2** | ≤ 500 ns/call forecast (HLA config) | 381 ns | ✅ PASS |
| **G3** | zero-alloc `forecast_into` | 0 allocs | ✅ PASS |
| **G4** | bit-reproducibility | byte-identical `Wout` | ✅ PASS |

**Phase 5 history (load-bearing negative result):** Phase 4 *interpolated* (without measuring) that K=8/M=8/R=2 d_h=18_720 would be the smallest config to pass both G1 legs. Phase 5 measured it directly: BOTH legs FAILED at λ=5e-3 — NRMSE 6.68e-3 (6.7× miss) because the K=8 system is heavily underdetermined (N=4050 samples, d_h=18_720 features → ≥14_670 zero eigenvalues; λ=5e-3 tuned for K=4 is too small to regularize K=8). Phase 5.1 ran the λ-sweep and recovered NRMSE at λ=5e-2 (10× larger). The threshold leg remains flat across λ (~7.0–7.2 LT) — confirming it's a capacity/delay problem, not a regularization problem.

**Compute blocker resolved.** Before Issue 186 Path B, d_h=18_720 was projected infeasible (~6 h via Jacobi eigendecomp). Householder+QL + full-rank direct Cholesky brought it to ~29 min wall. Any future config sweep is now cheap to test.

**Promotion paths (all open, all cheap to test):**
1. **K=10/M=8/R=2 at λ=5e-2** (~28 min Cholesky). Linear K-extrapolation from K=4=2.85 LT, K=8=7.23 LT predicts ~8.5 LT — PASS.
2. **Issue 186 Path D gate re-spec.** Promote on two-config evidence (Phase 5.1 K=8 NRMSE 9.43e-4 + Phase 1 K=8/M=24 threshold 8.16 LT — same K=8 delay length).
3. **More training data** (N=20_000+) — would make the Gram full-rank. Compute cost scales linearly.

**UPDATE (Phase 22, 2026-07-21): PROMOTED TO DEFAULT-ON.** All three compute paths above were tested (Phase 5.2 K=10 + Phase 5.3 R=1 K=8/M=24 λ-sweep) and none produced a single-config gate pass — the compound gate is structurally infeasible. The gate re-spec (Issue 186 Path D variant D3 — split-config gate) was accepted. See [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md) §Phase 5.3 for the full evidence. This section is retained for historical context; `karc_forecaster` now ships in the default feature set.

🔧 Feature flag: `karc_forecaster` — **DEFAULT-ON** (Phase 22, 2026-07-21).

📖 Plan: [`.plans/308_karc_delay_basis_ridge_forecaster.md`](../../.plans/308_karc_delay_basis_ridge_forecaster.md), Research: [`.research/288_KARC_Delay_Basis_Ridge_Forecaster.md`](../../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md), Benchmark: [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md), Paper: [arXiv:2606.19984](https://arxiv.org/abs/2606.19984)

### 24.2 Large-d_h Eigensolver — `karc_householder_eig` + `karc_householder_eig_par` (Issues 186 + 187)

The ALS B-step's original eigendecomp (`karc::jacobi_eigen`, O(d_h³·n_sweeps)) is infeasible at d_h > ~5000 — blocking Plan 308's K=8/M=24/R=2 config (d_h=18_720). Issue 186 Path B swaps in `linalg::symmetric_eig` (Householder tridiag + implicit-shift QL), which is ~5-10× faster at d_h ≥ 256 and feasible at d_h=18_720. The eigensolver is always compiled as a generic `linalg` primitive; the feature gates only the wiring in `karc::large_dh`.

**Measured speedup** (single-threaded, release build):

| n | Householder+QL | Jacobi | Speedup |
|---|---|---|---|
| 64 | 310 µs | 2.5 ms | 7.92× |
| 128 | 3.4 ms | 36.6 ms | 10.62× |
| 256 | 73.5 ms | 687 ms | 9.35× |
| 512 | 794 ms | 10.9 s | 13.69× |

`karc_householder_eig_par` (Issue 187) adds a row-parallel rayon variant. Four row-parallel hot loops (Householder matvec, rank-2 update, Q accumulation, QL eigenvector rotation) parallelize across rows via `par_chunk_mut(n)`; each row's work is fully sequential so the result is bit-identical to the serial path. **Landed a critical QL convergence fix** for near-singular Grams (the NR-local check `|e[m]| + dd == dd` cannot deflate tiny-eigenvalue matrices; added the LAPACK `dsteqr` global-scale criterion — affects both serial and parallel paths).

**Why both stay opt-in despite T1-T6 PASS.** The Phase 5 G1 measurement at d_h=18_720 showed the full-rank direct Cholesky path is BOTH faster and more accurate than Householder+QL for the actual G1 measurement (direct Cholesky ~22 min vs parallel eigendecomp ~87 min; NRMSE 6.68e-3 vs ALS-rank-8 4.71e-3 — 28× worse). The parallel path landed a critical bug fix that ships regardless (it affects the serial Householder path too), but there is no passing G1 gate to promote against. The serial `karc_householder_eig` path stays opt-in for the same reason.

🔧 Feature flags: `karc_householder_eig` (implies `karc_forecaster`) — **opt-in**; `karc_householder_eig_par` (implies `karc_householder_eig`) — **opt-in**.

📖 Benchmark: [`.benchmarks/308_karc_goat.md`](../../.benchmarks/308_karc_goat.md) §Phase 5 (issues 185/186/187 resolved + removed; resolution captured here).

### 24.3 Regime Gate — `karc_regime_gate` (Plan 556 Phase 1)

Closed-form residual-MSE mux between `KarcForecaster` (chaotic-regime specialist) and `SeasonalNaiveForecaster` (periodic-regime floor). Directly fixes KARC's structural periodic-blindness documented in [`.benchmarks/010_report_the_floor_consolidated.md`](../../.benchmarks/010_report_the_floor_consolidated.md) §T7 (K-sweep 2026-07-20 refuted the "K=4 too shallow" hypothesis: KARC's Chebyshev basis can't fit periodic data regardless of K).

Two `WelfordMse` accumulators + sigmoid confidence + cold-start floor. **Revised from variance-only to MSE (variance + bias²)** after Plan 514 surfaced the failure mode where a consistently-biased forecaster has variance 0 but large error. Implies `karc_forecaster` (the gate routes to KARC) + `conformal_predictive_intervals` (the floor the gate routes to in the periodic regime and during cold-start).

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | KARC ≥95% ticks on Lorenz-63; Seasonal ≥95% on period=12; mix ≤5% | PASS (Plan 514 Phase 1) | ✅ PASS |
| **G2** | `decide()` ≤ 50 ns/call | 37 ns median | ✅ PASS |
| **G3** | enabling gate does not perturb KARC forecasts | bit-identical (conformal_karc_no_regression.rs) | ✅ PASS |
| **G4** | 0 allocs/100 calls | 0 allocs | ✅ PASS |

**Runtime integration gain (riir-ai Plan 514 Phase 1):** G1 PASS — **92.45% MAE reduction** on mixed-regime NPC corpus (synthetic). G2 ~at-budget — 89 ns/tick. Pure modelless (two Welford accumulators + sigmoid).

**Why opt-in.** Primitive-level GOAT PASS + positive synthetic-corpus runtime gain. Stays opt-in pending a production-corpus gain measurement.

🔧 Feature flag: `karc_regime_gate` (implies `karc_forecaster` + `conformal_predictive_intervals`) — **opt-in**.

📖 Plan: [`.plans/556_karc_mitigations_open_primitives.md`](../../.plans/556_karc_mitigations_open_primitives.md), Benchmark: [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md), Companion runtime: [`riir-ai/.plans/514_karc_mitigations_runtime.md`](../../../riir-ai/.plans/514_karc_mitigations_runtime.md).

### 24.4 Batched MatVec — `karc_batched_matvec` (Plan 556 Phase 2)

SIMD-batched forecast across N forecasters of identical (D, M, K) shape. Crowd-scale perf primitive: amortizes memory bandwidth by laying out N `Wout` matrices contiguously and hoisting the per-output-row `simd::simd_matvec` call across the batch. Ships `KarcBatchForecaster` + `karc_batched_matvec_into`.

**G2 partial PASS (architectural finding).** Pure-matvec amortizes well, but full-forecast amortization does NOT materialize because `feature_expand` dominates the per-forecast cost.

| N | pure_matvec | batched_forecast_full | sequential_baseline | matvec amortization | full amortization |
|---|---|---|---|---|---|
| 1 | 104 ns | 408 ns | 411 ns | 1.0× | 1.0× |
| 8 | **815 ns** | **3.42 µs** | **3.33 µs** | **4.0×** | **0.97×** |
| 32 | 3.77 µs | 13.6 µs | 14.8 µs | **7.0×** | 1.09× |

The original G2 target (5.3× full-forecast amortization at N=8) assumed the matvec was the dominant cost. Measurement showed it's only ~25% — `feature_expand` is ~75% (delay state → ψ basis expansion, per-NPC, not amortizable by the batched matvec).

**Architectural redirect (Plan 514 Phase 3).** The right consumer for this primitive is cell-shared-KARC + per-NPC latent_functor deviation — ONE feature_expand per cell, batched matvec across N NPC Wouts — not per-NPC-Wout batching. A future `feature_expand_batched` primitive could also close the gap.

🔧 Feature flag: `karc_batched_matvec` (implies `karc_forecaster`) — **opt-in**.

📖 Plan: [`.plans/556_karc_mitigations_open_primitives.md`](../../.plans/556_karc_mitigations_open_primitives.md), Benchmark: [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md), Companion runtime: [`riir-ai/.plans/514_karc_mitigations_runtime.md`](../../../riir-ai/.plans/514_karc_mitigations_runtime.md).

### 24.5 LOD Tier — `karc_lod_tier` (Plan 556 Phase 3)

Config tag + tier-promotion Wout projection. Three nested tiers (LOD0 background D=8/M=4/K=2 d_h=64 / LOD1 midground D=8/M=8/K=4 d_h=256 / LOD2 hero D=8/M=8/K=8 d_h=512) map to different `KarcForecaster` const-generic monomorphizations. The nested-subset structure (LOD0 features are a strict prefix of LOD1; LOD1 of LOD2) makes tier promotion a pure index remap — down-tier preserves surviving Wout columns bit-identically; up-tier zero-fills new columns.

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | tier promotion preserves surviving Wout columns bit-identically | bit-identical | ✅ PASS |
| **G2** | tier promotion ≤ 10 µs (one-time) | worst-case 831 ns (Lod1→Lod2 release) | ✅ PASS |
| **G3** | default (LOD1) path unchanged | bit-identical | ✅ PASS |
| **G4** | per-tick dispatch zero-alloc | 0 allocs/tick | ✅ PASS |

**Config revision (load-bearing).** Lod2 ships as (D=8, M=8, K=8, R=1) → d_h=512, NOT the plan's original (8, 8, 8, 2) → d_h=18_720. The plan's figure doesn't math out (8·8·8·2 = 1024, not 18_720). R=2 promotion-gate config (the real d_h=18_720 from Issue 185/186/187) deferred — pair-product features break the nested-subset invariant.

**Runtime integration — honest split verdict (riir-ai Plan 514 Phase 2).** The primitive itself is correct (G1-G4 all PASS). The runtime integration's G2 has a split verdict:

| Scale | Savings | Verdict |
|---|---|---|
| **1k NPCs (production scale)** | 14.7% (re-validated 2026-07-20), 5.3× headroom | ✅ PASS |
| **10k NPCs (crowd scale)** | 4.9% | ❌ FAIL |

**Root cause of the 10k FAIL.** 10k-NPC state (~20 MB) exceeds L3 cache, so memory bandwidth dominates and the compute savings vanish. The dormant-Lod1 memory overhead cancels Lod0's 4× compute advantage at crowd scale. LOD is a **per-node compute optimization, not a per-cluster one** — 10k+ NPC scale belongs in a **sharding layer** (across game-server nodes). **The sharding substrate landed 2026-07-25** at `riir-engine/src/npc_shard.rs` (feature `npc_shard`); Issue 556 POC confirmed single-process sharding is ruled out (22% regression vs flat 10k) and multi-node distribution is required — see `riir-ai/.benchmarks/556_npc_shard_goat.md`. Plan 514 Phase 3/4 G2 targets revised from "10k NPCs on a single node" to "1k NPCs per shard".

**Why opt-in.** Primitive-level GOAT PASS + 1k-scale runtime PASS. Stays opt-in until either a pure-enum redesign (breaks `forecaster()` API) or a positive gain on a smaller-scale corpus.

🔧 Feature flag: `karc_lod_tier` (implies `karc_forecaster`) — **opt-in**.

📖 Plan: [`.plans/556_karc_mitigations_open_primitives.md`](../../.plans/556_karc_mitigations_open_primitives.md), Benchmark: [`.benchmarks/556_karc_mitigations_goat.md`](../../.benchmarks/556_karc_mitigations_goat.md), Runtime bench: [`riir-ai/.benchmarks/514_karc_lod_dispatch_goat.md`](../../../riir-ai/.benchmarks/514_karc_lod_dispatch_goat.md), Sharding substrate + POC verdict: [`riir-ai/.benchmarks/556_npc_shard_goat.md`](../../../riir-ai/.benchmarks/556_npc_shard_goat.md) (`riir-engine/src/npc_shard.rs`, feature `npc_shard`).

## 25. katgpt-canon — Canonical Intent Space Substrate (Proposal 009, Research 459)

**The crate.** `katgpt-canon` ships the `CanonicalIntent { tag, direction }` type + the `ModelAdapter` trait + three concrete adapters behind independent feature gates. The crate depends on `katgpt-core` (for SVD) and `katgpt-spectral` (for Procrustes) — both already in-tree; the crate itself is `publish = true` (crates.io-ready, MIT).

| Adapter | Feature | Math | G1/G2/G4 verdict (Bench 562) |
|---|---|---|---|
| `ProcrustesAdapter` | `canon` | Orthogonal Procrustes rotation `R` (from `orthogonal_procrustes`); `project_into` = `R·h` | G1 residual 0.0000% / round-trip 4.47e-8 / BLAKE3-deterministic; **G2 d=256 16.17µs** ≤ 50µs (post-SIMD, was 29µs); G4 0 allocs hot path. ⚠️ **d=2304 diagnostic = 1.328ms** (O(d²), NOT gated against 50µs — setup-time use only). |
| `SubspaceAdapter` | `canon_subspace` | Joint SVD `M=[A\|B] = UΣV^T`, top-k right singular vectors define the shared subspace; `project_into` = `V_k^T·h` | G1 fit shapes + no-NaN + held-out mean cos 0.257 (frac positive 0.78); **G2 k=4 d_b=1536 417ns** ≤ 50µs; G4 0 allocs. **Carries the load-bearing Bench 423 G5 GO at k∈{2,4}** (mean cosine +0.87/+0.75 on Gemma↔MiniCPM real weights). |
| `MaskAdapter` | `canon_mask` | Elementwise mask application (lottery ticket *apply*, not discovery); `project_into` = `mask ⊙ h` | G1 all-ones identity + half-zero preserve; **G2 d=2304 1.38µs** ≤ 50µs; G4 0 allocs. Discovery routes to riir-train per Research 459 §1.3. |

**All 17 GOAT sub-gates PASS** (Bench 562, 2026-07-28). The 8-wide FMA dot product SIMD optimization (commit `e5efd20e`) cut the d=256 Procrustes hot path 29→16µs and the d=2304 diagnostic 3.9→1.3ms (2.9× from the 8-wide accumulator pattern, mirroring `dot_8wide` in katgpt-attn-match).

**Why opt-in despite GOAT PASS.** The cross-arch Super-GOAT headline (Proposal 009's "plug-and-play any base model") was **permanently demoted** after four hidden-state construction methods failed the G6 cross-architecture discrimination gate (Bench 424/425/426/427, see `negative_results.md` §15). The substrate is useful (intra-arch snapshot swap, cross-arch ALIGNMENT preservation) but no longer the headline selling point. Promotion to default-on would require a new proposal re-arguing the value proposition post-demotion.

**Known limitation (honest).** `ProcrustesAdapter::project_into` at production model dim (d=2304, Gemma2-2B) is 1.328ms — O(d²) scaling, **not gated against the 50µs target**. The theoretical SIMD floor at d=2304 is ~220µs (5.3M flops / 8-wide AVX2 FMA / 3 GHz); even perfect SIMD can't hit 50µs. The 50µs G2 floor applies to the per-direction-per-tick hot path — that's SubspaceAdapter (O(d·k), k≪d) and MaskAdapter (O(d)). ProcrustesAdapter's use case is same-arch snapshot swap, a setup-time operation where 1.3ms is acceptable.

🔧 Feature flags: `canon`, `canon_subspace`, `canon_mask` (independent, default-off). Crates: `katgpt-canon`.

📖 Proposal: [`.proposals/009_canonical_intent_space.md`](../../.proposals/009_canonical_intent_space.md), Research: [`.research/459_canonical_intent_space_plug_and_play.md`](../../.research/459_canonical_intent_space_plug_and_play.md) (CLOSED), Benchmark: [`.benchmarks/562_katgpt_canon_goat.md`](../../.benchmarks/562_katgpt_canon_goat.md), Cross-arch demotion: [`negative_results.md`](negative_results.md) §15, Non-hidden-state follow-up: [`.proposals/010_non_hidden_state_canonical_construction.md`](../../.proposals/010_non_hidden_state_canonical_construction.md) (draft).

## 26. SipIt Transformer Inversion (Plan 561, arxiv 2510.15511)

**The primitive.** `invert_sequence` recovers the discrete input tokens `x ∈ V^T` from a transformer's observed hidden states `h̆_t` at positions `t ∈ [0, T)` — the inversion that Nikolaou et al. (ICLR 2026) prove is well-posed under the paper's injectivity theorem. Two policies: `RandomPolicy` (uniform-without-replacement enumeration, the paper's baseline) and `GradientGuidedPolicy` (paper Alg 3 — proxy hidden state + finite-difference gradient descent + periodic vocab projection + random fallback).

**Phases 1-4 DONE (2026-07-26), Phase 5 awaiting consumer.** G1-G4 PASS on the toy 2-layer GELU transformer (d=16, |V|=32, T=8):

| Gate | Target | Result |
|---|---|---|
| **G1** correctness | 8 random prompts recover exactly; Lemma D.2 causality; corrupted-observed rejects | 3/3 sub-tests PASS (20 unit tests green) |
| **G2** perf | random policy ~linear in \|V\|; gradient-guided fewer acceptance tests | random 37→130→1375µs/pos for \|V\| 32→128→512 (linear); **grad-guided 317 vs random 1075 acceptance tests across 64 positions (70.5% reduction, 3.4×)** |
| **G3** no-regression | default features unchanged | 1814 lib tests pass (zero leak behind feature gate) |
| **G4** alloc-free | hot path zero per-trial allocs | per-call 2 allocs (random) / 5 (grad); steady-state 10× per-call = no per-trial leak |
| **Phase 4 robustness** | Theorem 3.2 perturbation guarantee | recovery holds below `Δ_π/2` noise; degrades above; margin strictly positive on random init |

**The honest toy-scale caveat.** Gradient-guided sub-linear scaling in |V| (the paper's <0.25%·|V| claim for |V|≥32K) is **NOT validated on the toy** — the numerical finite-difference gradient dominates latency at d=16 (O(D) forward evals per step × 200 steps). Validating the paper's regime requires a real transformer (GPT-2/Llama) with an analytical gradient (1 fwd + 1 bwd ≈ 2× fwd cost) + |V| ≥ 32K. The toy proves the mechanism is correct + the strict-improvement A/B holds; it cannot prove the production latency/speedup tradeoff.

**The 1/sqrt(D) vs 1.0 weight-scale lesson (load-bearing for reproducibility).** Phase 1 used standard stable-training scale `1/sqrt(D)` — GELU saturates near the origin, the Jacobian is effectively zero, the loss landscape is flat, gradient steps move the proxy <0.1 units. Phase 2 uses `new_scaled(rng, 1.0)` explicitly — the Jacobian becomes well-conditioned, gradient norm ~700 (clipped to 1.0), proxy converges within ~20 steps. This is a **substrate-scale correction**, not hyperparameter tuning — real transformers (GPT-2, LLaMA) have weights large enough that the Jacobian is well-conditioned at `1/sqrt(D)` because they have many more layers and much larger D.

**Why opt-in.** No consumer wired yet (grep verified: zero `transformer_inversion` consumers across all 7 repos). The primitive is research infrastructure for transparency/audit tooling on standard text transformers — the open adoption hook. Phase 5 (T5.1) awaits a concrete consumer (e.g. a speculative-decode audit mode, or a transparency feature in riir-ai). If no consumer materializes within ~3 months, it stays parked as opt-in research infrastructure (T5.2, re-evaluate 2026-10-26).

**Rejected fusions (do NOT re-add without amending Plan 561).** Applying SipIt to HLA per-NPC state (HLA is a sigmoid-bounded kernel, not a text transformer — theorem doesn't transfer); activation-based sync compression (sync already commits 32-byte hash; transmitting activations is 96× bandwidth increase); lossless activation hashing (theorem is measure-zero over parameters, not bit-exact over f32); cold-tier prompt re-hydration (SipIt needs model weights + per-position matrix; activations are 15-1000× larger than prompts); transmitting compact h for quorum audit (violates the sync-boundary rule — sync scalars, not embeddings).

🔧 Feature flag: `transformer_inversion` (in katgpt-core, default-off); `grad_policy` adds the gradient-guided driver.

📖 Plan: [`.plans/561_transformer_inversion_sipit_open_primitive.md`](../../.plans/561_transformer_inversion_sipit_open_primitive.md), Research (Gain-Redirects cross-refs): [`.research/158_MUX_Multiplexed_Latent_Reasoning.md`](../../.research/158_MUX_Multiplexed_Latent_Reasoning.md) + [`.research/232_Task_Relevant_Identifiability_Specialist.md`](../../.research/232_Task_Relevant_Identifiability_Specialist.md) + [`.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md`](../../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md), Paper: [arXiv:2510.15511](https://arxiv.org/abs/2510.15511), Reference impl: <https://github.com/giorgosnikolaou/SIPIT>, Bench: `crates/katgpt-core/benches/bench_561_inversion_goat.rs`.

## 27. LatentConfounderAudit — CD-LAM §III-B Diagnostics (Issue 194, arxiv 2607.09185)

**The primitive.** Three modelless diagnostics distilling Wei et al. 2026 (*Causally Debiased Latent Action Model*, CD-LAM §III-B + Appendix A) — the confounder-purity audit that any direction-vector consumer (MAG/TILR/LatentFieldSteering/CommittedFieldBlend) can run before deploying a mined or constructed direction:

| Diagnostic | Formula | Clean value | What it tests |
|---|---|---|---|
| Zero-transition response | `R₀ = RMS(‖E(x, x)‖) / D` | ≈ 0 | No-op input pair should produce near-zero latent |
| Shift-invariance response | `R_shift = RMS(‖E(x, T(x))‖) / D` | ≈ 0 | Nuisance transform should produce near-zero latent |
| Shortcut leakage | `mean_cos(diff-action) − mean_cos(same-action)` | < 0 | Action similarity should dominate context similarity |

Where `D = RMS(‖E(x, x′)‖) + ε` over ordinary transitions. `LatentConfounderAudit::audit_confounders_into` takes a pre-allocated `AuditScratch`; the convenience `audit_confounders` wraps it. The encoder API is `Fn(&[f32], &[f32], &mut [f32])` — output buffer as 3rd arg, sidestepping HRTB lifetime issues.

**G1-G4 PASS modellessly (Bench 194, 2026-07-28).** 12 unit tests + 1 doctest on a synthetic encoder `E(x,x') = A(x,x') + c·confounder(x)` with known confounder coefficient `c`. Clean (c=0): R₀<1e-5, R_shift<1e-5, L<0. Confounded (c=2.0): R₀>0.1, R_shift>0.1, L>-0.5. Monotone across c∈{0, 0.5, 1, 2, 5} — the audit is a quantitative purity score, not just binary pass/fail.

| Gate | Target | Result |
|---|---|---|
| **G1** correctness | monotone in c; 12 tests | ✅ PASS |
| **G2** perf | sub-µs at HLA d=8 | **292 ns/call** at d=8 (3.4× under 1µs); d=32 = 750ns; d=64 = 1.38µs |
| **G3** no-regression | feature-gated, no existing code touched | 1814 → 1814 default; +12 with feature on |
| **G4** alloc-free | zero steady-state | 0 allocs / 100 audit calls (TrackingAllocator sentinel-verified) |

**What this does NOT prove (honest).** (1) Does not prove the audit catches real bugs in production-mined direction vectors — the G1 synthetic encoder has a known injected confounder; real mined directions (MAG/TILR/Steering/Blend) could have subtler confounders the diagnostics miss. (2) The "Report the Floor" rule (Research 322 / Plan 340) does NOT apply — the three metrics are raw geometric measurements (norm ratios, cosine gaps), NOT probabilities / confidence scores / predictive intervals; no distributional claim. (3) Does not prove a quality gain in a downstream consumer.

**Why opt-in.** Diagnostic primitive, not a capability. Promotion to default-on requires a concrete consumer (MAG/TILR/Steering/Blend) benchmarking a real-bug-caught gain (fewer misconfigured directions deployed). No consumer has adopted the audit yet. Re-opens when a consumer adopts + demonstrates a real-bug-caught gain. The CD-LAM training recipe (`L_emb + L_ctr + L_cal` + three-stage fine-tuning) is genuinely gradient-descent → routes to riir-train if a video world model or analogous training system is built.

**The false-PASS correction (documented for future maintainers).** The initial research verdict on CD-LAM was PASS; that was revised to Gain after honest re-review — the diagnostic FRAMEWORK is a real gain (3 modelless metrics + the encoder API contract), but the original PASS implied the primitives shipped CD-LAM's debiasing capability, which they do not (that's training-side, routes to riir-train). The bench file is the durable home of the GOAT verdict; the issue file was removed per noise-reduction rule.

🔧 Feature flag: `latent_confounder_audit` (in katgpt-core, default-off).

📖 Issue (removed, bench is durable home): [`.benchmarks/194_latent_confounder_audit_goat.md`](../../.benchmarks/194_latent_confounder_audit_goat.md), Research: [`.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md`](../../.research/460_CD_LAM_Latent_Confounder_Audit_Diagnostics.md), Paper: [arXiv:2607.09185](https://arxiv.org/abs/2607.09185), Bench: `crates/katgpt-core/benches/bench_194_latent_confounder_audit_goat.rs`.

## 28. EventLog Query Combinator — PRO-LONG Programmatic Search (Plan 562, arxiv 2607.20064)

**The primitive.** A deterministic query combinator over the existing `EventLog<A>` append-only event log (Plan 124). Distills PRO-LONG's load-bearing finding (Table 1): programmatic tools (grep + Python) account for +15.2 of the +18.1 gain on ARC-AGI-3 — the access pattern (search-at-read-time, decide-nothing-at-write-time) is the value, independent of the LLM that instantiates it. This primitive ships the **pattern-based search axis** (grep/regex/predicate analog) as a modelless, zero-allocation, composable API on top of the lossless log substrate.

Ships:
- `EventPredicate<A>` trait (object-safe, `Debug` supertrait) — the escape-hatch seam for consumer-defined predicates.
- `Predicate<A>` enum: `EventTypeIs` / `EventTypeIn` / `IdRange` / `IdRangeFrom` / `And` / `Or` / `Not` / `All` / `None_` / `Custom(Box<dyn EventPredicate<A>>)`. Constructor helpers: `event_type(t)` / `id_range(lo, hi)` / `id_range_from(from)` / `.and(...)` / `.or(...)` / `!pred` (via `std::ops::Not`) / `custom(p)`.
- `EventLog::filter(&self, &Predicate<A>) -> impl Iterator<Item = &Event<A>>` — lazy, zero-alloc. The direct "grep the log" analog.
- `EventLog::query_window(&self, Range<EventId>, Option<EventType>) -> impl Iterator` — contiguous slice + optional type filter. Sub-µs by construction.
- `EventLog::count_where(&self, &Predicate<A>) -> usize` — the `grep -c` analog.
- `EventLog::first_where` / `last_where(&self, &Predicate<A>) -> Option<&Event<A>>` — early-exit (`find` / `rfind`).

**G1–G4 PASS (Bench 564, 2026-07-29) — ship-quality gate, NOT promote-to-default.**

| Gate | Target | Result |
|---|---|---|
| **G1** correctness | 13 predicate combinations on a 100-event deterministic log | ✅ PASS — all 13 (EventTypeIs, count_where All/None_, first/last_where, query_window ±type filter, And, Or, Not, Custom payload>500) |
| **G2** perf | sub-µs / sub-100ns per operation | ✅ PASS — `filter` **4.99 ns/result-event** (200× under 1µs), `query_window` **0.46 ns/call** (217× under 100ns), `first_where`/`last_where` **4.04 / 5.71 ns** (24× / 17× under 100ns) |
| **G3** no-regression | feature-off build clean, Plan 124 API unchanged | ✅ PASS — `existing_api_unchanged` unit test; purely additive `impl EventLog` block gated `#[cfg(feature = "event_log_query")]` |
| **G4** alloc-free | zero steady-state allocation | ✅ PASS — filter collect capacity stable (512→512 across 1000 iterations); count/first/last/query_window zero-alloc by construction (lazy iterators / early-exit / slice) |

**Why opt-in.** This is a **ship-quality gate**, not a promote-to-default gate. Per the Gain-tier verdict (Research 461), the feature is a missing capability (the programmatic-search axis did not ship), not a measurable improvement over an existing approach — there is no incumbent query API on `EventLog` to beat (only `iter()`). Promotion to default-on requires a downstream consumer to prove a measurable gain over the no-query baseline. The three trigger conditions (Plan 562 Phase 3):
- **T3.1** riir-engine per-NPC cognition (CLR vote accuracy, KARC forecast skill, consolidation quality) — opens a riir-ai plan for the latent-predicate bridge.
- **T3.2** riir-neuron-db Raven/δ-Mem consolidation pipeline ("find all events matching P in last N ticks" quality/latency gain) — opens a riir-neuron-db plan.
- **T3.3** katgpt-pruners MCTS planner (`filter` for "find all evaluations matching P" search-efficiency gain) — opens a katgpt-rs plan.
- **T3.4** If any of T3.1–T3.3 pass → promote `event_log_query` to default features.

**Why this is Gain-tier, not GOAT-tier.** The gain is a missing feature, not a measurable improvement. The three retrieval axes (pattern / semantic / content-addressed) are orthogonal: the pattern axis (this primitive) composes with the semantic axis (`experience_graph` latent-seeded NS traversal, riir-neuron-db) and the content-addressed axis (`Engram` hash→slot) at the consumer layer via `Predicate::Custom`.

**The PRO-LONG Table 1 finding (load-bearing for this distillation).** Programmatic tools (grep + Python) drive +15.2 of the +18.1 gain on ARC-AGI-3; Write/Edit adds only +2.9. The value is in the log + programmatic search, not in self-authored notes (clearing the workspace every call costs PRO-LONG only 0.5 points). This primitive ships the deterministic analog of the grep/Python search axis — no LLM in the loop.

🔧 Feature flag: `event_log_query` (in katgpt-pruners, default-off; implies `event_log`). Root forwards via `event_log_query = ["katgpt-pruners/event_log_query"]`.

📖 Plan: [`.plans/562_event_log_query_combinator.md`](../../.plans/562_event_log_query_combinator.md), Research (Gain): [`.research/461_PRO_LONG_Programmatic_Memory_Log_Search.md`](../../.research/461_PRO_LONG_Programmatic_Memory_Log_Search.md), Paper: [arXiv:2607.20064](https://arxiv.org/abs/2607.20064) PRO-LONG (Fox et al., Duke, 2026-07-23), Bench: [`.benchmarks/564_event_log_query_goat.md`](../../.benchmarks/564_event_log_query_goat.md) (numbered 564 not 562 — `.benchmarks/562` was already allocated to `katgpt-canon`), Substrate: `crates/katgpt-pruners/src/event_log.rs`, Example: `crates/katgpt-pruners/examples/event_log_query_basic.rs`.

## 29. SWE Trajectory Freeze — Modelless Inference-Attempt Freezer (Proposal 011 Layer 4, Issues 569-571)

Modelless committed freeze of an inference attempt's trajectory through patch-space — the flipped R463 insight that **even when a model proposes zero valid patches, the inference loop's trajectory has measurable geometry that is freezable and comparable across snapshots**. Composes shipped DEFAULT-ON substrate (`tf_loop` + `latent_trajectory_geometry` + `committed_field_blend`/FAME + a local BLAKE3 envelope) into a two-stage pipeline:

1. **Fit** (offline) — `derive_directions` / `derive_directions_and_centroid` from cluster centroids of labeled training summaries (the T5.3b data-derived-directions fix — random directions degenerate via concentration-of-measure).
2. **Freeze** (online, per-attempt) — encode trajectory into summary → mean-center → project onto pre-fit directions via FAME sigmoid gates → commit via BLAKE3 envelope.

### Two encoders (two discrimination axes)

| Encoder | Method | Discriminates | d | Gate |
|---------|--------|---------------|---|------|
| `GeometrySummaryEncoder` | Trajectory SHAPE (length, curvature, step-to-step cosine, n_steps) | **Structural** — failure-mode classification (oscillation vs committed-wrong vs converged) | 4 (+replicate to D) | Bench 013 G1-G4 + Bench 014 G5 |
| `StateMagnitudeEncoder` | State MAGNITUDE (mean/std/max/min norm, initial/final norm, norm ratio, mean cosine) | **Value-level** — cross-snapshot identification (which checkpoint produced this trajectory?) | 8 | Bench 019 G1-G5 |

The two encoders are **complementary, not redundant**. Shape features are perturbation-INVARIANT (the failure-mode geometry is preserved across weight perturbations — Bench 015 NEGATIVE on value-level discrimination via geometry). Magnitude features are perturbation-SENSITIVE (activation scale is weight-determined — Bench 018 POSITIVE at 100% / σ≥0.1).

### Two freeze methods

| Method | Encoder | Output | Use case |
|--------|---------|--------|----------|
| `freeze_attempt[_into]` | GeometrySummaryEncoder | `FrozenAttempt<N,D>` (commits geometry triple + π + summary) | Classify the failure MODE of an attempt |
| `freeze_attempt_value[_into]` | StateMagnitudeEncoder | `FrozenValueAttempt<N,D>` (commits π + summary, no geometry triple) | Identify which SNAPSHOT produced an attempt |

Both commit via the same local `TrajectoryFreezeEnvelope` (BLAKE3-checked header + payload, matches the `MerkleFrozenEnvelope` pattern — no cross-repo dep). At production scale (N=3, D=32): geometry = 164 bytes, value = 140 bytes.

### GOAT gate results

**Geometry path (Bench 013 + Bench 014):**

| Gate | Status | Detail |
|------|--------|--------|
| G1 directions non-degenerate | ✅ PASS | All unit-norm, non-collinear (Bench 013) |
| G2 perf | ✅ PASS | 4582 ns/call < 5000 ns (Bench 013) |
| G3 cross-mode discrimination | ✅ PASS | 100% accuracy, oscillation 0.98 / committed_wrong 0.71 / converged 0.72 gates (Bench 013) |
| G4 alloc-free | ✅ PASS | `freeze_attempt_into` = 0 allocs (after `from_states_into` zero-alloc fix, Bench 014) |
| G5 cross-model | ✅ PASS | **100% accuracy** on real Kimi-K3 vs random weights (40/40 held-out, Bench 014) |

**Value path (Bench 019 + Benches 018/020):**

| Gate | Status | Detail |
|------|--------|--------|
| G1 correctness | ✅ PASS | Hand-computed 3-state dim-2 trajectory matches bit-identically (Bench 019) |
| G2 perf | ✅ PASS | 51.8µs vs geometry 100.7µs = 0.52× (faster, single-pass Welford, Bench 019) |
| G3 no-regression | ✅ PASS | 1851 lib tests; geometry path unaffected (Bench 019) |
| G4 tamper-evidence | ✅ PASS | Header verification; tampered merkle_root + commitment both fail (Bench 019) |
| G5 value discrimination | ✅ PASS | 100% on synthetic scale+variance-shift; **100% on real Kimi-K3 σ-perturbation** at σ≥0.1 (processing: Bench 018, generation: Bench 020) |

**The load-bearing discrimination result (Bench 018 + Bench 020):** The SEQUENCE trajectory (final hidden states across a prompt's tokens with growing KV cache, N=48-64 steps) achieves **100% per-prompt accuracy at σ≥0.1** via `StateMagnitudeEncoder`, with d_Mahalanobis = 14.526 at σ=0.5 (50× the depth trajectory's 0.285). This holds for BOTH processing trajectories (fixed-token reading) AND generation trajectories (greedy argmax decoding) — Bench 020 confirmed generation is equally discriminative (actually BETTER at σ=0.05: 100% vs 81.2%, because argmax choices amplify weight perturbation near decision boundaries).

The DEPTH trajectory (9 per-layer states per token, Benches 012-017) is the NEGATIVE control: shape features are perturbation-invariant, and the per-token Bayes-optimal ceiling for value-sensitive features is only ~54% (SNR ≈ 1.0 — token-to-token variance swamps the perturbation signal). The sequence trajectory overcomes this via the √N SNR boost (64 steps vs 9).

### Why opt-in

1. **Synthetic G5.** The substrate-level G5 test (Bench 019) uses synthetic scale+variance-shifted trajectories. Benches 018/020 use real Kimi-K3 weights with synthetic σ-perturbation (uniform multiplicative noise). **Real training drift may differ** (structured, non-uniform). Promotion should wait for a second real checkpoint.
2. **No production consumer.** The primitive has no downstream caller yet. Per codebase pattern, promotion follows a consumer demonstrating the gain. The consumer is the SWE-bench pruner runtime (Proposal 011 Layer 4), which is blocked on Layer 3 (rubrc WASM compiler maturity).

The substrate is READY — the remaining gap is real-checkpoint validation (an external dependency) + consumer wiring (blocked on Layer 3). Re-evaluate at the next SWE-bench integration milestone.

### The honest negative-result trail (Benches 012-017)

The path to the positive result was NOT linear — five benches explored + documented NEGATIVE results before the sequence-trajectory breakthrough:

| Bench | Question | Result | Lesson |
|-------|----------|--------|--------|
| 012 | Real Kimi-K3 depth trajectory geometry | PARTIAL — G3 FAIL 29% distinct across tokens | Depth geometry is model-determined, not input-determined |
| 015 | Geometry features + σ-perturbation | NEGATIVE — ~50% (coin flip) at all σ | Shape features are perturbation-INVARIANT |
| 016 | Value-sensitive per-layer displacement features | NEGATIVE for per-token — centroid works, per-token SNR ≈ 1.0 | Resolution floor, not information deficit |
| 017 | Mahalanobis/LDA covariance-aware classifier | NEGATIVE — Bayes-optimal ceiling ~54% | Per-token SNR floor is FUNDAMENTAL |
| 018 | **Sequence trajectory state magnitude** | **POSITIVE — 100% at σ≥0.1** | √N SNR boost overcomes the floor |

This trail is load-bearing: it documents WHY the sequence trajectory + state-magnitude encoder is the correct combination (not a lucky guess), and why the depth trajectory + geometry encoder is the wrong combination for value-level discrimination.

🔧 Feature flag: `swe_trajectory_freeze` (in katgpt-core, default-off; implies `latent_trajectory_geometry` + `committed_field_blend`). Root forwards via `swe_trajectory_freeze = ["katgpt-core/swe_trajectory_freeze"]`.

📖 Proposal: [`.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md`](../../.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md) (Layer 4), Issues: 569 + 570 + 571 (all resolved + removed per noise-reduction rule; resolutions captured above + in Benches 013/014/018/019/020), Substrate: `crates/katgpt-core/src/swe_trajectory_freeze.rs`, Benches: [013](../../.benchmarks/013_swe_trajectory_freezer_goat.md) (geometry GOAT) + [014](../../.benchmarks/014_swe_trajectory_freezer_g5.md) (cross-model G5) + [018](../../.benchmarks/018_sequence_trajectory.md) (sequence POSITIVE) + [019](../../.benchmarks/019_state_magnitude_encoder_substrate_goat.md) (substrate GOAT) + [020](../../.benchmarks/020_generation_trajectory.md) (generation POSITIVE), Negative controls: [015](../../.benchmarks/015_swe_trajectory_perturbation_sensitivity.md) + [016](../../.benchmarks/016_value_sensitive_encoder.md) + [017](../../.benchmarks/017_covariance_aware_classifier.md).

## 30. GDN-HOLA Dual-Path Tree Verification (Plan 430, Research 407 §2.2)

Fuses two shipped opt-in primitives — Plan 424 (GDN rollback-free tree verify, catalog §18) + Plan 395 (HOLA hippocampal exact KV cache) — into a **dual-path tree verifier** that scores speculative draft tree nodes against BOTH the GDN recurrent state (masked triangular solve) AND the HOLA hippocampal cache (ancestor-masked softmax read), with **zero rollback on either path**. Residual-add complement: `O[i] = O_gdn[i] + O_hola[i]` — the hippocampal cache complements the compressive recurrent state (HOLA §3.5), not replaces it.

**Why this matters:** GDN2's fixed-size recurrent state compresses context but loses exact long-range recall. HOLA's hippocampal cache recovers exact recall for high-surprise tokens but is a flat top-w set with no tree-verification story. Fusing them at the speculative tree-verify layer gives both exact-recall recovery (HOLA) AND rollback-free tree scoring (GDN masked solve) — the hippocampal complement to the compressive recurrent state, extended to branching speculative drafts.

**Key design decisions:** (1) ancestor masking by construction on the HOLA path (only ancestor tokens in `block_kv`, no bitmask needed — unlike the GDN solve which needs the explicit `X` interaction matrix); (2) read-only verify, dual-write commit (both `S₀` and cache are read-only during verify; commit writes both — GDN via `commit_accepted`, HOLA via `observe`); (3) commit pipes GDN residuals to HOLA (the `(beta_t, residual_norm_t)` already computed during GDN commit are exactly what `HippocampalCache::observe` needs — "β·‖e‖ is free").

| Gate | Status | Detail |
|------|--------|--------|
| **G1** correctness | ✅ PASS | Dual-path verify on random trees (T=16,32,64,128) within `1e-3` of per-branch sequential GDN2+HOLA reference |
| **G2** perf | ✅ PASS per gate def | `dual/(gdn+hola) ≈ 0.91–1.07` for T≥32 — fusion is cheaper than separate. **Aspirational 1.2× sub-bar FAILED**: `dual/gdn = 1.24–1.40×` — HOLA's W=64 softmax read adds 24–40% (inherent cost of exact recall, not an implementation deficiency). Pre-normalization optimization landed (chain T=128 −8.8%) |
| **G3** no-regression | ✅ PASS | With `hippocampal_cache` OFF, `verify_gdn_hola_tree_into` byte-identical to `verify_gdn_tree_into` (Plan 424) |
| **G4** alloc-free | ✅ PASS | `verify_gdn_hola_tree_into` = 0 allocs steady-state; stack-local `o_hola` buffer + reuse of Plan 424 scratch |
| **G5** retrieval | [-] deferred | Requires trained GDN2+HOLA model + multi-key retrieval harness; retrieval property inherited from Plan 395 G4 |

🔧 Feature flag: `gdn_hola_tree_verify` (in katgpt-core; implies `gdn_tree_verify` + `hippocampal_cache`). Stays **opt-in** — requires both opt-in parents + a trained γ vector or modelless γ=1. riir-ai consumer wiring (T3.2) + full integration test (T3.3) deferred (cross-repo; bridge adapter `verify_gdn2_hola_tree_layer` + `commit_gdn2_hola_tree_layer` + `forward_tree_gdn2_hola` shipped).

📖 Plan: [`.plans/430_dual_path_rollback_free_tree_verify_gdn_hola.md`](../../.plans/430_dual_path_rollback_free_tree_verify_gdn_hola.md), Research: [`.research/407_Trees_from_Marginals_GDN_Tree_Verify.md`](../../.research/407_Trees_from_Marginals_GDN_Tree_Verify.md) §2.2, Benchmark: [`.benchmarks/430_dual_path_verify_goat.md`](../../.benchmarks/430_dual_path_verify_goat.md), Parents: [Plan 424](../../.plans/424_gdn_tree_verification_primitive.md) (GDN tree verify, catalog §18) + [Plan 395](../../.plans/395_hippocampal_exact_kv_cache.md) (HOLA cache), Papers: [arXiv:2607.06763](https://arxiv.org/abs/2607.06763) §3.4 + [arXiv:2607.02303](https://arxiv.org/abs/2607.02303)

## 31. Loop Stability Fix — Inter-Loop RMSNorm (Plan 428, Research 414)

Parameter-free architectural fix for T-pass (LT2) looped inference stability: add `rmsnorm(&mut h)` between loop iterations (after the inner layer pass, before the residual gate saves `prev_h`). Byte-identical when `LoopStabilityMode::None` (default); activates only when the consumer sets `Config.loop_stability_mode = LoopStabilityMode::InterLoopNorm`. Zero-cost when off.

**The Phase 1 PoC defend-wrong verdict** (6 competitors head-to-head on a d_model=256 toy looped transformer, T=12 iterations) is the load-bearing finding:

| Competitor | G1 Norm Ratio | G2 KL | G3 OH | G4 Step | Verdict |
|---|---|---|---|---|---|
| Baseline (vanilla) | 11.19× | 0.0128 | 0% | 6.85 | Barely explodes; no convergence |
| **InterNorm** | **3.34×** | **0.0008** | -1.2% | 2.05 | **Only fix that controls norm**; converging |
| FLA-res | 2.2B× | 0.0000\* | 0.4% | 12.8B | **CATASTROPHIC explosion** — direct residual addition amplifies |
| AttnInj | 11.19× | 0.0128 | -0.6% | 6.85 | **No-op** — Q irrelevant for single-position softmax(1)=1.0 |
| Combined | 589M× | 0.0000\* | -1.0% | 3.3B | FLA-res dominates; InterNorm can't compensate |
| DecayGate (0.8) | 1309× | 0.0046 | -0.7% | 3914 | 0.8 accumulates; norm explodes |

\*KL=0.0000 for FLA-res/Combined is a **false pass** — logits so large that softmax saturates to a degenerate one-hot that doesn't change between loops.

**Only Inter-loop RMSNorm works.** FLA-res (direct residual addition of `prev_h` at every layer) causes catastrophic norm explosion (~7× growth per loop). The FLA paper likely uses a gated mechanism, not direct addition. Attention Injection is a no-op for single-position attention. Both were DROPPED from Phase 2 implementation per the defend-wrong verdict.

**Phase 2 GOAT gate** (implementation behind `loop_stability_fix` feature flag):

| Gate | Status | Detail |
|------|--------|--------|
| **G1** byte-identical when `None` | ✅ PASS | 11/11 LT2 tests pass in both configs |
| **G2** finite logits with `InterLoopNorm` at T=12 | ✅ PASS | 8 positions verified |
| **G3** latency overhead < 5% | ✅ PASS | 2.3% on micro model |
| **G4** norm control (InterLoopNorm ≤ baseline) | ✅ PASS | 0.88× ratio (non-worsening) |

**Why opt-in:** the micro model (`Config::micro()`, n_embd=16, 6 layers) doesn't exhibit norm explosion at T=12 (0.88× — norm slightly decreased). The PoC benchmark (`examples/loop_stability_poc.rs`) with d_model=256 + gaussian init std=0.02 showed the explosion (11.19× baseline → 3.34× InterNorm). Promotion requires a real-world model that exhibits T-pass norm explosion. The fix is parameter-free and zero-cost when `None`, so opt-in has no downside.

**Model-based path (riir-train):** documented in Proposal 018 §7.3 — train with FLT fixes (weights adapted to looped architecture), explicit norm penalty in training loss (Readout Blind Spot's training fix), stochastic loop count during training.

🔧 Feature flag: `loop_stability_fix` (forwarded: katgpt-rs → katgpt-core → katgpt-types). `LoopStabilityMode` enum in `crates/katgpt-types/src/enums.rs`; `Config.loop_stability_mode` field behind `#[cfg(feature="loop_stability_fix")]`, initialized to `None` in all 11 constructors.

📖 Plan: [`.plans/428_loop_stability_poc.md`](../../.plans/428_loop_stability_poc.md), Research: [`.research/414_Fully_Looped_Transformer_Readout_Blind_Spot.md`](../../.research/414_Fully_Looped_Transformer_Readout_Blind_Spot.md), PoC: `examples/loop_stability_poc.rs` (529 lines, std-only), Papers: [arXiv:2605.18797](https://arxiv.org/abs/2605.18797) (Fully Looped Transformer) + [arXiv:2606.24898](https://arxiv.org/abs/2606.24898) (Readout Blind Spot)

## 32. Quant-Error LoRA — Quantization-Error Compensating Reader-LoRA (Issue 565, Research 463)

Deterministically-constructed low-rank reader-LoRA that compensates for the quantization error of a weight matrix: given `W` (f32 reference) and `W_q` (quantized), compute `E = W − dequant(W_q)`, then `E ≈ A·B` via truncated SVD. At inference: `y = W_q·x + α·B·(A·x)`. Three variants shipped: `from_error` (weight-space SVD, calibration-free), `from_error_data_aware` (output-space SVD, calibration-conditioned), `SparseErrorBypass` (top-K COO). Pure modelless (closed-form SVD / partial-sort; no gradient descent). Gated on `subspace_phase_gate` for Plan 301's `thin_svd_into`.

**The Issue 565 defend-wrong PoC (2026-08-01) is the load-bearing finding.** Four strategies tested head-to-head on the 105K-param Moka CNN:

| Strategy | Cosine vs f32 | Δ vs ternary baseline | Verdict |
|---|---|---|---|
| B2 (ternary, no correction) | 0.9706 | — | baseline |
| **A (wSVD rank-32)** | **0.9939** | **+0.023** | ✅ G1 PASS (≥0.02 target) |
| A (wSVD rank-16) | 0.9888 | +0.018 | near-pass |
| B (data-aware SVD) | ~0.90 | −0.06 | ❌ WORSE — confirmed real (not PoC artifact) |
| D (sparse bypass) | 0.91–0.96 | −0.03 to −0.06 | ❌ WORSE — distributed error, not outlier-concentrated |

**Surprise:** Strategy A at rank-16+ genuinely improves the ternary forward pass (0.9939 cosine). The pre-PoC prediction ("rank-8 fails on small CNNs") was too pessimistic — the Small-Kernel Parameter Paradox is a **cost** issue (27.8% param overhead at rank-8 on a 32×288 conv vs 0.39% on an LLM linear layer), not a **quality** issue. Strategy B (data-aware SVD) confirmed WORSE: on small networks the weight structure dominates, and calibration-conditioned output error overfits to the calibration distribution — the opposite of large LLMs (GPTQ/OBQ).

**G5 (the load-bearing gate) — DECISIVELY NEGATIVE.** PUCT win-rate (n=20, b=50, vs greedy f32):

| Strategy | Win-rate |
|---|---|
| f32 baseline | 100% (20/20) |
| B2 (ternary-only) | **0% (0/20)** |
| A rank-32 (ternary+wSVD LoRA) | **0% (0/20)** |

The prediction was right about FAIL but wrong about WHY. The actual failure: residual error after rank-32 LoRA (cosine 0.9939, 78% error energy captured) is still large enough to **collapse PUCT strength entirely** — not just degrade it. PUCT's budget=50 simulations per move amplify the small policy perturbations (total abs diff ≈45 across 82 moves) through search-tree exploration. The value head residual error causes excessive passing (26+ passes/game vs ~17 actual moves). int8 (0.97 cosine, 85-95% win-rate) works because its error is UNIFORM (small, symmetric, ~5% relative); ternary's error is STRUCTURED (large, biased, 145% relative) and the LoRA removes the bias but leaves residual structure that PUCT amplifies.

**The lesson, generalized:** cosine ≥ 0.99 is NECESSARY but NOT SUFFICIENT for PUCT parity. PUCT's search amplifies residual errors, setting a higher bar than greedy-move parity. int8 is a surprising outlier that works because its error is uniform.

**Final verdict:** modelless quant-error-compensating LoRA is unviable for the ternary path on Moka v1. Issue 565 CLOSED. The primitive ships as reusable substrate for larger models where the error manifold is genuinely low-rank AND the target is greedy-move parity (not PUCT parity) — e.g., Gemma 2 2B (Proposal 008) if ever aggressively quantized for edge deployment. The trained-projection path (riir-train) is the only remaining option for Moka.

🔧 Feature flag: `quant_error_lora` (in katgpt-core). Stays **opt-in** — the PoC G5 DECISIVELY FAILED on the only consumer (Moka ternary path). Ships as reusable substrate for future larger-model consumers.

📖 Issue: 565 (removed per noise-reduction rule — resolution above), Research: [`.research/463_moka_freeze_thaw_lever_audit.md`](../../.research/463_moka_freeze_thaw_lever_audit.md) §7 (PoC Addendum 2026-08-01), PoC bench: `riir-poc/tests/quant_error_lora_poc.rs` (cross-repo, permanent regression check), Substrate: `crates/katgpt-core/src/quant_error_lora.rs` (7 unit tests)

## 33. Kimi-K3 Native Support — Model Loader + Decoder Layer Composition (Proposal 032)

Native Rust implementation of the Kimi-K3-0.40B model forward path — the composition layer that fuses four substrate primitives (MLA, KDA, MoE, AttnRes) into a working model + a safetensors weight loader. This is **infrastructure, not an algorithmic primitive** — it exists to enable real-model benchmarking (benches 012–020 use it to extract per-layer hidden-state trajectories from real Kimi-K3 weights). The architecture was distilled from Research 447 (KDA + AttnRes + Stable LatentMoE) + Research 331 (safetensors header analysis, cross-repo in riir-ai).

### Layer topology (verified against real `model.safetensors`)

| Layer | Attention | FFN |
|-------|-----------|-----|
| 0 | KDA | Dense |
| 1–2 | KDA | MoE |
| 3 | MLA | MoE |
| 4–6 | KDA | MoE |
| 7 | MLA | MoE |

8 layers total. KDA (Kimi Delta Attention) on 6 layers, MLA (Multi-head Latent Attention) on 2 layers. Layer 0 is the only dense-FFN layer; layers 1–7 use the Stable LatentMoE. Config: `hidden_size=1024`, `vocab_size=163840`, `n_routed_experts=8`, `num_experts_per_tok=2`.

### What the feature gates

| Feature | Gates | Contents |
|---------|-------|----------|
| `kimi_k3` | `mla_attention` + `kda_linear` + `transformer_moe` + `transformer_attn_res` | Decoder layer composition (`src/kimi_k3/decoder_layer.rs`) — fuses the four substrate primitives into `kimi_decoder_layer_forward` |
| `kimi_k3_loader` | `kimi_k3` + `dep:safetensors` + `dep:base64` + `dep:memmap2` | Adds the model-level forward path + safetensors loader + tiktoken tokenizer (`src/kimi_k3/{model,loader,tiktoken}.rs`) |

Both opt-in (default-off). The loader pulls heavy deps (safetensors tensor parsing, memmap2 for zero-copy mmap, base64 for tiktoken format).

### The forward path

```text
hidden = embed_tokens(input_ids)
block_state = AttnResBlockState::new()
for layer in 0..8:
    kimi_decoder_layer_forward(layer, ..., hidden, block_state)
hidden = apply_attn_res(hidden, block_state, output_attn_res)  // output mixing
hidden = rmsnorm(hidden, final_norm_weight)
logits = lm_head(hidden)
```

Matches `KimiLinearModel.forward` + `KimiLinearForCausalLM.forward` from `modeling_kimi_k3_linear.py` (verified against real source, Research 331). Three entry points: `kimi_k3_forward_token` (production), `kimi_k3_forward_token_timed` (per-phase latency breakdown), `kimi_k3_forward_token_traced` (per-layer hidden-state extraction — the bench 012/014/018/020 trajectory source).

### Zero-copy mmap loader

`load_kimi_k3` memory-maps `model.safetensors` (1.5 GB) via `memmap2`, parses the safetensors header once, and maps tensor names to the weight structs (MLA / KDA / MoE / attn-res) with zero tensor copies — weights are read directly from the mmap'd region. Tensor name mapping verified against the real file header (Research 331): all tensor names match, including the fused projections (`kv_a_proj_with_mqa` = `w_dkv` + `w_kr` split at `d_c=128`, etc.).

### GOAT gate status

The Kimi-K3 loader is **infrastructure**, not a modelless primitive — it has no GOAT gate of its own. Its value is enabling the GOAT gates of the primitives it composes + the benches that consume it:

- **Bench 012** (T5.4): substrate numerical stability at D=1024 on real architecture — G1+G2+G4 PASS, G3 FAIL 29% distinct across tokens (depth geometry is model-determined, not input-determined — the negative control for the sequence-trajectory breakthrough).
- **Bench 014** (G5): 100% cross-model discrimination (real Kimi-K3 vs random weights, 40/40 held-out) via `GeometrySummaryEncoder`.
- **Bench 018 + 020**: 100% value-level discrimination at σ≥0.1 via `StateMagnitudeEncoder` (the load-bearing result for Proposal 011 Layer 4).

The substrate primitives it composes each have their own GOAT gates (GDN2/KDA Plan 105 14/14, MLA, MoE, AttnRes).

### Why opt-in

1. **Heavy deps.** safetensors + memmap2 + base64 are not needed by the default modelless-inference path.
2. **1.5 GB model file.** The loader requires downloading `Kimi-K3-0.40B/model.safetensors` from HuggingFace (`inference-optimization/Kimi-K3-0.40B`) — not bundled.
3. **Research substrate.** The loader exists to enable trajectory-geometry benches, not as a production inference path. Promotion to default would require a production consumer.

### Example

`cargo run --release --features kimi_k3_loader --example kimi_k3_hello_world` — loads real weights, tokenizes a prompt, decodes N tokens, reports per-phase latency + tok/s. Override prompt via `KIMI_PROMPT` + decode length via `KIMI_N_TOKENS`.

🔧 Feature flags: `kimi_k3` (decoder layer composition) + `kimi_k3_loader` (model + safetensors + tiktoken). Both opt-in (default-off) at the root crate.

📖 Proposal: 032 (Kimi-K3 native support — referenced in source; no standalone `.proposals/032_*.md` file). Research: [447](../../.research/447_Kimi_K3_KDA_AttnRes_LatentMoE.md) (architecture distillation) + riir-ai [331](../../../riir-ai/.research/331_kimi_k3_phase6_safetensors_header_analysis.md) (tensor name verification) + [330](../../../riir-ai/.research/330_kimi_k3_modeling_code_divergences.md) (modeling divergences). Substrate: `src/kimi_k3/{mod,decoder_layer,model,loader,tiktoken}.rs`. Example: `examples/kimi_k3_hello_world.rs`. Consumer benches: [012](../../.benchmarks/012_kimi_k3_trajectory_geometry.md) + [014](../../.benchmarks/014_swe_trajectory_freezer_g5.md) + [018](../../.benchmarks/018_sequence_trajectory.md) + [020](../../.benchmarks/020_generation_trajectory.md).

> **Backward (BPTT) path:** see [§76](#76-kimi-k3-analytic-backward--training-reference-gradient-pass) for the analytic gradient pass (`kimi_k3_backward` + per-primitive `kda_backward` / `mla_backward` / `moe_backward`). Training-time reference consumed by riir-train; NOT a production inference path.

## 34. QGF — Q-Guided Flow Test-Time Gradient Guidance (Plan 268)

Distilled from [arXiv:2606.11087](https://arxiv.org/abs/2606.11087) — Zhou et al., *Q-Guided Flow*. Test-time Q-gradient guidance for any `SpeculativeGenerator`: steers discrete generation toward higher expected Q by tilting logits `p' = p_t + (1/β)·∇Q`. No continuous diffusion, no flow-matching training, no BPTT — pure inference-time steering.

### Five-tier routing (Plasma/Hot/Warm/Cold/Freeze)

| Tier | Oracle | Latency | Use case |
|---|---|---|---|
| **Plasma** | ternary `ActionBridge` i8 directions + f32 Q dot | < 100 ns | cheapest, deterministic |
| **Hot** | cached `LeoHead::all_goals_q` f32 values | < 1 µs | cached Q-table |
| **Warm** | GPU batched Q-critic forward | ~1 ms | real neural critic |
| **Cold** | Turso-encrypted Q-table snapshots | ~10 ms | encrypted archival |
| **Freeze** | `NoGuidanceOracle` (zero gradient) | 0 ns | pure BC fallback |

The Freeze tier is always available as graceful degradation — QGF with `guidance_weight = 0.0` is byte-identical to the unguided generator (G2 regression-safety gate).

### Feature gate split

| Feature | Role |
|---|---|
| `qgf` | Parent feature — enables the `qgf` module compilation |
| `qgf_projector` | F2 — `FirstOrderProjector`: one Euler-step lookahead projection |
| `qgf_oracle` | F3 — `QGradientOracle` trait: ∇_a Q(s,â_1), Jacobian dropped per paper §5 |
| `qgf_drafter` | F1 — `QGuidedDrafter`: wraps any `SpeculativeGenerator` + oracle; implies `qgf_projector` + `qgf_oracle` |
| `qgf_adaptive` | F4 — `VarianceAdaptiveGuidance`: sigmoid-gated `1/β = sigmoid(k·(conf−τ))`; implies `qgf_drafter` |

### GOAT gate status (katgpt-core mechanism: ALL PASS)

The katgpt-core mechanism gates (G1 correctness, G2 regression-safety, G3 no-regression, G4 overhead + alloc-free, G5 stability) **ALL PASS** (Bench 268, 2026-07-01):

- **G1** — `tilt_logits` shifts distribution toward higher Q by >10% relative. Includes 2 negative controls (anti-gradient decreases E[Q]; 200 random gradients don't systematically help).
- **G2** — `guidance_weight = 0.0` → byte-identical logits + `applied = false`. Freeze tier is pure BC.
- **G3** — feature combos clean; 42/42 lib tests pass.
- **G4** — tilt overhead ~33 ns (AXPAY + sample), zero hot-path allocation.
- **G5** — sigmoid-bounded weights; no NaN; no off-manifold collapse.

**Stays OPT-IN** — the downstream task-quality gates (Sudoku solve rate +3–8%, DDTree spec acceptance +5–12%, Bomber win rate +2–5%) require real generators + task harnesses that live **outside katgpt-core** (riir-ai integration). This is the katgpt-core → riir-ai scope split (same pattern as Plan 354 SetAttention: core proves the mechanism, riir-ai proves the selling point).

### Sibling: DualLeoOracle (Plan 467)

QGF's 3rd `QGradientOracle` impl (2026-07-18) fuses a LEO teacher head + UVFA student head via `DualLeoMixer::combine_into` at the gradient level. G1–G4 PASS mechanistically; **G5 measured FAIL** on both synthetic (riir-ai Bench 553: dual 0.00% vs single 0.50%) and civ real networks (Bench 558: dual +2.69% vs single 35.68% → 36.64%, <3% gate). The correctness invariant held bit-identically. Stays opt-in (`qgf_oracle + dual_leo`) with documented unproven G5.

### Examples

- `qgf_01_guided_drafter` — minimal walkthrough: KnownLandscapeOracle + UnitGen drafter, `tilt_logits` hot path, ≥10% E[Q] gain.
- `qgf_02_adaptive_weight` — F4 sigmoid `1/β`: low critic confidence collapses guidance, high confidence activates it.
- `qgf_03_tier_routing` — backend route (CpuSimd/GpuBatch/AneCritic) + oracle tier (Plasma/Hot/Warm/Cold/Freeze).

🔧 Feature flags: `qgf` + `qgf_projector` + `qgf_oracle` + `qgf_drafter` + `qgf_adaptive` (all opt-in, default-off).

📖 Plan: [268](../../.plans/268_qgf_test_time_q_guided_flow.md). Bench: [268](../../.benchmarks/268_qgf_goat.md) (mechanism gates). Sibling Plan: [467](../../.plans/467_qgf_dual_leo_oracle.md). Research: [236](../../.research/236_QGF_Test_Time_Q_Guided_Flow.md). Substrate: `crates/katgpt-core/src/qgf/` (2031 LoC, 6 files).

## 35. Sleep-Time Query Anticipator (Plan 334)

Distilled from [arXiv:2504.13171](https://arxiv.org/abs/2504.13171) — Lin et al. (Letta/Berkeley), *Sleep-Time Compute*. Offline query anticipation primitive: orchestrates per-direction sleep-time compute → emits reusable `AnticipatedQuerySet` (c' artifact, BLAKE3-committed) → wake-time `consume()` does cheap dot-product + sigmoid-gated lookup, falling through to fresh compute on low-predictibility queries.

### The curiosity↔predictability inversion

The load-bearing theoretical contribution: **low-curiosity contexts (on the manifold) → high predictability → should pre-compute; high-curiosity contexts (off the manifold) → low predictability → compute fresh.** A synthetic KARC-like ridge forecaster models context as evolving on a low-dim manifold; the curiosity residual = `||x − forecast(x)||`. The `PredictabilityScorer` trait-swap mechanism lets consumers supply their own predictability model.

### Layer split (katgpt-core leaf → riir-engine runtime)

| Layer | What |
|---|---|
| **katgpt-core** (`sleep_time_anticipation`) | Open primitive: `SleepTimeAnticipator`, `PredictabilityScorer` trait, `DotPredictabilityScorer` default, `AmortizationCostModel`, `IdentityFunctorOp` (synthetic default). Opt-in — re-exports `katgpt-sleep` crate. |
| **riir-engine** (`npc_sleep_time`) | Production runtime: game-specific direction-vector catalogs, NPC tiering, HLA wiring, chain commitment. DEFAULT-ON (Plan 341 Phase 7 GOAT G1–G4 PASS on real corpus). |

The katgpt-core leaf stays opt-in as layer split — the production runtime is riir-engine-side. This mirrors the `zone_affective_manifold` pattern.

### GOAT gate (katgpt-core synthetic gates)

G1/G2/G5/G6/G7 ALL PASS (Bench 334, 2026-06-27) on synthetic fixtures. The quality gates G2/G3/G4 (predictability correlation, cross-player amortization economics, wake-time latency under load) require a real predictability-labeled corpus — those are the riir-ai Plan 341 gates (which PASSed).

### Examples

- `sleep_time_01_basic` — K=4 hardcoded directions, `anticipate()` → c' artifact (BLAKE3-committed), `consume()` → sigmoid-gated blend, amortization cost model at N=1 vs N=10 (~2.5× gain regime).
- `sleep_time_02_curiosity_inversion` — the load-bearing theoretical demo: PredictabilityScorer trait-swap with a custom forecaster.

🔧 Feature flag: `sleep_time_anticipation` (opt-in, `dep:katgpt-sleep`).

📖 Plan: [334](../../.plans/334_sleep_time_query_anticipator_primitive.md). Bench: [334](../../.benchmarks/334_sleep_time_goat.md). Research: [318](../../.research/318_Sleep_Time_Compute_Offline_Query_Anticipation.md). Substrate: `katgpt-sleep` crate (Issue 007 Phase E Tier 2 #6).

## 36. HOLA Hippocampal Exact KV Cache (Plan 395)

Distilled from [arXiv:2607.02303](https://arxiv.org/abs/2607.02303) — HOLA (Hippocampal Ontological Long-context Attention). Surprise-evicted (β·‖e‖) bounded KV cache with decoupled RMSNorm-γ read. The hippocampus-inspired exact-retrieval complement to approximate KV compression: high-surprise tokens get evicted from the main cache but retained in the hippocampal store for exact ancestor-masked softmax recall.

### Architecture

```
Main KV cache (approximate, bounded)
    ↑ surprise signal (β·‖e‖)
    ↓ evict on high surprise
Hippocampal cache (exact, bounded W=64 window)
    ↑ read via ancestor-masked softmax
    ↓ decoupled RMSNorm-γ projection
Output = main_cache_output + hippocampal_complement
```

### GOAT gate

G1–G4 GOAT ALL PASS + consumer wiring PASS (modelless gain). Production wiring in `forward_gdn2` via `HippocampalCacheDyn` (Issue 038). **Opt-in until G5 riir-train gate** (perplexity on real text). Pure stdlib + katgpt-types (no extra deps).

| Gate | Result |
|---|---|
| G1 (correctness) | exact retrieval within window W=64 |
| G2 (synthetic retrieval) | surprise-evicted tokens recalled via ancestor-masked softmax |
| G3 (no-regression) | feature combos clean |
| G4 (alloc-free) | zero hot-path allocation |
| G5 (perplexity on real text) | ⏸ DEFERRED to riir-train |

### GDN-HOLA fusion (Plan 430, §30 in this catalog)

`gdn_hola_tree_verify` fuses the GDN masked triangular solve + HOLA ancestor-masked softmax read for speculative draft tree verification with zero rollback on either path. See §30 for the dual-path GOAT gate.

🔧 Feature flag: `hippocampal_cache` (opt-in). Implied by `gdn_hola_tree_verify`.

📖 Plan: [395](../../.plans/395_hippocampal_exact_kv_cache.md). Bench: `hippocampal_cache_goat` + `hippocampal_cache_retrieval`. Research: [378](../../.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md). Substrate: `crates/katgpt-core/src/hippocampal_cache.rs`.

## 37. Analytic Lattice + Transfer Matrix Band Structure (Plan 330 + Plan 458)

### Analytic Lattice (Plan 330)

Distilled from a fusion of Functional Attention × PJ-RoPE × Gyrocalculus (Research 311). k×k transport operator chain composer + ASOC trait shapes + direction-vector SIMD decoder + spectral audit. The katgpt-core half: pure math primitives (`compose_chain`, `batch_compose_chain`, `direction_vector_decode`, `spectral_audit`) + generic trait shapes (`PlasmaDraft`, `RederiveOp`, `ComposerCtx`) — NO `GpuFuture` import (leaf-clean).

**Layer split:** the `ComposerTick: GpuFuture` impl + `Join3` combinator ship in riir-engine under the `analytic_lattice_runtime` feature (Phase 1b — separate task). The katgpt-core half is opt-in; the runtime gates need a real GPU executor.

**GOAT gate (katgpt-core half):** G1+G2+G3+G5+G6 ALL PASS (10 GOAT tests + 36 unit tests = 46 tests, Bench 330, 2026-06-26). G4 latency + G1b/G1c/G1d non-blocking contract DEFERRED to riir-engine Phase 1b (need GpuFuture).

### Transfer Matrix Band-Structure Analyzer (Plan 458)

Sits on top of `analytic_lattice::compose_chain`: given a sequence of k×k TransportOperators or a periodic stack, compute the composite's eigenvalues, per-mode Bloch factor μ = λ^(1/N), and classify each mode as:

- **Propagating** (|μ|≈1, allowed band)
- **Decaying** (|μ|<1−ε, evanescent)
- **Growing** (|μ|>1+ε, unstable)

The ML anchor is Bai/Koltun/Kolter DEQ Jacobian Regularization (arXiv:2106.14342 ICML 2021): their `ρ(J_*)<1` stability condition is the scalar spectral-radius version of this primitive's per-mode band classifier. Zero-alloc hot paths (`band_classify_into`, `analyze_chain_into`, `analyze_periodic_into`).

**GOAT gate:** G1–G4 ALL PASS (Bench 458, 2026-07-18). G5 (cross-arch bit-identity) deferred to CI. Symmetric-matrix-only Jacobi eigensolver is a documented v1 limitation — non-symmetric chains need a QR-based eigensolver (deferred if a concrete consumer needs it).

🔧 Feature flags: `analytic_lattice` (opt-in) + `transfer_matrix_band_structure` (opt-in, implies `analytic_lattice` for `compose_chain`).

📖 Plans: [330](../../.plans/330_analytic_lattice_encoder_decoder_primitive.md) + [458](../../.plans/458_transfer_matrix_band_structure.md). Benches: [330](../../.benchmarks/330_analytic_lattice_goat.md) + [458](../../.benchmarks/458_transfer_matrix_band_structure_goat.md). Research: [311](../../.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md) + [451](../../.research/451_Delta_Lattice_Tunneling_Transfer_Matrix_Band_Structure.md). Substrate: `crates/katgpt-core/src/analytic_lattice/`.

## 38. Paired Loss Gap Diagnostic (Plan 335)

Distilled from [arXiv:2606.20936](https://arxiv.org/abs/2606.20936) — Li & Merrill (AI2), *Paired Token-Level Loss Gap*. Generic modelless A/B measurement primitive: `PairedLossGap` computes per-token `Δ_i = ℓ_A − ℓ_B` from two log-prob traces; tag-stratified means (Content/Function/Other/BracketOpen/BracketClose/CopyN); filtered aggregates (`ALL_TOKENS` / `TOP-K∩NO-COPY` / `COPY-N-ONLY`) that amplify small architecture gaps aggregate loss hides (paper §6 shows ~2× separation on Olmo 1B).

### ClassSizeBound — the raw-vs-latent sync justification

`ClassSizeBound` exposes Proposition 1: `DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|` — the volume-of-support bound justifying the codebase's raw-vs-latent sync rule (physical small V_τ → raw sufficient; semantic large V_τ → latent earns its keep). This is the theoretical-validation-of-raw-vs-latent artifact documented in Research 319 §2.2.

### GOAT gate

ALL PASS (Bench 335, 2026-06-27). **Not promoted to default** — measurement tool by nature, opt-in is the right shape (you run it when you have two models to compare, not every tick).

| Gate | Result |
|---|---|
| G1 (35 unit tests + bench sanity) | ✅ PASS |
| G2 (`from_log_probs` 0.875µs + `filtered_mean` 1.500µs at L=8192) | ✅ PASS (original 1µs combined target re-spec'd as structurally impossible for 2 memory-bound passes) |
| G2-alloc (scratch-reused SIMD path) | ✅ 0 allocs |
| G3 (feature matrix clean) | ✅ PASS |
| G4 (TopKNoCopy amplifies \|gap\| 13.9× vs AllTokens) | ✅ PASS (target ≥1.5×) |

### Examples

- `paired_loss_01_micro_gpt_ab` — runnable micro-GPT A/B diagnostic: two log-prob traces → PairedLossGap → tag-stratified means table → filtered aggregates table → Proposition 1 annotation.
- `paired_loss_02_class_size_bound` — the raw-vs-latent sync-boundary decision demonstration with a worked annotation.

🔧 Feature flag: `paired_loss_diagnostic` (opt-in). Pure measurement tool, NOT an inference mechanism (Research 319 §3: not Super-GOAT).

📖 Plan: [335](../../.plans/335_paired_loss_gap_diagnostic_primitive.md). Bench: [335](../../.benchmarks/335_paired_loss_goat.md). Research: [319](../../.research/319_Paired_Token_Loss_Gap_Discourse_State_Diagnostic.md). Substrate: `crates/katgpt-core/src/paired_loss_diagnostic.rs`.

## 39. Bisimulation Operator Inference (Plan 324)

Distilled from [arXiv:2602.19260](https://arxiv.org/abs/2602.19260) — Duggan, Lorang, Lu, Scheutz (Tufts), *The Price Is Not Right* (underlying NSM method: [arXiv:2508.21501](https://arxiv.org/abs/2508.21501)). Generic modelless primitive that quotients an observed transition graph into bisimulation-equivalent state classes and infers an abstract PDDL-like operator schema.

### What it produces

Given a stream of observed state transitions `(s, a, s′, label)`:

1. **Minimal bisimulation quotient** — partition of observed states into equivalence classes such that two states are equivalent iff their outgoing labeled transitions lead to equivalent successor classes. Paige-Tarjan O((S+E) log S) partition refinement.
2. **Inferred operator schema** — one abstract operator per edge-label in the quotient graph, with preconditions (src class membership) + effects (dst class membership).
3. **Chain-committable canonical form** — BLAKE3 hash of the quotient graph `(classes, edges, operator_labels)`, suitable for LatCal-style commitment + anti-cheat replay.

### PDDL-side counterpart to Induced CWM (Plan 296 / §10)

Where CWM induces *executable code* from trajectories via an LLM refinement loop (riir-ai Plan 326, private), this primitive induces a *symbolic operator schema* via a deterministic graph algorithm. The runtime picks per task: code induction for rich domains, operator-schema induction for structured/combinatorial domains.

### GOAT gate

ALL PASS (Bench 324, 2026-06-25). Opt-in by design — downstream pipelines opt in; the primitive is not a default-on capability.

| Gate | Result |
|---|---|
| G1 (bisimulation correctness) | Known graph → known minimal quotient, bit-identical across re-runs |
| G2 (operator inference soundness) | Every observed transition covered; no spurious operators |
| G3 (plan validity) | Planner on inferred schema produces executable plans (no precondition violations) |
| G4 (latency) | Partition refinement ≤ 1 ms for N=1024 nodes |
| G5 (zero-alloc hot path) | `class_id(state) -> u32` is O(1), no heap alloc across 10⁶ queries |

### Example

`bisimulation_demo` — Towers-of-Hanoi transition graph: build, refine, infer operators, plan, print quotient + schema.

🔧 Feature flag: `bisimulation_operator_inference` (opt-in). Sibling sub-features: `induced_cwm_ismcts` (Phase 2 IS-MCTS search) + `induced_cwm_tournament` (Phase 3 value-function tournament).

📖 Plan: [324](../../.plans/324_bisimulation_operator_inference.md). Bench: [324](../../.benchmarks/324_bisimulation_goat.md). Research: [308](../../.research/308_NSM_VLA_Price_Is_Not_Right_Bisimulation_Operator_Inference.md). Substrate: `crates/katgpt-core/src/bisimulation/`.

## 40. Lifelong LaCAM Multi-Agent Pathfinding Substrate (Plan 440)

Distilled from [arXiv:2605.16855](https://arxiv.org/abs/2605.16855) — Arita & Okumura, *Lifelong LaCAM with Local Guidance for Lifelong MAPF*, AAAI 2026. Generic modelless receding-horizon windowed multi-agent pathfinder. Scales to 10,000 agents at real-time per-step planning (paper-reported). **Entirely heuristic** — no training, no backprop.

### Four pluggable seams

| Seam | Role |
|---|---|
| `CostFn` | Collision-count cost on space-time A* guidance |
| `LocalGuidanceSource` | PIBT one-step generator |
| `WarmStartScheme` | LLLG_Π / LLLG_Φ / LLLG_∅ warm-start schemes |
| `HindranceEstimator` | Congestion-based hindrance scoring |

These seams enable the riir-ai Super-GOAT fusion (riir-ai/318: HLA × Crowd MCGS × P350) without forking the substrate.

### GOAT gate (honest partial FAIL)

| Gate | Result |
|---|---|
| G3 (no-regression) | ✅ PASS |
| G4 (latency) | ✅ PASS |
| G1 (paper reproduction) | ⚠️ **2/4 real maps PASS** (empty + random), warehouse 0.41 + ht_chantry 0.27 FAIL on real MovingAI maps (Issue 148) |
| G2 (congestion mitigation) | ❌ **FAIL** (warm-start non-consumable) |

**Stays opt-in** per Plan 440 Phase 5 verdict. The G1 partial fail on real MovingAI maps is documented — the substrate works on structured maps but struggles on real-world warehouse/labyrinth topologies. The Super-GOAT promotion requires riir-ai fusion gates G5–G7.

### Sibling: LaCAM Escalation (Plan 453)

`lacam_escalation` (bounded one-step LaCAM) replaces the fake "LaCAM escalation" (shuffled-priority retries) with the real constraint-tree search from Okumura 2023. The critical insight: LaCAM DOES use recursive PIBT, but it works because the constraint tree bounds the recursion. Issues 140/143 collapsed throughput because they used recursive PI WITHOUT the constraint tree. This feature ships the constraint tree — the missing half. All 5 phases DONE; stays opt-in.

🔧 Feature flags: `multi_agent_path` (opt-in) + `lacam_escalation` (opt-in, implies `multi_agent_path`).

📖 Plans: [440](../../.plans/440_lifelong_lacam_multi_agent_pathfinding_substrate.md) + [453](../../.plans/453_bounded_one_step_lacam_escalation.md). Benches: [440](../../.benchmarks/440_lllg_paper_repro_goat.md) + [453](../../.benchmarks/453_lacam_escalation_goat.md). Research: [424](../../.research/424_Lifelong_LaCAM_Local_Guidance_Multi_Agent_Pathfinding.md). Substrate: `crates/katgpt-core/src/multi_agent_path/`.

## 41. FlowField × DualLeoMixer Fusion — Post-Max Potential Blending (Plan 459 + Plan 460)

Composes two DEFAULT-ON primitives — `LeoHead` (Plan 155) + `DualLeoMixer` (Plan 155) + `FlowFieldCache` (Plan 242 Fourier flow fields) — to answer a focused question: does mixing LEO teacher Q-values with UVFA student Q-values produce a better navigation field than the LEO-only baseline? No new feature flag (composition of `flow_field_nav` + `dual_leo`); the API surface is the two new `FlowFieldCache` methods.

### Two fusion points, one decisive lesson

| Plan | Fusion point | Method | G5 (≥30% stuck-NPC reduction) | Verdict |
|---|---|---|---|---|
| **459** | **pre-max** raw Q-slices | `α·Q_leo[a] + (1−α)·Q_uvfa[a]` per action, then `max-over-actions` | ❌ FAIL (best 25.9% at α=0.1) | demoted to compat |
| **460** | **post-max** potentials | `α·potential_leo[x,y] + (1−α)·potential_uvfa[x,y]` after max-pool | ✅ PASS (31.5% at α=0.10) | **recommended path** |

The lesson is the load-bearing finding: **the nonlinear `max-over-actions` step washes out the pre-max α-mix**. `max_a (α·Q_leo + (1−α)·Q_uvfa) ≠ α·max_a Q_leo + (1−α)·max_a Q_uvfa`. Moving the blend to *after* the max-pool makes the blend linear in the FFT's input, so the α-ratio transfers cleanly to the smoothed gradient. The 5.6-percentage-point gain (25.9% → 31.5%) is exactly the size of the nonlinearity that was being washed out.

### Two-stage honest stop rule

1. **Plan 460 G5' PASS on synthetic adversarial heads** (broad multimodal LEO + sharp unimodal UVFA).
2. **Issue 549 real-network follow-up (2026-07-18): G5' FAIL on untrained CivLeoNet + CivLeoUVFA** — stuck-NPC reduction drops from 31.5% (synthetic, α=0.10) to 3.3% (untrained real, α=0.10). The postmax mechanism is correct (G1 bit-identity holds on both synthetic + untrained real) but the gain requires **trained** networks. Tracked in `riir-ai/.issues/552`.

The postmax API stays shipped — the mechanism is sound and `DualLeoMixer` now has a 4th consumer via Plan 467 `DualLeoOracle` (QGF test-time fusion, orthogonal axis). But reading Plan 460 as "the postmax mechanism works on civ" without the untrained-network caveat is misleading.

### The perf measurement honesty footnote

Plan 460's first bench run reported a 3.10× cache-miss overhead (would have failed G2's 1.5× gate). Five subsequent runs showed 0.72×–1.53× — the baseline itself swung 1.91ms→5.47ms. The bench was updated to take 3 trials and report the **median** (stable 1.22–1.26×). Lesson: single-shot `std::time::Instant` on macOS is unreliable for sub-10ms code; median-of-N is the pragmatic middle ground when Criterion (the crate convention) is avoided for cold-cache benches.

### API surface

- `FlowFieldCache::get_or_compute_dual` (Plan 459, pre-max) — compat/parity with QGF pre-max mix. NOT recommended for new flow-field consumers.
- `FlowFieldCache::get_or_compute_dual_postmax` (Plan 460, post-max) — **recommended** dual path.
- `LeoPotentialGrid::blend_into` (Plan 460) — linear affine combination of two post-max grids.

All 5 `ActingMode`s honored (Lc / LeoOnly / UvfaOnly / Max / Min). `LeoOnly` is bit-identical to the single-head path (G1).

🔧 Feature flags: `flow_field_nav` + `dual_leo` (both DEFAULT-ON; the dual methods are opt-in via API choice).

📖 Plans: [459](../../.plans/459_flow_field_dual_leo_mixer_fusion.md) + [460](../../.plans/460_flow_field_dual_leo_postmax_fusion.md). Benches: [459](../../.benchmarks/459_flow_field_dual_leo_mixer_goat.md) + [460](../../.benchmarks/460_flow_field_dual_leo_postmax_goat.md). Substrate: `crates/katgpt-core/src/flow/cache.rs`.

## 42. ICT Distributional Branching-Point Detector (Plan 294)

Distilled from [arXiv:2606.19771](https://arxiv.org/abs/2606.19771) — *Information-Coupling Theory for Multi-Agent Branching*. Detects distributional branching points in agent action distributions: `collision_purity β(π)`, Rényi entropy H₂, JS-divergence-to-group-mean, plus a `BranchingDetector` that flags top-k% outliers via stable sort.

### GOAT gate (G1–G10, ALL PASS)

| Gate | Target | Result |
|---|---|---|
| G1 (mechanics) | analytic anchors on synthetic distributions | ✅ PASS |
| G2 (synthetic) | branching-point recovery | ✅ PASS |
| G3 (no-regression) | feature combos clean | ✅ PASS |
| G4 (latency) | ≤ 50µs/call at K=8 D=32 | ✅ PASS — **1.96µs mean, p99 2.00µs** (25× headroom) |
| G5 (alloc-free) | 0 allocs after warmup | ✅ PASS — 0 allocs / 1000 calls |
| G6 (feature isolation) | cargo + nm verify | ✅ PASS |
| G7–G10 (Plan 324 T9.4 follow-ups) | integration + composition | ✅ ALL PASS (2026-06-20+) |

The hot path is 1.96µs because the inner loops are chunked-4 (per AGENTS.md "write chunked 4-wide loops so LLVM autovectorizes"): K × action_dim = 256 f32 adds autovectorize to NEON/AVX2, K × JS-divergence uses chunked-4 m-buffer + scalar log accumulator.

### Sibling: Bisimulation Operator Inference (Plan 324)

Plan 324 (§39) ships the PDDL-side operator inference; ICT (Plan 294) ships the action-distribution branching detector. The two together cover the action-side + state-side of multi-agent branching-point discovery.

🔧 Feature flag: `ict_branching` (opt-in).

📖 Plan: [294](../../.plans/294_ict_branching_detector.md). Benches: [G1](../../.benchmarks/294_ict_g1.md), [G2](../../.benchmarks/294_ict_g2.md), [G3](../../.benchmarks/294_ict_g3.md), [G4-G6 aggregate](../../.benchmarks/294_ict_goat_gates.md), [G10](../../.benchmarks/294_ict_g10.md), [promotion](../../.benchmarks/294_ict_promotion.md). Research: [270](../../.research/270_Beyond_Entropy_ICT_Distributional_Branching_Detector.md). Substrate: `crates/katgpt-core/src/ict/`.

## 43. FORE — Fitted Occupancy-Ratio Estimator (Plan 438)

Distilled from [arXiv:2607.05375](https://arxiv.org/abs/2607.05375) — van der Laan & Kallus, *Fitted Occupancy-Ratio Evaluation without Bellman Completeness*, 2026. Generic modelless fitted-iteration estimator for the discounted occupancy ratio `ω_{π,γ} = d^π,γ / d^ν` in offline policy evaluation.

### The substrate-independent contribution

The **adjoint Bellman KL contraction** (Lemma 3.1): `B^γ_π ω = (1−γ)ω_0 + γ·d((ων)P_π)/dν` contracts relative entropy by factor γ per iteration. FORE converges under realizability alone — **no Bellman completeness needed**. This is the load-bearing theoretical contribution distilled into the primitive.

### GOAT gate (G1–G5 ALL PASS, stays opt-in pending consumer)

| Gate | Target | Result |
|---|---|---|
| G1 (correctness) | Baird-MRP anchors within 2% rel err | ✅ PASS — 0.31% (upper), 0.74% (lower) at n=100k K=50 γ=0.95 |
| G2 (perf) | FORE fit n=10000 D=8 K=20 < 100ms | ✅ PASS — 48.63ms median |
| G3 (no-regression) | clippy + lib tests clean | ✅ PASS |
| G4 (alloc-free) | KL-projection loop 0 allocs after warmup | ✅ PASS — 0 allocs/100 calls |
| G5 (modelless) | no GD through base weights | ✅ PASS — only mutable state is `θ: Vec<f32>` |

**Stays opt-in** — promotion to default-on requires a downstream consumer (Fusion A CLR stabilization in `riir-poc`) to validate the gain empirically. The primitive is correct + fast + alloc-free, but has zero in-tree consumers today.

### Softmax carve-out

FORE's normalized exponential class is density-ratio normalization, not direction-vector projection — the sigmoid rule does not apply (same carve-out as `product_key_memory`).

🔧 Feature flag: `occupancy_ratio` (opt-in).

📖 Plan: [438](../../.plans/438_occupancy_ratio_estimator_primitive.md). Bench: [438](../../.benchmarks/438_occupancy_ratio_goat.md). Research: [423](../../.research/423_Adjoint_Bellman_KL_Contraction_Occupancy_Ratio.md). Substrate: `crates/katgpt-core/src/occupancy_ratio.rs`.

## 44. Group Invariance Probe — Lie Subgroup Discovery (Plan 356)

Distilled from [arXiv:2512.20043](https://arxiv.org/abs/2512.20043) — LieFlow symmetry discovery. The modelless residue: generalizes `subspace_phase_gate` from "subspace of `ℝᵈ`" to "subgroup of `G`" via direct invariance testing `σ(β·(1−d(q,g·q)))` + a dual-signal discrete-vs-continuous classifier.

### The dual-signal classifier (key design finding)

The discrete-vs-continuous classification needs **two complementary signals** because no single measure handles both regimes:

| Regime | Variance | Concentration | Correct signal |
|---|---|---|---|
| Large-fraction discrete (C₄ ⊂ C₈, 50%) | ≈ 0.25 (bimodal) | ≈ 0.5 (indistinguishable) | **Variance** |
| Small-fraction discrete (C₄ ⊂ C₆₄, 6%) | ≈ 0.06 (low) | ≈ 0.06 (peaked) | **Concentration** |
| Continuous (SO(2) ⊂ SO(3)) | low | high | Neither → Continuous |
| No symmetry (uniform low) | ≈ 0 | low | support < min → None |

`classify_subgroup` fires `Discrete` if EITHER signal triggers — the OR of two complementary detectors.

### GOAT gate (8/8 ALL PASS, stays opt-in)

| Gate | Target | Result |
|---|---|---|
| G1 (correctness) | C₈→C₄ recovers Discrete, n_support≥100, max_score>0.95 | ✅ PASS — n_support=131, max_score≈1.0 |
| G2a (no symmetry) | uniform low → None | ✅ PASS |
| G2b (continuous) | uniform high → Continuous | ✅ PASS |
| G2c (small-fraction discrete) | 4 peaks/64 → Discrete | ✅ PASS (via concentration) |
| G3a/b (no-regression) | `--all-features` + `--no-default-features` clean | ✅ PASS |
| G4a/b (alloc-free) | `discover_subgroup_into` + `classify_subgroup` 0 allocs | ✅ PASS |

**Stays opt-in** — no in-tree consumer today. Ships as the open primitive layer; the IP-bearing consumer-side fusion lives in riir-ai.

🔧 Feature flag: `group_invariance_probe` (opt-in).

📖 Plan: [356](../../.plans/356_group_invariance_probe.md). Bench: [356](../../.benchmarks/356_group_invariance_probe_goat.md). Research: [355](../../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md). Substrate: `crates/katgpt-core/src/group_invariance_probe.rs`.

## 45. FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory (Plan 278)

Distilled from the Self-Evolver cognitive integrity line (Research 244). Audit-cadence diagnostic: detects whether an injected memory vector is *faithfully consumed* by a downstream consumer, via causal intervention (perturb the injection, measure the consumer's output delta). Not a per-tick primitive — invoked at audit cadence (sleep cycle, GM inspection, drift detection).

### Four perturbation strategies

| Strategy | What it tests |
|---|---|
| `Empty` | Does the consumer behave identically with a zeroed injection? (faithful consumers diverge) |
| `Shuffle` | Does within-vector permutation matter? (position-sensitive consumers diverge) |
| `Corrupt` | Does Gaussian noise injection flip the output? (robustness probe) |
| `Irrelevant` / `Filler` | Does a semantically-unrelated injection change behavior? (specificity probe) |

### Composition with TriggeredInjectionGate

The probe ships alongside `TriggeredInjectionGate` (gated by `triggered_injection`) — a sigmoid-gated injection controller that decides per-tick whether to inject. The probe validates the gate's decisions at audit cadence: a gate that injects but the consumer doesn't faithfully use the injection is a wasted injection.

### GOAT gate (G1 + G1b + G2 + perf, ALL PASS)

| Gate | Target | Result |
|---|---|---|
| G1 (faithful consumer detected) | linear consumer passes | ✅ PASS |
| G1b (unfaithful consumer detected) | copy-ignoring consumer fails | ✅ PASS |
| G2 (attribution ranking, simplified) | Spearman ρ ≥ 0.8 vs reference IG on linear consumer | ✅ PASS (full IG deferred to Phase 3) |
| Perf (TriggeredInjectionGate latency) | ≤ target | ✅ PASS |

**Stays opt-in** — audit-cadence diagnostic, not every-tick. 24/24 Phase 1 unit tests; `ProfilePod` is 16 bytes (Copy); `Intervention` enum is `#[repr(u8)]` (1 byte).

### Layer split

The open primitive (katgpt-core `faithfulness_probe`) is the generic IP-free diagnostic substrate. The IP-bearing private consumer-side guide lives at `riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md` (the cognitive integrity layer composition).

🔧 Feature flags: `faithfulness_probe` (opt-in) + `triggered_injection` (opt-in, for the gate).

📖 Plan: [278](../../.plans/278_faithfulness_probe_modelless.md). Bench: [278](../../.benchmarks/278_faithfulness_probe_goat.md). Research: [244](../../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md). Private guide: [riir-ai 129](../../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md). Substrate: `crates/katgpt-core/src/faithfulness/`.

## 46. CODA Fused SIMD Kernels — Single-Pass Matmul+Residual+RMSNorm+Activation (Plan 103)

CODA-inspired fused SIMD kernels that combine matmul + residual + rmsnorm + activation into single-pass loops, eliminating intermediate buffer writes. **50% buffer write reduction** per layer (14 → 5 passes) with zero numerical drift (self-consistency cosine = 1.00000000).

### Per-layer buffer analysis

| Path | Passes/layer | Notes |
|---|:---:|---|
| Baseline (separate kernels) | ~14 | rmsnorm + memcpy + matmul + residual add × 2 (QKV + MLP) |
| CODA Fused | ~5 | Kernel 1: out_proj+residual+partial_rms; Kernel 2: matmul+rmsnorm+activation; Kernel 3: down_proj+residual |
| **Savings** | **64% reduction** | exceeds stretch goal of 30% |

### GOAT gate (15/15 ALL PASS, stays opt-in)

| Gate | Target | Result |
|---|---|---|
| G1 (correctness) | ε < 1e-5 | ✅ PASS — all logits finite, ε < 1e-5 |
| G2 (decode speedup micro) | ≥ 5% | ✅ PASS — perf parity at micro scale |
| G3 (buffer write reduction) | ≥ 20% | ✅ PASS (stretch) — **50% (10→5 passes)** |
| G4 (feature isolation) | compiles w/wo | ✅ PASS — no overhead when disabled |
| G5 (numerical stability) | cosine ≥ 0.9999 | ✅ PASS (stretch) — **1.00000000 self-consistency** |

**Stays opt-in** — micro-bench perf parity (the kernel fusion eliminates writes but doesn't show end-to-end decode speedup at the micro scale tested). The real gain materializes at the memory-bandwidth-bound decode regime on larger models. Forwarded to `katgpt-forward` so `ForwardContext.coda_partial_sums` compiles.

🔧 Feature flag: `coda_fusion` (opt-in; forwards to `katgpt-core` + `katgpt-forward`).

📖 Plan: [103](../../.plans/103_coda_fused_simd_kernels.md). Bench: [030](../../.benchmarks/030_coda_fusion_simd.md). Substrate: `crates/katgpt-core/src/coda_fusion.rs` + `crates/katgpt-forward/src/coda.rs`.

## 47. Energy-Gated Attention (EGA) — Spectral Salience Gating (Plan 139)

Gates value aggregation by the spectral energy of key token embeddings. Per-head `EgaGate { w_proj, α, τ }` projects keys to a 1-D energy score, z-normalizes, applies a sigmoid gate `g = σ(α·(ẽ − τ))`, then scales attention weights `Â_ij = A_ij · g_j` and renormalizes.

### Architecture

```
e = X · w_proj              [seq_len] energy scores
ẽ = z_normalize(e)          z-normalized energy
g = σ(α · (ẽ − τ))          sigmoid gate vector
Â_ij = A_ij · g_j           gate each key position
Â_ij /= Σ_k(Â_ik + ε)       renormalize (sum-to-one)
Y = Â · V                   value aggregation
```

### GOAT gate (ALL PASS, stays opt-in)

All gates pass on synthetic energy distributions. Stays opt-in because the gate parameters (`w_proj`, `α`, `τ`) need to be **trained** (per-head energy projection) — this is not a modelless primitive. The z-normalize + sigmoid gate machinery is modelless, but the `w_proj` projection vector is a learned parameter. Consumers opt in when they have trained `EgaGate` parameters.

🔧 Feature flag: `ega_attn` (opt-in; in `katgpt-attn`).

📖 Plan: [139](../../.plans/139_ega_energy_gated_attention.md). Benches: [038 GOAT](../../.benchmarks/038_ega_attn_goat.md) + [046 examples](../../.benchmarks/046_ega_examples_goat.md). Substrate: `crates/katgpt-attn/src/ega.rs`.

## 48. Epiplexity Structural Information Scoring (Plan 130)

Distilled from [arXiv:2601.03220](https://arxiv.org/abs/2601.03220) — *Epiplexity: Structural information extractable by computationally bounded observers*. Measures structural information as **area under the loss curve above the final loss**: `S_T = Σ max(0, loss_i − final_loss)`. Paired with `TimeBoundedEntropy H_T = final_loss × n_tokens` + structural fraction `S_T / H_T`.

### The screening pruner composition

`EpiplexityScreeningPruner<P>` blends a relevance signal from inner pruner `P` with the epiplexity signal: `relevance = inner.relevance() × (1−α) + epiplexity_signal × α`. Three `EpiplexityWeight` variants: `Uniform`, `LossDrop`, `CumulativeArea`.

### GOAT gate (ALL PASS, stays opt-in)

| Test class | Result |
|---|---|
| Constant loss → flat S | ✅ PASS |
| Structured loss (decreasing) → S > 1.0 | ✅ PASS |
| More structure → higher S | ✅ PASS |
| Per-sample monotonicity | ✅ PASS |
| Ring buffer caps at capacity | ✅ PASS |

**Stays opt-in** — the primitive is a correct building block for modelless distillation data selection, but requires a distillation harness (training-loop integration) to demonstrate the data-selection gain. Used by `epiplexity_bandit` (the SR²AM extension) which is DEFAULT-ON.

🔧 Feature flag: `epiplexity_scoring` (opt-in; in `katgpt-pruners`). Implied by `epiplexity_bandit` (DEFAULT-ON).

📖 Plan: [130](../../.plans/130_epiplexity_structural_information_scoring.md). Benches: [014 screening](../../.benchmarks/014_epiplexity_screening_bench.md) + [041 GOAT](../../.benchmarks/041_epiplexity_structural_information_goat.md). Research: [090](../../.research/090_Epiplexity_Structural_Information_Computationally_Bounded_Observers.md). Substrate: `crates/katgpt-pruners/src/epiplexity.rs`.

## 49. GPU Inference Backend — CubeCL Metal Compute Pipelines (Plan 176)

Metal GPU inference backend via [CubeCL](https://github.com/gabrielbizon/CubeCL) compute pipelines. Fused layer dispatch for Gemma 2 2B on Apple Silicon (M-series). Subgroup size 32 (Metal SIMD width). Autotune system selects optimal GEMV variant per dimension.

### The GeGLU double-gate bug fix (the load-bearing correctness finding)

`gelu_tanh(x)` already computes `0.5 · x · (1 + tanh(...))`, which includes the `x` factor. The CubeCL code multiplied by the gate `g` an extra time: `g · gelu_tanh(g) · u` (buggy) → `gelu_tanh(g) · u` (fixed). This is a subtle correctness bug that would silently corrupt GeGLU activations — the kind of bug that only surfaces under bit-exact verification against a reference implementation.

### GOAT status

| Gate | Result |
|---|---|
| Correctness (GeGLU bug fixed) | ✅ PASS |
| Autotune (selects optimal variant per dim) | ✅ PASS |
| CubeCL F32 decode parity with WGSL | ⏸ pending sync reduction |

**Stays opt-in** — Metal-specific (Apple Silicon only); the CubeCL dep is heavy. The `inference_router` feature combines this with ANE routing for full inference path selection.

🔧 Feature flags: `gpu_inference` (opt-in; implies `kog_cpu_fusion`) + `inference_router` (opt-in; implies `gpu_inference` + `ane`).

📖 Plan: [176](../../.plans/176_ane_inference_backend.md). Bench: [029](../../.benchmarks/029_cubecl_gpu_rewrite.md). Substrate: `crates/katgpt-backend/src/gpu_backend.rs`.

## 50. Motor-Gated DEC Field — Amari Neural-Field Evolution (Plan 357)

Distilled from [arXiv:2602.18690](https://arxiv.org/abs/2602.18690) — Amari-style motor-gated neural-field evolution. Unifies `hodge_laplacian` (Stokes DEC) + latent steering into a single grid-stencil evolution step: the motor gate `ReLU(h)` modulates which channels of the cochain field are active, then the Hodge Laplacian diffuses the gated field.

### GOAT gate (G1–G5 ALL PASS, stays opt-in by design)

| Gate | Metric | Result | Threshold |
|---|---|---|---|
| G1 (no-teleporting) | max centroid displacement / 50 ticks | **0.0009 cells** | ≤ 2.0 cells |
| G2 (motor-gate locality) | channel isolation ratio | ∞ (no leak) | > 100× |
| G3 (conservation) | `|Σ K[ReLU(h)]| / L1(h)` | 0.0000 | < 0.05 |
| G4 (zero-alloc) | allocs / 1000 ticks (64×64×16) | 0 | = 0 |
| G5 (latency) | per-call (64×64×16, release) | **~29 µs** | < 100 µs |

The grid-stencil fast path (Issue 001 fix) closed the G5 gap decisively: **120 µs → 29 µs** (4.1× speedup, 3.4× margin under target). The feature stays **opt-in by design** — it's a primitive for downstream consumption (riir-ai Research 168 Phase 2), not a default-on capability.

### Sibling DEC features

| Feature | Status | Role |
|---|---|---|
| `heat_kernel_trajectory` | DEFAULT-ON | heat-kernel trajectory on cochains |
| `sheaf_admm` | DEFAULT-ON | sheaf ADMM consensus |
| `grid_3d` | DEFAULT-ON | 3D cell-complex NCA |
| `se2_equivariant_lift` | DEFAULT-ON | SE(2) rotation-equivariant lift |
| `cochain_point_sampler` (§16) | opt-in | point sampling on cochains |
| `htno_v_cycle` (§13) | opt-in | multi-scale V-cycle |
| **`motor_gated_field`** (this) | opt-in | motor-gated field evolution |

🔧 Feature flag: `motor_gated_field` (opt-in; in `katgpt-dec`).

📖 Plan: [357](../../.plans/357_motor_gated_dec_propagation_primitive.md). Bench: [357](../../.benchmarks/357_motor_gated_field_goat.md). Substrate: `crates/katgpt-dec/src/motor_gated.rs`.

## 51. Flow Field Navigation — Fourier-Smoothed LEO Crowd Flow (Plan 242)

Distilled from Treuille et al. *Continuum Crowds* (2006) + the LEO Q-value framework. When 100+ NPCs share the same goal (e.g., "go to town square"), running individual LEO Q-value lookups per NPC per tick is wasteful. A shared 2D flow field computed once per tick, FFT-smoothed to eliminate local minima, lets all NPCs read their gradient direction via O(1) lookup.

**Key insight:** LEO already computes Q-values per goal. `LeoHead::all_goals_q()` produces `[goals × actions]`. For spatial goals, the max-Q action per cell IS a flow vector. FFT smoothing the resulting potential field removes discretization noise and local minima.

### Why opt-in

Only helps for crowd scenarios (many entities, shared goals). Individual explorers or small groups (<20) won't benefit. The FFT compute cost must amortize over enough NPCs.

### Dependencies

- `leo_all_goals` (DEFAULT-ON) — LeoHead + all_goals_q
- `dep:rustfft` — FFT for smoothing
- `spectral_hierarchy` (DEFAULT-ON) — Jacobi/Haar FFT pipeline

This feature is the **substrate dependency** of §41 (FlowField × DualLeoMixer Fusion). The dual-LEO mixing experiments compose on top of `FlowFieldCache`.

🔧 Feature flag: `flow_field_nav` (opt-in). Implies `leo_all_goals` + `dep:rustfft`.

📖 Plan: [242](../../.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md).

## 52. Wall Attention — Diagonal Forget Gates Replacing RoPE (Plan 173)

Wall Attention replaces RoPE with diagonal forget gates. Each token accumulates a per-head per-dim prefix sum `P_t = Σ_{i≤t} log(f_i)` where `f_i` is a learned forget gate. The factorized form `q̃_i = exp(P_i) ⊙ q_i`, `k̃_j = exp(−P_j) ⊙ k_j` means standard attention kernels work unchanged after Q/K rescaling.

### Key design decisions

1. Wall **replaces** RoPE entirely when enabled (not additive — paper confirms Wall(NoPE) > Wall+RoPE).
2. Key-projected gate variant (derive gate from K) is preferred for zero KV cache overhead.
3. Gate bias initialized to 6.0 (open-gate init matching vanilla attention).
4. KV-head gate tying by default in GQA configs (one gate per KV head).
5. Algorithmically identical to standard attention after Q/K rescaling — no attention kernel changes.

### Decode vs prefill

- **Decode:** maintain running `P_t` prefix sum (O(1) update per token), rescale only the current query.
- **Prefill:** compute prefix sum once over all positions, rescale Q and K in one pass.
- **Per-layer isolation:** prefix sums indexed by `[layer_idx × n_kv_head × head_dim + kv_head × head_dim + d]`.

### Sibling position-encoding features

| Feature | Status | Role |
|---|---|---|
| `wall_attention` (this) | opt-in | diagonal forget gates replacing RoPE |
| `position_group_action` (§21) | opt-in | unified position-encoding trait |
| `grapem_rodrigues` (§20) | opt-in | rank-2 Rodrigues exponential |
| `rotary_value_embedding` | opt-in | RoVE — rotary on V projection |

🔧 Feature flag: `wall_attention` (opt-in; forwards to `katgpt-types/wall_attention`).

📖 Plan: [173](../../.plans/173_wall_attention_diagonal_gate.md).

## 53. Sense Composition Family — KG Latent Octree + Children (Plan 221 + children)

A family of six opt-in features implementing modelless inference-time sense composition. The core idea: compress game domain knowledge into fixed-type ternary bit-plane sense modules (~232B each). Each module encodes a KG latent octree + direction vectors. NPCs compose modules at spawn time and query at ~45ns/tick via bitwise dot-product.

### Features

| Feature | Plan | Role |
|---|---|---|
| `sense_composition` | 221 | KG Latent Octree — modelless sense module composition |
| `merkle_octree` | 221-M | Merkle octree hierarchical commitment for SenseModule |
| `schema_centroid` | 237 | Schema-Centroid Informed KG Embedding Initialization |
| `bake_precision` | 236 | BAKE Precision-Gated Bayesian Embedding Update |
| `spectral_threat` | 241 | LinOSS Modal Threat Prediction |
| `sense_lod` | 240 | Spectral NPC Perception Compression |

### Architecture

- **Substrate lives in** `katgpt-sense` crate (Issue 007 Phase E Tier 2). `katgpt-core` forwards the feature + re-exports.
- `sense_composition` enables the re-export + forwards the flag.
- Each child feature gates a specific submodule.
- **GM override always wins** over autonomous behavior — every autonomous path has a manual override.
- **Fail-safe defaults:** if a sense module returns garbage, NPC falls back to rule-based behavior.

### Dependencies

`sense_composition` implies `plasma_path` (TernaryWeights) + `domain_latent` (DomainLatent). The octree + direction vectors live in fixed-type ternary bit-planes.

### Companion plans

- **riir-ai Plan 249:** model-based training counterpart (sense module learning via GD).
- **seal-online-remaster Plan 036:** Brain Annotation — KG/HLA schema metadata for GameComponent derive.

🔧 Feature flags: `sense_composition` (parent, implies plasma_path + domain_latent), `merkle_octree` (implies sense_composition), `schema_centroid` (implies sense_composition), `bake_precision` (implies sense_composition), `spectral_threat` (implies sense_composition + modal_spec), `sense_lod` (implies sense_composition + slod).

📖 Plan: [221](../../.plans/221_kg_latent_octree_sense_composition.md). Research: [196](../../.research/196_KG_Latent_Octree_WASM_Composition.md).

## 54. MUX Superposition Family — Vocabulary-Simplex Pruning + Tree Search (Research 158)

A family of seven opt-in features implementing superposition-based tree search. The core idea: instead of pruning vocabulary branches deterministically, maintain a superposition over multiple branches and collapse only when needed.

### Features

| Feature | Role |
|---|---|
| `mux_pruner` | MuxSpanPruner — vocabulary simplex pruning |
| `mux_ddtree` | MuxDdTree — superposition branch DDtree nodes (implies mux_pruner) |
| `mux_bandit_width` | MuxBanditWidth — adaptive superposition width |
| `mux_bfs` | MUX BFS — superposition-guided parallel tree search (implies mux_ddtree) |
| `mux_freeze_thaw` | MUX Freeze/Thaw — persistent superposition patterns (implies mux_pruner) |
| `comp_width` | Compositional DDTree partner-entropy width — continuous replacement for PEAK_DOMINANCE_RATIO (Plan 205) |
| `mux_demux` | MuxDemux Verifier — deterministic superposition recovery |

### Related MUX-latent features (Phase 12 absorption)

| Feature | Role |
|---|---|
| `mux_latent_context` | MUX-Latent Context Compression — inference-time context compression via vocabulary superposition (DEFAULT-ON in root) |
| `mux_latent_wire` | MUX-Latent Wire Patch — latent-to-latent patching without decompress/recompress (implies mux_latent_context) |
| `lclm_adaptive_lod` | LCLM Adaptive LOD — spectral energy-based adaptive compression ratio (implies mux_latent_context) |

### Architecture

The MUX family operates on the speculative decoding tree substrate. `mux_pruner` provides the vocabulary simplex pruning kernel; `mux_ddtree` extends DDtree nodes to carry superposition branches; `mux_bfs` performs parallel tree search guided by superposition; `mux_bandit_width` adaptively controls how many branches to maintain; `mux_freeze_thaw` persists patterns across sessions; `mux_demux` provides deterministic verification of the collapsed result.

🔧 Feature flags: all opt-in in `katgpt-core`. Root forwards them.

📖 Research: [158](../../.research/158_MUX_Multiplexed_Latent_Reasoning.md).

## 55. DEC Operator Substrate — Cell Complexes + Cochain Fields (Plan 251)

The foundational Discrete Exterior Calculus (DEC) substrate: cell complexes, cochain fields, and the d/δ/Δ operators (exterior derivative, codifferential, Hodge Laplacian). This is the substrate that §13 (htno_v_cycle), §16 (cochain_point_sampler), §50 (motor_gated_field), and the AGENTS.md Stokes Calculus vocabulary all build on.

### Key insight

Topology determines WHERE information flows (fixed); learning determines HOW features are mixed. For modelless inference, we only need the fixed part. DEC operators provide structured, conservation-guaranteed alternatives to ad-hoc gradient/flow computations.

### What ships

- `CellComplex` struct (vertices, edges, faces, volumes with incidence)
- `CochainField` typed cochain on cell complex
- `BoundaryMatrix` (sparse signed incidence matrix Bₖ as triplets)
- `exterior_derivative` (d = boundary operator)
- `codifferential` (δ = discrete divergence)
- `hodge_laplacian` (Δ = δd + dδ)
- `hodge_decompose` (exact ⊕ harmonic ⊕ coexact = Helmholtz)
- Tests enforce `BₖBₖ₊₁ = 0` (boundary-of-boundary is zero) by construction

### Layer split

The substrate now spins in the `katgpt-dec` crate and re-exports as `katgpt_core::dec` (Issue 007 Phase E Tier 1). The `dec_operators` feature in `katgpt-core` gates the re-export.

### Why opt-in at katgpt-core level

The `katgpt-dec` crate is always compiled; the `dec_operators` feature in `katgpt-core` gates only the re-export path. Consumer features (`motor_gated_field`, `heat_kernel_trajectory`, `sheaf_admm`, `se2_equivariant_lift`, `cochain_point_sampler`, `htno_v_cycle`, `tropical_algebra`) each imply `dec_operators` when needed.

### Vocabulary translation (AGENTS.md §Manifold Geometry)

| Paper term | Code primitive |
|---|---|
| divergence / flux / ∇·F | `codifferential`, δ |
| boundary / ∂M / frontier | `exterior_derivative`, d |
| line integral / trajectory energy | rank-1 `CochainField` sum along path |
| Stokes / divergence theorem | DEC identity d∘d=0, `hodge_decompose` |
| Hodge decomposition / Helmholtz | `hodge_decompose`, `DecFlowField` |
| Fokker-Planck / continuity equation | `codifferential` on belief cochain |

🔧 Feature flag: `dec_operators` (opt-in; gates `dep:katgpt-dec`).

📖 Plan: [251](../../.plans/251_dec_operators_cell_complex.md). Research: [219](../../.research/219_Topological_Neural_Operators_DEC_Inference.md).

## 56. Causal Head Importance Family — Activation Patching + Relocation (Plan 358 + Proposal 004 + Plan 431)

A family of three opt-in features for modelless causal intervention diagnostics on attention heads. Distilled from HydraHead (arXiv:2606.20097) + the Knowing-Using Gap paper (arXiv:2607.08393).

### Features

| Feature | Plan | Role |
|---|---|---|
| `causal_head_importance` | 358 | CausalHeadImportance + ScaleNormalizedFusion — activation patching (Eq 10) + path patching (Eq 11) + span-level logit-diff readout (Eq 9) + cross-capability fusion (Eq 12) |
| `adaptive_causal_calibration` | Proposal 004 | AdaptiveCausal calibration — cheap OV-circuit proxy escalates to Plan 358's causal patching only on k suspects instead of all n_heads. OUR INVENTION, not from HydraHead. |
| `cross_stage_relocation` | 431 | Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic (arXiv:2607.08393). Two primitives: `permeation_scan_into` (2D intervention heatmap) + `RelocateOp` (snapshot residual at one stage, overwrite at another). |

### Key findings

- **causal_head_importance (Plan 358):** GOAT G1-G4 ALL PASS. Stays opt-in — `causal_necessity` loses the RTPurbo calibration slot competition to `attention_mass` on most workloads (no quality gain).
- **adaptive_causal_calibration (Proposal 004):** The proxy (attention_mass / ||OV_out||) is an UNVALIDATED hypothesis — promotion blocked on G1 (proxy precision) + G2 (cost reduction), both deferred to riir-engine. Ships the open primitive leaf-clean.
- **cross_stage_relocation (Plan 431):** Phase 3 defend-wrong PoC DONE — verdict: REFUTE `LateEarly` default (CLOBBERS in 2/4 clean configs because op_b overwrites op_a's recovery). The mechanism itself works (single-op relocation recovers in all configs). Production path: permeation-map diagnostic + `RelocatePair::Custom`.

### Pure numeric substrate

All three operate on `&[f32]` + a caller-supplied patched-forward-pass closure. Zero deps. The patched forward pass itself is riir-engine/riir-games territory; these are the scorers/operators.

🔧 Feature flags: `causal_head_importance` (opt-in), `adaptive_causal_calibration` (implies causal_head_importance), `cross_stage_relocation` (implies causal_head_importance).

📖 Plans: [358](../../.plans/358_causal_head_importance_calibration.md), [431](../../.plans/431_cross_stage_residual_relocation_primitive.md).

## 57. Thinking/Cognition Substrate — Collapse Detection + Capability Routing + Belief Drafting (Plan 212 + 216 + 217)

A family of three opt-in features forming the thinking/cognition inference-time substrate.

### Features

| Feature | Plan | Role |
|---|---|---|
| `collapse_aware_thinking` | 212 | Collapse-Aware Adaptive Thinking — `CollapseDetector` trait for detecting reasoning collapse. Forwards to `katgpt-types/collapse_aware_thinking`. |
| `substrate_gate` | 216 | SubstrateGate passthrough — inference-time capability substrate routing. Empty feature gate; gates the module in `katgpt-core/src/`. |
| `belief_drafter` | 217 | NextLat Belief-State Speculative Drafter — entropy threshold config for belief-state speculative generation. Forwards to `katgpt-types/belief_drafter`. |

### Architecture

`collapse_aware_thinking` provides the `CollapseDetector` trait that consumers implement to detect when a reasoning chain has collapsed (repetition, divergence, etc.). `substrate_gate` routes inference requests to the appropriate capability substrate. `belief_drafter` provides the entropy threshold configuration for speculative belief-state generation.

These three are substrate primitives consumed by the CGSP (Curiosity-Guided Self-Play, Plan 274) runtime and other cognition pipelines.

🔧 Feature flags: all opt-in. `collapse_aware_thinking` + `belief_drafter` forward to `katgpt-types`; `substrate_gate` is a `katgpt-core`-local module gate.

📖 Plans: [212](../../.plans/212_collapse_aware_adaptive_thinking.md), [216](../../.plans/216_substrate_gate_capability_routing.md), [217](../../.plans/217_nextlat_belief_state_drafter.md).

## 58. Game Episode Evolution — Hierarchical Decomposition + Self-Play (Plan 190 + 191)

A family of four opt-in features for game-episode evolution and self-play curriculum.

### Features

| Feature | Plan | Role |
|---|---|---|
| `and_or_dtree` | 190 | AND-OR DDTree — generic AND-OR tree for hierarchical goal decomposition |
| `partial_scoring` | 191 | PartialScorer — graduated reward for game episodes |
| `problem_mutator` | 191 | ProblemMutator — game config evolution |
| `idea_divergence` | 191 | IdeaDivergence — strategic novelty filter (implies partial_scoring) |

### Architecture

`and_or_dtree` provides the generic hierarchical goal decomposition tree (AND nodes require all children, OR nodes require one). `partial_scoring` provides graduated reward signals for partial episode completion. `problem_mutator` evolves game configurations to generate curriculum. `idea_divergence` filters for strategic novelty in generated ideas.

These compose with CGSP (Plan 274, `cgsp` feature) and the speculative generator substrate (`speculative_generator`, Plan 193).

🔧 Feature flags: all opt-in in `katgpt-core`.

📖 Plans: [190](../../.plans/190_and_or_dtree_blueprint_decomposition.md), [191](../../.plans/191_open_ended_problem_evolution_arena.md).

## 59. RTDC Family — Resolution-Tiered Deterministic Commitment (Plan 302 + Issue 002)

A family of two opt-in features for multi-resolution Merkle commitment of game state.

### Features

| Feature | Plan | Role |
|---|---|---|
| `rtdc` | 302 | Resolution-Tiered Deterministic Commitment — multi-resolution Merkle roots per SLoD σ-boundary. Phase 1: open primitive only; LatCal-backed encoding + chain quorum live in riir-chain (Plan 003). Implies `slod` + `merkle_octree` + `sense_composition`. |
| `rtdc_subtree_inclusion` | Issue 002 | RTDC Phase 3 Candidate C — probabilistic cross-depth consistency proof. Curator publishes 73-hash octree + seed; verifier samples K leaves and checks parent reconstruction. Implies `rtdc`. |

### GOAT gate (rtdc_subtree_inclusion)

Bench 303 CG6 PASS (2026-06-22): cost gate PASSES at 4.60× vs 5.0× target (8% headroom). Detection sub-criteria verified. Stays opt-in per Bench 303 verdict — promote to candidate-default once `chain_rtdc_subtree` wiring lands.

### Architecture

RTDC composes the SLoD spectral pruner (Plan 235) + Merkle octree (Plan 221-M) + sense composition (Plan 221) into a multi-resolution commitment scheme. Each SLoD σ-boundary gets its own Merkle root; light clients can verify at the resolution tier they care about.

🔧 Feature flags: `rtdc` (opt-in; implies slod + merkle_octree + sense_composition), `rtdc_subtree_inclusion` (opt-in; implies rtdc).

📖 Plan: [302](../../.plans/302_rtdc_open_primitive.md). Bench: [303](../../.benchmarks/303_rtdc_subtree_inclusion_goat.md).

## 60. Additional Significant Opt-In Features

A consolidated section for standalone opt-in features with their own plans but not large enough for a dedicated section.

### Attention & inference variants

| Feature | Plan | Role |
|---|---|---|
| `tiled_attention` | 115 | Tiled online-softmax flash attention for CPU SIMD |
| `parallax_attn` | 135 | Parallax parameterized local linear attention (implies tiled_attention) |
| `moa_inference` | 158 | MoA Mixture of Activations — token-adaptive activation mixing |
| `deltanet_inference` | 182 | DeltaNet GPU inference — hybrid DeltaNet/Attention decode |
| `lt2_looped` | 108 | LT2 looped inference types — LoopMode, HybridPattern, SdpaOutputGate, ResidualGate |
| `tf_loop` | 136 | Training-free loop wrapper — ODE-refined sub-stepping (implies lt2_looped) |

### KV cache & memory

| Feature | Plan | Role |
|---|---|---|
| `rim_slots` | 172 | RiM Reasoning Buffer Slots — fixed latent workspace for DDTree |
| `drift_segment` | 652 / Research 482 | DriftSegmentStore — training-free drift-segmented multi-state memory: rising-edge drift boundaries open slots, adjacent-density merge enforces capacity-K (arXiv:2606.10650 modelless; Bench 635 GOAT PASS — G1 +46.09pp change-point / +75.00pp stationary needle recall vs fixed-LFU at matched budget, 12 ns/token, 0 allocs; consumers: riir-ai `npc_episodic` Bench 675 + neuron-db wake-merge policy Bench 480; promotion candidate at next re-gate) |
| `product_key_memory_freeze` | 408 Phase 4 | FrozenProductKeyMemory — freeze/thaw wrapper with BLAKE3 commitment |
| `product_key_memory_episodic` | 408 Phase 5 / Issue 650 | PkmEpisodicStore — δ-rule write gate (PKM × δ-Mem fusion) + TF-IDF non-interference slot selection (`write_idf`/`write_weighted_idf` + `BackgroundAccessStats` + `write_selected`; Bench 636 GOAT G1 +12.5pp retention at matched learning) |
| `mop_path_entropy` | 573 | MOP value-iteration primitive — reward-free optimal policy (paper Eq. 7 log-space LSE fixed point; `MopSolver` + `pi_star` + shared arenas; Bench 638 GOAT G1–G4 PASS, stays opt-in pending riir-ai Plan 538 integration) |
| `chunked_net_fetch` | 272 T3.3 | NetChunkFetcher — network chunk fetcher stub for ChunkedContentStore |

### Direction & steering diagnostics

| Feature | Plan | Role |
|---|---|---|
| `dirichlet_energy` | 149 | Dirichlet Energy structural alignment diagnostic |
| `gain_cost_halt` | 304 | Gain/Cost Loop Halting Primitive — per-loop halting kernel. GOAT G1-G5 ALL PASS. Stays opt-in at katgpt-core leaf; production wiring is riir-ai civ-side. |
| `smear_classifier` | 298 | SmearClassifier — ternary latent-mass distribution classifier. GOAT G1/G2/G3 ALL PASS. Stays opt-in (G2 evidence synthetic). |
| `self_advantage_gate` | 283 | Self-advantage recursion gate for HLA reconstruction |
| `recursion_logits` | 283 | RecursionLogits opt-in trait — pre/post recursion logits exposure |

### Routing & gating

| Feature | Plan | Role |
|---|---|---|
| `rv_gated_routing` | 202 | RV-Gated Compute Routing — AcceptanceVarianceTracker integration + TriggerGate::rv_tier_boost |
| `ssmax_adaptive` | 411 S2 | SSMax built-in rolling-Δ estimator. GOAT G1-G5 ALL PASS. Stays opt-in pending real-attention PoC. |
| `gold_share_probe` | 411 | GoldShare — content-specific output-fraction diagnostic |
| `indicator_cascade` | 320 Phase 3 | IndicatorCascade — two-stage verifier escalation. Stays OPT-IN permanently (implies stage-2 verifier impl, consumer-crate territory). |

### GPU & quantization

| Feature | Plan | Role |
|---|---|---|
| `binary_plasma` | Issue 145 | Binary {−1,+1} plasma tier — single bit-plane, group-wise FP16 scale. The fastest Plasma tier; ternary (plasma_path) moved to Hot. |
| `ternary_trit_pack` | Issue 582 | `TernaryTritWeights` — base-3 packing, 5 trits/byte, **1.725 bits/weight** vs the bit-plane tier's 2.125 (−18.8%; Bonsai-27B 5.82 GB vs 7.16 GB). G1–G4 GOAT ALL PASS and, against the filed prediction, **1.10–1.15× faster** than the bit-plane kernel on NEON as well as smaller ([Bench 582](../../.benchmarks/582_trit_pack_goat.md)). The AVX2 leg is measured too: on x86_64 trit is **15–31% slower** than bit-plane SIMD ([Bench 586](../../.benchmarks/586_avx2_ternary_t4_measurements.md)) — the wider AVX2 lanes favour SWAR over LUT decode. Implies `ternary_group_scale`. Opt-in for the same policy reason as its parent; on CPU it is the better choice for a ternary consumer **on aarch64**, and a footprint-only choice on x86_64. |
| `ternary_group_scale` | Issue 578 (closed) | `TernaryGroupWeights` — ternary {-1,0,+1} bit-planes + per-128 f16 group scale, the `Q2_0_g128` container (Ternary-Bonsai-27B). G1–G4 GOAT gate ALL PASS (2026-08-12), stays opt-in by policy: promoting it would transitively promote `binary_plasma` (opt-in by deliberate Issue 145 decision) and the tier is a model-specific container 1.3× slower than row-scale ternary. **Issue 650 (2026-08-13) added the block-contiguous AoS companion layout** (`TernaryBlockAoS` + `TernaryBlockContiguousWeights`, same feature gate — one 34-byte `#[repr(C)]` block per 128-weight group, G1 bit-identical to the SoA matvec) — and the GPU investigation **resolved as a negative result on M3 Metal**: 3.12× vs sequential but **0.82× vs the existing SoA simdgroup kernel** (worse coalescing). Bonus finding that closed it: SoA simdgroup already measures 5.89–6.27× vs sequential on M3 Metal, so the motivating 1.89× ceiling is not reproducible. The types stay in-tree as validated code for potential CUDA use. Implies `binary_plasma`. See [`../08_performance/ternary_group_q2_0_tier.md`](../08_performance/ternary_group_q2_0_tier.md) (incl. the §AoS negative result). |
| `gpart_adapter` | 257 | GPart isometric partition adapter loading |
| `gpart_pruning` | Issue 008 | GPart top-k group pruning — zero out low-magnitude groups at apply time (implies gpart_adapter) |
| `simd_sigmoid` | Issues 024/025 | SIMD-vectorized sigmoid→tanh→clamp fused pass for AttractorKernel::step() + BoMSampler |

### Smaller primitives

| Feature | Plan | Role |
|---|---|---|
| `questbench` | 110 | QuestBench underspecification scoring for modelless architecture |
| `peira_distill` | 153 | PEIRA inter-view regressor alignment |
| `rat_plus_bridge` | 225 | RAT+ Recurrence Bridge — modelless dilated inference via GDN2 state |
| `dendritic_gate` | 260 | DendriticGate NMDA-inspired adaptive branching types |
| `hydra_budget` | 165 | Hydra-Aware Adaptive Layer Budget types |
| `review_metrics` | 054 | ReviewMetrics — inference-time path-consistency / reward-hacking counter |
| `elasticity_gated_update` | 429 | Elasticity-Gated Update — DSOM error-scaled neighborhood update. Consumer GOAT PASS in riir-neuron-db; `elasticity_gated_heal` PROMOTED to default-on there. |
| `sphere_sampling` | Issue 544 | Sphere Sampling — modelless primitives for sampling from unnormalized densities on S^{d-1}. Opt-in pending defend-wrong PoC. |
| `newton_schulz` | 152 | Newton-Schulz orthogonalization + Muon momentum. NOT in katgpt-core `default`; root's default-on forwarder activates it. |
| `binary_plasma` | Issue 145 | Binary {−1,+1} plasma tier |
| `rt_turbo` | 126 | RTPurbo retrieval head sparse decode via low-dim indexing (root-level opt-in; implies `dash_attn`) |
| `spec_compile` | — | Full spec compilation suite — SpecAsPruner + SpecAsMarginals + SpecDFA + SpecProof + SpecChain (root-level opt-in; implies `spec_pruner`) |
| `stokes_calculus` | 314 | Stokes Calculus Wrappers — belief_mass_divergence + boundary_flux_mass + line_integral + circulation_integral (root alias for `katgpt-core/dec_operators`). Stays opt-in — see [negative_results §35](negative_results.md#35-stokes-calculus-wrappers---g-c-structural-fail--g-a-runtime-fail-stays-opt-in) for G-C/G-A FAIL details. |

### Speculative decoding substrate markers

These are tracking flags that gate substrate types in `speculative/types.rs`. Root forwards them so the gated items resolve:

| Feature | Plan | Role |
|---|---|---|
| `stability_metrics` | 102 | StabilitySnapshot + DraftResult.stability field |
| `spec_cost_model` | 096 | SpecCostSnapshot + DraftResult.cost_snapshot field |
| `kurtosis_gate` | 203 | RejectionReason::KurtosisRejection variant |
| `elf_sde` | 079 | EarlyStopGate<P> depth-aware screening wrapper |
| `tes_loop` | 086 | TesNode + TrajectoryCredit |
| `lattice_deduction` | 088 | LDT conflict detector + LdtPruneConfig substrate |
| `echo_env_predictor` | 247 | BudgetAdaptation::EchoConsistency variant |
| `q_sample_solver` | — | q-sampling for critical steps (implies critical_interval_gate) |
| `self_cond_draft` | — | 2-pass self-conditioned speculative draft |
| `mbr_tree_select` | — | MBR selection from DDTree |
| `d2f_3sr_warm_start` | 291 | D2F 3SR warm-start config |
| `rcd_residual` | 258 | Residual Context Diffusion — entropy-weighted residual injection (implies critical_interval_gate) |

### Phase 12 absorption (DEFAULT-ON in root, opt-in here)

These features are marked "DEFAULT-ON in root" but not in katgpt-core's `default`. The root crate's default-on forwarder activates them:

| Feature | Plan | Role |
|---|---|---|
| `critical_interval_gate` | 222 | Discrete Critical Interval Solver Switching. Transitively DEFAULT-ON via root's `rcd_residual`/`closure_instrument`. |
| `modality_pruned_load` | 227 Phase 3 | Modality-Pruned Context Loading — query classifier + pipeline selection |
| `mux_latent_context` | 238 | MUX-Latent Context Compression (DEFAULT-ON in root) |
| `closed_unit_compaction` | 333 | Closed-Unit Compaction Gate — rubric-gated trajectory compaction (DEFAULT-ON in root) |
| `breakeven_routing` | 250 | Breakeven complexity cost-aware inference routing (DEFAULT-ON in root) |
| `memory_soup_lora` | 290 | Memory Soup LoRA — MSP0 binary format parser |
| `skill_opt` | 144 | SkillOpt text-space skill optimization |
| `channel_simd_align` | 227 Phase 5 | Channel SIMD Alignment — cache-line-padded weight storage (DEFAULT-ON in root, promoted 2026-08-11 per Bench 580: 84.9%/86.7% release-mode throughput improvement) |

## 61. NFCoT Flow Family — Normalizing Flow Continuous CoT Drafting (Plan 229)

A family of six opt-in features for modelless normalizing-flow-based speculative drafting. Distilled from the NFCoT paper (Continuous CoT via normalizing flows).

### Features

| Feature | Plan | Role |
|---|---|---|
| `nf_flow_score` | 229 T1 | FlowScore Drafter — modelless normalizing flow density scoring for draft acceptance |
| `nf_flow_gate` | 229 T3 | FlowGate — adaptive acceptance criterion based on flow density |
| `nf_flow_budget` | 229 T4 | FlowBudget — sigmoid-weighted speculative depth allocation |
| `nf_flow_mux` | 229 T6 | FlowMUX — flow scoring for MUX trajectories (implies mux_pruner) |
| `nf_flow_fold` | 229 T7 | FlowFold — confidence-gated chain folding (implies chain_fold) |
| `nf_flow` | 229 | Parent feature — enables all flow components |

### Architecture

The normalizing flow provides a density estimate over the token space, which replaces the heuristic acceptance criteria in standard speculative decoding. The flow score drives three independent mechanisms: draft acceptance (FlowGate), depth budgeting (FlowBudget), and chain folding (FlowFold). FlowMUX composes with the MUX pruner family (§54) for superposition-based trajectories.

All components are default-OFF pending GOAT gate validation. Phase 12 (2026-07-04): `nf_flow_generator` + `nf_flow_qgf` modules moved to `katgpt-speculative`.

🔧 Feature flags: all opt-in. `nf_flow_score` + `nf_flow_budget` forward to `katgpt-speculative`; `nf_flow_gate` is root-local.

📖 Plan: [229](../../.plans/229_nf_flow_score_drafter.md). Research: [204](../../.research/204_NFCoT_Normalizing_Flow_Continuous_CoT.md).

## 62. FOL-LNN Rule Inference Family — DDTree→FOL Pipeline (Plan 209)

A family of features for first-order logic rule inference — extracting logical constraints from prompts and decision trees, then using them as modelless pruners.

### Features

| Feature | Plan | Default | Role |
|---|---|---|---|
| `fol_constraints` | 209 T1 | opt-in | FOL constraint extraction from prompts — keyword table + FolPruner |
| `rule_extraction` | 209 T2 | opt-in | Logical rule extraction from DDtree paths — ExtractedRule + RuleExtractor |
| `reward_mem` | 209 T3 | **DEFAULT-ON** | Reward-Weighted Branch Memorization — blake3 pattern hashing + EMA reward tracking (GOAT 6/6) |
| `decision_trace` | 209 T4 | **DEFAULT-ON** | Interpretable decision traces (transitively in `default` via `regex` dependency) |
| `egcs` | 206 | **DEFAULT-ON** | Episode-Guided Constraint Synthesis — reference-based structural constraint injection (GOAT 4/4) |
| `fol_lnn` | 209 | opt-in | Parent feature — all FOL-LNN fusions convenience |

### Architecture

The pipeline flows: prompt → `fol_constraints` (keyword table extraction) → `rule_extraction` (DDTree path → logical rule) → `reward_mem` (BLAKE3 pattern hash + EMA reward) → `decision_trace` (interpretable trace output). `egcs` (Episode-Guided Constraint Synthesis) provides the reference-based structural injection layer. `reward_mem` + `decision_trace` + `egcs` are all DEFAULT-ON (GOAT-proved); the constraint/rule extraction layers stay opt-in.

🔧 Feature flags: `fol_constraints` + `rule_extraction` + `fol_lnn` opt-in; `reward_mem` + `decision_trace` + `egcs` DEFAULT-ON. All forward to `katgpt-pruners`.

📖 Plan: [209](../../.plans/209_fol_logical_rule_inference.md). Bench: [209](../../.benchmarks/209_fol_lnn_goat.md).

## 63. K-Prior Algorithmic Probability Sampler Family (Plan 305)

A family of three opt-in features for algorithmic-probability-based K-prior sampling — injecting Solomonoff-style complexity priors into bandit/MCTS/speculative decision paths.

### Features

| Feature | Plan | Role |
|---|---|---|
| `complexity_prior_sampler` | 305 | Open primitive — algorithmic-probability sampler + coincidence gate |
| `bandit_k_prior` | 305 T3.2 | Bandit K-prior wrapper (KPriorBandit<K>) exposing arm_log_prior |
| `mcts_k_prior` | 305 T3.1 | MCTS expansion-prior adapter (MctsExpansionPrior / UniformExpansion / KPriorExpansion) |
| `spec_k_prior` | 305 T3.3 | Speculative drafter post-drafting re-ranker (KPriorDrafter<K>::rerank) |

### Architecture

The `complexity_prior_sampler` open primitive computes algorithmic-probability-based priors. Three adapters inject these priors into different decision frameworks: bandit arm selection (KPriorBandit), MCTS expansion (MctsExpansionPrior), and speculative draft re-ranking (KPriorDrafter). The caller adds the adapter to their existing framework — no invasive changes.

🔧 Feature flags: all opt-in. All forward to `katgpt-pruners`.

📖 Plan: [305](../../.plans/305_algorithmic_probability_sampler.md).

## 64. CoExplain Bidirectional Alignment — Pruner Evolution (Plan 214)

A family of three opt-in features for self-refining constraint pruners with bidirectional alignment. Distilled from the CoExplain methodology.

### Features

| Feature | Plan | Role |
|---|---|---|
| `ted_lite` | 214 P1 | TED-Lite Divergence Metric — pruner drift measurement |
| `coexplain_pruner` | 214 P2+3 | Self-Refining + Editable ConstraintPruner (implies ted_lite + bandit) |
| `coexplain_riir` | 214 P4 | RIIR Feedback Loop — Curator marketplace enabler (implies coexplain_pruner) |

### Architecture

`ted_lite` measures pruner drift (how far a pruner's behavior has diverged from its baseline). `coexplain_pruner` uses this drift signal to create a self-refining constraint pruner that can be edited by a curator. `coexplain_riir` extends this with a feedback loop for a curator marketplace — pruners evolve based on external feedback.

🔧 Feature flags: all opt-in. All forward to `katgpt-pruners`.

📖 Plan: [214](../../.plans/214_coexplain_bidirectional_alignment.md).

## 65. Cubical Category Interval Topology for Inference (Plan 252)

A family of three opt-in features for cubical-category-based interval topology — applying CAT(0) cubical complex theory to inference-time token set constraints.

### Features

| Feature | Plan | Role |
|---|---|---|
| `interval_pruner` | 252 Phase 1 | IntervalPruner — interval-closure for valid token sets |
| `lattice_operad` | 252 Phase 2 | LatticeOperad — canonical AND/OR pruner expression composition (`PrunerExpr` enum + `canonicalize()` + `eval()` + operadic composition + distributive lattice word problem solver). **DEFAULT-ON** (Phase 12 absorption — module in katgpt-pruners, root forwards). GOAT T25-T27 validated (Plan 252 Phase 4). Pure modelless (lattice algebra). |
| `cubical_nerve` | 252 Phase 3 | CubicalNerve — CAT(0) cubical complexes from zone posets |
| `cubical_topology` | 252 | Parent feature — canonical pruner composition (implies interval_pruner + lattice_operad + cubical_nerve) |

### Architecture

`interval_pruner` provides interval-closure computation for valid token sets (which tokens are valid within a given interval). `cubical_nerve` constructs CAT(0) cubical complexes from zone posets — a topological structure that captures the constraint geometry. `cubical_topology` composes these with the lattice operad into a canonical pruner.

Phase 12 (2026-07-04): `interval_pruner` moved to `katgpt-pruners`; `cubical_nerve` moved to `katgpt-core`. The 2026-07-18 cargo-comment sync corrected a stale "DEFAULT-ON in root" claim — both are genuinely opt-in.

🔧 Feature flags: all opt-in. `interval_pruner` → `katgpt-pruners`; `cubical_nerve` → `katgpt-core`.

📖 Plan: [252](../../.plans/252_cubical_category_interval_topology.md). Research: [220](../../.research/220_Convenient_Category_Cubes_Interval_Topology.md).

## 66. Lean4Agent Formal Verification Fusion (Plan 223)

A family of three opt-in features for formal-verification-guided inference — Hoare-logic-based predicate propagation, failure localization, and sigmoid-graded relevance.

### Features

| Feature | Plan | Role |
|---|---|---|
| `hoare_pruner` | 223 | Predicate propagation — Hoare-logic pre/post conditions as inference pruners (implies llmexec_guard) |
| `trajectory_doctor` | 223 | Failure localization — identifies where a trajectory diverged from its specification (implies hoare_pruner) |
| `workflow_lattice` | 223 | Sigmoid-graded relevance — lattice-structured workflow scoring (implies hoare_pruner) |

### Architecture

`hoare_pruner` applies Hoare-logic pre/post conditions as inference-time pruning constraints — tokens that would violate the post-condition are pruned. `trajectory_doctor` localizes failures by identifying the first point where a trajectory diverges from its formal specification. `workflow_lattice` provides sigmoid-graded relevance scoring on a lattice structure.

All three depend on `hoare_pruner` which in turn depends on `llmexec_guard` (DEFAULT-ON). Phase 11 (2026-07-04): `hoare_pruner` also forwards to `katgpt-validator` so `SynPruner::propagate` resolves.

🔧 Feature flags: all opt-in. All forward to `katgpt-pruners` + `katgpt-validator`.

📖 Plan: [223](../../.plans/223_lean4agent_formal_verification_fusion.md).

## 67. Self-Advantage Recursion Gate Family (Plan 283)

A family of three opt-in features extending the Self-Advantage recursion gate (already documented in §60 as `self_advantage_gate`) — dead-compute detection via pre/post log-ratio, weight-shared loop integration, and personality fingerprinting.

### Features

| Feature | Plan | Role |
|---|---|---|
| `self_advantage_gate` | 283 | Self-advantage recursion gate for HLA reconstruction (see §60) |
| `weight_shared_advantage_gate` | 283 T2.2 | Wire AdvantageMarginGate into forward_looped weight-shared loop — breaks early on dead-compute iterations (implies self_advantage_gate) |
| `product_policy_sharpen` | 283 | Product-policy inference sharpening — controllable reasoning trust weight |
| `advantage_freeze_thaw` | 283 T5.3 | AdvantageDirectionSnapshot + EMA accumulator — per-NPC personality fingerprint via BLAKE3-committed A(·) direction |

### Architecture

The `self_advantage_gate` (§60) detects dead compute via pre/post recursion log-ratio. `weight_shared_advantage_gate` integrates this into the weight-shared inference loop for early termination on dead iterations. `product_policy_sharpen` provides a trust-weight mechanism for sharpening inference decisions. `advantage_freeze_thaw` captures the advantage direction as a per-NPC personality fingerprint (BLAKE3-committed), enabling freeze/thaw of the agent's behavioral signature.

All four distilled from arXiv:2511.16886 (Self-Advantage Recursion).

🔧 Feature flags: all opt-in. `product_policy_sharpen` + `advantage_freeze_thaw` forward to `katgpt-pruners`.

📖 Plan: [283](../../.plans/283_self_advantage_recursion_gate.md). Research: [250](../../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md).

## 68. Additional Standalone Opt-In Features (Extended)

A second consolidated table for standalone opt-in features with their own plans, complementing §60.

### Speculative decoding & drafters

| Feature | Plan | Role |
|---|---|---|
| `domino_lora` | 231 | Domino LoRA causal correction adapter for speculative decoding. Plan 387: moved to `katgpt-speculative`; Plan 394: forwarded to `katgpt-forward`. |
| `acceptance_forecast` | — | Acceptance rate forecasting for speculative draft budgeting |
| `velocity_field_ensemble_heterogeneous` | — | Heterogeneous-D velocity fields via Cross-Resolution transport (implies velocity_field_ensemble + cross_resolution_transport) |
| `mtp_lora_drafter` | — | Multi-token-prediction LoRA drafter |
| `ssc_spec_draft` | — | SSC speculative draft variant |

### Attention & memory

| Feature | Plan | Role |
|---|---|---|
| `msa_*` family | 256 | MSA blockwise sparse attention — **3× GOAT FAILED**, see [negative_results §36](negative_results.md#36-msa-blockwise-sparse-attention-family---3-goat-failed-stays-opt-in-permanently) |
| `still_kv` | 245 | StillKV perceiver-based KV cache compaction — modelless |
| `maxsim` | 080 | MaxSim late-interaction scoring (Research 45). Forwards to `katgpt-quant` (TurboQuant + OCTOPUS integration). |
| `engram` | 299 | Engram — hash-addressed, sigmoid-fused static pattern memory (arXiv:2601.07372) |
| `memory_soup_dtree` | — | Experimental DDTree branch merging |
| `emotion_vector` | — | Emotion vector primitive |

### Tokenization & vocabulary

| Feature | Plan | Role |
|---|---|---|
| `convex_tok` | 127 | ConvexTok LP vocabulary optimizer (Research 087). Forwards to `katgpt-tokenizer`. |
| `toast_tokenizer` | 122 | ToaST split-tree tokenization (Research 081). Forwards to `katgpt-tokenizer`. |
| `datrie_vocab` | — | Double-array trie vocabulary lookup |
| `fast_bpe` | — | Fast BPE tokenizer variant |

### Search & graph

| Feature | Plan | Role |
|---|---|---|
| `progressive_mcgs` | 272 | Progressive MCGS — graph search with reference edges + entropy-gated schedule (Research 239) |
| `set_diffusion` | 401 | Set Diffusion — set-causal attention + DecodeStrategy::SetDiffusion |
| `hlplayer` blends | 436 | `binned_blend` (HARMFUL), `kernel_blend` (RECOMMENDED), `contextual_bandit` — see [negative_results §37](negative_results.md#37-binned-blend-estimator---real-arena-strictly-harmful-stays-opt-in) |

### Formal verification & proof

| Feature | Plan | Role |
|---|---|---|
| `proof_cert` | 145 | Hierarchical GOAT Proof Certificates — formal verification methodology (Research 106) |
| `proof_sketch_evolution` | 128 | Proof Sketch Evolution — Elo-rated population + global goal cache (Research 088) |
| `ruliology` | 188 | Ruliology Bandit — simple program strategies as bandit arms (Research 168). **GOAT PASS 2026-08-05 (Bench 572)** — 97/97 release tests; stays opt-in (niche tool). |
| `symbolic_distill` | 210 | Symbolic Expression Distillation — compact polynomial expressions for DDTree boundaries (GOAT 6/6, default-ON in root) |

### Hardware & deployment

| Feature | Plan | Role |
|---|---|---|
| `flashar_anchor` | 166 | FlashAR strided anchor-then-fill D2F (Research 149). Forwards to `katgpt-forward`. |
| `flashar_consensus` | 166 / 651 | FlashAR Consensus Tri-Mode with Ternary Thermal Paths (Research 149). Issue 651: Warm/Cold = FLARE Eq 21 exact acceptance, slot-aligned; Plasma/Hot skip-biased by design |
| `hardware_aware_scheduler` | 339 | Hardware-Aware Prefix Scheduler — multi-request verification budget allocator (DSpark §3.2.2) |
| `moka_ane` | — | Moka on Apple Neural Engine via CoreML |

### Skill & lifecycle

| Feature | Plan | Role |
|---|---|---|
| `skill_lifecycle` | 192 | Inference-Time Skill Evolution — per-pruner memory + test-gated registration + progressive disclosure catalog |
| `hebbian_kernel_memory` | 559 | Closed-Form Fact-Storing MLP Construction + MLP Swap (arXiv:2607.10034). **DEFAULT-ON in katgpt-core** — see §69. |

### Pruners & routing

| Feature | Plan | Role |
|---|---|---|
| `bckvss` | — | Backward-compatible KV state sharing |
| `bfcf_lfu_shard` | — | BFCF LFU shard variant |
| `bfcf_tree` | — | BFCF tree variant |
| `manifold_pruner` | — | Manifold-based pruning |
| `spectral_budget` | — | Spectral budget allocation |
| `expression_pruner` / `expression_pruner_dep` | — | Expression-based pruners |
| `fpcg_selector` | — | FPCG selector |
| `hoare_pruner` family | 223 | See §66 (Lean4Agent Formal Verification) |
| `cubical_topology` family | 252 | See §65 (Cubical Category Interval Topology) |
| `rv_bandit_pruning` | — | River-valley bandit pruning |
| `rv_gated_thinking` | — | River-valley gated thinking |
| `safe_bandit` | — | Safe exploration bandit |
| `safe_exploration_budget` | — | Safe exploration budget allocator |
| `sdpg_bandit` | — | SDPG bandit |
| `selectivity_router` | — | Selectivity router |
| `smooth_min_rerank` | — | Smooth-min re-ranking |

### Extended pruners, routing & decision (second batch)

These are additional standalone features with their own plans that were not covered by the family sections above:

| Feature | Plan | Role |
|---|---|---|
| `bandit_mcts` | 067 | Bandit-guided MCTS rollout policy — NFSP/MCTS duality |
| `budget_adaptation` | 167 | Compression-adaptive decode budget — PFlash complexity signal |
| `caddtree_budget` | 219 | CaDDTree — Cost-Aware Adaptive DDtree Budget Selection (GOAT 7/7, **DEFAULT-ON**) |
| `cgsp_dual_pool` | 282 | Dual-Pool Reachable Memory Router — DecentMem distillation (arXiv:2605.22721) |
| `deep_manifold` | 085 | Deep Manifold fixed-point residual scoring (Research 51) |
| `federation` | 085 | Deep Manifold federated boundary alignment — KL coupling (Research 51) |
| `federation_composer` | 231 | Explicit Model→Agent→Tool pipeline with residual checking (GOAT 7/7, **DEFAULT-ON**) |
| `lodestar` | 207 | Lodestar Completion-Distance Pruning — shortest-accepting-distance powers budget-aware masking |
| `nexus_elo` | 143 | Nexus Elo — Plackett-Luce + P-UCB + goal cache for DDtree/SR²AM (Research 104) |
| `ppot` | 026 | PPoT logit-parameterized CPU resampling. Forwards to `katgpt-speculative`. |
| `thinking_cot` | 194 | Adaptive CoT thinking — self-learning when to think |
| `thinking_prune` | 171 | Thinking Prune — FrozenBaseGuard for SpecHop/LT2 intermediate steps (Research 153) |
| `wealth_pruner` | 187 | WealthPruner — economic bandit arms via Hayek market selection (Research 167) |
| `adaptive_cot_compaction` | 271 | Entropy-thresholded bandit-tuned online compaction |
| `data_gate` | 111 | Task-level data gating for self-play training stability (Research 075) |
| `vortex_flow` | — | VortexFlow attention substrate (parent of MSA family) |

### Remaining smaller primitives (cross-reference)

These are opt-in substrate/utility features without dedicated plans or with minimal documentation. Listed for completeness:

`adaptive_gamma_forecast`, `async_qdq_overlap`, `attn_match`, `auto_constraint_synthesis`, `bandit_top_p`, `best_buddies`, `bfcf_lsh_cms`, `chiaroscuro`, `cna_steering`, `corr_budget`, `cs_kv_probe`, `cumprodsum`, `curvature_alloc`, `data_probe`, `dense_mesh`, `directional_credit`, `dmax_spd`, `domino_correction`, `eqr_convergence`, `fastrand`, `feature_class`, `four_regime_router_dep`, `freq_bandit`, `frozen_base_guard`, `functional_substitution_gate`, `future_probe`, `game_domain`, `game_state`, `gated_mlp`, `gauge_invariant`, `gepa_reflective`, `hla_attention`, `hybrid_oct_pq`, `ilc_distill`, `iso_quant`, `kv_share`, `kvarn`, `manifold_power_iter_router`, `mech_attribution`, `memo_reflections`, `micro_belief`, `mls_aggregate`, `module_energy_route`, `nds_proxy`, `nexus_elo_proxy`, `opus_selection`, `osc_kv`, `outlier_guard`, `pathway_tracker`, `percept_route`, `phrase_boost`, `planar_quant`, `precision_aware_draft`, `quantile_balance_router`, `randopt_weight`, `recfm`, `regime_transition`, `replaid_schedules`, `reward_calibrator`, `rosetta_pruner`, `segment_checkpoint`, `self_distilling_bandit`, `sigmoid_graded_reject`, `sleep_consolidation`, `sp_kv`, `spec_reconciliation`, `spectral_pruner`, `spectral_quant`, `spectral_rank`, `ss_pruner`, `state_source`, `static_cal_tables`, `step_attribution_qualifier`, `stiff_anomaly`, `swir_switch_thinking`, `targeted_precision`, `thicket_variance_probe`, `three_mode_router`, `trd_refined_draft`, `trust_region_spec`, `unit_distance`, `vocab_channel_pruner`, `vocab_coreset`, `wealth_pruner`.

## 69. DEFAULT-ON Undocumented Features (Cross-Reference)

These features are DEFAULT-ON (in katgpt-core or root) with significant plans/GOAT benches but were not previously documented in the opt-in catalog. They are listed here for completeness — they do NOT need opt-in catalog entries, but a future default-features doc should cover them.

### Significant DEFAULT-ON primitives

| Feature | Plan | Default layer | Role |
|---|---|---|---|
| `manifold_erasure` | 426 | katgpt-core | MANCE — Manifold-Aware Concept Erasure. Local tangent + spectral weighting + trust-bounded erasure. Pure modelless linear algebra. GOAT G1–G7 ALL PASS. |
| `dreamer` | 107 | root | Auto-Dreamer offline memory consolidation scheduler. GOAT 8/8. Full pipeline: scheduler → consolidator → decay → counterfactual. |
| `reward_mem` | 209 T3 | root | Reward-Weighted Branch Memorization — blake3 pattern hashing + EMA reward tracking. GOAT 6/6. (See §62.) |
| `river_valley` | 152 | root | River-valley diagnostic metrics — subspace ratios, effective rank, cosine similarity. GOAT 25/25 (Bench 050). Substrate in `katgpt-spectral`. |
| `union_bound_confidence` | 231 | root | Union bound additive branch confidence. GOAT 6/6. |
| `hebbian_kernel_memory` | 559 | katgpt-core | Closed-Form Fact-Storing MLP Construction + MLP Swap. GOAT G1+G2+G3+G4 ALL PASS (Bench 559). (See §68.) |
| `ica_lens` | 475 | root (forwards `katgpt-spectral`) | ICA Lens — FastICA non-Gaussian direction mining + ERF diagnostic. GOAT G1–G5 ALL PASS (Bench 475). The missing third corner of direction acquisition: unsupervised-statistical. (See README showcase.) |
| `similarity_inference` | 526 | katgpt-core — **DEMOTED to opt-in 2026-09-04** (Issue 867 T1.3: 24 days default-on, zero consumers) | Similarity Inference — endogenous correlation device from joint-action history. GOAT G1–G8 ALL PASS (Bench 579). Infers the `ω` that shipped CCE (Plan 295) takes exogenously. (See README showcase.) |
| `channel_simd_align` | 227 Phase 5 | root | Channel SIMD Alignment — cache-line-padded weight storage for vectorized matvec. GOAT G1–G5 ALL PASS (Bench 580: 84.9%/86.7% release-mode throughput). The last of 6 QAT Infusion phases. (See §60 Phase 12 absorption table.) |

### Other notable DEFAULT-ON features (substrate/utility)

These are DEFAULT-ON features that form the substrate of the default build. They are listed for cross-reference — most are small utility/substrate primitives that don't warrant individual catalog entries:

`sparse_mlp`, `plasma_path`, `leo_all_goals`, `dual_leo`, `sigmoid_margin`, `spectral_hierarchy`, `dual_gram_pca`, `roofline_cost`, `octree_ctc`, `sector_projection`, `action_bridge`, `triggered_injection`, `temporal_deriv`, `bom_sampling`, `personality_composition`, `depth_invariance`, `cross_resolution_transport`, `latent_field_steering`, `viable_manifold_graph`, `ac_prefix`, `geometric_product`, `fourier_continuation`, `spectral_differentiation`, `tucker_factorization`, `arg_protocol`, `indicator_probe_bank`, `indicator_similarity`, `phase_rotation_coupling`, `spherical_steering`, `closure_instrument`, `non_interference_branches`, `funcattn_structured_basis`, `best_belief`, `committed_field_blend`, `tropical_algebra`, `temp_loss_fingerprint`, `zone_density_routing`, `set_attention`, `clr_weighted_set_attention` (Plan 570 — CLR-amplified reliability-weighted sibling; closes Set Attention G8), `manifold_bandit`, `mean_field_regime`, `qmc_sampling`, `velocity_field_ensemble`, `cognitive_architecture_root`, `ptg_functor_edges`, `local_branch_routing`, `ane_roofline`, `ane_fused_chain`, `cce_moderator`, `llmexec_guard`, `ssd_block`, `salience_tri_gate`, `renoise_ce`, `product_key_memory`, `linking_fold_fold`, `ssmax_temperature`, `subspace_steering`, `region_subspace_steering`, `mag_mining`, `tilr_invariant_subspace`, `manifold_erasure`, `heal_validation`, `smooth_min_similarity`, `simd_lut_dequant`, `poincare_navigator`, `chunked_content_store`, `causal_identification`, `conformal_predictive_intervals`, `karc_forecaster`, `hope_capacity`, `hebbian_kernel_memory`, `claim_rubric`, `clr`, `decode_specialize`, `delta_routing`, `specialist_projection`, `phase_separation` (Plan 571 Phase 25 promotion — modular arithmetic LRC coverage; G1+G2+G3+G4 GOAT gate ALL PASS), `similarity_inference` (Plan 526 Phase 6 promotion — endogenous correlation device; G1–G8 GOAT gate ALL PASS per Bench 579; **DEMOTED to opt-in 2026-09-04** per Issue 867 T1.3), `ica_lens` (Plan 475 — FastICA non-Gaussian direction mining; G1–G5 GOAT gate ALL PASS per Bench 475), `channel_simd_align` (Plan 227 Phase 5 — cache-line-padded weight storage; G1–G5 GOAT gate ALL PASS per Bench 580, 84.9%/86.7% release throughput), `lattice_operad` (Plan 252 Phase 2 — canonical AND/OR pruner expression composition; see §65), `plot` (Issue 355 Phase 2a — plotters dep toggle for benchmark SVG output; pure utility).

The full DEFAULT-ON list lives in `crates/katgpt-core/Cargo.toml` `default = [...]` (73 features) + root `Cargo.toml` `default = [...]` (135 features). See the per-feature Cargo.toml comments for GOAT bench references.

## 70. Functional Attention Family — Spectral Transport Operator (Plan 286)

A family of opt-in features for Functional Attention — a closed-form Tikhonov k×k spectral transport operator that replaces standard attention with a linear-in-n variant.

### Features

| Feature | Plan | Default | Role |
|---|---|---|---|
| `funcattn` | 286 | opt-in | Core operator — closed-form Tikhonov spectral transport (arXiv:2605.31559) |
| `funcattn_structured_basis` | 332 | **DEFAULT-ON** | Multi-scale basis constructors (DCT-log, Haar-packet) for FUNCATTN (implies funcattn) |
| `funcattn_spectral_pre_rotate` | 286 | opt-in | FUNCATTN × SpectralQuant — pre-rotate basis weights into the calibrated eigenbasis (implies funcattn + spectral_quant) |
| `funcattn_chiar_blend` | 286 | opt-in | FUNCATTN × CHIAR — per-token spectral-entropy sigmoid blend of FUNCATTN (low-entropy) vs fallback (implies funcattn + chiaroscuro) |
| `funcattn_freeze_thaw` | 286 T5.3 | opt-in | FUNCATTN × BLAKE3-committed snapshot hot-swap (FuncAttnWeightsSnapshot + atomic RwLock<Arc<>>) |
| `funcattn_compose` | 286 Phase 5 | opt-in | Parent — all three FUNCATTN compositions (spectral_pre_rotate + chiar_blend + freeze_thaw) |

### Architecture

The core `funcattn` replaces the standard attention matrix with a closed-form Tikhonov k×k spectral transport operator, making attention linear in sequence length. `funcattn_structured_basis` adds principled multi-scale basis constructors (DCT-log, Haar-packet). Three compositions extend it: spectral pre-rotation (align basis with SpectralQuant), chiaroscuro blend (sigmoid-mix with fallback attention based on spectral entropy), and freeze/thaw (BLAKE3-committed weight hot-swap).

Note: `funcattn` is already mentioned in the DEFAULT-ON list as part of the default feature surface — the feature flag itself is opt-in in katgpt-core but the `funcattn_structured_basis` submodule is DEFAULT-ON.

🔧 Feature flags: `funcattn` core opt-in; `funcattn_structured_basis` DEFAULT-ON; three compositions opt-in. Core in `katgpt-core`; compositions in `katgpt-attn`.

📖 Plan: [286](../../.plans/286_functional_attention_spectral_transport.md). Research: [257](../../.research/257_Functional_Attention_Spectral_Transport_Operator.md).

## 71. Sparse Off-Principal Task Vector Family (Plan 264)

A family of features for modelless sparse task vector operations — OPD-grounded sparse delta format + cross-frame alignment + ranking correction.

### Features

| Feature | Plan | Default | Role |
|---|---|---|---|---|
| `sparse_task_vector` | 264 Phase 1 | opt-in | Sparse Off-Principal Task Vector storage — OPD-grounded sparse delta format (GOAT G1–G2 PASS, 2.9×) |
| `off_principal_retrieval` | 264 Phase 2 | opt-in | Off-Principal Task Vector Retrieval — projection + index + score (GOAT G3–G4 PASS, ≥9×) |
| `orthogonal_procrustes` | Issue 001 | opt-in | Orthogonal Procrustes — cross-frame embedding alignment via polar decomposition (GOAT 4/4 PASS, 365µs) |
| `dynamic_rank` | 232 | opt-in | Dynamic Rank Pruner — GATv2-inspired static ranking detection & correction |

### Architecture

`sparse_task_vector` stores task-specific weight deltas in a sparse OPD-grounded format. `off_principal_retrieval` provides projection + indexing + scoring for retrieving relevant task vectors. `orthogonal_procrustes` aligns embeddings across frames via polar decomposition. `dynamic_rank` detects and corrects static ranking issues (GATv2-inspired).

🔧 Feature flags: all opt-in. `sparse_task_vector` in `katgpt-sparse`; `off_principal_retrieval` + `orthogonal_procrustes` in `katgpt-spectral`; `dynamic_rank` in `katgpt-pruners`.

📖 Plan: [264](../../.plans/264_sparse_off_principal_task_vector_modelless.md). Research: [231](../../.research/231_Sparse_Off_Principal_Task_Vector_OPD.md).

## 72. Task-Relevant Identifiability Family (Plan 265)

A family of features for task-relevant identifiability — band-conditioned selection + collider consistency + adaptive CoT stopping.

### Features

| Feature | Plan | Default | Role |
|---|---|---|---|
| `band_conditioner` | 265 | **DEFAULT-ON** | Band conditioning set + CI test primitives for task-relevant identifiability (arXiv:2605.12733) |
| `bckvss` | 265 Phase 1 | opt-in | Fusion A: Band-Conditioned KV Segment Selector (GOAT G1–G3 ALL PASS) |
| `collider_consistency` | 265 Phase 3 | **DEFAULT-ON** | Fusion C: Collider-Consistency ConstraintPruner for DDTree (GOAT G7–G9 pass) |
| `adaptive_cot_identifiability` | 265 Phase 4 | opt-in | Theory-backed adaptive CoT stopping criterion (GOAT G10 PASS) |

### Architecture

The family implements three fusions of task-relevant identifiability theory. `band_conditioner` (DEFAULT-ON substrate) provides band conditioning + conditional independence tests. `bckvss` (Fusion A) selects KV segments based on band-conditioned relevance. `collider_consistency` (Fusion C, DEFAULT-ON) applies collider-consistency constraints as a DDtree pruner. `adaptive_cot_identifiability` (Phase 4) provides a theory-backed CoT stopping criterion.

🔧 Feature flags: `band_conditioner` + `collider_consistency` DEFAULT-ON; `bckvss` + `adaptive_cot_identifiability` opt-in. All in `katgpt-band`.

📖 Plan: [265](../../.plans/265_task_relevant_identifiability_modelless.md). Research: [232](../../.research/232_Task_Relevant_Identifiability_Specialist.md).

## 73. INSIGHT Symbolic Distillation & Explanation Family (Plan 210)

A family of features for the INSIGHT pipeline — modelless explore→distill→explain for symbolic expression distillation and decision explanation.

### Features

| Feature | Plan | Default | Role |
|---|---|---|---|
| `symbolic_distill` | 210 F1 | **DEFAULT-ON** | Symbolic Expression Distillation — compact polynomial expressions for DDtree boundaries (GOAT 6/6) |
| `concept_grounding` | 210 F2 | **DEFAULT-ON** | Concept Grounding — template-based pruner rule explanation (implies symbolic_distill, GOAT 6/6) |
| `decision_explain` | 210 Phase 4 | **DEFAULT-ON** | Perturbation-based decision explanation via sensitivity analysis (GOAT 6/6) |
| `insight_explain` | 210 | opt-in | Parent — full INSIGHT pipeline (implies symbolic_distill + concept_grounding + decision_explain + posterior_evolution + mux_latent_context + reward_calibrator) |

### Architecture

The INSIGHT pipeline flows: `symbolic_distill` (compact polynomial expressions for DDtree boundaries) → `concept_grounding` (template-based rule explanation) → `decision_explain` (perturbation-based sensitivity analysis). The parent `insight_explain` combines all three plus `posterior_evolution` + `mux_latent_context` + `reward_calibrator` into the full pipeline.

Three of four components are DEFAULT-ON (GOAT 6/6 each); only the parent aggregator `insight_explain` stays opt-in.

🔧 Feature flags: `symbolic_distill` + `concept_grounding` + `decision_explain` DEFAULT-ON; `insight_explain` opt-in. All in `katgpt-pruners`.

📖 Plan: [210](../../.plans/210_insight_symbolic_distillation_explanation.md).

## 74. CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall (Plan 567)

Associative-memory recall on the symmetric space CP^(d-1) = SU(d)/U(d-1) instead of the sphere S^(n-1): the memory kernel `K_i = Σ_μ O_μ^(i) |ξ^μ_i⟩⟨ξ^μ_i|` is a d×d Hermitian **spiked** random matrix, and recall = align `|s_i⟩` with its **top eigenvector**. The top eigenvector is BBP-protected (Baik-Ben Arous-Pêché) against GUE crosstalk — the structural reason CP^(d-1) capacity **grows** with d (asymptotic α_c: 0.05 at d=2, 0.62 at d=3, 2.41 at d=4, ~40 at d=8) while gapless vector alignment on S^(n-1) **decays** as 4/(27n).

**Paper:** Galitski, *High-Capacity Generalized Hopfield Networks* — [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks) (JQI/UMD, 2026-07-31). Ships the generalized Gell-Mann basis + `f_abc`/`d_abc` structure constants (computed in closed form, not tabulated — a d=8 dense `f_abc` table would be 63³ entries), Mattis overlaps in real Bloch arithmetic, shifted-power-iteration top eigenvector, EXACT closed-form CP^(d-1) constraint projection (the Bloch map is a Euclidean similarity, so the closest pure state is the top eigenvector of ρ — no iterative projected gradient needed), the generalized Landau-Lifshitz-Gilbert dissipative flow (precession conserves energy, Gilbert damping lowers it monotonically), and finite-N capacity measurement.

Pure modelless — closed-form construction + Rayleigh-quotient ascent, NO gradient descent, so memories load from a frozen snapshot (freeze/thaw Path 1).

### GOAT gate (Bench 567) — STAYS OPT-IN

| Gate | Criterion | Result |
|---|---|---|
| G1 correctness | recall recovers corrupted memories below α_c | **PASS** — 27 unit tests |
| G2 capacity | measured α_c at our real N; capacity grows with d | **PASS** — α_c(d=3,N=64)=1.295 vs α_c(d=2)=0.174 (7.4× gain from moving off the sphere) |
| G3 no-regression | opt-in, default-off; `--all-features` clean | **PASS** |
| G4 perf | O(d³) paths sub-µs at d ≤ 4, alloc-free | **PASS** (after fixing a plan cost-model error — the `d³` eigendecomp is trivial; the real cost is `O(P·N·D2)` Mattis overlaps, cached incrementally) |
| G5 Plan 276 unblock | flips ≤ 10× leaky AND tracking ≥ leaky − 0.05 | **PASS, narrowly** — CP² recall with task-aligned memories: flips 347→3, tracking 0.000→1.000. Haar-random memories fail tracking (memory set must align with beliefs to be recalled — exactly freeze/thaw Path 1). Non-monotone in snap strength — depends on a hyperparameter with no principled setting. |
| **G6 Fusion B (KG capacity)** | ≥ 3× cosine-ANN triple capacity | **❌ FAIL** — CP² recall worse than cosine at every N (capacity ratio 1.00×). LLG unblock follow-up REFUTED (bit-identical precision to single-step). Projected-cosine diagnostic: projection HELPS 3–9× as a denoising op, but associative recall destroys angular precision — not actionable on production retrieve paths (queries use clean centroids where raw cosine already hits 1.0). |
| G7 BBP gap | relative gap > 0.1 at finite N | **PASS** — strongest result (0.73–0.95 everywhere, 7× margin at worst) |

**Promotion decision: `cp_hopfield` STAYS OPT-IN.** Default-on requires G5 AND G6 AND G7. G6 FAILS, and G5 passes only in the narrow sense, so the promotion precondition is not met.

### Why opt-in

1. **G6 FAIL is load-bearing.** The Super-GOAT headline (KG retrieval capacity gain) is refuted — CP² associative recall trades angular precision for basin robustness, which is the wrong trade for type-hit retrieval.
2. **G5 snap-sensitivity.** The Plan 276 unblock depends on a snap hyperparameter with no principled setting; the result is real but not a clean margin.
3. **No production consumer wired.** The primitive is validated mechanistically; a concrete consumer (riir-neuron-db `ItemEmbedIndex`, riir-games personality recall) would need to demonstrate a gain the synthetic G6 could not.

🔧 Feature flag: `cp_hopfield` (katgpt-core, opt-in). Implies nothing — standalone.

📖 Plan: [567](../../.plans/567_cp_hopfield_top_eigenvector_recall.md). Research: [466](../../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md). Benchmark: [567](../../.benchmarks/567_cp_hopfield_goat.md).

## 75. Gemma 4 Inference Config (Issue 577) — Infrastructure

`gemma4_inference` adds the `Gemma4LayerType` enum + `Config::gemma4_12b()` preset + `ModelArchitecture::Gemma4` variant, enabling Gemma 4 (sliding + full attention + per-layer shape) model loading. **Infrastructure, not an algorithmic primitive** — exists so the riir-engine Plan 318 Phase F 4B/2B MLA-MoE student has a real 12B baseline to beat, and so future Gemma 4 consumers have a config entry point.

The feature is a leaf-clean type addition in `katgpt-types` (`Gemma4LayerType`, `Config` preset) forwarded through `katgpt-core`. The forward path + GGUF loader live in riir-engine (opt-in `gemma4_inference` there, gated on this config type).

### What ships

- `Gemma4LayerType` enum (sliding-window vs full-attention per-layer tag).
- `Config::gemma4_12b()` preset (hidden size, num layers, vocab, etc.).
- `ModelArchitecture::Gemma4` variant.
- `MultiLayerKVCache::new_with_per_layer_kv_dim` — supports per-layer KV dim (Gemma 4 mixes sliding + full attention, which can have different KV shapes).

### Why opt-in

1. **Gates the loader + forward path in riir-engine.** The config type alone is cheap; the model file + forward kernels are not in katgpt-rs.
2. **No own GOAT gate.** Infrastructure — its value is enabling the GOAT gates of consumers (riir-engine Plan 318 Phase F baseline).

🔧 Feature flag: `gemma4_inference` (katgpt-core → katgpt-types, opt-in). riir-engine re-exposes it.

📖 Issue: 577 (resolved; no standalone `.issues/577_*.md` file — reference is in commit `96ac0a18` + `20a088af`). Substrate: `crates/katgpt-types/src/` (Config + Gemma4LayerType) + `crates/katgpt-core/src/`.

## 76. Kimi-K3 Analytic Backward — Training-Reference Gradient Pass

The analytic backward (BPTT) path for the Kimi-K3 native support (§33). Composes the per-primitive backward modules — **MLA attention** (`mla_backward`), **KDA linear attention** (`kda_backward`), **MoE FFN** (`moe_backward`) — with model-level composition backward (attn-res + RMSNorm + dense SiTU FFN + LM head + embedding) into a full-model gradient pass. Also includes gradient checkpointing (C7) + training-suitable weight init (C9).

This is a **training-time reference**, NOT a production inference path — consumed by `riir-train`. It is the documented modelless-by-mandate exception: the gradient kernels are analytic (hand-derived Jacobians, finite-diff verified), but the *purpose* is gradient descent, which is training. The inference-side modelless mandate is unaffected.

### What ships

| Feature | Module | Role |
|---|---|---|
| `kda_backward` (katgpt-attn) | `kda/backward.rs` | KDA analytic backward + conv-ring cross-token BPTT (C5). f64 accumulation to prevent NaN from gradient overflow (Issue 389 T4 ripple fix). |
| `mla_backward` (katgpt-attn) | `mla/backward.rs` | MLA analytic backward (C4). Cross-token composition gradient check relaxed for finite-diff tolerance. |
| `moe_backward` (katgpt-transformer) | transformer MoE | MoE FFN analytic backward (C4). |
| `kimi_k3_backward` (root) | `src/kimi_k3/backward.rs` | Full-model composition: `kimi_k3_forward_token_saved` (forward with activation capture) + `kimi_k3_backward_sequence` + `attn_res_backward` + `dense_situ_ffn_backward` + `situ_backward` + `KimiK3ModelGradients`. Implies `kimi_k3_loader` + the three per-primitive backward features. |

### Gradient check results

- **Single-layer (KDA+Dense, L=1):** PASS, max rel_err 0.26%.
- **No-attn-res (3 layers, block_size=1000, L=2):** PASS, max rel_err 1.38%.
- **All-KDA (3 layers, block_size=4, L=2):** PASS, max rel_err 8.34%.
- **Smoke test (kimi_k3_0_40b dims, vocab=64):** PASS (no NaN/Inf).
- **Full-model (3 layers with MLA at layer 2):** FAIL at 36% — MLA integration issue (MLA backward validated in isolation at 1.77%). Under investigation.

### Why opt-in

1. **Training-time reference.** The modelless-by-mandate exception — consumed by riir-train, never on the production inference path.
2. **Heavy dep chain.** Implies `kimi_k3_loader` (safetensors + memmap2 + base64) + the three per-primitive backward features.
3. **Full-model MLA composition backward has a known FAIL.** Single-primitive backward passes its gradient check; the multi-layer MLA composition does not yet.

🔧 Feature flags: `kda_backward` + `mla_backward` (katgpt-attn), `moe_backward` (katgpt-transformer), `kimi_k3_backward` (root, composes all three + the model-level backward). All opt-in (default-off).

📖 Substrate: `crates/katgpt-attn/src/{kda,mla}/backward.rs` + `crates/katgpt-transformer/src/` + `src/kimi_k3/backward.rs`. Related: §33 (Kimi-K3 forward). The backward work landed across commits `e896771c` (KDA + grad check), `8d3bd4f8` (MLA), `25d0c031` (MoE), `1de0e634` (KDA multi-token BPTT), `beffe760` (full-model), `5b232f33` (gradient checkpointing), `b7d5b33f` (weight init), `dfd139ea` + `9bb31500` (f64 accumulation NaN fix).

## 77. FlashMemory — Periodic Sigmoid-Threshold Sparse Attention for MLA (Issue 584)

Distills FlashMemory-DeepSeek-V4's lookahead sparse attention ([arXiv:2606.09079](https://arxiv.org/abs/2606.09079), Research 436) into the shipped `VortexFlow` substrate, adapted for **Multi-head Latent Attention** (the 2 MLA layers of Kimi-K3's hybrid 6-KDA/2-MLA stack). Three mechanisms:

- **Periodic refresh (τ-step)** — `FlashMemorySelector::select()` caches the last decision and refreshes only when `step − last_refresh ≥ τ`, amortizing selection cost (the R436 §2.1 gap: every shipped scorer ran per-step).
- **Sigmoid threshold selection** — `sigmoid(score) ≥ threshold` per-head per-block (paper default 0.5), producing dynamic block counts per query instead of rigid top-k. Sigmoid, never softmax.
- **Block centroids from MLA latent KV** — `FlashMemoryBlockCache::rebuild_from_cache` builds block centroids from `MlaKVCache::latent_kv_at()` + `W_UK` per-head up-projection. The centroid is only the selection heuristic — the sparse forward (`mla_forward_token_flashmemory`) attends to real per-token keys inside selected blocks.

### GOAT gate (Issue 584 — resolved 2026-08-15; Phase 1 real weights M3 2026-08-13, Phase 2 4090 2026-08-14)

| Gate | Verdict | Evidence |
|---|---|---|
| **G1** correctness | ✅ **PASS** | Dense-vs-sparse MLA on real `model.safetensors` weights: median cos 0.9566/0.9663/0.9663 + median rel MSE 0.1327/0.0929/0.0993 at 128/512/1024 tokens, ~73-74% KV reduction (`bench_021_flashmemory_real_weights_retrieval`). Threshold sweep: 0.5 is the calibrated sweet spot (0.3 → cos 1.0 but no sparsity; 0.7 → cos 0.72). |
| **G3** no-regression | ✅ **PASS** | 169 VortexFlow tests incl. the periodic-refresh extension (2026-08-13). |
| **G4** alloc-free | ✅ **PASS** | 0 allocations across 256 steady-state decode tokens / 32 selector refreshes (`bench_022_flashmemory_alloc_free`). Fixed two per-token alloc sites: `blocks_to_attend` stack-array fallback + `PerHeadSelection::new` pre-reserved capacity. |
| **G5** KV reduction | ◐ **PARTIAL (data-dependent)** | Real Kimi-K3 weights: ~74% ≤8K (bench_024, G1 holds at all scales). Bench 671 (4090, synthetic): ~50% consistent 4K→256K — synthetic random per-block directions keep ~50% positive dot-products. Paper's 90% needs real long-context attention patterns (a trained indexer / real Bonsai serving, riir-train Plan 337). |
| **G2** decode latency @ 256K | ✅ **PASS** | Bench 671 (RTX 4090, 2026-08-14, Bonsai GQA dims, synthetic): **1.8×** decode at 64K (6.35 vs 11.35 ms/token); at 256K sparse 22.69 vs dense 43.73 ms/token. GQA substrate: `GqaFlashMemoryBlockCache`/`GqaFlashMemorySelector` (katgpt-attn, 5 tests) + `forward_attention_layer_flashmemory` (riir-engine `flashmemory_gqa`, 4 tests). |

**Honest negative result (NIAH semantic-needle diagnostic, `bench_023`):** single-layer retrieval is **not testable** — the needle block has near-uniform dense attention (rank 34/34, mass ~0.03) at layer 3/8 on raw-embedding inputs; retrieval is an emergent multi-layer property. Positive diagnostic: per-head Pearson r between FlashMemory centroid block-mass and dense per-token block-mass = 0.965 (min 0.765). Output accuracy (G1) is the load-bearing gate; full-model NIAH deferred to Phase 2.

### Trained dual-encoder indexer (`trained_indexer`, riir-train Plan 337)

Two tiny trained MLPs (Q-Indexer + K-Indexer, 2114 params at Kimi-K3 d_h=64) that replace the modelless centroid-dot scorer — an upgrade, never a requirement. FlashMemory's periodic batch-scoring works with ANY scorer.

- **Bench 025** (`bench_025_flashmemory_trained_indexer`): the full training pipeline works end-to-end on M3 (dense-MLA attention-mass extraction → golden labels → manual-backprop dual-encoder training → GOAT gate). **Honest caveat:** Kimi-K3-0.40B (395M) has near-uniform attention → golden labels are low-signal; modelless baseline recall = 59.05%; the trained indexer's training does not converge on this data. Meaningful quality results need Bonsai-27B on the 4090.
- **Bench 026** (`bench_026_flashmemory_indexer_synthetic_convergence`): synthetic convergence proof that **definitively separates algorithm from data** — Adam optimizer (β1=0.9, β2=0.999) + non-zero bias init (breaks the bilinear σ(q·k) symmetry) converge to 100% accuracy from Epoch 1, beating modelless by 50pp; gradient check rel-err 0.000075 (< 0.01%). Also fixed the PRNG pathology (xorshift64 + Box-Muller outliers → warm-up + Irwin-Hall Gaussian). Conclusion: the Bench 025 non-convergence is a **data-quality** issue, not an algorithm bug.

### Why opt-in

1. **Serving-regime specificity.** GOAT gate complete (G2 PASS via Bench 671 on 4090, 2026-08-14; G5 PARTIAL — reduction is data-dependent: ~50% synthetic / 74% real ≤4K). Stays opt-in on scope, not gate: the win shape (KV-constrained 256K serving on Bonsai/4090) is far from the default short-context workload, and promotion waits for the trained indexer (riir-train Plan 337) to prove it beats the modelless scorer at scale.
2. **Scale honesty.** Real-weights validation is Kimi-K3-0.40B's 2 MLA layers at ≤8K; the 4090 scale test (Bench 671) is single-layer synthetic at Bonsai dims. Real 256K Bonsai serving needs CPU-offload + GPU-prefetch (the issue documents why a dense 256K baseline is infeasible on one 4090 — ~128 GiB).
3. **`trained_indexer` is training-dependent** — never default-on per the modelless-first mandate; the modelless `FlashMemorySelector` is the production default.

🔧 Feature flags: `flashmemory_sparse` (root → `katgpt-attn/flashmemory_sparse`, implies `mla_attention` + `dash_attn`, opt-in) + `trained_indexer` (root → `katgpt-attn/trained_indexer`, implies `flashmemory_sparse`, opt-in never default-on).

📖 Substrate: `crates/katgpt-attn/src/dash_attn/flashmemory_sparse.rs` (14 tests; both the modelless selector and the `trained_indexer` DualEncoderIndexer live in this one module) — landed across commits `8f030de2` (Phase 1 mechanism), `5ab63aef` (G1 bench_021), `7030cefa` (G4 bench_022), `865362e3` (G5 bench_024), `5e2b5f2b` (Plan 337 Phase B DualEncoderIndexer), `bca98f08` (Plan 337 bench_025), `4cdd0dec` (Plan 337 Phase C bench_026). Issue 584 (removed per noise-reduction rule — resolved 2026-08-15; G1-G5 evaluated, G2 via Bench 671; Benches 021-026 + 671 are the record). Training recipe: [riir-train Plan 337](../../../riir-train/.plans/337_flashmemory_indexer_training_recipe.md) (NOT the katgpt-rs Plan 337 — that is Tropical algebra).


## 78. Clustered LM Head — two-stage admissible-set vocab head (Plan 574; Issues 657/658/661/666)

**Feature flag:** `cluster_lm_head` (opt-in **PERMANENTLY** — the load-bearing
real-checkpoint gate was measured 2026-08-17 and fired the pre-committed
negative rule: [riir-ai Bench 688](../../../riir-ai/.benchmarks/688_clustered_lm_head_real_checkpoint_harness.md)).

Stage 1 scores k≈V/128 clusters via ⟨h, centroid_c⟩; stage 2 runs the exact head
over the admissible set only. `ClusterStop::Admissible` (Cauchy–Schwarz radius bound)
+ `ClusterStop::TopK` budget stops. D² seeding (Bench 658's degenerate-strided-init
fix) is the construction default; the strided variant survives as `ClusterInit::Strided`
for bench attribution only.

### GOAT trail (the honest arc)

| Gate | Result | Record |
|---|---|---|
| G2b recall | **PASS after two root-cause fixes** — 0.675 → **1.0000**. Issue 657's own diagnosis (the scoring objective) was WRONG: the real defect was degenerate strided k-means init. The bound is a *worse* ranker than what it replaced; its value is making an **exact** (recall-1.0-by-construction) stop rule possible. | [Bench 658](../../.benchmarks/658_clustered_lm_head_admissible_goat.md) (supersedes 657) |
| G2 structured | **8.3–9.2×** — after the cluster-contiguous row permutation (Issue 666): 2.2× → 8.3× by making each cluster's vocab rows memory-contiguous (the stage-2 GEMV reads whole clusters, not strided rows). | Bench 658 §addendum |
| G2 wave-parallel stage 2 | **WASH (negative)** — parallelizing stage 2's per-cluster work gained nothing: the bottleneck is **locality, not work**. This finding directly motivated the row-permutation fix above. | Issue 661 |
| G3 unstructured control | **HONEST FAIL — 0.08× loss** on uniform-random rows (99.99% admissible → all the overhead, none of the savings). | Bench 658 §Promotion |
| Promotion | **PERMANENT NEGATIVE (measured 2026-08-17, Issue 662 resolved + removed)** — on Gemma 2 2B's tied wte at the shipping ratio (k≈2000), 123 real `after_final_norm` probes: **Admissible active 99.95%** — the uniform-random regime, not between the extremes. Packed head **0.44×** (2.3× slower, interleaved). Exactness holds (recall 1.0000 asserted — sound bound, all-inclusive on real geometry: real h is anisotropic, exactly as predicted). The D² clusters carry real structure (TopK 68% argmax recall at 1.58% active) but TopK is inexact and the value proposition required the exact bound. Per the issue's pre-committed rule: fixture NOT tuned further. | [riir-ai Bench 688](../../../riir-ai/.benchmarks/688_clustered_lm_head_real_checkpoint_harness.md) (run record) |

**Unconditional landing:** D² seeding replaces strided seeding as the default in
`cluster_map_from_embeddings` (regime-independent strict improvement).

## 79. Bigram Markov Head — modelless sequential drafter (Issue 659)

**Feature flag:** `bigram_markov = []` in katgpt-speculative (zero deps, opt-in).

Deterministic CSR top-m successor table built from corpus bigram counts (packed-u64
sort + two-pointer passes; `(count desc, next asc)` top-m tie-break — bit-identical
rebuilds, brute-force-reference-pinned). Zero-alloc marginal emission
(`BigramMarginalBuffer`: O(steps × top_m) touched-reset sparse writes), greedy-chain
conditioning, zero-row fallback for unseen prevs (the `build_dd_tree` seam skips
prob ≤ 0 — an unseen prev proposes nothing). Emits per-position marginals straight
into `build_dd_tree(marginals, config)` — the existing seam with zero `dflash`
coupling in production code.

**Why:** Bonsai ships DSpark (6-layer drafter, 1.34× on 4090) but has NO working
drafter on Apple Silicon — the forward doesn't amortize at batch-1 (Bench 656
mode 2). A bigram table lookup is not a forward pass; it does not incur mode 2.

### Primitive gate (Bench 663)

181 ns/call (**23 ns/step**) at Bonsai scale on M3 release — ~5,600× under a
6-layer drafter forward per step (the mode-2 avoidance, measured); 17 MB worst-case
table vs 268 MB low-rank; G1 bit-identical rebuilds + brute-force-pinned; G4
alloc-free steady state.

**Deferred:** the consumer gate (acceptance rate at equal draft depth + wall-clock on
Metal AND 4090 against the Bonsai target) belongs to the riir-ai Bonsai consumer
(Plan 528) — the feature stays opt-in until it passes.

## 80. Switch Cost Table — directed skill-entropy switch cost (Issue 663)

**Feature flag:** `switch_cost = []` in katgpt-core (opt-in; `switch_cost_demo` example).

`SwitchCostTable` — directed pairwise switch cost `ske(a,b) = ln(Z(a∪b)/Z(a))` over
per-skill Bernoulli success counters: asymmetric by construction (`ske(0,1)=3.0` vs
`ske(1,0)=0.667` on the hand-computed fixture), u32 counters commute exactly
(record-order independent — forward vs reverse replay produce bit-identical
`to_bits()`), zero allocs. For task-switch scheduling / router curriculum ordering
(distilled from TTT-Discover via Research 484; consumed by the riir-clippy
cross-domain switch-cost measurement — Bench 032 there).

### GOAT (Bench 660)

G1 formula PASS (fixture 3.0/0.667, tol 1e-6) · G1 directionality PASS (gap >1.0
pinned constructible) · G1 determinism PASS (replay-order independence) · modelless
throughout. Stays opt-in (diagnostic/curriculum tool; no default-on consumer yet).

## 81. Freedom Selection — extension-count (freedom-of-function) best-of-K criterion (Issue 665)

**Feature flag:** `freedom_selection = ["renoise_ce"]` in katgpt-core (opt-in).

Extension-count selection (Bennett, arXiv:2608.05423; Research 486): among
candidates within a loss gate of the winner, prefer the one that opens an
unoccupied output region — freedom of function provably orders generalization.
Ships `log_freedom` (Σ log(2^a − 1), a = occupied-cell counts per context over a
declared finite partition), `freedom_gain`, `LossGate` (absolute/relative
tolerance), `ExtensionOccupancy` (O(1) update state) in `src/extension_count.rs`,
plus the renoise-CE selection sibling `best_of_n_freedom` (drift-gate +
max Δ-log-extension-count over caller-owned occupancy). Modelless closed-form;
8 unit tests incl. the brute-force enumeration pin. Documented conventions:
empty contexts excluded from the product (2^0−1 zero-annihilates);
first-activation pinned `FIRST_ACTIVATION_GAIN = 2.0 > ln 3` (raw increment +∞);
3-arg `freedom_gain` (distinguishes fresh vs occupied cell).

### PoC gate (Research 486 §PoC Addendum; harness `riir-poc examples/freedom_best_of_k.rs`, riir-ai `8bc3f65d2`)

**PASS** — parent-hit **0.7075** vs min-loss 0.4453 vs random-near-best 0.5156
(the confound control, SAME gate) on a controlled 4-context × 8-cell toy under a
declared child→parent distribution shift: **64/64 per-seed wins vs BOTH**.
Decomposition: relaxation buys +0.070; freedom guidance buys +0.192 more — **73%
of the gain is the freedom signal**, the paper's missing control separated. Both
substrate arms call the REAL substrates; all arms replay identical pools
(matched budget K=8).

T5 (Theorem-7 allocation formula as a second primitive) deferred on promotion.
Stays opt-in until a production consumer A/B + GOAT gate (the `switch_cost` /
Issue-663 precedent). Issue 665 RESOLVED+REMOVED 2026-08-17 (T1–T4 in `96d01e91`).

## 82. Effective Degree — modelless function-space simplicity metric (Issue 668)

**Feature flag:** `effective_degree = ["karc_forecaster"]` in katgpt-core
(opt-in; implies an already-default-on feature, so zero added build cost).

ED = Σ|c_k|·k over Chebyshev coefficients fitted along data-pair interpolation
paths (arXiv:2605.29823, ICML 2026; Research 488) — a distribution-aware,
reparameterization-invariant simplicity probe computable on any frozen
function. Ships `EdConfig` (+ `cheap()` r=4/K=3 and `precise()` r=15/K=7
presets mirroring the paper's efficiency/performance points),
`randomized_cosine_nodes` (stratified splitmix64 sampler, deterministic from
seed), `effective_degree_along_path` (the whole (K+1)² solve on fixed-size
stack arrays — no scratch, never allocates), the `_multi` vector-output twin
(one Cholesky with `out_dim` RHS), the basis-agnostic `ed_from_coeff_norms`
reducer, and the generic `ed_over_pairs` driver whose decode closure is
consumer-supplied (a shard readout, an adapter, a policy — katgpt-core stays
domain-agnostic). Gram/RHS accumulate in f64 (f32 Cholesky is fragile at
ε = 1e-6 relative to matrix scale — the KARC `fit_direct` precedent).
**Substrate consumed, not rebuilt:** `karc::ChebyshevBasis<8>` +
`linalg::{cholesky_f64, chol_solve_f64}`; only the ED reduction and the node
sampler are new (~560 LOC incl. docs + 12 tests).

### GOAT (Bench 665)

G1a–G1g + G2 + G3 + G4 ALL PASS — order preservation strictly monotone over
the full degree 1..5 chain (stronger than the paper's 3-point protocol),
stable across 8 node seeds and a Legendre basis swap (ordering is
basis-invariant, magnitude is not); `ed` scales ×2.0000 with outputs while
`ed_norm` is invariant to 1e-4; **195.9 ns/path** at the cheap config (sub-µs
is marginal at precise: ≈0.9 µs quiet / ≈1.1 µs busy); **0 allocs** steady
state; lib tests 1893 → 1905 (+12, 0 regressions); clippy clean in all three
feature states.

**Honest finding (the DC term):** `ed_norm` is a degree-weighted mean over
ALL coefficients including k = 0, so a DC offset drags it below the algebraic
degree (deg-5 fixture reads 1.15; with `c₀` zeroed, 1.63). Ordering is
unaffected — `ed_norm` is comparative-only, never an absolute degree read.
An offset-free arm is one line via `ed_from_coeff_norms`.

### Consumer verdict — SCOPE-LIMITED, no gate change (riir-neuron-db Issue 602 / Bench 484)

Over 360 shard states, `ed_norm` out-correlates the incumbent
`output_flatness` **12.6×** pooled (0.598 vs 0.047, control 0.032, permutation
floor 0.042) and beat it 4/4 scenarios — but the **sign inverts between
grains** (pooled +0.598, within-regime all four negative), a Simpson reversal
on 3 disjoint seed sets, so no threshold wires it as the proposed one-sided
freeze gate. The Bench 665 DC finding drove that conclusion: zeroing
`coeff_norms[0]` collapses the correlation 0.598 → +0.122 — ED's power on that
substrate lives in the DC term (event *alignment*, not shape complexity). The
paper's thesis is therefore NOT confirmed at shard scale; the weaker claim
(data-anchored function-space beats data-blind parameter-space) is.
`EdConfig::cheap()` is sufficient on real substrate (0.598 vs precise's 0.623
for ~5× less work); Theorem 3.1's path-averaging shows 0 ranking flips across
`n_pairs ∈ {1..64}`. Surviving promotion axis: **cross-regime triage** (the
KARC regime-mismatch probe, Research 488 §4) — not freeze timing.
Documented risk: grain-dependent sign (module-doc caveat 4 — consumers must
state their grain and verify the sign there; the paper measures only the
across-model grain).

Stays opt-in + diagnostic-only (the no-default-consumer rule). ED is not
UQ-bearing (a complexity scalar, not a distribution/interval/coverage claim),
so the Issue 010 "Report the Floor" rule does not apply. Issue 668
RESOLVED+REMOVED 2026-08-18 (T1–T6 in `2d5a9efc`; consumer verdict folded in
`89e3910d`).

## 83. Ignition Schedule — closed-form logistic ignition primitive (Issue 459 T5)

**Feature flag:** `ignition_schedule = []` in katgpt-core (zero deps, opt-in).

`IgnitionSchedule` (katgpt-core/src/ignition.rs) — the Neural Quadratic Forms
ignition theorems (arXiv:2608.13335 Thms 5–8; riir-train Research 422 §3.5) as
pure closed-form math:

- `IgnitionSchedule::new(z0, k, zeta)` — contract-asserted (`0 < z0 < k`, `zeta > 0`)
- `at(t)` — `z(t) = K / (1 + ((K−z₀)/z₀)·e^{−ζt})` ≡ `K·σ(ζt − ln((K−z₀)/z₀))`, one `exp`, no iteration
- `time_to_reach(target)` — per-curve inverse
- `ignition_time(zeta, eps)` — the patience law `t* = ln(1/ε)/ζ` (Thm 8, capacity-free)
- `order_by_ignition_into(zetas, &mut [usize])` — ζ-descending ignition order, index-ascending tie-break, caller-owned buffer

Design anchors: sigmoid-in-time is the adoption shape GD itself produces (the
second grounding for sigmoid-not-softmax after R315); patience ∝ 1/ζ —
pre-ignition signal is ε-small, so keying on raw rates amplifies noise (the
measured riir-clippy Issue 026 starved-pool negative is the anchor this
predicts). Correctness pinned beyond the gates: RK4 ODE anchor to the GLV
dynamics (<5e-4 rel), exact-sigmoid identity (<1e-5), inverse roundtrip (<1e-5).

### GOAT (Bench 666)

G1 monotone ranking **PASS** (t* strictly decreasing over ζ ∈ [0.1, 4.0] at
ε ∈ {1e-2, 1e-3, 1e-4}; `order_by_ignition_into` == observed threshold-crossing
order) · G2 latency **PASS** (**3.88 ns/call** release / 13.15 ns debug, n=100k
— 12.9× release headroom) · G3 no-regression **PASS** (default 1897/0/6 exact
baseline; feature-on 1911/0/6 (+14); clippy 0 both states) · G4 alloc-free
**PASS** (TrackingAllocator: 0 allocs across 1000× `at()` + 1000×
`ignition_time()` + the ordering helper).

**Consumer pilot (promotion gate — OPEN):** riir-clippy selection patience
scaled by `ignition_time(ζ̂, ε)` vs fixed patience on the heal-loop fixture.
Promotion to default only on a measured win; stays opt-in otherwise.

## 84. spectral_pencil — the affine matrix pencil scalar gate (Issue 676)

> **Added:** 2026-08-21 (`9795a9bd` closeout; doc-sync catch 2026-08-22 — the
> closeout landed the narrative doc but missed this catalog + the README +
> overview rows). Source: arXiv:2608.08003 "The Spectral Neuron" (Shtoff,
> TII 2026) / [Research 495](../../.research/495_Spectral_Neuron_Affine_Pencil_Shape_Gates.md)
> / [Bench 671](../../.benchmarks/671_spectral_pencil_goat.md) ·
> Code: `katgpt-core/src/spectral_pencil/` ·
> Narrative: [`.docs/02_inference/spectral_pencil.md`](../02_inference/spectral_pencil.md)

The scalar decision function `f(x) = λk(A₀ + Σ xᵢAᵢ)` — the input enters
**linearly** into a symmetric matrix; the nonlinearity is reading **one
ordered eigenvalue**. Expressivity grows with matrix dimension d while
retaining linear-model-style transparency (shape by construction: k=1
concave, k=d convex, `Aᵢ ⪰ 0` ⇒ Loewner-monotone per feature; Weyl global
influence bounds; exact Hellmann–Feynman attribution `∂f/∂xᵢ = vᵀAᵢv`;
canonical-gauge commitment bytes; invertible monotone warp; the γk ≥ ½
eigengap-guaranteed seeded init from paper Lemma 2).

- `tridiag` eval — 748 ns @ d=8 → 3.71 µs @ d=32 (Sturm bisection; the
  per-tick path at the 10k NPC × 20 Hz production shape)
- `dense` eval — 3.95 µs @ d=8 → 166.7 µs @ d=32 (pinned Jacobi; the
  spawn-time/GM/canonical-gauge low-cardinality path)
- `count_below` — **51 ns** exact integer Sturm count
- PSD/NSD feature constructors (`shape.rs`), Lipschitz certificate
  (`bounds.rs`), `field::SpectralField` archetype adapter + `GenomePod`
  per-NPC genome persistence (riir-ai Issue 736 B1/B3 extensions)

### GOAT (Bench 671)

G1–G4 **ALL PASS** — determinism (pinned full sweeps, bit-identical),
latency (table above; tridiag is ~41% of one P-core at d=16 / ~15% at d=8
under the production shape), no-regression (default untouched — opt-in
feature implying `hebbian_kernel_memory` + `karc_forecaster`), alloc-free
hot paths. UQ-bearing scope-limit recorded honestly: no
calibrated-prediction claim (the Report-the-Floor rule would apply before
any).

**Consumers:** `riir_game_sdk::spectral` facade (`spectral_hero_gate`,
riir-ai Issue 736 B2) → riir-mmorpg-examples spectral fear-gate (`00fa172`)
+ the 4th `FusionArm::Spectral` (mmorpg Bench 028 — c the continuous
max↔mean interpolation knob; c=0 bit-identical to Max). Stays opt-in —
swapping a game gate onto the spectral neuron is a gameplay decision (the
CLR precedent).

## 85. signed_coupling_dynamics — signed-graph opinion dynamics + crowd order parameters (Issue 680)

> **Added:** 2026-08-22. Source: arXiv:2608.16578 "Physics of Agents" (El et
> al., Stanford, Aug 2026) /
> [Research 497](../../.research/497_Signed_Coupling_Opinion_Phase_Forecast.md)
> / [Bench 672](../../.benchmarks/672_signed_coupling_goat.md) ·
> Code: `katgpt-core/src/signed_coupling.rs`

Glauber (heat-bath) update on a **signed** social graph — ties are typed
(`J_ij = +1` ally, `−1` rival, `0` absent) and each type gets its own
coupling:

```text
h_i = β⁺·Σ J⁺s + β⁻·Σ J⁻s + β₀·Σ|J|s + g_i        P(s_i = +1) = σ(h_i)
```

`g_i` is the intrinsic field — in our stack a direction-vector dot product
(personality × question), which is why nothing here trains: the caller
**authors** the couplings, and the paper's fitted ranges ship as
designer-facing constants (`PAPER_BETA_*_RANGE`).

- `SignedGraph` — row-compressed signed adjacency (CSR, `u32` indices, no heap
  after construction). Symmetric (the paper's energy form) **and** directed
  (asymmetric social influence: a recruit weighs the veteran, not the reverse).
- `Couplings` + `at_social_temperature(t)` — the one-scalar designer dial. High
  T = apathetic milling, low T = decisive mob.
- `signed_coupling_update_into` — 2 conditional adds per edge, ~1.8 ns/edge.
- `signed_coupling_update_informed_into` + `InformedCouplings` — the paper's
  5-coupling **truth asymmetry** (correct neighbors pull harder on the
  concordant channel, wrong ones push harder on the discordant). A sibling fn,
  not a flag, so no branch enters the inner loop.
- `sample_states_into` — caller-supplied uniforms ⇒ RNG-free and replayable.
- **Order parameters:** `net_opinion` (mean), `crowd_conviction` (**mean of
  squares — new; nothing in the stack shipped a mean-square crowd reducer**),
  `SusceptibilityAccumulator` (Welford `Var_t(|n|)` in f64 → `χ = N·Var`).

### GOAT (Bench 672)

G1–G4 **ALL PASS**, stays **opt-in** (promotion waits on a production consumer
— the CLR precedent). G1a/G1b reproduce indifference / polarization /
consensus on three graph families both deterministically and via a seeded
stochastic rollout; G1c reproduces the paper's `β⁺ > β⁻` consensus bias as a
mechanism; G1d locates an interior χ peak over the paper's 41-point sweep; G2
is at parity-to-2%-faster than the naive three-accumulator form (median
pairwise ratio 0.97–1.02× over 9 interleaved rounds); G4 is 0 allocs on every
steady-state path.

**Two findings the gate forced out** (both now on the type docs): `β₀ > β⁻`
makes rivals *attractive* — at the range midpoints the discordant weight is
`+0.15`, so a frustrated graph converges and polarization needs the
`β⁻ > β₀` corner of the fitted ranges; and "cold ⇒ consensus" is a claim about
the *graph*, since a cold short-range lattice quench freezes into domains
(`|n|=0.27`, `c=0.98`) unless given a shared field.

**Vocabulary collision (load-bearing for greps):** `crowd_conviction` here is a
crowd **order parameter** `mean(s²)`; Sheaf-ADMM `conviction`
(`katgpt-dec::sheaf_admm`, `riir-agents::multi_agent`) is per-agent
**resistance** in the consensus quadratic. Different quantities — and they
compose, since a sheaf conviction vector is a natural `g_i` source.

**Not claimed:** no calibrated-forecast claim (`σ(h_i)` is a dynamics rule; any
prediction-quality claim owes the conformal floor per Issue 010), and no
framing novelty — Research 497 §3 scored Q1 NO (De Marzo arXiv:2605.10721 /
De Nobili arXiv:2608.02178 published the "stat-mech predicts LLM crowds"
headline first). Gain, not Super-GOAT.

**`verdict_margin` (Plan 545 T1, 2026-08-23).** The one-snapshot
crowd-manipulability forecast derived from this substrate: the
CLR-reliability-weighted verdict margin over binary verdictification — how
close the crowd's weighted verdict sits to its decision boundary. Measured on
N=200 signed ring crowds: ρ(margin₀, verdict-flip-frac) = **−0.65** (paper
LLM −0.59; riir-ai Issue 745 / Research 499) — firm crowds flip less under
equal pressure, *but only through the gate's budget allocation* (ungated
uniform pressure flips firm crowds MORE — they have more majority agents to
lose; the negative forecast correlation exists only through conviction-gated
spending). Ships in the same module (`signed_coupling::verdict_margin`);
consumed by riir-games `social_pressure` (riir-ai Plan 545 — the
conviction-gated broadcast runner, G8 gated-spares-firm-crowds).

## 86. gaussianity_probe — sketched projection-normality for embedding populations (Issue 681)

Cramér–Wold sketch for d-dimensional embedding populations (Research 498 —
LeVLJEPA arXiv:2607.00784 SIGReg, distilled from training loss to an
inference-time diagnostic). Second-moment metrics (erank, spectral_flatness)
cannot see distribution *shape*; this probe can.

- **16 fixed directions** — 4 coordinate-axis anchors (`e_0..e_3`, the honest
  fix for axis-aligned bimodality: a purely random sketch dilutes it by
  |cos| ≈ 1/√d) + 12 BLAKE3-derived Rademacher ±1 rows (seedable, exact in f32)
- **Per direction** — KS-vs-fitted-Gaussian D statistic, a verbatim port of
  `katgpt_spectral::ks_d_statistic` (the leaf constraint forbids the dep; the
  port is pinned bit-identical by `katgpt-spectral/tests/gaussianity_agreement.rs`)
- **Aggregate** — `score = sigmoid(ln(p_min / 0.01))` over the n-aware
  Kolmogorov min-p (KS critical ∝ 1/√n); per-direction statistic public as
  `ks_d_vs_fitted_gaussian`
- **Zero-alloc** after `GaussianityScratch::new` (G4: 0 allocs / 100 calls)

### GOAT (Bench 673)

G1–G5 + cross-crate agreement **ALL PASS**, stays **opt-in** per the issue's
own T5 (promotion is a consumer decision). The load-bearing row is the
non-redundancy pin: on a bimodal e_0 μ=3σ fixture the probe scores **2.4e-23**
while `effective_rank` reads **53.3/64 = 83.3% "healthy"** — the blind spot
pinned. G2: probe **4.20× faster** than `effective_rank` (697.7 µs vs
2928.0 µs at n=1024 d=64 — erank pays the O(d³) Jacobi sweep). G5: 3 runs
bit-identical.

**Honest scope (module docs):** a non-axis-aligned moderate (μ ≲ 3σ) bimodal
departure in high d is missed by all 16 directions — the sketch is the cheap
always-on audit; `ica_lens` (katgpt-spectral, FastICA) is the optimizing
locator a consumer runs when it trips. CLT smoothing hides per-coordinate
idiosyncrasy for d ≳ 32; margin-wide departures (mixtures across samples,
radial heavy tails) are caught at any d.

**Waiting consumers:** band_conditioner Fisher-z precondition guard, riir-ai
#743 edge_lora hidden-space monitor, riir-neuron-db freeze-gate advisory
(`FreezeGateReport` additive field — the bimodal-two-styles-before-freeze
case).

## 87. distributional_steering — mean-field population steering via Feynman-Kac weights (Plan 577)

Population-level steering toward a **measure-defined target** `μ* ∝ e^{λΨ} p`
(Howard & Nüsken, [arXiv:2608.08770](https://arxiv.org/abs/2608.08770);
Research 505): a closed-form first-variation reward table, FK log-weight
accumulation with the mean-field `Ψ̇` correction solved by damped Picard,
and the weighted empirical measure `μ̂ = Σ wᵢ δ_{Xᵢ}` as the converging
object (paper Thm 3.4).

- **Reward table** — `LinearReward` (Ψ = a·x), `MomentReward`
  (Ψ = F'(m)·(p·x)), `MmdReward` (Ψ = 2[emb_ν − emb_μ], second variation
  −2k). All rows finite-difference-verified; the MMD gradient is
  `λ·4γ·[S_pop − S_ν]` (target attraction)
- **FK stepper** — `begin_step` (∇Ψ steering increment) → consumer
  integrates → `finish_step` (damped-Picard Ψ̇ + `Aᵢ += (b·∇Ψ + Ψ̇)δt` with
  the paper's clip); kernel matrix built **once per step** (symmetric fill +
  one `simd_exp_inplace` pass) and reused across every Picard iteration
- **Resampling** — `residual_resample_into` / `systematic_resample_into` for
  sampling consumers ONLY (documented NOT for persistent agents — the
  theorem tracks the weighted measure; weights-only is the agent mode)
- **BoM adapter** (`bom_sampling` + `distributional_steering`): static tilt
  fixed point over K hypotheses as a principled alternative to
  `select_best` argmax — no UQ claim
- **Zero-alloc** steady state (G4: 0 allocs / 1000 steps @ N=1000);
  bit-identical determinism; `tilt_residual` convergence certificate

### GOAT (Bench 682) — FAIL (partial) ⇒ stays opt-in

The paper's 1-D falsifiable harness reproduced the targeting minimum at
**λ\*=5 in both noise schedules** (clean V-curves) and the exact J
trade-off structure, but λ\*=10 held only one of two schedules (flat curve
at the seed-noise floor) and the gradient-only **separation claim did not
reproduce** (in the 1-D broad-kernel Langevin regime the position steering
dominates; the FK weights are a third-decimal correction). G2: 9045
ns/particle/step @ N=1000 — the sub-µs gate is structurally infeasible for
exact O(N²) MMD (the kernel build alone is 10⁶ `fast_exp`; the paper's
"Picard = 0.04% of runtime" is relative to network evals a modelless stack
doesn't have).

**Consumer-critical findings** (all in the module docs + Bench 682):

1. **Picard damping must scale O(1/λ)** for broad kernels — the iteration
   Jacobian ≈ 0.2λ; damping 1.0 diverges for λ≳5 regardless of K_FP
   (weights collapse to ESS 1). Use `α = min(1, 2/λ)`.
2. **`b` = the FULL simulated drift** (base + steering) — the `b·∇Ψ` term
   carries the Girsanov overshoot correction; Ψ̇ must be pure measure drift
   (both Ψ terms at the advanced positions).
3. **Research 505's Table-2 MMD row has a sign slip** (`Ψ = 2∫k(μ−ν)` for
   `R = −MMD²`); the module ships the corrected `Ψ = 2[emb_ν − emb_μ]`,
   pinned by finite-difference tests.
4. **Weights-only degenerates** to ESS→1 by λ≈7.5 over 30 steps without
   resampling (a real property, clip-bounded — harmless for crowd-salience
   consumers, decisive for sampling consumers).

**Demo** (`--example distributional_steering_demo`): 2-D GMM 1:3 → 3:1 dial,
MMD² 0.331 → 0.011, shares 0.24/0.76 → 0.67/0.33. **Reopen paths:** a
diffusion-sampler-shaped harness (the prerequisite for the riir-ai
crowd-targeting plan, Guide 344); approximate kernel features for G2;
N≲300 populations are already sub-µs per particle.

## 88. risk_control_exit — dual-threshold risk-controlled compute exit (Plan 575)

Modelless compute-exit gating distilled from "Conformal Thinking"
(Xi Wang et al., JHU + Apple, ICML 2026,
[arXiv:2602.03814](https://arxiv.org/abs/2602.03814);
Plan 575 / [Research 494](../../.research/494_Conformal_Thinking_Dual_Threshold_Risk_Control_Exit.md)):
how much compute a query needs is a **risk-controlled** decision, and one
threshold is not enough.

- **`DualExitPolicy`** — upper stop-when-confident threshold `λ+` + squeezed-
  sigmoid lower stop-when-not-progressing schedule
  `λ−(t) = σ(c(ωt − sB), l, u)` (Phase 1 T1.1–T1.2): exit early when the
  answer is provably not improving, not merely when it looks confident
- **Four bounded losses** (paper Eq. 8–11, T1.3) — the UQ-bearing surface;
  the Report-the-Floor rule is instantiated as the naive-calibration contrast
  (G1) + the exit-floor family (G2) since CRPS/Winkler are undefined for a
  decision rule
- **UCB/Hoeffding calibrator** (T1.4–T1.5) — offline, per-candidate
  `Risk̂ + sqrt(ln(1/δ)/2n) ≤ ε`, two-step decoupled selection with
  efficiency-loss argmin + monotonicity refusal
- **App. C disarm tripwire** (T1.6) — `p_i ≥ p_c` candidates are disarmed,
  the paper's safety condition against a degenerate lower schedule

### GOAT (Bench 681) — ALL PASS ⇒ stays opt-in (no-default-consumer rule)

| Gate | Result |
|---|---|
| G1 risk hold | UCB holds realized exit-FP-risk ≤ ε on **40/40** resplits at both validation sizes (n=40 max risk 0.0100; n=400 0.0200); naive no-correction violates **7/40 at n=40** (realized 0.18–0.35 ≫ ε) and is safe at n=400 — the paper's Fig. 4 small-n shape |
| G2 exit floor | At matched realized risk: dual compute **0.417** vs single-threshold **0.609** vs fixed-budget **1.000** (means over 3:1/1:1/1:3 trivial:stuck; wins-or-ties per composition; the gap grows 0.075→0.310 with stuck share — Fig. 6 shape) |
| G3 no-regression | Default lib 1951 unchanged; feature-on compiles green |
| G4 alloc/perf | 0 allocs steady state; **~4–5 ns/exit** (3.90/5.18, two release runs); calibration 0 allocs after init |

**Stays opt-in** — all gates pass modellessly but no in-tree consumer compiles
this module yet. Phase 3 is what flips it (each with its own consumer gate):
MCTS termination, Plan 304 fusion `GainCostLoopHalter`, Bebop Issue 023
re-gate, riir-ai Research 339 wiring.

## 89. prover_selection — prover-selection statistics: rank by complementarity, not strength (Issue 692)

Prover-selection kernels distilled from "Rewarding Progress: Scaling
LLM Inference via Data-Centric Verifiers" (Setlur et al., 2024,
[arXiv:2410.08146](https://arxiv.org/abs/2410.08146);
Issue 692 / [Research 509](../../.research/509_Rewarding_Progress_PAV_Prover_Advantage.md)):
which prover's signal should a consumer trust? Not the strongest one — the
paper's core finding is that the strongest prover is often the least
*distinguishable* from the policy it scores (A^μ ≈ 0 everywhere).

- **`distinguishability` / `alignment` / `theorem_bound` / `selection_gate`**
  (T1) — the D/Al complementarity selector: estimators over logged Bernoulli
  means + Theorem 3.1's predicted-gain bound `γ·(D+Al)` with a sigmoid-gated
  exposure
- **`first_pit`** (T2) — the changepoint kernel (first index where Q̂ < ε),
  consumed by riir-clippy's PAV data curation (riir-train Plan 356 A1) via a
  same-signature twin
- **`k_star` / `bok_advantage`** (T3) — the K* interior-optimum law: the
  closed-form rollout count maximizing the gap + the BoK gap
  `A(K) = (1−V)^K − (1−Q)^K`, gate-pinned against the empirical argmax on an
  exhaustive (Q,V) grid

### GOAT (Bench 684) — ALL PASS ⇒ promoted DEFAULT-ON 2026-08-27 (the `rating` precedent)

| Gate | Result |
|---|---|
| G1 correctness | Exhaustive unit gates; `k_star` pinned against empirical argmax on the exhaustive (Q,V) grid |
| G2 selector | The complementarity bound picks the peer that delivers a wired gain over the strength-trap top prover (`strong_flat`: strength 0.948, D 0.003) at **every paper α**, 16 seeds, on the controlled PAV retention harness (64×8, mean true θ of the retained beam strictly higher) |
| G3 no-regression | katgpt-core default lib **1978/0/7** post-promotion (the module's 27 tests now default-included); clippy 0 |
| G4 alloc/perf | Pure modelless arithmetic — no allocation sites, no deps, zero-cost-unless-invoked |
| G5 consumer | riir-clippy PAV data curation (Plan 356 A1) consumes `first_pit` via the same-signature twin |

**DEFAULT-ON** — pure math, no dep surface, zero-cost-unless-invoked (the
`rating` precedent); the Cargo feature remains as an inert alias. Companion
refutation recorded (Issue 692 T4): the dd_tree `BestAdvantage` variant
(score by `Q_i − mean_j Q_j`) is rank-invariant vs `BestQ` by mechanism —
not shipped.

## 90. incidence_algebra — the thought × agent incidence-mask algebra (riir-ai Issue 874 T1)

Distilled from ThoughtComm (arXiv:2510.20733, NeurIPS 2025) via riir-ai
Research 364 / Issue 874: the transferable artifact is not thought *recovery*
but the **incidence mask as a first-class object** (paper Thm 3 — the
who-shares-what structure is the identifiable thing). This stack already
builds such masks by construction (CLR observer sets, sheaf restriction maps,
npc_comms slices, healer fan-out hits) and computed none of the algebra over
them. `katgpt_core::incidence` ships it (feature `incidence_algebra`, opt-in;
11 module unit tests):

- **`agreement_counts_into` / `support_sizes_into`** — αⱼ (per-thought witness
  counts) and per-agent support sizes over the agent-major row-major mask;
  zero-alloc `_into` forms, deterministic fixed-index order (no hashing)
- **`agreement_score`** — one monotone σ-saturated tier curve, α ≤ 1 anchored
  to EXACTLY 0.5; two consumer maps that never gate:
  `routing_weight` (α=1 returns 1.0 bit-identically — a private thought is
  never penalized, the Bench-013 soft-bias lesson) and `contagion_strength`
  (α=1 returns 0.0 exactly — a single witness cannot stampede the crowd, the
  measured Plan-019 CLR failure this fixes; κ=0 kill-switch)
- **shared/private split + `private_fractions_into`** — shared = support with
  α ≥ 2, private = α == 1 (paper Thms 1+2); the retention counter is the
  Appx-C.2 discipline: never report agreement without it — collapse means
  conformity, not correctness
- **`hall_max_matching_into`** — Hall feasibility (can every agent be matched
  to a DISTINCT supported thought), Hopcroft–Karp, zero-alloc scratch
- **`audit_mask`** — support sizes + α distribution + the `DENSITY_ALERT`
  warning (a dense mask is the crowd-panic precondition — a warning, never a
  gate)
- **`rank_by_agreement_into`** — deterministic tier ordering (α desc, index
  asc — the Issue-849 lesson: a partial order truncated at a cap must never
  leave the tie-break to a per-process hasher)

Think-brain local: masks, α counts, and tier weights are never synced — what
crosses a sync surface stays raw (contagion intensity, witness counts; the
"sync the scalars" doctrine). First consumer chain: riir-games-shared
`agreement_tier` → riir-stealth `agreement_contagion` (riir-ai Issue 874 T2;
see the game-sdk book's `stealth_alarm.md` §α-weighted contagion for the
consumer A/B). **Opt-in with no standalone GOAT** — the substrate's gate is
its unit suite; a consumer-level A/B is the promotion instrument when a
routing consumer materializes (the same posture `signed_coupling` shipped
with).

## 91. usage_rate_eviction — mass/age KV eviction scoring + the generation-runaway canary (Plan 585)

Distilled from H2O-normalized (arXiv:2608.19920 "Learning how to Forget",
Seeger et al., AWS 2026 §3.2/A.4.3) via Research 523: the paper's normalized
H2O score `cum_mass / max(1, age)` as an O(1)/row/step incremental estimator
over caller-supplied attention-mass increments (the `suspect_indices` house
pattern — katgpt-core stays leaf-clean; mass producers are consumer-side,
riir-ai Issue 836 pull-gated). Fixes raw-H2O's age bias (an old-but-cold row
ties a young-but-hot row at equal cumulative mass); per-(b,h) selection by
construction; lowest-k eviction selection with pinned-sink exclusion +
`float_order` NaN-safe comparators + ascending-index tie-break (the Issue-849
determinism discipline).

- **`observe` / `score` / `UsageScoreTable`** — incremental mass+age state,
O(1) per row per step; `select_evict` / `select_evict_into` lowest-k with
pinned-sink mask
- **`RunawayStats` / `runaway_gate`** — the R/p128 generation-runaway canary
(Bench 696 lineage): output-length runaway is invisible to perplexity-style
and tolerant substring metrics — the paper measured 35–128× blowups while
SubEM read fine. Any lossy KV policy promoted to default MUST pass it on a
sealed long-context eval
- **`PolicyControl` / `beats_random_prompt_pin`** (Bench 697 null-control
addendum, Research 531) — the strict null bar: a candidate policy must beat
the prompt-pinned per-head random baseline at matched budget AND matched
protection; equal recall hands the slot to the cheaper null; NaN fail-closed.
Standing promotion rule: `runaway_gate` ∧ `beats_random_prompt_pin` ∧ the
protection factorial

**Bench 697 verdict — MIXED, opt-in:** G1–G4 PASS; 2–4× raw-H2O deep-needle
recall at cap ≥ 32 (signal value CONFIRMED against the controlled null —
5.0×/4.8×/3.8× at cap 32/48/64); one honest miss at the 8%-budget extreme
(cap 16, where even the random null's geometric survival beats mass_age);
G8 regime-bounded. Round cost 1.22 ns/row (4090) / 1.78 ns/row (M3), ≥5×
under the paper's +32–43% serving margin — the paper's kernel-pass margin
lives where scoring rides paged state (T4.1 byproduct kernel, Issue 836
pull-gated). The registered "skip the kernel, ship the null" alternative is
a measured cell: `rand_keystone` = 100% needle recall at every cap with zero
scoring work — the null ships only with a structural keystone oracle. No
consumer = no promotion (the no-default-consumer rule); promotion re-gate =
an Issue-836 consumer + real-corpus re-run against the controlled null.

## 92. contrastive_scope — the input-scope gate (Issue 674 / Research 493; DEFAULT-ON since 2026-08-20)

LittleLearner scope-gated epistemics (arXiv:2608.13545):
`ContrastiveScoreTable` (two-corpus log-odds, BLAKE3-committed,
freeze/thaw-able), `scope_score` D(x) (Naive-Bayes log-LLR sparse GEMV), the
epistemic haircut ĉ = c·sigmoid(−κ·D) + decline wiring, + the paired OOS
probe battery (the Report-the-Floor extension). "A relevance check is not a
scope check" — a syntactic pass is not evidence the input is in-domain.

**DEFAULT-ON since 2026-08-20** (Bench 669 T5 rule): the riir-clippy consumer
adoption landed (their Bench 040 / Plan 016 — the input-scope gate at the
`heal()` seam: `ScopeModel` over the domain corpus vs a canonical non-Rust
corpus; out-of-scope inputs declined 8/8 vs served 8/8 un-gated;
in-distribution healing bit-identical over the full corpus; 529 ns/input;
steady-state alloc delta −1). κ/θ re-pinned by the consumer from its measured
gap (θ=0, κ=4.5). No papaya dep — the built table is immutable (lock-free
reads by construction). Pure modelless (counts + log2 + sigmoid + BLAKE3);
zero runtime cost unless a table is constructed.

## 93. anti_common_mode + anchored_reach — the Research 433 (RVM) pair (Issue 696 T1/T2)

Both from arXiv:2608.23664 (RVM §C.3 / Eq. 7), landed together 2026-08-29,
independent features:

- **`anti_common_mode`** (T1) — the DT2 dynamic-tracking reward shape as a
  scalar gate; live consumer: the riir-ai riir-poc Issue-696-T3 CLR
  crowd-panic re-enable PoC (the `tick_swarm_emotions_collective` promotion
  is the named future consumer).
- **`anchored_reach`** (T2) — the anchored signed-reach blend
  `out = anchor + A·(cand − anchor)`: scalar or per-axis A, zero-alloc
  `*_into` forms, BIT-IDENTICAL pole fast paths at A ∈ {0, 1} (the composed
  form loses bits: 1e-30 + (1e-40 − 1e-30) = 0.0), and the A(r) schedule
  constructors (linear r / house-sigmoid 2σ(kr)−1 / sign-flip (2r−1)/β̄).
  Five regimes: clamp / blend / adopt / overshoot / repel.
  Sigmoid-not-softmax by construction (pointwise — RVM never normalizes
  across the group). The A=1 pole is the recorded operator behind riir-ai's
  belief-lead dead-reckon read (their Bench 796: 2.5–5× better mean lead
  error than the frozen belief, bits-equal to this fast path). Opt-in POC —
  promotion via the T4 consumer A/Bs.
