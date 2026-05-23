# Research 74: Subterranean Agents — Compiling Agentic Workflows into LLM Weights

> **Paper:** [Compiling Agentic Workflows into LLM Weights: Near-Frontier Quality at Two Orders of Magnitude Less Cost](https://arxiv.org/pdf/2605.22502) — Dennis, Patil, Shabahang, Guo (i14, University of Melbourne), May 2026
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 21 (G-Zero Self-Play), 36 (ROPD Rubric), 37 (REAP Model-Based/Modelless), 38 (SDAR), 40 (Bradley-Terry), 51 (Deep Manifold), 54 (ASFT), 60 (MeMo), 62 (SHINE)
> **Related Plans:** 049 (G-Zero Self-Play), 052 (GFlowNet Modelless), 071 (ROPD Modelless), 072 (SDAR Modelless), 092 (Freeze/Thaw), 094 (MeMo Reflections)
> **Verdict: SELECTIVE ADOPTION — The "compile procedure into weights" pattern directly maps to our G-Zero modelless→model-based pipeline. Key takeaway: procedural knowledge is NOT low-rank (LoRA fails), full fine-tuning required. Our Freeze/Thaw + BanditPruner already implements a lightweight version of this. Distill the flowchart→synthetic data→fine-tune pipeline as a new `subterranean` feature gate for the Productions system. The flowchart-as-directed-graph abstraction fills a gap in our procedure representation.**
>
> **Related Plans:** 110 (Subterranean Procedure Compilation)

---

## TL;DR

The paper introduces **subterranean agents** — LLMs with procedural workflows compiled into their weights via full fine-tuning on synthetic conversations generated from flowcharts. Key results across 3 domains (travel 14 nodes, Zoom 14 nodes, insurance 55 nodes):

- **Quality**: 8B compiled model achieves 87–98% of frontier in-context quality
- **Cost**: 128–462× cheaper per conversation than in-context baseline (grows with complexity)
- **Flexibility**: 30–50 min recompile cycle on production hardware (CI/CD cadence)
- **Critical finding**: LoRA fails for procedural knowledge — full parameter updates required
- **Architecture**: Flowchart → synthetic conversations → full fine-tune → deploy without orchestrator

**For our stack:** Our G-Zero self-play pipeline already does modelless→model-based distillation. The gap is the **flowchart abstraction** for representing procedures as directed graphs and generating structured training data from them. This maps directly to our game state machines (Bomber, Go, FFT) and can extend to non-game domains. The LoRA limitation finding is critical — our Productions system should support full fine-tuning paths for procedural knowledge.

---

## Paper Architecture: Subterranean vs. Surface Orchestration

### Surface Orchestration (Current Industry Standard)
```
User → Orchestrator → LLM
         ↑              |
    inject prompts ← parse output
```
- External orchestrator sits between user and LLM
- Injects instructions and routing decisions every turn
- Frameworks: LangGraph, CrewAI, Google ADK, OpenAI Agents SDK (290K+ GitHub stars combined)
- **Problems**: Token cost, context window consumption, exposes proprietary procedures, routing failures

### Subterranean Agent (Paper's Approach)
```
Training: User → Orchestrator → LLM → synthetic conversations
Runtime:  User → LLM (no orchestrator, procedure in weights)
```
- Orchestrator used ONLY during training data generation
- At runtime, user talks directly to the LLM
- Procedure shapes training data; model learns to self-orchestrate through statistical regularities

### Procedure Representation
Flowchart as directed graph: `F = (N, E, n₀, T)`
- `N`: Nodes with role (agent/user) and prompt template
- `E ⊆ N × N × C`: Edges with optional conditions
- `n₀ ∈ N`: Start node
- `T ⊆ N`: Terminal nodes (success, abandonment, escalation)

---

## Key Empirical Findings

### 1. Quality: Compilation Helps (Same-Model A/B)

| Criterion | 3B Subterranean | 3B Orchestrated | Δ | p |
|---|---|---|---|---|
| Task Success | 4.11 | 3.93 | +0.18 | <0.001 |
| Info Accuracy | 4.75 | 4.69 | +0.05 | 0.29 |
| Consistency | 4.34 | 4.12 | +0.22 | <0.001 |
| Graceful Handling | 4.07 | 3.87 | +0.20 | <0.001 |
| Naturalness | 4.12 | 3.96 | +0.17 | <0.001 |

**Insight**: Compilation itself helps — same model, same procedure, different architecture. The compiled model beats explicit orchestration on 4/5 metrics (p<0.001).

### 2. 8B Compiled ≈ 70× Larger Frontier Orchestrator

| Domain | Metric | 8B Compiled vs LG Orch |
|---|---|---|
| Insurance (55 nodes) | Graceful Handling | 4.81 vs 4.38 (p<0.001) |
| Insurance (55 nodes) | Naturalness | 4.92 vs 4.58 (p<0.001) |
| Zoom (14 nodes) | Naturalness | 4.87 vs 4.64 (p<0.001) |
| Zoom (14 nodes) | Info Accuracy | 4.26 vs 4.75 (p<0.001) ← |

**Insight**: Compiled model wins on naturalness/graceful handling, loses on broad knowledge accuracy. The capacity gap matters most for world knowledge, not procedure following.

### 3. Cost Scales with Procedure Complexity

| Domain | Nodes | In-Context | Subterranean | Ratio |
|---|---|---|---|---|
| Travel | 14 | $0.133 | $0.0010 | 128× |
| Zoom | 14 | $0.103 | $0.0003 | 296× |
| Insurance | 55 | $0.327 | $0.0007 | 462× |

**Insight**: Cost advantage GROWS with procedure complexity because compiled model's prompt is constant-size. The more complex the procedure, the bigger the win.

### 4. LoRA Fails for Procedural Knowledge (Critical!)

> "Procedural internalization requires modifying the model's implicit state-tracking behavior — a deeper change than stylistic alignment. A systematic study of parameter-efficient (LoRA) fine-tuning across ranks 16–128 found that low-rank methods fail to approach full fine-tuning on procedural tasks [Dennis et al., 2026b]."

**This is the most important finding for our stack.** Our Productions pipeline currently uses LoRA (riir-burner). For procedural domains (game strategies, workflow logic), we need a full fine-tuning path.

### 5. Failure Rates

| Domain | Compiled | Orchestrated |
|---|---|---|
| Travel | 5.5% | 24.0% |
| Insurance | 9.0% | 17.0% |
| Zoom | 11.0% | 9.0% |

**Insight**: Compiled models have lower failure rates because they eliminate routing failures by construction.

### 6. Recompile Cycle = CI/CD (Not Paradigm Shift)

| Stage | 8×H200 | Single A100 |
|---|---|---|
| Data Generation | 15–30 min | 15–30 min |
| Fine-tuning | 10–15 min | ~3 hours |
| Evaluation | 5–15 min | 5–15 min |
| **Total** | **30–50 min** | **3–4 hours** |

**Insight**: Production hardware makes recompile a CI/CD cycle. This is the same cadence as our Freeze/Thaw pipeline (Plan 092).

---

## Mapping to Our Architecture

### Direct Equivalences

| Paper Concept | Our Implementation | Location |
|---|---|---|
| Flowchart `F = (N, E, n₀, T)` | GameState FSM + ActionSpaceLog | `src/pruners/game_state/` |
| Synthetic conversation generation | Self-play arena + replay backward | `src/pruners/arena/`, `src/pruners/bomber/replay_backward.rs` |
| Full fine-tuning | LoRA training (riir-burner) — **GAP**: no full FT path | `riir-ai/crates/riir-gpu/` |
| Runtime without orchestrator | WASM Validator + BanditPruner | `src/pruners/bandit.rs` |
| Recompile cycle | Freeze/Thaw pipeline | `src/pruners/freeze.rs` |
| Quality scoring (LLM-as-judge) | Bradley-Terry pairwise ranking | `src/pruners/bt_rank.rs` |
| Multi-criteria rubric | ROPD RubricVector | `src/pruners/ropd_rubric/types.rs` |

### What We Already Have

1. **Modelless→Model-Based Pipeline** (G-Zero, Plan 049)
   - Phase 1: Modelless (Hint-δ, bandit learns from self-play)
   - Phase 2: Model-Based (DPO/GRPO weight updates)
   - This IS the subterranean agent pattern for game domains

2. **Procedure-as-State-Machine** (Game arenas)
   - Bomber: 12×12 grid, action space, constraint validation
   - Go: 19×19 board, capture rules, komi scoring
   - FFT: ATB combat, status effects, turn order
   - Each is a directed graph of valid state transitions

3. **Training Data from Traversals** (Replay backward, template proposer)
   - `ReplayBackwardWalker`: walks winning replays backward through validator
   - `TemplateProposer`: generates (query, hint) pairs from game states
   - `MeMo Reflection`: 5-step QA synthesis from game replays

4. **Compiled Knowledge** (Freeze/Thaw)
   - `BomberFrozenBandit`, `GoFrozenBandit`: bandit knowledge as fixed binary blobs
   - Load at runtime, no external orchestrator needed

### What's Missing (The Gap)

1. **Directed Graph Abstraction for Procedures**
   - We represent procedures as `enum Action` + `GameState` trait
   - Paper uses explicit `F = (N, E, n₀, T)` with edge conditions
   - Gap: No generic `ProcedureGraph` that can enumerate all valid paths
   - Our games enumerate paths via self-play exploration, not graph traversal

2. **Synthetic Data Generation from Graph Traversal**
   - Paper: sample path → generate conversation → fine-tune
   - We: self-play → collect (state, action, reward) → bandit update
   - Gap: We don't generate structured training data from procedure graphs
   - Our bandit updates are online; paper does offline batch fine-tuning

3. **Full Fine-Tuning Path for Procedural Knowledge**
   - Paper proves LoRA fails for procedures
   - Our Productions pipeline uses LoRA exclusively
   - Gap: No full parameter update option for procedural domains
   - Only matters when we want to compile procedures into actual LLM weights

4. **Procedure Complexity Scaling**
   - Paper shows cost advantage grows with procedure complexity
   - We don't measure this in our distillation benchmarks
   - Gap: No complexity-proportional cost model

---

## Distillable Ideas

### D1: ProcedureGraph Trait (High Value, Low Effort)

```rust
/// Directed graph representation of a procedural workflow.
/// Paper: F = (N, E, n₀, T)
pub trait ProcedureGraph {
    type NodeId: Copy + Eq + Hash;
    type Condition;

    fn start_node(&self) -> Self::NodeId;
    fn terminal_nodes(&self) -> &[Self::NodeId];
    fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)];
    fn node_prompt(&self, node: Self::NodeId) -> &str;

    /// Enumerate all unique acyclic paths (paper: 86 paths for travel, 2381 for insurance)
    fn enumerate_paths(&self) -> Vec<Vec<Self::NodeId>>;

    /// Total path count (complexity metric for cost model)
    fn path_count(&self) -> usize {
        self.enumerate_paths().len()
    }
}
```

This fills the gap between our `GameState` trait (runtime game loop) and a declarative procedure representation that can generate training data.

### D2: PathSampler for Training Data (Medium Value, Medium Effort)

```rust
/// Sample paths through a procedure graph and generate training data.
/// Paper: "sample a path through the flowchart and a set of scenario variables,
/// then generate the conversation turn by turn along the path"
pub struct PathSampler<G: ProcedureGraph> {
    graph: G,
    rng: fastrand::Rng,
}

impl<G: ProcedureGraph> PathSampler<G> {
    /// Generate N synthetic trajectories by sampling paths
    pub fn sample_trajectories(&self, n: usize) -> Vec<Trajectory<G::NodeId>> {
        // Sample path, then generate turn-by-turn data
    }
}
```

### D3: Cost-Proportional Distillation Budget (Low Value, Low Effort)

```rust
/// Cost model: more complex procedures get more training budget.
/// Paper: "advantage grows with procedure complexity because compiled model's
/// prompt is constant-size"
pub fn distillation_budget(path_count: usize, base_budget: usize) -> usize {
    // Scale budget with procedure complexity
    base_budget * (path_count as f64).ln().ceil() as usize
}
```

### D4: Full Fine-Tuning Flag for Productions (High Value, High Effort)

```rust
/// Our Productions pipeline should support both paths:
/// - LoRA for stylistic/alignment knowledge (existing)
/// - Full FT for procedural/state-tracking knowledge (new)
pub enum TrainingMode {
    /// Low-rank adaptation — works for style, fails for procedures
    Lora { rank: usize },
    /// Full parameter updates — required for procedural internalization
    FullFineTune,
    /// QLoRA middle ground — 4-bit quant + full parameter update
    Qlora { bits: u8 },
}
```

This maps to the existing `TuningMethod` enum in riir-ai.

### D5: Compiled Failure Rate Metric (Low Value, Low Effort)

```rust
/// Paper: compiled models have lower failure rates than orchestrators.
/// Measure this in our arena system.
pub fn compiled_failure_rate(results: &[MatchupResult]) -> f64 {
    let failures = results.iter().filter(|r| r.score <= 3.0).count();
    failures as f64 / results.len() as f64
}
```

---

## What We Should NOT Adopt

1. **Claude Sonnet as data generator** — We use self-play + validator, not frontier model API
2. **LLM-as-judge scoring** — We have Bradley-Terry pairwise ranking (better calibrated)
3. **In-context baseline** — We don't serve frontier APIs; our baseline is modelless bandit
4. **LangGraph orchestrator comparison** — Not relevant to our Rust-native stack
5. **Dynamic user simulation** — Our game environments provide ground-truth opponents
6. **5-criteria rubric** — We already have ROPD with 5 criteria (TaskFulfillment, OutputStructure, ConstraintSatisfaction, Completeness, Correctness)

---

## Verdict

### SELECTIVE ADOPTION — Three actionable items

| Item | Value | Effort | Priority |
|---|---|---|---|
| **D1: ProcedureGraph trait** | Fills architectural gap, enables structured training data generation | Low | P1 |
| **D4: Full FT flag in Productions** | Critical for procedural domains (games, workflows) | High | P2 |
| **D2: PathSampler** | Enables offline training data from graph traversal | Medium | P3 |
| **D3: Cost-proportional budget** | Nice-to-have for scaling | Low | P4 |
| **D5: Failure rate metric** | Easy addition to arena system | Low | P5 |

### Feature Gate

```toml
subterranean = ["bandit"]  # ProcedureGraph + PathSampler for compiling workflows into weights
```

### Why This Matters for Us

The paper validates our G-Zero architecture at a fundamental level:
- **Modelless first** (Phase 1) = Paper's "training data generation via graph traversal"
- **Model-based second** (Phase 2) = Paper's "compile into weights via fine-tuning"
- **Freeze/Thaw** = Paper's "recompile cycle" at game-speed (seconds, not minutes)
- **BanditPruner** = Paper's "statistical regularities learned from data"

The key new insight is the **LoRA limitation for procedural knowledge**. Our game domains are inherently procedural. When we move from bandit-level distillation to actual weight updates (riir-burner), we need the full fine-tuning path. The ProcedureGraph abstraction gives us a generic way to represent any procedure (game rules, workflow logic, validation pipelines) as a directed graph that can enumerate paths for training data generation.

### Honest Assessment

- **Strength**: Paper validates our modelless→model-based pipeline pattern
- **Weakness**: Paper uses frontier model as data generator; we use self-play (different signal source)
- **Risk**: Full fine-tuning requires more compute than LoRA; may not fit our edge-deployment target
- **Opportunity**: ProcedureGraph abstraction could unify our game state machines under one trait

---

## References

- Dennis et al., 2026a: "In-context prompting obsoletes agent orchestration for procedural tasks"
- Dennis et al., 2026b: "Procedural knowledge is not low-rank: Why LoRA fails to internalize multi-step procedures"
- SimpleTOD (Hosseini-Asl et al., 2020): Single-sequence task-oriented dialogue
- FireAct (Chen et al., 2023): Fine-tuning on ReAct trajectories
- SynTOD (Samarinas et al., 2024): Synthetic data from state transition graphs (closest to our approach)
- WorkflowLLM (Fan et al., 2024): 106K workflow samples → 8B model
- Agent Lumos (Yin et al., 2024): Planning + grounding modules