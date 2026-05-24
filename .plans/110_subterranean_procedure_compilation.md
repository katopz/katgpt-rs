# Plan 110: Subterranean Procedure Compilation

> **Research:** 074 (Subterranean Agents — Compiling Workflows into Weights)
> **Paper:** [arXiv:2605.22502](https://arxiv.org/pdf/2605.22502) — Dennis et al., May 2026
> **Related Plans:** 049 (G-Zero Self-Play), 052 (GFlowNet Modelless), 071 (ROPD Modelless), 092 (Freeze/Thaw), 094 (MeMo Reflections)
> **Feature Gate:** `subterranean = ["bandit"]`
> **Status:** Complete

## Tasks

- [x] T1: `ProcedureGraph` trait + `ProcedureEdge` + `ProcedureNode` types
- [x] T2: `PathEnumerator` — enumerate all acyclic paths through procedure graph
- [x] T3: `PathSampler` — sample trajectories with scenario variables
- [x] T4: `ProcedureCostModel` — complexity-proportional budget allocation
- [x] T5: GOAT proof — Bomber procedure graph + path enumeration benchmark
- [x] T6: GOAT proof — Go procedure graph + complexity scaling benchmark
- [x] T7: Integration — `ProcedureGraph` ↔ existing `GameState` trait bridge
- [x] T8: Integration — `PathSampler` → bandit training data (replace template proposer)
- [x] T9: Full fine-tuning flag — `TrainingMode::FullFineTune` variant in Productions config

## Motivation

The paper proves that **compiling procedural workflows into model weights** achieves 87–98% of frontier quality at 128–462× lower cost. Our G-Zero self-play pipeline already follows the modelless→model-based pattern, but lacks:

1. **Declarative procedure representation** — Our `GameState` trait is runtime-only; we can't enumerate valid paths offline
2. **Structured training data from graph traversal** — We rely on self-play exploration, not exhaustive path sampling
3. **Full fine-tuning path for procedural knowledge** — Paper proves LoRA fails for procedures (ranks 16–128 all fail)

The `ProcedureGraph` trait fills an architectural gap: it lets us represent any procedure (game rules, validation pipelines, workflow logic) as a directed graph that can enumerate paths for training data generation, estimate complexity for cost modeling, and bridge to our existing `GameState` trait for runtime execution.

## Architecture

### Core Abstraction

```
ProcedureGraph (directed graph)     ← New: declarative, offline
    ↓ enumerate_paths()
    ↓ sample_trajectories()
PathSampler (training data gen)     ← New: structured data from paths
    ↓ Vec<Trajectory>
    ↓ to_bandit_sessions()
BanditPruner (online learning)      ← Existing: learns from trajectories
    ↓ freeze()
FrozenBandit (compiled knowledge)   ← Existing: no orchestrator needed
```

### Feature Gate

```toml
subterranean = ["bandit"]  # ProcedureGraph + PathSampler for compiling workflows into weights
```

---

## T1: ProcedureGraph Trait

**File:** `src/pruners/subterranean/mod.rs` (new)
**Feature gate:** `subterranean`

```rust
/// Directed graph representation of a procedural workflow.
/// Paper: F = (N, E, n₀, T) where N=nodes, E=edges with conditions,
/// n₀=start, T=terminal nodes.
pub trait ProcedureGraph {
    type NodeId: Copy + Eq + Hash + fmt::Debug;
    type Condition: fmt::Debug;

    fn start_node(&self) -> Self::NodeId;
    fn terminal_nodes(&self) -> &[Self::NodeId];
    fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)];
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;

    /// Node label for human-readable path descriptions
    fn node_label(&self, node: Self::NodeId) -> &str;
}
```

**Types file:** `src/pruners/subterranean/types.rs`

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcedureNode {
    pub id: u32,
    pub label: String,
    pub is_terminal: bool,
    pub is_start: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcedureEdge {
    pub from: u32,
    pub to: u32,
    pub condition: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Trajectory<NodeId> {
    pub path: Vec<NodeId>,
    pub conditions_met: Vec<Option<String>>,
}
```

**Acceptance criteria:**
- Trait compiles under `subterranean` feature gate
- Types are `serde`-serializable
- No dependencies beyond `std` + `serde`

---

## T2: PathEnumerator

**File:** `src/pruners/subterranean/path_enumerator.rs`
**Feature gate:** `subterranean`

Implements exhaustive acyclic path enumeration using DFS with visited tracking.

```rust
/// Enumerates all unique acyclic paths through a procedure graph.
/// Paper: travel=86 paths, zoom=60 paths, insurance=2,381 paths.
pub struct PathEnumerator<'a, G: ProcedureGraph> {
    graph: &'a G,
    max_depth: usize, // Safety limit to prevent exponential blowup
}

impl<'a, G: ProcedureGraph> PathEnumerator<'a, G> {
    pub fn new(graph: &'a G, max_depth: usize) -> Self;

    /// Enumerate all acyclic paths from start to any terminal node.
    /// Returns empty if max_depth exceeded.
    pub fn enumerate(&self) -> Vec<Trajectory<G::NodeId>>;

    /// Count paths without materializing them (for cost estimation).
    pub fn count_paths(&self) -> usize;

    /// Sample N random paths using weighted selection.
    /// Shorter paths get higher weight (paper: paths vary 4–39 turns).
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Trajectory<G::NodeId>>;
}
```

**Acceptance criteria:**
- Correctly enumerates paths for 14-node graph (should match paper's 86/60 counts for similar topologies)
- `count_paths()` matches `enumerate().len()`
- `max_depth` prevents infinite loops on cyclic graphs
- Fuzzer test: random graphs with up to 100 nodes

---

## T3: PathSampler

**File:** `src/pruners/subterranean/path_sampler.rs`
**Feature gate:** `subterranean`

Generates training data from sampled trajectories by mapping graph nodes to game actions.

```rust
/// Generates structured training data from procedure graph paths.
/// Paper: "sample a path → scenario variables → generate turn-by-turn"
pub struct PathSampler<'a, G: ProcedureGraph> {
    enumerator: PathEnumerator<'a, G>,
    rng: fastrand::Rng,
}

/// A single training sample extracted from a trajectory.
#[derive(Debug, Clone)]
pub struct Sample<NodeId> {
    pub trajectory: Trajectory<NodeId>,
    pub turn_index: usize,
    pub node_label: String,
    pub valid_next_actions: Vec<NodeId>,
    pub chosen_action: NodeId,
}
```

**Bridge to BanditPruner:**

```rust
/// Convert sampled trajectories into bandit training sessions.
impl<G: ProcedureGraph> PathSampler<'_, G> {
    pub fn to_bandit_sessions(&self, trajectories: &[Trajectory<G::NodeId>])
        -> Vec<BanditSession>;
}
```

**Acceptance criteria:**
- Generates N samples from graph paths
- Each sample has valid `chosen_action` from `valid_next_actions`
- BanditSession conversion produces correct arm counts

---

## T4: ProcedureCostModel

**File:** `src/pruners/subterranean/cost_model.rs`
**Feature gate:** `subterranean`

```rust
/// Cost model based on paper's finding: cost advantage scales with procedure complexity.
/// Paper: 128× for 14 nodes, 462× for 55 nodes.
#[derive(Debug, Clone)]
pub struct ProcedureCostModel {
    pub node_count: usize,
    pub path_count: usize,
    pub avg_path_length: f64,
}

impl ProcedureCostModel {
    /// Estimate cost ratio: compiled vs. in-context.
    /// Paper: ratio = base × ln(path_count) factor
    pub fn cost_ratio_vs_in_context(&self) -> f64 {
        let base = 65.0; // Per-token cost ratio (paper: ~65×)
        let volume_factor = 1.0 + (self.path_count as f64).ln().max(1.0);
        base * volume_factor
    }

    /// Recommended training budget proportional to complexity.
    pub fn recommended_budget(&self, base_budget: usize) -> usize {
        base_budget * (self.path_count as f64).ln().ceil().max(1.0) as usize
    }

    /// Training time estimate (single A100, paper: ~3-4 hours for 55 nodes)
    pub fn estimated_training_hours(&self) -> f64 {
        1.0 + (self.path_count as f64 / 500.0).min(3.0)
    }
}
```

**Acceptance criteria:**
- Cost model returns values in paper's ballpark (128–462× for 14–55 node graphs)
- Budget scaling is monotonic with path count
- Unit tests against paper's reported numbers

---

## T5: GOAT Proof — Bomber Procedure Graph

**File:** `src/pruners/subterranean/bomber_procedure.rs`
**Feature gate:** `subterranean`
**Benchmark:** `.benchmarks/024_subterranean_bomber_procedure.md`

Implement `ProcedureGraph` for Bomberman game:

```rust
/// Bomberman game as a procedure graph.
/// Nodes: game states (start, placing_bomb, moving, waiting, game_over)
/// Edges: valid transitions (place_bomb → wait → explosion → check_alive)
pub struct BomberProcedure {
    nodes: Vec<ProcedureNode>,
    edges: Vec<ProcedureEdge>,
}
```

**GOAT proof targets:**
- Path enumeration completes in <100ms for 12×12 grid
- Enumerated paths cover >90% of self-play game trajectories
- Cost model predicts correctly for Bomber complexity

**Benchmark metrics:**
1. `path_count` — Total unique acyclic paths
2. `enumerate_time_ms` — Time to enumerate all paths
3. `path_coverage` — % of self-play trajectories covered by enumerated paths
4. `cost_ratio` — Predicted cost savings vs. in-context

---

## T6: GOAT Proof — Go Procedure Graph

**File:** `src/pruners/subterranean/go_procedure.rs`
**Feature gate:** `subterranean`
**Benchmark:** `.benchmarks/025_subterranean_go_procedure.md`

Implement `ProcedureGraph` for Go game (9×9 for tractability):

```rust
/// Go game as a procedure graph.
/// Nodes: game phases (opening, midgame, endgame, scoring, resigned)
/// Edges: valid transitions conditioned on board state
pub struct GoProcedure {
    nodes: Vec<ProcedureNode>,
    edges: Vec<ProcedureEdge>,
    board_size: usize, // 9 for tractability
}
```

**GOAT proof targets:**
- 9×9 path enumeration completes in <1s
- Path count grows exponentially with board size (validate complexity scaling)
- Cost model shows increasing advantage with board size

**Benchmark metrics:**
1. `path_count_9x9` — Paths for 9×9 board
2. `enumerate_time_9x9_ms` — Enumeration time
3. `cost_ratio_vs_board_size` — Scaling curve for 5×5, 7×7, 9×9, 13×13, 19×19

---

## T7: Integration — ProcedureGraph ↔ GameState Bridge

**File:** `src/pruners/subterranean/game_bridge.rs`
**Feature gate:** `subterranean`, `game_state`

```rust
/// Bridge between declarative ProcedureGraph and runtime GameState.
/// Allows generating training data from graph traversal, then validating
/// at runtime via GameState.
pub trait ProcedureGameState: ProcedureGraph + GameState {
    /// Map a graph node to a game state
    fn node_to_state(&self, node: Self::NodeId) -> Option<Self::State>;

    /// Map a game state to the closest graph node
    fn state_to_node(&self, state: &Self::State) -> Option<Self::NodeId>;
}
```

**Acceptance criteria:**
- Bridge compiles for existing game implementations
- Round-trip: node → state → node is identity for terminal nodes
- Zero-cost abstraction (no runtime overhead when feature disabled)

---

## T8: Integration — PathSampler → Bandit Training Data

**File:** `src/pruners/subterranean/bandit_bridge.rs`
**Feature gate:** `subterranean`

```rust
/// Convert PathSampler output to BanditPruner training sessions.
/// This replaces the manual template proposer with structured graph-based data.
pub fn graph_trajectories_to_sessions<G: ProcedureGraph>(
    trajectories: &[Trajectory<G::NodeId>],
    graph: &G,
) -> Vec<BanditSession> {
    trajectories.iter().map(|t| {
        let mut session = BanditSession::new(t.path.len());
        for (i, &node) in t.path.iter().enumerate() {
            let arms: Vec<String> = graph.edges_from(node)
                .iter().map(|(next, _)| format!("{node:?}->{next:?}")).collect();
            if i + 1 < t.path.len() {
                let chosen = format!("{node:?}->{:?}", t.path[i + 1]);
                session.record_choice(&arms, &chosen, 1.0);
            }
        }
        session
    }).collect()
}
```

**Acceptance criteria:**
- Generated sessions produce valid bandit updates
- Bandit trained on graph data matches bandit trained on self-play data (within 10%)
- Benchmark: graph-based vs self-play-based bandit convergence rate

---

## T9: Full Fine-Tuning Flag — Productions Config

**File:** `src/pruners/subterranean/training_mode.rs`
**Feature gate:** `subterranean`

```rust
/// Training mode for compiling procedures into weights.
/// Paper: "Procedural internalization requires modifying the model's implicit
/// state-tracking behavior — a deeper change than stylistic alignment.
/// LoRA fails to approach full fine-tuning on procedural tasks."
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SubterraneanTrainingMode {
    /// Low-rank adaptation — works for style, FAILS for procedures
    Lora { rank: usize },
    /// Full parameter updates — REQUIRED for procedural internalization
    FullFineTune,
    /// QLoRA middle ground — 4-bit quant + full parameter update
    Qlora { bits: u8 },
}

impl Default for SubterraneanTrainingMode {
    fn default() -> Self {
        Self::FullFineTune // Paper proves this is needed for procedures
    }
}
```

**Note:** Actual full fine-tuning implementation belongs in `riir-ai` (riir-gpu). This task only adds the config type and feature gate in katgpt-rs.

---

## Module Structure

```
src/pruners/subterranean/
├── mod.rs                    # Module index, re-exports
├── types.rs                  # ProcedureNode, ProcedureEdge, Trajectory
├── path_enumerator.rs        # PathEnumerator (DFS acyclic path enumeration)
├── path_sampler.rs           # PathSampler (training data from paths)
├── cost_model.rs             # ProcedureCostModel (complexity-based budgeting)
├── bomber_procedure.rs       # BomberProcedure (GOAT proof T5)
├── go_procedure.rs           # GoProcedure (GOAT proof T6)
├── game_bridge.rs            # ProcedureGraph ↔ GameState bridge (T7)
├── bandit_bridge.rs          # PathSampler → BanditPruner conversion (T8)
└── training_mode.rs          # SubterraneanTrainingMode enum (T9)
```

## Feature Gate in Cargo.toml

```toml
subterranean = ["bandit"]  # ProcedureGraph + PathSampler for compiling workflows into weights
```

Add to `full` feature:
```toml
full = [..., "subterranean"]
```

## Testing Strategy

1. **Unit tests:** PathEnumerator correctness with known graphs (paper's 14-node, 55-node topologies)
2. **Property tests:** Proptest for random graphs — all enumerated paths start at `n₀`, end at terminal
3. **GOAT proofs:** T5 (Bomber), T6 (Go) — benchmarks with target thresholds
4. **Integration tests:** BanditPruner trained on graph data vs self-play data

## Dependencies

- `bandit` (existing) — BanditPruner, BanditSession
- `game_state` (optional, for T7 bridge) — GameState trait
- `bomber` (optional, for T5 GOAT proof) — BomberProcedure
- `go` (optional, for T6 GOAT proof) — GoProcedure

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Path enumeration exponential blowup | `max_depth` safety limit, early termination |
| Graph doesn't match self-play trajectories | Measure coverage, not exact match |
| Full FT requires GPU training | Config type only in katgpt-rs; actual training in riir-ai |
| LoRA vs Full FT only matters at scale | Feature gate keeps it optional |

## Success Criteria

1. `ProcedureGraph` trait compiles and is implemented for Bomber + Go
2. Path enumeration produces correct counts for known topologies
3. BanditPruner trained on graph data converges within 10% of self-play baseline
4. Cost model estimates match paper's reported ratios (128–462×)
5. All GOAT proofs pass with target thresholds
6. Zero overhead when `subterranean` feature disabled