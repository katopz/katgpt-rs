# Lean4Agent: Formal Workflow Verification → Modelless Fusion

**Date:** 2026-06-08
**Paper:** arXiv:2606.06523 — Lean4Agent: Formal Modeling and Verification for Agent Workflow and Trajectory
**Domain:** Modelless (katgpt-rs) + Model-Based (riir-ai)

---

## Paper Distillation

Lean4Agent proposes a **3-layer Lean4 formal verification framework** for LLM agent workflows:

| Layer | What | Analogy |
|-------|------|---------|
| **Layer 1** | Structural verification — type system, graph well-formedness, read/write consistency | Compiler frontend checks |
| **Layer 2** | Static semantic verification — Hoare-style pre/postconditions under `LLMExec` assumption | Type + contract checker |
| **Layer 3** | Trajectory verification — localize which step failed via Lean/external/LLM-as-judge | Runtime assertion + backtrace |

**Key Results:**
- Verification-passing workflows outperform failing ones by **11.94% avg** (14.80% SWE, 9.07% paper understanding)
- `LeanEvolve` (formal-guided workflow revision) adds **+7.47%** on SWE-Bench
- Formal-guided evolution beats pure-LLM evolution by **7.00%** on paper understanding
- Gains larger for smaller models (27.33% for Gemma-4-31B on SWE)

**Core Innovation — `LLMExec` Assumption:**
```
∀s_i ∈ S, ∀Π, Π |= ρ(pre)_i ⟹ ∃Π' = LLM(v_i, Π) ∧ Π' |= ρ(post)_i
```
If preconditions hold, the LLM locally satisfies postconditions. This is a **sound assumption** for short-horizon steps, making static verification tractable without executing the model.

**Core Innovation — Predicate System:**
- `PredicateType`: inductive type with base predicates (nameExists, isNonEmptyString, matchesJsonSchema, etc.) + compositional (AND, OR) + extensible (user-defined via `PredicateKey`)
- `toProp`: maps each predicate to a Lean proposition
- **Graph-level predicates**: information flow, context management, evaluation independence — catch non-trivial errors that local predicates miss

**Core Innovation — LeanEvolve:**
1. Layer-3 localizes failure-inducing step + violated predicates
2. LLM revises that step's instruction using formal diagnostics
3. Re-run evolved workflow on same problem
4. Optional pure-LLM evolve add-on for broader exploration

---

## Mapping to Our Architecture

### What We Already Have (GOAT)

| Lean4Agent | Our Stack | Status |
|------------|-----------|--------|
| Layer 1 (structural) | `ConstraintPruner::is_valid()` + DDTree graph well-formedness | ✅ Working |
| Layer 2 (semantic predicates) | `ScreeningPruner::relevance()` + `GenerativeConstraintPruner<Output>` | ✅ Working |
| Layer 3 (trajectory) | `VrLoop` (verify-refine) + `SpecReconciler` | ✅ Working |
| LeanEvolve | Episode-Guided Constraint Synthesis (Plan 206) | 🟡 Planned |
| PredicateType | DDTree path constraints + WASM validator truth table | ✅ Working |
| LLMExec assumption | `ConstraintPruner` assumes valid-prefix-pruning | ✅ Implicit |

### What We're Missing (Gap)

| Concept | Status | Impact |
|---------|--------|--------|
| **Hoare-style pre/postcondition propagation** | ❌ None | Our ConstraintPruner is stateless — it checks per-token, not per-step with propagated state |
| **Implicit variable tracking** | ❌ None | No tracking of "which information does this step consume/produce" |
| **Graph-level predicates** | ❌ None | No detection of context-management errors (e.g., step reads from context that isn't propagated) |
| **Formal specification language** | ❌ None | FOL-LNN (184) extracts rules, but no declarative spec language for workflow contracts |
| **Proof certificates** | ❌ None | Research 106 proposed them, never implemented |

---

## Creative Fusion Ideas (Not Direct Mapping)

### Idea 1: `HoarePruner` — Stateful Predicate Propagation for DDTree

**Insight:** Lean4Agent's Layer 2 is essentially a **constraint propagation system** where postconditions from step N become preconditions for step N+1. Our `ConstraintPruner` already does this per-token (ancestor tokens constrain child tokens). The gap is at the **step level** — we don't track what semantic properties each DDTree path establishes.

**Fusion:** Extend `ConstraintPruner` with a `SemanticState` accumulator:

```rust
trait ConstraintPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[u16]) -> bool;
    // NEW: predicate propagation across steps
    fn propagate(&self, state: &SemanticState, depth: usize, token_idx: usize) -> Option<SemanticState>;
}

struct SemanticState {
    established_predicates: Vec<Predicate>,
    /// BLAKE3 hash of the state for deterministic replay
    hash: [u8; 32],
}
```

Each DDTree node carries a `SemanticState` — postconditions from parent become preconditions for children. This is **modelless** (no LLM training), purely inference-time constraint propagation. It's also **zero-cost** when predicates are trivially satisfied (common case).

**Why this is novel:** Lean4Agent uses Lean4 (heavy formal verification). We use Rust + BLAKE3 + sigmoid — lightweight, deterministic, fast. No proof assistant needed. The predicate system is extensible via WASM validators (our existing moat).

**GOAT Gate:** `hoare_pruner` — off by default, on if benchmark shows improvement.

### Idea 2: `WorkflowLattice` — Compositional Predicate Lattice for Speculative Decoding

**Insight:** Lean4Agent's `PredicateType` with AND/OR composition forms a **lattice** — predicates are partially ordered by implication. This maps directly to DDTree's AND-OR decomposition (Plan 190). The lattice structure means we can **propagate predicate satisfaction incrementally** — if we know predicate P is satisfied at depth D, we don't need to re-check it at D+1.

**Fusion:** Build a predicate lattice that composes with DDTree's BFS expansion:

```rust
struct WorkflowLattice {
    /// Predicates ordered by implication strength
    predicates: Vec<PredicateNode>,
    /// Join: P1 ∧ P2 → least upper bound
    join_table: Vec<Vec<usize>>,
    /// Meet: P1 ∨ P2 → greatest lower bound
    meet_table: Vec<Vec<usize>>,
}

impl ScreeningPruner for WorkflowLattice {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[u16]) -> f32 {
        // Check which predicates are satisfied by this token
        // Return sigmoid(dot(established, required)) — how many required predicates are met
    }
}
```

**Why this is novel:** Lean4Agent checks full predicate satisfaction (boolean). We use sigmoid-graded relevance — partial satisfaction gets partial score, allowing the DDTree to explore promising-but-incomplete paths. This is a continuous relaxation of Lean4Agent's discrete verification.

**GOAT Gate:** `workflow_lattice` — off by default.

### Idea 3: `TrajectoryDoctor` — Modelless Failure Localization via DDTree Replay

**Insight:** Lean4Agent's Layer 3 localizes failures by checking trajectory against step-level postconditions. We already have `VrLoop` (verify-refine) but it doesn't localize **which token** caused the failure — it just rejects/accepts entire candidates.

**Fusion:** When `GenerativeConstraintPruner<Output>` rejects an output, trace back through the DDTree path to find the **first token where a predicate became unsatisfiable**:

```rust
trait TrajectoryDoctor {
    /// Given a rejected output, find the earliest DDTree node where a predicate was violated
    fn localize_failure(&self, trajectory: &[u16], pruner: &dyn ConstraintPruner) -> Option<FailureSite>;
}

struct FailureSite {
    /// Depth in DDTree where failure originated
    depth: usize,
    /// Token that caused the cascade
    token_idx: usize,
    /// Which predicate was violated
    violated_predicate: Predicate,
    /// Suggested replacement tokens (from DDTree siblings)
    suggested_alternatives: Vec<u16>,
}
```

This is **modelless** — no LLM needed for localization, just DDTree replay with predicate checking. The suggested alternatives come from DDTree siblings that pass the pruner.

**Connection to existing plans:** This is the inference-time component of Episode-Guided Constraint Synthesis (Plan 206). Episodes store failure sites, not just raw tokens.

**GOAT Gate:** `trajectory_doctor` — off by default.

### Idea 4 (GOAT — Must Default On): `LLMExecGuard` — Collapse-Aware Adaptive Step Verification

**Insight:** Lean4Agent's `LLMExec` assumption is that short-horizon LLM steps are locally correct. We already have Collapse-Aware Adaptive Thinking (Plan 212) that detects when CoT is not helping. The fusion: **verify the LLMExec assumption at inference time using entropy collapse detection**.

When entropy collapses (Plan 212), the model is confident → LLMExec holds. When entropy is high (model uncertain), LLMExec may not hold → we need extra verification (ScreeningPruner + VrLoop rescue).

**Fusion:** Use entropy as a **runtime LLMExec confidence proxy**:

```rust
fn llmexec_confidence(entropy: f32, depth: usize) -> f32 {
    // sigmoid over entropy — high entropy = low confidence = LLMExec may not hold
    sigmoid(-entropy * LAMBDA + depth * BETA)
}
```

- High confidence (sigmoid > 0.8): Skip expensive verification, LLMExec holds
- Low confidence (sigmoid < 0.3): Engage ScreeningPruner + TrajectoryDoctor
- Medium: Use ScreeningPruner relevance to gate

**Why GOAT and default-on:** This makes Plans 212 (Collapse-Aware) and 196 (VortexFlow) work together — entropy collapse detection drives verification budget allocation. It's zero-cost when LLMExec holds (no extra verification), and pays for itself by catching failures early. No perf hurt because it replaces the existing `ScreeningPruner::relevance()` call with a cheaper entropy check in the common case.

**Constraint:** Must benchmark against current ScreeningPruner baseline. If regression > 2%, gate behind `llmexec_guard`.

### Idea 5: `WASMProofWitness` — Proof Certificates for WASM Validators

**Insight:** Lean4Agent produces Lean proof artifacts. Our WASM validators produce boolean results. If we add **witnesses** (why the validator accepted/rejected), we get proof certificates without Lean.

**Fusion:** Extend WASM validator interface to return a witness:

```rust
#[repr(C)]
struct ValidationResult {
    valid: u32,           // 0 or 1
    witness_hash: [u8; 32], // BLAKE3 hash of the proof witness
    violated_rule: u32,   // index of violated rule (if invalid)
}
```

The witness is a compact trace of which rules were checked, which passed, which failed. BLAKE3 hash ensures determinism. This connects to Research 106 (Shock: proof certificates) and our anti-cheat replay system (seal-online-remaster).

**Domain:** Both modelless (katgpt-rs) and model-based (riir-ai). The WASM validator is shared infrastructure.

**GOAT Gate:** `wasm_proof_witness` — off by default (changes WASM ABI).

---

## Verdict

| Idea | Domain | Novel | GOAT | Default | Rationale |
|------|--------|-------|------|---------|-----------|
| **Idea 4: LLMExecGuard** | katgpt-rs | ✅ | ✅ | ✅ ON | Entropy-driven verification budget — fuses Plans 212+196, zero-cost common case, catches failures early |
| **Idea 1: HoarePruner** | katgpt-rs | ✅ | ✅ | OFF | Stateful predicate propagation — novel modelless Hoare-style reasoning for DDTree |
| **Idea 2: WorkflowLattice** | katgpt-rs | ✅ | ✅ | OFF | Continuous relaxation of Lean4 discrete verification via sigmoid-graded lattice |
| **Idea 3: TrajectoryDoctor** | katgpt-rs | ✅ | ✅ | OFF | Failure localization via DDTree replay — connects to Plan 206 (Episode-Guided) |
| **Idea 5: WASMProofWitness** | Both | ✅ | ✅ | OFF | Proof certificates for WASM validators — shared moat infrastructure |

### Why These Are Fundamentally Different From Lean4Agent

1. **No Lean4 dependency** — all verification is Rust-native + WASM + BLAKE3. No external proof assistant.
2. **Sigmoid, not boolean** — continuous predicate satisfaction via sigmoid, not discrete Lean propositions. Allows partial credit.
3. **DDTree-native** — verification is embedded in the DDTree BFS expansion, not a separate verification pass. Zero overhead when pruner is trivially satisfied.
4. **Modelless first** — LLMExecGuard, HoarePruner, TrajectoryDoctor all work without any model training. They're inference-time constraint systems.
5. **Commercial moat** — WASM validators with proof witnesses become the "validator.wasm" fuel from Strategy Verdict (003). Lean4Agent is open academic work; our validators are private SaaS intelligence.

### Commercial Alignment (per 003 Verdict)

These ideas strengthen the Engine/Fuel split:
- **Engine (MIT):** `HoarePruner` trait, `TrajectoryDoctor` trait, `LLMExecGuard` entropy logic — all open, all inference-time
- **Fuel (SaaS):** `WASMProofWitness` proof certificates, domain-specific predicate libraries, WASM validators with witnesses — closed, proprietary
- **Flywheel:** Every failure localized by TrajectoryDoctor feeds into Episode DB → better validators → better translations → more episodes

---

## Related Research

| # | File | Connection |
|---|------|-----------|
| 184 | FOL-LNN Logical Rules | DDTree IS a Logical Neural Network — FOL extraction is the inverse of predicate injection |
| 170 | LEAP Blueprint DAG | AND-OR DAG proof search validates DDTree + ConstraintPruner stack |
| 212 | Collapse-Aware Thinking | LLMExecGuard fuses entropy collapse with verification budget |
| 206 | Episode-Guided Constraint Synthesis | TrajectoryDoctor is the inference-time component |
| 156 | Speculative Reconciliation | SpecReconciler = Layer 3 trajectory verification analogue |
| 182 | STV Self-Trained Verification | Episode-conditioned verification maps to predicate propagation |
| 050 | LDT Lattice Deduction | WorkflowLattice extends LDT's sound deduction to predicate lattices |
| 005 | Artifact Definition | Canonical spec for ConstraintPruner interface |
| 011 | PPoT Probabilistic Programs | Post-DDTree rescue = LeanEvolve analogue |

---

## TL;DR

Lean4Agent proves that **formal predicate verification of agent workflows gives 12% average improvement** — significant, reproducible across 5 models. The key insight is the `LLMExec` assumption: short-horizon LLM steps are locally correct if preconditions hold. We fuse this insight into our existing `ConstraintPruner` → `ScreeningPruner` → `DDTree` stack by adding **entropy-driven verification budgeting** (LLMExecGuard, default-on), **stateful predicate propagation** (HoarePruner), and **failure localization** (TrajectoryDoctor). All modelless, all Rust-native, all feeding the Episode DB flywheel. The WASM proof witness (Idea 5) extends our commercial moat by making validators produce auditable proof certificates.

**Decision:** 5 research-grade ideas distilled. 1 default-on (LLMExecGuard), 4 opt-in behind feature gates. Plan to follow.
