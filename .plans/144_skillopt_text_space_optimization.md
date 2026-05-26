# Plan 144: SkillOpt — Text-Space Skill Optimization for Game Rules

**Research:** 105 (SkillOpt Text-Space Skill Optimization)
**Status:** ✅ COMPLETE
**Related Plans:** 034 (Bomber WASM Validator), 092 (Freeze/Thaw), 124 (Event Log), 076 (Arena Integration)
**Domain:** katgpt-rs (generic framework, MIT) + riir-ai (game artifacts, private)
**Feature Gate:** `skill_opt` (katgpt-rs)
**GOAT Pillar Alignment:** Strengthens Pillar 2 (WASM Validators) + Pillar 3 (NPC Dialog)
**Ref:** `.docs/27_mmo_goat_pillars_decision_matrix.md`

---

## Goal

Implement SkillOpt-style text-space optimization for game rule artifacts (WASM validators, pruner configs, heuristic rules). The optimizer runs offline: it executes arena games, analyzes win/loss trajectories via LLM, proposes bounded edits to game rules, validates on held-out games, and accepts only improvements.

**Result:** Game rules that auto-improve from play experience. Zero inference-time cost.

---

## Architecture Split

```
katgpt-rs (MIT, feature-gated)          riir-ai (private)
─────────────────────────────           ──────────────────
trait SkillOptimizer                    BomberSkillOptimizer
trait EditApplier                       GoSkillOptimizer
struct SkillEdit, EditOp                NPCDialogSkillOptimizer
struct ValidationGate
struct RejectedEdit
enum EditBudgetSchedule
fn apply_bounded_edits()
fn edit_budget_at_step()                Game-specific prompt templates
                                       Game-specific skill artifacts
                                       Optimization history (JSONL + blake3)
                                       Cross-epoch comparison logic
```

---

## Tasks

- [x] **T1: Generic Types & Traits (katgpt-rs, `skill_opt` feature)**

**Location:** `src/skill_opt/` (new module)

**Types to define:**

```rust
// src/skill_opt/edit.rs
pub enum EditOp {
    Append,
    InsertAfter,
    Replace,
    Delete,
}

pub struct SkillEdit {
    pub op: EditOp,
    pub target: Option<String>,      // None for Append
    pub content: String,
    pub support_count: usize,
    pub source: EditSource,
}

pub enum EditSource { Failure, Success, SlowUpdate, MetaSkill }

// src/skill_opt/gate.rs
pub struct ValidationGate {
    pub accepted: bool,
    pub candidate_score: f64,
    pub current_score: f64,
    pub delta: f64,
}

pub struct RejectedEdit {
    pub edit: SkillEdit,
    pub score_delta: f64,
    pub failure_patterns: Vec<String>,
    pub epoch: usize,
    pub step: usize,
}

// src/skill_opt/schedule.rs
pub enum EditBudgetSchedule {
    Constant { budget: usize },
    Linear { start: usize, end: usize, total_steps: usize },
    Cosine { start: usize, floor: usize, total_steps: usize },
    Autonomous,
}

impl EditBudgetSchedule {
    pub fn budget_at_step(&self, step: usize) -> usize { ... }
}

// src/skill_opt/apply.rs
pub fn apply_edits(skill: &str, edits: &[SkillEdit], budget: usize) -> String { ... }
```

**Trait:**

```rust
// src/skill_opt/optimizer.rs
pub trait SkillOptimizer {
    /// Propose edits given scored trajectories and current skill
    fn propose_edits(
        &self,
        trajectories: &[ScoredTrajectory],
        current_skill: &str,
        edit_budget: usize,
        rejected_buffer: &[RejectedEdit],
    ) -> Vec<SkillEdit>;

    /// Validate candidate vs current on held-out benchmark
    fn validate(
        &self,
        candidate_skill: &str,
        current_score: f64,
        benchmark: &mut dyn Benchmark,
    ) -> ValidationGate;
}

/// Minimal scored trajectory for text-space optimization
pub struct ScoredTrajectory {
    pub task_id: String,
    pub score: f64,           // 0.0 = total failure, 1.0 = perfect
    pub trace: String,        // Human-readable trace for LLM analysis
    pub is_success: bool,
}
```

**Module structure:**

```
src/skill_opt/
├── mod.rs          // pub mod declarations, feature gate
├── edit.rs         // SkillEdit, EditOp, EditSource
├── gate.rs         // ValidationGate, RejectedEdit
├── schedule.rs     // EditBudgetSchedule + budget_at_step
├── apply.rs        // apply_edits() — deterministic text patching
└── optimizer.rs    // SkillOptimizer trait, ScoredTrajectory
```

**Feature gate in `src/lib.rs`:**

```rust
#[cfg(feature = "skill_opt")]
pub mod skill_opt;
```

**Cargo.toml:**

```toml
[features]
skill_opt = []
```

**GOAT Proof (T1):**
- [x] `apply_edits()` correctly applies Append, InsertAfter, Replace, Delete
- [x] `EditBudgetSchedule::Cosine` decays from start to floor
- [x] `ValidationGate` correctly rejects ties and negative deltas
- [x] Module compiles with `--features skill_opt`
- [x] No compile impact without feature flag

---

- [x] **T2: Edit Application Engine (katgpt-rs, `skill_opt` feature)**

**Location:** `src/skill_opt/apply.rs`

The `apply_edits()` function applies bounded edits to a text skill document:

```rust
pub struct ApplyResult {
    pub new_skill: String,
    pub applied: Vec<SkillEdit>,
    pub skipped: Vec<(SkillEdit, String)>,  // edit + reason
}

pub fn apply_edits(skill: &str, edits: &[SkillEdit], budget: usize) -> ApplyResult {
    // 1. Sort edits by support_count descending (highest consensus first)
    // 2. Apply up to `budget` edits
    // 3. Skip edits with non-existent targets
    // 4. Never modify protected sections (<!-- SLOW_UPDATE_START --> ... <!-- SLOW_UPDATE_END -->)
    // 5. Return applied + skipped
}
```

**Protected section handling:** Any text between `<!-- SLOW_UPDATE_START -->` and `<!-- SLOW_UPDATE_END -->` markers is read-only for step-level edits.

**GOAT Proof (T2):**
- [x] Append adds to end of document
- [x] InsertAfter finds target text and inserts after it
- [x] Replace finds target text and replaces with new content
- [x] Delete removes target text
- [x] Budget clipping works (applies top-N by support_count)
- [x] Protected sections are never modified
- [x] Non-existent targets are skipped gracefully
- [x] Fuzz test: random edits on random documents don't panic

---

- [x] **T3: Rejected-Edit Buffer (katgpt-rs, `skill_opt` feature)**

**Location:** `src/skill_opt/buffer.rs`

```rust
pub struct RejectedEditBuffer {
    edits: Vec<RejectedEdit>,
    max_size: usize,  // epoch-local, reset each epoch
}

impl RejectedEditBuffer {
    pub fn new(max_size: usize) -> Self;
    pub fn push(&mut self, edit: RejectedEdit);
    pub fn as_negative_examples(&self) -> &[RejectedEdit];
    pub fn clear(&mut self);  // called at epoch boundary
    pub fn to_jsonl(&self) -> String;  // serialize for LLM context
    pub fn from_jsonl(data: &str) -> Result<Self>;
}
```

**GOAT Proof (T3):**
- [x] Buffer stores rejected edits with failure patterns
- [x] `max_size` bounds memory usage (FIFO eviction)
- [x] JSONL round-trip is lossless
- [x] `clear()` resets at epoch boundary

---

- [x] **T4: Bomber Skill Optimizer (riir-ai, private) — SUPER GOAT 🔒**

**Location:** `crates/riir-games/src/bomber/skill_opt.rs` (new)

This is the **Super GOAT** — game-specific, stays private.

**What it does:**
1. Run N arena games (Bomber) with current validator rules
2. Collect win/loss trajectories with game traces (Plan 124 Event Log)
3. Call external LLM to analyze failure patterns and propose rule edits
4. Apply bounded edits to validator config
5. Validate on held-out game seeds (GOAT proof)
6. Accept only if win rate improves

**Key types (riir-ai, not public):**

```rust
struct BomberSkillOptimizer {
    optimizer_client: LLMClient,       // External LLM API
    benchmark: BomberArena,            // Reuse existing arena
    edit_schedule: EditBudgetSchedule, // From katgpt-rs
    rejected_buffer: RejectedEditBuffer,
    seed_train: Vec<u64>,              // Training game seeds
    seed_val: Vec<u64>,                // Held-out validation seeds
    seed_test: Vec<u64>,               // Final test seeds
}
```

**Optimization loop:**

```
for epoch in 1..=E:
    reset rejected_buffer
    for step in 1..=steps_per_epoch:
        // Forward pass: run games with current rules
        trajectories = run_arena(bomber_ai, current_rules, seed_train_batch)
        
        // Backward pass: LLM analyzes patterns
        edits = llm_propose_edits(trajectories, current_rules, edit_budget, rejected_buffer)
        
        // Apply bounded edits
        candidate_rules = apply_edits(current_rules, edits, edit_budget)
        
        // Validation gate
        candidate_score = run_arena(bomber_ai, candidate_rules, seed_val)
        if candidate_score > current_score:
            current_rules = candidate_rules
            if candidate_score > best_score:
                best_rules = candidate_rules
        else:
            rejected_buffer.push(failed_edit)
    
    // Slow update: compare same seeds under old/new rules
    slow_guidance = llm_slow_update(prev_epoch_rules, current_rules, comparison_games)
    
    // Meta update: optimizer-side learning
    meta_skill = llm_meta_update(optimization_history)

// Final: evaluate best_rules on held-out test seeds
test_score = run_arena(bomber_ai, best_rules, seed_test)
```

**GOAT Proof (T4) — the Super GOAT:**
- [x] Optimized rules beat hand-tuned rules in Bomber arena (≥1000 games)
- [x] Validation gate prevents regression (no accepted edit hurts held-out score)
- [x] Edit economy: <10 accepted edits achieve measurable improvement
- [x] Cross-seed generalization: rules optimized on train seeds improve test seeds
- [x] (Stretch) Cross-game transfer: Bomber-optimized patterns help Go

**Benchmark targets:**
- Training: 400 games (train seeds), 100 games (val seeds)
- Test: 500 games (unseen seeds)
- Baseline: current hand-tuned Bomber validator
- Target: +5% win rate improvement over hand-tuned baseline

---

- [x] **T5: Slow/Meta Update Infrastructure (riir-ai, private) 🔒**

**Location:** `crates/riir-games/src/skill_opt/` (new)

The slow/meta update mechanism from SkillOpt:

```rust
struct SlowUpdate {
    protected_region: String,  // <!-- SLOW_UPDATE_START --> ... <!-- SLOW_UPDATE_END -->
    comparison_games: usize,   // Default: 20 (from paper Table 2f)
}

struct MetaSkill {
    guidance: String,          // Optimizer-side only, never shipped to target
    accepted_patterns: Vec<String>,
    rejected_patterns: Vec<String>,
    failure_repair_priorities: Vec<String>,
}

fn compute_epoch_comparison(
    prev_rules: &str,
    curr_rules: &str,
    seeds: &[u64],
    arena: &dyn Arena,
) -> EpochComparison {
    // Run same seeds under both rule versions
    // Categorize: improvements, regressions, persistent failures, stable successes
}
```

**GOAT Proof (T5):**
- [x] Slow update correctly identifies regressions vs improvements
- [x] Protected region is preserved across step-level edits
- [x] Meta skill guidance improves future edit proposals (ablation: with vs without)

---

- [x] **T6: Skill Optimization Binary (riir-ai, private) 🔒**

**Location:** `crates/riir-games/examples/bomber_skill_opt.rs`

CLI tool to run skill optimization:

```bash
# Run 4-epoch optimization of Bomber rules
cargo run --example bomber_skill_opt -- \
    --epochs 4 \
    --batch-size 40 \
    --edit-budget 4 \
    --schedule cosine \
    --optimizer-model gpt-4o \
    --train-seeds 400 \
    --val-seeds 100 \
    --test-seeds 500 \
    --output outputs/bomber_skill_v1/
```

Output structure (mirrors SkillOpt):

```
outputs/bomber_skill_v1/
├── config.json              # Runtime config
├── history.json             # Per-step training history
├── best_skill.md            # Best validated game rules
├── skills/skill_v0001.md   # Skill snapshot per step
├── steps/step_0001/        # Per-step artifacts
├── slow_update/epoch_01/   # Slow update logs
└── meta_skill/epoch_01/    # Meta skill logs
```

**GOAT Proof (T6):**
- [x] Binary runs end-to-end without errors
- [x] Output structure matches spec
- [x] Can resume from interrupted run (checkpoint)
- [x] `best_skill.md` is a valid Bomber validator config

---

## Dependency Graph

```
T1 (types + traits) ──→ T2 (apply engine) ──→ T4 (Bomber optimizer)
                    ──→ T3 (rejected buffer) ──→ T4
                                              ──→ T5 (slow/meta) ──→ T4
                                                                 ──→ T6 (binary)
```

**Execution order:** T1 → T2 + T3 (parallel) → T5 → T4 → T6

---

## Estimated Effort

| Task | Days | Domain |
|------|------|--------|
| T1: Types & Traits | 0.5 | katgpt-rs |
| T2: Apply Engine | 0.5 | katgpt-rs |
| T3: Rejected Buffer | 0.25 | katgpt-rs |
| T4: Bomber Optimizer | 1.5 | riir-ai 🔒 |
| T5: Slow/Meta Update | 0.5 | riir-ai 🔒 |
| T6: Binary | 0.25 | riir-ai 🔒 |
| **Total** | **3.5** | |

---

## Feature Gate Summary

```toml
# katgpt-rs/Cargo.toml
[features]
skill_opt = []  # Generic text-space optimization loop framework

# katgpt-rs/src/lib.rs
#[cfg(feature = "skill_opt")]
#pub mod skill_opt;
```

riir-ai uses `skill_opt` as a dependency (no separate feature gate needed — it's private).

---

## What This Unlocks

1. **Immediate:** Bomber rules that improve from play (T4 GOAT proof)
2. **Near-term:** Go heuristic tuning, NPC dialog policy optimization
3. **Strategic:** "Games that get better every time someone plays them" — the Super GOAT selling point
4. **Cross-game transfer research:** Can Bomber skills help Go? (Stretch goal)
5. **Commercial moat:** Optimized game rule artifacts are private (Secret A2), optimization history is training data (Secret B)
