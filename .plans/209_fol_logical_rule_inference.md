# Plan 209: FOL Logical Rule Inference — Modelless DDTree→FOL Pipeline

**Date:** 2026-06-07
**Status:** 🔧 In Progress
**Research:** `.research/184_FOL_LNN_Inference_Time_Logical_Rules.md`
**Depends On:** Plan 190 (AND-OR DDTree), Plan 206 (EGCS/EpisodePruner), BanditPruner, SynPruner, `ConstraintPruner` trait
**Feature Gates:** `fol_constraints`, `rule_extraction`, `reward_mem`, `decision_trace` (each independently gateable)
**GOAT Criteria:** FOL extraction accuracy ≥80% on Rust prompts; rule reuse ≥30%; reward gain ≥10%; zero overhead on miss path

---

## Problem

DDTree AND-OR decomposition (Plan 190) already produces weighted branch paths at inference time. The FOL-LNN paper (arXiv 2110.10963) proves that AND-OR-NOT gates with learnable weights can both solve RL tasks 10-100× faster AND extract interpretable first-order logic rules. Our DDTree IS the LNN — but constructed on-the-fly from marginals, not pre-trained. We currently throw away the logical structure after each inference. Four fusions extract value from this structure.

**Current:** DDTree explores AND-OR paths with weights → generates tokens → discards path structure.
**Proposed:** Extract FOL constraints from prompts (no LLM), extract logical rules from top-K DDTree paths, reward-weight branches via compilation success, produce interpretable decision traces.
**Result:** Self-improving inference cycle: rules → episodes → better future inference → better rules. All modelless.

---

## Architecture

```
Prompt Input
    │
    ├── T1: FolConstraintExtractor (NEW, feature: fol_constraints)
    │   ├── Keyword extraction: "async function" → ⟨keyword=async⟩ ∧ ⟨keyword=fn⟩
    │   ├── Type constraints: "returns Result<T,E>" → ⟨return_type=Result⟩
    │   ├── Negation: "no unsafe" → ¬⟨keyword=unsafe⟩
    │   └── Output: Vec<FolConstraint> → FolPruner wraps inner ConstraintPruner
    │
    ├── DDTree AND-OR Exploration (existing, Plan 190)
    │   ├── Marginals + Bandit scores → weighted branch paths
    │   └── FolPruner injects constraints → prunes disallowed branches
    │
    ├── T2: RuleExtractor (NEW, feature: rule_extraction)
    │   ├── After DDTree: extract TOP-K highest-scoring paths as FOL-like rules
    │   ├── Rule deduplication: similar paths → single rule
    │   └── Store extracted rules in EpisodePruner for future constraint injection
    │
    ├── T3: RewardMemPruner (NEW, feature: reward_mem)
    │   ├── Compilation success/failure → reward signal
    │   ├── blake3 hash of (prompt_type, path_pattern) → PatternHasher
    │   ├── Positive reward on compile → boost future similar prompt branches
    │   └── Wire into BanditPruner update loop
    │
    └── T4: DecisionTrace (NEW, feature: decision_trace, opt-in)
        ├── Extract human-readable decision traces from DDTree exploration
        ├── rules_applied + alternatives_rejected + confidence
        └── to_string(vocab) → human-readable audit trail
```

---

## Tasks

### Phase 1: T1 — Prompt→FOL Constraint Extraction

- [x] **T1.1:** Create `src/pruners/fol_pruner.rs` with `FolConstraint` + `FolPruner<P>` structs
  - `FolConstraint { depth_range: (usize, usize), allowed: Vec<usize>, disallowed: Vec<usize>, confidence: f32 }`
  - `FolPruner<P: ConstraintPruner> { inner: P, constraints: Vec<FolConstraint> }`
  - Feature-gated behind `#[cfg(feature = "fol_constraints")]`

- [x] **T1.2:** Implement `extract_fol_constraints(prompt: &str, vocab: &[String]) -> Vec<FolConstraint>`
  - Keyword extraction: "async function" → `⟨keyword=async⟩ ∧ ⟨keyword=fn⟩`
  - Type constraints: "returns Result<T,E>" → `⟨return_type=Result⟩`
  - Negation: "no unsafe" → `¬⟨keyword=unsafe⟩`

- [x] **T1.3:** Build static keyword→token index lookup table
  - ~100 Rust keywords/patterns pre-computed at compile time
  - `const RUST_KEYWORD_TABLE: &[(&str, &[&str])]`
  - Pre-allocated, zero alloc at inference time

- [x] **T1.4:** Implement `ConstraintPruner` for `FolPruner<P>`
  - `relevance()` applies constraints then delegates to inner
  - Zero cost on miss path (empty constraints → inner only)

- [x] **T1.5:** Unit tests for constraint extraction (14/14 passing)
  - Test: "async function returning Result" extracts correct keywords
  - Test: "no unsafe" produces negation constraint
  - Test: empty prompt → zero constraints (miss path)

- [x] **T1.6:** Benchmark: constraint extraction overhead (zero alloc on hot path)
  - Target: <1μs for typical prompt
  - Zero alloc on hot path

### Phase 2: T2 — DDTree Path→Logical Rule Extraction

- [x] **T2.1:** Create `src/pruners/rule_extractor.rs` with `RuleExtractor` + `ExtractedRule`
  - `ExtractedRule { conditions: Vec<(usize, usize)>, action: (usize, usize), score: f32, support: u32 }`
  - `RuleExtractor { top_k: usize, min_score: f32 }`
  - Feature-gated behind `#[cfg(feature = "rule_extraction")]` — depends on `and_or_dtree`

- [x] **T2.2:** Implement path extraction from DDTree nodes
  - `RuleExtractor::extract(tree: &[TreeNode], top_k: usize) -> Vec<ExtractedRule>`
  - Walk DDTree, collect paths with score ≥ `min_score`, return TOP-K

- [x] **T2.3:** Implement rule deduplication
  - Similar paths (Hamming distance ≤ threshold) → single merged rule
  - Increment `support` count on merge

- [x] **T2.4:** Store extracted rules in EpisodePruner
  - Wire into post-DDTree pipeline for future constraint injection
  - Rules become episodes → self-improving cycle

- [x] **T2.5:** Unit tests for rule extraction (9/9 passing)
  - Test: top-K extraction from known tree structure
  - Test: deduplication merges similar paths
  - Test: min_score threshold filters low-quality rules

- [x] **T2.6:** Create `examples/rule_extraction_demo.rs`
  - Shows DDTree paths → extracted rules in human-readable format

### Phase 3: T3 — Reward-Weighted Branch Memorization

- [x] **T3.1:** Create `src/pruners/reward_mem_pruner.rs` with `RewardMemPruner`
  - `RewardMemPruner<P: ConstraintPruner> { inner, rewarded_patterns, current_prompt_type }`
  - Feature-gated behind `#[cfg(feature = "reward_mem")]` — depends on `egcs`, `bandit`

- [x] **T3.2:** Implement `PatternHasher`
  - blake3 hash of (prompt_type, path_pattern) — per `.agents` rules
  - `PatternHasher::hash(prompt_type: &str, path: &[usize]) -> [u8; 32]`
  - Zero-alloc: stack-only blake3

- [x] **T3.3:** Wire compilation success → bandit reward update
  - `CompileOutcome::Success` → reward = 1.0
  - `CompileOutcome::Error(_)` → reward = -0.5
  - EMA update: `new_score = old_score + lr * (reward - old_score)` with lr=0.1

- [x] **T3.4:** On inference: look up rewarded patterns → `get_boost()`
  - Pattern lookup by prompt_type hash → retrieve rewarded paths
  - Returns 0.0 on miss (zero overhead)

- [x] **T3.5:** Unit tests for reward propagation (12/12 passing)
  - Test: compilation success propagates positive reward
  - Test: compilation error propagates negative reward
  - Test: pattern lookup retrieves rewarded branches
  - Test: miss path (no reward history) → zero overhead
  - Test: blake3 hash deterministic
  - Test: blake3 hash differs for different paths
  - Test: blake3 hash differs for different prompt types
  - Test: EMA convergence
  - Test: ConstraintPruner delegation
  - Test: batch_is_valid delegation
  - Test: reset clears state
  - Test: prompt type isolation

- [x] **T3.6:** Integration test: before/after on known hard problems
  - Measure accuracy gain after N compilation cycles

### Phase 4: T4 — Interpretable Decision Traces

- [x] **T4.1:** Create `src/pruners/decision_trace.rs` with `DecisionTrace`
  - `DecisionTrace { rules_applied: Vec<ExtractedRule>, alternatives_rejected: Vec<ExtractedRule>, confidence: f32 }`
  - Feature-gated behind `#[cfg(feature = "decision_trace")]` — depends on `rule_extraction`

- [x] **T4.2:** Implement `to_string(&self, vocab: &[String]) -> String`
  - Human-readable format:
    ```
    Decision trace: "Chose `match` over `if-else` because:
      ⟨token=enum⟩ ∧ ⟨branch_count≥3⟩ ∧ ¬⟨simple_comparison⟩ → match"
    ```
  - Alternative rejection explanation with score comparison

- [x] **T4.3:** Example showing trace output
  - `decision_trace_demo.rs` with sample DDTree → trace output

- [x] **T4.4:** Make opt-in (not default-on)
  - Debug/audit feature only — adds extraction cost, no accuracy/perf benefit
  - Document as transparency/audit tool

### Phase 5: T5 — GOAT Proof + Feature Gates

- [x] **T5.1:** Add feature gates to `Cargo.toml`
  ```toml
  fol_constraints = []                          # T1: standalone
  rule_extraction = ["and_or_dtree"]            # T2: depends on DDTree
  reward_mem = ["egcs", "bandit_pruner"]        # T3: depends on EGCS + BanditPruner
  decision_trace = ["rule_extraction"]          # T4: depends on rule extraction
  ```

- [ ] **T5.2:** GOAT test: FOL constraint extraction accuracy
  - Target: ≥80% correct constraint extraction on Rust prompts
  - Test corpus: 50+ Rust prompts with known expected constraints

- [ ] **T5.3:** GOAT test: Rule extraction support threshold
  - Target: ≥30% pattern reuse (rules hit ≥30% of future similar prompts)

- [ ] **T5.4:** GOAT test: Reward propagation improves future inference
  - Target: ≥10% accuracy gain after reward warm-up (N=50 compilations)

- [x] **T5.5:** GOAT test: Zero overhead on miss path
  - No constraints → performance identical to baseline (inner pruner only)
  - Benchmark: latency delta < 0.5% on unconstrained prompts

- [ ] **T5.6:** Default-on if GOAT passes with no perf regression
  - Default: `fol_constraints` + `reward_mem`
  - Opt-in: `decision_trace` (debug/audit, not default)
  - Conditional: `rule_extraction` (default-on if support threshold ≥30%)

- [x] **T5.7:** Wire modules into `src/pruners/mod.rs`
  - Add `pub mod fol_pruner;` behind `#[cfg(feature = "fol_constraints")]`
  - Add `pub mod rule_extractor;` behind `#[cfg(feature = "rule_extraction")]`
  - Add `pub mod reward_mem_pruner;` behind `#[cfg(feature = "reward_mem")]`
  - Add `pub mod decision_trace;` behind `#[cfg(feature = "decision_trace")]`
  - Re-export key types

### Phase 6: T6 — Documentation + Examples

- [x] **T6.1:** Create `examples/fol_constraint_demo.rs`
  - Prompt → FOL constraint extraction → FolPruner → DDTree search

- [x] **T6.2:** Create `examples/rule_extraction_demo.rs`
  - DDTree paths → extracted rules with human-readable output

- [ ] **T6.3:** Update README with FOL-LNN section
  - Architecture diagram of the self-improving cycle
  - Feature gate documentation
  - Performance characteristics

- [x] **T6.4:** Create benchmark report `.benchmarks/209_fol_lnn_goat.md`
  - G1: Constraint extraction accuracy ≥80%
  - G2: Rule support threshold ≥30%
  - G3: Reward accuracy gain ≥10%
  - G4: Miss path overhead <0.5%
  - G5: Constraint extraction latency <1μs
  - G6: All tests pass with/without each feature gate

### Phase 7: Plan Update

- [ ] **T7.1:** Update plan status to Done after all GOAT gates pass

---

## Feature Gate Configuration

```toml
[features]
# T1: FOL constraint extraction from prompts — standalone, zero deps
fol_constraints = []

# T2: Logical rule extraction from DDTree paths — needs AND-OR tree
rule_extraction = ["and_or_dtree"]

# T3: Reward-weighted branch memorization — needs EGCS + BanditPruner
reward_mem = ["egcs", "bandit_pruner"]

# T4: Interpretable decision traces — needs rule extraction
decision_trace = ["rule_extraction"]

# Convenience: all FOL-LNN features
fol_lnn = ["fol_constraints", "rule_extraction", "reward_mem"]

# Default-on after GOAT proof (TBD)
# default = ["fol_constraints", "reward_mem"]
```

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `src/pruners/fol_pruner.rs` | NEW | 1 |
| `src/pruners/rule_extractor.rs` | NEW | 2 |
| `src/pruners/reward_mem_pruner.rs` | NEW | 3 |
| `src/pruners/decision_trace.rs` | NEW | 4 |
| `src/pruners/mod.rs` | EXTEND | 5 |
| `Cargo.toml` | EXTEND | 5 |
| `examples/fol_constraint_demo.rs` | NEW | 6 |
| `examples/rule_extraction_demo.rs` | NEW | 6 |
| `.benchmarks/209_fol_lnn_goat.md` | NEW | 6 |
| `.plans/209_fol_logical_rule_inference.md` | THIS | 7 |

## SOLID Compliance

- **S:** Each file handles one concern: constraint extraction, rule extraction, reward memorization, decision traces.
- **O:** All new types implement existing traits (`ConstraintPruner`, `ScreeningPruner`) — extend without modifying.
- **L:** `FolPruner<P>` wraps any `ConstraintPruner`, `RewardMemPruner<P, L>` wraps any `ConstraintPruner + EpisodeLookup`.
- **I:** Thin public APIs: `extract_fol_constraints()`, `RuleExtractor::extract()`, `reward_path()`, `DecisionTrace::to_string()`.
- **D:** Depend on traits (`ConstraintPruner`, `EpisodeLookup`), not concrete implementations.

## Expected Performance

| Metric | Without FOL-LNN | With FOL-LNN | Delta |
|--------|-----------------|--------------|-------|
| Accuracy on constrained prompts | Baseline | +2-5× | FOL constraint pruning |
| Future inference accuracy | Baseline | +10-20% | Reward-weighted branches |
| Constraint extraction latency | 0 | <1μs | Static keyword table |
| Rule extraction latency | 0 | Post-DDTree (not hot path) | After exploration |
| Miss path overhead | 0 | 0 | Inner pruner only |
| Memory per pattern | 0 | ~32 bytes (blake3 hash + score) | PatternHasher |

## Key Constraints

- **Modelless only** — no LLM training, all inference-time
- **Feature-gated** — each fusion independently gateable, zero-cost when off
- **Zero-cost miss path** — empty constraints / no reward history → identical to baseline
- **Pre-allocated lookup tables** — keyword table is `const`, zero alloc hot path (per optimization.md)
- **blake3 for hashing** — per `.agents` rules (not SHA1/SHA256)
- **Per Verdict 003:** generic AND-OR/FOL is MIT engine, game-specific is riir-ai private
- **Per research:** DDTree IS the LNN — no training needed, constructed from marginals at inference time

## Commercial Alignment

| Component | License | Rationale |
|---|---|---|
| T1: FolConstraintExtractor | MIT | Regex + keyword table — too simple to hide |
| T2: RuleExtractor | MIT | Generic logical rule extraction — engine fuel |
| T3: RewardMemPruner | MIT | Extends EGCS — engine feature |
| T4: DecisionTrace | **SaaS surface** | "Explain why this translation was chosen" — premium |
| Rust Concept Graph | MIT | Engine fuel — grows from community |
| Episode DB of extracted rules | **Secret B** | Proprietary data flywheel |

## Cross-Repo Alignment (riir-ai ↔ katgpt-rs)

| riir-ai Plan | Relationship | Notes |
|---|---|---|
| **239** FOL Game Rule Extraction | Mirror — training-side rule extraction | 239 T1 extracts FOL from LoRA weights (training fuel); 209 T1 extracts FOL from prompts (inference engine). Different lifecycle stages, same logical representation. |
| **240** EQL Symbolic LoRA | Complementary — quantitative rules | 240 produces EQL expressions (numeric); 209 produces FOL constraints (logical). Both feed 211's mode router. |
| **211** Three-Mode Router | Consumer — 209 T4 DecisionTrace | **DRI decision:** 209 T4 (`DecisionTrace`) is the canonical home for "DDTree path → human-readable rule". 211 F5 re-exports via `decision_trace` feature. Do not duplicate. |

### Execution Order

| Phase | Plan | Rationale |
|-------|------|-----------|
| 1 | 210 F4 (Reward Calibration) | Formalizes existing AbsorbCompress — zero risk |
| 2 | 212 (Collapse-Aware Thinking) | Independent, proven by S2F paper |
| 3 | **209** (this plan) | Foundation for 211's mode router |
| 4 | 210 F1-F3 (Distillation + Explanation) | Core novelty, needs 209 for grounding |
| 5 | 211 (Three-Mode Router) | Consumes 209 + 210 outputs |

---

## TL;DR

Plan 209 = **FOL Logical Rule Inference** — four modelless fusions from FOL-LNN paper (arXiv 2110.10963). T1: Extract FOL constraints from prompts via regex+keyword table (no LLM, <1μs). T2: Extract TOP-K logical rules from DDTree paths → self-improving cycle. T3: Reward-weight DDTree branches via compilation success → better future inference. T4: Interpretable decision traces (opt-in debug/audit). All feature-gated, zero-cost miss path, pre-allocated lookup tables. DDTree IS the LNN — no training needed. MIT engine, Secret B data flywheel. ~600-800 lines new code across 4 files.
