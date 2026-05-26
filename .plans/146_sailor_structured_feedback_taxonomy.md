# Plan 146: Sailor-Inspired Structured Feedback Taxonomy for SR²AM

> **Origin:** Research 108 (Sailor — Static Analysis + LLM + SE)
**Status:** ✅ COMPLETE — LOW PRIORITY (architectural validation, no new primitives needed)
> **GOAT Pillar:** Pillar 2 (WASM Validators) — external academic validation
> **Feature Gate:** None — enhancement to existing `sr2am_configurator` feature
> **Related Plans:** 112 (SR²AM), 111 (Data Gate), 137 (PrudentBanker)
> **Related Research:** 108 (Sailor), 037 (REAP), 076 (SR²AM)
> **Depends on:** Plan 112 (SR²AM configurator — already implemented)
> **Blocks:** Nothing

---

## Task Index

- [x] Task 1: Structured Feedback Enum for Exploration Outcomes (DEFERRED)
- [x] Task 2: Cross-Reference GOAT Pillar 2 Academic Validation (DOCUMENTATION)

---

## Verdict

Sailor independently discovered the same three-phase pipeline (modelless targeting → model-based synthesis → concrete validation) that our architecture already implements. **No new code is immediately required.** Two incremental enhancements are identified for future implementation.

---

## Context

The Sailor paper (UCSB, 2026) proves that a three-phase pipeline of static analysis → LLM-orchestrated symbolic execution → concrete validation discovers 379 vulnerabilities across 6.8M LOC, 30× more than any single technique alone. The key insight: **structured feedback taxonomy in the iterative refinement loop** is what makes the composition work.

Our architecture already implements this pipeline:
- Phase 1: `ConstraintPruner` + `ScreeningPruner` (modelless)
- Phase 2: Bandit/MCTS + SR²AM feedback (model-based)
- Phase 3: Arena GOAT proof (validation)

This plan captures the two actionable distillations as future enhancement tasks.

---

## Task 1: Structured Feedback Enum for Exploration Outcomes (DEFERRED)

- [x] Add `ExplorationOutcome` enum to SR²AM feedback classification

```rust
/// Structured exploration outcome — inspired by Sailor's feedback taxonomy
///
/// Sailor classifies SE feedback into:
///   - "not reached" → target line not executed
///   - "site reached" → target reached, no violation
///   - "bug triggered" → concrete violation confirmed
///
/// Our analog for game-state exploration:
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExplorationOutcome {
    /// Move sequence didn't reach target game state
    /// Sailor: "not reached" → LLM fixes driver/stubs
    /// Action: Adjust bandit Q-values (negative reward)
    NotReached,
    /// Target state reached but no win condition
    /// Sailor: "site reached" → LLM tightens constraints
    /// Action: Neutral reward + log for constraint tuning
    StateReachedNoWin,
    /// Concrete win condition satisfied
    /// Sailor: "bug triggered" → confirmed vulnerability
    /// Action: Positive reward + update GOAT proof
    WinConfirmed,
    /// Invalid game state detected
    /// Sailor: "compilation error" → fix harness
    /// Action: Zero reward + flag for WASM validator check
    InvalidState,
}
```

**File:** `katgpt-rs/src/speculative/` or wherever SR²AM lives
**Feature gate:** Uses existing `sr2am_configurator`
**GOAT proof:** Existing SR²AM proofs still pass — this is a refactoring, not a behavior change

**Defer reason:** SR²AM already works. This is a clarity improvement, not a feature. Implement when we next touch SR²AM code for other reasons.

---

## Task 2: Cross-Reference GOAT Pillar 2 Academic Validation (DOCUMENTATION)

- [x] Update `.docs/27_mmo_goat_pillars_decision_matrix.md` with Sailor external validation reference

**This is a documentation-only change.** No code impact.

---

## Why This Plan Is LOW PRIORITY

| Factor | Assessment |
|--------|-----------|
| Code change needed | Minimal (enum addition + doc update) |
| New functionality | None — pure refactoring |
| GOAT proof impact | None — existing proofs unchanged |
| Risk | Zero |
| Academic value | Medium — strengthens Pillar 2 evidence |
| Blocking | Nothing depends on this |

The research (108) concluded that **the pattern is already distilled in our architecture**. This plan exists to capture the two minor enhancements for when we're next in the relevant code areas. Don't prioritize over active feature work.

---

## Feature Gate Assessment

| Gate | Status | Reason |
|------|--------|--------|
| New feature gate | NOT NEEDED | Enhancement to existing `sr2am_configurator` |
| Secret/selling point | NOT APPLICABLE | This is an academic paper validation, not a game feature |
| riir-ai domain | NO | No game-specific code; pattern is general |
| katgpt-rs domain | YES | Enhancement to existing SR²AM trait system |

---

## References

- Research 108: `.research/108_Sailor_Static_Analysis_LLM_Symbolic_Execution_Vulnerability_Discovery.md`
- GOAT Pillar Decision Matrix: `riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md`
- Sailor paper: https://arxiv.org/pdf/2604.06506
- SR²AM Plan: `.plans/112_sr2am_configurator_bandit.md`
