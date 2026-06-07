# Plan 201: Rosetta Pruners — Universal Cross-Domain Meta-Pruner

**Date**: 2026-06-07
**Status**: ✅ Implemented
**Research**: `.research/178_Rosetta_Neurons_Cross_Model_Alignment.md` (Section 2.2)
**GOAT Rank**: #2 (highest commercial value, medium effort)

---

## Context

Rosetta Neurons shows that universal concepts emerge across different systems processing the same data. We have multiple `ConstraintPruner` and `ScreeningPruner` implementations (SynPruner, game validators, WASM validators). Mining cross-pruner agreement reveals universal constraint concepts — tokens/positions where ALL pruners agree → high confidence. Positions where pruners disagree → uncertain, needs more exploration.

This creates a **meta-pruner** that is stronger than any individual pruner: universal concepts give certainty, disagreements signal where to invest compute.

---

## Architecture

```rust
/// Meta-pruner built from cross-pruner Rosetta alignment.
///
/// Mines universal constraint concepts from multiple pruners:
/// - Universal: ALL pruners agree → fast path (skip verification)
/// - Contested: pruners disagree → invest more compute (deeper DDTree)
pub struct RosettaPruner<P: ConstraintPruner> {
    /// Underlying pruners
    pruners: Vec<P>,
    /// Pre-computed universal concept map: (depth, token_idx) → agreement ratio
    concept_map: papaya::HashMap<(usize, usize), f32>,
    /// Universal concepts: positions where agreement > threshold
    universal_concepts: Vec<ConstraintConcept>,
    /// Number of pruners
    n_pruners: usize,
    /// Agreement threshold for "universal" (default: 0.9)
    threshold: f32,
}

pub struct ConstraintConcept {
    id: usize,
    depth: usize,
    tokens: Vec<usize>,
    /// Fraction of pruners that agree on validity
    agreement_ratio: f32,
}

impl<P: ConstraintPruner> ConstraintPruner for RosettaPruner<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Fast path: check pre-computed concept map
        if let Some(&agreement) = self.concept_map.pin().get(&(depth, token_idx)) {
            if agreement > self.threshold {
                return true; // Universal concept — all pruners agree
            }
        }
        // Slow path: query all pruners, majority vote
        let valid_count = self.pruners.iter()
            .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
            .count();
        valid_count > self.n_pruners / 2
    }
}
```

---

## Tasks

- [ ] Add `ConstraintConcept`, `RosettaPruner<P>` structs to `katgpt-rs/src/pruners/`
- [ ] Implement `RosettaPruner::mine_concepts()` — run probe inputs through all pruners, compute agreement
- [ ] Implement `ConstraintPruner` for `RosettaPruner` with fast-path concept map lookup
- [ ] Implement `ScreeningPruner` for `RosettaPruner` with agreement-weighted relevance
- [ ] Add `RosettaPruner` to `build_dd_tree_pruned` integration path
- [ ] Add feature flag `rosetta_pruner` (opt-in initially, default-on after GOAT proof)
- [ ] Write test: verify RosettaPruner matches majority vote, fast-path hits for universal concepts
- [ ] Write benchmark: measure DDTree build time with vs without RosettaPruner
- [ ] Write example: Sudoku with RosettaPruner combining row/col/box constraints
- [ ] GOAT gate: measure DDTree build time reduction. If ≥ 20% → default-on.
- [ ] Update README feature flags section

---

## Commercial Play

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:
- `RosettaPruner` composition is open (MIT) — it's the engine
- Pre-computed concept maps from specific game domains are fuel (SaaS)
- More pruners you mine → better universal concepts → better the engine
- Competitors can implement the composition, but without mined concept maps, they get suboptimal pruning

---

## Expected Performance

- **Fast path**: O(1) papaya lock-free HashMap lookup for universal concepts
- **Slow path**: O(N_pruners) majority vote for contested positions (rare after mining)
- **DDTree reduction**: 20-40% fewer branches explored (universal concepts pruned early)
- **Mining cost**: O(N_inputs × N_pruners × N_tokens) — offline, one-time per domain

---

## TL;DR

**Rosetta Pruners** mine universal constraint concepts across all pruners → meta-pruner with O(1) fast path for universal concepts. Expect 20-40% DDTree reduction. Feature-gated `rosetta_pruner`, default-on after GOAT proof. Commercial: concept maps are fuel.
