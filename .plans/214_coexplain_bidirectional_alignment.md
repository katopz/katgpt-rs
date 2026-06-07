# Plan 214: CoExplain Bidirectional Alignment — Modelless Pruner Evolution

**Date:** 2026-06
**Status:** Plan
**Depends On:** Research 189 (Editable XAI CoExplain Bidirectional Alignment)
**Feature Flags:** `coexplain_pruner`, `ted_lite`, `coexplain_riir`

---

## Goal

Implement CoExplain's Read/Write/Enhance cycle for the modelless inference pipeline, enabling self-refining pruners and the Curator marketplace without lora.bin.

## Tasks

### Phase 1: TED-Lite Divergence Metric (Enabling Infrastructure)
- [x] Add `PrunerDivergence` struct to `katgpt-rs/src/pruners/ted_lite.rs`
  - `threshold_divergence: f32` — Σ |τ_current - τ_original| / N
  - `topology_divergence: f32` — Hamming distance on branch existence vectors
  - `lambda_t: f32` — developer-configurable divergence clamp (default: 0.1)
- [x] Implement `PrunerDivergence::compute()` — O(k) per pruner
- [x] Add `PrunerDivergence::clamp_adjustment()` — reject bandit updates that exceed lambda_t
- [x] Add diagnostic log: emit divergence metrics per N tokens (behind `log`)

### Phase 2: Self-Refining Pruner (Extends BanditPruner)
- [x] Add `PrunerAccuracy` tracker to `katgpt-rs/src/pruners/self_refining.rs`
  - Track TP, TN, FP, FN per pruner slot
  - Compute precision, recall, F1 per slot
- [x] Implement threshold mode: adjust SynPruner rejection threshold based on FP/FN ratio
  - `threshold_adjustment = sigmoid(α * (FP_rate - FN_rate))` where α is learning rate
  - Clamp via TED-Lite lambda_t
- [x] Implement topology mode: prune low-value DDTree branches (acceptance rate < ε)
  - Expand high-value branches (acceptance rate > 1-ε)
  - Clamp via TED-Lite lambda_t
- [x] Feature gate: `coexplain_pruner`

### Phase 3: CoEditable ConstraintPruner (Bidirectional)
- [x] Add `EditableConstraintPruner` trait extending `ConstraintPruner` in `katgpt-rs/src/pruners/editable_constraint.rs`
  - `fn edit_threshold(&mut self, slot: usize, new_threshold: f32) -> Result<(), DivergenceError>`
  - `fn edit_topology(&mut self, branch: &[usize], action: TopologyAction) -> Result<(), DivergenceError>`
  - `fn snapshot(&self) -> PrunerSnapshot` — captures golden reference for TED-Lite
  - `fn divergence(&self) -> &PrunerDivergence` — current divergence from snapshot
- [x] Implement `PrunerSnapshot` with blake3 integrity hashing
  - Dynamic threshold adjustment (Tier 0 bracket balancer thresholds)
  - Topology changes (add/remove validation rules)
- [x] Add rule editor backend: accept JSON rule format → compile to ConstraintPruner config
  - Input: `{"rules": [{"attribute": "bracket_depth", "threshold": 0, "action": "reject"}]}`
  - Output: Updated SynPruner config
- [x] Feature gate: `coexplain_pruner`

### Phase 4: Neuro-Symbolic RIIR Feedback Loop (Curator Marketplace)
- [x] Add rule extraction from successful RIIR translations
  - After successful translation, extract DDTree paths that led to compilable output
  - Store as "translation rules" in Episode DB
- [x] Add Curator rule ingestion endpoint (depends on riir-ai Curator API)
  - Accept Curator decision tree rules via MCP/Web UI
  - Compile to WASM ConstraintPruner via riir-validator-sdk
  - Hot-swap into inference pipeline
- [x] Add bandit refinement loop for Curator rules
  - Track translation success rate per Curator rule
  - Adjust thresholds via Phase 2 self-refining mechanism
  - Report accuracy back to Curator (Read mode)
- [x] Add before/after comparison tests
  - Without CoExplain: baseline pruner accuracy on Python→Rust corpus
  - With CoExplain: improved accuracy after bandit refinement
  - With Curator rules: further improvement from domain knowledge injection
- [x] Feature gate: `coexplain_riir` (implies `coexplain_pruner`)

### Phase 5: CPU/GPU Auto-Route Integration
- [x] Add CoExplain workload to inference router (`katgpt-rs/src/inference_router.rs`)
  - Bandit updates → CPU (lightweight, O(1) per token)
  - Rule compilation → async worker (WASM compile is CPU-bound but infrequent)
  - TED-Lite computation → CPU (O(k), negligible)

## Tests/Examples

- [x] `tests/coexplain_goat.rs` — GOAT verification tests (6 tests: G1-G6)
- [x] `examples/coexplain_demo.rs` — full pipeline demo (5 sections)
- [x] `.benchmarks/214_coexplain_goat.md` — GOAT proof report

## Expected Results

| Metric | Before | After (Self-Refining) | After (Curator Rules) |
|--------|--------|----------------------|----------------------|
| Pruner accuracy | ~90% (SynPruner) | ~95% (bandit-refined) | ~98% (domain knowledge) |
| Invalid tokens accepted | 10% | 5% | 2% |
| Compilation success rate | ~60% | ~70% | ~85% |

## Cross-Repo Alignment (riir-ai ↔ katgpt-rs)

| riir-ai Plan | Relationship | Notes |
|---|---|---|
| **239** FOL Game Rules | Upstream — rules feed CoExplain | 239 T1 extracts game rules from trained LoRA. These rules become Curator input for 214's `EditableConstraintPruner`. Pipeline: 239 extracts → Curator reviews → 214 ingests → bandit refines. |
| **244** Rule-Init LoRA + TED | Same concept, different lifecycle | 244 uses TED regularization (`L_topology ≈ ||ΔW - ΔW_rule||²`) during LoRA training. 214 uses `PrunerDivergence` (`threshold_divergence, topology_divergence, lambda_t`) during inference pruning. **Same `lambda_t` parameter name** — good consistency. **Metric difference documented:** 244 uses Frobenius norm (L2, standard for training). 214 uses Hamming distance (cheaper for inference). Both valid. |

### Execution Order

| Phase | Plan | Rationale |
|-------|------|-----------|
| 1 | 210 F4 (Reward Calibration) | Zero risk |
| 2 | 212 (Collapse-Aware Thinking) | Independent |
| 3 | 209 (FOL Inference) | Foundation |
| 4 | 210 F1-F3 (Distillation) | Core novelty |
| 5 | 211 (Three-Mode Router) | Consumer |
| 6 | 213 (BFCF Tree) | Needs stable calibration |
| 7 | **214** (this plan) | Curator marketplace enabler, depends on 239 rules |

---

## Feature Gate Summary

```
[features]
ted_lite = []                           # Phase 1: Divergence metric
coexplain_pruner = ["ted_lite"]         # Phase 2+3: Self-refining + editable
coexplain_riir = ["coexplain_pruner"]   # Phase 4: Full RIIR feedback loop
```

Default: `ted_lite` ON after GOAT proof (no perf hurt).
