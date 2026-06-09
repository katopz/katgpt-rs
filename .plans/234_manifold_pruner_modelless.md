# Plan 234: ManifoldE Point-to-Manifold Pruner — Modelless Inference-Time Geometry

> **Source:** Research 207 — ManifoldE Point-to-Manifold Principle
> **Date:** 2026-06
> **Status:** 📋 Plan
> **Feature Gate:** `manifold_pruner` (opt-in, promote to default if GOAT)
> **Related:** Plan 207 (Lodestar Completion Distance Pruning), Plan 201 (Rosetta Pruner), Plan 232 (DynamicRankPruner)

---

## Background

ManifoldE embeds knowledge graph triples as points near a manifold (sphere/hyperplane) rather than point-to-point. The key insight: validity is a **distance to a geometric surface**, not a binary inside/outside check.

**In katgpt-rs:**
- `ConstraintPruner::is_valid()` returns `bool` — binary accept/reject
- `ScreeningPruner::relevance()` returns `f32` — already scalar, but linear
- DDTree expansion uses boolean pruning — boundary tokens (barely valid, barely invalid) are either fully accepted or fully rejected
- BFCP regions use fixed thresholds — no adaptation based on access frequency

**The gap:** No pruner knows *how close* a token is to the constraint boundary. ManifoldE's geometry gives us soft validity scores, half-space intersection, and kernel-tricked relevance — all modelless, all at inference time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ConstraintPruner Trait                     │
│                                                              │
│  is_valid()           → bool        (existing, unchanged)   │
│  manifold_score()     → f32        (NEW, default=binary)    │
│  constraint_vector()  → Option<(&[f32], f32)>  (NEW, None)  │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ HyperplanePruner │  │ ManifoldPruner   │                 │
│  │ (half-space ∩)   │  │ (soft sigmoid)   │                 │
│  │ ≥2 pruners       │  │ single wrapper   │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │          ScreeningPruner Trait           │               │
│  │                                          │               │
│  │  KernelScreeningPruner                   │               │
│  │  ┌─────────┬───────────┬──────────────┐ │               │
│  │  │ Linear  │ Gaussian  │ Polynomial   │ │               │
│  │  │ dot(q,c)│ exp(-d/σ) │ (dot+c)^deg │ │               │
│  │  └─────────┴───────────┴──────────────┘ │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │          BFCP Region Radius              │               │
│  │  D_r = base_radius * sigmoid(freq/scale) │               │
│  │  hot → wide manifold, cold → tight       │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Trait extensions with defaults:** `manifold_score()` defaults to binary `{0.0, 1.0}` via `is_valid()`. Zero cost if not overridden. Backward compatible.
2. **Half-space intersection:** `HyperplanePruner` composes constraint vectors from multiple pruners. Valid = geometric intersection of all half-spaces.
3. **Sigmoid not softmax:** Per user rules, soft scoring uses sigmoid projection — no normalization dependency across candidates.
4. **Kernel trick on `ScreeningPruner`:** Lift relevance computation to implicit feature space. SIMD-accelerated Gaussian.
5. **BFCP radius adaptation:** LFU frequency → sigmoid → manifold radius. Hot regions expand, cold regions contract. Wired into existing `bfcf_lfu_shard` feature.

### File Layout

| File | Purpose | ~LOC |
|------|---------|------|
| `crates/katgpt-core/src/traits.rs` | Trait extensions (`manifold_score`, `constraint_vector`) | ~30 |
| `src/pruners/hyperplane_pruner.rs` | `HyperplanePruner` half-space intersection | ~200 |
| `src/pruners/manifold_pruner.rs` | `ManifoldPruner` soft sigmoid wrapper | ~150 |
| `src/pruners/kernel_scoring.rs` | `KernelKind` enum + SIMD kernel functions | ~200 |
| `src/pruners/kernel_screening_pruner.rs` | `KernelScreeningPruner<P>` wrapper | ~120 |
| `src/pruners/mod.rs` | Module glue (feature-gated) | ~8 |
| `tests/goat_234_manifold_pruner.rs` | GOAT proof benchmark | ~150 |

---

## Tasks

### Phase 1: Trait Extensions (Backward Compatible)

- [ ] Add `manifold_score(&self, depth: usize, token_idx: usize, prefix: &[usize]) -> f32` default method to `ConstraintPruner` trait in `crates/katgpt-core/src/traits.rs`. Default: `if self.is_valid(depth, token_idx, prefix) { 1.0 } else { 0.0 }`. Zero cost if not overridden.
- [ ] Add `constraint_vector(&self, depth: usize, prefix: &[usize]) -> Option<(&[f32], f32)>` optional method to `ConstraintPruner` trait. Returns `(normal_vector, threshold)` for half-space constraint. Default: `None` (fall back to `is_valid()`).
- [ ] Add `KernelKind` enum: `Linear`, `Gaussian { sigma: f32 }`, `Polynomial { degree: f32, c: f32 }`.
- [ ] Test: default `manifold_score()` returns 1.0 for valid, 0.0 for invalid tokens (backward compat)
- [ ] Test: default `constraint_vector()` returns `None` (backward compat)

### Phase 2: HyperplanePruner — Geometric Constraint Intersection (P0)

- [ ] Create `src/pruners/hyperplane_pruner.rs` behind `#[cfg(feature = "manifold_pruner")]`
- [ ] Implement `HyperplanePruner` struct that takes a slice of `&dyn ConstraintPruner` and composes their constraint vectors via half-space intersection
- [ ] Half-space intersection: for each pruner returning `Some((normal, threshold))`, compute `normal · token_embedding >= threshold`. Valid = intersection of all half-spaces.
- [ ] For pruners returning `None`, fall back to `is_valid()` boolean check
- [ ] `manifold_score()` implementation: `product of sigmoid(-distance_to_boundary / temperature)` for each constraint
- [ ] SIMD batch: `batch_manifold_score()` processes all candidates in one pass
- [ ] Test: single pruner with constraint vector → HyperplanePruner matches its constraint
- [ ] Test: two pruners with non-parallel normals → intersection is stricter than either alone
- [ ] Test: two pruners with parallel normals → intersection = stricter of the two
- [ ] Test: pruner returning `None` → falls back to boolean AND correctly

### Phase 3: ManifoldPruner — Soft Validity Scoring (P1)

- [ ] Create `src/pruners/manifold_pruner.rs` behind `#[cfg(feature = "manifold_pruner")]`
- [ ] Implement `ManifoldPruner` wrapper that converts any `ConstraintPruner` into a soft scorer via temperature-controlled sigmoid
- [ ] `manifold_score()` returns `sigmoid(-distance / temperature)` where distance is derived from `is_valid()` boundary proximity
- [ ] For pruners with `constraint_vector()`: distance = `|normal · token - threshold|`
- [ ] For pruners without: distance = `0.0` if valid, `∞` if invalid → sigmoid → `{1.0, 0.0}` fallback
- [ ] Wire into DDTree: use `manifold_score` instead of `is_valid` when feature enabled, expand children with score > 0.5
- [ ] Test: ManifoldPruner with high temperature → nearly uniform scores (permissive)
- [ ] Test: ManifoldPruner with low temperature → near-binary scores (conservative)
- [ ] Test: DDTree expansion with ManifoldPruner captures boundary tokens that binary pruner misses

### Phase 4: Kernel-Tricked Relevance (P1)

- [ ] Create `src/pruners/kernel_scoring.rs` with kernel scoring functions
- [ ] `kernel_score(query: &[f32], candidate: &[f32], kind: KernelKind) -> f32`:
  - Linear: `dot(query, candidate)`
  - Gaussian: `exp(-||q-c||²/σ²)` — SIMD-accelerated (chunked f32, 4 or 8 per iteration)
  - Polynomial: `(dot(q,c) + c)^degree`
- [ ] Implement `KernelScreeningPruner<P>` that wraps any `ScreeningPruner` and applies kernel transformation
- [ ] Test: Gaussian kernel returns 1.0 for identical vectors, ~0.0 for distant
- [ ] Test: Polynomial kernel preserves sign of dot product
- [ ] Benchmark: `kernel_score` SIMD vs scalar on 256-dim vectors

### Phase 5: BFCP Region Radius Adaptation (P2)

- [ ] Extend BFCP region scoring to use LFU frequency as manifold radius `D_r`
- [ ] `region_radius(freq: f32) -> f32`: `D_r = base_radius * sigmoid(freq / freq_scale)`
- [ ] Hot regions: high freq → large `D_r` → wide manifold → more candidates pass
- [ ] Cold regions: low freq → small `D_r` → tight manifold → fewer candidates pass
- [ ] Wire into existing `bfcf_lfu_shard` feature — no new feature gate
- [ ] Test: hot region radius > cold region radius for same `base_radius`
- [ ] Test: `sigmoid(0) = 0.5` → default radius at zero frequency

### Phase 6: GOAT Proof + Benchmark

- [ ] Create benchmark: `cargo test --features manifold_pruner --release -- --nocapture`
- [ ] Benchmark 1: HyperplanePruner vs boolean AND composition — measure valid candidate count and downstream DDTree acceptance rate
- [ ] Benchmark 2: KernelScreeningPruner (Gaussian) vs linear ScreeningPruner — measure ranking quality (Kendall τ)
- [ ] Benchmark 3: BFCP region radius adaptation — measure throughput vs fixed threshold
- [ ] GOAT gate: promote `manifold_pruner` to default if ≥3% improvement in DDTree acceptance rate without throughput regression
- [ ] If GOAT fails: document negative result, keep as opt-in, demote in README

### Phase 7: Documentation + Integration

- [ ] Add `manifold_pruner` to feature flags table in `README.md`
- [ ] Add to `.docs/01_overview.md` module structure
- [ ] Add to `.docs/07_adaptation.md` as technique
- [ ] Add to `.docs/15_paper_feature_comparison.md`
- [ ] Add example: before/after DDTree expansion showing boundary token recovery

---

## GOAT Gate

### Acceptance Criteria

| Gate | Condition | Measurement |
|------|-----------|-------------|
| **G1: Acceptance gain** | HyperplanePruner ≥ 3% higher DDTree acceptance rate vs boolean AND | `tests/goat_234_manifold_pruner.rs` |
| **G2: Kernel quality** | Gaussian kernel Kendall τ ≥ 0.05 improvement over linear on same candidate set | Benchmark 2 |
| **G3: Zero regression** | No throughput regression when `manifold_pruner` enabled vs disabled | Benchmark 1+3 |

### Failure Outcome

If G1 fails (no ≥3% gain):
- Feature stays default-off (`manifold_pruner` remains opt-in)
- Trait extensions (`manifold_score`, `constraint_vector`) stay — zero cost, useful for future work
- Document negative result in `.docs/` and plan status
- Do NOT promote to default

### Dependencies

- Requires `bfcf_lfu_shard` feature for Phase 5
- Requires `ddtree` feature for DDTree integration (Phase 3)
- No new crate dependencies — SIMD via chunked f32 loops, not external crate

---

## Cross-Repo Alignment

| Repo | What | Why |
|------|------|-----|
| `riir-ai` | `KernelKind` enum + kernel scoring functions | Reusable math for LoRA training manifold loss (Research 092) |
| `seal-online-remaster` | Soft validity scoring for NPC behavior | Replace binary action validation with manifold scoring |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| SIMD perf regression from soft scoring | Keep binary fast-path as default; only compute `manifold_score` when feature enabled |
| `constraint_vector()` allocation in hot loop | Return `&[f32]` slice from pre-allocated per-pruner buffer, not `Vec` |
| HyperplanePruner composition blowup with many pruners | Cap intersection at 8 pruners; fall back to boolean AND for remaining |
| Temperature/hyperparameter sensitivity | Make all params configurable in `domains.toml` (fuel layer) |

---

## SOLID/DRY Compliance

- **S:** Each pruner has single responsibility — `HyperplanePruner` composes, `ManifoldPruner` softens, `KernelScreeningPruner` lifts
- **O:** Open for extension — new kernel kinds plug into `KernelKind` enum without touching pruner logic
- **L:** All wrappers implement the traits they wrap — substitutable for inner pruner
- **I:** Trait extensions have defaults — existing implementors need zero changes
- **D:** HyperplanePruner depends on `ConstraintPruner` trait, not concrete pruner types
- **DRY:** Kernel scoring functions shared via `kernel_scoring.rs`, not duplicated per pruner

---

## TL;DR

ManifoldE's point-to-manifold principle applied to three core traits: `ConstraintPruner` gets soft validity scoring + geometric half-space intersection, `ScreeningPruner` gets kernel-tricked relevance, BFCP gets frequency-adapted manifold radius. All modelless, feature-gated as `manifold_pruner`. P0 = `HyperplanePruner` (highest value — strengthens the commercial moat). GOAT-gate: promote to default if ≥3% DDTree acceptance gain without throughput regression. Trait extensions are zero-cost defaults — backward compatible, no breakage.
