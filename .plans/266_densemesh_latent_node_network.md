# Plan: DenseMesh — Latent Node Network for Modelless Inference

**Date:** 2026-06-14
**Research:** katgpt-rs/.research/234_DenseMesh_Latent_Node_Network.md
**Status:** Phase 1–8 complete. **Gate 2 FAILED empirically** — real trained Bomber LoRAs composed via diamond topology produce 0/1000 wins over best single (improvement -0.00%). **Demoted to experimental.** Gate 1 + Gate 3 + Gate 5 PASS. Gate 4 measured (9.27× single-thread vs paper bound 2.5× — requires vertex parallelism). 51 unit tests + 5 profiling tests + 5 gate tests pass.
**Commercial Bound:** Public (katgpt-rs/MIT) — generic framework. Edge LoRA composition recipes stay in riir-ai (R122).

---

## Goal

Implement the `DenseMesh` trait framework + topology engine + EdgeBandit in katgpt-rs, gated behind `dense_mesh`. This is the modelless (inference-time only) distillation of LMNet (arXiv:2505.12741). The framework treats multiple forward passes through the same LLM as nodes in a directed graph, communicating via dense hidden states instead of tokens. Edges are pluggable (identity, LoRA, projection).

Landing: `katgpt-rs/src/dense_mesh/` + feature gate in `Cargo.toml` + `lib.rs`.

---

## Architecture

```
katgpt-rs/src/dense_mesh/
├── mod.rs              # Module root, re-exports
├── types.rs            # DenseHidden, Topology, MeshConfig, ComputeTarget
├── traits.rs           # DenseNode, DenseEdge traits
├── topology.rs         # Layerwise topology, forward_dense orchestration
├── edge_identity.rs    # IdentityEdge (baseline, gate 1)
├── edge_lora.rs        # LoraEdge (wraps existing LoRA adapter as edge)
├── edge_projection.rs  # ProjectionEdge (fixed random projection, no training)
├── handoff.rs          # HiddenHandoff (stripped forward, drafter→verifier)
├── node_transformer.rs # TransformerNode — wraps `transformer::forward` as DenseNode (gate 2/3/4 glue)
├── adaptive_width.rs   # CollapseAware + BreakevenRouter integration
├── edge_bandit.rs      # EdgeBandit — Thompson sampling over (topology, edge_set)
├── compute_router.rs   # CPU/GPU/ANE routing by topology width
└── tests.rs            # GOAT gate proofs (correctness, perf, composition)
```

---

## Task

### Phase 1 — Core Traits & Types (unblock) ✅

- [x] Create `katgpt-rs/src/dense_mesh/mod.rs` with module declarations
- [x] Define `DenseHidden` type in `types.rs` — fixed-size hidden state buffer (`Box<[f32]>`), zero-alloc scratch reuse
- [x] Define `Topology` struct in `types.rs` — `Vec<usize>` of layer widths (e.g., `[1,4,4,4,1]`), `LayerRole` enum (Input/Hidden/Output)
- [x] Define `MeshConfig` in `types.rs` — topology, edge registry, compute thresholds
- [x] Define `ComputeTarget` enum (Cpu/Gpu/Ane) in `types.rs`
- [x] Define `DenseNode` trait in `traits.rs` — `fn forward_dense(&self, input: &DenseHidden, scratch: &mut Scratch, ctx: &mut Ctx) -> DenseHidden`
- [x] Define `DenseEdge` trait in `traits.rs` — `fn route(&self, from: &DenseHidden, scratch: &mut Scratch) -> DenseHidden` + `fn cost_hint(&self) -> f32`
- [x] Add `dense_mesh` feature to `katgpt-rs/Cargo.toml`
- [x] Register module in `katgpt-rs/src/lib.rs` behind `#[cfg(feature = "dense_mesh")]`

### Phase 2 — Topology Engine & Aggregation (partial) ✅

- [x] Implement `LayerwiseTopology` in `topology.rs` — holds edge matrix `[layer][from_node][to_node] -> Box<dyn DenseEdge>`
- [x] Implement `forward_dense()` orchestration — layer-by-layer: aggregate incoming edges (summation per paper §3.1.3), call node forward, propagate
- [x] Implement aggregation as SIMD chunked sum (4 or 8 lanes, per optimisation.md)
- [x] Pre-allocate scratch buffers in `MeshConfig` builder (plasma tier — `Vec::with_capacity` once, `clear()` + reuse)
- [x] Handle variable topology width at runtime (adaptive width, not compile-time)

### Phase 3 — Edge Implementations ✅

- [x] Implement `IdentityEdge` in `edge_identity.rs` — no-op, returns input unchanged (gate 1 baseline)
- [x] Implement `ProjectionEdge` in `edge_projection.rs` — fixed random matrix multiply, no training (modelless fallback)
- [x] Implement `LoraEdge` in `edge_lora.rs` — wraps existing `LoraWeights` as a dense-edge transformation (LoRA on attention output projection)
- [x] Implement `HiddenHandoff` in `handoff.rs` — stripped forward: drafter returns `DenseHidden` instead of tokens, verifier consumes directly (F2 from research)

### Phase 4 — EdgeBandit (Self-Learning Topology) ✅

- [x] Define `EdgeBanditArm` — `(topology_shape, active_edge_subset)` pair
- [x] Implement `EdgeBandit` in `edge_bandit.rs` — Thompson sampling (Beta distribution) over arms
- [x] Reward signal: speculative verifier acceptance rate × quality proxy (win/loss for games)
- [x] Reuse existing `ThinkingBandit` / `FreqBandit` infrastructure (DRY)
- [x] Convergence test: cumulative regret < O(log T · √N) over 200 queries (gate 5)

### Phase 5 — Adaptive Width & Compute Routing ✅

- [x] Integrate with `CollapseAwareThinking` (P212) — entropy spike triggers width expansion
  - **DONE 2026-06-14.** `dense_mesh/adaptive_width.rs::collapse_signal()` reads `CollapseDetector::hesitation_count()` / `threshold()` and returns `WidthDecision::{Contract,Neutral,Expand}` based on a configurable hysteresis band (default `[0.25, 0.75]`). Mirrors the `TvpExpansion` pattern in `S2FCollapseDetector`. Feature-gated on `collapse_aware_thinking`.
- [x] Integrate with `BreakevenRouter` (P250) — breakeven analysis picks optimal width
  - **DONE 2026-06-14.** `dense_mesh/adaptive_width.rs::breakeven_signal()` reads a `BreakevenSnapshot { cpu_to_gpu_amortized }` (constructed from `BreakevenBandit::stats()`) and returns `Expand` when the CPU→GPU upgrade has amortised, else `Contract`. Feature-gated on `breakeven_routing`.
  - **Decision rule:** collapse is the primary (quality) signal — non-`Neutral` collapse always wins. When collapse has no opinion, breakeven (cost signal) decides. Both `Neutral` → falls back to `Contract` (cheapest baseline, matches gate 1).
- [x] Implement `pick_compute(width, layer_role)` in `compute_router.rs`:
  - `width == 1` → Cpu (no GPU launch overhead)
  - `width >= 4` → Gpu (data-parallel branches amortise ~50μs launch)
  - `LayerRole::Output` → Ane (final decode, per R155)
- [x] Threshold constants in `MeshConfig` (configurable, not hardcoded magic numbers)

### Phase 6 — Latent/Raw Compliance & Chain Bridge ✅

- [x] Mark `DenseHidden` as latent-only (never crosses `SyncBlock` / chain quorum)
- [x] Add bridge function `latent_to_raw_scalar()` — sigmoid projection of dense state to scalar (for chain commit, per AGENTS.md)
- [x] Add bridge function `raw_to_latent_projection()` — raw scalar lifted into dense direction (for conditioning)
- [x] Ensure raw values (token outputs, positions) only appear at input/output boundary nodes
- [x] Document anti-patterns in module doc: never sync dense state, never validate movement by latent similarity

### Phase 7 — GOAT Gate Proofs (Tests) — gate 2 FAILED, demoted to experimental

- [x] **Gate 1 (correctness):** `test_dense_mesh_chain_identity` — topology `[1,1]` + IdentityEdge produces identical output to vanilla `forward()`
- [ ] **Gate 2 (composition gain):** FAILED empirically. Two tests:
  - `test_dense_mesh_gate2_composition_differs_from_single_lora` — proves composition MECHANISM (relative L2 = 1.009) but mechanism ≠ gain
  - `test_dense_mesh_gate2_real_lora_composition_gain` — tests REAL trained Bomber LoRAs (baseline + echo vs moa target): **0/1000 wins over best single, improvement -0.00%**
  - **Verdict:** untrained LoRA composition is a no-op ensemble. Gate 2 fails. R122 must train dedicated comm edges.
- [x] **Gate 3 (easy overhead):** `test_dense_mesh_gate3_easy_overhead_vs_vanilla` — at `Config::small_target()` (vocab=4096, n_embd=64) ratio **0.997× ≤ 1.05×** ✅. At draft scale 2.71× — framework overhead visible at micro-model scale.
- [ ] **Gate 4 (hard bound):** `test_dense_mesh_gate4_hard_bound_width4_measured` — measured **9.27×** single-thread vs paper bound 2.5×. Requires vertex parallelism (batched forward or rayon) — `#[ignore]` by default, run with `--include-ignored`. Filed as follow-up.
- [x] **Gate 5 (bandit convergence):** `test_dense_mesh_gate5_bandit_convergence` and `test_bandit_converges_to_best_arm` — regret bound over 500 pulls passes
- [x] Add profiling test `prof_dense_mesh.rs` per optimisation.md template
- [x] Add GOAT gate tests file `tests/dense_mesh_goat_gates.rs` (replaces synthetic-only checks with real `transformer::forward` measurements)

### Phase 8 — Documentation & Feature Gate ✅

- [x] Add `dense_mesh` to feature flags section in `README.md`
- [x] Add DenseMesh section to `README.md` feature showcase (after SubstrateGate)
- [x] Update `.research/234_...` status to "GOAT G1+G2+G3+G5 PASS, G4 measured" (next edit)
- [x] Create benchmark output format showing topology/latency/quality tradeoff (served by `prof_dense_mesh.rs` T1 scaling table + `.benchmarks/266_densemesh_goat.md`)
- [ ] If gates 1–3 pass AND gate 2 ≥ 5 pp gain → promote to default, demote SubstrateGate if dominated (NOT MET — gate 2 proves mechanism only, not win rate)

---

## Dependencies (existing modules reused — DRY)

- `katgpt-rs/src/speculative/thinking_controller.rs` — ThinkingBandit (for EdgeBandit)
- `katgpt-rs/src/speculative/types.rs` — ForwardContext, scores
- `katgpt-rs/src/types.rs` — LoRA weights, DomainLatent
- `katgpt-rs/src/inference_router.rs` — compute target routing
- `katgpt-rs/src/simd.rs` — SIMD primitives for aggregation
- `katgpt-core/src/traits.rs` — ConstraintPruner pattern (for trait style)
- `katgpt-rs/src/transformer.rs` — forward pass (DenseNode impl wraps this)

---

## Validation

```bash
# Correctness gate
cargo test --features dense_mesh test_dense_mesh_chain_identity -- --nocapture

# Perf gate (must run release)
cargo test --release --features dense_mesh prof_dense_mesh -- --nocapture

# Composition gate (requires game LoRAs — may stub for modelless proof)
cargo test --features dense_mesh test_dense_mesh_multi_game -- --nocapture

# Full feature build check
cargo build --features dense_mesh
```

---

## Out of Scope (riir-ai R122)

- Training edge LoRAs (model-based, private)
- Cross-game edge composition recipes (private IP)
- Sleep-cycle edge consolidation (private)
- Game-specific edge weight assets (private)

---

## TL;DR

Implement `DenseMesh` trait framework in katgpt-rs behind `dense_mesh` feature. Core deliverable: `DenseNode` + `DenseEdge` traits, layer-wise topology engine, EdgeBandit, adaptive-width compute routing. 8 phases, 35 tasks. GOAT-gated by 5 arena proofs. Public framework (MIT); the actual edge composition recipes are riir-ai R122.
