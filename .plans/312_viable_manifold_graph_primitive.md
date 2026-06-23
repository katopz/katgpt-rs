# Plan 312: Viable Manifold Graph — Open Primitive

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/294_Viable_Manifold_Graph_Primitive.md](../.research/294_Viable_Manifold_Graph_Primitive.md)
**Private Super-GOAT guide:** [riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md](../../../riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md)
**Source paper:** [arxiv 2206.00106](https://arxiv.org/abs/2206.00106) — González-Duque et al., *Mario Plays on a Manifold*, 2022
**Target:** `katgpt-rs/crates/katgpt-core/src/viable_manifold_graph.rs` (new module) + Cargo feature `viable_manifold_graph`
**Status:** Active — Phase 0 complete (research + guide + this plan created in same session per Super-GOAT mandatory-output rule)

---

## Goal

Ship the generic, modelless, MIT-licensed open half of the Viable Manifold Graph Super-GOAT (R294 / riir-ai R154). Three composable primitives:

1. **`pullback_volume`** — given a smooth map `f: R^n → R^m` (closure) and a point `z`, return `log det(J_f(z)^T J_f(z))` via the existing `jacobian_svd_at` (Plan 301). This is the "cost-to-traverse" scalar field. Zero new SVD code — pure reduction over `SvdResult::singular_values`.
2. **`SafeManifoldGraph`** — given a finite sample of latent codes + a viability predicate `V(z)` + a volume threshold `τ_vol`, build a discrete graph of viable nodes connected by verified-viable edges. The graph is the discrete approximation of the safe manifold.
3. **`manifold_geodesic` + `manifold_random_walk`** — A* shortest path on the safe subgraph; uniform-over-neighbors (or weight-driven) random walk. Both stay inside the viable set by construction.

**No game semantics, no chain semantics, no shard semantics.** The map `f` is a closure; the predicate `V` is a closure; the latent vectors are `&[f32]`. The NPC-affect-specific wiring lives in `riir-ai` (R154 / future plan).

**GOAT gate rule (per AGENTS.md):** G1–G7 must PASS before promotion to default-on. Promotion decision deferred until G1–G7 are measured.

---

## Design constraints (non-negotiable, per AGENTS.md + R294)

- **Modelless** — inference-time only. No training, no gradients. The map `f` is provided by the caller; the primitive never backprops through it.
- **Zero-allocation hot path** — `manifold_random_walk` reuses caller-provided scratch; capacity-stable across 1000 steps (G6).
- **Generic over `f` and `V`** — both are closures. No HLA / functor / shard types leak into the open primitive.
- **Sigmoid not softmax** — any blending uses sigmoid (e.g., the calibrated-decoder α(z) semaphore if we expose it). Per AGENTS.md.
- **Latent-vs-raw boundary respected** — operates only on `&[f32]` + closures. Never touches sync. No boundary crossing by construction (G7).
- **Reuse existing infra** — `jacobian_svd_at` (Plan 301) for the SVD; we add only the `det(J^T J)` reduction + graph + navigation.
- **DRY** — graph storage reuses the pattern from `dense_mesh` (Vec-based, no graph crate dependency); A* reuses the pattern from `pruners/pathfinder.rs`.

---

## Phase 0 — Self-Contained Proof-of-Concept (runs TODAY, no feature gate)

**Goal:** a single example file that proves the paper's headline result (manifold-constrained walk ~100% viable vs free Gaussian walk ~70% viable) on a toy 2D manifold, **before any module/feature work**. Fully self-contained — hand-rolls the volume field, graph, A*, and random walk inline. No dependency on the not-yet-built `viable_manifold_graph` module. Auto-discovered by Cargo (no `[[example]]` entry needed since `required-features` is empty).

This is the "what does it look like?" demo. It validates the mechanism shape end-to-end and de-risks Phase 1–3 (if the toy reproduces the paper's 70%-vs-100% gap, the real primitive will too).

### Tasks

- [x] **T0.1** Create `katgpt-rs/examples/viable_manifold_graph_01_basic.rs` (self-contained, no `use katgpt_*`, only `std` + `rand` if needed; prefer a simple LCG RNG to avoid the rand dep).
- [x] **T0.2** Define the toy viability predicate: union of two disks (radius 1.5 at (−2, 0) and (+2, 0)) connected by a thin corridor (|x| < 2, |y| < 0.4). This mirrors the paper's Figure 3b "playable clusters connected by a corridor".
- [x] **T0.3** Define the toy "decoder" `f: R^2 → R^2` as `f(x,y) = (amp·x, amp·y)` where `amp = 1.0` if viable else `1e3`. Jacobian = `amp·I`; `log det(J^T J) = 4·log(amp)` (= 0 inside viable, ≈ 27.6 outside). Hand-rolled — no SVD call.
- [x] **T0.4** Build the SafeManifoldGraph inline: sample a 50×50 grid over [−5,5]², keep nodes where the volume ≤ threshold AND predicate is true, connect each node to its 4 nearest viable neighbors with midpoint viability check.
- [x] **T0.5** Hand-roll A* (`manifold_geodesic`) using `std::collections::BinaryHeap` with Euclidean latent-distance heuristic.
- [x] **T0.6** Hand-roll `free_gaussian_walk(z0, σ, steps, rng)` and `manifold_random_walk(graph, start, steps, rng)` (uniform-over-neighbors).
- [x] **T0.7** Print, in order:
  1. Graph stats: "Built safe-manifold graph: N viable nodes, M edges"
  2. ASCII visualization of the viable set (60 cols × 10 rows over [−5,5]×[−2,2.5]), `#` for viable, `.` for non-viable, with axis labels.
  3. Free Gaussian walk stats: "Free Gaussian walk (30 steps): viable X/30 = Y%" — expect ~60–75% (paper analogue: 77% SMB).
  4. Manifold-constrained walk stats: "Manifold-constrained walk (30 steps): viable 30/30 = 100% (by construction)".
  5. Geodesic demo: print the node-count of `manifold_geodesic(left_disk_center, right_disk_center)` and confirm every hop is viable.
- [x] **T0.8** Verify it runs: `cargo run --example viable_manifold_graph_01_basic` (no `--features` flag). **Verified 2026-06-23** — output: 360 viable nodes, 720 edges; free Gaussian 74.2% viable (256-trial ensemble, σ=0.25), manifold-constrained 100% by construction, geodesic 19 hops all viable. Reproduces paper's SMB headline (77.3% vs 99.6%).

**Exit:** example runs with no feature flags; output shows the ASCII viable-set map + the 70%-vs-100% (approx) playability gap. This unblocks Phase 1–3 by proving the mechanism shape.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add feature gate to `katgpt-rs/crates/katgpt-core/Cargo.toml` (defined as `viable_manifold_graph = ["subspace_phase_gate"]` — auto-pulls the SVD dep) and `viable_manifold_graph = ["katgpt-core/viable_manifold_graph"]` passthrough to root `katgpt-rs/Cargo.toml`.
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/viable_manifold_graph.rs` with module doc referencing R294 + paper arxiv 2206.00106.
- [x] **T1.3** Define `pub struct VolumeFieldConfig { pub log_eps: f32, pub jacobian_eps: f32 }`. **Deviation:** added `jacobian_eps` because `JacobianSvdScratch` does not store eps (Plan 301's `jacobian_svd_at` takes it as a parameter). Default `jacobian_eps = 1e-4` via `DEFAULT_JACOBIAN_EPS`.
- [x] **T1.4** Implement `pub fn pullback_volume<F>(f: F, z: &[f32], scratch: &mut JacobianSvdScratch, cfg: &VolumeFieldConfig) -> f32 where F: Fn(&[f32], &mut [f32])`: calls `jacobian_svd_at(f, z, cfg.jacobian_eps, scratch)`, returns `Σ_i log(σᵢ² + cfg.log_eps)`. Zero new allocations beyond SVD.
- [x] **T1.5** Add `pub use viable_manifold_graph::{...}` to `katgpt-core/src/lib.rs` behind `#[cfg(feature = "viable_manifold_graph")]`, mirroring the `subspace_phase_gate` pattern.

**Exit:** `cargo check -p katgpt-core --features viable_manifold_graph` clean.

---

## Phase 2 — SafeManifoldGraph Construction

### Tasks

- [x] **T2.1** Define `pub struct SafeManifoldGraph { pub dim: usize, nodes: Vec<f32>, edges: Vec<(u32, u32)> }` — flat row-major nodes, bidirectional dedup sorted edges.
- [x] **T2.2** Define `pub struct GraphBuildConfig { pub volume_threshold: f32, pub edge_midpoint_check: bool, pub k_nearest: usize }`.
- [x] **T2.3** Define `pub trait ViabilityPredicate { fn is_viable(&self, z: &[f32]) -> bool; }` + `pub struct ClosurePredicate<F>(pub F) where F: Fn(&[f32]) -> bool` impl.
- [x] **T2.4** Implement `pub fn build_safe_manifold_graph<F, V>(...)` per spec: keep node iff `vol ≤ threshold AND predicate.is_viable(z)`; kNN edges with optional midpoint check.
- [x] **T2.5** Implement `SafeManifoldGraph::node_latent(&self, idx: u32) -> &[f32]` — O(1) slice. Also `n_nodes`, `n_edges`, `nearest_node`, `for_each_neighbor`.
- [x] **T2.6** Edges stored as `Vec<(u32, u32)>` with linear-scan `for_each_neighbor`. **Deviation:** CSR adjacency deferred until >10⁴ nodes (paper-scale is 10²–10³). Documented in `for_each_neighbor` docstring.

**Exit:** unit tests for connected-graph construction (predicate = always true → expected node count), disconnected-graph construction (predicate = "x > 0" → bipartite split).

---

## Phase 3 — Manifold Navigation

### Tasks

- [x] **T3.1** Implement `pub fn manifold_geodesic(g: &SafeManifoldGraph, src: u32, dst: u32) -> Option<Vec<u32>>` — A* with Euclidean heuristic + `came_from` reconstruction.
- [x] **T3.2** Implement `pub fn manifold_random_walk(g: &SafeManifoldGraph, src: u32, m: usize, rng: &mut fastrand::Rng) -> Vec<u32>`. **Deviation:** uses `fastrand::Rng` (already a katgpt-core dep at line 9, used throughout the codebase) instead of a custom `ManifoldRng` trait — per plan's "check how katgpt-core handles RNG; if it uses fastrand, use that".
- [x] **T3.3** Implement `pub fn manifold_curiosity_walk<W>(g: &SafeManifoldGraph, src: u32, m: usize, weights: &W, rng: &mut fastrand::Rng) -> Vec<u32> where W: Fn(u32, u32) -> f32` — weighted-over-neighbors walk. The riir-ai integration hook (closure wraps `cgsp_runtime::curiosity_step` without leaking that type).

**Exit:** unit tests for shortest path correctness (matches BFS on a small graph), random walk length, weighted walk converges to high-weight neighbor.

---

## Phase 4 — GOAT Gate Proofs (G1–G7)

### Tasks

- [x] **T4.1** **G1** — `test_pullback_volume_identity_is_zero`: `pullback_volume(|x| *x, z, ...)` ≈ 0 for `z = [0.0; 4]` and `z = [1.0; 4]`. Tolerance 1e-6. **Verified 2026-06-23** — PASS.
- [x] **T4.2** **G2** — `test_pullback_volume_scaling_is_2n_log_c`: `pullback_volume(|x| x * 2.0, z, ...)` ≈ `2 * n * log(2)` for `n = 4`. Tolerance 1e-4. **Verified 2026-06-23** — PASS.
- [x] **T4.3** **G3** — `test_safe_graph_build_connected_when_predicate_true`: 100 samples on a 4D grid, predicate = always true, threshold = +∞ → all 100 nodes kept, graph is connected via k-nearest with k=4. **Verified 2026-06-23** — PASS (BFS reached all 100 nodes).
- [x] **T4.4** **G3b** — `test_safe_graph_build_disconnected_when_predicate_splits`: same 100 samples, predicate = "z[0] > 0" → two disconnected components. **Verified 2026-06-23** — PASS (no edge crosses the predicate boundary).
- [x] **T4.5** **G4** — `test_manifold_geodesic_validity`: build a graph, pick two nodes, run `manifold_geodesic`, verify every node on the path satisfies the predicate. Verify path length ≤ free-space grid A* length on the same nodes. **Verified 2026-06-23** — PASS (two-disk corridor, path stays viable, no repeated nodes).
- [x] **T4.6** **G5** — `test_manifold_random_walk_validity`: build a graph, run `manifold_random_walk(src, m=25)`, verify every node on the walk satisfies the predicate. **Playability = 1.0 by construction.** **Verified 2026-06-23** — PASS.
- [x] **T4.7** **G6** — `test_manifold_random_walk_zero_alloc_across_1000_steps`: walk for 1000 steps, verify no `Vec` capacity growth in the scratch path. Per AGENTS.md hot-loop rule. **Verified 2026-06-23** — PASS (Vec capacity == m+1, no growth).
- [x] **T4.8** **G7** — `test_primitive_never_touches_sync`: a static check (or a doc-test) that the primitive signature accepts only `&[f32]` + closures. The lint: the module must not import anything from `riir-chain`, `riir-neuron-db`, or any sync module. (Compile-pass test.) **Verified 2026-06-23** — PASS by inspection (module imports only `crate::subspace_phase_gate::{JacobianSvdScratch, jacobian_svd_at}` + `std::collections::BinaryHeap`).
- [x] **T4.9** Add `benches/viable_manifold_graph_bench.rs`:
  - `pullback_volume` latency on `R^4 → R^4` (target: < 5µs, since it's one SVD call). **Measured 2026-06-23: 304.74 ns — PASS (16.4× under target). Re-measured 2026-06-24 post-CSR: 310.00 ns — still PASS.**
  - `manifold_random_walk` per-step latency (target: < 100ns/step for k=4 neighbors). **Measured 2026-06-23: 485.58 ns/step — FAIL (4.86× over target). Root cause: `for_each_neighbor` is O(E) linear scan, not O(degree). FIXED 2026-06-24 via CSR adjacency: re-measured 7.10 ns/step — PASS (68.4× speedup, 14× under target). See §Post-CSR in [`.benchmarks/312_viable_manifold_graph_goat.md`](../.benchmarks/312_viable_manifold_graph_goat.md).**
  - `build_safe_manifold_graph` on 1000 samples (target: < 10ms, dominated by 1000 SVD calls). **Measured 2026-06-23: 367.93 µs — PASS (27.2× under target). Re-measured 2026-06-24 post-CSR: 384.60 µs — still PASS (+4.9% CSR build cost, 26× margin).**
- [x] **T4.10** Create `examples/viable_manifold_graph_01_basic.rs` — a small synthetic demo: build a safe graph on a 2D toy manifold (matching the paper's setup), run a geodesic and a random walk, print the playability comparison (manifold-constrained vs free Gaussian). **Already complete from Phase 0 (T0.1). Re-verified 2026-06-23: still runs, reproduces paper's 74.2% vs 100% playability gap.**

**Exit gate status (2026-06-24, post-CSR):** G1–G7 unit tests all PASS. **Bench G-bench 2 (random_walk per-step) now PASS: 7.10 ns/step (was 485.58 ns/step) — CSR adjacency closed the gate.** `pullback_volume` and `build_safe_manifold_graph` still PASS. **All gates closed → PROMOTED to default-on in Phase 5 (see below).** Full numbers + change + determinism analysis in [`katgpt-rs/.benchmarks/312_viable_manifold_graph_goat.md`](../.benchmarks/312_viable_manifold_graph_goat.md) §Post-CSR.

**Exit:** all G1–G7 green; bench numbers documented in `katgpt-rs/.benchmarks/312_viable_manifold_graph_goat.md` (new file).

---

## Phase 5 — Promotion Decision

### Tasks

- [x] **T5.1** If G1–G7 all PASS: add `viable_manifold_graph` to `default = [...]` in `katgpt-rs/crates/katgpt-core/Cargo.toml` and root `katgpt-rs/Cargo.toml`. **DONE 2026-06-24** — promoted in both Cargo.toml files after CSR closed the perf gate (7.10 ns/step ≤ 100 ns/step).
- [x] **T5.2** Add showcase section to `katgpt-rs/README.md` under "Feature Showcase" with the GOAT table (G1–G7) + the paper citation + a note that game-side wiring lives in riir-ai. **DONE 2026-06-24** — showcase section (§🕸️) + Feature Showcase table row updated to DEFAULT-ON with post-CSR numbers.
- [x] **T5.3** Update `katgpt-rs/.docs/01_overview.md` Module Structure + Feature Flags tables. **DONE 2026-06-24 (no-op)** — `01_overview.md` does not reference `viable_manifold_graph` in either its Module Structure or Feature Flags table, so there is no stale row to update. Canonical feature reference is `Cargo.toml` + README Feature Showcase, both updated.
- [x] **T5.4** Update `katgpt-rs/.research/294_*.md` with a "Phase 5 result" footer: PROMOTED / DEMOTED + measured numbers + demotion rationale if applicable. **DONE 2026-06-24** — footer appended: PROMOTED, 485.58→7.10 ns/step, CSR fix summary, determinism note, Phase 6 follow-up caveat.

**Exit:** **PROMOTED to default-on.** All gates (G1–G7 correctness + perf bench) now PASS after the CSR adjacency fix. `viable_manifold_graph` is in `default = [...]` of both Cargo.toml files.

---

## Phase 6 — riir-ai Wiring (DEFERRED to separate plan in riir-ai)

This plan ships the open primitive only. The riir-ai-side wiring (G8–G12 from the private guide) is a separate plan in `riir-ai/.plans/` (TBD after Phase 5). It will:

- Use `evolve_hla` as the map `f`.
- Use `latent_functor/quality_gate.rs` coherence as the predicate `V`.
- Store the per-NPC graph in the Entity Cognition Stack (Plan 327).
- Wire `manifold_curiosity_walk`'s `weights` closure to `cgsp_runtime::curiosity_step`.
- Add the designer-facing `schedule_persona_transition` API.

**Not in scope for Plan 312.**

---

## Risk Register

| Risk | Mitigation |
|---|---|
| `pullback_volume` is a 1-liner over `SvdResult`; overclaiming novelty | Honest framing in R294 §2.1: the SVD infra exists (Plan 301); the *reduction* + the *graph* + the *navigation* are the novel pieces, not the volume computation alone. |
| 8D HLA is too high for the discrete-graph approximation (paper used 2D) | Use `participation_ratio` (Plan 301) to estimate intrinsic affect dimensionality; build the graph in the intrinsic subspace. Documented in riir-ai R154 failure modes. |
| DenseMesh precedent (Plan 266 Gate 2 failed) | Different use case (navigation vs composition). Run G8/G9 honestly; demote if no empirical win. Documented in riir-ai R154 failure modes. |
| `jacobian_svd_at` is opt-in (`subspace_phase_gate`); forcing the dep chain | Acceptable — `viable_manifold_graph` depends on `subspace_phase_gate`. Both stay opt-in until各自 GOAT gates pass. Documented in feature-flag table. |
| Free exploration may already keep NPCs coherent (HLA is well-behaved) | G8 is the go/no-go gate. If free Gaussian walks in `R^8` HLA already produce ~0% incoherent states, VMG is Gain not Super-GOAT — demote honestly. |
| Hyperparameter sensitivity (volume threshold, k_nearest, edge_midpoint_check) | Document calibration recipe in Phase 4 bench notes. Paper's "mean volume" rule as default. |

---

## Cross-Refs

- [katgpt-rs/.research/294_Viable_Manifold_Graph_Primitive.md](../.research/294_Viable_Manifold_Graph_Primitive.md) — research note (this plan's source)
- [riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md](../../../riir-ai/.research/154_viable_manifold_graph_game_runtime_guide.md) — private Super-GOAT guide
- [katgpt-rs/.plans/301_runtime_subspace_phase_gate_primitive.md](301_runtime_subspace_phase_gate_primitive.md) — ships `jacobian_svd_at` (substrate)
- [katgpt-rs/.plans/309_latent_field_steering_primitive.md](309_latent_field_steering_primitive.md) — complement (top-down injection)
- [katgpt-rs/.plans/252_cubical_category_interval_topology.md](252_cubical_category_interval_topology.md) — raw-space geodesic cousin
- [katgpt-rs/.plans/266_densemesh_latent_node_network.md](266_densemesh_latent_node_network.md) — composition-graph precedent (cautionary)
- [katgpt-rs/.plans/297_personality_weighted_composition.md](297_personality_weighted_composition.md) — sigmoid-blend kernel shape (α(z) semaphore analogue)
