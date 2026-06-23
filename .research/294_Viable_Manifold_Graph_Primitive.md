# Research 294: Viable Manifold Graph — Geodesic / Random-Walk on a Safe Subgraph of Latent Space

> **Source:** [Mario Plays on a Manifold: Generating Functional Content in Latent Space through Differential Geometry](https://arxiv.org/pdf/2206.00106) — González-Duque, Palm, Hauberg, Risi (ITU Copenhagen / DTU / modl.ai), arXiv:2206.00106v1, 31 May 2022
> **Date:** 2026-06-23
> **Status:** Active — Fusion Research (Super-GOAT)
> **Related Research:** 290 (Latent Field Steering — top-down injection; this is the *constrained-exploration* complement), 276 (PersonalityWeightedComposition — same sigmoid-blend kernel shape as the α(z) semaphore), 279 (Subspace Phase-Gate — ships `jacobian_svd_at`, the SVD-of-J primitive this builds on), 257 (Functional Attention Spectral Transport — operator on belief manifold), 234 (DenseMesh — *composition* graph, failed Gate 2; this is the *navigation* graph, different use), 051 (Deep Manifold fixed-point boundary), 270 (ICT branching detector — gates *when* to walk)
> **Related Plans:** 312 (this primitive — open), 309 (Latent Field Steering — companion), 297 (PersonalityWeightedComposition), 301 (Subspace Phase-Gate — ships the SVD infra), 252 (Cubical CAT(0) geodesic — raw-space cousin; this is the latent-space upgrade), 266 (DenseMesh — failed composition graph)
> **Cross-ref (riir-ai):** Research 154 (Viable Manifold Graph — Game Runtime Guide, **private Super-GOAT moat**)
> **Classification:** Public — generic modelless math (no game semantics, no chain semantics, no shard semantics)

---

## TL;DR

The paper trains a VAE on Mario/Zelda levels and uses Riemannian geometry to construct a discrete graph of "playable" latent nodes, on which interpolations (A* shortest path) and random walks stay reliably within functional content (~99% playability vs ~77% for naive Gaussian walks). The VAE training, hierarchical categorical decoder, and the Baumgarten A* Mario solver are all out of scope here (training pipeline → `riir-train`; Mario agent → game-specific).

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **pure geometry**. Given any smooth map `f: R^n → R^m` (a "decoder" in the paper; in our stack it can be `evolve_hla`, a `latent_functor` application, a `SenseModule::project`, or any closure) and any viability predicate `V(z)` over latent codes:

1. **Pullback volume field** — `vol(z) = log det(J_f(z)^T J_f(z))` is computable from the singular values returned by our existing `jacobian_svd_at` (Plan 301). High volume ⇒ small latent perturbations produce large output changes ⇒ "expensive to traverse". This is the geometric signal the paper uses to mark non-playable regions.
2. **Safe-manifold subgraph** — sample a finite set of latent codes, keep the ones where `vol(z) ≤ τ_vol` (geometric viability) AND `V(z) = true` (semantic viability, pluggable). Connect two nodes by an edge iff their interpolation midpoint is also viable. This is a discrete graph approximation of the safe manifold.
3. **Manifold-constrained navigation** — `manifold_geodesic(g, src, dst)` runs A* on the safe subgraph (paper §III-C); `manifold_random_walk(g, src, m)` does uniform-over-neighbors walk for `m` inner steps (paper §III-C). Both stay inside the viable set by construction.

This is the **constrained-exploration complement** to Latent Field Steering (R290): LFS injects a direction vector top-down; VMG constrains where exploration is allowed to go. Together they cover both halves of "steering" — push toward + keep within.

**All three pieces are engine plumbing — no know-how leak, no game IP, no chain IP, no shard IP.** The game-AI selling point (NPC crowd-scale persona exploration that stays coherent by construction) lives in the private `riir-ai/.research/154` guide.

---

## 1. Paper Core Findings

### 1.1 The problem: linear interpolation and Gaussian random walks leave the playable manifold

A trained VAE's latent space does not uniformly decode to functional content. Linear interpolation `z(t) = (1−t)·z_a + t·z_b` between two playable codes can cross unplayable regions; Gaussian random walks `z_{n+1} = z_n + ε`, ε ∼ N(0, I) drift out of the playable set. Paper Table I: naive Gaussian walk on SMB reaches only **77.3%** playability; on Zelda, **56.7%**. This is the failure mode the paper fixes.

### 1.2 The pullback metric as a "cost-to-traverse" field

For a decoder `dec: Z → R^D`, the pullback of the Euclidean metric is `M(z) = J_dec(z)^T J_dec(z)` where `J_dec` is the Jacobian. The volume `det M(z)` measures how much latent-space volume expands near `z` under decoding — high volume ⇒ decoding is locally near a discontinuity / extrapolation regime. Paper Eq. (4) approximates `J_dec` by forward differences.

**The calibration trick (paper §III-B, Eq. 5):** replace the decoder's σ with `10^5` when near non-playable regions, blending via a sigmoid semaphore:

```
dec_calibrated(z) = α(z) · dec_{μ,σ}(z) + (1 − α(z)) · dec_{μ, 10^5}(z)
α(z) = sigmoid((minDist(z) − β·k) / β)         // paper Eq. 6, k ≈ 6.9, β = 5.5
```

where `minDist(z)` is the distance to the nearest known non-playable training code. This forces the calibrated decoder to output noise near non-playable regions, blowing up the Jacobian, hence `det M(z)`.

### 1.3 The discrete-graph approximation (paper §III-C)

Sample a fine grid (paper uses 100×100). Compute `log det M(z)` for every node. Threshold at the mean volume. Keep low-volume nodes; connect adjacent low-volume nodes. The result is a finite graph that approximates the playable manifold. A* on this graph gives manifold-aware shortest paths; uniform-over-neighbors random walks on it stay inside the playable set.

### 1.4 Empirical results (paper Table I, 10 VAE runs SMB + 4 VAE runs Zelda)

| Game | Method | Interp. playability | RW playability |
|---|---|---|---|
| SMB | Ours | **0.993** ± 0.033 | **0.996** ± 0.010 |
| SMB | Baseline (center-of-mass RW) | 0.953 ± 0.084 | 0.963 ± 0.026 |
| SMB | Normal (Gaussian RW) | 0.949 ± 0.093 | 0.773 ± 0.169 |
| Zelda | Ours | **0.961** ± 0.068 | **0.995** ± 0.011 |
| Zelda | Normal | 0.896 ± 0.105 | 0.567 ± 0.257 |

Tradeoff: gain in reliability comes at a small cost in diversity of decoded samples. The "jump submanifold" experiment (Table I, bottom) shows the method generalizes to *submanifolds* (e.g., "playable AND requires jump") without retraining.

### 1.5 What does NOT transfer (training-only)

- The VAE training recipe, ELBO optimization, hierarchical categorical decoder, Adam schedule → `riir-train`. One-line note: *training pipeline → riir-train, not distilled here.*
- The Baumgarten A* Mario agent and the Zelda grammar checker are domain-specific viability predicates. The primitive is **predicate-agnostic**.

---

## 2. Distillation

### 2.1 What's already in katgpt-rs (verified by notes + code grep, 2026-06-23)

| Paper concept | Existing codebase analogue | Status |
|---|---|---|
| Jacobian `J_dec(z)` forward-difference approximation | `jacobian_svd_at(f, x, eps, scratch)` (Plan 301, Bench 301 G1 PASS) — already returns singular values of `J_f(x)` | ✅ Shipped, **opt-in** (`subspace_phase_gate` feature). Returns `SvdResult { singular_values, ... }`. `det(J^T J) = Π σᵢ²` is a one-line product over the returned singular values. |
| Intrinsic-dimension phase gate `N ≥ d` | `phase_transition_gate(n, d)` (same module) | ✅ Shipped. Different use (sample sufficiency), same math substrate. |
| Sigmoid semaphore `α(z)` blend | `PersonalityWeightedComposition::compose_into` (Plan 297) kernel: `Σ sigmoid(wᵢ/τ) · belief_confidence_i · dᵢ` | ✅ Shipped, **default-on**. Same mathematical shape as the calibrated-decoder blend. |
| Top-down direction-vector injection (the "steer toward" half) | `apply_latent_steering(state, field)` (Plan 309, R290) | ✅ Shipped, **default-on**, Super-GOAT. VMG is the "stay within" complement. |
| Quality gate on direction vectors (Dirichlet separation ratio) | `latent_functor/quality_gate.rs::DirectionQuality` | ✅ Shipped in riir-ai. Predicate-shape analogue for *directions*, not *states*. |
| Coherence-decay → re-estimation trigger | `latent_functor/reestimation.rs::tick` (Plan 303) | ✅ Shipped in riir-ai. The "low coherence = drift off safe manifold" signal exists; VMG adds the *graph* substrate to make drift recoverable by navigation, not just by re-estimation. |
| Generic A* on graphs | `pruners/pathfinder.rs::find_path` (Plans 017/018, raw grid A*) | ✅ Shipped. Raw-grid, not graph-of-latent-nodes. Different substrate. |
| CAT(0) geodesic on safe nodes (raw space) | `CubicalNerve::cat0_geodesic()` (Plan 252) | ✅ Shipped. **Raw navigation space, not latent.** Direct structural cousin — same shape (shortest path on a subgraph of "safe" nodes), different domain. VMG is the latent-space upgrade. |
| Latent node network substrate | `DenseMesh` (Plan 266) | ⚠️ Shipped but **Gate 2 FAILED** empirically (composition of untrained LoRA edges = no-op). Substrate exists; the *composition* use case failed. VMG uses a latent node graph for *navigation*, not composition — different use case, not blocked by DenseMesh's failure. |

### 2.2 What's NOT in katgpt-rs (the gap — three missing primitives)

1. **Pullback volume field** — `pub fn pullback_volume<F>(f: F, z: &[f32], scratch: &mut JacobianSvdScratch) -> f32` returning `log(Π σᵢ² + ε)` where `σᵢ` are singular values of `J_f(z)`. Trivially derived from `jacobian_svd_at`'s `SvdResult` — but **not currently exposed**. The SVD primitive exists; the `det(J^T J)` reduction does not.
2. **SafeManifoldGraph** — a `SafeManifoldGraph { nodes: Vec<f32>, dim, edges: Vec<(u32, u32)> }` built from a finite sample + a viability predicate. Construction: sample → filter by `(vol(z) ≤ τ_vol) AND V(z)` → connect adjacent viable nodes → verify edge midpoints. **No shipped equivalent** (DenseMesh is a *composition* graph, not a *navigation* graph).
3. **Manifold-constrained navigation** — `manifold_geodesic(g, src, dst) -> Vec<u32>` (A* on the safe subgraph) + `manifold_random_walk(g, src, m, rng) -> Vec<u32>` (uniform-over-neighbors walk, `m` inner steps). **No shipped equivalent** in latent space. `pathfinder.rs` is raw-grid; `cat0_geodesic` is raw navigation.

### 2.3 Closest cousins (3)

1. **Latent Field Steering (R290 / Plan 309)** — `apply_latent_steering(state, field)`. Top-down injection. **Complement, not duplicate**: LFS has no notion of "stay within safe region"; it just adds a vector. Without VMG, LFS-steered exploration can drift off the coherent affect manifold. With VMG, LFS provides the per-step push and VMG provides the constraint that the push stays inside the safe graph. They are the two halves of "steering".
2. **Subspace Phase-Gate (R279 / Plan 301)** — `jacobian_svd_at`. Computes the SVD of `J_f`. **Same math substrate, different use**: Plan 301 uses it for intrinsic-dimension estimation and phase-transition gating; VMG uses the *same* SVD's singular values for volume-field computation. Plan 301 is the infrastructure; VMG is a new consumer.
3. **Cubical CAT(0) geodesic (Plan 252)** — `cat0_geodesic()` on the cubical nerve of a game zone poset. **Same structural shape** (shortest path on a subgraph of safe nodes), **different domain** (raw navigation space vs latent belief space). Plan 252 is the raw-space cousin; VMG is the latent-space upgrade.

### 2.4 Fusion — what novel combination does this enable?

**Fusion A (PRIMARY — Super-GOAT, see riir-ai R154): VMG × Latent Field Steering × PersonalityWeightedComposition × cgsp curiosity → coherent crowd-scale persona emergence**

- *Latent Field Steering* (R290) = top-down direction push
- *PersonalityWeightedComposition* (R276) = sigmoid-blend of personality directions
- *cgsp curiosity* (riir-ai R126) = "what should this NPC explore next?"
- *VMG* (this note) = "the exploration must stay inside the coherent affect manifold"

Without VMG: crowd-scale curiosity-driven exploration of HLA affect space produces broken intermediate states (NPCs that randomly wander into degenerate valence/arousal configurations and behave erratically). With VMG: every curiosity step is a one-edge walk on the safe-manifold graph, so the NPC's affect trajectory is *guaranteed* to pass only through coherent intermediate personas. This is the paper's headline result (~99% playability vs ~77% for free walks) translated to NPC affect.

**Fusion B (open, secondary): VMG × jacobian_svd_at × reestimation → drift-recovery by navigation, not by re-estimation**

Today, `latent_functor/reestimation.rs` triggers when coherence decays — it re-derives the direction vector from scratch. VMG offers an alternative recovery: instead of re-estimating, *navigate back* along the safe-manifold graph to the nearest coherent node. This is cheaper (graph lookup vs SVD) and preserves the learned direction (no re-derivation).

**Fusion C (latent-space reframing of an existing raw-space primitive): VMG × Cubical CAT(0) → unified safe-graph navigation across raw and latent space**

Plan 252 ships `cat0_geodesic` for raw navigation. VMG ships `manifold_geodesic` for latent navigation. Together: an NPC navigating from zone A to zone B uses `cat0_geodesic` (raw); the same NPC transitioning from persona X to persona Y uses `manifold_geodesic` (latent). Same algorithm, two domains, unified interface.

---

## 3. Latent-space reframings (mandatory per workflow §1.5 step 3)

The paper operates on a VAE decoder. Our reframings operate on each Super-GOAT factory module:

### 3.1 HLA per-NPC latent state (`katgpt-rs/crates/katgpt-core/src/sense/`)

HLA state is `R^8` (valence, arousal, desperation, calm, fear + 3 reserved). The "playable manifold" becomes the **coherent affect manifold**: the subset of `R^8` where projected scalars remain behaviorally meaningful (not collapsed to a degenerate attractor, not exploding in magnitude). The "decoder" `f` is `evolve_hla` itself; `J_f(z)` is its Jacobian; `det(J^T J)` measures local sensitivity. High-sensitivity regions = small latent perturbations produce large affect shifts = "expensive to traverse". NPC affect evolution becomes a constrained random walk on the safe affect subgraph.

### 3.2 `latent_functor/` (`zone_gating`, `reestimation`, `arithmetic`, `cross_game`, `k_selector`, `quality_gate`)

A functor application `extract_functor_into(src, dst, f)` is a map `R^d → R^d`. The safe manifold around a functor's direction `f` is the set of source/target pairs whose coherence (Plan 303) exceeds `τ_reest`. VMG builds the discrete graph over these pairs; navigation on it gives coherent interpolations between learned relations (e.g., "fears X" → "fears Y" via a path of intermediate fears that all cohere). This is a **new capability** for the latent functor runtime — today, functors are atomic point-to-point; with VMG, they support manifold-aware traversal.

### 3.3 `cgsp_runtime/` (curiosity-guided self-play)

Curiosity drives exploration. Today, exploration is a free Gaussian step in latent space. VMG constrains curiosity steps to the safe-manifold graph — every exploration move lands on a viable node. This is the paper's headline win (~99% vs ~77%) translated to NPC curiosity.

### 3.4 LatCal fixed-point commitment (`riir-chain/src/encoding/latcal*.rs`)

If the safe-manifold graph itself is a chain artifact (e.g., a "faction persona manifold" committed by the faction's founding snapshot), the graph's BLAKE3-committed structure is the IP. LatCal bridges: the *graph adjacency* is committed raw; the *latent positions* of nodes stay latent. The bridge function is `manifold_graph_commitment_hash(g) → [u8; 32]` — a deterministic BLAKE3 over the sorted edge list, not over the latent coordinates.

### 3.5 `NeuronShard` (`riir-neuron-db/src/shard.rs`)

A shard's `style_weights[64]` is a frozen latent direction. The "playable manifold around this shard" = the set of latent states reachable from the shard's projection that remain coherent. Freeze/thaw versions the manifold: each snapshot defines its own safe subgraph, and persona divergence between NPCs is measurable as graph-distance between their current node in their respective manifolds. **Cross-ref for `riir-neuron-db` follow-up**: a `shard_manifold_graph` view (gated behind a new feature) would be the shard-side analogue of Plan 301's `subspace_phase_gate` wrapper.

---

## 4. Verdict

| Question | Answer | Evidence |
|---|---|---|
| **Q1** No prior art? | **YES** | Three-layer check done. Paper-vocab grep (`pullback`, `Riemannian`, `playability`, `Jacobian volume`, `manifold graph`) → zero code hits, zero research-note hits (except abstract math references). Codebase-vocab grep (`safe_graph`, `viable_node`, `metric_volume`, `det_JtJ`, `manifold_walk`) → zero code hits. Existing primitives cover *pieces* (`jacobian_svd_at` computes the SVD; LFS injects direction; Cubical CAT(0) does raw-space geodesic; DenseMesh is a *composition* graph that failed Gate 2), but **no shipped primitive computes the pullback volume, builds a safe-manifold subgraph, or runs geodesic/random-walk navigation on it in latent space**. |
| **Q2** New capability class? | **YES** | Today NPC latent state evolves by `evolve_hla` (deterministic from raw inputs) or by Latent Field Steering (top-down designer injection). There is **no mechanism** for "explore the safe affect manifold" — i.e., NPCs cannot random-walk their own affect to discover emergent personas, and designers cannot interpolate between two persona shards along a coherent path. VMG adds this. New capability class (coherent exploration of latent state), not an optimization. |
| **Q3** Product selling point? | **YES** | *"NPCs random-walk and interpolate their affect within a learned viable-manifold graph; every explored state stays coherent by construction. Designers author 'persona A → persona B' transitions and the runtime finds a coherent path through affect space — no broken intermediate states, no per-NPC training, crowd-scale at 20Hz tick."* Defensible, demoable, no flat-exploration competitor can match. |
| **Q4** Force multiplier? | **YES** | Touches ≥6 pillars: (1) HLA kernel (`sense/reconstruction.rs`), (2) `latent_functor/reestimation` + `quality_gate`, (3) Latent Field Steering (R290), (4) cgsp curiosity exploration, (5) Cubical CAT(0) geodesic (raw→latent upgrade), (6) NeuronShard freeze/thaw (persona versioning via per-snapshot manifolds), (7) `jacobian_svd_at` (new use of existing primitive). |

**All 4 YES → Super-GOAT.**

**Selling point (one sentence):** *Every NPC's affect evolution is a constrained walk on a learned viable-manifold graph, so crowd-scale curiosity-driven persona emergence stays coherent by construction — no per-NPC training, no decoded-to-broken-intermediate-personas, 20Hz tick.*

**Tier reasoning:** Novel mechanism (no shipped prior art for latent-space safe-manifold navigation), new capability class (coherent exploration vs free drift), defensible selling point (crowd-scale persona emergence), force multiplier across 6+ existing pillars. Moat: the private game-runtime guide (`riir-ai/.research/154`) contains the NPC-affect-specific wiring; the open primitive is the generic math hook.

**Mandatory outputs (this session):**
1. ✅ Open primitive note — this file (`katgpt-rs/.research/294_*.md`).
2. ✅ Private game-runtime guide — `riir-ai/.research/154_*.md` (created next).
3. ✅ Open plan — `katgpt-rs/.plans/312_*.md` (created next).

**Redirect to riir-train:** the VAE training recipe, ELBO optimization, hierarchical categorical decoder, Adam schedule, and Baumgarten A* Mario viability agent are all training- or game-specific. One-line note: *training pipeline + Mario agent → riir-train / game-specific; not distilled here.*

---

## 5. Validation protocol (how to prove it's Super-GOAT, not just hype)

The validation protocol lives in the private guide (`riir-ai/.research/154` §Validation). The open-side gates that must pass before promotion to default-on in katgpt-rs:

- **G1 (correctness)** — `pullback_volume(identity_map, z)` returns 0 for all `z` (identity Jacobian → `det(I) = 1` → `log 1 = 0`).
- **G2 (correctness)** — `pullback_volume(f, z)` for `f(x) = 2x` returns `2·n·log 2` (Jacobian = 2I, det = 2^n).
- **G3 (graph construction)** — `SafeManifoldGraph::build(samples, predicate)` produces a connected graph when the viable set is connected; produces the expected node count when the predicate is "always true".
- **G4 (geodesic)** — `manifold_geodesic(g, a, b)` returns a path whose every node satisfies the viability predicate; the path length is ≤ free-space A* length on the same node set.
- **G5 (random walk)** — `manifold_random_walk(g, src, m)` returns a walk of length `m` whose every node satisfies the predicate; empirical playability (fraction of walk nodes that satisfy V) = **1.0 by construction**, vs < 1.0 for free Gaussian walk (paper's headline: 0.99 vs 0.77).
- **G6 (zero-alloc hot path)** — `manifold_random_walk` capacity-stable across 1000 steps (per AGENTS.md hot-loop rule).
- **G7 (latent-vs-raw boundary)** — the primitive operates only on `&[f32]` latent vectors and a closure predicate; never touches sync, never emits raw scalars. No boundary crossing by construction.

Quality / behavior gates (the "is it actually better?" question) are riir-ai's responsibility — they require HLA, game scenarios, and crowd-scale benchmarks that don't belong in the public engine.

---

## 6. What stays open vs private

| Piece | Location | Why |
|---|---|---|
| `pullback_volume`, `SafeManifoldGraph`, `manifold_geodesic`, `manifold_random_walk` (generic math) | `katgpt-rs/crates/katgpt-core/src/viable_manifold_graph.rs` (new, feature `viable_manifold_graph`) | Generic — applies to any `Fn(&[f32], &mut [f32])` map and any closure predicate. No game/chain/shard semantics. |
| NPC-affect-specific wiring (HLA as `f`, coherence as `V`, crowd-scale dispatch) | `riir-ai/crates/riir-engine/src/` (private) | Game IP — the *selling point*. |
| Per-NPC persona-versioned manifolds via NeuronShard freeze/thaw | `riir-neuron-db/src/` (future, follow-up) | Shard IP — *if* we decide persona manifolds are shard-side artifacts. |
| Chain-committed manifold graphs (faction persona manifold as BLAKE3-committed artifact) | `riir-chain/src/` (future, follow-up) | Chain IP — *if* we decide manifolds cross the sync boundary. |

---

## 7. References

- **Source paper:** [arXiv:2206.00106](https://arxiv.org/abs/2206.00106) — González-Duque et al., 2022.
- **Prior art the paper builds on:** Arvanitidis et al. (Latent Space Oddity, ICLR 2018) — original pullback-metric idea for VAEs. Detlefsen et al. 2020 — categorical extension. Arvanitidis et al. 2019 — shortest paths on learned manifolds.
- **Our substrate:** `katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md` (Plan 301 ships `jacobian_svd_at`); `katgpt-rs/.research/290_latent_field_steering_open_primitive.md` (complement); `katgpt-rs/.research/276_Personality_Weighted_Latent_Layer_Composition.md` (sigmoid-blend kernel shape); `katgpt-rs/.plans/252_cubical_category_interval_topology.md` (raw-space geodesic cousin); `katgpt-rs/.plans/266_densemesh_latent_node_network.md` (composition-graph precedent, Gate 2 failed); `riir-ai/.research/153_latent_field_steering_game_runtime_guide.md` (template for the private guide).
- **→ riir-train redirect:** VAE training, hierarchical categorical decoder, ELBO, Adam schedule, Baumgarten A* Mario viability agent. Not distilled here.

---

## Phase 5 result (2026-06-24)

**PROMOTED to default-on.** All gates closed:

- **G1–G7 correctness:** all PASS (10/10 unit tests).
- **Perf bench — `manifold_random_walk/k=4` per-step:** 485.58 ns/step → **7.10 ns/step** post-CSR (68.4× speedup, 14× under the 100 ns/step target).
- **Fix:** CSR (Compressed Sparse Row) adjacency added to `SafeManifoldGraph`; `for_each_neighbor` went from O(E) linear edge-list scan to O(degree) direct index. ~30 lines, no public-API change. Per-node neighbor order preserved byte-for-byte (`[< node ascending] ++ [> node ascending]`, matching the pre-CSR linear-scan emission order) so `manifold_random_walk` output is identical for any fixed RNG seed.
- **`pullback_volume`:** 310 ns (PASS, 16× under 5 µs).
- **`build_safe_manifold_graph`:** 384.60 µs (PASS, 26× under 10 ms; CSR build cost is +4.9%, negligible).

Open follow-up (not a blocker): Phase 6 riir-ai wiring on real 8D HLA (R154 G8–G12) lives in a separate plan. If free Gaussian walks in `R^8` HLA already keep playability ~99% (HLA is well-behaved), this primitive is Gain not Super-GOAT — demote honestly at that point.

📖 Full GOAT record: [`katgpt-rs/.benchmarks/312_viable_manifold_graph_goat.md`](../.benchmarks/312_viable_manifold_graph_goat.md).
