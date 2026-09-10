# Research 544: Programmable Cellular Automata — DEC as the Global-Function Layer

**Status:** SUPER-GOAT (fusion, modelless track) — open primitive spec filed (`katgpt-rs/.plans/591_pca_dec_global_function_layer.md`); game application gated on riir-ai Proposal 012 revival (`.issues/911_pca_global_functions_unblock_living_dungeon.md`). LLM-evolution track: audited discard (see §3). Training track: vacuous (see §3).

**Date:** 2026-09-10
**Paper:** Khalifa, Nasir, Siper, James, Togelius, *Programmable Cellular Automata* — [arXiv:2609.06102](https://arxiv.org/abs/2609.06102)
**Closest cousins:** Research 404 (Cells2Pixels NCA) → riir-ai Proposal 012 (Living Dungeon, shelved) · `katgpt-dec` operators · Plan 305 `zone_gating`

---

## TL;DR

The paper modularizes cellular automata into **local functions** (3×3 Moore neighborhood → value), optional **global functions** (whole-grid state → scalar), and a **decision function** (composition → next cell value), all as human-readable programs, evolved by LLM-as-mutation genetic programming for PCG-Benchmark level generation. Two measured findings matter more than the method: (1) **purely-local CA cannot solve some domains at all** (Zelda: 0% playability with zero global functions — counting/connectivity by local propagation needs O(board) iterations), and (2) **one global function collapses the iteration count** (Sokoban ~98 → ~19 iterations; Zelda 0% → 99% playability). The LLM evolution then *rediscovered* a universal global-function set: counting, connectivity, perimeter, distribution.

**The distillate is the finding, not the method.** The search loop exists to discover the global-function layer; the discovered set maps 1:1 onto DEC operators this stack already ships closed-form — `boundary_flux_mass` (count/perimeter), `betti_numbers` b0 + `hodge_decompose` (connectivity), `codifferential` (clustering). The paper is therefore independent empirical validation that (a) the global-function layer is the load-bearing piece of any grid-automaton generator and (b) the universal function set is exactly our Stokes/DEC operator family. Fusion verdict: **Super-GOAT** — wire shipped DEC aggregates into the decision function of a modular CA (`stochastic_birth_death_step` is the ready local kernel), which un-blocks the shelved Living-Dungeon lane (Proposal 012) whose #1 honest risk was exactly this iteration/quality cost.

---

## 1. Paper Core Findings

- **PCA formulation:** chromosome = array of `l` local functions + `g` global functions + 1 decision function. Local: neighborhood → value. Global: whole state → value. Decision: takes *only* the function outputs (not raw state) → next cell value. Synchronous CA retained; **global functions run Asynchronous** (their writes are visible mid-pass — §3's counter example: a sync global makes every cell see count=0 and over-place; async fixes the dynamics, not just the cost).
- **Evolution:** population 40, 50 generations, tournament-7, uniform crossover, LLM (Claude Opus, 4096-token cap) as the mutation operator (random + context sampling). Fitness = quality with diversity as cascaded tiebreaker (Eq. 1–3).
- **Result tables:** Zelda with g=0 is **0% playable across every l** — pure-local cannot deliver "≥1 player, ≥1 key, ≥1 door, ≥3 enemies, path ≥18" within 100 iterations. With g=1: 99%. Iterations-to-best drop ~3–8× (Sokoban 98.4 → 19.1 at l=1,g=1; Binary 13.6 → 15.6 is the noisy exception where l=1,g=0 already suffices).
- **Discovered globals (Table 4):** Counting (61%/61%/43% frequency), Connectivity/largest-component (17/4/19%), Border Detector, Spread (bounding-box perimeter), Row Distribution, Spatial Distribution, Clustering, Distance (max Manhattan pair), **Perimeter** (edges touching inactive/boundary), Run-Length.
- **Discovered locals (Table 5):** neighborhood count (100%/85%/66%), Alone (isolated center), Orientation (h/v alignment), Encoding, Vertical gradient, Weighting (orthogonal 20 / other 7 / center 1), Pattern (corner-vs-edge hash), Corner, Hashing, Contrast.
- **Their own open questions:** visible/hidden state split (hidden channels as memory); richer global returns (e.g. a Dijkstra map) with the stated cheat risk ("the global function can cheat and return the final values, and the decision function just copies it").

## 2. Distillation

### 2.1 Vocabulary crosswalk — the evolved functions ARE DEC operators in Python

The strongest single observation of this distillation: the LLM search rediscovered our operator vocabulary without knowing it. Paper-vocabulary-only grep returns zero hits; operator names are the defense (R296 lesson, working in reverse).

| Evolved function (paper T4/T5) | Frequency (B/Z/S) | Shipped equivalent (`katgpt-dec`, re-exported `katgpt_core::dec`) | Delta |
|---|---|---|---|
| **Counting Function** (global, tiles of a type) | 61/61/43% | `boundary_flux_mass` / `belief_mass_divergence` — mass of the tile-indicator cochain (Stokes: region mass = boundary flux, `stokes_calculus.rs` L127/L64) | closed-form, O(n^{(d−1)/d}) vs evolved 5±3-line Python |
| **Perimeter Function** (global, edges touching inactive/boundary) | 6/0/4% | `boundary_flux_mass` **literally** — the boundary measure of the tile set | identical quantity |
| **Connectivity Function** (global, largest connected area) | 17/4/19% | `betti_numbers → [usize;4]` (b0 = component count, `hodge.rs` L404) + `hodge_decompose` harmonic dimension | b0 gives component *count*, not max-*size*; largest-size is the one genuine extension (§7) |
| **Clustering Function** (global, density of clustering) | 3/0/5% | `codifferential` δ — divergence of the indicator cochain (AGENTS.md: "curiosity = positive divergence") | identical semantics |
| **Border / Spatial Distribution** | 0–10% | `exterior_derivative` on boundary 0-cochain; centroid moments | partial |
| **Spread / Row Distribution / Distance** | 2–11% | `line_integral` + first moments of dω | partial |
| Counting (local, 3×3) | 100/85/66% | the `graph_laplacian` 7-point stencil count term (`birth_death.rs`) | identical |
| **Weighting Function** (20/7/1) | 3/6/31% | the 7-point stencil's orth/diag/center weights | the LLM re-derived our stencil weights |
| **Alone Function** | 3/24/46% | Δx == 0 (isolated-vertex test) | identical |
| Orientation / Vertical / Contrast | 14–36% | directional `exterior_derivative`; anisotropic Laplacian difference | identical family |

### 2.2 Closest cousins (3)

1. **Research 404 / `birth_death.rs` (Plan 454)** — modelless 3D NCA: `stochastic_birth_death_step` over the 7-point stencil, sigmoid alive gate, crowding death, `argmax_block_type` voxel classes, **explicit caveat "no such consumer exists today"**. This is the local half + decision kernel, awaiting exactly what this paper supplies: a reason (content generation) and a global layer.
2. **riir-ai Proposal 012 (Living Dungeon, REJECTED/shelved 2026-07-10, no PoC)** — NCA×dungeon generation: grow-from-seed, regenerate-after-destruction, morph-on-operator-swap. Honest risk #1: "is DEC-growth a *good* dungeon rule, or mush? Cannot claim without running it." This paper's tables are measured evidence that the missing ingredient was never a better local rule — it was the global-function layer.
3. **Plan 305 `zone_gating` + swarm aggregates** — global scalar (`I_d`, `ripe_per_band`, `apple_centroid_direction`) → local compute parameters/steering/hibernation. The decision-shape (global reads gate local work) already ships and is GOAT'd; what never shipped is feeding those globals into a *per-cell content update rule*.

### 2.3 What is genuinely unshipped

- **The composition:** a global aggregate wired INTO the per-cell decision function to collapse iterations. Nothing in the stack does this (grep: zero hits for global→cell-update wiring; `zone_gating` gates *compute budgets*, not cell states; birth-death consumes only local terms + dropout).
- **Async global feedback with deterministic traversal:** fixed row-major scan + O(1) incremental counter update = deterministic (matches the swarm's `ripe_per_band` incremental pattern). The sync variant over-places ("all cells see stale count") — the paper's §3 example is a real dynamics difference, not a perf nicety.
- **CA-driven content generation at all:** shipped generation is `katgpt-pruners/map_generator.rs` (random-walk + BFS acceptance) and factory maps. No CA lane.
- **Largest-connected-component size** (b0 gives count only) — the one crosswalk row needing new math (max-component size from the harmonic projector's support, or a committed union-find pass).

### 2.4 Fusion

**paper (global-function layer is load-bearing, measured) × `birth_death` local kernel × DEC closed-form globals × Proposal 012's sellable surface** = a content generator whose growth rule reads *committed topological measurements* (b0, boundary mass, divergence) instead of propagating rumors cell-to-cell:

- **Zero-search delivery:** the paper needs an LLM+GA loop to find count/connectivity; we ship them as operators. Their Table 4 is our function library, discovered independently — third-party validation of the DEC moat.
- **The cheat channel is structurally bounded (answers the paper's own open question):** a global function "can cheat and return the final values" — but a *topological scalar cannot*. b0 ∈ {small int}, flux mass is one f32, divergence is one f32. The information-theoretic bandwidth of the DEC global layer is tiny by construction, so the decision function must do the generation work locally. Closed-form DEC globals are a *regularizer* the evolved-Python version is not.
- **d ≤ 3 guard:** boundary-vs-volume wins only for d ≤ 3 (AGENTS.md manifold rule) — 2D tile maps and 3D voxels are exactly the d ≤ 3 regime; never advertise this layer for high-dim shard fields.
- **Freeze/thaw attractor:** converged grid = `MerkleFrozenEnvelope` snapshot; morph = operator hot-swap; regenerate = thaw + re-evolve with the globals re-measured. Proposal 012's modelless unblock, now with the iteration cost measured by an external group.

## 3. Per-track verdicts (§1.5 — one verdict per track)

- **Track (a) modelless inference — the PCA×DEC fusion: SUPER-GOAT** (gate in §5). Outputs filed this session: open-primitive plan (katgpt-rs 591), game-application issue (riir-ai 911), Proposal 012 addendum. No separate private guide is written: **Proposal 012 already is the selling-point guide for this exact surface** (selling point ¶, force-multiplier map, modelless unblock §) — duplicating it would violate the no-parallel-substrate rule; its addendum carries the new evidence.
- **Track (b) LLM-as-mutation evolutionary search: audited discard.** The search layer's entire job in this paper is *discovering* the global functions; this stack already ships the discovered set closed-form (§2.1), so the search has no residual job here — and the landscape is crowded (FunSearch → AlphaEvolve → ShinkaEvolve/OpenEvolve/CodeEvolve commoditizing it). If CA-rule search is ever wanted anyway (e.g. discovering *local* kernels beyond the stencil family), the nearest seams are `katgpt-ruliology/mutation.rs` (`FsmTemplateProposer`, delta-gated co-evolve) and riir-clippy's mining loop — recorded, no plan.
- **Track (c) model-based/training: vacuous.** The paper contains no training loop — the LLM mutation operator is inference-time code generation (dev-time, like healer mining). No riir-train deferral to justify; the §3.5 adversarial panel is skipped per the workflow's clear-inference-side exemption, and pre-flight #5 found no training surface to mis-route (the only training-adjacent angle — training the generator LLM on CA-rule corpora — is not the paper's claim and has no consumer here).

## 4. Reframes (workflow §1 steps 3–4, both mandatory)

- **Latent-space reframe:** the global functions are latent-space aggregations of the state cochain — `belief_mass_divergence` already feeds the CGSP Stokes validator and zone scheduling (Plan 352). The new move is *feeding the aggregate back into the local update* — a one-directional bridge (global scalar → local gate), same shape as the two-brain bridge (info brain = cell states, think brain = global measurements), and like it, must stay one-way per iteration to keep determinism.
- **Game-context reframe:** per-tick, a zone's cave/dungeon layer grows via local kernels while reading zone-level scalars the game already commits (`ripe_per_band` counts, threat centroid, `I_d` density). Selling point: *"world content grows from seeds with provable connectivity (b0 == 1 checked in O(boundary)), in a fraction of the ticks, because the growth rule reads zone measurements instead of propagating rumors."* Converged layouts freeze into quorum-verifiable snapshots — content generation that satisfies the sync-boundary rule by construction (generated geometry is raw tile data; the globals are scalars).
- **Consumer/healer reframe (priority-#2 check, asked explicitly):** the healer already composes global + local — `select_best_candidate` blends global trajectory statistics (`W_EVO·evolution + W_RATE·reliability`) with local span match, and `RerankMode::Structural` is a global structural prior over local candidates. The paper validates that shape; it adds no primitive the healer lacks. The AST chunker is not a lattice; no CA lane applies. Recorded, no action.

## 5. Novelty gate — pinned claim

**Pinned claim (§4 precondition):** *Closed-form DEC boundary aggregates (betti numbers, boundary-flux mass, codifferential) as the global-function layer of a modular cellular automaton for game content generation, consuming the CellComplex/lattice state, distinguished from arXiv:2609.06102's LLM-evolved Python globals by (i) zero-search closed-form delivery — the paper's own Table 4 discoveries map 1:1 onto shipped operators, (ii) sublinear O(n^{(d−1)/d}) evaluation for d ≤ 3, and (iii) an information-bounded cheat channel (topological scalars cannot leak the solution).*

- **Q1 No prior art?** YES, scoped to the claim. Web sweep: local+global hybrid CA is NOT novel (Kaneko globally coupled maps 1990; Extended Cellular Automata arXiv:2502.17065 proves a formal global-rule extension; Mitchell's density-classification is the canonical pure-local hardness result; Empowered NCA arXiv:2205.06771 adds a signaling channel for the same propagation cost), the term "programmable cellular automata" is not a first coin (VLSI/BIST literature since ~1988, CAM-6 1987 — first use *in PCG* is this paper), LLM-as-mutation is crowded. **DEC-for-procgen: no published work found** (DEC is standard in geometry processing/graphics; procgen uses flood-fill/graph grammar, never Betti/Hodge vocabulary). Codebase sweep: composition unshipped, CA content-gen unshipped (Proposal 012 proposed then shelved with zero PoC). The claim is scoped to the DEC instantiation + the wiring, never to the hybrid idea.
- **Q2 New behavior class?** YES — generation-time topological guarantees + iteration collapse is a capability no shipped generator here has (`map_generator.rs` checks connectivity *post hoc* with BFS; this makes the global measurement *causal* in the generation loop).
- **Q3 Product selling point?** YES — the sentence in §4; strengthens the Living-Dungeon surface (Proposal 012's ¶) with measured external evidence.
- **Q4 Force multiplier?** YES — DEC substrate × `birth_death` NCA × `zone_gating` decision shape × freeze/thaw attractor × Proposal 012's game surface; ≥2 pillars.

**All 4 YES → Super-GOAT.** No "candidate" hedge.

## 6. MOAT gate per domain

| Domain | Verdict |
|---|---|
| `katgpt-rs` | **Ships here** — DEC/Stokes is in-scope; open primitive (generic: cell complex + local kernels + DEC global reads + decision composer, no game semantics), opt-in behind a feature flag until the GOAT gate passes; slot = DEC content-dynamics layer alongside birth_death/motor_gated. |
| `riir-ai` | **Pillar-adjacent** (content-generation lane). Proposal 012 addendum + Issue 911 record the unblock; revival is an owner call. Game semantics (dungeon tile vocabulary, monster/loot placement) stay here, never in katgpt-rs. |
| `riir-chain` | None — no new commitment surface (converged snapshots reuse the existing Merkle envelope pattern; nothing chain-specific). |
| `riir-neuron-db` | None — shards untouched; d≤3 guard explicitly excludes shard-dim fields. |
| `riir-train` | None — track (c) vacuous (§3). |
| `riir-clippy` | Validated-only (§4 consumer reframe); no actionable gap. |
| `riir-game-sdk` | None directly; consumers reach any future game wiring through the facade per usual. |

## 7. Open Primitive Spec (Plan 591 scope — `katgpt-dec`, feature `pca_global`)

- `PcaGlobalFn` — enum over closed-form globals: `Betti0`, `BoundaryFluxMass`, `BeliefMassDivergence`, `Codifferential`; evaluated per iteration (sync) via `DecCache`, or incrementally (async, deterministic fixed row-major traversal + O(1) counter update — the paper's dynamics, our swarm pattern).
- `PcaDecision` — consumes (local neighborhood values, global scalars) → next cell value; the birth-death sigmoid alive gate is the seed implementation with a global-termination term (e.g. stop placing when `count ≥ k` — the exact construct pure-local CA cannot express).
- `step_pca(...)` — the loop; `*_into` scratch variants, zero-alloc per tick; `debug_assert!(complex.dim() <= 3)` for every boundary-vs-volume path (curse-of-dim guard, AGENTS.md).
- **Largest-component size** (the one crosswalk gap): max-component cardinality alongside b0 — either harmonic-support extraction or a committed scan; decide during implementation, keep closed-form where possible.
- GOAT gate: **G1** correctness (final state connectivity verified by an *independent* `betti_numbers` call; same seed + traversal → bit-identical grid), **G2** perf (iterations-to-b0==1 ≥3× better than pure-local birth-death on a 64×64 tile map; paper suggests 3–8×), **G3** no-regression (flag-off path bit-identical to current `birth_death`), **G4** alloc-free per tick. Bench → `.benchmarks/` per numbering; promote to default only on pass, demote nothing (new slot).

## 8. What goes where (7-repo discipline)

- **katgpt-rs (public):** this note; Plan 591 primitive (generic DEC+CA composition — no tile vocabulary, no game semantics); bench.
- **riir-ai (private):** Issue 911 + Proposal 012 addendum (game surface: living dungeons, zone content); any future System impl wiring globals into `riir-games` content tick.
- **Consumers (seal-remake, riir-mmorpg-examples):** eventual content pipelines via the facade — not filed now.
- **Not chain / not neuron-db / not train / not clippy** (§6).

## 9. Constraints check

| Constraint | Status |
|---|---|
| Three-track system | ✅ all three classified per-track (§3); no repo characterized as modelless-only |
| Latent-to-latent, sigmoid not softmax | ✅ global scalars are latent aggregations; alive gate stays sigmoid (birth-death); softmax nowhere |
| Freeze/thaw over fine-tuning | ✅ converged attractor = Merkle frozen snapshot; no weight mutation anywhere |
| Sync boundary | ✅ generated tile geometry = raw committed data; globals cross as scalars; bridge one-way per iteration |
| Curse of dimensionality | ✅ d ≤ 3 guard in the spec — never advertised for shard-dim fields |
| Zero-alloc hot path | ✅ `*_into` scratch variants specified |
| Determinism | ✅ async feedback + fixed traversal order = deterministic; G1 asserts bit-identity |

## 10. References

- Khalifa, Nasir, Siper, James, Togelius, *Programmable Cellular Automata* — [arXiv:2609.06102](https://arxiv.org/abs/2609.06102)
- Prior-art context (web sweep 2026-09-10): Kaneko, *globally coupled maps* (1990) — the original local+global lattice coupling; *Extended Cellular Automata* [arXiv:2502.17065] — formal global-rule CA theory; Mitchell/Crutchfield — pure-local global-computation hardness (density classification); Earle et al., *Illuminating diverse neural cellular automata for level generation* (GECCO 2022); Siper et al. [arXiv:2608.17947]; FunSearch (Nature 2024) / AlphaEvolve [arXiv:2506.13131] / ShinkaEvolve (ICLR 2026) — the LLM-evolution lineage this note's track (b) discards
- Existing: Research 404 (Cells2Pixels NCA) · riir-ai Proposal 012 (Living Dungeon, shelved) · Plan 454 (`birth_death`) · Plan 357 (`motor_gated`) · Plan 305 (`zone_gating`, Bench 267 GOAT) · Plan 314 (`belief_mass_divergence`) · Plan 352 (zone scheduling) · `riir-games` swarm aggregates (`ripe_per_band`, `apple_centroid_direction`, crowd threat centroid)
