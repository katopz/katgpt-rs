# Research 404: Cells2Pixels — Resolution-Decoupled Cellular Automata (Continuous Cochain Sampling)

> **Source:** Pajouheshgar, Xu, Abbasi, Mordvintsev, Jakob, Süsstrunk. *Neural Cellular Automata: From Cells to Pixels*. SIGGRAPH 2026. [arxiv:2506.22899](https://arxiv.org/abs/2506.22899) · code: [TheDevilWillBeBee/Cells2Pixels](https://github.com/TheDevilWillBeBee/Cells2Pixels)
> **Date:** 2026-07-10
> **Status:** Done
> **Related Research:** 291 (cross-resolution spectral transport), 359 (motor-gated DEC propagation), 365 (heat-kernel trajectory), 219 (DEC/TNO), 305 (phase-modulated cross-domain coupling)
> **Related Plans:** 310 (cross-resolution transport), 357 (motor-gated DEC), 413 (multi-scale V-cycle), 416 (region subspace field)
> **Classification:** Public

---

## TL;DR

The paper's headline contribution — a Neural Cellular Automaton evolving on a
**coarse** lattice paired with a lightweight **Local Pattern Producing Network
(LPPN)** decoder that renders the field at **arbitrary fine resolution** — is a
computer-graphics / texture-synthesis result whose *value* is in the joint
end-to-end **training** of the NCA rule + LPPN. That half → `riir-train` if
anyone wants the graphics capability; it is out of scope for this workflow.

**Distilled for katgpt-rs (modelless, inference-time):** the transferable
*inference* primitive is narrower than the paper's headline: **continuous
intra-primitive sampling of a cochain field conditioned on a local coordinate**.
Given a `CochainField` on a `CellComplex` (the coarse lattice) and a query point
`p` inside a cell (quad / triangle / tet / hex), compute
`(s̄(p), u(p)) → output`, where `s̄(p) = Σ λⱼ(p)·sⱼ` is the λ-weighted
locally-interpolated cell state and `u(p)` is a compact intra-cell local
coordinate. This turns the *discrete* cochain into a *continuous* field queryable
at any resolution, while the dynamics stay on the coarse lattice.

Both halves of the paper's coarse/fine split already ship in `katgpt-dec`:
coarse-grid local dynamics = `evolve_motor_gated_field` (Plan 357) + heat kernel
(Plan 359) + DEC operators (Plan 251); coarse↔fine *discrete* transfer =
`htno_v_cycle` restrict/prolongate (Plan 413) + `CrossResolutionTransport`
asymmetric basis projection (Plan 310). **The one genuinely unshipped piece is
the continuous intra-cell sampler with local-coordinate conditioning** — Plan
413's prolongate is a discrete vertex-to-vertex scatter; it cannot answer "what
is the threat value at continuous point (3.7, 5.2) inside this quad?"

**Verdict: Gain.** A clean, provable, narrow geometric primitive (continuous
cochain sampling) that amplifies the DEC substrate's existing consumers (motor-
gated field, heat-kernel trajectory, terrain cochains, zone geometry) by making
their outputs continuously queryable at arbitrary LOD. Not a new capability class
and not a pillar — Q3 (product selling point) is marginal ("smooth fields at any
LOD" is a quality knob, not a headline), which blocks Super-GOAT.

---

## 1. Paper Core Findings

### 1.1 The NCA + LPPN hybrid

An NCA evolves cell states `sᵢᵗ ∈ ℝᶜ` on a lattice via a shared local update rule:

```
sᵢ^{t+Δt} = sᵢᵗ + 𝒜(𝒵(sᵢᵗ, sⱼᵗ))·Δt,   j ∈ 𝒩(i)
```

where `𝒵` is local perception (convolution-like neighborhood aggregation) and
`𝒜` is an MLP adaptation function. This is, structurally, a cellular automaton
on a graph/grid — and it is structurally identical to one Amari-style tick of
our `evolve_motor_gated_field` on a `CellComplex` (Research 359 / Plan 357).

The paper's contribution is **decoupling dynamics from appearance**: evolve the
NCA on a coarse lattice (e.g. 128×128), then render at arbitrary resolution
(e.g. 8192×8192) via an LPPN — a lightweight coordinate-based MLP decoder:

```
o(p) = f_θ( s̄(p), u_aug(p) ) ∈ ℝᴷ
```

where:
- `s̄(p) = Σⱼ∈Ω λⱼ(p)·sⱼ` — λ-weighted interpolation of the cell states in the
  primitive Ω enclosing `p` (λ = partition-of-unity, non-negative, linear-
  precision barycentric/bilinear/mean-value coordinates).
- `u(p)` — a compact *intra-primitive local coordinate* (Cartesian for
  quad/cube, sorted+remapped barycentric for triangle/polygon), rescaled to
  `[-1,1]`.
- `u_aug` — a continuity-enforcing encoding: `sin/cos` harmonic basis for
  Cartesian (C⁰ across primitive boundaries), or sort+CDF-remap for barycentric.

The LPPN adds ~25% parameters; recurrent NCA updates all run coarse; inference
stays real-time and parallelizable because both NCA and LPPN are local.

### 1.2 Why NCAs were stuck (the three problems the paper solves)

1. Training time/memory grow **quadratically** with grid size.
2. Information propagates **locally** (one neighborhood hop per step) → slow
   long-range coordination on large grids.
3. Real-time inference at high resolution is **compute-heavy**.

Their fix (coarse dynamics + local fine decoder) addresses all three: the
expensive recurrence stays small; the decoder is local + weight-shared; render
resolution is a runtime compute-quality knob.

### 1.3 Properties preserved (the self-organization story)

- **Regeneration**: after severe damage, the system re-grows missing regions and
  returns to the same attractor.
- **Parameter-swap morphing**: swapping model parameters at test time produces a
  gradual morph between converged morphologies — an *emergent* attractor
  transition, not explicitly optimized.
- **Resolution invariance**: a model trained at one lattice size can be evaluated
  at arbitrary larger lattices (texture expansion, Appendix E).

### 1.4 Loss functions (training-side, → riir-train)

Patch-based multi-scale OT texture loss, pseudo-targets for PBR cross-map
alignment, FFT auto-correlation regularizer for long-range structure, morphology
loss (RGBA recon + living-mask shape loss + LPIPS). All training-only.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase)

| Paper term | Codebase equivalent | Shipped? |
|---|---|---|
| "cell" / "cellular automaton" / "lattice" | `CellComplex` vertex + `CochainField`; `grid_2d` | ✅ DEC substrate (Plan 251) |
| "cell state `sᵢ`" | `CochainField` value at a vertex; HLA 8-dim per-NPC state | ✅ |
| "local update rule" `𝒵 + 𝒜` | `evolve_motor_gated_field` (Plan 357); `hodge_laplacian` tick; heat-kernel roll-forward (Plan 359) | ✅ the coarse dynamics |
| "neighborhood" `𝒩(i)` | cell-complex adjacency; coboundary / stencil | ✅ |
| "primitive" (quad/tri/tet/hex) | `CellComplex` cell shape | ✅ (topology); ❌ (no continuous sampler) |
| "λ-coordinate" / "barycentric" / "partition of unity" | `FuncAttnBasis` partition-of-unity rows; `BSplineBasis` partition-of-unity; **no λ-coordinate sampler on cochains** | partial — concept ships elsewhere, not on DEC fields |
| "local coordinate `u(p)`" / "intra-primitive" | **unshipped** — no intra-cell coordinate concept in `katgpt-dec` | ❌ the gap |
| "LPPN decoder" `(s̄(p), u(p)) → output` | latent→raw scalar bridge (`project_to_scalars`, direction-vector projection); **no continuous-point cochain sampler** | ❌ the gap |
| "decouple dynamics from appearance" / "coarse grid + fine render" | `htno_v_cycle` restrict/prolongate (Plan 413, *discrete*); `CrossResolutionTransport` asymmetric basis (Plan 310, *basis-dim*); `HtnoMultiScale` V-cycle (riir-train, nearest-neighbor) | ✅ at discrete/basis level; ❌ at continuous-point level |
| "compute-quality knob" (render scale) | plasma/hot/warm/cold tiering; thermal LOD | ✅ conceptually |
| "regeneration after damage" | MAPE-K self-healing loop (riir-neuron-db) | ✅ (shard domain) |
| "parameter-swap morphing" | freeze/thaw hot-swap; committed personality blend | ✅ |
| "attractor" | frozen snapshot; `MerkleFrozenEnvelope`; `KarcShard` | ✅ |
| "living mask" / "growth from seed" | fog-of-war visibility gate; spectral lottery-ticket init | ✅ conceptually |
| NCA rule **training** + LPPN **training** | — | → riir-train (out of scope) |

### 2.2 Closest cousins (3)

1. **Plan 413 (`htno_v_cycle`)** — closest. Ships discrete coarse↔fine V-cycle on
   cell complexes: `restrict` (fine→coarse selector gather) + `prolongate`
   (coarse→fine adjoint scatter). This is the discrete analog of the paper's
   coarse/fine split. **What it does NOT cover:** continuous-point sampling
   inside a primitive with local-coordinate conditioning — prolongate scatters to
   fine *vertices*, not to arbitrary continuous `p`.
2. **Plan 310 (`CrossResolutionTransport`)** — asymmetric basis projection
   (`d_src ≠ d_dst`) for train-on-small-deploy-on-large shards. This is
   resolution decoupling at the *basis-dimension* level, not the *spatial-
   sampling* level. Research 305 §1.2 explicitly states "discretization
   decoupling … is exactly Research 291's cross-resolution transport thesis …
   we shipped that in Plan 310."
3. **Plan 357 (`evolve_motor_gated_field`)** — the coarse-dynamics half. An
   Amari-style neural-field tick on a cell complex, structurally identical to an
   NCA update. The grid-stencil fast path (Issue 001) is exactly the NCA-style
   local update pattern.

### 2.3 What is genuinely unshipped (the narrow gap)

```
sample_cochain_at_point(
    cx: &CellComplex,
    field: &CochainField,
    point: &[f32; D],        // continuous query location
    local_coord_encode: LocalCoordEncode,  // Cartesian-sincos | Barycentric-sort-cdf
) -> Vec<f32>                // interpolated state s̄(p)  (+ optional u_aug(p))
```

plus the local-coordinate framework:
- `λ_coordinate(p, primitive) -> Vec<f32>` — partition-of-unity weights
  (bilinear for quad, barycentric for tri, trilinear for hex).
- `local_coord(p, primitive) -> Vec<f32>` — compact `u(p) ∈ [-1,1]ᵈ`.
- `local_coord_aug(p, primitive) -> Vec<f32>` — C⁰-continuous encoding
  (`sin/cos` harmonic for Cartesian; sort+CDF-remap for barycentric).

This is the modelless LPPN *input* computation. The LPPN *decoder weights*
themselves (`f_θ`) are training-side (→ riir-train); the modelless analog is
that the caller supplies a frozen direction vector / projection (existing
`project_to_scalars` pattern) and the primitive only computes the continuous
`(s̄(p), u_aug(p))` conditioning.

### 2.4 Fusion

**F1 (PRIMARY — katgpt-dec): continuous sampler × existing DEC field consumers.**
`sample_cochain_at_point` makes the outputs of `evolve_motor_gated_field`,
`heat_kernel_trajectory`, and the terrain cochains (Safety/Threat/Occupancy/
Destruction) continuously queryable. Today these fields are read only at discrete
vertices; NPCs that query "threat at my exact continuous position" get the
nearest-cell value. With the sampler, they get a λ-interpolated + local-
coordinate-conditioned value — smoother spatial reasoning at arbitrary LOD.

**F2 (SECONDARY — katgpt-dec): continuous sampler × Plan 413 V-cycle.**
Compose `sample_cochain_at_point` with `htno_v_cycle`: evolve on the coarse
complex, prolongate to the fine complex, then sample the fine complex at
continuous points. This is the full Cells2Pixels pipeline (coarse dynamics →
fine discrete reconstruction → continuous render) at the cochain level —
modelless, no training.

**F3 (TERTIARY — speculative, riir-ai): continuous HLA field × zone geometry.**
If HLA per-NPC state is laid out as a cochain on a belief-manifold cell complex
(per Research 168 Integration 1), the continuous sampler lets a downstream
consumer query the affect field at any point in the zone — not just at NPC
positions. This is the "continuous emotional weather map" angle. Speculative;
depends on the HLA→cell-complex wiring landing first.

---

## 3. Latent-Space Reframing (mandatory per workflow §1.5 step 3)

How does the mechanism look operating on each latent-state kernel?

- **(a) HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`):
  each NPC = a "cell" in a social/manifold lattice; the 8-dim HLA state is the
  cell state. The sampler + a frozen direction vector decode
  `(s̄(p), u(p)) → fine behavior scalar` at any point in the zone — a continuous
  affect field, not just per-NPC values. Stays latent; only the 5 scalar
  projections cross sync (same boundary discipline as existing HLA).
- **(b) `latent_functor/`**: the sampler is a *continuous* functor application —
  instead of `apply_functor` at discrete cells, it's functor evaluation at
  continuous points. Composes with Plan 413 V-cycle for multi-resolution functor
  application.
- **(c) `cgsp_runtime/`**: curiosity signal becomes a continuous field — query
  curiosity at any point, not just at the NPC. Marginal value; curiosity is
  inherently per-agent.
- **(d) DEC Stokes operators**: the sampler is the **Whitney/de-Rham
  reconstruction operator** — the standard DEC map from a discrete cochain
  (differential form sampled at cells) to a continuous differential form. This
  is genuine DEC theory: cochains are discrete forms; Whitney forms reconstruct
  the continuous form via precisely the λ-basis the paper uses. The paper
  re-invents the Whitney form input as "locally interpolated cell state".
- **(e) NeuronShard `style_weights[64]`**: a frozen direction vector over the
  shard's style space is the LPPN analog; the sampler queries the personality at
  continuous "resolution". Marginal — shards are not spatial.
- **(f) DEC `hodge_decompose` / `DecFlowField`**: sampling the exact/coexact/
  harmonic flow channels at continuous points gives a continuous Helmholtz
  decomposition of the game-world flow. This is the strongest DEC-theory
  connection: the sampler is the continuous Hodge star.

**Assessment:** the latent reframing confirms the primitive is a DEC-substrate
amplifier (makes discrete cochains continuously queryable), not a new latent-
state kernel. The Whitney-form / continuous-Hodge-star framing is the deepest
statement; it lands squarely in `katgpt-dec` (the DEC substrate), confirming the
MOAT-gate routing below.

---

## 4. Novelty Gate (Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| Q1 — No prior art? | **Partial.** Coarse dynamics ✅ shipped (`evolve_motor_gated_field`, heat kernel). Discrete coarse↔fine ✅ shipped (`htno_v_cycle` Plan 413). Basis-dim resolution decoupling ✅ shipped (`CrossResolutionTransport` Plan 310). **Continuous intra-primitive cochain sampling with local-coordinate conditioning ❌ unshipped** — grep across `katgpt-dec/src/` for `sample_at_point\|bilinear\|trilinear\|barycentric\|local_coord\|intra_cell` returns zero relevant hits; Plan 413 prolongate is vertex-to-vertex scatter. The narrow gap is real. | vocabulary translation §2.1 |
| Q2 — New class of behavior? | **Yes-but-narrow.** Today, cochain fields are read only at discrete cells. Continuous-point queries are a new capability for the DEC substrate. But it is "make existing outputs continuously queryable", not a new behavior class — the field already exists; the sampler changes its *read resolution*. | §2.3, §3 |
| Q3 — Product selling point? | **Marginal.** "Our game-world fields (threat, safety, occupancy, emotion) are continuous — query them at any resolution, from plasma-tier coarse to cold-tier fine, while dynamics evolve on a fixed coarse lattice." Decent for spatial-AI LOD quality; NOT a headline selling point no competitor can match. Cannot finish the Super-GOAT sentence strongly. | §3 |
| Q4 — Force multiplier? | **Yes (≥5 systems):** DEC substrate, motor-gated field, heat-kernel trajectory, terrain cochains, zone geometry, Plan 413 V-cycle, thermal LOD. But the multiplier is "read resolution quality", not "new composition". | §2.4 |

**Q3 fails the Super-GOAT bar.** A marginal selling point that is a quality knob
on existing outputs does not clear "can you finish the sentence: our NPCs do X
that no competitor can". The honest verdict is below Super-GOAT.

---

## 5. Verdict + MOAT Gate per Domain

### Tier: **Gain**

| Tier | Criteria | Routing |
|---|---|---|
| ~~Super-GOAT~~ | ~~Novel mechanism + new capability class + selling point + ≥2 pillars~~ | Q3 (selling point) fails — "continuous fields at arbitrary LOD" is a quality knob, not a pillar-class moat. |
| ~~GOAT~~ | ~~Provable gain over existing approach, promotes to default if it wins~~ | Not a default-on candidate — continuous sampling is opt-in quality, not a correctness/perf win on the default path. |
| **Gain** | Incremental, useful, behind feature flag | **Plan only** — narrow open primitive in `katgpt-dec`, opt-in feature. |

**One-line reasoning:** the paper's value is its trained NCA+LPPN for texture
synthesis (→ riir-train); the modelless inference transfer is a narrow geometric
primitive (continuous cochain sampling with local coordinates) that is genuinely
unshipped but is a quality amplifier on existing DEC consumers, not a new
capability class or pillar.

### MOAT gate per domain — `katgpt-rs` (public engine)

- **In scope?** Yes — "DEC/Stokes substrate" is explicitly in the `katgpt-rs`
  domain table. Continuous cochain sampling is fundamental DEC math (Whitney/
  de-Rham reconstruction), the kind of substrate primitive that belongs in the
  public engine.
- **Strengthens the engine moat?** Marginally — fills a read-resolution gap in
  the DEC substrate that every DEC consumer benefits from. Neutral-to-positive
  for the adoption funnel (the DEC crate becomes more complete).
- **Routing:** open primitive → `katgpt-dec` (the DEC subcrate of katgpt-core).
  No private guide needed (Gain, not Super-GOAT). No riir-ai/riir-chain/riir-
  neuron-db guide — the selling point does not clear the pillar bar in any
  private repo.

### §3.6 PoC consideration

No parity claim is made. The modelless analog (continuous cochain sampling via
frozen λ-basis + local-coordinate conditioning) is **architecturally distinct**
from the paper's trained LPPN (a learned SIREN MLP). We do NOT claim "matches
the paper's image quality" — that would require a defend-wrong PoC in
`riir-poc/`. The Gain verdict rests on the *capability* (continuous queries
where only discrete existed), which is provable by construction (λ-basis is
exact for linear fields by the linear-precision property), not on a quality
comparison with the paper's texture outputs.

---

## 6. Open Primitive Spec (Plan scope)

A narrow plan (`katgpt-rs/.plans/422_*`) for `katgpt-dec`:

- `sample_cochain_at_point(cx, field, point, encode) -> Vec<f32>` — λ-weighted
  interpolated state `s̄(p)`.
- `local_coordinate(point, primitive) -> Vec<f32>` — compact `u(p) ∈ [-1,1]ᵈ`.
- `local_coordinate_aug(point, primitive, n_harmonics) -> Vec<f32>` — C⁰ encoding.
- `λ_coordinate(point, primitive) -> Vec<f32>` — partition-of-unity weights.
- Feature flag `cochain_point_sampler` in `katgpt-dec`, opt-in.
- GOAT gate G1 (linear-precision exactness on linear fields), G2 (partition-of-
  unity Σλⱼ=1), G3 (C⁰ continuity across primitive boundaries for the aug
  encoding), G4 (zero-alloc `*_into` variants), G5 (sub-µs per query on a grid).

**Primitive types to support initially:** quad (2D grid, bilinear + Cartesian
sincos), triangle (mesh, barycentric sort+CDF-remap per paper Appendix B). Hex
and tet deferred (the `grid_2d` fast path in `CellComplex` makes quad the
default; triangle covers mesh consumers).

---

## 7. What goes where (5-repo discipline)

| Component | Repo | Why |
|---|---|---|
| `sample_cochain_at_point` + local-coordinate framework | `katgpt-rs` (`katgpt-dec`) | Generic DEC math (Whitney/de-Rham reconstruction). No game/chain/shard semantics. |
| Wiring into motor-gated field / heat kernel / terrain cochains | `riir-ai` (consumer) | Game-runtime spatial reasoning. Optional follow-up, not part of this note. |
| Trained NCA rule + trained LPPN for texture synthesis | `riir-train` | Out of scope for this workflow. Note only: if a game ever needs procedural texture/morphogenesis, the paper's training recipe lives there. |
| Continuous HLA affect field (F3 speculative) | `riir-ai` (speculative) | Depends on HLA→cell-complex wiring (Research 168). Not started. |

---

## 8. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Sampler is pure geometric interpolation (λ-basis + coordinate encode); no training, no gradients. The LPPN *decoder weights* are training-side and out of scope. |
| Latent-to-latent preferred | ✅ Sampler operates on the latent cochain; only the caller's frozen direction projection (existing pattern) would cross to raw scalars. |
| Use sigmoid not softmax | ✅ N/A — λ-coordinates are a partition-of-unity by linear-precision, not a softmax. No gating involved. |
| Freeze/thaw over fine-tuning | ✅ N/A — sampler is parameter-free geometry. Any decoder direction vectors would be frozen artifacts (existing pattern). |
| 5-repo discipline | ✅ Open primitive → katgpt-dec; no private guide (Gain, not Super-GOAT). |
| Raw scalars at sync boundary | ✅ Sampler output is latent (interpolated cochain value); only caller's scalar projections cross sync, same as existing HLA bridge. |
| Zero-alloc hot path | ✅ `*_into` variants write into caller-supplied scratch (same pattern as `FuncAttnScratch`, `VCycleScratch`). |

---

## 9. References

- Pajouheshgar et al., *Neural Cellular Automata: From Cells to Pixels*, SIGGRAPH 2026 — [arxiv:2506.22899](https://arxiv.org/abs/2506.22899)
- Mordvintsev et al., *Growing Neural Cellular Automata*, Distill 2020 — the original NCA
- Stanley, *Compositional Pattern Producing Networks* (the LPPN namesake), 2007
- Existing: Plan 251 (DEC operators), Plan 310 (cross-resolution transport), Plan 357 (motor-gated DEC), Plan 359 (heat-kernel trajectory), Plan 413 (multi-scale V-cycle), Plan 416 (region subspace field), Research 219 (TNO/DEC), Research 291 (cross-resolution), Research 305 (phase-modulated coupling), Research 359 (motor-gated DEC propagation), Research 365 (PhysiFormer heat kernel)
- Follow-up: Research 544 (Programmable Cellular Automata, arXiv:2609.06102) — measured that the global-function layer (count/connectivity = our `boundary_flux_mass`/`betti_numbers`) is the load-bearing ingredient CA generation was missing; Proposal 012 revival candidate

---

## TL;DR

Cells2Pixels (SIGGRAPH 2026) pairs a coarse-grid Neural Cellular Automaton with
a lightweight Local Pattern Producing Network decoder to render self-organizing
patterns at arbitrary resolution. The paper's value is its **trained** NCA+LPPN
for texture synthesis (→ riir-train, out of scope). The modelless inference
transfer is narrow: **continuous intra-primitive sampling of a cochain field
conditioned on a local coordinate** — the one piece not already covered by
`evolve_motor_gated_field` (coarse dynamics), `htno_v_cycle` (discrete coarse↔
fine), and `CrossResolutionTransport` (basis-dim resolution decoupling).
**Verdict: Gain** — a useful, provable, narrow geometric primitive (the
Whitney/de-Rham continuous reconstruction for discrete cochains) that amplifies
the DEC substrate's existing consumers by making their outputs continuously
queryable at arbitrary LOD. Q3 (selling point) fails Super-GOAT — "continuous
fields at any LOD" is a quality knob, not a pillar. Short plan for
`katgpt-dec`/`cochain_point_sampler`; no private guide (Gain, not Super-GOAT).
No parity claim with the paper's trained texture outputs (§3.6).
