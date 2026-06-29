# Research 324: Trajectory Geometry of Transformer Representations Across Layers

> **Source:** [Trajectory Geometry of Transformer Representations Across Layers](https://arxiv.org/abs/2606.09287) — Pandey, Singh, Mahdid (Jun 2026, arXiv:2606.09287v2)
> **Date:** 2026-06-29
> **Status:** Done
> **Related Research:** 219 (DEC), 242 (Topological State Tracking / micro_belief), 286 (Depth-Invariance), 296 (Stokes Calculus)
> **Related Plans:** 251 (DEC operators), 276 (MicroRecurrentBeliefState / AttractorKernel), 301 (SubspacePhaseGate), 303 (latent_functor reestimation), 314 (Stokes wrappers), 334 (Stokes validator)
> **Classification:** Public

---

## TL;DR

Observational interpretability paper that treats the transformer forward pass as a discrete population trajectory through a high-dimensional representation manifold and computes 5 probe-free geometric metrics over per-layer hidden states (trajectory length, curvature, semantic convergence index, layerwise cosine similarity, representational stability). Reports four empirical findings: (1) semantic categories converge into attractor-like basins in middle-to-late layers (CI 0.41–0.58, p<0.001), (2) trajectory curvature encodes computational complexity (reasoning 0.71–0.83 rad vs lexical 0.27–0.31 rad), (3) ambiguous tokens undergo progressive trajectory bifurcation beginning at ~20–25% of network depth (5.6× separation), (4) a universal three-phase computational structure (encoding / elaboration / output preparation).

**Verdict: Gain.** The paper is descriptive (observational, no causal interventions, no training). The five geometric metrics are basic linear-algebra primitives. Every substrate the metrics describe — attractor basins, divergence-as-signal, phase transitions, coherence-driven re-estimation — already ships in this codebase under operational (not interpretive) form: `AttractorKernel` (Plan 276), `subspace_phase_gate` (Plan 301), `belief_mass_divergence` (Plan 314), `latent_functor/reestimation.rs` (Plan 303), `entity_cognition/species_transition.rs`. The single transferable *primitive* is a small zero-alloc diagnostic struct wrapping trajectory-length + turning-angle + cosine-drop for an arbitrary sequence of latent states, useful as an opt-in probe for crowd-NPC coherence auditing and as a difficulty hint for breakeven-complexity routing. Not a new capability class; not Super-GOAT.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable insight is the **probe-free diagnostic trio**: trajectory length, mean turning-angle curvature, and adjacent-step cosine similarity, computed on any sequence of latent vectors (HLA evolution, functor applications, consolidation ticks, even per-layer transformer hidden states). The paper's hypothesis — "curvature as a probe-free complexity readout" — is exactly the kind of zero-supervision signal this codebase already uses elsewhere (`belief_mass_divergence` → `divergence_to_curiosity` sigmoid gate in `cgsp_runtime/stokes_validator.rs`). A named, reusable `LatentTrajectoryGeometry` struct turns that pattern into a generic open primitive.

---

## 1. Paper Core Findings

### 1.1 Framework
- Treats layer sequence `(h⁽⁰⁾, h⁽¹⁾, …, h⁽ᴸ⁾)` of mean-pooled hidden states as a discrete trajectory in `R^d`.
- All metrics computed in the **full ambient space** `R^d`; PCA/UMAP/t-SNE used only for visualization.
- Validated on GPT-2 Small (12 layers), TinyLlama-1.1B (22), Qwen2.5-1.5B (28); 150 prompts across 5 families.
- Four controls: random labels (C1), random embeddings (C2), shuffled layer order (C3), multiple projections (C4). All effects vanish under C1–C3.

### 1.2 The five metrics
| Metric | Definition | Paper's claim |
|---|---|---|
| Trajectory length `L(τ)` | `Σ ‖h⁽ˡ⁺¹⁾ − h⁽ˡ⁾‖₂` | More computation → longer path. |
| Curvature `κ(l)` | `arccos( v⁽ˡ⁾·v⁽ˡ⁺¹⁾ / (‖v⁽ˡ⁾‖·‖v⁽ˡ⁺¹⁾‖) )` where `v⁽ˡ⁾ = h⁽ˡ⁾ − h⁽ˡ⁻¹⁾` | Mean curvature `κ̄` is a probe-free complexity readout. |
| Convergence Index `CI(l)` | `D_between(l) − D_within(l)` over semantic category | Attractor-like clustering in middle-to-late layers. |
| Layerwise cosine similarity `SIM(l)` | `cos(h⁽ˡ⁾, h⁽ˡ⁺¹⁾)` | Sharp drops = phase transitions. |
| Representational stability `STAB(l)` | `cos(h_p⁽ˡ⁾, h_p′⁽ˡ⁾)` for lexical perturbation `p′` | Surface-form abstraction by layer `l`. |

### 1.3 Findings
1. **Attractor basins**: peak CI 0.41–0.58 in middle-to-late layers, collapses to ≈0 under random labels.
2. **Curvature ∝ complexity**: F4 reasoning 0.78 rad (GPT-2) vs F2 lexical 0.31 rad, d > 1.8 across all models.
3. **Bifurcation as disambiguation**: ambiguous-token pair separation rises monotonically from ~22% depth, 5.6× by final layer; absent in unambiguous controls.
4. **Universal three-phase structure**: encoding (`l ≤ L/4`), elaboration (`L/4 < l ≤ 3L/4`), output preparation (`l > 3L/4`) — boundaries consistent across all 3 architectures; collapses to flat high-similarity under random embeddings.
5. **Curvature peaks in the "computational inflection zone"** at 20–45% of network depth — corresponds to known induction-head and MLP-knowledge-retrieval layer ranges.

### 1.4 Limitations (paper-stated)
- Only decoder-only models, ≤ 1.5B parameters, English-only, 150 prompts.
- **Observational only** — no causal claims. No activation patching.
- Mean-pooling across tokens (sequence-level), not token-level.
- Coordinate-dependent geometry; paper explicitly suggests persistent homology / Betti numbers as a future coordinate-free extension.

---

## 2. Distillation

### 2.1 Why this is Gain, not GOAT/Super-GOAT

**Novelty gate:**
1. **No prior art?** NO. The substrate the metrics describe already ships:
   - Attractor basins → `katgpt-micro-belief::AttractorKernel` (Plan 276), `MicroRecurrentBeliefState`. Per-NPC attractor dynamics operational, not just observational.
   - Divergence-as-curiosity-signal → `katgpt-dec::belief_mass_divergence` (Plan 314) → `cgsp_runtime/stokes_validator.rs::divergence_to_curiosity` sigmoid gate (Plan 334). The paper's "curvature as probe-free readout" hypothesis is *already in production* as curiosity signal on the HLA emotion manifold.
   - Phase transitions → `subspace_phase_gate::phase_transition_gate` (Plan 301, `participation_ratio` + `numerical_rank` + `jacobian_svd_at`); `entity_cognition/species_transition.rs` (per-NPC Wildlife→Pet→NPC→Criminal phase transitions).
   - Coherence-driven re-estimation on drift → `latent_functor/reestimation.rs` (Plan 303, DiPOD pattern).
2. **New class of behavior?** NO. The paper is descriptive; it does not propose a new mechanism. A diagnostic primitive is incremental, not a new capability class.
3. **Product selling point?** Weak. "Our NPCs use trajectory curvature as a probe-free complexity readout" — true, but we already say this about divergence→curiosity. No new moat.
4. **Force multiplier?** Loose. Could connect to BreakevenComplexityRouter (Plan 250), SubspacePhaseGate (301), AttractorKernel (276), Stokes validator (334). But the fusion is "add one more diagnostic to existing routers" — Gain-tier.

**Verdict: Gain.** Plan only, opt-in feature flag, no Super-GOAT guide, no promotion to default unless a downstream fusion proves a measurable gate win.

### 2.2 The transferable primitive (open, generic, modelless)

A single zero-allocation diagnostic struct over an arbitrary sequence of `d`-dim latent vectors:

```text
LatentTrajectoryGeometry {
    length: f32,            // Σ ‖Δ‖₂        (metric 3.2 L(τ))
    mean_curvature: f32,    // mean arccos(v_l · v_{l+1})  (metric 3.2 κ̄)
    min_adjacent_cosine: f32, // min_l cos(h_l, h_{l+1})    (metric 3.2 SIM)
    n_steps: u16,
}
```

Two API surfaces:
- **Per-trajectory**: `from_states(states: &[&[f32]]) -> LatentTrajectoryGeometry`. Streaming fold, no allocation, O(L·d).
- **Pairwise bifurcation**: `bifurcation_ratio(a: &[&[f32]], b: &[&[f32]]) -> (f32, Option<u16>)` returning (final/initial separation ratio, onset-step index). Mirrors Finding 3.

Three semantic re-casts the paper itself suggests:
1. **Probe-free difficulty hint** (paper §6: "curvature could potentially flag inputs that require complex reasoning"). Feed `mean_curvature` into the existing `BreakevenComplexityRouter` (Plan 218/250) as an alternative signal to entropy.
2. **Crowd-NPC coherence audit**: per-NPC `LatentTrajectoryGeometry` over the HLA `evolve_hla` history across `K` ticks → flag NPCs whose emotion trajectory has anomalous length or curvature (e.g., ping-ponging between attractor basins).
3. **Phase boundary detector for staged functors**: `min_adjacent_cosine` step index localizes phase boundaries in latent-functor application sequences — the operational analog of the paper's three-phase structure on transformer layers.

### 2.3 Fusion

**Strongest fusion idea — Curvature as BreakevenComplexityRouter secondary signal (Plan 218/250 × Research 324):**

`BreakevenComplexityRouter` currently routes by entropy + structural complexity. Research 324's `mean_curvature` is an entropy-orthogonal signal (a trajectory can have low entropy per step but high turning-angle — e.g., tight oscillation). Fusing `mean_curvature > τ_curvature` as a *secondary* router arm could catch the "low-entropy-but-non-geodesic" failure mode that pure-entropy routers miss. **This is the only fusion worth a plan, and only behind a feature flag with a benchmark vs entropy-only.**

**Why not a Super-GOAT fusion:**
- The paper's other findings (attractor basins, three-phase structure) describe mechanisms we already *operate* (AttractorKernel, phase_transition_gate). Re-describing them as "trajectory geometry" is a vocabulary swap, not a new capability.
- The bifurcation finding (F3) is the most novel, but we already have `BranchingDetector` (ICT Plan 294, JS-divergence) doing this on the HLA manifold; `bifurcation_ratio` would be an *alternative* signal, not a new capability.

**Closest cousins across all five repos:**
| Source | Why it's a cousin |
|---|---|
| `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` | Same "geometry-as-inference-primitive" frame; ships DEC. |
| `katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md` | Per-NPC belief trajectory topological tracking. |
| `katgpt-rs/.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md` | Diagnostic over depth — same observational stance. |
| `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` | Divergence/boundary as probe-free signal — operationalized in Plan 314/334. |
| `katgpt-rs/.research/323_TEMP_Perturbed_Loss_Vector_Fingerprint.md` | Perturbation-based probe-free signal (nearest in spirit to stability metric). |
| `riir-ai/.research/123_Latent_Functor_Runtime_Guide.md` | Functor application = "stage"; coherence decay = "stability drop". |
| `riir-ai/.research/142_Distributional_Branching_Point_NPC_Guide.md` | Per-NPC bifurcation already shipped as BranchingDetector. |
| `riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md` | Phase transition detection on shards; same conceptual frame. |

---

## 3. Verdict

| Question | Answer |
|---|---|
| Training-only? | NO — purely inference-time / observational. |
| No prior art? | NO — attractor, divergence, phase-gate, re-estimation all ship. |
| New capability class? | NO — diagnostic primitive, not a new mechanism. |
| Product selling point? | Weak — "probe-free difficulty signal" is already shipped as `divergence_to_curiosity`. |
| Force multiplier? | Loose — at best a secondary signal for `BreakevenComplexityRouter`. |

**Tier: Gain.**

**One-line reasoning:** Every geometric phenomenon the paper describes operationally (attractor convergence, divergence-as-signal, phase transitions, coherence drift) already ships under non-paper vocabulary; the only transferable piece is a small reusable `LatentTrajectoryGeometry` diagnostic struct that may feed BreakevenComplexityRouter as an entropy-orthogonal secondary signal — opt-in, benchmarked, no Super-GOAT guide.

**Routing:** Open primitive → `katgpt-rs` (`katgpt-core` or `katgpt-types` — likely the latter since it's pure math over `&[f32]`). Plan only — no private guide needed (no game IP, no chain IP, no shard IP).

---

## 4. Caveats and explicit non-claims

1. **The paper itself flags "curvature as complexity readout" as a hypothesis**, not a proven result. Any plan that uses `mean_curvature` as a router signal must benchmark against entropy-only, not assume the paper's transformer-layer result transfers to HLA-emotion-trajectories without evidence.
2. **Transformers have layers; NPCs have ticks.** The mapping is not 1:1. The paper's three-phase structure (encoding/elaboration/output) is layer-resolved in a feedforward stack; per-NPC recurrent state evolves cyclically. The closest analog is *within-tick multi-stage functors* (latent_functor applications, not tick sequence).
3. **Paper is observational.** It explicitly does NOT establish that geometric properties cause behavior. Treating curvature as a routing signal is a modelless bet, not a paper-endorsed claim.
4. **Bifurcation onset depth is transformer-specific.** The 22%-of-depth number does not transfer. The *mechanism* (progressive separation between two contextual interpretations) might, but the onset threshold needs its own calibration on the target substrate.
5. **Not Super-GOAT.** Per the workflow's "no candidate escape hatch" rule, no `LatentTrajectoryGeometry` guide is created in riir-ai/riir-chain/riir-neuron-db. If a future gate proves the curvature secondary signal materially beats entropy on a crowd-NPC routing task, this can be re-evaluated at that time.

## TL;DR

Gain-tier observational paper. Five probe-free trajectory metrics; all substrates already operational in this codebase. One small transferable primitive (`LatentTrajectoryGeometry`) worth an opt-in plan as a secondary signal for `BreakevenComplexityRouter`. Not Super-GOAT — no new capability class, no new moat, no private guide required.
