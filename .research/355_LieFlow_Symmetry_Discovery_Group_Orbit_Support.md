# Research 355: LieFlow — Symmetry Discovery as Support Concentration on a Hypothesis Lie Group

> **Source:** Chen, Park, Eijkelboom, Yang, van de Meent, Wong, Walters — *Discovering Symmetry Groups with Flow Matching* — [arXiv:2512.20043](https://arxiv.org/abs/2512.20043), ICML 2026 (PMLR 306).
> **Date:** 2026-07-01
> **Status:** Active — GOAT (modelless distillation + fusion Super-GOAT TBD, tracked in [`.issues/011`](../.issues/011_lieflow_fusion_super_goat_investigation.md))
> **Related Research:** 279 (subspace phase gate — the linear-algebraic cousin), 299 (Clifford wedge — rotational recovery gate), 305 (phase rotation — deterministic 2D rotation), 314 (f-divergence group invariance — theoretical cousin), 321 (tropical equivariant operators), [riir-ai 166](../../riir-ai/.research/166_SE2_Equivariant_Game_Maps_Guide.md) (SE(2) lifting — the APPLY-known-group sibling), [riir-ai 020](../../riir-ai/.research/020_Orbit_OFT_Adapter_First_RL.md) (OFT — orthogonal adapter, redirects to riir-train)
> **Related Plans:** 301 (subspace phase gate — shipped), 319 (Clifford wedge — shipped, default-on), 322 (phase rotation — shipped), [riir-ai 354](../../riir-ai/.plans/354_se2_equivariant_substrate.md) (SE(2) substrate — pending), [riir-neuron-db 002](../../riir-neuron-db/.plans/002_phase_transition_subspace_phase_gate.md) (can_freeze two-sided gate — shipped)
> **Classification:** Public

---

## TL;DR

LieFlow reframes symmetry **discovery** as a distribution-learning problem on a Lie group `G`: train a velocity field `v_θ` via flow matching such that the induced distribution `p_θ(g)` over `G` has its **support concentrate on the true symmetry subgroup** `H ⊆ G`. Continuous symmetries yield a smooth spread over a submanifold; discrete symmetries (e.g. `C₄`, `Ico`) yield sharp peaks at the finite group elements. The paper's headline trick is a **power time schedule** (`p(t) = n·t^{n−1}`, `n=5`) that fixes a "last-minute convergence" pathology where the discrete-group flow stays near-zero until `t→1`.

**For katgpt-rs (modelless, inference-time):** the value is **NOT** the trained `v_θ` (that is a flow-matching training loop → riir-train). The transferable primitives are three: (1) the **reframe** — "symmetry discovery = support concentration on a hypothesis group", which is exactly what our shipped `subspace_phase_gate` + `spectral_flatness` + `can_freeze` already detect in linear-algebraic terms, just without the group-orbit vocabulary; (2) the **group-orbit interpolation** (Eq. 4: `x_t = exp(t·A)·x_0` with `A = log(g⁻¹)`) which sidesteps the "second moments are invariant under orthogonal transformations" trap that bit Plan 318 T4.8 (the latent functor couldn't recover rotations because dual-form fitting uses second moments — LieFlow's first-order orbit construction is the escape); (3) the **stabilizer-ambiguity diagnosis** (`ker J_x = stab_x`), a structural theorem explaining *when* a discovered symmetry coordinate is identifiable.

**Verdict: GOAT.** A small modelless primitive (direct invariance testing on a hypothesis group, scored by `spectral_flatness`-style concentration) plus a vocabulary crosswalk that names an existing capability gap. A fusion Super-GOAT (LieFlow × SE(2) Research 166 × Committed Personality Plan 336 × Plan 318 T4.8 stabilizer insight → per-NPC committed personality symmetry fingerprint) is plausible but **NOT committed** here — the Q2 (new capability class) and Q3 (selling point) answers need a design pass before claiming all 4 novelty-gate YES. Tracked in [`.issues/011`](../.issues/011_lieflow_fusion_super_goat_investigation.md) per the "no candidate escape hatch" rule.

**Distilled for katgpt-rs (modelless, inference-time):**

> Given a hypothesis matrix Lie group `G` (e.g. `SO(8)` acting on HLA, or `O(d)` acting on a shard's `style_weights[d]`), and a stream of observations `x_i`, compute the **invariance score** `s(g) = sigmoid(−β · d(q, g·q))` for sampled `g ∈ G` (where `q` is an empirical summary statistic of the stream — mean, covariance, or `style_weights` itself). The support of the high-`s(g)` region is the discovered subgroup `H`. Concentration of `s` (via `spectral_flatness` of the `s(g)` histogram, or participation ratio of the orbit Gram matrix) classifies `H` as continuous (spread) or discrete (peaked). No training, no `v_θ`.

This is a **one-page primitive** sitting next to `subspace_phase_gate.rs` — it adds the "group orbit" axis the existing linear-algebraic phase gate lacks.

---

## 1. Paper Core Findings

### 1.1 The reframe — symmetry discovery as distribution learning on `G`

Prior symmetry-discovery work (LieGAN, Augerino, SGM) estimates the **Lie algebra generators** of `H` directly. LieFlow instead learns a full distribution `p_θ(g)` over a hypothesis group `G ⊇ H`, then reads `H` off as the **closure of the support** of `p_θ`. Three advantages:

1. **No fixed Lie-algebra basis** — Benton/Augerino assume a fixed basis + uniform coefficient prior; LieFlow learns the basis implicitly via the support shape.
2. **Unified continuous + discrete** — LieGAN's Gaussian-coefficient assumption forces continuous groups; LieFlow's distribution can be unimodal (continuous) or multi-modal-peaked (discrete) in the same framework.
3. **Partial symmetries fall out for free** — the conditional `p_θ(g | x)` has support `H_x ⊆ H`, the per-input symmetry subset (§4.3).

### 1.2 The flow matching loop on Lie groups (TRAINING — redirects to riir-train)

Per global rule, the flow matching objective `L_LieCFM = E_{t,x}[‖v_θ(x_t) − A‖²_G]` is a training loop and redirects to riir-train. The interesting *structural* pieces are:

- **Eq. 4 group-orbit interpolation:** `x_t = exp(t·A)·x_0` where `A = log(g⁻¹) ∈ g`. The interpolation stays on the group orbit of `x_1` by construction.
- **Algorithm 3 group-element generation:** accumulate `M = exp(Δt·A_t)·M` along the sampled trajectory; the composed transform `M·g` is a group element `h ∈ H`.
- **Proposition 4.1:** under (B.1) ideal convergence, (B.2) globally generated support, (B.3) point stabilizers are global symmetries, the sampled `h` are supported in `H`.

### 1.3 The stabilizer-ambiguity theorem (Appendix A) — the structural insight

The infinitesimal action map `J_x : g → T_x(G·x_1)` has `ker J_x = stab_x` (the Lie algebra of the point stabilizer). Consequence:

- **Finite stabilizers** (e.g. `C_2` self-symmetry of a rectangle): `stab_x = {0}` → `A` is uniquely identifiable → LieFlow works (confirmed experimentally, Appendix G.1).
- **Continuous stabilizers** (e.g. `SO(2)` rotating a sphere): `stab_x ≠ {0}` → `A` is identifiable only modulo `stab_x` → needs a gauge choice, quotient formulation, or auxiliary supervision.

**This is the exact mathematical content of Plan 318 T4.8's "dual-form operator cannot distinguish a rotation from the identity" null result** — second moments `TᵀT = SᵀRᵀR·S = SᵀS` are blind to `R ∈ O(d)` because the dual form quotients out the orthogonal group. LieFlow's first-order orbit construction (Eq. 4) is the escape hatch: it lifts the fit from second-moment space to the orbit itself.

### 1.4 "Last-minute convergence" + power time schedule (§5.4, Appendix E)

For discrete groups, the velocity field averages to ~0 until `t → 1` because the modes are symmetric (by definition of subgroup) — the flow "doesn't know which mode to commit to" until time is almost out. Fix: sample `t ∼ p(t) = n·t^{n−1}` with `n = 5` (power schedule) to concentrate training signal near `t = 1`. Ablation: `n=1` (uniform) → W₁ = 0.470 on `SO(3) → Ico`; `n=5` → W₁ = 0.104 (4.5× improvement, Table 4).

### 1.5 Empirical results

| Setting | LieFlow W₁ (RI%) | LieGAN W₁ (RI%) | Notes |
|---|---|---|---|
| `SO(2) → C₄` | 0.054 (90.0%) | 1.07–1.71 (negative) | Clean recovery of 4 peaks |
| `SO(3) → Tet` | 0.066 (92.7%) | 1.26 (−40%) | Synthetic tetrahedron |
| `SO(3) → Ico` (w/ power) | 0.104 (80.0%) | 1.20 (−131%) | Power schedule required |
| `SO(3) → SO(2)` z-axis | 0.027 (98.3%) | 0.032 (98.0%) | Continuous group — both work |
| ModelNet10 `SO(3) → Ico` + noise (σ=0.1, 10% mask) | 0.31 | — | Graceful degradation |

Robustness: at σ=0.1 perturbation (~6°) + 10% point masking, W₁ degrades from 0.129 → 0.31 — the discovered group is still approximately correct. **This graceful-degradation property is the modelless hook**: a *non-learned* invariance test inherits the same robustness profile.

---

## 2. Distillation

### 2.1 The modelless core — direct invariance testing on a hypothesis group

Strip the trained `v_θ`. What's left is a **deterministic invariance test**:

```text
Given: hypothesis group G (with sampling prior p(G)),
       observation stream {x_i} or summary statistic q,
       concentration threshold τ.
For g sampled from p(G):
    invariance_score(g) = sigmoid(−β · distribution_distance(q, g·q))
    where distribution_distance is task-dependent:
      - shard style_weights: ‖style_weights − g·style_weights‖₂
      - HLA trajectory summary: Wasserstein-1 between {x_i} and {g·x_i}
      - DEC cochain field: ‖cochain − g·cochain‖_{L²}
Discover H as closure of {g : invariance_score(g) > τ}.
Classify H:
  - discrete  ⟺  spectral_flatness({score(g)}) < 0.3  (peaked support)
  - continuous⟺  spectral_flatness({score(g)}) ≥ 0.3  (spread support)
```

This is a **direct port** of the existing `subspace_phase_gate` + `spectral_flatness` machinery, lifted from "subspace of `ℝᵈ`" to "subgroup of `G`". The new piece is the group action `g·q` — which requires `G` to act on the data type. For `SO(d)`/`O(d)` acting on `ℝᵈ` vectors (HLA, shard style_weights), this is a matrix-vector product. For `SE(2)` acting on a 2D cochain, this is the lift/rotate/project pipeline of Research 166.

**Latent reframing:** the invariance score lives in `[0,1]` (sigmoid), the discovered subgroup `H` is identified by support concentration (entropy collapse), and `H`'s discrete-vs-continuous character is read off `spectral_flatness` of the score distribution. **Every step is a latent-space operation; nothing crosses sync until the resulting subgroup descriptor (a list of group elements, or a Lie-algebra basis) is committed.**

### 2.2 Fusion candidates (the closest cousins across all 5 repos)

| Cousin | What it ships | LieFlow fusion |
|---|---|---|
| **`subspace_phase_gate`** (Plan 301, Research 279) | `participation_ratio`, `numerical_rank`, `phase_transition_gate(N, d)`, `jacobian_svd_at` | The linear-algebraic cousin. LieFlow is the group-orbit generalization: "subspace" → "subgroup", "participation ratio of eigenvalues" → "spectral flatness of orbit invariance scores". Same concentration mechanism, different domain. |
| **`spectral_flatness`** (riir-neuron-db) | Wiener entropy of `style_weights[64]` — 0 = single-mode, 1 = uniform | **Direct reuse** as the discrete-vs-continuous classifier on the invariance score histogram. Zero new code. |
| **`can_freeze` two-sided gate** (riir-neuron-db Plan 002) | `FreezeGateReport { N, d, input_sufficient, output_flatness, can_freeze }` | The closest shipped instance of "support concentration detected, freeze the result". LieFlow adds the group axis: the freeze report would record the discovered `H`, not just `d`. |
| **Clifford wedge** (Plan 319, Research 299) | `geometric_product_wedge_into` — anti-symmetric bivector, G2 rotational recovery gate | The wedge is the **differential** of the LieFlow invariance score at identity: `wedge(u, v) ≈ d/dt [invariance_score(exp(t·A))]|_{t=0}` for `A = u∧v`. Already-shipped rotational signal. |
| **Phase rotation** (Plan 322, Research 305) | Deterministic 2D rotation `cos(α)·a + sin(α)·b` with `α = sigmoid(⟨state, dir⟩)·π/2` | The construction primitive for `g·x` when `G = SO(2)`. LieFlow generalizes "one rotation" to "discover the rotation subgroup". |
| **SE(2) lifting** (Research 166, Plan 354) | Lift → group-conv → project on the SE(2) cell complex | **The APPLY-known-group sibling.** LieFlow DISCOVERS the group; SE(2) APPLIES it. Fusion: use LieFlow to discover that the relevant group is *not* SE(2) but e.g. `C₄` (discrete), then route to a discrete-group lift instead of the continuous SE(2) lift. |
| **Plan 318 T4.8 stabilizer insight** (latent functor rank-k) | "Dual-form operator cannot recover rotations because second moments are `O(d)`-invariant" | **LieFlow's `ker J = stab` theorem is the general statement of this null result.** The first-order orbit construction (Eq. 4) is the principled escape — a follow-up could revisit rank-k functor fitting via orbit interpolation rather than second moments. |
| **f-divergence group invariance** (Research 314) | Fisher-Rao distance depends only on the double coset `H g₁⁻¹ g₂ H` | Theoretical cousin. Research 314 says "invariant statistics are functions of the double coset"; LieFlow operationalizes this by *learning the double coset structure from data*. |

**Fusion (the Super-GOAT-TBD combination):** LieFlow × Research 166 (SE(2)) × Plan 336 (Committed Personality) × Plan 318 T4.8 (stabilizer). The combination: **discover each NPC's effective symmetry group from its runtime HLA trajectory modellessly (via direct invariance testing on `O(8)` or a subgroup), commit the discovered group elements as an `ArchetypeBlendShard`-style freeze/thaw artifact, and use the discovered group to route between continuous (SE(2)-style) and discrete (`C₄`-style) perception operators.** This would give per-NPC *committed personality symmetry fingerprints* — a new capability class. **Not committed here** — see [`.issues/011`](../.issues/011_lieflow_fusion_super_goat_investigation.md).

### 2.3 What redirects to riir-train

- The trained velocity field `v_θ` and the flow matching loss `L_LieCFM` — training loop, full stop.
- The power time schedule ablation (`n=5`) — only meaningful in the trained setting; the modelless invariance test has no time axis.
- The Wasserstein-1 evaluation metric on generated group elements — evaluation methodology for the trained model.

---

## 3. Verdict

**Tier: GOAT** — provable vocabulary + small-primitive gain, not a new capability class by itself.

| Tier criterion | This paper |
|---|---|
| Novel mechanism (no prior art)? | **Partial.** The group-orbit framing of support concentration is new vocabulary; the underlying concentration detection already ships (`subspace_phase_gate`, `spectral_flatness`, `can_freeze`). The modelless invariance-test primitive itself is small and arguably a generalization of an existing one. |
| New capability class? | **No, not standalone.** Direct invariance testing is a measurement, not a behavior. The new-capability claim depends on the fusion (committed personality symmetry fingerprint) — tracked in Issue 011, not claimed here. |
| Product selling point? | **Indirect.** "Our NPCs discover their own symmetry groups from runtime data" is a selling point only when fused with commitment (Plan 336) and applied perception (Research 166). Standalone, it's a measurement primitive. |
| Force multiplier? | **Yes** — connects to `subspace_phase_gate`, `spectral_flatness`, `can_freeze`, Clifford wedge, phase rotation, SE(2), Plan 318 stabilizer, Research 314. ≥4 systems. |

**One-line reasoning:** LieFlow's training loop redirects to riir-train; the modelless residue (group-orbit invariance test + support-concentration classifier via existing `spectral_flatness`) is a clean GOAT that generalizes `subspace_phase_gate` from subspaces to subgroups. The Super-GOAT fusion (committed per-NPC symmetry fingerprints) is plausible but needs a Q2/Q3 design pass before committing — tracked in Issue 011, not claimed as candidate here.

### 3.1 What ships (the GOAT plan scope)

A new opt-in feature `group_invariance_probe` in `katgpt-core` (sibling of `subspace_phase_gate`), providing:

1. `invariance_score(g, q, distance_fn) -> f32` — the per-element score.
2. `discover_subgroup<G: GroupAction>(samples: &[G::Elem], q: &Q, distance_fn, τ, β) -> SubgroupReport` — batch score + support extraction.
3. `classify_subgroup(scores: &[f32]) -> SubgroupClass` — `Discrete` / `Continuous` / `Partial` via `spectral_flatness`.
4. `SubgroupReport { n_support, class, intrinsic_dim, score_flatness, support_elements }` — mirrors `FreezeGateReport` shape.

**GOAT gate (the G1–G4):**

| Gate | Test | Target |
|---|---|---|
| **G1** Correctness | On synthetic `SO(2) → C₄` data (4 orbit points), `discover_subgroup` recovers exactly the 4 group elements; `classify_subgroup` returns `Discrete`. | 4/4 elements, `Discrete` |
| **G2** Non-redundancy | On data with **no** symmetry (uniform on `SO(2)`), `discover_subgroup` returns the full `SO(2)` support and `Continuous`. Distinguishes from a `subspace_phase_gate`-only baseline that cannot tell "uniform on a group" from "no structure". | Classification matches ground truth on ≥ 5 synthetic settings |
| **G3** No regression | `cargo check -p katgpt-core --all-features` clean; no `default` feature change. | Clean |
| **G4** Alloc-free | `discover_subgroup_into` variant takes `&mut [f32]` scratch; zero allocations in the hot scoring loop. | 0 allocs (CountingAllocator) |

**Promotion:** G1–G4 PASS → opt-in `group_invariance_probe` ships. **Do NOT promote to default** until a downstream consumer (riir-neuron-db `can_freeze` extension, or riir-ai committed personality fusion from Issue 011) demonstrates the selling point.

### 3.2 The fusion Super-GOAT — tracked in Issue 011

Per the research skill's "no candidate escape hatch" rule: I am **not confident enough** to commit all 4 novelty-gate YES right now. The honest state is:

- **Q1 (no prior art?):** likely YES — no shipped primitive discovers a symmetry group from data. Research 166 applies a known group; this discovers one.
- **Q2 (new capability class?):** UNCERTAIN — "committed per-NPC symmetry fingerprint" is plausibly new, but it may reduce to "a new shard field" rather than a new capability. Needs a design pass.
- **Q3 (selling point?):** UNCERTAIN — "NPCs discover their own symmetry" is catchy but the game-side value depends on whether discovered symmetries are *actionable* (do they change NPC behavior?) or merely *descriptive* (a new field in the freeze report).
- **Q4 (force multiplier?):** YES — connects to HLA, functor, neuron-db, DEC, LatCal, SE(2), committed personality.

Two UNCERTAIN answers → **do not claim Super-GOAT candidate**. Issue 011 tracks the design investigation; if it returns Q2+Q3 = YES, the Super-GOAT guide lands in `riir-ai/.research/` (game-runtime selling point) and the plan lands in `riir-ai/.plans/`.

---

## 4. Latent vs Raw Boundary

- **Invariance scores `s(g) ∈ [0,1]`** — latent, per-NPC or per-shard, never synced directly.
- **Discovered subgroup descriptor** (list of group elements for discrete `H`, or Lie-algebra basis for continuous `H`) — latent, but **committable** via `MerkleFrozenEnvelope` (the descriptor is fixed-size and BLAKE3-hashable). This is the natural freeze artifact for the fusion Super-GOAT.
- **Classification label** (`Discrete` / `Continuous` / `Partial`) — raw, deterministic, may cross sync as a `u8` tag (mirrors `FreezeGateReport`'s raw fields).
- **Group action `g·q`** — operates entirely in latent space; never touches the raw sync layer.

**Bridge pattern (per global AGENTS.md):** the discovery is latent; only the resulting classification tag (and optionally the committed subgroup descriptor hash) crosses sync. Anti-cheat sees the tag, not the scores.

---

## 5. References

- **Paper:** [arXiv:2512.20043](https://arxiv.org/abs/2512.20043) — Chen et al., ICML 2026 (PMLR 306).
- **Linear-algebraic cousin:** [`katgpt-rs/.research/279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md`](279_Diffusion_Curse_Dimensionality_Subspace_Clustering_Fusion.md) + [`katgpt-rs/.plans/301_subspace_phase_gate.md`](../.plans/301_subspace_phase_gate.md).
- **Closest shipped instance:** [`riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md`](../../riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md) (the `can_freeze` guide).
- **Apply-known-group sibling:** [`riir-ai/.research/166_SE2_Equivariant_Game_Maps_Guide.md`](../../riir-ai/.research/166_SE2_Equivariant_Game_Maps_Guide.md).
- **Stabilizer insight precedent:** [`riir-ai/.plans/318_latent_functor_rank_k_upgrade.md`](../../riir-ai/.plans/318_latent_functor_rank_k_upgrade.md) T4.8 (the dual-form orthogonal-blindness null result).
- **Theoretical cousin:** [`katgpt-rs/.research/314_Group_Invariance_f_Divergences_Fisher_Rao.md`](314_Group_Invariance_f_Divergences_Fisher_Rao.md).
- **Training-loop redirect:** riir-train (the flow matching `v_θ` and power time schedule).
- **Follow-up:** [`katgpt-rs/.issues/011_lieflow_fusion_super_goat_investigation.md`](../.issues/011_lieflow_fusion_super_goat_investigation.md).

---

## TL;DR

LieFlow discovers symmetry groups by learning a distribution over a hypothesis Lie group `G` whose support concentrates on the true subgroup `H` — continuous `H` spreads smoothly, discrete `H` peaks sharply. The trained velocity field redirects to riir-train; the modelless residue is a small primitive (`group_invariance_probe`) that generalizes our shipped `subspace_phase_gate` from "subspace of `ℝᵈ`" to "subgroup of `G`", reusing `spectral_flatness` as the discrete-vs-continuous classifier. **Verdict: GOAT** — the standalone primitive ships behind `group_invariance_probe`, G1–G4 defined, do not promote to default until a downstream consumer justifies it. The LieFlow × SE(2) × Committed Personality × Plan 318 stabilizer fusion is a plausible Super-GOAT (per-NPC committed symmetry fingerprints) but Q2+Q3 are uncertain — tracked in Issue 011, not claimed as candidate per the no-escape-hatch rule.
