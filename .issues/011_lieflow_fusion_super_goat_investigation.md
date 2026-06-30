# Issue 011: LieFlow Fusion Super-GOAT Investigation — Per-NPC Committed Symmetry Fingerprints

> **Origin:** [`katgpt-rs/.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md`](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md) §3.2.
> **Date:** 2026-07-01
> **Status:** **Closed 2026-07-01** — Q2+Q3 = NO (conditional on Plan 354), GOAT-only is final scope.
> **Prerequisite:** GOAT plan for `group_invariance_probe` (Research 355 §3.1) — SHIPPED (Plan 356 Phase 1, 8/8 GOAT gates PASS, opt-in).
> **Re-open condition:** Plan 354 (SE(2) substrate) Phases 1–3 ship AND a discrete-group-lift companion is built → re-run T3/T4; if both flip to YES, create `riir-ai/.research/NNN_committed_personality_symmetry_fingerprint_guide.md` + `riir-ai/.plans/NNN_*.md`.

---

## Why this issue exists

Research 355 distilled LieFlow (arXiv:2512.20043) to a **GOAT** (modelless `group_invariance_probe` primitive that generalizes `subspace_phase_gate` from subspaces to subgroups). The fusion combination (LieFlow × SE(2) Research 166 × Committed Personality Plan 336 × Plan 318 T4.8 stabilizer insight) is a plausible **Super-GOAT** — per-NPC committed personality symmetry fingerprints — but two of the four novelty-gate questions are **UNCERTAIN**:

- **Q2 (new capability class?):** is "committed per-NPC symmetry fingerprint" a new capability, or just a new field on `ArchetypeBlendShard`?
- **Q3 (product selling point?):** is "NPCs discover their own symmetry groups from runtime data" actionable (changes NPC behavior) or merely descriptive (a new freeze-report field)?

Per the research skill's **"no candidate escape hatch"** rule, two UNCERTAIN answers means: do NOT claim Super-GOAT candidate, do NOT create the private guide yet, track the design investigation here. This issue closes when Q2 + Q3 are answered YES (→ create `riir-ai/.research/NNN_*.md` guide + `riir-ai/.plans/NNN_*.md` plan) or NO (→ demote to "GOAT only, no Super-GOAT", close this issue).

## The fusion hypothesis (what would be built if Q2+Q3 = YES)

```
Per-NPC runtime HLA trajectory {h_t}
    ↓ (modelless invariance testing, Research 355 §2.1)
Discovered subgroup H_npc ⊆ O(8)   [or a chosen hypothesis group]
    ↓ (classify: Discrete / Continuous / Partial)
Subgroup descriptor + classification tag
    ↓ (commit via MerkleFrozenEnvelope — sibling of ArchetypeBlendShard)
Committed per-NPC symmetry fingerprint
    ↓ (consume: route perception operator)
If Discrete C_n  → discrete-group lift (cheap, n orientations)
If Continuous    → SE(2) continuous lift (Research 166, expensive)
If Partial       → per-context conditional routing (p_θ(·|x) analog)
```

The selling-point sentence (TBD): *"Our NPCs discover and commit their own effective emotional symmetry groups from runtime data — an NPC whose affect trajectory is invariant under a `C₄` rotation of its HLA axes gets a `C₄`-equivariant perception pipeline for free, while an NPC with no symmetry gets the full `SO(8)`-equivariant pipeline. No competitor ships per-NPC discovered symmetry."*

## Tasks (design investigation) — COMPLETE 2026-07-01

Grounded by codebase investigation: HLA struct (`riir-engine/src/committed_blend/archetypes.rs` L11–21), Plan 318 T4.8 (`riir-engine/src/latent_functor/arithmetic.rs` L2041–2057), Plan 354 state (`riir-ai/.plans/354_se2_equivariant_substrate.md`), `ArchetypeBlendShard` (`riir-neuron-db/src/archetype_blend_shard.rs` L82–190).

- [x] **T1 — Hypothesis group `G` for HLA → SO(2)×SO(2)×SO(2)×SO(2) on named emotion pairs. NOT full `O(8)`/`SO(8)`.**
  HLA is 8-dim (`HLA_DIM = 8`): `[valence, arousal, desperation, calm, fear, reserved, reserved, reserved]` (`committed_blend/archetypes.rs` L11–21). The codebase already treats `(arousal, fear)` as an SO(2) rotation plane — `CautiousField` (L139–218) applies `R(θ) − I` in that plane with `lipschitz_bound = 1.0` ("rotation is an isometry"). Natural pairings the code suggests:
  - (valence, calm) = approach/settle [axes 0, 3]
  - (arousal, fear) = activation/vigilance [axes 1, 4] — *already an SO(2) in `CautiousField`*
  - (desperation, reserved) [axes 2, 5]
  - (reserved, reserved) [axes 6, 7, or identity]
  Full `SO(8)` is wrong: it mixes semantically-distinct axes (rotating valence into fear has no behavioral grounding) AND hits the reserved-axis null subspace (axes 5–7 carry no variance in the default library → large stabilizer → discovery vacuous). The 4-factor product respects the named-axis pairing and mirrors an existing shipped pattern.

- [x] **T2 — Distance `d(q, g·q)` → mean-shift `‖μ − g·μ‖₂` as primary.**
  First-order statistics (mean) are immune to the T4.8 second-moment blindness (see T5). Cost is O(d), not O(d²). Optional secondary signal: covariance-shift `‖Σ_block − g·Σ_block·gᵀ‖_F` **restricted to within-pair 2×2 blocks** — informative there (a 30° rotation in the (valence, calm) plane changes the off-diagonal covariance unless the block is isotropic), blind only at full `O(d)`. Wasserstein-1 redirects to riir-train — it's the paper's *evaluation metric* for the trained `v_θ`, not a modelless probe. The shipped `group_invariance_probe` API takes distance via the caller (`GroupAction::act` produces `g·q`; caller computes distance), so this is a wiring recommendation for the future consumer, not an API change.

- [x] **T3 — Q2 design pass → NO (conditional).**
  The behavior change ("route perception operator: discrete-lift vs SE(2)-lift vs none") requires the SE(2) lift + group-conv pipeline to EXIST. **Plan 354 (SE(2) substrate) Phases 1–3 are NOT STARTED** — verified: no `riir-engine/src/equivariant/` module, no `se2_equivariant` feature flag, all checkboxes `[ ]` in `riir-ai/.plans/354_se2_equivariant_substrate.md`. Research 166 (SE(2) Game Maps) is research-only; no implementation. Without a perception operator to route to, the only consumer is "a new field in the freeze report" → Q2 = NO. **Re-opens to YES if/when Plan 354 Phases 1–3 ship AND a discrete-group-lift companion is built.**

- [x] **T4 — Q3 design pass → NO (conditional).**
  Selling-point sentence: *"Our NPCs discover and commit their own effective emotional symmetry groups from runtime data."* Without Plan 354 (the T3 blocker), this reduces to *"slightly cheaper freeze reports for symmetric NPCs"* — an optimization, not a pillar. The pillar reading (*"rotation-discovered NPC perception"*) requires the SE(2)/discrete lift pipeline to exist. Q3 = NO until Plan 354 lands. (The other modelless consumer from Plan 356 — a `can_freeze` extension adding a group-axis field to `FreezeGateReport` — is also descriptive-only, with no anti-cheat value since latent HLA invariance has no anti-cheat application per the latent/raw boundary rules.)

- [x] **T5 — Plan 318 T4.8 stabilizer interaction → CONFIRMED: hypothesis group MUST be strict subgroup of `O(d)`.**
  The T4.8 null result (`riir-engine/src/latent_functor/arithmetic.rs` L2041–2057, verbatim): *"the DUAL-form operator `C = Q̃·reg⁻¹·K̃ᵀ` is fit from second moments `Q̃ = Φ_tᵀ·Targets` and `K̃ = Ψ_sᵀ·Sources`. Second moments are INVARIANT under orthogonal transformations of the source (`RᵀR = I` ⇒ `TᵀT = SᵀRᵀR·S = SᵀS = K̃`). So even with linear (non-sigmoid) bases, the dual form CANNOT distinguish a rotation from the identity."* Covariance-Frobenius scoring `‖Σ − g·Σ·gᵀ‖_F` inherits this: for isotropic Σ every `R ∈ O(d)` scores ≈ 0 → discovery vacuous; even for anisotropic Σ, the score conflates "R is a symmetry" with "R preserves Σ's eigenspaces". **Resolution (consistent with T1+T2): use SO(2)×SO(2)×SO(2)×SO(2) + mean-shift distance.** Per-pair 2×2 rotations are not O(2)-blind in the same way.

- [x] **T6 — Commitment artifact scope → fixed-size Pod sibling of `ArchetypeBlendShard`.**
  Mirror `riir-neuron-db/src/archetype_blend_shard.rs` (224 bytes, `#[repr(C)]` Pod, BLAKE3-chunk-aligned, two-tier commitment via own `commitment` field + `freeze_envelope()`/`thaw_envelope()` gated `merkle_freeze`). Cap N at a fixed `MAX_GROUP_ELEMENTS` (e.g. 8 generators × `[f32; 8]` = 256 bytes for the SO(2)⁴ case), store `n_elements: u8` + class tag. Gets zero-copy mmap + the freeze envelope for free. Variable-size `Vec`-based descriptors ARE committable today (`MerkleFrozenEnvelope` is payload-agnostic — `StateTransition.delta: Vec<u8>` is frozen in `freeze.rs` tests), but the Pod pattern is cleaner and matches the existing template. Constructor pattern: 4 constructors (`new`/`new_unchecked`/`from_bytes`/optional deterministic) per the AGENTS.md "constructor audit" rule; mandatory `layout_has_no_implicit_padding` test.

- [x] **T7 — Decision → NO (close issue, GOAT-only is final scope).**
  Q2+Q3 = NO (both conditional on Plan 354 Phases 1–3, which are NOT STARTED). Per the research skill's no-escape-hatch rule, two NO answers → do NOT create the Super-GOAT guide/plan. The modelless primitive in Research 355 §3.1 (`group_invariance_probe`, shipped Plan 356 Phase 1, opt-in) is the final scope. **Re-open condition:** Plan 354 Phases 1–3 ship AND a discrete-group-lift companion is built → re-run T3/T4; if both flip to YES, create `riir-ai/.research/NNN_committed_personality_symmetry_fingerprint_guide.md` + `riir-ai/.plans/NNN_*.md`.

## Blockers (resolved)

- ~~The GOAT plan for `group_invariance_probe` (Research 355 §3.1) must ship first~~ → **SHIPPED** (Plan 356 Phase 1, 8/8 GOAT gates PASS, opt-in feature).
- Plan 354 (SE(2) substrate) Phases 1–3 must land before the "route to discrete vs continuous lift" consumer in the fusion hypothesis can be wired → **NOT STARTED**. This is the binding re-open condition; see T3/T4/T7.

## Non-goals

- Re-litigating the LieFlow training loop. It redirects to riir-train; this issue is purely about the modelless fusion.
- Re-running the Research 355 novelty gate. Q1 (no prior art) and Q4 (force multiplier) are already YES; only Q2 + Q3 are open.
