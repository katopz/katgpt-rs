# Issue 011: LieFlow Fusion Super-GOAT Investigation — Per-NPC Committed Symmetry Fingerprints

> **Origin:** [`katgpt-rs/.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md`](../.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md) §3.2.
> **Date:** 2026-07-01
> **Status:** Open — design investigation (NOT a Super-GOAT candidate yet)
> **Prerequisite:** GOAT plan for `group_invariance_probe` (Research 355 §3.1) — the open primitive this investigation consumes.

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

## Tasks (the design investigation)

- [ ] **T1** Define the hypothesis group `G` for HLA. Candidates: `O(8)` (full orthogonal — too big?), `SO(8)`, a block-diagonal `SO(2)×SO(2)×SO(2)×SO(2)` (per affect-pair), or the discrete subgroup `{I, R_π}` (valence↔arousal swap). The choice determines the invariance-test cost and the granularity of the fingerprint.
- [ ] **T2** Define the distribution distance `d(q, g·q)` for HLA. Candidates: `‖μ − g·μ‖₂` (mean shift), Wasserstein-1 between trajectory distributions, `‖Σ − g·Σ·gᵀ‖_F` (covariance shift). The choice determines what "symmetric" means operationally.
- [ ] **T3** **Q2 design pass** — sketch the *behavior* change an NPC exhibits when its committed symmetry fingerprint is `C₄` vs `SO(2)` vs `Partial`. If the only consumer is "a new field in the freeze report", Q2 = NO. If the fingerprint changes which perception operator runs (discrete-lift vs SE(2)-lift vs none), Q2 = likely YES.
- [ ] **T4** **Q3 design pass** — write the one-sentence selling point as if for the `riir-ai/.docs/pillars/` index. If it reads as a pillar ("rotation-discovered NPC perception"), Q3 = YES. If it reads as an optimization ("slightly cheaper perception for symmetric NPCs"), Q3 = NO.
- [ ] **T5** Check the **Plan 318 T4.8 stabilizer** interaction concretely: does the latent functor's orthogonal-blindness null result mean the discovered `H` is *also* blind to `O(d)` rotations? If so, the hypothesis group must be a strict subgroup (e.g. signed permutation, not full `O(d)`), or the discovery is ill-posed.
- [ ] **T6** Scope the **commitment artifact**. Is the discovered `H` a fixed-size descriptor (e.g. "8 group elements + a u8 class tag")? If variable-size, the `MerkleFrozenEnvelope` commitment needs a serialization format — non-trivial.
- [ ] **T7** Decide: **YES → create `riir-ai/.research/NNN_committed_personality_symmetry_fingerprint_guide.md` + `riir-ai/.plans/NNN_*.md`. NO → close this issue, the GOAT-only primitive in Research 355 §3.1 is the final scope.**

## Blockers

- The GOAT plan for `group_invariance_probe` (Research 355 §3.1) must ship first — this investigation consumes its API.
- Plan 354 (SE(2) substrate) Phases 1–3 must land before the "route to discrete vs continuous lift" consumer in the fusion hypothesis can be wired.

## Non-goals

- Re-litigating the LieFlow training loop. It redirects to riir-train; this issue is purely about the modelless fusion.
- Re-running the Research 355 novelty gate. Q1 (no prior art) and Q4 (force multiplier) are already YES; only Q2 + Q3 are open.
