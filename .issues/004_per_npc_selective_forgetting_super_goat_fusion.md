# Issue 004: CLOSED — Per-NPC Selective Forgetting Not Novel (covered by R158/R161/R155)

> **Origin:** Research 320 (Red Queen Gödel Machine — Selective Erasure & Best-Belief Selection, arXiv:2606.26294)
> **Date:** 2026-06-28
> **Status:** ❌ CLOSED 2026-06-28 — not Super-GOAT; candidate selling point already shipped under different vocabulary.
> **Resolution:** Candidate selling point was a paraphrase of Research 158 (Committed Personality Blend) §1.3 property #3 + §2.4. No new capability. Plan 336 retained in reduced form (GOAT for `best_belief` only; DRY trait extraction is Gain).

---

## Why this closed (the evidence)

The original Issue 004 proposed a candidate selling point:

> "Our NPCs co-evolve their personality direction vectors and their theory-of-mind models of other NPCs. When an NPC's snapshot is swapped (personality divergence, tame event, faction change), only the memories that depended on the OLD personality are erased — position, HP, wallet balance (raw, synced) survive bit-identical; affect projections, KG triples, cached policies (latent, local) are selectively invalidated."

Reading the already-committed riir-ai Super-GOAT corpus shows this is **a restatement**, not a new capability:

### Research 158 (Committed Personality Blend) already ships the headline

Verbatim from `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md` §1.3 property #3:

> "Sampling invariance — the personality survives observation gaps. Fog-of-war, network desync, snapshot thaw all preserve the committed blend. This is the FAME Proposition 3 property, made real for game AI."

And §2.4 (sync boundary): only the K-weight vector `π` (K=3 floats = 12 bytes) crosses sync as LatCal-committed raw scalars; HLA state evolution `z(t)` (8-dim) stays local per-NPC. **This IS the "position/HP survive bit-identical; affect stays local" claim** — already committed, BLAKE3-tagged, quorum-verifiable.

### Research 161 (Cognitive Branch) already ships the per-domain partition

`riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md` §1.2: each NPC has a `BranchBank` of ≤ D=8 cognitive branches, each branch accumulates verifier-approved experience **without cross-contamination** (non-interference by orthogonal construction), failures stored branch-local. When a branch is swapped, only that branch's episodic store is affected — by construction. This IS "selective forgetting on swap", at branch granularity.

### Research 155 (Sub-Goal Compaction) already ships the compaction gate

CUCG (closed-unit compaction gate) at MMO scale. The shard-side `can_freeze` gate (riir-neuron-db Plan 002) and the trajectory-side rubric (katgpt-rs Plan 320/333) are already recognized as isomorphic (riir-neuron-db Research 007).

## Q1–Q4 verdict (all answered by the above)

| Q | Original concern | Answer from existing corpus |
|---|---|---|
| **Q1** (no prior art?) | Needs deeper code audit | **Prior art IS Research 158 + 161 + 155.** The "selective forgetting on personality swap" is the design of CommittedFieldBlend. Dead. |
| **Q2** (new class?) | Plausible but unverified | **Not a new class.** It's the defining property of the already-committed per-NPC committed-personality Super-GOAT (R158). |
| **Q3** (selling point?) | Depends on riir-train | **Moot** — the selling point is already claimed by R158. No new selling point to validate. |
| **Q4** (force multiplier?) | ≥5 pillars | **N/A** — applies to R158, not to a new fusion. |

## Lesson (for the research workflow)

This is the canonical failure mode the skill warns about: **notes-only grep + paper-vocabulary-only grep misses the riir-ai Super-GOAT guides**. Research 320's fusion protocol §1 step 1 mandates grepping `riir-ai/.research/` + `riir-ai/.plans/` — and the listing was done (158, 161, 155 appeared in `list_directory riir-ai/.research`), but the guides were not READ because the grep was scoped to "selective erasure" / "co-evolution" keywords. The guides frame the same mechanism under "committed personality", "non-interference branches", "sampling invariance" — different vocabulary. **Vocabulary translation across repos is insufficient if the translated terms are only grepped, not read.** Future mitigation: when a candidate selling point mentions per-NPC + memory + swap, mandatorily `read_file` the R136/R146/R149/R152/R155/R158/R161/R163 guide set before claiming novelty.

## What survives from Research 320

- **`best_belief.rs`** (ε-quantile Beta lower bound for conservative selection) — genuinely new, no prior art in the codebase (confirmed: `sample_beta` exists for Thompson sampling, but no inverse-CDF quantile function for selection). Stays GOAT.
- **`CriterionVersionedCache<V>` trait extraction** — DRY refactor over `DecCache` (katgpt-core `dec/cache.rs`) + `ZoneGeometryCache` (Plan 335 Phase 2). Both already ship the pattern. Downgraded to **Gain**.
- **Controlled-utility-evolution freeze/thaw reframe** — architectural observation, no new primitive. Lives as a paragraph in Research 320 §2.2.3, not a plan.

## References

- [Research 158 (Committed Personality Blend)](../../riir-ai/.research/158_per_npc_committed_personality_blend_guide.md) — the already-committed Super-GOAT that ships the candidate selling point.
- [Research 161 (Cognitive Branch)](../../riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md) — per-domain non-interference memory partition.
- [Research 155 (Sub-Goal Compaction)](../../riir-ai/.research/155_Per_NPC_Sub_Goal_Compaction_Guide.md) — CUCG at MMO scale.
- [Research 320](../.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md) — corrected distillation.
- [Plan 336](../.plans/336_controlled_utility_primitives.md) — reduced to `best_belief` GOAT + DRY trait Gain.
