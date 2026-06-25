# Issue 002: AC-Prefix × Engram × Latent Field Steering — Super-GOAT Quality Gate

**Date:** 2026-06-24
**Status:** **CLOSED — negative Super-GOAT verdict** (2026-06-26). The fusion is not realizable: no Transformer-in-the-loop game-AI workload exists in riir-ai (verified by API audit), compute economics are catastrophic (100×–377,000× cost asymmetry vs additive latent fusion), multi-layer correctness requires riir-train (Issue 003), and Research 295 §2.4 itself rates the novelty gate as borderline-to-negative. AC-Prefix stays shipped as a standalone token-level conditional evaluation primitive (default-on, GOAT-passed). Re-open only if a Transformer-in-the-loop host workload emerges naturally from other work. See Issue 009 Resolution.
**Origin:** Plan 313 Phase 4 T4.4 (AC-GPT Prefix Primitive GOAT PASS)
**Related:** katgpt-rs/.plans/313 (AC-GPT Prefix), katgpt-rs/.research/295 §2.4 (fusion table), katgpt-rs/.plans/299 (Engram), katgpt-rs/.plans/309 (Latent Field Steering)

## Context

Plan 313 shipped the AC-GPT arbitrary-conditional prefix primitive (modelless mask builder + sequence augmenter). The GOAT gate G1–G4 passed on 2026-06-24 (`katgpt-rs/.benchmarks/313_ac_prefix_goat.md`):

- **G1:** primitive buffer construction bit-identical to manual reference (0.000000 diff).
- **G2:** 27.258× speedup vs iterative-MLM unmasking (1.39 ms vs 37.9 ms on 128-token base, 64 conditioning).
- **G3:** `AcPrefix::empty()` bit-identical to vanilla causal forward (0 mismatches).
- **G4:** 0 allocations on `attends(i,j)` and `mask.get(i,j,n)` hot paths.

This proves the primitive is *fast and correct as a mask builder*. It does **not** prove the primitive delivers a *quality* win on a real workload — that's the Super-GOAT question.

## The Question

**Does the AC-Prefix × Engram × Latent Field Steering fusion deliver a measurable quality win over Engram × Latent Field Steering at iso-compute on a real game-AI workload?**

The fusion (from Research 295 §2.4):

| Signal | Source | Role |
|--------|--------|------|
| Known future outcome | AC-Prefix (this plan) | Position-aware conditioning set (mask-disciplined, leakage-free) |
| Retrieved similar past pattern | Engram (P299) | Hash-addressed pattern memory, fused into hidden state |
| Designer-authored steering | Latent Field Steering (P309) | Top-down direction-vector injection |

The three together produce: "NPC samples behavior conditioned on a known future outcome AND a retrieved similar past pattern AND a designer-authored steering direction" — three conditioning signals, one forward pass, no leakage. None of the three alone composes all three signals.

## Why this is Super-GOAT not GOAT

- The GOAT gate (speedup + correctness + no-regression + alloc-free) is satisfied — that's what Plan 313 proved.
- The Super-GOAT gate requires a **quality** measurement on a real workload, which needs:
  1. A riir-ai game runtime benchmark harness (MultiThreatArena-style, per Plan 314 precedent).
  2. A baseline: Engram × Latent Field Steering (without AC-Prefix).
  3. A treatment: Engram × Latent Field Steering × AC-Prefix.
  4. An iso-compute constraint (same forward-pass budget).
  5. A quality metric (win rate, survival time, task completion, etc.).

This is riir-ai's job — the katgpt-rs primitive is the open half, the riir-ai runtime wiring + benchmark is the private half.

## Prerequisites (blocking)

- [ ] riir-ai Plan for AC-Prefix runtime wiring (consume `katgpt_core::ac_prefix::AcPrefix` from riir-engine).
- [ ] riir-ai benchmark harness with Engram × Latent Field Steering baseline already instrumented.
- [ ] A game-AI workload where "conditioning on a known future outcome" is semantically meaningful (e.g., hindsight policy evaluation, counterfactual curiosity queries, or dreamer-style rollout conditioning).

**✅ RESOLVED (Issue 009, 2026-06-26):** The compute-substrate gap is confirmed and closes the issue with a negative verdict. Five verified structural facts (no shared compute graph, no host workload in riir-ai, catastrophic compute economics, multi-layer correctness gap, borderline novelty) make the fusion not realizable without negative-ROI infrastructure investment. Direction B (build a Transformer-in-the-loop arena, ~2–3 plans) is rejected as negative-ROI: the additive baseline (Engram × Latent Field Steering) already covers ~70% of the use case at 100×–377,000× lower compute cost. AC-Prefix stays shipped as a standalone primitive. See `katgpt-rs/.issues/009_ac_prefix_fusion_compute_substrate_gap.md` § Resolution.

## Falsifiable prediction

If the fusion delivers ≥5% quality improvement at iso-compute over the Engram × Latent Field Steering baseline on the chosen workload, the Super-GOAT gate passes and the fusion becomes a default riir-engine wiring. If it delivers <5% or regresses, the AC-Prefix primitive stays shipped-but-underused (still useful for ad-hoc conditional evaluation queries, just not a default wiring).

## Cross-references

- **Research:** `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md` §2.4 (fusion table), §3 (GOAT verdict).
- **Plan:** `katgpt-rs/.plans/313_AC_GPT_Prefix_Primitive.md` (this primitive).
- **Bench:** `katgpt-rs/.benchmarks/313_ac_prefix_goat.md` (GOAT gate results).
- **Cousin plans:** P299 (Engram), P309 (Latent Field Steering), P012 Phase 5 (Target-Conditioned Draft).
- **riir-ai precedent:** Plan 314 (BoM arena benchmark — the template for the Super-GOAT harness).
