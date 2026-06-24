# Issue 003: AC-Prefix G1 — Modelless Unblock Investigation (freeze/thaw + raw/lora + latent correction)

**Date:** 2026-06-24 (revised from "riir-train dependency" framing after §3.5 protocol)
**Status:** Open — Phase 0 (modelless unblock) UNVERDICTED, blocks `ac_prefix` promotion
**Origin:** Plan 313 Phase 4 audit (revert of premature promotion in commit `154c0333`)
**Related:** katgpt-rs/.plans/313 (AC-GPT Prefix), katgpt-rs/.benchmarks/313_ac_prefix_goat.md (G1 reformulation analysis), katgpt-rs/.issues/002 (Super-GOAT quality gate), katgpt-rs/.research/295 (paper analysis), katgpt-rs/AGENTS.md (modelless-first mandate), research skill §3.5 (modelless unblock protocol)

## Context

Plan 313 shipped the AC-GPT arbitrary-conditional prefix primitive. The GOAT gate G1 ("AC-GPT conditional logprob matches iterative-MLM to 1e-4") **failed at 7.5e-4** on untrained micro-GPT.

**The failure cause is systematic and characterizable:** AC-GPT intentionally doubles the conditioning signal — each conditioning token `xc` appears both as a copy in region r0 (bidirectional self-attention cluster) AND in-place in region r1 (causal). An untrained model treats both appearances as real signal → biased likelihood.

## Why this issue was revised (the §3.5 lesson)

The original framing of this issue (v1, 2026-06-24) concluded "riir-train dependency" without checking whether the doubled-signal bias could be corrected **modellessly**. This violates the research skill §3.5 modelless unblock protocol (added 2026-06-24 in the same audit). The bias is systematic and characterizable — exactly the case where a deterministically constructed reader-LoRA or latent correction might work without gradient descent.

**The revised framing:** before deferring to riir-train, exhaust the three modelless unblock paths. Only if all three fail is this a genuine riir-train dependency.

## Phase 0 — Modelless unblock investigation (MANDATORY before riir-train deferral)

Per research skill §3.5, check all three paths:

### Path 1: Freeze/thaw snapshot correction

- [ ] Can a frozen snapshot state, thawed at inference, eliminate the doubled-signal bias?
- **Analysis needed:** the bias comes from the attention pattern (r1 eval tokens attend to both r0 copies AND r1 in-place xc). Freeze/thaw preserves weight state; it doesn't change attention topology. **Likely insufficient alone** — but check whether a freeze-state that has learned to down-weight xc (via any modelless construction) can be thawed.
- **Verdict:** [ ] TODO

### Path 2: Raw/lora reader-writer hot-swap (PRIMARY CANDIDATE)

- [ ] Can a **deterministically constructed** reader-LoRA fix the doubled-signal bias?
- **Analysis needed:** `LoraPair { reader, writer }` (Plan 025) supports hot-swap between bidirectional-prefill (reader) and causal-decode (writer) adapters. The question: can we construct a reader adapter IN CLOSED FORM (no gradient descent) that:
  - Scales the value projection contribution from r1 in-place xc positions by 0.5 (compensating for the doubling)?
  - OR zeros out the key/query projection for r1 in-place xc positions (forcing eval tokens to get conditioning ONLY from r0 copies)?
  - OR applies any other analytic correction derived from the known bias structure?
- **Key question:** is the bias linear (clean 2× that a scale-0.5 fixes) or nonlinear (attention softmax normalization makes "just halve it" non-trivial)?
- **Verdict:** [ ] TODO — **THIS IS THE MOST PROMISING PATH**

### Path 3: Latent-space correction

- [ ] Can a dot-product projection + sigmoid gate correct the bias in latent space?
- **Analysis needed:** project the AC-GPT output latent onto a "correction direction" (derived analytically from the bias structure) and gate the output. This is the modelless analog of a trained adapter.
- **Verdict:** [ ] TODO

### Phase 0 exit criteria

- If ANY path passes G1 (AC-GPT ≈ iterative-MLM to 1e-4) → **MODELLESS-VALIDABLE** → implement the correction → re-promote `ac_prefix` to default-on.
- If ALL paths fail → document WHY each failed → proceed to Phase 1 (riir-train dependency).

## Phase 1 — riir-train dependency (ONLY if Phase 0 fails all three paths)

**The Blocking Question:** Does AC-GPT conditional logprob match iterative-MLM conditional logprob to within 1e-4 after LoRA fine-tuning at game-AI context lengths (1024+ tokens)?

### Prerequisites (blocking, Phase 1 only)

- [ ] riir-train plan for AC-GPT LoRA fine-tuning recipe (consume `katgpt_core::ac_prefix::AcPrefix` + `ForwardForAcPrefix` trait).
- [ ] A pretrained base model available to riir-train (Qwen3-1.5B or similar).
- [ ] A corpus where "conditioning on a subset of tokens" is semantically meaningful (game transcripts, dialogue, or code completion).
- [ ] Evaluation harness: AC-GPT single-pass conditional logprob vs iterative-MLM conditional logprob, same base model + LoRA, same conditioning set, tolerance 1e-4.

### Falsifiable prediction (Phase 1)

- **PASS → re-promote `ac_prefix` to default-on in katgpt-rs.**
- **FAIL → the paper's equivalence claim does not hold at game-AI scale. Document as negative result, keep `ac_prefix` opt-in permanently as a fast but non-equivalent conditioning modality.**

## What katgpt-rs already proved (independent of this issue)

The primitive is correct and fast as a modelless mask builder + sequence augmenter:

- **Buffer construction bit-identical to manual reference** (reformulated G1, 0.0 diff).
- **27.258× speedup vs iterative-MLM** (G2, single forward vs 64 forwards).
- **No regression on empty prefix** (G3, 0 mismatches — degenerates to vanilla causal).
- **Zero allocations on hot path** (G4, `attends(i,j)` and `mask.get(i,j,n)` both 0 allocs).
- **Leakage-prevention property** unit-tested in Phase 1 (`attends_three_region_rule_small_example`, `materialize_from_matches_attends_for_all_pairs`).

These results hold regardless of the Phase 0/1 outcome. The primitive is usable today via `--features ac_prefix` for any caller that wants single-pass arbitrary-conditional evaluation.

## Cross-references

- **Research:** `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md` §2.3 (latent reframing), §3 (GOAT verdict).
- **Plan:** `katgpt-rs/.plans/313_AC_GPT_Prefix_Primitive.md` (T3.1 original G1 spec, T4.1 revert note).
- **Bench:** `katgpt-rs/.benchmarks/313_ac_prefix_goat.md` (G1 reformulation analysis).
- **Super-GOAT:** `katgpt-rs/.issues/002_ac_prefix_super_goat_gate.md` (quality gate, separate from this equivalence gate).
- **Modelless unblock protocol:** `katgpt-rs/.agents/skills/research/SKILL.md` §3.5.
- **Paper:** [arXiv:2606.14943](https://arxiv.org/abs/2606.14943) — Lu, Elmoznino, Gagnon, Mittal, Kasetty, Lajoie. AC-GPT. Mila, 12 Jun 2026.
