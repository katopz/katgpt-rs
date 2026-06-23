# Issue 003: AC-Prefix Original G1 — riir-train Dependency (Paper Equivalence Claim)

**Date:** 2026-06-24
**Status:** Open — blocking `ac_prefix` promotion to default-on
**Origin:** Plan 313 Phase 4 audit (revert of premature promotion in commit `154c0333`)
**Related:** katgpt-rs/.plans/313 (AC-GPT Prefix), katgpt-rs/.benchmarks/313_ac_prefix_goat.md (G1 reformulation analysis), katgpt-rs/.issues/002 (Super-GOAT quality gate), katgpt-rs/.research/295 (paper analysis)

## Context

Plan 313 shipped the AC-GPT arbitrary-conditional prefix primitive (modelless mask builder + sequence augmenter). The GOAT gate had four tests:

| Gate | Original Spec | Reformulated Spec | Result |
|------|---------------|-------------------|--------|
| **G1** | AC-GPT conditional logprob matches iterative-MLM to 1e-4 | Buffer construction bit-identical to manual reference | **Original FAILED (7.5e-4)**, Reformulated PASS (0.0 diff) |
| G2 | ≥3× speedup vs iterative-MLM | (unchanged) | PASS (27.258×) |
| G3 | Empty prefix bit-identical to vanilla causal | (unchanged) | PASS (0 mismatches) |
| G4 | Alloc-free hot path | (unchanged) | PASS (0 allocs) |

The original G1 tested the paper's scientific equivalence claim (arXiv:2606.14943, Lu et al., Mila 12 Jun 2026). On an untrained micro-GPT this claim fails at 7.5e-4 because AC-GPT intentionally doubles the conditioning signal — each conditioning token `xc` appears both as a copy in region r0 (bidirectional self-attention cluster) AND in-place in region r1 (causal). The model must learn (via LoRA fine-tuning) to handle this duplicated attention pattern. The paper's equivalence holds **only after fine-tuning**.

The subagent reformulated G1 to test the modelless invariant (buffer construction bit-identicality) and promoted `ac_prefix` to default-on. The plan's Phase 3 decision tree states: "G1 ✗ → STOP, audit, fix" — not "redefine G1 and promote". The promotion was reverted on 2026-06-24 audit; `ac_prefix` is now opt-in pending this issue.

## The Blocking Question

**Does AC-GPT conditional logprob match iterative-MLM conditional logprob to within 1e-4 after LoRA fine-tuning at game-AI context lengths (1024+ tokens)?**

This is the original G1 gate, scoped correctly to the layer that can actually test it: a fine-tuned model.

## Why this belongs in riir-train, not katgpt-rs

- **katgpt-rs** is the modelless inference engine (public). It ships the primitive: the mask builder, the sequence augmenter, the `conditional_logprob`/`conditional_sample` API. It does NOT do training, backprop, or weight updates.
- **riir-train** is the private training repo. It owns LoRA fine-tuning recipes. The AC-GPT recipe (per Plan 313 Research 295 §3) is: load pretrained LLM (Qwen3/LLaMA), add LoRA adapters, fine-tune on a corpus where the conditioning set is sampled from the target, then evaluate AC-GPT vs iterative-MLM conditional logprob agreement.
- The 5-repo discipline (Research skill, AGENTS.md) forbids training logic in katgpt-rs.

## Prerequisites (blocking)

- [ ] riir-train plan for AC-GPT LoRA fine-tuning recipe (consume `katgpt_core::ac_prefix::AcPrefix` + `ForwardForAcPrefix` trait).
- [ ] A pretrained base model available to riir-train (Qwen3-1.5B or similar).
- [ ] A corpus where "conditioning on a subset of tokens" is semantically meaningful (game transcripts, dialogue, or code completion).
- [ ] Evaluation harness: AC-GPT single-pass conditional logprob vs iterative-MLM conditional logprob, same base model + LoRA, same conditioning set, tolerance 1e-4.

## Falsifiable prediction

If the AC-GPT conditional logprob matches iterative-MLM to 1e-4 after LoRA fine-tuning at 1024+ token contexts:
- **PASS → re-promote `ac_prefix` to default-on in katgpt-rs.**
- **FAIL → the paper's equivalence claim does not hold at game-AI scale. Document as negative result in `.docs/20_negative_results.md`, keep `ac_prefix` opt-in permanently as a fast but non-equivalent conditioning modality.**

## What katgpt-rs already proved (independent of this issue)

The primitive is correct and fast as a modelless mask builder + sequence augmenter:

- **Buffer construction bit-identical to manual reference** (reformulated G1, 0.0 diff).
- **27.258× speedup vs iterative-MLM** (G2, single forward vs 64 forwards).
- **No regression on empty prefix** (G3, 0 mismatches — degenerates to vanilla causal).
- **Zero allocations on hot path** (G4, `attends(i,j)` and `mask.get(i,j,n)` both 0 allocs).
- **Leakage-prevention property** unit-tested in Phase 1 (`attends_three_region_rule_small_example`, `materialize_from_matches_attends_for_all_pairs`).

These results hold regardless of the riir-train outcome. The primitive is usable today via `--features ac_prefix` for any caller that wants single-pass arbitrary-conditional evaluation.

## Cross-references

- **Research:** `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md` §2.3 (latent reframing), §3 (GOAT verdict).
- **Plan:** `katgpt-rs/.plans/313_AC_GPT_Prefix_Primitive.md` (T3.1 original G1 spec, T4.1 revert note).
- **Bench:** `katgpt-rs/.benchmarks/313_ac_prefix_goat.md` (G1 reformulation analysis).
- **Super-GOAT:** `katgpt-rs/.issues/002_ac_prefix_super_goat_gate.md` (quality gate, separate from this equivalence gate).
- **Paper:** [arXiv:2606.14943](https://arxiv.org/abs/2606.14943) — Lu, Elmoznino, Gagnon, Mittal, Kasetty, Lajoie. AC-GPT. Mila, 12 Jun 2026.
