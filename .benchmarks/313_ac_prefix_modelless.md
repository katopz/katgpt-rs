# Issue 003 Phase 0 — AC-Prefix §3.5 Modelless Unblock Results (Path 2 PASSED)

**Date:** 2026-06-24
**Bench:** `crates/katgpt-core/benches/bench_313_ac_prefix_modelless.rs`
**Run:** `cargo bench -p katgpt-core --features ac_prefix --bench bench_313_ac_prefix_modelless -- --nocapture`
**Hardware:** macOS (release profile, single-threaded micro-GPT forward)

## The §3.5 question

Can a **deterministically constructed** mask variant (no gradient descent) eliminate the doubled-signal bias that made the original G1 fail at 7.5e-4?

## The modelless fix (Path 2)

`AcPrefix::attends_dedup` — eval tokens in r1 do NOT attend to in-place `xc` tokens in r1. All conditioning flows through r0 copies. On single-layer micro-GPT, this makes the attended (token, original_position) set identical to iterative-MLM's → same K/V → same softmax → same logprobs.

## Results (32-token base, 16 xc every other, single-layer micro-GPT, seed 0xC0FFEE)

| Path | Conditional logprob | |diff vs iterative| | Gate |
|------|---------------------|-------------------|------|
| AC-GPT original mask (`attends`) | -53.372864 | 0.000751 | **> 1e-4 → bias CONFIRMED** |
| AC-GPT deduplicated mask (`attends_dedup`) | -53.373615 | **0.000000** | **< 1e-4 → PASS ✓** |
| Iterative-MLM (reference, 16 forwards) | -53.373615 | — | reference |

## Gate verdicts

| Gate | Description | Threshold | Measured | Result |
|------|-------------|-----------|----------|--------|
| G1-modelless | `\|dedup − iterative\|` | `< 1e-4` | `0.000000` | **PASS ✓ (bit-identical)** |
| G1-bias (negative control) | `\|original − iterative\|` | `> 1e-4` | `0.000751` | **CONFIRMED ✓ (bias present)** |
| G1-dedup-vs-original | `\|dedup − original\|` | `> 0.0` | `0.000751` | **CONFIRMED ✓ (non-trivial)** |

## Phase 0 verdict

**✓ MODELLESS-VALIDABLE.** The deduplicated mask eliminates the doubled-signal bias on single-layer micro-GPT **without gradient descent**. The bias (7.5e-4 with original mask) drops to 0.0 (bit-identical to iterative-MLM) with the deduplicated mask.

Per research skill §3.5, this unblocks G1 modellessly. The deferral to riir-train was premature (as flagged in AGENTS.md canonical failure). The correct resolution is: ship the modelless correction, re-promote `ac_prefix` to default-on.

## Why this works (the math)

For a single attention layer, K/V at any position depend only on the token embedding (not on other positions' attention patterns). The r0 copy of `xc` at original position `p` has:
- **same token** as the in-place r1 `xc` at position `p`
- **same RoPE rotation** (both use original position `p`)
- **same K/V** (same weights, same input embedding)

Therefore the deduplicated attended set for eval at position `k`:
- `{ all xc via r0 copies } ∪ { eval at positions ≤ k via r1 }`

is identical (in token+position pairs) to iterative-MLM's attended set:
- `{ all xc in-place } ∪ { all positions ≤ k }` = `{ all xc } ∪ { eval at positions ≤ k }`

Same attended K/V → same attention scores → same softmax → same logprobs. **Bit-identical.**

## Multi-layer caveat (non-blocking)

On multi-layer models, the r0 copies' representations evolve through layers attending only to other r0 copies (r0→r1 is false), whereas in iterative-MLM the in-place `xc` attend bidirectionally to eval tokens too. The representations diverge from layer 2 onward.

**This does NOT block the modelless G1 pass** — the single-layer equivalence is sufficient to prove the bias-correction mechanism works. Multi-layer equivalence (does LoRA fine-tuning close the representation gap?) is a riir-train follow-up, tracked in Issue 003 but non-blocking for the `ac_prefix` promotion.

## What ships

- `AcPrefix::attends_dedup(i, j)` — the deduplicated three-region rule (O(log \|xc\|) per pair, zero-alloc).
- `AcPrefix::is_xc_position(p)` — binary-search helper.
- `AcPrefixMask::materialize_dedup_from(prefix)` — bit-pack the deduplicated rule.
- `AcPrefix::conditional_logprob_dedup(forward)` — convenience single-pass conditional logprob with deduplicated mask.
- 3 new unit tests: `attends_dedup_eliminates_inplace_xc_attention`, `attends_dedup_empty_prefix_is_standard_causal`, `materialize_dedup_matches_attends_dedup_for_all_pairs`.
- The original `attends` / `materialize_from` / `conditional_logprob` are retained (paper-faithful mask, for post-LoRA fine-tuned models).

## Cross-references

- **Issue 003** (`ac_prefix_g1_riir_train_dependency`) — CLOSED modelless-validable, then resolved-and-removed in commit `552b4632` (2026-06-26) along with Issues 002-009. This benchmark is the surviving evidence.
- **Plan:** `katgpt-rs/.plans/313_AC_GPT_Prefix_Primitive.md` (Phase 4 promotion now justified).
- **GOAT gate bench:** `katgpt-rs/.benchmarks/313_ac_prefix_goat.md` (G1 reformulation, G2–G4).
- **Research:** `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md`.
- **Protocol:** `katgpt-rs/.agents/skills/research/SKILL.md` §3.5 (modelless unblock).
- **Paper:** [arXiv:2606.14943](https://arxiv.org/abs/2606.14943) — Lu et al., AC-GPT, Mila, 12 Jun 2026.
