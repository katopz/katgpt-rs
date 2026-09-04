# Bench 700 — cond_audit PoC: per-junction forward-KL + Pinsker TV gate (Issue 719 T1)

**Status:** RECORD — T1 landed (opt-in `cond_audit`); T2–T4 trigger-gated; NO consumer, NO GOAT claim

**Date:** 2026-09-03
**Issue:** `.issues/719_conditioning_consistency_audit_poc.md` (T1 + embedded G8 + G2 only)
**Source:** Research 528 (arXiv:2609.00865 "MemoryWalker") — the modelless conditioning-consistency audit. **The paper's drift figures are theirs, not ours** (their harness, their eviction densities); every number below is from OUR synthetic fixtures.

## What landed

`crates/katgpt-core/src/cond_audit.rs` behind opt-in `cond_audit = ["stale_residual"]` (zero new deps):

- **Seam:** `audit_conditioning(positions, vocab, student, teacher, cfg)` — two `FnMut(u32, &mut [f32])` arms producing next-token LOGITS for the same decode positions. Student = compressed conditioning; teacher = full conditioning (conceptually no-grad — at inference just a second forward). Future consumers (Gemma sliding ring, TokenBudgetPacker, H2O) adapt by wrapping their forwards in the closure contract; katgpt-core depends on none of them. **No consumer wiring exists** (T2–T4 stay `[-]`).
- **Per-junction forward-KL:** `KL(teacher ‖ student)` — forward direction per Research 528: the student is charged for every token the full-context teacher finds plausible, so a deficit ("the model was taught it knows the weather") cannot hide; reverse-KL is mode-seeking and blind on exactly the deficit side. Sum over positions → `eps_kl`.
- **Verdict:** `TV <= sqrt(eps_kl / 2)` (Pinsker, unconditional). Report also carries `tv_bound_chain = sqrt(K · eps_kl / 2)` (triangle+Cauchy–Schwarz telescoped, the safe multi-junction form) and per-junction KLs for exact per-junction Pinsker — `sqrt(eps_kl/2)` is exact only at K=1; the issue's form is the paper's O() convention.
- **Greedy flip counter:** positions where `argmax(student) != argmax(teacher)`, first-index tie-break (the engine's pinned argmax convention).
- **Calibrated-zero arm:** bit-identical arms measure `eps_kl` EXACTLY 0.0 (every `t·(log t − log s)` term is `t·0`) — the compression-off control, and the non-vacuity baseline.

### Substrate composition (not duplication)

The numeric core DELEGATES to the existing `stale_residual::kl_logits` (stable max-shift + log-sum-exp, q-side underflow floored at ln(1e-30) → prohibited tokens read large-but-finite, never +inf). The Research 528 §4 "no KL instrument" claim is thereby refined, not contradicted: a stable categorical KL over logits existed inside the `stale_residual` feature; what was missing — and what 719 adds — is the conditioning-pair seam, the sum → Pinsker-TV verdict chain, the multi-junction honest bound accounting, and the calibrated-zero discipline. `cond_audit` implies `stale_residual`; no code was duplicated.

### Numerics note (why softmax here is house-legal)

The house "sigmoid not softmax" rule targets latent gating / latent→raw projection. Next-token logits ARE a categorical distribution and KL between categoricals requires normalization — log-softmax (max-shift + LSE) is the categorical-KL diagnostic, implemented in the delegated substrate.

## Gates (test binary `cond_audit_poc`, release, deterministic fixtures, CPU)

Fixture: vocab 512, 8 junctions at scattered positions; logits = deterministic SplitMix-keyed 16-dim vocab projection per position (seeded, zero runtime randomness — see the fixture doc for why it is a projection and not a table lookup).

| Gate | Result |
|---|---|
| **Calibrated-zero (control)** | eps_kl = **0.0 exactly** (bit-level), 0 flips, verdict PASS — the numeric floor is zero, not "small" |
| **G8 non-vacuity (planted deficit)** | student drops the teacher's argmax by 12 nats → eps_kl = **49.327026 nats**, tv_bound = **4.966238** ≫ 0.05 threshold, tv_chain = 14.046640, flips **8/8** → verdict **FAIL** — the audit can fail |
| **Monotone response** | noise scale 0.0 / 0.25 / 1.0 / 4.0 → eps_kl = 0.0 / 0.095305 / 1.265889 / 19.705107 nats (strictly increasing) |
| **Determinism ×3** | all report fields bit-identical (`.to_bits()`) across 3 runs — and the deficit arm's eps_kl was also bit-identical across debug AND release profiles (0.695021 under the first fixture, re-measured 49.327026 under the final fixture in both profiles) |
| **Report consistency** | eps_kl = ordered junction sum (bit-exact), max_junction_kl = max, both Pinsker forms match closed formulas, chain ≥ single-junction ≥ per-junction triangle structure |

Debug suite: 5 passed / 1 ignored (G2 release-gated); release: 6/6. Module unit tests 3/3 (`cargo test --features cond_audit --lib cond_audit`).

## G2 — measured audit overhead (release-gated test)

Method: interleaved median-of-ratios over 15 reps × 25 calls, arm A = paired forwards only, arm B = full audit (forwards + KL math + report allocs); ratio-based so box contention hits both arms (the Bench 728 discipline; absolute wall-clock is unusable at this box's load).

**Measured: median audit/forward ratio = 1.487 (overhead fraction 0.487)** against the gate budget 4.0.

Box-load caveat: measured on the shared M3 under sibling cargo load (load avg 60+); the interleaved-ratio shape is load-robust, absolute µs are not and are not asserted. Fixture caveat: the forwards model the MINIMAL real shape (16-dim vocab projection per token) — a real transformer forward costs orders of magnitude more, so 1.487 UPPER-bounds the production fraction (a real serving pair makes the audit ≈ noise). The first fixture draft (pure table-lookup forwards) measured 5.42× — recorded as a fixture artifact: zero-compute forwards do not exist in serving, and a gate against them would police the wrong thing. The 4.0× budget catches pathological audit regressions (per-element allocs, O(vocab²) passes) against the cheapest honest forward.

## Discipline (binding, from Issue 719)

- Opt-in `cond_audit`; default build untouched — `cargo test -p katgpt-core --lib` (default features) after landing: **1992 passed / 0 failed / 7 ignored**, count-identical no-regression (the new module is cfg-gated out of the default build).
- **No default promotion. No GOAT claim — no consumer exists.** Every shipped numeric-compression surface is gated stronger at bit-identity; T2 (Gemma-4 sliding ring), T3 (TokenBudgetPacker), T4 (H2O) stay trigger-gated in the issue until a semantic-compression consumer lands.
- No paper drift figures cited for our surfaces; the synthetic numbers above are fixture-relative and exist to prove the instrument can measure, not to characterize any real model.

## Files

- `crates/katgpt-core/src/cond_audit.rs` — the audit (seam + KL/Pinsker/flip + calibrated-zero docs + 3 unit tests)
- `crates/katgpt-core/tests/cond_audit_poc.rs` — 6 PoC gates (calibrated / G8 / monotone / determinism / consistency / G2 release-gated)
- `crates/katgpt-core/Cargo.toml` — `cond_audit = ["stale_residual"]` + `[[test]]` required-features
- `crates/katgpt-core/src/lib.rs` — feature-gated module wiring

## Addendum 2026-09-04 — trigger-2 disposition (evidence re-gathered; the issue file was removed before it could be recorded)

Trigger 2 of the T2–T4 defer list ("riir-train Plan 343 T1.6 (Gemma-4 ring)") FIRED on 2026-08-25 — T1.6 landed (`maglev_drafter/infer.rs`, riir-train). Disposition: **no cond_audit work is warranted**, for two measured reasons:

1. **The ring is train-time-faithful structure, not inference-time semantic eviction.** The Maglev drafter is consistency-trained WITH the ring (W=512) and self-injection present — the windowed behavior is the intended, trained operating point, not a compression of full-context conditioning. `cond_audit`'s protocol (paired compressed-conditioned forward vs full-context teacher → per-junction KL → Pinsker) targets the opposite case: a model trained on full context whose conditioning gets compressed at inference. A positional ring that simply drops old KV rows makes no per-junction semantic compression decision to audit — the eviction is physical and total, and the audit would measure the training contract, not a defect.
2. **Consumer census (grep 2026-09-04): zero production consumers of `new_all_sliding_bounded`.** Callers are the katgpt-rs unit tests and riir-ai's Issue 752 bit-identity gates (`gemma4_sliding_ring_wrap.rs`) only. The production ring consumer (`maglev_drafter::InferenceCache`) rides `new_gemma4_sliding_bounded` with explicit per-layer kv_dim (d-wide K/V rows) — the generic constructor was the plan's *ask*; the landed consumer needed the per-layer-dim variant.

Reopen conditions are unchanged: a semantic eviction/windowing PR that compresses full-context conditioning at inference (e.g. token merging, learned eviction, H2O — trigger 3), or any consumer that introduces a compressed-conditioning junction on a full-context-trained model.
