# Issue 976: JetSpec++ outcome-distribution speculation cache — modelless fusion idea (novelty TBD, posture-gated)

**Status:** OPEN — fusion idea filed per Research 641 (§4 modelless track); not armed. Arming condition: a pipelined or multi-request speculative-serving posture exists in the fleet (riir-infer `disaggregated.rs`/`multi_host.rs` lane opens, or a real multi-request batch caller for Plan 339).

> **Source:** arXiv:2610.19792v2 "JetSpec++" (Research 641, this session). Every mechanism cell is published — this issue is an ENGINEERING combination onto shipped katgpt-rs substrate, filed novelty-TBD by design; closest published cells: D2SD (arXiv:2606.04446 — k-distribution from draft confidences + second draft model), SSD (ICLR 2026 — speculation cache, fixed geometric fan-out), JetSpec++ itself (per-round adaptive selection + conformal threshold).

## The idea

Generalize the shipped acceptance machinery from a **marginal** to an **outcome distribution**, and spend a cache budget against it:

1. **Outcome distribution** `P(k, a) = Πᵢ≤ₖ p̂ᵢ · p̂(a|k)` (JetSpec++ Eq 6; Prop 1: exact under speculative sampling with a perfect proxy) — per-position draft confidences chain-multiplied (one `cumprodsum` pass — shipped SIMD primitive), bonus distribution from a k-indexed histogram (M7: freeze/thaw-legal fixed-size Pod, BLAKE3-committed, KL-convergence-gated).
2. **Calibration**: split-conformal threshold λ on the outcome scores (shipped `conformal_predictive_intervals` substrate; calibration affects only what is pre-drafted, never the verified output) + optional 6-feature logistic recalibration (hand-prior weights or deterministic offline IRLS — degraded mode keeps AUC loss measurable).
3. **Selection**: top-B outcomes by score (Eq 7: disjoint events → hit mass additive → sort-by-score optimal for fixed B); **Prop 2b** — per-round posterior selection ≥ ANY fixed allocation (SSD's geometric fan-out included), measured by JetSpec++ Table 11. This is also one-line validation that Plan 339's per-round greedy admission is the right SHAPE on the verification-budget side.
4. **Consumers**: `AcceptanceSurrogate`/`AcceptanceForecast` (Bebop) as the confidence producer; `HardwareAwarePrefixScheduler` (Plan 339) for the multi-request allocation variant; `bigram_markov` as a possible modelless confidence source.

## Why not now

- Single-stream batch-1 CUDA-graph league serving gains nothing from a pre-draft cache (one outcome is realized per round; no second device to draft on).
- D2SD already publishes the draft-confidence k-distribution; SSD publishes the cache; the delta is conformal calibration + per-round dominance + our substrate — real but engineering-class until a posture pays for it.

## Gates (when armed)

Paired hit-rate vs a fixed-allocation cache at equal B on a frozen trace (strict ≥ required — Prop 2b is a dominance claim); conformal coverage at nominal rate; forecast-vs-realized E[accepted] (MAE/CRPS **vs the conformal-naive floor** — UQ "Report the Floor" extension); G4 alloc-free; Issue-940 determinism (first-index tie-break on (k,a)); M6's staleness ladder reported (stale → predicted → true), distribution not pooled.

## References

- Research 641 (`../.research/641_JetSpecPP_Foresight_Pipelined_Self_Refining_Drafts.md`) — full distill + track table
- Research 316 (DSpark) · Research 638 (pipelining cell table) · Plan 339 · riir-infer Issue 028 (disaggregated handoff)
