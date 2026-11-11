# Issue 991: Attractor fixed-point early exit for latent-thought kernels

**Status:** Open — small latency lever (Research 662, reframed at verdict round 1); fixture gate pre-registered below
**Claimed:** zed-m3-idle-991, 1794362080, expires 2026-11-13T01:54:40+00:00
**Date:** 2026-11-09
**Origin:** katgpt-rs Research 662 ([arXiv:2610.42149](https://arxiv.org/abs/2610.42149) LARC) motivated the look at step-to-step geometry on latent-thought kernels, but the mechanism here is NOT LARC's: `LatentThoughtKernel` is an **attractor iteration** (its module doc: one observed signal "should move the belief further toward a fixed point than a single attractor step would") — successive cosine approaching 1 on Family B is the kernel *doing its job*, not a failure. LARC's collapse (pairwise cos 0.97 under RL pressure, Table 12) is a different failure mode that requires a consumer reading **every** step separately — no such consumer exists in this stack (the kernel's callers read only the final state). What survives is a standard convergence-detection lever the reframe surfaced: **stop iterating once the iteration has effectively converged.**

## Why an issue, not a plan

The early-exit test is closed-form (residual norm between successive states, µs-scale) but its **value is unmeasured**: K is caller-set (benches run K=1/3/5; callers may set larger K) and the saving is only the tail steps after convergence. If most real inputs converge before K, this is a free latency cut that grows with K; if not, it closes negative in one bench run. The census MUST cover the K values callers actually set — the savings scale with K, so a "converges before K" result at K=3 says little about the saving at the larger-K postures.

## Mechanism

- **Convergence detector:** residual `‖s_{t+1} − s_t‖` (or `1 − cos(s_t, s_{t+1})`) below `ε` for `w` consecutive inner iterations ⇒ the fixed point is effectively reached; break the K-loop and return `s_{t+1}` (the returned state differs from the full-K run by at most `ε`-scale — declare the tolerance in the API, never hide it).
- **Where:** `LatentThoughtKernel::step` (and any Family A caller iterating manually), behind an opt-in knob; disabled ⇒ byte-identical full-K loop.
- **Census arm (report-only, run first):** `KATGPT_BELIEF_CONVERGENCE_CENSUS=1` logs per-input converged-at-step over the existing micro-belief bench families; answers the premise question before any behavior change.

## Signal-diff vs shipped cousins (why this is not already shipped)

| Cousin | Signal | This issue |
|---|---|---|
| `AttractorKernel::step_with_precomputed_wx` | reuses the invariant input matvec across K (a *compute* reuse) | *iteration-count* cut once the state stops moving |
| `cgsp::EntropyCollapse` / `CollapseDetector` | output-distribution entropy of the sampling loop | state residual between successive attractor iterations |
| LARC Table 12 collapse (motivating cousin only) | mean **pairwise** cosine over trace steps read separately downstream | successive residual on a kernel whose consumers read only the final state — different failure channel, different statistic (residual norm, not pairwise cosine; τ=0.97 does not carry over) |

## Tasks

- [ ] T1 Census arm: `KATGPT_BELIEF_CONVERGENCE_CENSUS=1` report of converged-at-step over the bench families (`micro_belief_bench`, coherence families A/B) AND at enlarged K postures beyond the bench trio (the stack's callers set K; the census sweeps K until the converged-at distribution stops changing); report-only, byte-identical when unset.
- [ ] T2 Fixture gate (pre-registered): a synthetic recurrence at a known contraction rate converges at the analytically-predicted step (detector must fire within ±1); a non-contracting sequence MUST NOT trip within K; detector cost ≤ 0.5 µs/call at dim 32 (one norm + compare).
- [ ] T3 Census verdict: if inputs do not converge before K on real bench distributions, record the negative and CLOSE this issue (premise failed honestly). If they do: open the early-exit follow-up with the measured converged-at distribution as the expected-savings bar.
- [ ] T4 (conditional on T3-positive) Early-exit arm: break at the detected step; **tolerance-declared** returned state (bit-identity vs full-K is NOT claimed — the returned state may differ by ≤ε; the API must name the tolerance). Paired gate: quality-neutral on coherence benches (LB95) + latency delta; GOAT gate before any default promotion, demote the loser.

## Non-goals

- No step-diversity/collapse machinery — no consumer in this stack reads intermediate kernel states (LARC's failure channel); if one ever lands, file a new issue with the pairwise-cosine statistic and LARC's Table-12 signature as its gate.
- No batch-level diversity machinery — owned by Plan 634 / riir-ai Plan 646 (Research 660).
