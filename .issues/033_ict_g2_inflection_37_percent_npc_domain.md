# Issue 033 — ICT G2 inflection sits at 37.5% on synthetic-NPC suite, not 10%

**Date:** 2026-06-19
**Plan:** 294 (ICT Distributional Branching-Point Detector) Phase 3 T3.3
**Severity:** Medium — informative, NOT a blocker for G3
**Status:** Documented; k_percent sweep recommended for NPC-scale consumers

## Summary

Plan 294 Phase 3 GOAT Gate G2 asserts the median inflection location of the
JS-uniqueness curve sits in `[5%, 20%]` (paper §A.4.1 reports ~10% on LLM
token distributions). On the synthetic-NPC decision suite shipped in
`tests/bench_294_ict_g2.rs`, the **median inflection location is 37.5%**
(IQR 25% – 50%) — well outside the 10% band.

This is the failure mode Plan §Risks anticipates:

> "G2 fails (no 10% inflection) — The 10% is LLM-token-specific. Sweep k%
> to find our inflection. May be 20-30% for NPCs. Document in T3.3."

(37.5% is above the 20-30% range Plan §Risks conjectured, but the
qualitative finding — paper's 10% does not transfer to NPC-scale synthetic
workloads — is confirmed.)

## Result

```
Decision points: 1000, K=8, action_dim=6
Regime mix: committed=566 (56.6%), undecided=290 (29.0%), noise=144 (14.4%)

Inflection-location histogram:
   0.1 | ██████████████████████████████           (243)
   0.3 | ████████████████████                     (159)
   0.4 | █████████████                            (107)
   0.5 | ████████████████████████████████████████ (316)
   0.6 | ███████████                              (94)
   0.8 | ██████████                               (81)

Median = 0.3750 (37.5%)   IQR = [0.25, 0.50]
```

The distribution is multi-modal: a peak at 10% (the paper's band — 243 / 1000
decision points do show the LLM-style inflection), a peak at 50% (316 / 1000 —
no clean inflection), and spread in between. This is consistent with the
regime mixture: "committed" decisions have a sharp 1-vs-7 inflection at 12.5%;
"undecided" and "noise" decisions have flatter uniqueness curves with the
second-difference maximum landing mid-range.

## Interpretation

The paper's 10% rule is calibrated on **LLM next-token distributions** where
the vocabulary is large (32K+) and the long-tail structure is sharp. NPC
action distributions are typically small (6-32 discrete actions) and the
"committed vs undecided" mixture is less bimodal than "obvious-token vs
long-tail-token".

**Implication for `BranchingDetector` callers:**
- The default `k_percent = 0.10` (paper-recommended) is fine for LLM workloads.
- For NPC-scale workloads, callers should sweep `k_percent` and pick the
  empirical inflection. This Issue's measurement suggests `k_percent ≈ 0.38`
  for the synthetic mixture shipped here; real game workloads will differ.

## Mitigation

The math primitives (`collision_purity`, `js_divergence`, `BranchingDetector`)
work correctly at any `k_percent` — the API exposes it as a constructor
parameter precisely so callers can tune per-domain. No code change needed.

The default `k_percent = 0.10` in `BranchingDetector::new` examples stays at
the paper's value; callers wanting the synthetic-NPC value pass `0.38`
explicitly.

## Does this block G3?

**No.** Per Plan §Implementation Order, the Super-GOAT-vs-Gain decision point
is after G3 (orthogonality to H_1), not after G2. G2 borderline-fail just
means we sweep `k_percent` empirically rather than hard-coding 0.10. G3 still
runs and still decides the verdict.

## Does this block shipping?

**No.** T1 (primitives), T2 (G1 paper-proof), T6 (Bebop H_1→H_2 upgrade),
T7 (Curiosity Pulse spec), T8 (docs) all ship regardless of G2 outcome. The
plan explicitly states "Keep T6 (H_1→H_2 upgrade is independently valuable)"
even if the upstream gates fail.

## Follow-up

- If real-game NPC data becomes available (riir-ai Plan 324 / 314), re-run
  G2 on it. The synthetic mixture here may not be representative.
- If G3 also fails (Spearman ρ(H_1, JS-uniqueness) ≥ 0.5), the entire
  Super-GOAT verdict downgrades to Gain per Plan §Phase 4 Downgrade path.
- The `k_percent` sweep should be a one-liner caller-side override; no
  module-level change required.

## References

- Plan 294 §Phase 3 T3.3, §Risks row "G2 fails", §Implementation Order
- Research 270 §1.4 (10% empirical), §A.4.1 (paper measurement)
- arxiv 2606.19771 §A.4.1
- Test: `tests/bench_294_ict_g2.rs`
- Bench doc: `.benchmarks/294_ict_g2.md`
