# Bench 973 — NucleusAccept (PLM relaxed acceptance): accepted/call A/B on fixtures

**Status:** DONE 2026-10-30. Counts are valid. Latency is NOT a measurement (box refused, see below).
**Issue:** [973](../.issues/973_relaxed_acceptance_policy.md) · **Feature:** `nucleus_accept` (opt-in)

## Command

```bash
CARGO_TARGET_DIR=/tmp/i973_target cargo test --release -p katgpt-forward \
  --features nucleus_accept --lib nucleus::ab_throughput -- --ignored --nocapture
```

## Fixture

64 LM-shaped target laws, V=4096: Zipf(s), with s drawn per law in [1.2, 2.4] and the head order
shuffled within the top 8. The drafter proposes the target's rank-r token: r=0 with probability .55,
r=1 with .2, r=2..7 with .15, and a junk token (rank 64..4095) with .1. Each call is a K=8 chain
that stops at the first rejection. There are 20,000 calls per arm, all seeded. The strict arm is
`SoftmaxArgmax` (FLARE Eq 21).

## Result (accepted draft tokens per call, the bonus token excluded)

| Arm | accepted/call | vs strict |
|---|---|---|
| `SoftmaxArgmax` (strict) | 0.482 | 1.00× |
| `NucleusAccept` corner ρ=0 ε=1 | 0.479 | 0.99× (same per-position acceptance probability p(d); only the per-draw decisions differ) |
| `NucleusAccept` ε=2e-3 | 5.129 | 10.6× |
| `NucleusAccept` ρ=0.94 | 4.948 | 10.3× |
| `NucleusAccept` ρ=0.94 ε=2e-3 (paper constants) | 5.177 | 10.7× |

How to read this:
- The corner matching strict is the expected null. It confirms the corner is the exact policy, not
  a free lunch.
- The relaxed arms are capped by the drafter's 10% junk rate. With P(junk-free position) = 0.9, the
  ceiling is E = Σ_{i=1..8} 0.9^i ≈ 5.7. So the ~10× figure is a property of THIS fixture: peaked
  laws and a mostly-head drafter. It is not a transferable speedup. The paper's +46.5% is on real
  models with real drafters.
- No quality metric is reported here, by design. The (ρ, ε) MAUVE-budget calibration is riir-ai
  Issue 1142.

## Latency (NOT a measurement)

`PROVENANCE: power=AC Power load=102.99 swap=1342.44M canary=skipped powermode=2(high)` —
`bench_preflight.sh` **REFUSED**, and the 1-min load average reached ~130 during the run.
Printed step times: strict 85 ns, corner 405 ns, ε 331 ns, ρ 7.2 µs, ρ+ε 1.8 µs. They are
recorded only for the structural cost ladder, which is a code fact and does not depend on load:
the ε term is O(1) (reads p(d)); the corner adds the prefix sum p[..d] plus a CDF walk on
rejection; the ρ term is a full-vocab fold, O(V), per position. A consumer that wants ρ at large V
should feed a top-k buffer (scope extension), not the full row.
