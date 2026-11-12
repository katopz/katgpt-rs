# Plan 636 — Issue 997 e-means PoC: split-revival vs farthest-reinit head-to-head

**Status:** CLOSED — T7 NEGATIVE (see the outcome block in issue 997; Research 665 GOAT demoted; the split arm is unreachable in batch Lloyd's — the head-to-head moves to T4's mini-batch lane) — claimed 2026-11-12 (zed-84acf899@amo; the 997 issue row carries the live lease); amo free (the measurement ended), box law satisfied.

**Issue:** [997_emeans_codebook_fit_upgrade.md](../.issues/997_emeans_codebook_fit_upgrade.md) · **Research:** [665_EMeans_EMA_MiniBatch_KMeans.md](../.research/665_EMeans_EMA_MiniBatch_KMeans.md) · **Posture:** bench_375 (K=128, D=32, patch corpus from N_TRANSITIONS=1000 x 16 patches)

## Arm accounting (deviation from the issue's four arms, disclosed)

T1 landed between filing and execution, so "shipped" and "fixed-farthest" are the same
code at HEAD. The split+EMA arm needs T4's (B, a0) grid to mean anything (the paper's
constants are OUT-OF-RANGE at n=16K — Research 665), so it ships with T4, not T2.
T2-as-executed = **two arms**: `farthest` (incumbent, T1-fixed) and `split`
(e-means proportional-split revival). The head-to-head the issue names as its
information value ("the paper's ablation NEVER compared split against farthest-reinit")
is fully covered by these two.

## Tasks

- [x] Claim row on issue 997 pushed BEFORE authoring (b6b85a906)
- [x] `emeans_poc` feature in katgpt-core (opt-in; default byte-identical — the
      Plan-287 G3 lesson; feature-isolation gate both postures)
- [x] Split revival implementation behind the gate: dead cluster seeds from the
      member of the max-inertia alive cluster farthest from that cluster's centroid
      (e-means proportional split); `d2_nearest` refreshed via the T1 helper (same
      staleness class); f64/last-index tie rules preserved
- [x] PoC harness `crates/katgpt-core/tests/bench_997_emeans_split_poc.rs` (release
      posture, isolated CARGO_TARGET_DIR): bench_375-style corpus generation, both
      arms on the SAME seeded corpus; per-arm: final SSE, wall-time, revival count
      (DIAGNOSTIC ONLY — never a veto at unequal SSE)
- [x] T3 stuck-rate fixture: ~50 seeds/arm on the uneven multi-blob fixture
      (two blobs of very different mass + light middle blob); STUCK = final SSE
      > 1% above best-of-arms; report COUNTS
- [x] T6 determinism gate: two runs at the same seed produce byte-identical
      codebooks per arm (BLAKE3 over the canonical codebook payload); sequential
      f64 accumulation asserted
- [x] G1 no-regression: default build byte-identical (feature off) + the bench_375
      factorized-vs-monolithic margin holds on the gated path
- [x] T7 verdict: raw numbers into Research 665 "PoC Addendum" in the SAME commit
      as the outcome; split beats farthest on SSE at <= wall-time AND stuck-rate
      improves => GOAT stands (promote winner behind its flag); otherwise strike
      the GOAT grade in 665 (demote), keep only T1, record the negative
- [x] Commit + push (claim folded to lease-closed at landing)
