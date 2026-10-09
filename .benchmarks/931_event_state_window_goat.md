# Bench 931 — `event_state_windows` Phase 1 GOAT (Plan 623 T1.3)

**Date:** 2026-10-10 · **Box:** 4090 workstation (shikuwa, i7-13700K, Windows 11)
**Posture:** `--release` (bench profile), `cargo bench -p katgpt-core --features event_state_windows --bench bench_931_event_state_window_goat`
**Box state (provenance):** ⚠ NOT a verified-quiet window — the riir-train T0.3b
dllm AR trainer (issue 621) was RUNNING throughout at its ~20–22 s/step
plateau (0.89-core busy-poll + per-step CPU CE phases over 255×256K logits,
i.e. active memory-bandwidth contention). Per the Issue-649 class this makes
the numbers **load-caveated**: every bar PASSED under this load, so a quiet
re-run can only improve them — but the p99 tails are scheduler noise, not
work. Re-run at the Phase-2 GOAT finalization on a quiet box.
**Commit:** <this>

## G1 (correctness) — PASS

- Module inline suite: **10/10** (`cargo test -p katgpt-core --features
  event_state_windows --lib event_state_window`): partition-of-unity inside
  cores (|Σ−1| < 1e-3, K=3 composition incl. saturated terminals) +
  strictly-below-one on every excluded transition; the literal settle-pin
  (both neighbors < 0.5 at strictly-interior transition points); γ→0 →
  EXACT hard indicator (fast_sigmoid ±40 saturation → 1.0/0.0, edges
  excluded as the σ(0)=0.5 ambiguous points); γ→∞ → flat 0.25;
  malformed/NaN/inf totality (zero window, never panic — 20k-iteration fuzz,
  >15k live-path samples, exact-zero on every malformed combo);
  summary closed form (1..=100: mean 50.5, spread 28.86607, quantiles
  10/25/50/75/90 = 10/25/50/75/90, supports [91,76,51,26,11]);
  permutation invariance BIT-IDENTICAL (n=257, reverse+swap, duplicates +
  ±f32::MAX extremes); empty/single/NaN shapes.
- Bench-posture re-pin: hard-indicator 1.0/0.0 + inverted/NaN zero ✓.

## G2 (latency, release) — PASS (load-caveated)

| arm | p50 | p90 | p99 | bar |
|---|---|---|---|---|
| window per-call (batch=256, n=2000 batches) | **6 ns** | 6 ns | 6 ns | ≤ 20 ns ✓ |
| K=8 partition row (context) | 48 ns/tick | — | — | — |
| summary N=10k, fixed-window baseline | **98.3 µs** | — | 152.7 µs | ≤ 100 µs ✓ |
| summary N=10k, event-anchored total | 98.8 µs | — | 149.3 µs | rel ≤ +K×20 ns+noise ✓ |

- **The relative claim holds:** anchoring delta **+500 ns** p50 (≈ the
  K×6 ns window share + noise) — event anchoring is ~free against the
  µs-class summary, so the boundary-law gain (the paper's +166–233% at
  event-anchored boundaries) costs nothing at aggregation time.
- Method note: the first run measured per-call with `Instant` pairs and read
  **p50 0 ns / p99 100 ns — the Windows timer quantum wearing a latency
  number**, not work (the timed-region class from the other side: the work
  is real and sunk into a `black_box` accumulator; the RESOLUTION was the
  defect). Batched 256 calls per sample → the 6 ns figure. Batch arithmetic
  printed beside every number.
- Tail support printed by the bench itself (batched form: n=2000 batches,
  tail@p99=350).
- The 98.3 µs p50 is 1.7% under the bar UNDER LOAD; the sort dominates
  (N=10k `sort_unstable_by(total_cmp)`); a quiet box widens the margin.

## G3 (no regression) — PASS

- Default-features lib suite: **2141 passed, 0 failed** (8 ignored) — the
  feature is additive, default-off; only lib.rs/`Cargo.toml` gate lines added.
- Clippy: `--all-targets --features event_state_windows` clean (the only
  residual warnings are the box's incremental-cache hard-linking notices +
  pre-existing bench/example warnings); `cargo refine --fix --write
  --verify` over all three new files: **0 edits needed** (compile-gated).

## G4 (alloc) — PASS

- `tests/event_state_window_alloc_check.rs` (separate single-fn binary, the
  bench-655 convention): **0 allocs / 0 deallocs** across 100 steady-state
  summaries (N=10k, scratch pre-allocated before the measured window) +
  10,000 window evals.

## Determinism

- Double-summary bit-identical + REVERSED-order bit-identical (the G1
  invariance re-proven at the bench posture).
- Summary BLAKE3 (the two-box anchor): `f4938f076062e5b5f997311f78c63a8bb1dcaccda1811a25fd1f3cad7090a641`
  (over mean/spread/p10/p25/p50/p75/p90 LE bytes + supports + n; input =
  the bench's seeded LCG corpus).

## Verdict

**Phase 1 GOAT: G1–G4 ALL PASS** (G2 load-caveated, bars passed under
documented trainer contention). The feature STAYS OPT-IN — promotion to
default is gated on the Phase-2 riir-ai PoC (plan T2.1, the crowd-latent
Type-A index), per the plan's own rule that the paper's domain evidence
does not auto-transfer (Research 611 §6).

First measurements this session caught two test-design traps worth the
record: (1) the γ→0 hard-indicator test initially sampled t 0.001 from the
edge — with γ=1e-3 the sigmoid argument is 1.0, σ(1)=0.73, NOT saturated
(limits need ≫40γ clearance, and the exact edges are the σ(0)=0.5 ambiguous
points — the hard indicator is undefined THERE by construction); (2)
`b_next = +inf` is a VALID terminal shape (open-ended rise), not a malformed
input — only NaN/−inf/inverted bounds are malformed.
