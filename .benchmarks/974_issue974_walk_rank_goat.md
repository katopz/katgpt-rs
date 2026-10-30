# Bench — Issue 974 (T3): walk_rank GOAT — the katgpt-core upstream copy

**Status:** RECORD — G2 PASS (defaults ≈ 0.28 ms/call on the synthetic view, 36× under the 10 ms issue budget); G1 asserted on EVERY timed iteration; G4 alloc gate green in-module.
**Feature:** `walk_rank` (opt-in, katgpt-core). Bench source: `crates/katgpt-core/benches/walk_rank_goat.rs` (criterion, harness=false, required-features `walk_rank`).
**Run:** `cargo bench -p katgpt-core --features walk_rank --bench walk_rank_goat` (RELEASE, bench profile).

## Box state

- Apple M3 Max, 16 cores, macOS 27.0.1, AC power, charged (Zed session report).
- Concurrent sibling agents active in OTHER repos (seal-render; riir-refine Batch 320); this lane built in an isolated `CARGO_TARGET_DIR=/tmp/walk_lift`. Criterion estimates over 100 samples; the fixture is deterministic (LCG construction — the unseeded-global-RNG gate discipline; the ranker's own rng is per-config seeded).
- Synthetic view = ring backbone + 2 LCG shortcut chords per node (4 listed neighbors/node — HIGHER degree than the ndb graph fixtures, so per-call cost here is an upper bound on the ndb bench 517 shapes).

## G2 — latency (criterion median of the per-batch estimates; batches = 8 spread queries)

| group | time (median) | per call | vs 10 ms bar |
|---|---|---|---|
| `g2_latency/defaults_w32_k8_n100k` | 2.22 ms | ≈ 0.28 ms | **36× headroom** |
| `g2_latency/sensitivity_w16_k8_n100k` | 0.97 ms | ≈ 0.12 ms | — |
| `g2_latency/sensitivity_w8_k4_n100k` | 0.19 ms | ≈ 0.023 ms | — |

**Verdict: PASS** — matches the ndb bench 517 shape (2.3×/12× cheaper at the two smaller postures — 2×/8× fewer steps — vs ndb's 3.4×/6.2×). The upstream copy adds ZERO overhead over the ndb measurements (the trait boundary is the same static dispatch).

## G1 — determinism, asserted in the timed loop

`g1_determinism/rerun_identity_w32_k8_n10k` (1.90 ms/batch): the fixture's ranking is captured ONCE, verified byte-identical on a re-run BEFORE timing, then **asserted `assert_eq!` on EVERY timed iteration** — a regression to non-determinism fails the bench, not just a number. Loud-zero: every group asserts a non-empty warmup ranking first.
`g1_determinism/budgeted_context_defaults_n10k` (88 µs/batch ≈ 11 µs/call): stage-2 priced (the ranked list × default budget 64/depth-2).

## G4 — alloc (in-module test, not a bench group)

`walk_rank::tests::alloc_per_call_is_bounded_not_per_step` (the counting allocator; green in `cargo test -p katgpt-core --features walk_rank --lib`): one stage-1 + one stage-2 call at defaults allocate ≤ 16 each against 256 walk steps — the once-per-call law, never per-step. Runs in dev by the Issue-741 gate (`any(debug_assertions, alloc_tracking)`).

## In-module test suite (T2)

6 tests green at the feature: `deterministic_reruns`, `temporal_guard_leak_free_and_insert_invariant` (future-insert invariance + non-vacuity), `budget_accounting_exact_and_capped`, `dangling_neighbors_skipped` (listing + query sides), `alloc_per_call_is_bounded_not_per_step`, `hub_dominates_on_visit_count` (the ndb T1 mechanism at fixture scale — the hub strictly dominates on visit COUNT, never tiebreak). Clippy `-D warnings` clean at default and `--features walk_rank` postures.

## Verdict

Issue 974 T3: **PASS**. The upstream copy is proven at the same bars as the ndb landing (516 mechanism / 517 latency); the ndb re-export flip (974 T1's consumer half) landed in the same arc — ndb `walk_rank` re-exports `katgpt_core::walk_rank` at the same paths, adapters stay consumer-side, and the full ndb suite (512 tests) + bench re-ran green THROUGH the shim.
