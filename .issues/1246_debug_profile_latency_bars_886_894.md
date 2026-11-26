# Issue 1246: two debug-profile latency-bar suites fail on amo — bench_886 G2a axpy + bench_894 kvarn zip gates (pre-existing, proven at clean HEAD)

**Status:** Open — filed 2026-11-26 (zed-riir-refine-idle@amo) during the i1245-followup lint-drain verification.
**Class:** debug-profile latency bars that their own `cargo test` (debug) posture cannot hold on this box — the "profile is part of the claim" trap one layer down: the bars were calibrated (presumably release or on another box) and the suites run and FAIL at the default debug profile on amo (M5 Ultra, AC, quiet at run time).

## The two suites

- **`crates/katgpt-core/tests/bench_886_differential_anchor_goat` G2a** — `axpy @ d=512` bar 1.0 µs; measured 2.38–2.61 µs/call at debug on amo, reproducibly, ALONE (not load noise). The in-file comment says a single 512-wide axpy is "below timer resolution in release" — the batch-of-64 harness implies a RELEASE-calibrated bar, but the suite ships in the plain `cargo test` (debug) sweep and fails there. Reproduced at clean HEAD `8619ecffa` in a detached worktree (2.5690 µs) — NOT caused by the lint drain (`13c62d2d3`).
- **`crates/katgpt-kv/tests/bench_894_kvarn_dequant_zip_goat`** (feature `kvarn`) — 6 G2 timing gates (old-vs-new sequential debug-timed arms, 4096-position passes): value-4-bit "≥10% faster" gate reads 1.00x, and the five no-regression gates read 0.92x/0.25x/0.38x. The 4x-class key-arm ratios are far outside any load-noise band. Reproduced at clean HEAD `8619ecffa` (same 6 gates, same magnitudes) — pre-existing.

## What this issue asks

1. **Adjudicate the intended profile per suite** (the AGENTS.md profile-matrix law): if the bars are release-calibrated, gate the runs (`#[cfg_attr(debug_assertions, ignore)]` is the riir-train precedent) or re-pin as debug bars measured on a quiet box with box-state provenance; if debug-calibrated, re-pin from a fresh quiet-box measurement (the sequential-A/B box-state law applies to bench_894's old-vs-new arms).
2. **bench_894's 4x key-arm regressions deserve a real look regardless of profile** — "new" arms at 0.25x on the key side is either a genuine regression landed unnoticed in debug-shaded timing, or an arm-measures-the-box artifact (the sequential-A/B audit class). Interleaved A/B + release re-run decides.
3. Neither suite blocks this filing session's landing; both were disclosed in `13c62d2d3`'s commit body.

## Provenance

- Found 2026-11-26 during the i1245-followup workspace test sweep (`zed-riir-refine-idle@amo`).
- Clean-HEAD reproductions: detached worktrees at `8619ecffa` (886: 2.5690 µs; 894: same 6 gates).
- Related doctrine: katgpt-rs AGENTS.md §"A ratio of two SEQUENTIALLY-timed arms measures the BOX", §cfg-gated/profile traps, §"A latency number without its BOX STATE is not a measurement".
