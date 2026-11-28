# Issue 1246: two debug-profile latency-bar suites fail on amo — bench_886 G2a axpy + bench_894 kvarn zip gates (pre-existing, proven at clean HEAD)

**Status:** Open — filed 2026-11-26 (zed-riir-refine-idle@amo) during the i1245-followup lint-drain verification.
**Lease closed:** zed-9b1c8276@amo, 1795660538, was expires 2026-11-28T02:35:00+00:00 — expired unworked (no commits, no WIP in the tree at the 02:5xZ sweep read); folded 2026-11-28 by the re-claim below.
**Claimed:** zed-25331aaa@amo, 1795913500, expires 2026-11-30T02:45:00+00:00
(scope: the remaining fix-shape decision + the gate membership rows — per suite pick required-features on the `[[test]]` row vs a debug-ignore posture vs release-only gate membership, the green-zero trap ruled out by construction, and land the fix + any moved `scripts/test_gate.sh` floor in one commit. Execution sequenced AFTER this lane's i697 C1 fold lands tonight; the release-posture adjudication `810552b2e` is the deciding input; the filing lane's disclosure stands as provenance.)
**Class:** debug-profile latency bars that their own `cargo test` (debug) posture cannot hold on this box — the "profile is part of the claim" trap one layer down: the bars were calibrated (presumably release or on another box) and the suites run and FAIL at the default debug profile on amo (M5 Ultra, AC, quiet at run time).

## The two suites

- **`crates/katgpt-core/tests/bench_886_differential_anchor_goat` G2a** — `axpy @ d=512` bar 1.0 µs; measured 2.38–2.61 µs/call at debug on amo, reproducibly, ALONE (not load noise). The in-file comment says a single 512-wide axpy is "below timer resolution in release" — the batch-of-64 harness implies a RELEASE-calibrated bar, but the suite ships in the plain `cargo test` (debug) sweep and fails there. Reproduced at clean HEAD `8619ecffa` in a detached worktree (2.5690 µs) — NOT caused by the lint drain (`13c62d2d3`).
- **`crates/katgpt-kv/tests/bench_894_kvarn_dequant_zip_goat`** (feature `kvarn`) — 6 G2 timing gates (old-vs-new sequential debug-timed arms, 4096-position passes): value-4-bit "≥10% faster" gate reads 1.00x, and the five no-regression gates read 0.92x/0.25x/0.38x. The 4x-class key-arm ratios are far outside any load-noise band. Reproduced at clean HEAD `8619ecffa` (same 6 gates, same magnitudes) — pre-existing.

## What this issue asks

1. **Adjudicate the intended profile per suite** (the AGENTS.md profile-matrix law): if the bars are release-calibrated, gate the runs (`#[cfg_attr(debug_assertions, ignore)]` is the riir-train precedent) or re-pin as debug bars measured on a quiet box with box-state provenance; if debug-calibrated, re-pin from a fresh quiet-box measurement (the sequential-A/B box-state law applies to bench_894's old-vs-new arms).
2. **bench_894's 4x key-arm regressions deserve a real look regardless of profile** — "new" arms at 0.25x on the key side is either a genuine regression landed unnoticed in debug-shaded timing, or an arm-measures-the-box artifact (the sequential-A/B audit class). Interleaved A/B + release re-run decides.
3. Neither suite blocks this filing session's landing; both were disclosed in `13c62d2d3`'s commit body.

## ADJUDICATED 2026-11-26 (same day, release posture on amo — asks 1+2 answered empirically)

`cargo test --release` on amo (M5 Ultra, AC, quiet):

- **bench_886: ALL GATES PASS** — G2a axpy **0.0228 µs/call** (bar 1.0 µs, ~44× under), G2b 1.0038 (bar ≤ 1.02), G4 zero allocs. The bar is RELEASE-calibrated; the plain-debug `cargo test` sweep simply cannot hold it (debug runs ~100× slower on this micro-kernel class). Fix shape: gate the debug run of the suite — but note the AGENTS.md green-zero trap (a cfg-gated whole-file harness compiles to an empty binary that reports `ok. 0 passed`); the owning lane should pick between `required-features` on the `[[test]]` row, a debug-`ignore` with the count treated accordingly, or leaving the suite release-only in the gates.
- **bench_894: ALL GATES PASS** — G3 bitwise 12/12 (0 differing elements across all bits/hadamard/kv_dim combos), G4 0 allocs, and ALL six G2 gates green at release: value 4-bit **2.28× faster** (0.4377, bar ≤ 0.90 — the "no faster" debug reading was pure debug-shading), value 2-bit 1.53×, value 8-bit 1.27×, key 4-bit 1.35×, key 8-bit 1.42×, key 2-bit +1.3% (within the 1.05 no-regression bar), A/A control 0.9992. **The debug "4× regressions" were the sequential-debug-A/B-measures-the-box artifact in full** — the shipped dequant zip is a real win, no regression exists at the calibrated posture.

Remaining work: only the fix-shape decision in the first bullet (a scoped lane that owns test_gate.sh membership/floors); the measurement questions are closed.

## Provenance

- Found 2026-11-26 during the i1245-followup workspace test sweep (`zed-riir-refine-idle@amo`).
- Clean-HEAD reproductions: detached worktrees at `8619ecffa` (886: 2.5690 µs; 894: same 6 gates).
- Related doctrine: katgpt-rs AGENTS.md §"A ratio of two SEQUENTIALLY-timed arms measures the BOX", §cfg-gated/profile traps, §"A latency number without its BOX STATE is not a measurement".
