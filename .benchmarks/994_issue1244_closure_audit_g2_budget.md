# Bench 994 — Issue 1244 G2 budget: `closure_audit` ridge-sweep floor bar (spectral sweep)

**Status:** G2 **PASS** — ridge sweep (both sides, full λ grid) median-of-3 **38.776 ms ≤ 50.0 ms** at T=1e4, D=64, release, on a preflight-PASSED quiet rig.
**Type:** budget floor bar on an offline instrument (the primitive claims NO GOAT — ungoat discipline; the bar prices the instrument's fitness at the consumer-census scale, never a perf claim).

## Provenance (box state)

- `PROVENANCE: power=AC Power load=3.06 swap=3966.69M canary=skipped powermode=0(auto) ane=ok` — amo shared lane, preflight **PASSED** before the run (the gate line is in the job log; the run refuses to be quoted without it).
- Commit `8adb663a9` (the job's ff-only checkout pinned it first line). Release lib tests in the same run: 15/15 closure_audit (`test result: ok. 15 passed`).
- Job: amo `2e226a91` (log `01a20eaa-2eb1-76d1-a812-a2ee2e226a91.log`).

## The measured numbers

| line | value |
|---|---|
| G2 ridge sweep (both sides, median of 3) | **38.776 ms** (runs 40.586 / 38.776 / 38.594) — bar ≤ 50.0 ms → **PASS**, 22 % headroom |
| kNN FULL pass (cap none) | 3731.1 ms — disclosed, no bar (the issue's own text prices ≈0.3–2 s; the honest measured full pass at T=1e4 D=64 is 3.7 s on this box — consumers wanting the full posture budget ~4 s, the capped posture ~1.5 s) |
| kNN CAPPED pass (ref cap 2048) | 1484.6 ms — disclosed, no bar |
| alloc census (full run) | 64 allocations — disclosed offline budget (the offline instrument is bounded-by-arms, never per-window; the ONLINE half is the `WindowRing` G4 hard-zero gate, separate) |

## The arc that produced this number (recorded because the bar is pre-registered, never re-pinned)

The bar FAILED twice before passing, and both failures were implementation waste, not the design's floor — the discipline ran investigate-before-verdict each time:

| shape | ridge sweep | verdict vs bar |
|---|---|---|
| as landed (`c80cf19e7`): gram+rhs rebuilt PER λ | 142.7 ms (m3, preflight-passed) | FAIL — 5λ × 2 sides rebuilt ~37M FMA of λ-independent work each |
| + gram/rhs hoist (`8adb663a9` interim) | 60.7 ms (m3, preflight-passed) | FAIL — remaining cost = ~164M FMA of per-λ held-out predictions that cancel algebraically |
| + spectral sweep (`8adb663a9` landed) | **38.776 ms** (amo, preflight-passed) | **PASS** |

The spectral form: `gram = V·diag(δ)·Vᵀ` via the crate's own UNGATED deterministic `linalg::symmetric_eig` (Householder + QL, fixed sweep order — zero new deps, substrate consumed not forked); `W_λ = V·diag(1/(δᵢ+λ))·S`; held-out SSE through the trace expansion `‖Y‖² − 2ΣdᵢUᵢᵢ + ΣdᵢdⱼCᵢⱼTᵢⱼ` (A = PᵀY, C = PᵀP, U = SᵀA, T = SSᵀ built ONCE per side) — per-λ cost O(d²), whole sweep ≈ 90M FMA. Secondary: the alloc census dropped 1976 → 64 (no per-λ matrix copies).

## Equivalence + determinism (the rewrite's proof obligations)

- **spectral == direct**: new cross-check test (`spectral_sweep_matches_direct_residual_form`) pins the trace expansion against an independent per-λ Cholesky + direct-residual evaluation within **1e-9 on every λ** (injection corpus, 5/5 grid points).
- **Determinism**: every fold sequential f64 in a pinned order (Issue-940); the trace expansion shifts LAST-ULP rounding relative to the direct form (disclosed in the module docs) — the G1 battery pins band asserts, and 15/15 hold on the spectral shape. Not sync-crossing (local instrument), so no cross-boundary order claim is weakened.
- G1 battery 15/15 (identity/noise/injection/degenerate plants + determinism + boundaries + ring round-trip); G4 alloc canary 4/4 (ring push hard-zero; offline budget disclosed).

## Gate ledger for issue 1244 (T0–T2 complete)

| gate | result |
|---|---|
| G1 fixtures (injection > bar / noise ≈ 0 / identity ≈ exact / degenerate plant NOT Closed) | PASS (in-module battery, 15/15 incl. the spectral-equality pin) |
| G2 budget (ridge ≤ 50 ms @ T=1e4 D=64 release, median of 3) | **PASS (38.776 ms, this bench)** |
| G4 (WindowRing zero per-tick alloc; offline budget disclosed) | PASS (`issue_1244_closure_audit_g4_alloc`, 4/4) |
| kNN budgets | disclosed only (full 3.7 s / capped 1.5 s at the G2 corpus) — no bar |
| GOAT | **no claim** (ungoat discipline) — report-only consumers first |

## What remains (issue 1244)

- T3 (optional, this lane, separate claim): light phase-clustering arm (Lloyd's + agglomerative, pinned seeds) + NMI-across-≥5-pinned-seeds disclosure.
- T4: hand consumers their first census (riir-ai `.issues/1259` sync surface; riir-refine `.issues/256` fix trajectories) — cross-repo, now unblocked by T1+T2.
