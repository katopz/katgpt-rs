**Lease closed:** zed-4b76d49c@amo, 1794880283, expires 2026-11-19T12:00:00Z, closed 2026-11-17T02:50Z — all three tasks adjudicated; the issue CLOSES (the sweep is GREEN at both postures).
Claim scope was: the release-gate fix (option (a)) + the release-posture re-verify.
**Status:** Closed — fixed `6ad125427`, verified at both postures 2026-11-17. The profile split is the fix (the issue's option (a)): fit-cost bars (500/2000 ns) + node-sampling bar (200 ns) are RELEASE-only; DEBUG asserts ORDER (cheap fit < precise fit — load-independent direction); the pair-scaling arm stays in BOTH profiles (a relative ratio — its 2.5× tolerance was sized for a loaded box at the 718(a) pricing run, and profile inflation cancels in a ratio). Fit cost extracted to `fit_per_path_ns` (one home, both configs). VERIFIED: debug on busy amo 7/7 GREEN (the sweep that reded at `d77d27f31` now passes unoptimized on a loaded box); release on quiet bmo 7/7 GREEN — `PROVENANCE: power=AC Power load=1.20 swap=66006.38M canary=skipped powermode=0(auto)` (preflight PASSED, swap warning disclosed) — fit r=4 **108.9 ns/path** (bar 500, 4.6× headroom), fit r=15 **399.0 ns/path** (bar 2000, 5× headroom), node sampling 0.3 ns (bar 200), pair scaling 103–104 ns/pair FLAT across 4→32 (ratio ≈1.0×; the debug numbers: 1599.8 ns r=4 class inflation → the release-only bar was correct as pre-registered, no re-pin needed so task 2's branch is not taken).

# Issue 1227: bench_668 G2 latency bar reds in debug profile on a busy box

`cargo test -p katgpt-core --features ternary_circuit` (the full-feature sweep any
ternary_circuit arming runs) executes `tests/bench_668_effective_degree_goat.rs`
`g2_per_path_latency_and_pair_scaling` — a wall-clock latency bar
(`≤ 500 ns/path`, measured 1599.8 ns/path) — in whatever profile the sweep
runs. In DEBUG on a busy amo (the i1186 quantize pair + cq3_gpu_g1 GPU lanes)
it fails:

```
r=4 K=3: 1599.756455 ns/path exceeds 500 ns
```

At clean HEAD `dd970a948` (detached worktree, my diff absent): same failure,
same magnitude — **pre-existing, not the i1225 lane's regression**. Release is
not exercised by this sweep (the test runs at the sweep's profile).

## The class (standing law)

riir-ai AGENTS.md: **"Run armed latency gates with `--release`"** — a
nanosecond-class wall-clock bar is unadjudicable in debug (unoptimized kernel
bodies inflate it 3–10×) and load-sensitive on a shared box (box state is
part of every latency claim). A latency bar runnable at the default test
profile is a flake that reds every full-feature sweep run on a loaded box —
exactly the shared-box green-zero/TRANSIENT misdiagnosis class the x86 matrix
rails against.

## Tasks

- [x] Read bench_668's intent: if G2's bar is a RELEASE-posture claim, gate
      the latency arm to release (`#[cfg(not(debug_assertions))]` on the
      timing assertions with a debug arm that asserts ORDER (e.g. r-scaling
      direction) instead of absolute ns — the pairing-scaling half may
      already be profile-tolerant; check its own margin first) or mark the
      test `#[ignore]`-with-runbook like its GOAT siblings. — **DONE via the
      `cfg!(debug_assertions)` split (option (a), the cfg! form — the test
      stays one test so the sweep still executes it everywhere; measured in
      debug, barred in release). Margin check on the pairing-scaling half:
      profile-tolerant CONFIRMED — 103–104 ns/pair flat in release on bmo,
      and its 2.5× tolerance was sized for a loaded release run (718(a));
      the arm stays dual-profile so the regression class keeps sweep
      coverage.**
- [x] If the bar is meant to hold in DEBUG too, re-measure and re-pin it
      from a QUIET-box DEBUG run (bench preflight provenance quoted beside
      the number — 1599 ns on a loaded box is evidence of nothing). — **NOT
      TAKEN: the bar was a release-posture claim (the GOAT landing's own
      posture); the debug 1599 ns reading is expected debug inflation
      (3–10×), not a moved bar. No re-pin.**
- [x] Sweep the sibling GOAT bench files for the same shape (a debug-runnable
      absolute-latency assertion) — file each hit separately or fold here if
      trivial; the class, not the instance, is the finding. — **DONE 2026-11-17,
      verdict: bench_668 was the ONLY instance.** Every `Instant::now` test
      target swept across katgpt-core + katgpt-types: `sync_bank_goat` and
      `bench_680_kinematic_rollout_goat` already release-lock via
      `#[cfg_attr(debug_assertions, ignore = "...")]` (the ugc_g1b/funcattn
      precedent); `bench_578_ternary_group_goat` bars a RELATIVE ratio (vs
      row-scale, ceiling 1.50× — profile-tolerant, inflation cancels); the
      remaining Instant users have zero absolute bars or are `#[ignore]`d
      runbook cells. No new files needed — the class is closed by the one
      fix.

## Defer-marking

Found inside the i1225 G4-A landing's validation sweep; not fixed there
(scope law — the landing is the ternary-circuit lane). The full-feature
ternary_circuit sweep stays RED on loaded boxes until the G2 arm is
release-gated or re-pinned; the i1225 gates themselves
(transcode/forward/alloc) are all GREEN and unaffected.
