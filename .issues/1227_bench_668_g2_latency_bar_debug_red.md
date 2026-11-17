**Status:** OPEN — pre-existing red found 2026-11-17 by the i1225 G4-A lane's full-feature test sweep (zed-f3873962@amo, katgpt-rs `d77d27f31`); proven PRE-EXISTING at clean HEAD `dd970a948` in a detached worktree (issue-825 discipline: cross-run before recording).
**Claimed:** zed-4b76d49c@amo, 1794880283, expires 2026-11-19T12:00:00Z
Claim scope: the release-gate fix (option (a)) landed `6ad125427`; release-posture re-verify queued on quiet bmo (the i1225 G2-A window).

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

- [ ] Read bench_668's intent: if G2's bar is a RELEASE-posture claim, gate
      the latency arm to release (`#[cfg(not(debug_assertions))]` on the
      timing assertions with a debug arm that asserts ORDER (e.g. r-scaling
      direction) instead of absolute ns — the pairing-scaling half may
      already be profile-tolerant; check its own margin first) or mark the
      test `#[ignore]`-with-runbook like its GOAT siblings.
- [ ] If the bar is meant to hold in DEBUG too, re-measure and re-pin it
      from a QUIET-box DEBUG run (bench preflight provenance quoted beside
      the number — 1599 ns on a loaded box is evidence of nothing).
- [ ] Sweep the sibling GOAT bench files for the same shape (a debug-runnable
      absolute-latency assertion) — file each hit separately or fold here if
      trivial; the class, not the instance, is the finding.

## Defer-marking

Found inside the i1225 G4-A landing's validation sweep; not fixed there
(scope law — the landing is the ternary-circuit lane). The full-feature
ternary_circuit sweep stays RED on loaded boxes until the G2 arm is
release-gated or re-pinned; the i1225 gates themselves
(transcode/forward/alloc) are all GREEN and unaffected.
