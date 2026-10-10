# Bench 932 — Plan 622 Phase 7 GOAT: the CPU legs

**Status:** COMPLETE (CPU legs) — 2026-10-11
**Feature:** `scale_invariant_attn` (opt-in; NOT promoted — the GPU legs are open)
**Plan:** `.plans/622_scale_invariant_attn.md` Phase 7
**Box state:** `PROVENANCE: power=AC Power load=2.37 swap=0.00M canary=skipped powermode=2(high)` — M3 Max, macOS 26.6.2. One concurrent CPU consumer (`plan404_rank_selection_ab`, sibling session, 1 core of 16); **GPU-exclusive** (no GPU compute consumers — verified `ps aux` before the run; the preflight canary was unbuilt and is DISCLOSED as skipped, not silently absent: the load/swap/power axes all pass and the G2 leg is CPU-only where the missing canary is irrelevant).

## What ran

`cargo test -p katgpt-core --features scale_invariant_attn --test scale_invariant_goat --release` — the NEW Phase-7 test file `crates/katgpt-core/tests/scale_invariant_goat.rs` (required-features row per the green-zero rule), 3 tests:

| leg | verdict | measured |
|---|---|---|
| **G1** decade-mass flatness at T = 10⁶ | **PASS** | window (10⁵, 10⁶]: f64 reference 37.96165 inside the Lean interval [37.96157, 37.96165]; f32-LUT mass sum within 1e-4 relative of the reference; LUT sum inside the Θ(1) band — ≤ ατ·ln Δ = 37.9631, ≥ the T-independent floor 9.227. The length-generalization claim holds at the extreme where SSMax's position-independent scalar has flattened the distribution. |
| **G2** LUT build < 1 ms @ 64k | **PASS** | **0.233 ms** @ 64k (release, best-of-7, `black_box` loud-zero defence: every timed build's boundary pin asserted exact + top pair finite). Dev profile measured 1.088 ms — printed, bar skipped loud (a debug-profile latency number is not a measurement of the shipped binary). |
| **G4** apply is allocation-free | **PASS** | 0 allocations across 12 apply calls (both variants × full/mid-row entry × logit_scale 1.0/8.0 × head_scale 1.0/1.7) with the per-thread counting allocator; loud-zero defence: the transformed rows asserted ≠ pristine, sink asserted untouched. |

Sibling suites at the same posture (no regression):
- `scale_invariant_spec_match` 6/6 (Phase 6 Lean mirrors unchanged).
- src unit tests `--lib scale_invariant` 25/25 at `scale_invariant_attn,tiled_attention,parallax_attn` — includes the G1 legs already pinned in Phase 1–3: LUT vs closed form ≤ 1e-6 relative, boundary pins exact, the Φ(−1/a_t) tilt law within C·a_t⁻³ (C = 1.0 fixed once, values derived at test time), the calibration pass-through, the sidecar corruption matrix.
- No-double-temperature (the ordering law): `si_and_ssmax_armed_is_a_loud_config_error` (parallax tests) — executes in the same feature posture.
- clippy `--features scale_invariant_attn --all-targets`: **0 findings**.

## What did NOT run here (the GPU legs — open, tracked in the plan)

- **G2** pp2048/tg128 ±3% and Metal/CUDA parity — needs the engine wiring (no riir-engine/riir-infer consumer forwards `scale_invariant_attn` yet; verified by grep).
- **G3** the Bonsai-27B zero-shot split (calibrated / uncalibrated / pp-RoPE expected-FAIL rows) — same wiring + the serving-loader knob.
- **G1** Bonsai real-logit decade diagnostics — rides the G3 harness (the Phase-5 `si_probes` report structs are the output shape).
- **League** pp16384/pp65536 cells + train-length provenance — rides the G2 engine run.
- **Promotion** — therefore NOT decided; `scale_invariant_attn` stays opt-in (correct per the GOAT law: no promotion until G1–G4 all pass).

## Notes

- The G1 window-sum bound transcription was caught by the gate itself on the first run (a T vs TΔ mix-up in the interval formula — `big_t·d` instead of `T = big_t/d`): the Lean bounds are tight enough at T = 10⁵ that a wrong-window reference violates them by ~4e-5 relative. The gate discriminates — that is what it is for.
- G4 reuses the house per-thread counting allocator (`tests/common/mod.rs`, Issue 714).
