# 875 — rate_control GOAT (Issue 873 primitive B)

**Status:** GOAT G1/G2/G3/G4 PASS (2026-09-22) — lands OPT-IN (`rate_control = []`, katgpt-core). NOT PROMOTED: no consumer-measured gain exists (the stack-slot rule); the first consumer A/B is riir-train Plan 416 Phase 2 (vs cosine at fixed budget + regime-change arm). Report-first posture (R135/Bench 047): gates nothing until evidence volume exists.

Owner: Issue 873 (closed 2026-09-25, HISTORY.md § Issue 873) · Source: [Research 581](../.research/581_Mini_AGI_Governed_Pool_Modelless.md) (volotat/mini-AGI @ `96784b7`, `plasticity.py:63-358`, MIT). Companion landings the same day: A `pool_admission` ([Bench 873](873_pool_admission_goat.md), `eabd0cb8`) and C `dying` ([Bench 874](874_dying_goat.md), `650faeac`) by the parallel session — a same-day TWIN landing; origin's A/C commits are canonical and this session's duplicate A/C implementations were discarded per the Batch-169 precedent. This bench covers B only.

## Box state (the G2 rule)

- Windows 11 Pro, i7-13700K (16 cores), RTX 4090 idle, AC power.
- CPU ~76% at session start — ≥2 sibling agent sessions active throughout; best-of-3 timing is the load discipline for this state.
- Release profile, isolated `git worktree` at `C:/tmp/kt873` (detached @ `2331961c` + this change), own `target/`. (The shared `E:\git\katgpt-rs` worktree was contended by the parallel session's in-flight edits at measurement time.)

## G1 — correctness (in-module `#[cfg(test)]`, 13 tests)

| Arm | Pins |
|---|---|
| `plateau_factor_flat` | y=5.0 constant ×300: factor stays within [0.98, 1.02] (the σ-floor keeps e = slope-noise/σ bounded; y=0 is exactly flat) |
| `improving_factor_climbs` / `deteriorating_factor_falls_fast` | exact ramps move the factor the right way, hard |
| `asymmetry_down_is_steeper_than_up` | equal-magnitude opposite trends: down-arm displacement ≥ 2× up-arm (gain 0.025 vs 0.005), both inside the clamps |
| `regime_jump_confirms_steps_and_resets` | jump > max(4σ, floor) then still > 2σ next obs → Note::RegimeJumpConfirmed, factor ×2, fits reset (slow weight below the cold bar), no re-confirm on the new level |
| `regime_jump_discarded_on_single_spike` | one-off spike → Note::RegimeJumpDiscarded, no step, pending cleared |
| `no_window_edge_artifacts` | **the B3 window-edge arm** — stationary noise + ONE negative outlier (negative residuals never fire the jump path, isolating window geometry): (1) every single-step \|Δln f\| ≤ down-gain (no ×2-style discontinuity); (2) the recovery envelope is NON-INCREASING at every candidate window length L ∈ {50,100,128,200,256} — a windowed controller's largest recovery step lands exactly at spike+L (the outlier leaving the window); an EWLS controller has no such rebound |
| `bad_inputs_dropped_fail_closed` | NaN/±inf val, negative/NaN se: dropped, factor unmoved |
| `determinism_bit_identical_across_runs` | 2000 LCG observations incl. a planted regime jump: identical (acc, state) across runs |
| `recentering_is_fit_invariant` | 3× the recenter period of a noisy ramp: still tracks, factor finite and clamped |
| `save_restore_roundtrip` | snapshot → both branches evolve identically; restore(state) == original |
| `factor_respects_clamps` | ±steep ramps ×5000: exactly FACTOR_CEIL / FACTOR_FLOOR |
| `observe_is_allocation_free` | 1000 observes = 0 allocs (TrackingAllocator) |

Test-gate row: `katgpt-core:2076:rate_control` (2076 = 2063 default + 13).

## G2 — latency (bench_875, best-of-3, release)

| Path | Measured | Ceiling |
|---|---|---|
| `observe(val, se)` — dual EWLS update + effect-size/t-stat readouts + tanh/exp nudge + jump detect | **42.0 ns** | 1,000 |

## G3 — no-regression

Default-feature lib count measured **unchanged at 2063** at this landing
base (HEAD `2331961c`); the feature compiles to nothing at default
features. Rows added to `scripts/test_gate.sh` for all three Issue-873
features (2076 rate_control / 2079 pool_admission / 2097 dying — the
latter two close the green-zero gap on the parallel session's already-
landed features: their gated tests were invisible to every pinned suite
before these rows). Clippy `--all-targets` at the feature: zero warnings.

## G4 — alloc-free

`observe` = 0 allocs (in-module debug suite). `RateController` is
plain-`Copy` (save/restore = assignment); `EwlsFit` is 6 f32 sums + a u32
clock; `Note` is a bare enum returned by value.

## Pinned constants (B2 — the Issue-033 never-adaptive law)

From mini-AGI (Research 581): `LAMBDA_SLOW 0.97`, `LAMBDA_FAST 0.85`,
`GAIN_UP 0.005`, `WIDTH_UP 0.75`, `GAIN_DOWN 0.025`, `WIDTH_DOWN 6.0`.
House-pinned at first landing: `T_MID 0.0`, `EFFECT 10.0`,
`FACTOR_FLOOR 0.25`, `FACTOR_CEIL 4.0`, `JUMP_K_CONFIRM 4.0`,
`JUMP_K_STILL 2.0`, `JUMP_FLOOR 1.0`, `MIN_WEIGHT 5.0`,
`RECENTER_EVERY 256`, `SIGMA_FLOOR_ABS 1e-6`, `SIGMA_FLOOR_REL 1e-3`.
Editable as constants only, never by feedback.

## Notable implementation decisions

- **`MIN_WEIGHT` constraint (found by probe on landing)**: must stay
  strictly below the fast fit's steady-state ceiling `1/(1−0.85) = 6.67`.
  The first draft used 8.0 → the fast arm was permanently gated cold →
  `v = min(t_slow, EFFECT·e_slow, EFFECT·e_fast) = 0` forever → factor
  deadlocked at exactly 1.0 (the improving-trend G1 arm caught it).
  Shipped 5.0; the constraint is written on the constant.
- **σ floors bound `e = b/σ` against cancellation noise** on near-perfect
  series (an exact ramp has σ → rounding-level; noise/noise is
  unbounded). `max(1e-6, 1e-3·|ȳ|)`.
- **Rolling-origin recenter** (every 256 ticks) is an exact algebraic
  shift of the EWLS sums (sx' = sx − d·s0; sxx' = sxx − 2d·sx + d²·s0;
  sxy' = sxy − d·sy) — keeps |x| ≤ 256 so the quadratic sums stay
  f32-precise over unbounded runs; fit-invariant by algebra, tested.
- **Jump detection reads the PRE-observation prediction** and combines
  fit σ with the caller's se in quadrature; a flagged candidate is still
  incorporated into the fits (confirmation lives one observation ahead).
