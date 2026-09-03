# Issue 722 — `stability_score` is 0.0 for every non-empty input, and a 13/13 GOAT never noticed

**Status:** OPEN — **T1 (formula) + T2 (non-vacuity test) + T3 (the class
swept, population 1) DONE 2026-09-04. Only T4 remains and it is an OWNER
COST CALL** (which threshold Plan 102 T6's dropped assertion should carry —
its two candidates are 2.3× apart and the data is below). Found sideways,
while printing `p99_tail_support` next to a P99 in `bench_102` for the
Issue-713-adjacent percentile work.

## The defect

`crates/katgpt-core/src/speculative/types.rs`, `StabilitySnapshot::compute`,
feature `stability_metrics` (**default-on**):

```rust
/// Stability score: 1.0 - (P99 / P50), 1.0 = perfect
pub stability_score: f64,
...
let stability_score = if p50 > 0 {
    1.0 - (p99 as f64 / p50 as f64).min(1.0)
} else {
    1.0
};
```

The input is documented as **sorted ascending**, so `p99 >= p50` by
construction — the p99 rank is never below the p50 rank. Therefore
`p99/p50 >= 1.0`, therefore `.min(1.0) == 1.0`, therefore

> **`stability_score == 0.0` for every non-empty sorted input. Always.**

Not "usually", not "in these benches": the metric cannot take any other value.
The doc comment is self-contradictory in the same two lines — for
`1.0 - P99/P50` to reach its stated "1.0 = perfect" you would need
`P99/P50 == 0`, which is impossible for a non-negative sorted sample.

The single-sample case is the same defect at the boundary and reads worse:
`p50 == p99` gives `1.0 - 1.0 = 0.0`, i.e. one perfectly repeatable
observation is scored **maximally unstable**.

## Measured, not argued

`cargo test --release -p katgpt-rs --test bench_102_tilert_pipeline_goat --
--nocapture`, every printed score from real timing data:

```
  1-layer: P50=709ns  P99=1042ns (sup=6) CV=0.1221 stability=0.0000
  2-layer: P50=1416ns P99=1875ns (sup=6) CV=0.1032 stability=0.0000
  4-layer: P50=2709ns P99=3500ns (sup=6) CV=0.0918 stability=0.0000
│ Stability│     0.0000  (1.0 = perfect)          │
```

Three different workloads, CV varying 0.09–0.12, `stability` pinned at
0.0000 — and the same bench prints, four lines later:

```
    Now we can detect latency spikes, regressions, and instability.
║  D1 (Stability Metrics):  ✅ Production-ready observability          ║
```

## Why the GOAT gate did not catch it

`stability_metrics` is recorded in `Cargo.toml` as *"Plan 102 GOAT 13/13"*.
Grepped for every assertion on the field across katgpt-rs, riir-ai, riir-train
and riir-game-sdk: there is exactly **one**, and it is the arm the constant
does not cover —

```rust
assert_eq!(empty.stability_score, 1.0);   // the n == 0 early return
```

So 13 gates passed over a metric whose only asserted value comes from the
branch that never computes it. This is the workspace's *authored-but-unreachable*
lesson in a new place: a unit test proves a scalar **can** be produced, never
that it ever **varies**. Compare `.docs/10_audits/` on non-vacuity — a gate
that cannot fail proves nothing, and a metric that cannot vary is the same
statement about a measurement.

## The fix (T1)

The stated intent — "1.0 = perfect" — has one natural reading for a
tail-vs-median ratio: `p50 / p99`. Equal ranks (no tail) → 1.0; a tail that
grows without bound → 0. Clamped to `[0, 1]`, `p99 == 0` → 1.0 (a sample that
is all zeros is perfectly stable at this timer's resolution). The `n == 0`
early return keeps its 1.0, so the one existing assertion is untouched.

**Direction of the change, for anyone reading a historical number:** every
recorded `stability_score` of 0.0000 in a `.benchmarks/` doc or bench log is
the constant, not a measurement, and is not comparable with anything produced
after this commit. No gate can regress, because none asserted the value.

## The spec'd gate that would have caught it was never implemented

Plan 102 T6 is marked `[x]` and specifies, verbatim:

> - Assert: `stability_score > 0.7` (P99 < 3.3× P50) for all sizes
> - Assert: `cv < 0.5` for micro config

**Neither assert exists.** `grep -rn "stability_score\s*>" --include="*.rs"`
over the repo returns zero sites predating this issue, and the `cv` gate
shipped as `cv < 1.0` (`bench_102:661`), not 0.5. The stability gate could not
have shipped: at a constant 0.0 it would have failed every run, so the path of
least resistance was to leave it out — and the bench then reported green
forever over the one number the plan wanted gated. A `[x]` on a task whose
assertions were dropped is how a constant survives a "GOAT 13/13" record.

The spec's parenthetical also settles the intended DIRECTION, which is why T1
is a fix and not a redesign:

| reading | `score > 0.7` means | satisfiable? |
|---|---|---|
| `1.0 - p99/p50` (shipped) | P99 < 0.3 × P50 | **no** — p99 ≥ p50 always |
| `p50 / p99` (T1) | P99 < 1.43 × P50 | yes |

and "P99 < 3.3× P50" is score **0.303** under `p50/p99`. So the author had the
direction right — 1.0 = perfect is reachable only by `p50/p99` — and both the
formula and the threshold arithmetic wrong.

## T4 — the threshold is an owner call, and here is the data for it

Measured after the T1 fix, `cargo test --release -p katgpt-rs --test
bench_102_tilert_pipeline_goat bench_f`, 4 runs × 3 configs on the M3 (shared
box, so read the spread, not any single row):

| config | stability_score observations | CV |
|---|---|---|
| 1-layer | 0.708, 0.667, 0.667, 0.680 | 0.25–0.29 |
| 2-layer | 0.744, 0.780, 0.780, 0.730 | 0.096–0.103 |
| 4-layer | 0.765, 0.660, 0.771, 0.625 | 0.091–0.150 |

**The plan's `> 0.7` would fail 5 of these 12 observations.** Its
parenthetical's 0.303 passes all 12 with wide margin. Both are defensible and
they are 2.3× apart, so picking one here would be picking a number rather than
measuring one — and a wall-clock gate on a shared box is the load-flaky class
this workspace already documents. Left as an owner call with the spread
attached rather than defaulted. Note `n_steps = 500` gives tail support **6**,
under the audit's `MIN_SUPPORT` of 10, so whatever threshold is chosen should
come with a larger n or an explicit acknowledgement that one preemption moves
the input.

## Tasks

- [x] **T1 — the formula.** `p50/p99` clamped, with the derivation of why the
  old form is constant written at the site so it cannot be "simplified" back.
- [x] **T2 — a non-vacuity test.** Not "the score is 0.7 for this input", which
  the constant would also have satisfied at 0.0: assert the metric **orders**
  three distributions (tight > moderate > wide, strictly), that a single
  sample is 1.0 rather than 0.0, and that the old expression is constant —
  pinned as an explicit `assert_eq!(1.0 - ratio.min(1.0), 0.0)` so the
  regression is caught by name.
- [ ] **T4 — pick the gate threshold** and implement Plan 102 T6's dropped
  assertion, or record deliberately that it stays print-only. Data above;
  raise `n_steps` first if it is to be a gate (tail support 6 today).
- [x] **T3 — the class, not the instance — MEASURED, population is 1.**
  `stability_score` was not found by looking for it; a print happened to sit
  next to a number being changed for other reasons. So the class was swept
  rather than left to luck: `1.0 - (a / b).min(1.0)` across all 16 contract
  repos (the exact spelling — `.min` on the RATIO, which is what makes the
  collapse invisible) gives **41 sites: katgpt-rs 6, riir-ai 31, riir-train 4,
  zero in the other 13.** Read per site, not counted:
  - katgpt-rs's 6 are 3 code sites + 3 of this issue's own comments/test.
    `flow_pruner.rs:147` divides entropy by `max_entropy = ln(V)` and
    `caddtree_budget.rs:117/139` raise `(1 - top1) ∈ [0,1]` to a power — both
    ratios are **≤ 1 by definition**, so the `.min` is a defensive clamp that
    never binds. Correct.
  - riir-ai's 31 are almost entirely `distance / max_distance` and
    `count / cap` saturations (`leo_obs.rs` ×11, the `npc_clr` extractors,
    threat urgency). The ratio genuinely **can** exceed 1 there — that is what
    the clamp is for — and the result stays in `[0, 1)` otherwise.
  - riir-train's 4 (`game_smt_teacher` ×2, `channel_lora`, `rat_plus_replay`)
    are the same saturation shape against a stated bound.

  **No second instance.** What made `stability_score` unique is that its
  numerator is `>=` its denominator by an invariant the API's own contract
  guarantees (sorted input ⇒ the p99 rank is never below the p50 rank), so the
  `.min` binds on **every** input rather than on outliers. That is the
  discriminator worth remembering, and it is a property of the *invariant*,
  not of the spelling.

  Scope of this closure, stated so it is not over-read: it covers one spelling
  of one shape. The class as written above — "a formula whose range collapses
  under an invariant its own inputs guarantee" — is not decidable by grep, and
  a wider instrument would need the invariant, not the expression. Closing T3
  means "swept for a second `1.0 - (a/b).min(1.0)` and found none", which is
  what the population table says and no more.

## Discipline

No promotion claim is added by this fix: `stability_metrics` was already
default-on, and this makes an already-shipped field mean what it says. What
changes is that "Plan 102 GOAT 13/13" in `Cargo.toml` should not be read as
evidence about `stability_score` — 13 green gates and a constant metric are
compatible, which is the finding.
