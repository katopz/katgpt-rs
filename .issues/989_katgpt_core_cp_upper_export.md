# Issue 989 — export the one-sided Clopper–Pearson UPPER bound from katgpt-core (the reflex Issue 112 T3a consumer ask)

**Status:** OPEN — filed 2026-11-08 from riir-reflex Issue 112 T3a (the substrate-first consumed-as-issue row; reflex currently carries a TEMPORARY local bisection marked `// TEMPORARY: superseded by the katgpt-core export (T3a)` at `src/engine/threshold.rs::cp_upper_bound`).
**Claimed:** zed-m3-i989-cpexport (mac m3), epoch=1794158329, expires 2026-11-10T17:18:49Z


## What exists (the seeds, verified 2026-11-08)

- `cp_upper` — a PRIVATE fn inside the bench file `crates/katgpt-core/benches/bench_615_two_fidelity_bai_goat.rs` (the Plan 615 / Bench 615 one-sided Clopper–Pearson PAC precedent): exact-CP upper bound via bisection, with the E=0 closed form `cp_upper(0, n) = 1 − delta^(1/n)` as the canary.
- `certified_frontier.rs` — exports only the closed-form `beta_union_bound`; its own doc names the exact CP/Beta-quantile as the upgrade.
- `riir-reflex/src/engine/threshold.rs` (the consumer) — a temporary f64 port: regularized incomplete beta (Lentz continued fraction + Lanczos ln-gamma) under a bounded deterministic bisection; validated against an independent direct-binomial-CDF root check (`cp_upper_matches_the_independent_binomial_cdf_root`, 1e-9) and the E=0 closed form (1e-15).

## The ask

Export from `katgpt-core` a one-sided Clopper–Pearson UPPER bound API:

- `pub fn cp_upper(errors: usize, n: usize, delta: f64) -> f64` behind a feature alongside `certified_frontier` (same opt-in posture; naming per the house vocabulary — the exact quantile IS `beta_union_bound`'s named upgrade path, consider `certified_frontier`-adjacent placement or a sibling module, never a new dep).
- `errors >= n` → `1.0`; `errors == 0` → the E=0 closed form (canary, pinned); otherwise the Beta `(1−delta)`-quantile of `Beta(errors + 1, n − errors)` via deterministic bisection on the regularized incomplete beta (f64 sequential folds; fixed iteration bounds — determinism per the Issue 940 law: same inputs → bit-identical output).
- Seed implementation: lift the Bench-615 private `cp_upper` (the precedent), NOT a rewrite — one home, bench re-pointed or left citing (DRY; the goat-audit eye).
- Fixture tests: E=0 closed form, monotonicity in `errors`, never-undercuts-the-observed-rate, and the independent binomial-CDF root identity `P(X <= k | p_u) = delta` (the reflex test trio is the template — port all three).
- Consumers re-pointed in the same family: `riir-reflex` retires its TEMPORARY block and consumes the export (greppable marker: the `// TEMPORARY:` line in `threshold.rs` disappears); Bench 615's private fn cites the export.

## Why katgpt-core

The CP bound is a modelless decision primitive (exact order statistics, no deps) — katgpt-core is its substrate home per the boundary laws; reflex consumes upstream primitives, never forks them (the `step_noise`/`elo_k_eff` forward precedent). A second fork surviving in reflex's engine would be the DRY violation the T3a marker exists to prevent.

## Constraints

- Feature-gated alongside `certified_frontier`; clippy `-D warnings` at both postures; `CARGO_TARGET_DIR=/tmp/plan989`; rustfmt per-file; never `cargo fmt -p`.

Refs: riir-reflex `.issues/112` (the consumer lane + the pre-registered harness gates) · Bench 615 (the seed) · Bench 688 (`beta_union_bound`, the closed-form sibling) · ThinkJev §3.6 (the worked recipe the bound serves).
