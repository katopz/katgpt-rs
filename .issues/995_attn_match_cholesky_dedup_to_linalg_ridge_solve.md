# Issue 995 — dedup: attn-match's `beta_fitter` Cholesky vs the leaf's `linalg::ridge_solve` family

**Status:** OPEN — filed 2026-11-12 (amo, from issue 988 T1's landing; the dedup issue 988's own fallback branch named — "keep its local copy and file a dedup issue — never a second solver").

**Lease closed:** LANDED 2026-11-12 by zed-84acf899@amo (the dfa5bc1f lineage continuation) — value_fitter's 4-copy local family (dispatcher + unblocked + blocked + multi-RHS solve) DELETED, re-pointed onto `linalg::ridge_solve::{try_cholesky_strict_f32, chol_solve_f32}`; `try_cholesky_strict_f32` added to the family (strict `diag <= 0` failure twin of `try_cholesky_f32` — the relative-tolerance clamp was absorbing rank-deficient pivots, starving the callers' jitter escalation; measured on G5: clamped-success 0.095 rel err vs strict-then-jitter 0.010). beta_fitter KEEPS its local scalar solve as the reviewed exception (nested-chain sensitivity, measured 0.0102 vs 0.0708 — see its module doc); katgpt-core non-optional (option 1, features de-`dep:`-ed); G5 graded on the relative-error floor: **PASS 0.0102 vs floor 0.05** (all-local baseline 0.0071 — the numerics delta is disclosed and bounded; bit-pin not claimed). bench_271 GOAT 10/10, attn-match lib 120/0 `attn_match` + 150/0 all-features, clippy `-D warnings` ×3 postures, root workspace check clean.
(decision recorded at claim: OPTION 1 — katgpt-core goes non-optional in attn-match. Rationale: the base leaf adds no new dependency class (attn-match already carries rayon/serde/postcard/half; core adds fastrand/log/bytemuck/arrayvec — same class), the workspace ships core to crates.io as its central crate anyway, and option 2's dual-codepath-until-retirement is the exact smell this dedup exists to kill. Shape: beta_fitter's jitter loop STAYS caller-side (retry policy), the factorisation/solve re-points onto linalg::{try_cholesky_f32, chol_solve_f32}, the local cholesky_decompose deletes, value_fitter's same-family copy (unblocked+blocked) sweeps in the same landing, the G5 gate grades on the RELATIVE-ERROR floors (no bit-pin — the numerics delta scalar→SIMD-interiors is disclosed at the landing), the f978a20b summation-order lesson stays as a doc note in the consuming module. Implementation + compile gates deferred until amo's window frees (the 1165 attempt-5 measurement holds the box; the zero-contention law).)

**Repos:** [katgpt-rs]

## The duplication

`crates/katgpt-attn-match/src/beta_fitter.rs` carries a private
`cholesky_decompose` (sequential-scalar Cholesky with a geometric
diagonal-jitter retry loop, `f978a20b`'s numerical-cliff rationale) plus
its normal-equations warm-start path — while
`crates/katgpt-core/src/linalg/ridge_solve.rs` ships the solver FAMILY
(`try_cholesky_f32` / `cholesky_f32` / `chol_solve_f32` /
`ridge_solve_direct_f32/f64`, relative-tolerance pivot clamp, SIMD
interior dots). Two f32 Cholesky factorisations in one workspace; the
two differ in numerics discipline (attn-match: scalar-only +
exponential jitter; linalg: `simd_dot_f32` interiors + relative-tol
clamp) and neither is strictly wrong — but a numerics fix to one will
not reach the other (the canonical failure the leaf's own doc records:
`f978a20b`, 0.7% → 7.5% relative error from a summation-order change).

## Why the re-point is not mechanical (the 988 T1 finding)

`beta_fitter` does not currently depend on katgpt-core — the dep is
optional behind `maxsim`/`smooth_min_rerank`/`rotary_value_embedding`
(attn-match Cargo.toml). Consuming `katgpt_core::linalg` from
`beta_fitter` means either:

1. **Making katgpt-core non-optional** in attn-match (a dep-graph
   change with compile-time cost for the attn-match-only consumers), or
2. A new attn-match feature threading the dep (e.g. `shared_lstsq`)
   with the local copy kept for the dep-free postures — two live
   codepaths again, until the local one is retired.

Neither is a mechanical same-commit move, so 988 T1 took the fallback
branch: graft fitting consumes the leaf's `ridge_solve_direct_f32`
directly (one solver per purpose), attn-match keeps its local copy
TEMPORARILY, and THIS issue owns the retirement.

## The shape of the fix (when taken)

- Decide 1-vs-2 (the dep-graph cost against the divergence risk).
- Re-point `beta_fitter::fit_beta_nnls`'s warm-start solve +
  `cholesky_decompose` onto `linalg::ridge_solve` (or its `try_*` twins
  with the jitter loop kept as the caller's retry discipline — the
  jitter loop is CALLER-side policy, the factorisation is the
  duplicate).
- **Bit-exactness gate**: attn-match's G5 gate must hold byte-identical
  across the re-point OR the numerics delta is disclosed and
  re-pinned — the two factorisations differ today (scalar vs SIMD
  interiors), so a bitwise pin CANNOT survive the swap; the gate is the
  relative-error floors, not bit identity. Say which at the landing.
- Delete the local `cholesky_decompose`; leave the `f978a20b`
  summation-order lesson as a doc note in the consuming module.

## Consumer watch

- `value_fitter::cholesky_decompose` (attn-match) — the same-family
  second copy; sweep it in the same landing.
- Any new fit lane (988's graft consumers, future overlay families)
  consumes `linalg::ridge_solve` from birth — the graft lane already
  does.
