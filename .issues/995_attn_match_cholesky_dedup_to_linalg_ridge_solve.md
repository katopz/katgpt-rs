# Issue 995 — dedup: attn-match's `beta_fitter` Cholesky vs the leaf's `linalg::ridge_solve` family

**Status:** OPEN — filed 2026-11-12 (amo, from issue 988 T1's landing; the dedup issue 988's own fallback branch named — "keep its local copy and file a dedup issue — never a second solver").

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
