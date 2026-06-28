# Issue 009: `ahla_step` math divergence (riir-engine vs katgpt-core)

## Status

OPEN — discovered during Plan 008 Phase 2.1 (2026-06-28).

## Summary

`ahla_step` in riir-engine and katgpt-core compute **different mathematical
results** for the same inputs. This blocks the HLA substrate dedup
(Plan 008 Phase 2.1) until reconciled.

## The divergence

Both crates compute the AHLA readout intermediate `r` from query `q` and the
PKV state matrix. The math differs:

| Crate | Code | Computes |
|---|---|---|
| **riir-engine** `hla/kernel.rs:376` | `simd::simd_matvec(&mut tmp_r, pkv, q, hd, hd)` | `tmp_r = PKV · q` (standard row-major matrix-vector product) |
| **katgpt-core** `hla/kernel.rs:460-469` | manual loop: `for i { for j { tmp_r[j] += q[i] * PKV[i*hd+j] } }` | `tmp_r = qᵀ · PKV` (= `PKVᵀ · q`) |

Since `PKV = Σ k·vᵀ` is non-symmetric in general, `PKV · q ≠ qᵀ · PKV`.

The same divergence applies to the output computation `o = ... · E` where
`E = Σ k·rᵀ`:
- riir-engine: `simd_matvec(out, &q_head.e, q, hd, hd)` → `o = E · q`
- katgpt-core: manual loop → `o = qᵀ · E`

## Docstring mismatch in riir-engine

riir-engine's `ahla_step` docstring (line 374) says:
> `r = q_tᵀ · PKV_t`

But the code computes `r = PKV · q`. **The docstring and code disagree.**
katgpt-core's manual loop matches its docstring intent (`qᵀ · PKV`).

## Why tests don't catch it

The existing AHLA tests (`ahla_step_basic`, `ahla_step_nonzero_output`) use
sparse inputs like `q = [1, 0]` where `PKV · q` and `qᵀ · PKV` produce
identical results (only the first row/column contributes). A test with a
dense `q` (e.g., `q = [0.5, 0.7]`) and a non-symmetric PKV would expose the
divergence.

## Impact

- **Blocks Plan 008 Phase 2.1**: riir-engine's `ahla_step` cannot be replaced
  with `katgpt_core::hla::ahla_step` without changing behavior.
- **Possibly a long-standing bug**: if the paper derivation says `qᵀ · PKV`
  (as both docstrings claim), riir-engine has been computing the wrong thing.
  But models may have been trained/tuned against riir-engine's behavior.
- **Modelless-first mandate**: per the project AGENTS.md, behavior changes
  require either (a) freeze/thaw reconciliation or (b) riir-train validation.
  A silent "fix" that changes outputs is forbidden.

## Decision needed

1. **Is riir-engine's `PKV · q` an intentional variant** (e.g., a different
   but valid AHLA formulation that emerged from training)?
2. **Or is it a bug** (docstring intent is `qᵀ · PKV`, code drifted)?

## Resolution paths (per modelless-first mandate)

- **If bug (most likely):** The fix is to change riir-engine's
  `simd_matvec(tmp_r, pkv, q, ...)` to a manual `qᵀ · PKV` loop (or use a
  transpose-aware matvec). But this changes outputs for any model trained
  against the current behavior → requires riir-train validation OR a
  freeze/thaw reconciliation.
- **If intentional variant:** Document as such, rename the function
  (`ahla_step_transposed`?), and update the docstring to match the code.
- **Either way:** Add a test with dense `q` + non-symmetric PKV to lock in
  the intended behavior and prevent future silent drift.

## References

- Plan 008 Phase 2.1 (katgpt-rs/.plans/008_katgpt_core_substrate_extraction.md)
- HLA paper: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."
- katgpt-rs/.research/28_Higher_order_Linear_Attention.md

## Priority

Medium — blocks full HLA substrate dedup but doesn't affect correctness of
either repo in isolation (both are internally consistent). The partial dedup
(HlaVariant re-export + kernel optimization ports) already landed.
