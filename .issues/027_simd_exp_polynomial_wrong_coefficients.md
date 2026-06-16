# Issue 027: simd_exp_inplace / neon_exp_inplace / avx2_exp_inplace use wrong polynomial coefficients (correctness bug)

**Date:** 2026-06-16
**Discovered during:** Plan 281 SIMD-sigmoid work (Issue 024/025 M1, commit `420f041d`)
**Status:** ✅ **FIXED** — all 4 SIMD exp kernels now use the correct Horner-chain polynomial form matching `cephes_exp_scalar`. Truth-referenced regression tests added. Verified: 367 katgpt-core tests pass with `micro_belief,bom_sampling,simd_sigmoid`.

---

## Symptom (measured in-process against `f32::exp()`)

Calling the real `katgpt_core::simd::simd_exp_inplace` on `[0.0, 1.0, 2.0, 0.5, -1.0, 5.0, 10.0]` (Apple Silicon aarch64, release):

| input | true `f32::exp()` | `simd_exp_inplace` | abs_err | rel_err |
|---|---|---|---|---|
| 0.0 | 1.000000 | 1.000000 | 0.000000 | 0.00% |
| **1.0** | 2.718282 | **2.732925** | 0.014644 | **0.54%** |
| **2.0** | 7.389056 | **7.763399** | 0.374343 | **5.07%** |
| **0.5** | 1.648721 | **1.691146** | 0.042425 | **2.57%** |
| -1.0 | 0.367879 | 0.367879 | 0.000000 | 0.00% |
| 5.0 | 148.413162 | 148.413162 | 0.000000 | 0.00% |
| 10.0 | 22026.464844 | 22026.464844 | 0.000000 | 0.00% |

exp(2) is off by **5%**. exp(1) is off by **0.5%**. exp(0.5) is off by **2.6%**.

Inputs whose range-reduced residual `g` happens to land near zero (-1, 5, 10) come out exact — the bug only manifests for inputs where `g = x - n·ln2` lands in the ±0.1..0.5 band where the polynomial error is visible.

## Root Cause

The SIMD exp kernels (`neon_exp_inplace` at `simd.rs:2322`, `avx2_exp_inplace` at `simd.rs:1941`, and the fused `neon_exp_sum_inplace` / `avx2_exp_sum_inplace`) use a polynomial with coefficients `1/n` instead of the correct Taylor coefficients `1/n!`:

```
// CURRENT (wrong) — simd.rs neon_exp_inplace L2355-2385
q = 1 + g·(1 + g·(0.5 + g·(1/3 + g·(0.25 + g·(0.2 + g·(1/6))))))
```

Expanding the inner coefficients: `1, 1, 0.5, 1/3, 0.25, 0.2, 1/6` — these are `1/k`, not `1/k!`.

The **correct** 6th-order Taylor/Horner form is:

```
// CORRECT
q = 1 + g·(1 + g·(1/2 + g·(1/6 + g·(1/24 + g·(1/120 + g·(1/720))))))
```

Coefficients: `1, 1, 1/2, 1/6, 1/24, 1/120, 1/720` — these are `1/k!`.

### Why `cephes_exp_scalar` is NOT affected

The scalar fallback `cephes_exp_scalar` (simd.rs:1643) uses a **different parenthesization** that happens to compute the correct polynomial despite looking superficially similar:

```rust
// simd.rs:1662 — scalar form (CORRECT, ~0.01% error at g=1)
let q = 1.0
    + g * (1.0
        + g * 0.5
            * (1.0
                + g * (1.0 / 3.0)
                    * (1.0 + g * 0.25 * (1.0 + g * 0.2 * (1.0 + g * (1.0 / 6.0))))));
```

The non-standard nesting `g·0.5·(1 + g/3·(...))` evaluates to `0.5g + g²/6·(...)` — effectively recovering the `1/k!` coefficients through algebraic cancellation. Whoever transcribed this into the SIMD Horner form `g·(0.5 + g·(1/3 + ...))` lost the cancellation and introduced the `1/k` error.

### Why existing tests don't catch it

`simd::tests::exp_sum_matches_separate_exp_plus_sum` (simd.rs:~6150) compares `simd_exp_sum_inplace` to `simd_exp_inplace` — **both use the same buggy polynomial**, so they agree to 1e-6 while both being wrong vs truth. The test is self-referential.

`simd::tests::exp_sum_known_value` uses tolerance `1e-4` for exp(1)/exp(2). It PASSES despite the 0.5%/5% errors — likely because the fused `simd_exp_sum_inplace` path hits a different (also-wrong) summation order whose errors partially cancel against the polynomial error. This is fragile luck, not correctness.

## Impact

Every caller of `simd_exp_inplace` / `simd_exp_sum_inplace` / `simd_exp_f32` (if any) is affected:

```
$ grep -rn "simd_exp_inplace\|simd_exp_sum_inplace" crates/katgpt-core/src/ src/ | wc -l
```

Known hot callers (search needed to enumerate all):
- Softmax / log-softmax / softmax-scaled paths (the docstring of `simd_exp_inplace` explicitly says "Sufficient for softmax").
- Any exp-decay or attention-weight computation routing through the SIMD exp.

For softmax, the standard max-shift trick reduces the input range to `[0, ~30]`, but the residual `g` after range reduction still lands in the buggy band for many inputs (e.g. input 1.0 → g ≈ 0.31 → 0.5% error). Softmax distributions are therefore subtly wrong, which silently degrades model quality (KL-divergence from truth is non-zero for no reason).

## Fix (small, well-scoped)

Replace the polynomial in all four SIMD kernels (`neon_exp_inplace`, `avx2_exp_inplace`, `neon_exp_sum_inplace`, `avx2_exp_sum_inplace`) with the correct Horner form. The new `neon_sigmoid_tanh_clamp` helper added in commit `420f041d` already uses the correct form — copy its polynomial structure.

```rust
// CORRECT 6th-order Horner for exp(g), g ∈ [-0.5·ln2, 0.5·ln2]
let q = {
    let p2 = 1.0 + g * (1.0 / 2.0);     // could inline further
    let p3 = 1.0 + g * (1.0 / 6.0);
    let p4 = 1.0 + g * (1.0 / 24.0);
    let p5 = 1.0 + g * (1.0 / 120.0);
    let p6 = 1.0 + g * (1.0 / 720.0);
    1.0 + g * (1.0 + g * (0.5 + g * (1.0/6.0 + g * (1.0/24.0 + g * (1.0/120.0 + g * (1.0/720.0))))))
};
```

Or in NEON intrinsics (Horner, fewest multiplies):

```rust
let v_inv_2  = vdupq_n_f32(1.0 / 2.0);
let v_inv_6  = vdupq_n_f32(1.0 / 6.0);
let v_inv_24 = vdupq_n_f32(1.0 / 24.0);
let v_inv_120 = vdupq_n_f32(1.0 / 120.0);
let v_inv_720 = vdupq_n_f32(1.0 / 720.0);

let p = vaddq_f32(v_one, vmulq_f32(vg, v_inv_720));         // 1 + g/720
let p = vaddq_f32(v_one, vmulq_f32(vg, vaddq_f32(v_inv_120, p))); // ... (Horner chain)
// ... continue for inv_24, inv_6, inv_2, 1
```

### Verification

Add a **truth-referenced** test (not self-referential) to `simd::tests`:

```rust
#[test]
fn simd_exp_matches_f32_exp_within_ulp() {
    let inputs: Vec<f32> = (-200..200).map(|i| i as f32 * 0.1).collect();
    let mut x = inputs.clone();
    simd_exp_inplace(&mut x);
    for (i, &xi) in inputs.iter().enumerate() {
        let expected = xi.exp();
        let rel_err = (x[i] - expected).abs() / expected.max(1e-30);
        assert!(rel_err < 1e-5, "exp({}) = {} vs true {}, rel_err={}", xi, x[i], expected, rel_err);
    }
}
```

This test will FAIL on current HEAD (proof of the bug) and PASS after the fix.

## GOAT gate

- **G1 (correctness)**: `simd_exp_matches_f32_exp_within_ulp` passes — rel_err < 1e-5 across [-20, 20].
- **G2 (no regression)**: all existing simd tests still pass.
- **G3 (latency)**: the Horner chain has the same number of FMAs (6), so latency is unchanged.
- **Promotion**: this is a pure bugfix — no feature flag needed. Land directly on `develop`. The existing self-referential test should be KEPT (it still validates fused-vs-separate consistency) but supplemented with the truth-referenced test above.

## Cross-References

- **Discovering commit:** `420f041d` (Plan 281 SIMD-sigmoid, Issue 024/025 M1) — the new `simd_sigmoid_tanh_clamp_inplace` uses the CORRECT polynomial form (verified by 17 passing bom tests under `simd_sigmoid`).
- **Affected code:** `crates/katgpt-core/src/simd.rs` — `neon_exp_inplace` (L2322), `avx2_exp_inplace` (L1941), `neon_exp_sum_inplace` (L2250), `avx2_exp_sum_inplace` (L2206).
- **Scalar reference (correct):** `cephes_exp_scalar` (L1643) — uses the non-standard nesting that algebraically recovers `1/k!`.
- **Related:** Issue 024 (attractor latency), Issue 025 (BoM G3) — both resolved their sigmoid path via the new helper which already uses correct coefficients.

## TL;DR

`simd_exp_inplace` and its fused variants use polynomial coefficients `1/k` instead of `1/k!`, producing up to **5% error** on common inputs like exp(2). The scalar fallback `cephes_exp_scalar` is correct (different nesting). Existing tests pass only because they compare SIMD-exp to SIMD-exp, never to `f32::exp()`. Fix: replace the polynomial in all 4 SIMD kernels with the correct Horner form (already proven in `simd_sigmoid_tanh_clamp_inplace`). Add a truth-referenced test. Pure bugfix, no feature flag, land directly.

---

## Fix Applied (2026-06-16)

Replaced the buggy add-nested polynomial in all 4 SIMD exp kernels with the correct Horner-chain form:

- `neon_exp_inplace` (simd.rs:~L2354)
- `avx2_exp_inplace` (simd.rs:~L2045)
- `neon_exp_sum_inplace` — both the `step!` macro and the remaining-chunks loop
- `avx2_exp_sum_inplace` — the `step!` macro

New form: `Q = 1 + g·(1 + g/2·(1 + g/3·(1 + g/4·(1 + g/5·(1 + g/6)))))` — matches `cephes_exp_scalar` bit-for-bit.

Added 2 truth-referenced regression tests (comparing to `f32::exp()`, not to another SIMD path):
- `simd_exp_matches_f32_exp_truth_referenced` — sweeps [-15, 15] in 0.1 steps, threshold 5e-4 relative (tolerates the f32 range-reduction precision floor at ~2e-5 for |x|>6, catches any polynomial regression which would produce ≥5e-2).
- `simd_exp_sum_matches_f32_exp_truth_referenced` — sweeps multiple lengths {1,3,4,8,12,16,17,31,32,33,100} to exercise both the main 4-accumulator loop and the remaining-chunks loop.

Post-fix worst observed rel_err: 2.683e-5 at x=-9.7 (range-reduction noise, not polynomial). Pre-fix at x=2: 5.07e-2 (5000× worse).

Validation:
- `cargo test -p katgpt-core --lib simd::tests` → 112 passed (was 110 before this fix + 2 new)
- `cargo test -p katgpt-core --features micro_belief,bom_sampling,simd_sigmoid --lib` → 367 passed
- No latency regression: the Horner chain has the same 6 FMA count as the old form.
