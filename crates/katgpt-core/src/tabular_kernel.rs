//! Shared tabular-kernel helpers — hoisted verbatim from `mop::solve`
//! (Plan 573) so `hmm_control` (Plan 590) consumes the same one-hot fast
//! path + SIMD dot without requiring the `mop_path_entropy` feature.
//!
//! DRY: one implementation, two gated modules (`mop_path_entropy`,
//! `hmm_homeostasis`). Hoisting is a move, not a rewrite — both callers
//! inherit the same bit-level behavior, and the `mop` golden-parity tests
//! still pin it empirically.
//!
//! Ungated by design: pure generic functions over caller-owned slices, no
//! feature semantics, zero allocation.

use crate::simd::simd_dot_f32;

/// Sentinel for [`row_onehot`]: the row is NOT one-hot (0 or ≥2
/// nonzeros) → dense-SIMD dot. `u32::MAX` can never be a valid column.
pub(crate) const DENSE_ROW: u32 = u32::MAX;

/// One-hot row detection (Issue 654): the single nonzero column `j` of a
/// kernel row, or [`DENSE_ROW`] when the row has ≠ 1 nonzeros.
///
/// `pj != 0.0` treats `-0.0` as zero (IEEE: `-0.0 == 0.0`) and NaN as a
/// nonzero (degenerate kernels are out of contract — both paths yield NaN).
/// Early-exits on the 2nd nonzero so dense rows cost O(2), not O(N) — the
/// Bench 638 dense fixtures must not pay a scan tax.
#[inline]
pub(crate) fn row_onehot(row: &[f32]) -> u32 {
    let mut nz = DENSE_ROW;
    let mut seen = 0u8;
    for (j, &pj) in row.iter().enumerate() {
        if pj != 0.0 {
            seen += 1;
            if seen > 1 {
                return DENSE_ROW;
            }
            nz = j as u32;
        }
    }
    nz
}

/// Row dot `Σ_j row[j]·vec[j]` — one-hot fast path (Issue 654) with
/// dense-SIMD fallback.
///
/// **Bit-identity argument** (why the fast path is NOT a behavior change):
/// when `onehot = j*` is the row's single nonzero column, the dense dot
/// reduces to that one term — every zero entry contributes `±0`, an exact
/// no-op against finite accumulators (`acc + ±0 = acc`, and every
/// accumulation starts at `+0`, so any all-zero partial stays `+0`), while
/// the surviving term is correctly rounded in both paths (an FMA into a
/// zero accumulator equals the plain product). The only bit divergence is
/// the sign of a ±0 dot, which the callers' additive composition absorbs
/// exactly (in `mop`: `h_bar + dot` with `h_bar` never `-0`; in
/// `hmm_control`: products with non-negative emissions). Arch-independent:
/// the proof does not depend on the SIMD lane layout. Rows with ≥2
/// nonzeros keep the dense path — f32 addition order is not associative,
/// and replicating per-arch lane structure would be fragile for marginal
/// gain (blended zone-KG rows are the minority).
///
/// Consumers: `mop::solve` (`Σ p·ln z`, the LSE bootstrap) and
/// `hmm_control::solve` (`Σ p·β`, the reachability expectation).
#[inline]
pub(crate) fn row_dot(row: &[f32], vec: &[f32], onehot: u32, n: usize) -> f32 {
    if onehot == DENSE_ROW {
        simd_dot_f32(row, vec, n)
    } else {
        let j = onehot as usize;
        row[j] * vec[j]
    }
}
