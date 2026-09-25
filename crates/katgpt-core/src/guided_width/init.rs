//! T4 — deterministic diversity: Sobol/BLAKE3-seeded branch initialisation
//! and farthest-point returned-set selection.
//!
//! Both consume shipped substrate: the low-discrepancy draw is
//! [`crate::speculative::qmc::SobolQmc`] (dimensions beyond its
//! `SOBOL_MAX_DIM` fall back to the BLAKE3 ε source), and the returned set is
//! the TEMP farthest-point selector's allocation-free core
//! ([`crate::diversity::temp::select_diverse_subset_in_place`]).

use crate::diversity::temp::{blake3_noise_fill, select_diverse_subset_in_place};
use crate::speculative::qmc::{SOBOL_MAX_DIM, SobolQmc};

use super::types::{GuidedWidthScratch, MAX_BRANCHES};

/// splitmix64 finaliser — the per-(branch, step, generation) seed mixer.
#[inline]
pub(crate) fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Write `n` initial branch states into `out[..n·d]` (branch-major).
///
/// Branch 0 is `center` exactly (the deterministic branch). Branch `b ≥ 1`
/// is `center + spread · (2u_b − 1)` with `u_b` the b-th point of a
/// `min(d, SOBOL_MAX_DIM)`-dimensional scrambled Sobol sequence seeded by
/// `seed`; coordinates past `SOBOL_MAX_DIM` take the BLAKE3 source.
/// `spread == 0` (or non-finite) makes every branch `center`, bit-exactly.
///
/// `sobol` is a caller-owned source built once with
/// `SobolQmc::new_multi(_, min(d, SOBOL_MAX_DIM))` — it is [`SobolQmc::reseed`]ed
/// here, which is bit-identical to a fresh `new_multi(seed, ..)` without the
/// allocating primitive-polynomial search. `sobol_buf` must hold
/// `(n − 1) · min(d, SOBOL_MAX_DIM)` floats. Deterministic in
/// `(center, n, spread, seed)`; zero-allocation.
///
/// # Panics
///
/// If `sobol.dim() != min(d, SOBOL_MAX_DIM)` (and a draw is needed).
pub fn sobol_init_into(
    center: &[f32],
    n: usize,
    spread: f32,
    seed: u64,
    sobol: &mut SobolQmc,
    sobol_buf: &mut [f32],
    out: &mut [f32],
) {
    let d = center.len();
    for b in 0..n {
        out[b * d..(b + 1) * d].copy_from_slice(center);
    }
    if n <= 1 || d == 0 || !(spread.is_finite() && spread != 0.0) {
        return;
    }
    let dd = d.min(SOBOL_MAX_DIM);
    assert_eq!(sobol.dim(), dd, "sobol_init_into: Sobol source dim mismatch");
    sobol.reseed(seed);
    let pts = &mut sobol_buf[..(n - 1) * dd];
    sobol.draw_nd(n - 1, pts);
    for b in 1..n {
        let row = &mut out[b * d..(b + 1) * d];
        let p = &pts[(b - 1) * dd..b * dd];
        for i in 0..dd {
            row[i] += spread * (2.0 * p[i] - 1.0);
        }
        if d > dd {
            // BLAKE3 tail for dimensions Sobol does not carry: the draw is
            // already in [-1, 1], so scale by `spread` and add in place.
            let tail = &mut row[dd..];
            let mut block = [0.0f32; 8];
            for (c, chunk) in tail.chunks_mut(8).enumerate() {
                let s = mix64(seed ^ ((b as u64) << 32) ^ (c as u64));
                let blk = &mut block[..chunk.len()];
                blake3_noise_fill(s, spread, blk);
                for (x, &e) in chunk.iter_mut().zip(blk.iter()) {
                    *x += e;
                }
            }
        }
    }
}

/// Farthest-point returned set over the LAST decision's branches: write up
/// to `k` branch indices into `out`, greedy max-min (L∞) from the farthest
/// pair — a coverage set of distinct hypotheses to return alongside the
/// scored best. Returns the number written (`0` after the kill-switch path).
///
/// Zero-allocation (the TEMP selector's in-place core, workspaces owned by
/// `scratch`).
pub fn diverse_set_into(scratch: &mut GuidedWidthScratch, k: usize, out: &mut [usize]) -> usize {
    let n = scratch.last_n.min(MAX_BRANCHES);
    if n == 0 || k == 0 {
        return 0;
    }
    let k = k.min(n).min(out.len());
    let d = scratch.d;
    let GuidedWidthScratch {
        states,
        min_dist,
        is_selected,
        ..
    } = scratch;
    let mut refs: [&[f32]; MAX_BRANCHES] = [&[]; MAX_BRANCHES];
    for (b, r) in refs.iter_mut().enumerate().take(n) {
        *r = &states[b * d..(b + 1) * d];
    }
    select_diverse_subset_in_place(&refs[..n], k, out, min_dist, is_selected);
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch0_is_center_and_draws_are_bounded_and_reproducible() {
        let c = [0.1f32, -0.2, 0.3, 0.0, 0.5, -0.5, 0.25, 0.0];
        let mut buf = [0.0f32; 64];
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        let mut src = SobolQmc::new_multi(0, 8);
        sobol_init_into(&c, 8, 0.4, 11, &mut src, &mut buf, &mut a);
        sobol_init_into(&c, 8, 0.4, 11, &mut src, &mut buf, &mut b);
        assert_eq!(&a[..8], &c);
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        for br in 1..8 {
            for i in 0..8 {
                assert!((a[br * 8 + i] - c[i]).abs() <= 0.4 + 1e-6);
            }
        }
        assert_ne!(&a[8..16], &a[16..24]);
    }

    #[test]
    fn zero_spread_is_all_center() {
        let c = [0.7f32; 5];
        let mut buf = [0.0f32; 40];
        let mut a = [9.0f32; 20];
        let mut src = SobolQmc::new_multi(0, 5);
        sobol_init_into(&c, 4, 0.0, 3, &mut src, &mut buf, &mut a);
        assert!(a.iter().all(|&x| x == 0.7));
    }

    #[test]
    fn high_dimension_uses_the_blake3_tail() {
        let d = SOBOL_MAX_DIM + 5;
        let c = vec![0.0f32; d];
        let mut buf = vec![0.0f32; 3 * SOBOL_MAX_DIM];
        let mut a = vec![0.0f32; 4 * d];
        let mut src = SobolQmc::new_multi(0, SOBOL_MAX_DIM);
        sobol_init_into(&c, 4, 1.0, 5, &mut src, &mut buf, &mut a);
        let tail = &a[d + SOBOL_MAX_DIM..2 * d];
        assert!(tail.iter().any(|&x| x != 0.0));
        assert!(tail.iter().all(|&x| x.abs() <= 1.0));
    }
}
