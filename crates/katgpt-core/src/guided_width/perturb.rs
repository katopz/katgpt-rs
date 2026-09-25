//! T1 structured perturbation — the pluggable ε arm of the rollout driver.
//!
//! Arm (a) [`Transversal`] is for DENSE low-d latents (the `[f32; 8]` belief):
//! `ε = σ_t · P_⊥ v` with `P_⊥ = I − ûûᵀ`, `û` the unit deterministic update
//! of this step. Exploration is orthogonal to the proposal's own progress, so
//! noise never fights (or fakes) the refinement it rides on; when the branch
//! is stuck (`‖Δu‖ ≈ 0`) there is no `û` and the draw is isotropic. No
//! cell-complex / Hodge claim is made at d = 8 (curse-of-dimensionality rule:
//! the DEC boundary story is a d ≤ 3 story) — arm (b) lives in `hodge_arm`.
//!
//! The ε SOURCE is the shipped BLAKE3 one
//! ([`crate::diversity::temp::blake3_noise_fill`]): zero-mean uniform in
//! `[-1, 1]^d`, one hash per 8 coordinates, bit-reproducible.

use crate::diversity::temp::blake3_noise_fill;

/// A perturbation arm the rollout driver can call once per noisy step.
///
/// `perturb` adds the ZERO-MEAN exploration part of `ε` to `h` in place; the
/// μ≠0 guidance term is added by the driver itself (and only when
/// [`Self::admits_guidance`] says the arm's invariant survives it).
pub trait Perturbation {
    /// `h += σ · P(v)` where `v` is drawn from `seed`. `delta` is this step's
    /// deterministic update `u_t − h_{t−1}`; `noise` is a caller-owned
    /// `h.len()` buffer. Must be deterministic in `(seed, sigma, delta)` and
    /// allocation-free. `sigma == 0` must leave `h` bit-identical.
    fn perturb(&mut self, h: &mut [f32], delta: &[f32], noise: &mut [f32], seed: u64, sigma: f32);

    /// Whether the driver may add the μ≠0 table term on top. An arm whose
    /// invariant the table's directions would break (the mass-conserving
    /// arm: a table direction is not divergence-free) returns `false`.
    fn admits_guidance(&self) -> bool {
        true
    }
}

/// Arm (a): transversal (`project = true`) or isotropic (`project = false`)
/// zero-mean BLAKE3 noise for dense latents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transversal {
    /// Project the draw off the deterministic update direction.
    pub project: bool,
}

impl Transversal {
    /// Transversal arm (the T1 (a) default).
    pub const TRANSVERSAL: Self = Self { project: true };
    /// Isotropic arm (the ablation / GRAM zero-mean analogue).
    pub const ISOTROPIC: Self = Self { project: false };
}

impl Default for Transversal {
    fn default() -> Self {
        Self::TRANSVERSAL
    }
}

impl Perturbation for Transversal {
    #[inline]
    fn perturb(&mut self, h: &mut [f32], delta: &[f32], noise: &mut [f32], seed: u64, sigma: f32) {
        let dir = if self.project { Some(delta) } else { None };
        transversal_perturbation_into(h, dir, noise, seed, sigma);
    }
}

/// Arm (a) as a free function: `h += σ · P_⊥ v`, `v ~ U[-1,1]^d` from `seed`.
///
/// `update_dir = Some(Δu)` projects `v` off `û = Δu/‖Δu‖`; `None`, a zero or
/// non-finite `‖Δu‖`, leaves `v` isotropic. `sigma` that is `0`, negative or
/// non-finite is an exact no-op (NaN never becomes noise). Scalar sequential
/// loops only — the result is bit-identical on every platform.
///
/// Post-condition (asserted in tests): when projected, `ε · û ≈ 0` to f32
/// rounding.
pub fn transversal_perturbation_into(
    h: &mut [f32],
    update_dir: Option<&[f32]>,
    noise: &mut [f32],
    seed: u64,
    sigma: f32,
) {
    if !(sigma.is_finite() && sigma > 0.0) || h.is_empty() {
        return;
    }
    let d = h.len();
    let v = &mut noise[..d];
    blake3_noise_fill(seed, 1.0, v);
    if let Some(u) = update_dir {
        let u = &u[..d];
        let mut nsq = 0.0f32;
        for &x in u {
            nsq += x * x;
        }
        if nsq.is_finite() && nsq > 1e-24 {
            let mut vu = 0.0f32;
            for i in 0..d {
                vu += v[i] * u[i];
            }
            // v − û(û·v) = v − u (u·v)/‖u‖²
            let c = vu / nsq;
            for i in 0..d {
                v[i] -= c * u[i];
            }
        }
    }
    for i in 0..d {
        h[i] += sigma * v[i];
    }
}

/// Add the μ≠0 guidance term `scale · s · d` to `h` (T5). `scale` that is
/// `0`/non-finite is an exact no-op.
#[inline]
pub(crate) fn add_guidance(h: &mut [f32], dir: &[f32], sign: i8, scale: f32) {
    if !(scale.is_finite() && scale != 0.0) {
        return;
    }
    let s = scale * f32::from(sign);
    for (x, &di) in h.iter_mut().zip(dir) {
        *x += s * di;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transversal_noise_is_orthogonal_to_the_update() {
        let delta = [0.3f32, -0.1, 0.7, 0.0, 0.2, -0.4, 0.05, 0.9];
        for seed in 0..64u64 {
            let mut h = [0.0f32; 8];
            let mut buf = [0.0f32; 8];
            transversal_perturbation_into(&mut h, Some(&delta), &mut buf, seed, 0.5);
            let dot: f32 = h.iter().zip(&delta).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-5, "seed {seed}: ε·û = {dot}");
            assert!(h.iter().any(|&x| x != 0.0));
        }
    }

    #[test]
    fn zero_or_nan_sigma_is_an_exact_noop() {
        let delta = [1.0f32; 8];
        for sigma in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut h = [0.25f32; 8];
            let mut buf = [0.0f32; 8];
            transversal_perturbation_into(&mut h, Some(&delta), &mut buf, 7, sigma);
            assert_eq!(h, [0.25f32; 8]);
        }
    }

    #[test]
    fn stuck_update_falls_back_to_isotropic_bit_identically() {
        let mut a = [0.1f32; 8];
        let mut b = [0.1f32; 8];
        let mut buf = [0.0f32; 8];
        transversal_perturbation_into(&mut a, Some(&[0.0; 8]), &mut buf, 3, 0.2);
        transversal_perturbation_into(&mut b, None, &mut buf, 3, 0.2);
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }

    #[test]
    fn deterministic_in_seed() {
        let delta = [0.5f32, 0.1, -0.2, 0.3, 0.0, 0.0, 0.4, -0.6];
        let mut a = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        let mut buf = [0.0f32; 8];
        transversal_perturbation_into(&mut a, Some(&delta), &mut buf, 99, 0.3);
        transversal_perturbation_into(&mut b, Some(&delta), &mut buf, 99, 0.3);
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        let mut c = [0.0f32; 8];
        transversal_perturbation_into(&mut c, Some(&delta), &mut buf, 100, 0.3);
        assert_ne!(a, c);
    }
}
