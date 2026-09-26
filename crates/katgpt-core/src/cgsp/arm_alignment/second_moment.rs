//! `SecondMomentDrift` — Issue 899's redesign of the Plan 610 drift summary.
//!
//! ```text
//! s_j   = p_j / Σp                                simplex share (max-renorm safe)
//! d_j   = fast_ema(s_j) − slow_ema(s_j)           per-arm derivative (10:1 kernel)
//! v_j   ← (1−α)·v_j + α·(s_j(t) − s_j(t−1))²      per-arm step-noise variance
//! C_kj  = cos²(g_k, g_j)                          pool kernel, built once
//! z_k   = (C·d)_k / √((C∘C)·v)_k                  null-normalized alignment
//! r̃_k   = sigmoid(β_z · |z_k|)                    β_z = 1 (z is in null-sd units)
//! ```
//!
//! Why each piece exists (Bench 900 measured the defects it answers):
//!
//! - **Squared cosines** — `(C·d)_k = ĝ_kᵀ (dM) ĝ_k` for the second moment
//!   `M = Σ_j s_j g_j g_jᵀ`. Mass moving onto a `±e_i` pair raises both arms'
//!   read, where the first-moment pull `Σ s_j g_j` cancels it to zero.
//! - **Null normalization** — under i.i.d. arm noise, `Var((C·d)_k)` is
//!   proportional to `Σ_j C_kj² v_j`, so `z_k` has the same null variance for
//!   every arm whatever the pool geometry. The per-coordinate preconditioner
//!   tried to buy this in an arbitrary latent basis and could not.
//! - **Simplex share** — `renormalize_priorities` rescales to max = 1, and
//!   every max change would otherwise inject a common-mode drift `∝ p_j`.
//!
//! Cost: `O(n)` per observation, `O(n)` per pool-indexed candidate (two row
//! dots, evaluated lazily), `O(n·dim)` per off-pool candidate. State is
//! fixed-size per arm plus the two `n×n` kernels, which are allocated once at
//! construction. Zero allocations once built.

use super::{DEFAULT_SCALE_ALPHA, DriftSummary, SCALE_EPS};
use crate::cgsp::types::{Candidate, DEFAULT_POOL_SIZE, Direction, Priority, sigmoid};
use crate::simd::simd_dot_f32;
use crate::temporal_deriv::TemporalDerivativeKernel;

/// Default β for the z-score form: a 2σ excursion maps to `sigmoid(2) ≈ 0.88`
/// (the point the first-moment form reaches at `|cos| = 0.5`, β = 4), and
/// ranks survive up to `|z| = 40` before `fast_sigmoid` saturates to 1.0.
pub const DEFAULT_Z_BETA: f32 = 1.0;

/// Denominator guard: below it the arm has no measured noise yet and scores
/// the neutral `z = 0`.
const DEN_EPS: f32 = 1e-9;

/// Second-moment, null-normalized drift summary (Issue 899).
///
/// `A` bounds the ARM count (unlike [`FirstMomentDrift`](super::FirstMomentDrift),
/// whose `D` bounds the latent dimension). The pool is fixed at construction.
#[derive(Clone, Debug)]
pub struct SecondMomentDrift<const A: usize = DEFAULT_POOL_SIZE> {
    kernel: TemporalDerivativeKernel<A>,
    share: [f32; A],
    prev: [f32; A],
    drift: [f32; A],
    noise_var: [f32; A],
    /// `Σ_j v_j` (for the centered null's `m² Σ v` term).
    noise_sum: f32,
    /// `C`, `n×n` row-major: `C_kj = cos²(g_k, g_j)`.
    gram_sq: Vec<f32>,
    /// `C∘C` (entrywise square) for the null denominator.
    gram_sq2: Vec<f32>,
    n: usize,
    var_alpha: f32,
    beta: f32,
    primed: bool,
}

/// `cos²` between two directions (0 for a zero vector).
#[inline]
fn cos_sq(a: &Direction, b: &Direction) -> f32 {
    let len = a.dim().min(b.dim());
    let den =
        simd_dot_f32(&a.coords, &a.coords, a.dim()) * simd_dot_f32(&b.coords, &b.coords, b.dim());
    match den > SCALE_EPS {
        true => {
            let c = simd_dot_f32(&a.coords, &b.coords, len);
            c * c / den
        }
        false => 0.0,
    }
}

impl<const A: usize> SecondMomentDrift<A> {
    /// Build over a frozen pool: precomputes `C` and `C∘C` (`2·n²` floats,
    /// the only allocation this type ever makes).
    ///
    /// # Panics
    ///
    /// Panics if the pool has more than `A` arms.
    pub fn for_pool(pool: &[Direction]) -> Self {
        let n = pool.len();
        assert!(n <= A, "pool has {n} arms, SecondMomentDrift<A={A}>");
        let mut gram_sq = vec![0.0f32; n * n];
        let mut gram_sq2 = vec![0.0f32; n * n];
        for (k, gk) in pool.iter().enumerate() {
            for (j, gj) in pool.iter().enumerate() {
                let c2 = cos_sq(gk, gj);
                gram_sq[k * n + j] = c2;
                gram_sq2[k * n + j] = c2 * c2;
            }
        }
        Self {
            kernel: TemporalDerivativeKernel::default(),
            share: [0.0; A],
            prev: [0.0; A],
            drift: [0.0; A],
            noise_var: [0.0; A],
            noise_sum: 0.0,
            gram_sq,
            gram_sq2,
            n,
            var_alpha: DEFAULT_SCALE_ALPHA,
            beta: DEFAULT_Z_BETA,
            primed: false,
        }
    }

    /// Override β_z.
    #[inline]
    pub fn with_beta(mut self, beta: f32) -> Self {
        debug_assert!(
            beta.is_finite() && beta > 0.0,
            "beta must be finite and positive, got {beta}"
        );
        self.beta = beta;
        self
    }

    /// Override the kernel's fast/slow EMA coefficients.
    #[inline]
    pub fn with_alphas(mut self, alpha_fast: f32, alpha_slow: f32) -> Self {
        self.kernel = TemporalDerivativeKernel::new(alpha_fast, alpha_slow);
        self
    }

    /// Null-normalized alignment `z_k` of pool arm `k` from the latest
    /// observation (`O(n)`: two row dots). `0` for an out-of-range arm.
    #[inline]
    pub fn z(&self, k: usize) -> f32 {
        let n = self.n;
        match k < n {
            true => {
                let row = k * n..(k + 1) * n;
                let (c, c2) = (&self.gram_sq[row.clone()], &self.gram_sq2[row]);
                let (d, sh, v) = (&self.drift[..n], &self.share[..n], &self.noise_var[..n]);
                let num = simd_dot_f32(c, d, n);
                // Simplex-centered null (Issue 899 v2): with `m = s·C_k`,
                // `Σ (C_kj − m)² v_j = Σ C_kj² v_j − 2m Σ C_kj v_j + m² Σ v_j`.
                // Four SIMD dots measured faster than one fused scalar pass
                // (2.08× vs 2.47× the incumbent's `sample_candidates`).
                let m = simd_dot_f32(c, sh, n);
                let var = simd_dot_f32(c2, v, n) - 2.0 * m * simd_dot_f32(c, v, n)
                    + m * m * self.noise_sum;
                Self::z_from(num, var)
            }
            false => 0.0,
        }
    }

    /// Write every arm's `z` into `out[..n]` (`O(n²)`; telemetry / tests).
    pub fn z_scores_into(&self, out: &mut [f32]) {
        for (k, o) in out.iter_mut().enumerate().take(self.n) {
            *o = self.z(k);
        }
    }

    /// `C` (`n×n`, row-major; read-only).
    #[inline]
    pub fn kernel_matrix(&self) -> &[f32] {
        &self.gram_sq
    }

    #[inline]
    fn z_from(num: f32, var: f32) -> f32 {
        let den = var.max(0.0).sqrt();
        if den > DEN_EPS { num / den } else { 0.0 }
    }

    /// `z` for a direction outside the pool (perturbed / off-pool candidate).
    fn z_off_pool(&self, direction: &Direction, pool: &[Direction]) -> f32 {
        let (mut num, mut m, mut q2, mut q1) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (j, gj) in pool.iter().enumerate().take(self.n) {
            let c = cos_sq(direction, gj);
            num += c * self.drift[j];
            m += c * self.share[j];
            q2 += c * c * self.noise_var[j];
            q1 += c * self.noise_var[j];
        }
        Self::z_from(num, q2 - 2.0 * m * q1 + m * m * self.noise_sum)
    }
}

impl<const A: usize> DriftSummary for SecondMomentDrift<A> {
    /// Returns `‖d‖₂` of the per-arm share derivative.
    fn observe(&mut self, priorities: &[Priority], _pool: &[Direction]) -> f32 {
        let n = self.n;
        debug_assert_eq!(
            priorities.len(),
            n,
            "priority vector length must equal the pool size"
        );
        let clean = |p: f32| if p.is_finite() && p > 0.0 { p } else { 0.0 };
        let total: f32 = priorities.iter().take(n).map(|&p| clean(p)).sum();
        match total > 0.0 {
            true => {
                let inv = 1.0 / total;
                for (s, &p) in self.share.iter_mut().zip(priorities.iter().take(n)) {
                    *s = clean(p) * inv;
                }
            }
            false => self.share[..n].fill(1.0 / n.max(1) as f32),
        }
        self.share[n..].fill(0.0);

        // Warm-start both EMAs on the first observation. From the kernel's
        // zero init, every arm reads a common-mode upward drift that decays at
        // the SLOW rate (5% left after 100 steps at α = 0.03). `C·d` reads
        // that drift in proportion to each row's sum, so it favored dense
        // clusters under pure noise. Measured: noise AUC 0.949 before this.
        if !self.primed {
            self.kernel.fast = self.share;
            self.kernel.slow = self.share;
        }
        self.drift = self.kernel.observe(&self.share);
        match self.primed {
            true => {
                let (a, keep) = (self.var_alpha, 1.0 - self.var_alpha);
                for ((v, &s), &q) in self.noise_var[..n]
                    .iter_mut()
                    .zip(&self.share[..n])
                    .zip(&self.prev[..n])
                {
                    let step = s - q;
                    *v = keep * *v + a * step * step;
                }
            }
            false => self.primed = true,
        }
        self.prev[..n].copy_from_slice(&self.share[..n]);
        self.noise_sum = self.noise_var[..n].iter().sum();
        // `z` is evaluated lazily per scored arm (`O(n)` each), so an
        // observation costs `O(n)`, not the `O(n²)` of every row.
        simd_dot_f32(&self.drift, &self.drift, A).max(0.0).sqrt()
    }

    #[inline]
    fn score(&self, candidate: &Candidate, pool: &[Direction]) -> f32 {
        let z = match candidate.pool_index < self.n {
            true => self.z(candidate.pool_index),
            false => self.z_off_pool(&candidate.direction, pool),
        };
        sigmoid(self.beta * z.abs())
    }

    fn reset(&mut self) {
        self.kernel.reset();
        self.share = [0.0; A];
        self.prev = [0.0; A];
        self.drift = [0.0; A];
        self.noise_var = [0.0; A];
        self.noise_sum = 0.0;
        self.primed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::super::FirstMomentDrift;
    use super::*;

    fn axis(dim: usize, i: usize, sign: f32) -> Direction {
        let mut coords = vec![0.0f32; dim];
        coords[i] = sign;
        Direction { coords }
    }

    fn cand(pool: &[Direction], k: usize) -> Candidate {
        Candidate::new(pool[k].clone(), k)
    }

    fn zs<const A: usize>(sm: &SecondMomentDrift<A>) -> Vec<f32> {
        let mut out = vec![0.0f32; sm.n];
        sm.z_scores_into(&mut out);
        out
    }

    /// Mass moving onto a ±e1 pair: invisible to the first moment (the pair
    /// cancels in `Σ s_j g_j`), visible to the second.
    #[test]
    fn sees_drift_onto_a_zero_centroid_pair() {
        let pool = vec![
            axis(4, 0, 1.0),
            axis(4, 0, -1.0),
            axis(4, 1, 1.0),
            axis(4, 1, -1.0),
        ];
        let mut second: SecondMomentDrift<4> = SecondMomentDrift::for_pool(&pool);
        let mut first: FirstMomentDrift<4> = FirstMomentDrift::new();
        let mut w = [1.0f32; 4];
        for t in 0..60 {
            w[2] += 0.05;
            w[3] += 0.05;
            // Tiny alternating wobble so the step-noise variance is non-zero.
            let e = if t % 2 == 0 { 1e-3 } else { -1e-3 };
            let p = [w[0] + e, w[1] - e, w[2] + e, w[3] - e];
            second.observe(&p, &pool);
            first.observe(&p, &pool);
        }
        // Second moment: the gaining pair reads positive, the losing pair
        // negative, and each ±pair reads identically (squared cosines).
        let z = zs(&second);
        assert!(z[2] > 2.0 && z[3] > 2.0, "gaining pair not seen: {z:?}");
        assert!(z[0] < -2.0 && z[1] < -2.0, "losing pair not seen: {z:?}");
        assert!((z[2] - z[3]).abs() < 1e-4, "±pair must read equally: {z:?}");
        // First moment: the pair drift cancels in `Σ s_j g_j`, so only the
        // wobble moves `m` and every arm scores the same — no information.
        let s1: Vec<f32> = (0..4)
            .map(|k| first.score(&cand(&pool, k), &pool))
            .collect();
        for s in &s1 {
            assert!(
                (s - s1[0]).abs() < 1e-3,
                "first moment should not discriminate: {s1:?}"
            );
        }
    }

    /// Under i.i.d. arm noise the z-score's spread is geometry-free: an arm in
    /// an 8-arm cluster and an isolated arm have comparable null std.
    #[test]
    fn null_variance_is_geometry_free() {
        let mut pool = Vec::new();
        for i in 0..8 {
            let mut c = vec![0.0f32; 16];
            c[0] = 1.0;
            c[1 + i] = 0.1;
            pool.push(Direction { coords: c });
        }
        for i in 0..8 {
            pool.push(axis(16, 8 + i, 1.0));
        }
        let mut sm: SecondMomentDrift<16> = SecondMomentDrift::for_pool(&pool);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let (mut sq_c, mut sq_s, mut cnt) = (0.0f64, 0.0f64, 0usize);
        for t in 0..4000 {
            let p: Vec<f32> = (0..16).map(|_| 1.0 + 0.2 * next()).collect();
            sm.observe(&p, &pool);
            if t >= 500 {
                let z = zs(&sm);
                sq_c += (z[0] as f64).powi(2);
                sq_s += (z[12] as f64).powi(2);
                cnt += 1;
            }
        }
        let (sd_c, sd_s) = ((sq_c / cnt as f64).sqrt(), (sq_s / cnt as f64).sqrt());
        let ratio = sd_c / sd_s;
        assert!(
            (0.67..=1.5).contains(&ratio),
            "null std not geometry-free: cluster {sd_c:.3} vs isolated {sd_s:.3}"
        );
    }

    #[test]
    fn off_pool_path_matches_pool_path_on_a_pool_direction() {
        let pool: Vec<Direction> = (0..4).map(|i| axis(4, i, 1.0)).collect();
        let mut sm: SecondMomentDrift<4> = SecondMomentDrift::for_pool(&pool);
        for t in 0..40 {
            let e = if t % 2 == 0 { 0.01 } else { -0.01 };
            sm.observe(&[1.0 + 0.02 * t as f32, 1.0 + e, 1.0 - e, 1.0], &pool);
        }
        for k in 0..4 {
            let on = sm.score(&cand(&pool, k), &pool);
            let off = sm.score(&Candidate::new(pool[k].clone(), usize::MAX), &pool);
            assert!((on - off).abs() < 1e-5, "arm {k}: {on} vs {off}");
        }
    }

    #[test]
    fn max_renormalization_is_invisible() {
        // The same distribution at two scales must give identical z.
        let pool: Vec<Direction> = (0..4).map(|i| axis(4, i, 1.0)).collect();
        let mut a: SecondMomentDrift<4> = SecondMomentDrift::for_pool(&pool);
        let mut b: SecondMomentDrift<4> = SecondMomentDrift::for_pool(&pool);
        for t in 0..30 {
            let e = if t % 3 == 0 { 0.03 } else { -0.01 };
            let p = [1.0 + 0.05 * t as f32, 1.0 + e, 1.0, 1.0 - e];
            let scale = if t % 2 == 0 { 1.0 } else { 7.5 };
            a.observe(&p, &pool);
            b.observe(&p.map(|x| x * scale), &pool);
        }
        for (x, y) in zs(&a).iter().zip(zs(&b).iter()) {
            assert!((x - y).abs() < 1e-3 * x.abs().max(1.0), "{x} vs {y}");
        }
    }

    #[test]
    fn reset_and_neutral_start() {
        let pool: Vec<Direction> = (0..4).map(|i| axis(4, i, 1.0)).collect();
        let mut sm: SecondMomentDrift<4> = SecondMomentDrift::for_pool(&pool);
        // One observation primes the variance only: every arm is neutral.
        sm.observe(&[0.7, 0.1, 0.1, 0.1], &pool);
        for k in 0..4 {
            assert!((sm.score(&cand(&pool, k), &pool) - 0.5).abs() < 1e-6);
        }
        sm.observe(&[0.1, 0.7, 0.1, 0.1], &pool);
        sm.reset();
        assert!(zs(&sm).iter().all(|&z| z == 0.0));
        assert_eq!(sm.kernel_matrix().len(), 16);
    }
}
