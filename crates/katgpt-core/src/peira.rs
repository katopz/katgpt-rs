//! PEIRA: Predictive Encoders through Inter-View Regressor Alignment
//!
//! Implementation of the PEIRA auxiliary loss (arXiv:2605.17671) for
//! collapse-free representation alignment. The core computation maintains
//! EMA estimates of cross-view (Σ) and within-view (N) covariance matrices,
//! then computes a closed-form predictor and auxiliary loss without
//! backpropagating through the matrix inverse.
//!
//! All matrices are k×k where k is the representation dimension (typically
//! 128–512), so inversion is O(k³) which is negligible on CPU. No GPU/WGSL
//! needed.
//!
//! SIMD kernels use `float64x2_t` (NEON, 2-wide) or `__m256d` (AVX2, 4-wide)
//! for the outer-product and dot-product hot loops.

// ── SIMD f64 helpers (only used in this module) ──────────────────

/// SIMD outer product + EMA update for f64 covariance tracking.
///
/// Computes for each (i,j):
///   sigma_ij = s[i] * t[j]
///   n_ij     = (s[i] * s[j] + t[i] * t[j]) / 2
///
/// Then applies EMA:
///   dst_sigma[idx] = alpha * dst_sigma[idx] + (1 - alpha) * sigma_ij   (or sigma_ij if first_step)
///   dst_n[idx]     = alpha * dst_n[idx]     + (1 - alpha) * n_ij       (or n_ij if first_step)
#[inline]
fn simd_outer_product_ema_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
    alpha: f64,
    first_step: bool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_outer_product_ema_f64(dst_sigma, dst_n, student, teacher, k, alpha, first_step)
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            unsafe {
                avx2_outer_product_ema_f64(dst_sigma, dst_n, student, teacher, k, alpha, first_step)
            }
        } else {
            scalar_outer_product_ema_f64(dst_sigma, dst_n, student, teacher, k, alpha, first_step)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_ema_f64(dst_sigma, dst_n, student, teacher, k, alpha, first_step)
    }
}

/// SIMD outer product (no EMA) for f64 sample covariance.
///
/// Computes for each (i,j):
///   dst_sigma[i*k+j] = s[i] * t[j]
///   dst_n[i*k+j]     = (s[i] * s[j] + t[i] * t[j]) / 2
#[inline]
fn simd_outer_product_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_outer_product_f64(dst_sigma, dst_n, student, teacher, k) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            unsafe { avx2_outer_product_f64(dst_sigma, dst_n, student, teacher, k) }
        } else {
            scalar_outer_product_f64(dst_sigma, dst_n, student, teacher, k)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_f64(dst_sigma, dst_n, student, teacher, k)
    }
}

/// SIMD f64 dot product: `a · b` for length `len`.
#[inline]
fn simd_dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_f64(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            unsafe { avx2_dot_f64(a, b, len) }
        } else {
            scalar_dot_f64(a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_dot_f64(a, b, len)
    }
}

// ── Scalar fallbacks ─────────────────────────────────────────────

#[inline]
#[allow(dead_code)]
fn scalar_outer_product_ema_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
    alpha: f64,
    first_step: bool,
) {
    let one_minus_alpha = 1.0 - alpha;
    for i in 0..k {
        let si = student[i] as f64;
        let ti = teacher[i] as f64;
        for j in 0..k {
            let sj = student[j] as f64;
            let tj = teacher[j] as f64;
            let sigma_ij = si * tj;
            let n_ij = (si * sj + ti * tj) / 2.0;
            let idx = i * k + j;
            if first_step {
                dst_sigma[idx] = sigma_ij;
                dst_n[idx] = n_ij;
            } else {
                dst_sigma[idx] = alpha * dst_sigma[idx] + one_minus_alpha * sigma_ij;
                dst_n[idx] = alpha * dst_n[idx] + one_minus_alpha * n_ij;
            }
        }
    }
}

#[inline]
#[allow(dead_code)]
fn scalar_outer_product_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
) {
    for i in 0..k {
        let si = student[i] as f64;
        let ti = teacher[i] as f64;
        for j in 0..k {
            let sj = student[j] as f64;
            let tj = teacher[j] as f64;
            dst_sigma[i * k + j] = si * tj;
            dst_n[i * k + j] = (si * sj + ti * tj) / 2.0;
        }
    }
}

#[inline]
#[allow(dead_code)]
fn scalar_dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..len {
        unsafe {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
    }
    sum
}

// ── NEON (aarch64) f64 SIMD ──────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_outer_product_ema_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
    alpha: f64,
    first_step: bool,
) {
    use core::arch::aarch64::{
        vdupq_n_f64, vld1q_dup_f64, vld1q_f64, vmlaq_f64, vmulq_f64, vst1q_f64,
    };

    unsafe {
        let one_minus_alpha = 1.0 - alpha;
        let v_alpha = vdupq_n_f64(alpha);
        let v_oma = vdupq_n_f64(one_minus_alpha);
        let n_chunks = k / 2;

        for i in 0..k {
            let si = *student.get_unchecked(i) as f64;
            let ti = *teacher.get_unchecked(i) as f64;
            let v_si = vld1q_dup_f64(&si);
            let v_ti = vld1q_dup_f64(&ti);

            let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
            let row_n = dst_n.as_mut_ptr().add(i * k);

            let mut j = 0;
            for _ in 0..n_chunks {
                // Load f32 pairs and convert to f64
                let sj0 = *student.get_unchecked(j) as f64;
                let sj1 = *student.get_unchecked(j + 1) as f64;
                let v_sj = vld1q_f64([sj0, sj1].as_ptr());

                let tj0 = *teacher.get_unchecked(j) as f64;
                let tj1 = *teacher.get_unchecked(j + 1) as f64;
                let v_tj = vld1q_f64([tj0, tj1].as_ptr());

                // sigma_ij = si * tj
                let v_sigma = vmulq_f64(v_si, v_tj);
                // n_ij = (si * sj + ti * tj) / 2
                let v_n = vmulq_f64(v_si, v_sj);
                let v_n = vmlaq_f64(v_n, v_ti, v_tj);
                let half = vdupq_n_f64(0.5);
                let v_n = vmulq_f64(v_n, half);

                if first_step {
                    vst1q_f64(row_sigma.add(j), v_sigma);
                    vst1q_f64(row_n.add(j), v_n);
                } else {
                    let v_old_sigma = vld1q_f64(row_sigma.add(j));
                    let v_old_n = vld1q_f64(row_n.add(j));
                    vst1q_f64(
                        row_sigma.add(j),
                        vmlaq_f64(vmulq_f64(v_alpha, v_old_sigma), v_oma, v_sigma),
                    );
                    vst1q_f64(
                        row_n.add(j),
                        vmlaq_f64(vmulq_f64(v_alpha, v_old_n), v_oma, v_n),
                    );
                }
                j += 2;
            }

            // Scalar tail
            while j < k {
                let sj = *student.get_unchecked(j) as f64;
                let tj = *teacher.get_unchecked(j) as f64;
                let sigma_ij = si * tj;
                let n_ij = (si * sj + ti * tj) * 0.5;
                if first_step {
                    *dst_sigma.get_unchecked_mut(i * k + j) = sigma_ij;
                    *dst_n.get_unchecked_mut(i * k + j) = n_ij;
                } else {
                    let idx = i * k + j;
                    *dst_sigma.get_unchecked_mut(idx) =
                        alpha * *dst_sigma.get_unchecked(idx) + one_minus_alpha * sigma_ij;
                    *dst_n.get_unchecked_mut(idx) =
                        alpha * *dst_n.get_unchecked(idx) + one_minus_alpha * n_ij;
                }
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_outer_product_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
) {
    use core::arch::aarch64::{
        vaddq_f64, vdupq_n_f64, vld1q_dup_f64, vld1q_f64, vmulq_f64, vst1q_f64,
    };

    unsafe {
        let n_chunks = k / 2;
        let half = vdupq_n_f64(0.5);

        for i in 0..k {
            let si = *student.get_unchecked(i) as f64;
            let ti = *teacher.get_unchecked(i) as f64;
            let v_si = vld1q_dup_f64(&si);
            let v_ti = vld1q_dup_f64(&ti);

            let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
            let row_n = dst_n.as_mut_ptr().add(i * k);

            let mut j = 0;
            for _ in 0..n_chunks {
                let sj0 = *student.get_unchecked(j) as f64;
                let sj1 = *student.get_unchecked(j + 1) as f64;
                let v_sj = vld1q_f64([sj0, sj1].as_ptr());

                let tj0 = *teacher.get_unchecked(j) as f64;
                let tj1 = *teacher.get_unchecked(j + 1) as f64;
                let v_tj = vld1q_f64([tj0, tj1].as_ptr());

                let v_sigma = vmulq_f64(v_si, v_tj);
                // n = (si*sj + ti*tj) / 2
                let v_n = vmulq_f64(v_si, v_sj);
                let v_n = vaddq_f64(v_n, vmulq_f64(v_ti, v_tj));
                let v_n = vmulq_f64(v_n, half);

                vst1q_f64(row_sigma.add(j), v_sigma);
                vst1q_f64(row_n.add(j), v_n);
                j += 2;
            }

            while j < k {
                let sj = *student.get_unchecked(j) as f64;
                let tj = *teacher.get_unchecked(j) as f64;
                *dst_sigma.get_unchecked_mut(i * k + j) = si * tj;
                *dst_n.get_unchecked_mut(i * k + j) = (si * sj + ti * tj) * 0.5;
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    use core::arch::aarch64::{vaddq_f64, vaddvq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64};

    unsafe {
        let n_chunks = len / 4; // accumulate 2 pairs of 2-wide
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);

        let mut i = 0;
        for _ in 0..n_chunks {
            let va0 = vld1q_f64(a.as_ptr().add(i));
            let vb0 = vld1q_f64(b.as_ptr().add(i));
            acc0 = vaddq_f64(acc0, vmulq_f64(va0, vb0));

            let va1 = vld1q_f64(a.as_ptr().add(i + 2));
            let vb1 = vld1q_f64(b.as_ptr().add(i + 2));
            acc1 = vaddq_f64(acc1, vmulq_f64(va1, vb1));

            i += 4;
        }

        let mut sum = vaddvq_f64(vaddq_f64(acc0, acc1));

        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

// ── AVX2 (x86_64) f64 SIMD ──────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_outer_product_ema_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
    alpha: f64,
    first_step: bool,
) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_broadcast_sd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_storeu_pd,
    };

    unsafe {
        let one_minus_alpha = 1.0 - alpha;
        let v_alpha = _mm256_set1_pd(alpha);
        let v_oma = _mm256_set1_pd(one_minus_alpha);
        let n_chunks = k / 4;

        for i in 0..k {
            let si = *student.get_unchecked(i) as f64;
            let ti = *teacher.get_unchecked(i) as f64;
            let v_si = _mm256_broadcast_sd(&si);
            let v_ti = _mm256_broadcast_sd(&ti);

            let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
            let row_n = dst_n.as_mut_ptr().add(i * k);

            let mut j = 0;
            for _ in 0..n_chunks {
                // Load 4 student/teacher f32 values and broadcast to f64
                let mut sj_buf = [0.0f64; 4];
                let mut tj_buf = [0.0f64; 4];
                for b in 0..4 {
                    sj_buf[b] = *student.get_unchecked(j + b) as f64;
                    tj_buf[b] = *teacher.get_unchecked(j + b) as f64;
                }
                let v_sj = _mm256_loadu_pd(sj_buf.as_ptr());
                let v_tj = _mm256_loadu_pd(tj_buf.as_ptr());

                // sigma_ij = si * tj
                let v_sigma = _mm256_mul_pd(v_si, v_tj);
                // n_ij = (si * sj + ti * tj) / 2
                let v_n = _mm256_mul_pd(v_si, v_sj);
                let v_n = _mm256_add_pd(v_n, _mm256_mul_pd(v_ti, v_tj));
                let half = _mm256_set1_pd(0.5);
                let v_n = _mm256_mul_pd(v_n, half);

                if first_step {
                    _mm256_storeu_pd(row_sigma.add(j), v_sigma);
                    _mm256_storeu_pd(row_n.add(j), v_n);
                } else {
                    let v_old_sigma = _mm256_loadu_pd(row_sigma.add(j));
                    let v_old_n = _mm256_loadu_pd(row_n.add(j));
                    // EMA: alpha * old + (1-alpha) * new
                    _mm256_storeu_pd(
                        row_sigma.add(j),
                        _mm256_add_pd(
                            _mm256_mul_pd(v_alpha, v_old_sigma),
                            _mm256_mul_pd(v_oma, v_sigma),
                        ),
                    );
                    _mm256_storeu_pd(
                        row_n.add(j),
                        _mm256_add_pd(_mm256_mul_pd(v_alpha, v_old_n), _mm256_mul_pd(v_oma, v_n)),
                    );
                }
                j += 4;
            }

            // Scalar tail
            while j < k {
                let sj = *student.get_unchecked(j) as f64;
                let tj = *teacher.get_unchecked(j) as f64;
                let sigma_ij = si * tj;
                let n_ij = (si * sj + ti * tj) * 0.5;
                if first_step {
                    *dst_sigma.get_unchecked_mut(i * k + j) = sigma_ij;
                    *dst_n.get_unchecked_mut(i * k + j) = n_ij;
                } else {
                    let idx = i * k + j;
                    *dst_sigma.get_unchecked_mut(idx) =
                        alpha * *dst_sigma.get_unchecked(idx) + one_minus_alpha * sigma_ij;
                    *dst_n.get_unchecked_mut(idx) =
                        alpha * *dst_n.get_unchecked(idx) + one_minus_alpha * n_ij;
                }
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_outer_product_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_broadcast_sd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_storeu_pd,
    };

    unsafe {
        let n_chunks = k / 4;
        let half = _mm256_set1_pd(0.5);

        for i in 0..k {
            let si = *student.get_unchecked(i) as f64;
            let ti = *teacher.get_unchecked(i) as f64;
            let v_si = _mm256_broadcast_sd(&si);
            let v_ti = _mm256_broadcast_sd(&ti);

            let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
            let row_n = dst_n.as_mut_ptr().add(i * k);

            let mut j = 0;
            for _ in 0..n_chunks {
                let mut sj_buf = [0.0f64; 4];
                let mut tj_buf = [0.0f64; 4];
                for b in 0..4 {
                    sj_buf[b] = *student.get_unchecked(j + b) as f64;
                    tj_buf[b] = *teacher.get_unchecked(j + b) as f64;
                }
                let v_sj = _mm256_loadu_pd(sj_buf.as_ptr());
                let v_tj = _mm256_loadu_pd(tj_buf.as_ptr());

                let v_sigma = _mm256_mul_pd(v_si, v_tj);
                let v_n = _mm256_mul_pd(v_si, v_sj);
                let v_n = _mm256_add_pd(v_n, _mm256_mul_pd(v_ti, v_tj));
                let v_n = _mm256_mul_pd(v_n, half);

                _mm256_storeu_pd(row_sigma.add(j), v_sigma);
                _mm256_storeu_pd(row_n.add(j), v_n);
                j += 4;
            }

            while j < k {
                let sj = *student.get_unchecked(j) as f64;
                let tj = *teacher.get_unchecked(j) as f64;
                *dst_sigma.get_unchecked_mut(i * k + j) = si * tj;
                *dst_n.get_unchecked_mut(i * k + j) = (si * sj + ti * tj) * 0.5;
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_castpd256_pd128, _mm256_extractf128_pd, _mm256_loadu_pd,
        _mm256_mul_pd,
    };

    unsafe {
        let n_chunks = len / 8; // 2× __m256d accumulators
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();

        let mut i = 0;
        for _ in 0..n_chunks {
            let va0 = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_pd(b.as_ptr().add(i));
            acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(va0, vb0));

            let va1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
            let vb1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(va1, vb1));

            i += 8;
        }

        // Handle remaining 4-wide chunks
        let mut acc = _mm256_add_pd(acc0, acc1);
        let n_rem4 = (len - i) / 4;
        for _ in 0..n_rem4 {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
            i += 4;
        }

        // Horizontal sum of 4-wide accumulator
        let mut sum = horizontal_sum_256d(acc);

        // Scalar tail
        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn horizontal_sum_256d(v: core::arch::x86_64::__m256d) -> f64 {
    use core::arch::x86_64::{
        _mm_add_pd, _mm_add_sd, _mm_castpd_ps, _mm_cvtss_f32, _mm_unpackhi_pd,
        _mm256_castpd256_pd128, _mm256_extractf128_pd,
    };
    unsafe {
        let hi = _mm256_extractf128_pd(v, 1);
        let lo = _mm256_castpd256_pd128(v);
        let sum128 = _mm_add_pd(lo, hi);
        // sum128 has [s0, s1], shuffle to get [s1, s1]
        let shuf = _mm_unpackhi_pd(sum128, sum128);
        let result = _mm_add_sd(sum128, shuf);
        // Extract the lower f64
        let mut dst = [0.0f64; 2];
        core::arch::x86_64::_mm_storeu_pd(dst.as_mut_ptr(), result);
        dst[0]
    }
}

/// Configuration for PEIRA distillation.
///
/// Controls the regularization strength (λ), EMA momentum for covariance
/// tracking, and representation dimension.
#[derive(Debug, Clone)]
pub struct PeiraConfig {
    /// Regularization parameter λ > 0.
    ///
    /// Controls the effective rank of recovered CCA subspace:
    /// - Larger λ → fewer canonical directions recovered (more conservative)
    /// - Smaller λ → more directions (more expressive, potentially noisy)
    ///
    /// Default: 0.1
    pub lambda: f64,
    /// EMA momentum for covariance estimates (0 < α < 1).
    ///
    /// Higher = slower tracking, more stable. Lower = faster adaptation.
    ///
    /// Default: 0.9
    pub ema_rate: f64,
    /// Representation dimension k.
    /// All internal matrices are k×k.
    pub dim: usize,
}

impl Default for PeiraConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            ema_rate: 0.9,
            dim: 8,
        }
    }
}

impl PeiraConfig {
    /// Create a new config with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Set regularization λ.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        assert!(lambda > 0.0, "PEIRA λ must be positive, got {lambda}");
        self.lambda = lambda;
        self
    }

    /// Set EMA momentum.
    pub fn with_ema_rate(mut self, rate: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&rate),
            "EMA rate must be in (0, 1), got {rate}"
        );
        self.ema_rate = rate;
        self
    }
}

/// EMA covariance tracker for PEIRA.
///
/// Maintains running estimates of:
/// - **Σ** (cross-view covariance): how student and teacher representations co-vary
/// - **N** (within-view covariance): auto-covariance averaged over both views
///
/// Both are k×k matrices stored in row-major flat layout.
#[derive(Debug, Clone)]
pub struct PeiraCovariance {
    /// Cross-view covariance Σ (k×k), row-major.
    sigma: Vec<f64>,
    /// Within-view covariance N (k×k), row-major.
    n: Vec<f64>,
    /// Configuration.
    config: PeiraConfig,
    /// Number of EMA updates applied.
    step_count: usize,
    /// Pre-allocated scratch for peira_aux_loss
    sigma_sample: Vec<f64>,
    n_sample: Vec<f64>,
    pm: Vec<f64>,
}

impl PeiraCovariance {
    /// Create a new zero-initialized covariance tracker.
    pub fn new(config: PeiraConfig) -> Self {
        let k = config.dim;
        Self {
            sigma: vec![0.0; k * k],
            n: vec![0.0; k * k],
            config,
            step_count: 0,
            sigma_sample: vec![0.0; k * k],
            n_sample: vec![0.0; k * k],
            pm: vec![0.0; k * k],
        }
    }

    /// Get the dimension k.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Get the number of updates.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Update EMA covariance estimates with a student-teacher pair.
    ///
    /// Both slices must have length `dim`.
    pub fn update(&mut self, student: &[f32], teacher: &[f32]) {
        let k = self.config.dim;
        assert_eq!(student.len(), k, "student repr length mismatch");
        assert_eq!(teacher.len(), k, "teacher repr length mismatch");

        let alpha = self.config.ema_rate;

        // SIMD-accelerated outer product + EMA update
        simd_outer_product_ema_f64(
            &mut self.sigma,
            &mut self.n,
            student,
            teacher,
            k,
            alpha,
            self.step_count == 0,
        );
        self.step_count += 1;
    }

    /// Compute the closed-form predictor matrices (P*, Q*).
    ///
    /// - P* = Σ (N + λI)⁻¹  — the optimal linear predictor
    /// - Q* = (N + λI)⁻¹     — the regularized inverse
    ///
    /// Returns (P*, Q*) as flat k×k row-major vectors.
    pub fn predictor(&self) -> (Vec<f64>, Vec<f64>) {
        let k = self.config.dim;
        let lambda = self.config.lambda;

        // Build N + λI
        let mut n_reg = self.n.clone();
        for i in 0..k {
            n_reg[i * k + i] += lambda;
        }

        // Invert N + λI
        let q_star = invert_spd(&n_reg, k);

        // P* = Σ @ Q*
        let p_star = matmul(&self.sigma, &q_star, k);

        (p_star, q_star)
    }

    /// Get a reference to the current Σ matrix (row-major).
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Get a reference to the current N matrix (row-major).
    pub fn n_matrix(&self) -> &[f64] {
        &self.n
    }

    /// Reset covariance estimates (e.g., at episode boundaries).
    pub fn reset(&mut self) {
        self.sigma.fill(0.0);
        self.n.fill(0.0);
        self.sigma_sample.fill(0.0);
        self.n_sample.fill(0.0);
        self.pm.fill(0.0);
        self.step_count = 0;
    }
}

/// Compute the PEIRA auxiliary loss L_aux.
///
/// L_aux = -½ Tr(Σ A^T) + ¼ Tr(A (N + λI) A^T)
///
/// This formulation avoids differentiating through the matrix inverse.
/// At the optimum A* = (N + λI)⁻¹ Σ^T, L_aux equals the PEIRA objective.
///
/// # Arguments
/// * `student` — Student representation (length k)
/// * `teacher` — Teacher representation (length k)
/// * `p_star` — Predictor matrix P* = Σ(N + λI)⁻¹ (k×k row-major)
/// * `q_star` — Inverse Q* = (N + λI)⁻¹ (k×k row-major)
/// * `lambda` — Regularization parameter
///
/// # Returns
/// The scalar auxiliary loss value.
pub fn peira_aux_loss(
    student: &[f32],
    teacher: &[f32],
    p_star: &[f64],
    q_star: &[f64],
    lambda: f64,
    sigma_sample: &mut [f64],
    n_sample: &mut [f64],
    pm: &mut [f64],
) -> f64 {
    let k = student.len();
    assert_eq!(teacher.len(), k);
    assert_eq!(p_star.len(), k * k);
    assert_eq!(q_star.len(), k * k);
    assert_eq!(sigma_sample.len(), k * k);
    assert_eq!(n_sample.len(), k * k);
    assert_eq!(pm.len(), k * k);

    // Compute the auxiliary loss using the closed-form predictor:
    // L_aux = -½ Tr(Σ P*^T) + ¼ Tr(P* (N + λI) P*^T)
    //
    // Since P* = Σ Q*, and Q* = (N + λI)⁻¹:
    // The loss simplifies to: -½ Tr(Σ Q* Σ^T) + ¼ Tr(Σ Q* Σ^T)
    //                        = -¼ Tr(P* Σ^T)
    //
    // But for numerical accuracy, we compute the full form using the
    // current sample's outer products.

    // Compute sample cross-covariance: sigma_sample = u ⊗ v
    // and sample within-covariance: n_sample = (u ⊗ u + v ⊗ v) / 2
    // SIMD-accelerated outer product
    simd_outer_product_f64(sigma_sample, n_sample, student, teacher, k);

    // Term 1: -½ Tr(Σ_sample @ P*^T) = -½ Σ_{i,j} sigma_sample[i,j] * p_star[j,i]
    // In row-major: Tr(A @ B^T) = Σ_{i,j} A[i,j] * B[i,j] when both are row-major
    let term1 = -0.5 * simd_dot_f64(sigma_sample, p_star, k * k);

    // Term 2: ¼ Tr(P* @ (N_sample + λI) @ P*^T)
    // = ¼ Tr(P* @ M @ P*^T) where M = N_sample + λI
    // P* @ M is k×k, then (P* @ M) @ P*^T trace
    pm.fill(0.0);
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f64;
            for l in 0..k {
                let m_lj = if l == j {
                    n_sample[l * k + j] + lambda
                } else {
                    n_sample[l * k + j]
                };
                sum += p_star[i * k + l] * m_lj;
            }
            pm[i * k + j] = sum;
        }
    }

    // Tr(PM @ P^T) = Σ_{i,j} pm[i,j] * p_star[i,j]
    let term2 = 0.25 * simd_dot_f64(pm, p_star, k * k);

    // Add the regularization penalty: + λ/2 (||u||² + ||v||²)
    // Use SIMD dot(x, x) = ||x||² instead of scalar .powi(2)
    let norm_sq_u = crate::simd::simd_dot_f32(student, student, student.len()) as f64;
    let norm_sq_v = crate::simd::simd_dot_f32(teacher, teacher, teacher.len()) as f64;
    let reg = lambda / 2.0 * (norm_sq_u + norm_sq_v);

    term1 + term2 + reg
}

/// Invert a symmetric positive definite (SPD) matrix using Cholesky decomposition.
/// More efficient than Gauss-Jordan for SPD matrices: exploits symmetry,
/// no partial pivoting needed, uses half the memory.
fn invert_spd(mat: &[f64], k: usize) -> Vec<f64> {
    // Step 1: Cholesky decomposition — L such that L * L^T = mat
    let mut l = vec![0.0f64; k * k];
    for j in 0..k {
        // Diagonal
        let mut sum = 0.0f64;
        for p in 0..j {
            sum += l[j * k + p] * l[j * k + p];
        }
        let diag = mat[j * k + j] - sum;
        assert!(
            diag > 0.0,
            "Matrix not positive definite in Cholesky decomposition"
        );
        l[j * k + j] = diag.sqrt();

        // Off-diagonal (lower triangle only)
        for i in (j + 1)..k {
            let mut sum = 0.0f64;
            for p in 0..j {
                sum += l[i * k + p] * l[j * k + p];
            }
            l[i * k + j] = (mat[i * k + j] - sum) / l[j * k + j];
        }
    }

    // Step 2: Invert lower triangular L → L_inv
    let mut l_inv = vec![0.0f64; k * k];
    for i in 0..k {
        l_inv[i * k + i] = 1.0 / l[i * k + i];
        for j in 0..i {
            let mut sum = 0.0f64;
            for p in j..i {
                sum += l[i * k + p] * l_inv[p * k + j];
            }
            l_inv[i * k + j] = -sum / l[i * k + i];
        }
    }

    // Step 3: M_inv = L_inv^T * L_inv (only lower triangle, then mirror)
    let mut inv = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0f64;
            for p in i..k {
                sum += l_inv[p * k + i] * l_inv[p * k + j];
            }
            inv[i * k + j] = sum;
            inv[j * k + i] = sum; // symmetric
        }
    }

    inv
}

/// Compute matrix product C = A @ B where all are k×k row-major.
/// Uses SIMD f64 dot product for the inner accumulation loop.
fn matmul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * k];
    // Transpose B for sequential access in the inner loop
    let mut bt = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            bt[j * k + i] = b[i * k + j];
        }
    }
    for i in 0..k {
        let a_row = &a[i * k..i * k + k];
        for j in 0..k {
            let b_col = &bt[j * k..j * k + k];
            c[i * k + j] = simd_dot_f64(a_row, b_col, k);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peira_config_validates() {
        let cfg = PeiraConfig::new(16).with_lambda(0.5).with_ema_rate(0.95);
        assert_eq!(cfg.dim, 16);
        assert_eq!(cfg.lambda, 0.5);
        assert_eq!(cfg.ema_rate, 0.95);
    }

    #[test]
    #[should_panic(expected = "must be positive")]
    fn peira_config_rejects_zero_lambda() {
        PeiraConfig::new(4).with_lambda(0.0);
    }

    #[test]
    fn matrix_inverse_identity() {
        let k = 3;
        let identity: Vec<f64> = (0..k)
            .flat_map(|i| (0..k).map(move |j| if i == j { 1.0 } else { 0.0 }))
            .collect();
        let inv = invert_spd(&identity, k);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * k + j] - expected).abs() < 1e-10,
                    "inv[{i},{j}] = {} expected {expected}",
                    inv[i * k + j]
                );
            }
        }
    }

    #[test]
    fn matrix_inverse_known() {
        // [[2, 1], [1, 3]] inverse is [[3/5, -1/5], [-1/5, 2/5]]
        let k = 2;
        let mat = vec![2.0, 1.0, 1.0, 3.0];
        let inv = invert_spd(&mat, k);
        let expected = vec![0.6, -0.2, -0.2, 0.4];
        for i in 0..4 {
            assert!(
                (inv[i] - expected[i]).abs() < 1e-10,
                "inv[{i}] = {} expected {}",
                inv[i],
                expected[i]
            );
        }
    }

    #[test]
    fn ema_covariance_tracks_identity() {
        // Feed many identical student=teacher pairs → Σ and N should converge to I
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k).with_ema_rate(0.5));

        let repr: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        for _ in 0..100 {
            cov.update(&repr, &repr);
        }

        // Σ[0,0] should be ~1.0, Σ[i,j] for i≠j should be ~0
        let sigma = cov.sigma();
        assert!((sigma[0] - 1.0).abs() < 0.1, "Σ[0,0] = {}", sigma[0]);
        assert!(sigma[1].abs() < 0.1, "Σ[0,1] = {}", sigma[1]);

        // N[0,0] should be ~1.0 (auto-covariance of [1,0,0,0])
        let n = cov.n_matrix();
        assert!((n[0] - 1.0).abs() < 0.1, "N[0,0] = {}", n[0]);
    }

    #[test]
    fn predictor_yields_valid_matrices() {
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k).with_lambda(0.1));

        // Feed correlated views
        for _ in 0..50 {
            let student: Vec<f32> = vec![1.0, 0.5, 0.0, 0.0];
            let teacher: Vec<f32> = vec![0.8, 0.4, 0.0, 0.0];
            cov.update(&student, &teacher);
        }

        let (p_star, q_star) = cov.predictor();
        assert_eq!(p_star.len(), k * k);
        assert_eq!(q_star.len(), k * k);

        // Q* should be symmetric positive definite (diagonal dominant)
        for i in 0..k {
            assert!(
                q_star[i * k + i] > 0.0,
                "Q*[{i},{i}] = {} not positive",
                q_star[i * k + i]
            );
        }
    }

    #[test]
    fn aux_loss_is_finite() {
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k));
        cov.update(&[1.0, 0.5, -0.3, 0.0], &[0.8, 0.4, -0.2, 0.1]);
        let (p, q) = cov.predictor();
        let mut sigma_sample = vec![0.0f64; k * k];
        let mut n_sample = vec![0.0f64; k * k];
        let mut pm = vec![0.0f64; k * k];
        let loss = peira_aux_loss(
            &[1.0, 0.5, -0.3, 0.0],
            &[0.8, 0.4, -0.2, 0.1],
            &p,
            &q,
            0.1,
            &mut sigma_sample,
            &mut n_sample,
            &mut pm,
        );
        assert!(loss.is_finite(), "Loss is not finite: {loss}");
    }
}
