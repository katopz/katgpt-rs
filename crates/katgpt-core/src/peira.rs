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
#[allow(clippy::too_many_arguments)]
fn simd_outer_product_ema_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
    alpha: f64,
    first_step: bool,
    s_scratch: &mut [f64],
    t_scratch: &mut [f64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (s_scratch, t_scratch); // unused in NEON path
        unsafe {
            neon_outer_product_ema_f64(dst_sigma, dst_n, student, teacher, k, alpha, first_step)
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            let _ = (s_scratch, t_scratch); // unused in AVX2 path
            unsafe {
                avx2_outer_product_ema_f64(dst_sigma, dst_n, student, teacher, k, alpha, first_step)
            }
        } else {
            scalar_outer_product_ema_f64(
                dst_sigma, dst_n, student, teacher, k, alpha, first_step, s_scratch, t_scratch,
            )
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_ema_f64(
            dst_sigma, dst_n, student, teacher, k, alpha, first_step, s_scratch, t_scratch,
        )
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
    s_scratch: &mut [f64],
    t_scratch: &mut [f64],
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (s_scratch, t_scratch); // unused in NEON path
        unsafe { neon_outer_product_f64(dst_sigma, dst_n, student, teacher, k) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            let _ = (s_scratch, t_scratch); // unused in AVX2 path
            unsafe { avx2_outer_product_f64(dst_sigma, dst_n, student, teacher, k) }
        } else {
            scalar_outer_product_f64(dst_sigma, dst_n, student, teacher, k, s_scratch, t_scratch)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_f64(dst_sigma, dst_n, student, teacher, k, s_scratch, t_scratch)
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
#[allow(clippy::too_many_arguments)]
fn scalar_outer_product_ema_f64(
    dst_sigma: &mut [f64],
    dst_n: &mut [f64],
    student: &[f32],
    teacher: &[f32],
    k: usize,
    alpha: f64,
    first_step: bool,
    s_scratch: &mut [f64],
    t_scratch: &mut [f64],
) {
    // Pre-cast f32 → f64 into caller-provided scratch (avoids per-call Vec allocation).
    for i in 0..k {
        s_scratch[i] = student[i] as f64;
        t_scratch[i] = teacher[i] as f64;
    }
    if first_step {
        // First step: direct assignment (no EMA blending)
        for i in 0..k {
            let si = s_scratch[i];
            let ti = t_scratch[i];
            let row_off = i * k;
            for j in 0..k {
                let sj = s_scratch[j];
                let tj = t_scratch[j];
                let idx = row_off + j;
                dst_sigma[idx] = si * tj;
                dst_n[idx] = (si * sj + ti * tj) / 2.0;
            }
        }
    } else {
        // Subsequent steps: EMA blending
        let one_minus_alpha = 1.0 - alpha;
        for i in 0..k {
            let si = s_scratch[i];
            let ti = t_scratch[i];
            let row_off = i * k;
            for j in 0..k {
                let sj = s_scratch[j];
                let tj = t_scratch[j];
                let idx = row_off + j;
                dst_sigma[idx] = alpha * dst_sigma[idx] + one_minus_alpha * (si * tj);
                dst_n[idx] = alpha * dst_n[idx] + one_minus_alpha * ((si * sj + ti * tj) / 2.0);
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
    s_scratch: &mut [f64],
    t_scratch: &mut [f64],
) {
    // Pre-cast f32 → f64 into caller-provided scratch (avoids per-call Vec allocation).
    for i in 0..k {
        s_scratch[i] = student[i] as f64;
        t_scratch[i] = teacher[i] as f64;
    }
    for i in 0..k {
        let si = s_scratch[i];
        let ti = t_scratch[i];
        let row_off = i * k;
        for j in 0..k {
            let sj = s_scratch[j];
            let tj = t_scratch[j];
            dst_sigma[row_off + j] = si * tj;
            dst_n[row_off + j] = (si * sj + ti * tj) / 2.0;
        }
    }
}

#[inline]
#[allow(dead_code)]
fn scalar_dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    // 4 independent accumulators (4 elements per outer iter) — same pattern as
    // the f32 SIMD kernels. Single-accumulator dot is FMA-latency-bound; 4 lanes
    // keep the pipeline full and let LLVM emit 4-wide unrolled FMA on targets
    // without hardware f64 SIMD (WASM, RISC-V, debug builds).
    let mut acc = [0.0f64; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] += *a.get_unchecked(i) * *b.get_unchecked(i);
            acc[1] += *a.get_unchecked(i + 1) * *b.get_unchecked(i + 1);
            acc[2] += *a.get_unchecked(i + 2) * *b.get_unchecked(i + 2);
            acc[3] += *a.get_unchecked(i + 3) * *b.get_unchecked(i + 3);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f64>();
    while i < len {
        unsafe {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        i += 1;
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
        vcvt_f64_f32, vdupq_n_f64, vld1_f32, vld1q_dup_f64, vld1q_f64, vmlaq_f64, vmulq_f64,
        vst1q_f64,
    };

    unsafe {
        let n_chunks = k / 2;
        let half = vdupq_n_f64(0.5);

        if first_step {
            // First step: direct assignment (no EMA blending, no load from dst)
            for i in 0..k {
                let si = *student.get_unchecked(i) as f64;
                let ti = *teacher.get_unchecked(i) as f64;
                let v_si = vld1q_dup_f64(&si);
                let v_ti = vld1q_dup_f64(&ti);

                let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
                let row_n = dst_n.as_mut_ptr().add(i * k);

                let mut j = 0;
                for _ in 0..n_chunks {
                    // Hardware f32→f64 widening via vcvt_f64_f32 — avoids 2 scalar casts + stack roundtrip.
                    let v_sj = vcvt_f64_f32(vld1_f32(student.as_ptr().add(j)));
                    let v_tj = vcvt_f64_f32(vld1_f32(teacher.as_ptr().add(j)));

                    let v_sigma = vmulq_f64(v_si, v_tj);
                    let v_n = vmulq_f64(v_si, v_sj);
                    let v_n = vmlaq_f64(v_n, v_ti, v_tj);
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
        } else {
            // Subsequent steps: EMA blending
            let one_minus_alpha = 1.0 - alpha;
            let v_alpha = vdupq_n_f64(alpha);
            let v_oma = vdupq_n_f64(one_minus_alpha);

            for i in 0..k {
                let si = *student.get_unchecked(i) as f64;
                let ti = *teacher.get_unchecked(i) as f64;
                let v_si = vld1q_dup_f64(&si);
                let v_ti = vld1q_dup_f64(&ti);

                let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
                let row_n = dst_n.as_mut_ptr().add(i * k);

                let mut j = 0;
                for _ in 0..n_chunks {
                    // Hardware f32→f64 widening via vcvt_f64_f32 — avoids 2 scalar casts + stack roundtrip.
                    let v_sj = vcvt_f64_f32(vld1_f32(student.as_ptr().add(j)));
                    let v_tj = vcvt_f64_f32(vld1_f32(teacher.as_ptr().add(j)));

                    let v_sigma = vmulq_f64(v_si, v_tj);
                    let v_n = vmulq_f64(v_si, v_sj);
                    let v_n = vmlaq_f64(v_n, v_ti, v_tj);
                    let v_n = vmulq_f64(v_n, half);

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
                    j += 2;
                }

                while j < k {
                    let sj = *student.get_unchecked(j) as f64;
                    let tj = *teacher.get_unchecked(j) as f64;
                    let idx = i * k + j;
                    let sigma_ij = si * tj;
                    let n_ij = (si * sj + ti * tj) * 0.5;
                    *dst_sigma.get_unchecked_mut(idx) =
                        alpha * *dst_sigma.get_unchecked(idx) + one_minus_alpha * sigma_ij;
                    *dst_n.get_unchecked_mut(idx) =
                        alpha * *dst_n.get_unchecked(idx) + one_minus_alpha * n_ij;
                    j += 1;
                }
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
        vaddq_f64, vcvt_f64_f32, vdupq_n_f64, vld1_f32, vld1q_dup_f64, vmulq_f64, vst1q_f64,
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
                // Hardware f32→f64 widening via vcvt_f64_f32 — avoids 2 scalar casts + stack roundtrip.
                let v_sj = vcvt_f64_f32(vld1_f32(student.as_ptr().add(j)));
                let v_tj = vcvt_f64_f32(vld1_f32(teacher.as_ptr().add(j)));

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
        _mm_loadu_ps, _mm256_add_pd, _mm256_broadcast_sd, _mm256_cvtps_pd, _mm256_loadu_pd,
        _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    unsafe {
        let n_chunks = k / 4;
        let half = _mm256_set1_pd(0.5);

        if first_step {
            // First step: direct assignment (no EMA blending, no load from dst)
            for i in 0..k {
                let si = *student.get_unchecked(i) as f64;
                let ti = *teacher.get_unchecked(i) as f64;
                let v_si = _mm256_broadcast_sd(&si);
                let v_ti = _mm256_broadcast_sd(&ti);

                let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
                let row_n = dst_n.as_mut_ptr().add(i * k);

                let mut j = 0;
                for _ in 0..n_chunks {
                    // Hardware f32→f64 widening via _mm256_cvtps_pd — avoids 4 scalar casts + stack roundtrip.
                    let v_sj = _mm256_cvtps_pd(_mm_loadu_ps(student.as_ptr().add(j)));
                    let v_tj = _mm256_cvtps_pd(_mm_loadu_ps(teacher.as_ptr().add(j)));

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
        } else {
            // Subsequent steps: EMA blending
            let one_minus_alpha = 1.0 - alpha;
            let v_alpha = _mm256_set1_pd(alpha);
            let v_oma = _mm256_set1_pd(one_minus_alpha);

            for i in 0..k {
                let si = *student.get_unchecked(i) as f64;
                let ti = *teacher.get_unchecked(i) as f64;
                let v_si = _mm256_broadcast_sd(&si);
                let v_ti = _mm256_broadcast_sd(&ti);

                let row_sigma = dst_sigma.as_mut_ptr().add(i * k);
                let row_n = dst_n.as_mut_ptr().add(i * k);

                let mut j = 0;
                for _ in 0..n_chunks {
                    // Hardware f32→f64 widening via _mm256_cvtps_pd — avoids 4 scalar casts + stack roundtrip.
                    let v_sj = _mm256_cvtps_pd(_mm_loadu_ps(student.as_ptr().add(j)));
                    let v_tj = _mm256_cvtps_pd(_mm_loadu_ps(teacher.as_ptr().add(j)));

                    let v_sigma = _mm256_mul_pd(v_si, v_tj);
                    let v_n = _mm256_mul_pd(v_si, v_sj);
                    let v_n = _mm256_add_pd(v_n, _mm256_mul_pd(v_ti, v_tj));
                    let v_n = _mm256_mul_pd(v_n, half);

                    let v_old_sigma = _mm256_loadu_pd(row_sigma.add(j));
                    let v_old_n = _mm256_loadu_pd(row_n.add(j));
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
                    j += 4;
                }

                while j < k {
                    let sj = *student.get_unchecked(j) as f64;
                    let tj = *teacher.get_unchecked(j) as f64;
                    let idx = i * k + j;
                    let sigma_ij = si * tj;
                    let n_ij = (si * sj + ti * tj) * 0.5;
                    *dst_sigma.get_unchecked_mut(idx) =
                        alpha * *dst_sigma.get_unchecked(idx) + one_minus_alpha * sigma_ij;
                    *dst_n.get_unchecked_mut(idx) =
                        alpha * *dst_n.get_unchecked(idx) + one_minus_alpha * n_ij;
                    j += 1;
                }
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
        _mm_loadu_ps, _mm256_add_pd, _mm256_broadcast_sd, _mm256_cvtps_pd, _mm256_mul_pd,
        _mm256_set1_pd, _mm256_storeu_pd,
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
                // Hardware f32→f64 widening via _mm256_cvtps_pd — avoids 4 scalar casts + stack roundtrip.
                let v_sj = _mm256_cvtps_pd(_mm_loadu_ps(student.as_ptr().add(j)));
                let v_tj = _mm256_cvtps_pd(_mm_loadu_ps(teacher.as_ptr().add(j)));

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
#[derive(Debug, Clone, Copy)]
pub struct PeiraConfig {
    /// Representation dimension k.
    /// All internal matrices are k×k.
    pub dim: usize,
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
    /// Configuration.
    config: PeiraConfig,
    /// Number of EMA updates applied.
    step_count: usize,
    /// Cross-view covariance Σ (k×k), row-major.
    sigma: Vec<f64>,
    /// Within-view covariance N (k×k), row-major.
    n: Vec<f64>,
    /// Pre-allocated scratch for peira_aux_loss
    sigma_sample: Vec<f64>,
    n_sample: Vec<f64>,
    pm: Vec<f64>,
    /// Pre-allocated scratch for invert_spd_into (L factor, L_inv, matmul bt)
    inv_l_scratch: Vec<f64>,
    inv_l_inv_scratch: Vec<f64>,
    inv_matmul_bt_scratch: Vec<f64>,
    /// Pre-allocated scratch for matmul_into (transposed B)
    matmul_bt_scratch: Vec<f64>,
    /// Pre-allocated output buffers for predictor_with_scratch
    q_star: Vec<f64>,
    p_star: Vec<f64>,
    /// Pre-allocated scratch for f32→f64 conversion in scalar outer product paths.
    s_scratch: Vec<f64>,
    t_scratch: Vec<f64>,
}

impl PeiraCovariance {
    /// Create a new zero-initialized covariance tracker.
    pub fn new(config: PeiraConfig) -> Self {
        let k = config.dim;
        Self {
            config,
            step_count: 0,
            sigma: vec![0.0; k * k],
            n: vec![0.0; k * k],
            sigma_sample: vec![0.0; k * k],
            n_sample: vec![0.0; k * k],
            pm: vec![0.0; k * k],
            inv_l_scratch: vec![0.0; k * k],
            inv_l_inv_scratch: vec![0.0; k * k],
            inv_matmul_bt_scratch: vec![0.0; k * k],
            matmul_bt_scratch: vec![0.0; k * k],
            q_star: vec![0.0; k * k],
            p_star: vec![0.0; k * k],
            s_scratch: vec![0.0; k],
            t_scratch: vec![0.0; k],
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
    #[inline]
    pub fn update(&mut self, student: &[f32], teacher: &[f32]) {
        let k = self.config.dim;
        debug_assert_eq!(student.len(), k, "student repr length mismatch");
        debug_assert_eq!(teacher.len(), k, "teacher repr length mismatch");

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
            &mut self.s_scratch,
            &mut self.t_scratch,
        );
        self.step_count += 1;
    }

    /// Compute the closed-form predictor matrices (P*, Q*).
    ///
    /// - P* = Σ (N + λI)⁻¹  — the optimal linear predictor
    /// - Q* = (N + λI)⁻¹     — the regularized inverse
    ///
    /// Returns (P*, Q*) as flat k×k row-major vectors.
    ///
    /// # Performance
    ///
    /// **This method allocates 5 vectors on every call.** For hot paths,
    /// prefer [`predictor_with_scratch()`] or [`predict_and_loss()`] which
    /// reuse pre-allocated internal buffers and are zero-alloc.
    #[deprecated(note = "allocates 5 vectors per call; use `predictor_with_scratch()` instead")]
    pub fn predictor(&self) -> (Vec<f64>, Vec<f64>) {
        let k = self.config.dim;
        let lambda = self.config.lambda;

        // Build N + λI (allocate once, copy — avoids self.n.clone() overhead)
        let mut n_reg = vec![0.0f64; k * k];
        n_reg.copy_from_slice(&self.n);
        for i in 0..k {
            n_reg[i * k + i] += lambda;
        }

        // Invert N + λI (allocates scratch internally)
        let mut q_star = vec![0.0f64; k * k];
        let mut l_scratch = vec![0.0f64; k * k];
        let mut l_inv_scratch = vec![0.0f64; k * k];
        // Single `bt_scratch` reused for both `invert_spd_into` and `matmul_into`.
        // Previously this was declared twice — the first allocation was leaked.
        let mut bt_scratch = vec![0.0f64; k * k];
        invert_spd_into(
            &mut q_star,
            &mut l_scratch,
            &mut l_inv_scratch,
            &mut bt_scratch,
            &n_reg,
            k,
        );

        // P* = Σ @ Q*
        let mut p_star = vec![0.0f64; k * k];
        matmul_into(&mut p_star, &mut bt_scratch, &self.sigma, &q_star, k);

        (p_star, q_star)
    }

    /// Compute predictor matrices, fully zero-alloc using pre-allocated scratch.
    ///
    /// Same as [`predictor()`] but avoids all allocations by using pre-allocated
    /// internal buffers. Returns references valid until the next mutable call.
    pub fn predictor_with_scratch(&mut self) -> (&[f64], &[f64]) {
        let k = self.config.dim;
        let lambda = self.config.lambda;

        // Build N + λI in scratch buffer (avoids clone)
        self.pm[..k * k].copy_from_slice(&self.n);
        for i in 0..k {
            self.pm[i * k + i] += lambda;
        }

        // Invert N + λI using pre-allocated scratch
        invert_spd_into(
            &mut self.q_star,
            &mut self.inv_l_scratch,
            &mut self.inv_l_inv_scratch,
            &mut self.inv_matmul_bt_scratch,
            &self.pm[..k * k],
            k,
        );

        // P* = Σ @ Q* using pre-allocated scratch
        matmul_into(
            &mut self.p_star,
            &mut self.matmul_bt_scratch,
            &self.sigma,
            &self.q_star,
            k,
        );

        (&self.p_star, &self.q_star)
    }

    /// Combined predictor + auxiliary loss computation (zero-alloc hot path).
    ///
    /// Computes predictor matrices, then evaluates the auxiliary loss on the
    /// given (student, teacher) pair, all using pre-allocated internal scratch.
    /// Returns `(auxiliary_loss, p_star, q_star)` where the matrix references
    /// are valid until the next mutable call.
    pub fn predict_and_loss(&mut self, student: &[f32], teacher: &[f32]) -> (f64, &[f64], &[f64]) {
        let k = self.config.dim;
        let lambda = self.config.lambda;

        // Build N + λI in scratch buffer
        self.pm[..k * k].copy_from_slice(&self.n);
        for i in 0..k {
            self.pm[i * k + i] += lambda;
        }

        // Invert N + λI
        invert_spd_into(
            &mut self.q_star,
            &mut self.inv_l_scratch,
            &mut self.inv_l_inv_scratch,
            &mut self.inv_matmul_bt_scratch,
            &self.pm[..k * k],
            k,
        );

        // P* = Σ @ Q*
        matmul_into(
            &mut self.p_star,
            &mut self.matmul_bt_scratch,
            &self.sigma,
            &self.q_star,
            k,
        );

        // Compute auxiliary loss using sigma_sample/n_sample scratch
        // (pm is currently N + λI, will be overwritten by aux loss)
        let loss = peira_aux_loss(
            student,
            teacher,
            &self.p_star,
            &self.q_star,
            lambda,
            &mut self.sigma_sample,
            &mut self.n_sample,
            &mut self.pm,
            &mut self.s_scratch,
            &mut self.t_scratch,
        );

        (loss, &self.p_star, &self.q_star)
    }

    /// Get a reference to the current Σ matrix (row-major).
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Get a reference to the current N matrix (row-major).
    pub fn n_matrix(&self) -> &[f64] {
        &self.n
    }

    /// Compute the PEIRA auxiliary loss using internal scratch buffers.
    ///
    /// Zero-alloc convenience method that delegates to [`peira_aux_loss`]
    /// with the pre-allocated `sigma_sample`, `n_sample`, and `pm` buffers.
    ///
    /// Note: `p_star` and `q_star` should come from a recent call to
    /// [`predictor_with_scratch()`] for consistency.
    pub fn compute_aux_loss(
        &mut self,
        student: &[f32],
        teacher: &[f32],
        p_star: &[f64],
        q_star: &[f64],
        lambda: f64,
    ) -> f64 {
        peira_aux_loss(
            student,
            teacher,
            p_star,
            q_star,
            lambda,
            &mut self.sigma_sample,
            &mut self.n_sample,
            &mut self.pm,
            &mut self.s_scratch,
            &mut self.t_scratch,
        )
    }

    /// Reset covariance estimates (e.g., at episode boundaries).
    pub fn reset(&mut self) {
        self.sigma.fill(0.0);
        self.n.fill(0.0);
        self.sigma_sample.fill(0.0);
        self.n_sample.fill(0.0);
        self.pm.fill(0.0);
        self.inv_l_scratch.fill(0.0);
        self.inv_l_inv_scratch.fill(0.0);
        self.inv_matmul_bt_scratch.fill(0.0);
        self.matmul_bt_scratch.fill(0.0);
        self.q_star.fill(0.0);
        self.p_star.fill(0.0);
        self.s_scratch.fill(0.0);
        self.t_scratch.fill(0.0);
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
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn peira_aux_loss(
    student: &[f32],
    teacher: &[f32],
    p_star: &[f64],
    q_star: &[f64],
    lambda: f64,
    sigma_sample: &mut [f64],
    n_sample: &mut [f64],
    pm: &mut [f64],
    s_scratch: &mut [f64],
    t_scratch: &mut [f64],
) -> f64 {
    let k = student.len();
    debug_assert_eq!(teacher.len(), k);
    debug_assert_eq!(p_star.len(), k * k);
    debug_assert_eq!(q_star.len(), k * k);
    debug_assert_eq!(sigma_sample.len(), k * k);
    debug_assert_eq!(n_sample.len(), k * k);
    debug_assert_eq!(pm.len(), k * k);
    debug_assert_eq!(s_scratch.len(), k);
    debug_assert_eq!(t_scratch.len(), k);

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
    simd_outer_product_f64(
        sigma_sample,
        n_sample,
        student,
        teacher,
        k,
        s_scratch,
        t_scratch,
    );

    // Term 1: -½ Tr(Σ_sample @ P*^T) = -½ Σ_{i,j} sigma_sample[i,j] * p_star[j,i]
    // In row-major: Tr(A @ B^T) = Σ_{i,j} A[i,j] * B[i,j] when both are row-major
    let term1 = -0.5 * simd_dot_f64(sigma_sample, p_star, k * k);

    // Term 2: ¼ Tr(P* @ (N_sample + λI) @ P*^T)
    // = ¼ Tr(P* @ M @ P*^T) where M = N_sample + λI
    // P* @ M is k×k, then (P* @ M) @ P*^T trace
    // Build M = N_sample + λI in-place (branch-free diagonal add)
    for i in 0..k {
        n_sample[i * k + i] += lambda;
    }
    // P*M = matmul(P*, M) — SIMD-optimized, sigma_sample reused as bt_scratch
    matmul_into(pm, sigma_sample, p_star, n_sample, k);
    // Restore n_sample by subtracting λ from diagonal
    for i in 0..k {
        n_sample[i * k + i] -= lambda;
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
#[allow(dead_code)]
fn invert_spd(mat: &[f64], k: usize) -> Vec<f64> {
    let mut inv = vec![0.0f64; k * k];
    let mut l = vec![0.0f64; k * k];
    let mut l_inv = vec![0.0f64; k * k];
    let mut bt = vec![0.0f64; k * k];
    invert_spd_into(&mut inv, &mut l, &mut l_inv, &mut bt, mat, k);
    inv
}

/// Zero-alloc variant of [`invert_spd`]: writes into caller-provided scratch buffers.
///
/// - `inv`: output k×k inverse matrix
/// - `l_scratch`: k×k scratch for Cholesky factor L
/// - `l_inv_scratch`: k×k scratch for L⁻¹
/// - `matmul_bt_scratch`: k×k scratch for transposed matrix in Step 3
/// - `mat`: input k×k SPD matrix
/// - `k`: matrix dimension
#[inline]
fn invert_spd_into(
    inv: &mut [f64],
    l_scratch: &mut [f64],
    l_inv_scratch: &mut [f64],
    matmul_bt_scratch: &mut [f64],
    mat: &[f64],
    k: usize,
) {
    // Step 1: Cholesky decomposition — L such that L * L^T = mat
    l_scratch.fill(0.0);
    for j in 0..k {
        let j_row = j * k;

        // Diagonal: sum of squares of L[j, 0..j]
        let sum = if j > 0 {
            simd_dot_f64(
                &l_scratch[j_row..j_row + j],
                &l_scratch[j_row..j_row + j],
                j,
            )
        } else {
            0.0
        };
        let diag = mat[j_row + j] - sum;
        assert!(
            diag > 0.0,
            "Matrix not positive definite in Cholesky decomposition"
        );
        let diag_sqrt = diag.sqrt();
        l_scratch[j_row + j] = diag_sqrt;

        // Off-diagonal (lower triangle only)
        for i in (j + 1)..k {
            let i_row = i * k;
            let sum = if j > 0 {
                simd_dot_f64(
                    &l_scratch[i_row..i_row + j],
                    &l_scratch[j_row..j_row + j],
                    j,
                )
            } else {
                0.0
            };
            l_scratch[i_row + j] = (mat[i_row + j] - sum) / diag_sqrt;
        }
    }

    // Step 2: Invert lower triangular L → L_inv
    l_inv_scratch.fill(0.0);
    for i in 0..k {
        let i_row = i * k;
        let inv_diag = 1.0 / l_scratch[i_row + i];
        l_inv_scratch[i_row + i] = inv_diag;
        for j in 0..i {
            // j < i always true in this loop; use SIMD for the inner dot product
            let sum = simd_dot_f64(
                &l_scratch[i_row + j..i_row + i],
                &l_inv_scratch[j * k + j..j * k + i],
                i - j,
            );
            l_inv_scratch[i_row + j] = -sum * inv_diag;
        }
    }

    // Step 3: M_inv = L_inv^T * L_inv
    // Transpose L_inv into l_scratch (Cholesky L no longer needed), then
    // use matmul_into which transposes B internally and uses simd_dot_f64.
    for i in 0..k {
        let i_row = i * k;
        for j in 0..k {
            l_scratch[j * k + i] = l_inv_scratch[i_row + j];
        }
    }
    // inv = l_scratch @ l_inv_scratch = L_inv^T @ L_inv
    matmul_into(inv, matmul_bt_scratch, l_scratch, l_inv_scratch, k);
}

/// Compute matrix product C = A @ B where all are k×k row-major.
/// Uses SIMD f64 dot product for the inner accumulation loop.
#[allow(dead_code)]
fn matmul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * k];
    let mut bt = vec![0.0f64; k * k];
    matmul_into(&mut c, &mut bt, a, b, k);
    c
}

/// Zero-alloc variant of [`matmul`]: writes into caller-provided buffers.
///
/// - `c`: output k×k result matrix
/// - `bt_scratch`: k×k scratch for transposed B
/// - `a`, `b`: input k×k matrices
/// - `k`: matrix dimension
fn matmul_into(c: &mut [f64], bt_scratch: &mut [f64], a: &[f64], b: &[f64], k: usize) {
    // Transpose B for sequential access in the inner loop
    for i in 0..k {
        let i_row = i * k;
        for j in 0..k {
            bt_scratch[j * k + i] = b[i_row + j];
        }
    }
    for i in 0..k {
        let i_row = i * k;
        let a_row = &a[i_row..i_row + k];
        for j in 0..k {
            let b_col = &bt_scratch[j * k..j * k + k];
            c[i_row + j] = simd_dot_f64(a_row, b_col, k);
        }
    }
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
        let expected = [0.6, -0.2, -0.2, 0.4];
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

    // Tests the public deprecated `predictor()` API surface — keep the
    // method exercised so regressions in the allocating path don't slip in.
    #[test]
    #[allow(deprecated)]
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

    // Uses the deprecated allocating `predictor()` for a one-shot test —
    // aux_loss path is already covered zero-alloc elsewhere.
    #[test]
    #[allow(deprecated)]
    fn aux_loss_is_finite() {
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k));
        cov.update(&[1.0, 0.5, -0.3, 0.0], &[0.8, 0.4, -0.2, 0.1]);
        let (p, q) = cov.predictor();
        let mut sigma_sample = vec![0.0f64; k * k];
        let mut n_sample = vec![0.0f64; k * k];
        let mut pm = vec![0.0f64; k * k];
        let mut s_scratch = vec![0.0f64; k];
        let mut t_scratch = vec![0.0f64; k];
        let loss = peira_aux_loss(
            &[1.0, 0.5, -0.3, 0.0],
            &[0.8, 0.4, -0.2, 0.1],
            &p,
            &q,
            0.1,
            &mut sigma_sample,
            &mut n_sample,
            &mut pm,
            &mut s_scratch,
            &mut t_scratch,
        );
        assert!(loss.is_finite(), "Loss is not finite: {loss}");
    }
}
