use super::*;

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol * (1.0 + a.abs() + b.abs())
}

#[test]
fn delay_ring_orders_newest_first() {
    let mut ring: DelayRing<2, 3> = DelayRing::new();
    ring.push(&[1.0, 1.0]);
    ring.push(&[2.0, 2.0]);
    ring.push(&[3.0, 3.0]);
    assert!(ring.is_full());
    let mut out = [0.0f32; 6];
    assert!(ring.flatten_into(&mut out));
    // newest first: [3,3, 2,2, 1,1]
    assert_eq!(out, [3.0, 3.0, 2.0, 2.0, 1.0, 1.0]);
}

#[test]
fn delay_ring_wraps_around() {
    let mut ring: DelayRing<1, 3> = DelayRing::new();
    for v in 0..5 {
        ring.push(&[v as f32]);
    }
    let mut out = [0.0f32; 3];
    assert!(ring.flatten_into(&mut out));
    // last three: 4, 3, 2
    assert_eq!(out, [4.0, 3.0, 2.0]);
}

#[test]
fn fourier_basis_is_bounded() {
    let basis: FourierBasis<8> = FourierBasis::new(1.0);
    let mut out = [0.0f32; 8];
    for &x in &[0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 3.7] {
        basis.eval_into(x, &mut out);
        for &v in &out {
            assert!(v.abs() <= 1.0 + 1e-5, "Fourier value out of [-1,1]: {v}");
        }
    }
}

#[test]
fn chebyshev_basis_recurrence() {
    let basis: ChebyshevBasis<4> = ChebyshevBasis::new();
    let mut out = [0.0f32; 4];
    basis.eval_into(0.5, &mut out);
    // T0=1, T1=0.5, T2=2*0.5*0.5 - 1 = -0.5, T3 = 2*0.5*(-0.5) - 0.5 = -1.0.
    assert!(approx_eq(out[0], 1.0, 1e-5));
    assert!(approx_eq(out[1], 0.5, 1e-5));
    assert!(approx_eq(out[2], -0.5, 1e-5));
    assert!(approx_eq(out[3], -1.0, 1e-5));
}

#[test]
fn bspline_partition_of_unity() {
    let basis: BSplineBasis<8> = BSplineBasis::new();
    let mut out = [0.0f32; 8];
    for x in (0..=100).map(|i| i as f32 / 100.0) {
        basis.eval_into(x, &mut out);
        let sum: f32 = out.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-3), "B-spline sum at x={x} = {sum}");
    }
}

#[test]
fn feature_expand_layout() {
    // 2 coords, M=2 basis each → 4 features.
    let basis: ChebyshevBasis<2> = ChebyshevBasis::new();
    let delay = [0.5f32, 0.0];
    let mut out = [0.0f32; 4];
    feature_expand::<ChebyshevBasis<2>, 2>(&delay, &basis, &mut out);
    // coord 0 (x=0.5): T0=1, T1=0.5
    assert!(approx_eq(out[0], 1.0, 1e-5));
    assert!(approx_eq(out[1], 0.5, 1e-5));
    // coord 1 (x=0.0): T0=1, T1=0
    assert!(approx_eq(out[2], 1.0, 1e-5));
    assert!(approx_eq(out[3], 0.0, 1e-5));
}

#[test]
fn forecaster_fits_and_forecasts_linear_map() {
    // Build a forecaster where the true map is û = 2·x_coord_0 (linear).
    // Use Chebyshev with M=2 so T0=1 (bias) and T1=x are present.
    type F = KarcForecaster<ChebyshevBasis<2>, 1, 2, 1>;
    let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 20);
    // 20 samples: u_t = x, u_{t+1} = 2x.
    for i in 0..20 {
        let x = (i as f32) * 0.1 - 1.0; // x in [-1, 0.9]
        let delay = [x];
        let target = [2.0 * x];
        f.accumulate_pair(&delay, &target);
    }
    f.fit_ridge(1e-6).unwrap();
    assert!(f.is_fitted());
    // Forecast at x=0.5 → expect ≈ 1.0.
    let mut out = [0.0f32];
    assert!(f.forecast_into(&[0.5], &mut out));
    assert!(
        approx_eq(out[0], 1.0, 1e-2),
        "forecast at x=0.5: {} (expected ~1.0)",
        out[0]
    );
}

#[test]
fn forecaster_rejects_zero_lambda() {
    type F = KarcForecaster<ChebyshevBasis<2>, 1, 2, 1>;
    let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 4);
    f.accumulate_pair(&[0.0], &[0.0]);
    let err = f.fit_ridge(0.0).unwrap_err();
    assert_eq!(err, FitError::NonPositiveLambda);
}

#[test]
fn forecaster_rejects_no_samples() {
    type F = KarcForecaster<ChebyshevBasis<2>, 1, 2, 1>;
    let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 4);
    let err = f.fit_ridge(1e-6).unwrap_err();
    assert_eq!(err, FitError::NoSamples);
}

#[test]
fn fit_error_singular_display_pin() {
    // Issue 904: the Woodbury path degrades to FitError::Singular on an
    // indefinite f32 sample Gram (previously a cholesky_f32 panic through the
    // real tick path). This Display feeds the engine's
    // "fit_ridge failed: {e}; keeping previous fit" warn line.
    assert_eq!(
        FitError::Singular.to_string(),
        "f32 sample Gram not positive definite"
    );
}

#[test]
fn forecaster_fit_woodbury_path_forecasts_linear_map() {
    // Force the Woodbury sample-space path: d_h (D*M*K = 2*3*1 = 6) > n (4).
    // Verifies the reused scratch buffers (sample_gram/sample_chol/sample_z/
    // w_t_transpose) produce correct forecasts after the zero-alloc refactor.
    type F = KarcForecaster<ChebyshevBasis<3>, 2, 3, 1>;
    let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 8);
    // 4 samples (n=4 < d_h=6 → Woodbury path).
    for i in 0..4 {
        let x = (i as f32) * 0.3 - 0.5;
        let delay = [x, x + 0.1];
        let target = [2.0 * x, 3.0 * x];
        f.accumulate_pair(&delay, &target);
    }
    f.fit_ridge(1e-3).unwrap();
    assert!(f.is_fitted());
    let mut out = [0.0f32; 2];
    assert!(f.forecast_into(&[0.2, 0.3], &mut out));
    // The fit should approximate the linear map target ≈ [2*0.2, 3*0.2] = [0.4, 0.6].
    assert!(
        approx_eq(out[0], 0.4, 0.15),
        "woodbury forecast[0] at x=0.2: {} (expected ~0.4)",
        out[0]
    );
    assert!(
        approx_eq(out[1], 0.6, 0.15),
        "woodbury forecast[1] at x=0.2: {} (expected ~0.6)",
        out[1]
    );
}

// ── Phase 2 tests (Plan 308 T2.1–T2.5) ──

#[test]
fn higher_order_feature_count_formula() {
    // d_h_1 = 4, R=1 → 4.
    assert_eq!(higher_order_feature_count(4, 1), 4);
    // d_h_1 = 4, R=2 → 4 + 4*5/2 = 14.
    assert_eq!(higher_order_feature_count(4, 2), 14);
    // d_h_1 = 96, R=2 → 96 + 96*97/2 = 96 + 4656 = 4752.
    assert_eq!(higher_order_feature_count(96, 2), 4752);
    // Plan config D=3, M=24, K=8 → d_h_1=576, R=2 → 576 + 576*577/2 = 576 + 166176 = 166752.
    assert_eq!(higher_order_feature_count(576, 2), 166752);
}

#[test]
fn higher_order_r1_matches_feature_expand() {
    // R=1 must produce identical output to feature_expand.
    let basis: ChebyshevBasis<4> = ChebyshevBasis::new();
    let delay = [0.5f32, -0.3, 0.8];
    let mut out_first = [0.0f32; 12];
    let mut out_higher = [0.0f32; 12];
    feature_expand::<ChebyshevBasis<4>, 4>(&delay, &basis, &mut out_first);
    feature_expand_higher_order::<ChebyshevBasis<4>, 4, 1>(&delay, &basis, &mut out_higher);
    for i in 0..12 {
        assert_eq!(
            out_first[i].to_bits(),
            out_higher[i].to_bits(),
            "R=1 mismatch at idx {i}"
        );
    }
}

#[test]
fn higher_order_r2_count_and_symmetry() {
    // 2 coords, M=2 → d_h_1 = 4. R=2 → 4 + 4*5/2 = 14 features.
    let basis: ChebyshevBasis<2> = ChebyshevBasis::new();
    let delay = [0.5f32, 0.0];
    let d_h = higher_order_feature_count(4, 2);
    assert_eq!(d_h, 14);
    let mut out = vec![0.0f32; d_h];
    feature_expand_higher_order::<ChebyshevBasis<2>, 2, 2>(&delay, &basis, &mut out);
    // First 4 features = first-order.
    // Chebyshev at x=0.5: T0=1, T1=0.5. At x=0: T0=1, T1=0.
    assert!(approx_eq(out[0], 1.0, 1e-5)); // T0(0.5)
    assert!(approx_eq(out[1], 0.5, 1e-5)); // T1(0.5)
    assert!(approx_eq(out[2], 1.0, 1e-5)); // T0(0)
    assert!(approx_eq(out[3], 0.0, 1e-5)); // T1(0)
    // Pair products: 4*5/2 = 10 pairs.
    // Pairs in order: (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3).
    let psi = [&out[0], &out[1], &out[2], &out[3]];
    let mut idx = 4;
    for f1 in 0..4 {
        for f2 in f1..4 {
            let expected = psi[f1] * psi[f2];
            assert!(
                approx_eq(out[idx], expected, 1e-5),
                "pair ({},{}) at idx {}: got {}, expected {}",
                f1,
                f2,
                idx,
                out[idx],
                expected
            );
            idx += 1;
        }
    }
    assert_eq!(idx, d_h);
}

#[test]
fn chunked_gram_matches_direct() {
    // Build a small synthetic feature set, compare chunked_gram_into against
    // a hand-computed XᵀX + λI.
    let d_h = 3;
    let n = 4;
    let features: Vec<f32> = vec![1.0, 2.0, 3.0, 0.5, 1.0, 1.5, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let lambda = 0.1f64;
    // Direct XᵀX.
    let mut direct_gram = vec![0.0f64; d_h * d_h];
    for r in 0..n {
        let row = &features[r * d_h..(r + 1) * d_h];
        for i in 0..d_h {
            for j in 0..d_h {
                direct_gram[i * d_h + j] += row[i] as f64 * row[j] as f64;
            }
        }
    }
    for i in 0..d_h {
        direct_gram[i * d_h + i] += lambda;
    }
    // Chunked.
    let mut chunked = vec![0.0f64; d_h * d_h];
    let iter = (0..n).map(|r| &features[r * d_h..(r + 1) * d_h] as &[f32]);
    chunked_gram_into(iter, &mut chunked, lambda, d_h);
    for i in 0..d_h * d_h {
        assert!(
            (direct_gram[i] - chunked[i]).abs() < 1e-10,
            "gram mismatch at {}: direct={}, chunked={}",
            i,
            direct_gram[i],
            chunked[i]
        );
    }
}

#[test]
fn forecast_low_rank_matches_full_rank_matvec() {
    // Construct A (D×r) and B (r×d_h) from a known Wout = A·B, then verify
    // the two-stage matvec A·(B·ψ) matches the direct Wout·ψ.
    let d_h = 4usize;
    let r = 2usize;
    let d_out = 2usize;
    // A = [[1, 0], [0, 2]], B = [[1, 0, 1, 0], [0, 1, 0, 1]]
    // Wout = A·B = [[1,0,1,0], [0,2,0,2]]
    let a: Vec<f32> = vec![1.0, 0.0, 0.0, 2.0];
    let b: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let mut wout = vec![0.0f32; d_out * d_h];
    for d in 0..d_out {
        for j in 0..d_h {
            for k in 0..r {
                wout[d * d_h + j] += a[d * r + k] * b[k * d_h + j];
            }
        }
    }
    let psi: Vec<f32> = vec![0.5, -1.0, 0.3, 0.7];
    // Direct: out_direct[d] = Σ_j Wout[d,j] * psi[j].
    let mut out_direct = [0.0f32; 2];
    for d in 0..d_out {
        for j in 0..d_h {
            out_direct[d] += wout[d * d_h + j] * psi[j];
        }
    }
    // Two-stage.
    let mut out_lr = [0.0f32; 2];
    let mut mid = [0.0f32; 2];
    forecast_low_rank_apply(&a, &b, &psi, &mut mid, &mut out_lr, d_h, r, d_out);
    for d in 0..d_out {
        assert!(
            approx_eq(out_direct[d], out_lr[d], 1e-4),
            "matvec mismatch at d={}: direct={}, low_rank={}",
            d,
            out_direct[d],
            out_lr[d]
        );
    }
}

#[test]
fn low_rank_fit_r_equals_d_recovers_forecast_quality() {
    // Fit full-rank Wout via fit_ridge, then fit low-rank A·B via low_rank_fit
    // with r=D. The low-rank A·B should approximate the full-rank Wout
    // (within a tolerance that accounts for the ALS gauge freedom and the
    // float precision of the Kronecker B-step).
    type F = KarcForecaster<ChebyshevBasis<3>, 2, 3, 2>;
    let mut f: F = KarcForecaster::with_capacity(ChebyshevBasis::new(), 100);
    // Build a rich nonlinear 2D signal.
    for i in 0..80 {
        let t = i as f32 * 0.07;
        let x0 = (0.9 * t).sin();
        let x1 = (1.4 * t).cos() + 0.3 * (2.1 * t).sin();
        let prev_t = (i - 1) as f32 * 0.07;
        let prev_x0 = (0.9 * prev_t).sin();
        let prev_x1 = (1.4 * prev_t).cos() + 0.3 * (2.1 * prev_t).sin();
        let delay = [x0, x1, prev_x0, prev_x1];
        let next_t = (i + 1) as f32 * 0.07;
        let target = [
            (0.9 * next_t).sin(),
            (1.4 * next_t).cos() + 0.3 * (2.1 * next_t).sin(),
        ];
        f.accumulate_pair(&delay, &target);
    }
    let lambda = 1e-4f32;
    f.fit_ridge(lambda).expect("fit_ridge");
    let iters = f.fit_low_rank(2, lambda, 100, 1e-10).expect("fit_low_rank");
    assert!(f.is_low_rank_fitted());
    assert!(iters > 0, "ALS should run at least 1 iteration");
    // Compare A·B vs full-rank Wout directly.
    let r = 2usize;
    let d_h = F::D_H;
    let mut ab = vec![0.0f32; 2 * d_h];
    for d in 0..2 {
        for j in 0..d_h {
            for k in 0..r {
                ab[d * d_h + j] += f.a_low_rank[d * r + k] * f.b_low_rank[k * d_h + j];
            }
        }
    }
    // Max absolute weight difference.
    let max_w = f.wout.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let max_diff = f
        .wout
        .iter()
        .zip(ab.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel = max_diff / max_w.max(1e-6);
    // With r=D=2, the factorization can represent Wout, but the ALS gauge
    // freedom + float precision leaves some residual. We check <15% relative.
    assert!(
        rel < 0.15,
        "low-rank A·B diverges from Wout: max_diff={max_diff}, max_w={max_w}, rel={rel:.4}"
    );
    // Also verify forecasts match within a looser tolerance (Chebyshev
    // expansion amplifies weight differences).
    let mut max_rel_err = 0.0f32;
    for probe_t in 0..20 {
        let t = (probe_t as f32 + 0.5) * 0.3;
        let delay = [
            (0.7 * t).sin(),
            (1.1 * t).cos(),
            (0.7 * (t - 0.07)).sin(),
            (1.1 * (t - 0.07)).cos(),
        ];
        let mut out_full = [0.0f32; 2];
        let mut out_lr = [0.0f32; 2];
        let delay_copy = delay;
        assert!(f.forecast_into(&delay, &mut out_full));
        assert!(f.forecast_low_rank_into(&delay_copy, &mut out_lr));
        for d in 0..2 {
            let denom = out_full[d].abs().max(0.5);
            let rel = (out_full[d] - out_lr[d]).abs() / denom;
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    assert!(
        max_rel_err < 0.15,
        "low-rank forecast diverges from full-rank: max_rel_err={max_rel_err:.4}"
    );
}

#[test]
fn low_rank_fit_is_deterministic() {
    // Two ALS runs on identical Gram/Cov must produce bit-identical A, B.
    let d_h = 6usize;
    let d_out = 2usize;
    let r = 2usize;
    // Synthetic Gram (SPD, well-conditioned).
    let mut gram = vec![0.0f64; d_h * d_h];
    for i in 0..d_h {
        for j in 0..d_h {
            gram[i * d_h + j] = if i == j { 3.0 } else { 0.3 };
        }
    }
    let mut cov = vec![0.0f64; d_h * d_out];
    for i in 0..d_h {
        for d in 0..d_out {
            cov[i * d_out + d] = (i as f64 + 0.1) * ((d as f64) + 0.5);
        }
    }
    let lambda = 1e-3f64;
    let mut a1 = vec![0.0f64; d_out * r];
    let mut b1 = vec![0.0f64; r * d_h];
    let mut a2 = vec![0.0f64; d_out * r];
    let mut b2 = vec![0.0f64; r * d_h];
    let mut scr1 = LowRankFitScratch::with_capacity(d_h, d_out, r);
    let mut scr2 = LowRankFitScratch::with_capacity(d_h, d_out, r);
    let n1 = low_rank_fit(
        &gram, &cov, d_h, d_out, r, lambda, 30, 1e-10, &mut a1, &mut b1, &mut scr1,
    );
    let n2 = low_rank_fit(
        &gram, &cov, d_h, d_out, r, lambda, 30, 1e-10, &mut a2, &mut b2, &mut scr2,
    );
    assert_eq!(n1, n2, "iteration count must match");
    for i in 0..d_out * r {
        assert_eq!(a1[i].to_bits(), a2[i].to_bits(), "A bit mismatch at {i}");
    }
    for i in 0..r * d_h {
        assert_eq!(b1[i].to_bits(), b2[i].to_bits(), "B bit mismatch at {i}");
    }
}

#[test]
fn frozen_a_fit_b_step_is_valid_ridge_solution() {
    // Verify the frozen-A B-step produces a valid ridge solution: the
    // returned B minimizes ‖Y - A·B·Xᵀ‖² + λ‖B‖² for the frozen A.
    //
    // We check this by verifying the gradient is approximately zero at
    // the solution: d/dB [‖Y - A·B·Xᵀ‖² + λ‖B‖²] = -2·Aᵀ·(Y·X - A·B·Xᵀ·X) + 2λB
    // should be ≈0. Equivalently: Aᵀ·Covᵀ = (AᵀA)·B·G + λB (the normal eq).
    let d_h = 6usize;
    let d_out = 2usize;
    let r = 2usize;
    let mut gram = vec![0.0f64; d_h * d_h];
    for i in 0..d_h {
        for j in 0..d_h {
            gram[i * d_h + j] = if i == j { 2.0 + (i as f64) * 0.1 } else { 0.3 };
        }
    }
    let mut cov = vec![0.0f64; d_h * d_out];
    for i in 0..d_h {
        for d in 0..d_out {
            cov[i * d_out + d] = (i as f64 + 0.1) * ((d as f64) + 0.5);
        }
    }
    let lambda = 1e-3f64;
    // Arbitrary frozen A (not from ALS — just a valid D×r matrix).
    let a_frozen: Vec<f64> = (0..d_out * r).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let mut b_out = vec![0.0f64; r * d_h];
    let mut scr = LowRankFitScratch::with_capacity(d_h, d_out, r);
    low_rank_fit_b_with_frozen_a(
        &gram, &cov, d_h, d_out, r, lambda, &a_frozen, &mut b_out, &mut scr,
    );
    // Verify the normal equation: (AᵀA)·B·G + λB == Aᵀ·Covᵀ.
    // Compute AᵀA (r×r).
    let mut ata = vec![0.0f64; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0;
            for d in 0..d_out {
                s += a_frozen[d * r + i] * a_frozen[d * r + j];
            }
            ata[i * r + j] = s;
        }
    }
    // Compute LHS = (AᵀA)·B·G + λB (r × d_h).
    let mut lhs = vec![0.0f64; r * d_h];
    for i in 0..r {
        for j in 0..d_h {
            let mut s = 0.0;
            for k in 0..r {
                for l in 0..d_h {
                    s += ata[i * r + k] * b_out[k * d_h + l] * gram[l * d_h + j];
                }
            }
            lhs[i * d_h + j] = s + lambda * b_out[i * d_h + j];
        }
    }
    // Compute RHS = Aᵀ·Covᵀ (r × d_h). (Covᵀ = Cov transposed: d_h × D → D × d_h)
    // Cov is d_h × d_out, so Covᵀ is d_out × d_h. Aᵀ is r × d_out.
    // Aᵀ·Covᵀ = r × d_h. (Aᵀ·Covᵀ)[i,j] = Σ_d A[d,i]·Cov[j,d].
    let mut rhs = vec![0.0f64; r * d_h];
    for i in 0..r {
        for j in 0..d_h {
            let mut s = 0.0;
            for d in 0..d_out {
                s += a_frozen[d * r + i] * cov[j * d_out + d];
            }
            rhs[i * d_h + j] = s;
        }
    }
    // LHS should equal RHS (the normal equation is satisfied).
    let max_resid = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_resid < 1e-9,
        "frozen-A B does not satisfy normal equation: max residual = {max_resid:e}"
    );
}

#[test]
fn frozen_a_fit_forecaster_method_works() {
    // End-to-end: accumulate trajectory, ALS-fit A, then re-fit B via
    // fit_low_rank_with_frozen_a. The forecast should be close to the
    // ALS forecast (same A, B solved from same data).
    let mut f = KarcForecaster::<ChebyshevBasis<4>, 2, 4, 2>::with_capacity(
        ChebyshevBasis::<4>::new(),
        256,
    );
    // Synthetic trajectory identical to low_rank_fit_r_equals_d.
    for i in 0..200i32 {
        let t = i as f32 * 0.07f32;
        let x0 = (0.9f32 * t).sin();
        let x1 = (1.4f32 * t).cos() + 0.3f32 * (2.1f32 * t).sin();
        let prev_t = (i.saturating_sub(1)) as f32 * 0.07f32;
        let prev_x0 = (0.9f32 * prev_t).sin();
        let prev_x1 = (1.4f32 * prev_t).cos() + 0.3f32 * (2.1f32 * prev_t).sin();
        let delay = [x0, x1, prev_x0, prev_x1];
        let next_t = (i + 1) as f32 * 0.07f32;
        let target = [
            (0.9f32 * next_t).sin(),
            (1.4f32 * next_t).cos() + 0.3f32 * (2.1f32 * next_t).sin(),
        ];
        f.accumulate_pair(&delay, &target);
    }
    // ALS reference fit (r = D = 2) — used only to extract a
    // plausible A for the frozen-A test.
    let lambda = 1e-4f32;
    f.fit_low_rank(2, lambda, 50, 1e-10).unwrap();
    let a_ref: Vec<f32> = f.a_low_rank.clone();
    // Re-fit B with A frozen at the ALS A.
    f.fit_low_rank_with_frozen_a(&a_ref, 2, lambda).unwrap();
    // The frozen-A fit must produce a usable low-rank forecaster.
    assert!(f.is_low_rank_fitted());
    assert_eq!(f.low_rank_r(), 2);
    // A must be preserved verbatim (frozen means frozen).
    for (i, a) in a_ref.iter().enumerate() {
        assert_eq!(
            a.to_bits(),
            f.a_low_rank[i].to_bits(),
            "frozen A modified at idx {i}"
        );
    }
    // B must be non-trivial (not all zeros — the fit found a solution).
    let b_norm: f32 = f.b_low_rank.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(b_norm > 1e-6, "frozen-A B is all zeros: b_norm={b_norm:e}");
    // Forecast must produce finite values at several probe points.
    let mut max_abs = 0.0f32;
    for probe_t in 0..10i32 {
        let t = (probe_t as f32 + 0.5) * 0.3f32;
        let delay = [
            (0.7f32 * t).sin(),
            (1.1f32 * t).cos(),
            (0.7f32 * (t - 0.07f32)).sin(),
            (1.1f32 * (t - 0.07f32)).cos(),
        ];
        let mut out = [0.0f32; 2];
        assert!(f.forecast_low_rank_into(&delay, &mut out));
        for o in out.iter() {
            assert!(o.is_finite(), "non-finite forecast at probe {probe_t}");
            max_abs = max_abs.max(o.abs());
        }
    }
    assert!(max_abs > 0.0, "all forecasts are zero");
}

#[test]
fn warm_start_with_game_a_factors_converges_to_valid_solution() {
    // Smoke test for `low_rank_fit_warm_start`: given Game A's fitted
    // (A, B) as init, ALS must converge and produce finite factors that
    // forecast to finite values. Also verifies that warm-start from a
    // *valid* Game-A solution on the SAME data reproduces the from-scratch
    // optimum (the ALS fixed point is the same regardless of init).
    //
    // This does NOT test cross-game transfer (that's the bench's job) —
    // it just verifies the API is wired correctly and produces valid math.
    let d_h = 6usize;
    let d_out = 2usize;
    let r = 2usize;
    let mut gram = vec![0.0f64; d_h * d_h];
    for i in 0..d_h {
        for j in 0..d_h {
            gram[i * d_h + j] = if i == j { 2.0 + (i as f64) * 0.1 } else { 0.3 };
        }
    }
    let mut cov = vec![0.0f64; d_h * d_out];
    for i in 0..d_h {
        for d in 0..d_out {
            cov[i * d_out + d] = (i as f64 + 0.1) * ((d as f64) + 0.5);
        }
    }
    let lambda = 1e-3f64;

    // Step 1: fit from scratch to get a reference (A*, B*).
    let mut a_ref = vec![0.0f64; d_out * r];
    let mut b_ref = vec![0.0f64; r * d_h];
    let mut scr_ref = LowRankFitScratch::with_capacity(d_h, d_out, r);
    let _n_ref = low_rank_fit(
        &gram,
        &cov,
        d_h,
        d_out,
        r,
        lambda,
        50,
        1e-12,
        &mut a_ref,
        &mut b_ref,
        &mut scr_ref,
    );

    // Step 2: warm-start ALS from the reference solution. With max_iters=0,
    // the factors must be returned UNCHANGED (no ALS steps taken).
    let mut a_ws = vec![0.0f64; d_out * r];
    let mut b_ws = vec![0.0f64; r * d_h];
    let mut scr_ws = LowRankFitScratch::with_capacity(d_h, d_out, r);
    let n_ws = low_rank_fit_warm_start(
        &gram,
        &cov,
        d_h,
        d_out,
        r,
        lambda,
        0,
        1e-12,
        &a_ref,
        &b_ref,
        &mut a_ws,
        &mut b_ws,
        &mut scr_ws,
    );
    assert_eq!(n_ws, 0, "max_iters=0 should run zero ALS iterations");
    for i in 0..d_out * r {
        assert_eq!(
            a_ws[i].to_bits(),
            a_ref[i].to_bits(),
            "max_iters=0 warm-start must copy A unchanged at {i}"
        );
    }
    for i in 0..r * d_h {
        assert_eq!(
            b_ws[i].to_bits(),
            b_ref[i].to_bits(),
            "max_iters=0 warm-start must copy B unchanged at {i}"
        );
    }

    // Step 3: warm-start from a PERTURBED init (A_ref + noise). After
    // enough ALS iterations, it must converge back to the same fixed
    // point (A*, B*) — the ALS objective is convex in each factor
    // individually, so the fixed point is unique up to scale balancing.
    let mut a_perturbed = a_ref.clone();
    for v in a_perturbed.iter_mut() {
        *v += 0.05; // small perturbation
    }
    let mut b_perturbed = b_ref.clone();
    for v in b_perturbed.iter_mut() {
        *v += 0.05;
    }
    let mut a_ws2 = vec![0.0f64; d_out * r];
    let mut b_ws2 = vec![0.0f64; r * d_h];
    let mut scr_ws2 = LowRankFitScratch::with_capacity(d_h, d_out, r);
    let n_ws2 = low_rank_fit_warm_start(
        &gram,
        &cov,
        d_h,
        d_out,
        r,
        lambda,
        100,
        1e-12,
        &a_perturbed,
        &b_perturbed,
        &mut a_ws2,
        &mut b_ws2,
        &mut scr_ws2,
    );
    assert!(
        n_ws2 > 0,
        "perturbed warm-start should run ≥1 iteration to converge"
    );

    // The converged Wout = A·B must match the reference Wout = A*·B*
    // (up to numerical tolerance — scale balancing may differ but the
    // product is the fixed point).
    let mut wout_ref = vec![0.0f64; d_out * d_h];
    let mut wout_ws = vec![0.0f64; d_out * d_h];
    for d in 0..d_out {
        for j in 0..d_h {
            let mut s_ref = 0.0;
            let mut s_ws = 0.0;
            for k in 0..r {
                s_ref += a_ref[d * r + k] * b_ref[k * d_h + j];
                s_ws += a_ws2[d * r + k] * b_ws2[k * d_h + j];
            }
            wout_ref[d * d_h + j] = s_ref;
            wout_ws[d * d_h + j] = s_ws;
        }
    }
    let mut max_diff = 0.0f64;
    for i in 0..d_out * d_h {
        max_diff = max_diff.max((wout_ref[i] - wout_ws[i]).abs());
    }
    assert!(
        max_diff < 1e-3,
        "warm-start from perturbation must converge to same Wout fixed point (max_diff={max_diff:e})"
    );
}
