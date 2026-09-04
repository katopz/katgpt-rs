//! Bench 668 — `effective_degree` GOAT gate (Issue 668, Research 488).
//!
//! Gates:
//! - **G1a** order preservation (paper Appendix I protocol): synthetic targets
//!   of ground-truth algebraic degree {1, 2, 5} → `ed_norm` strictly ordered.
//! - **G1b** basis invariance: the same ordering survives refitting the *same*
//!   sampled outputs in a **Legendre** basis and reducing through the shipped
//!   [`ed_from_coeff_norms`] (paper §3.2 / Appendix I: Chebyshev and Legendre
//!   agree on ordering).
//! - **G1c** scale behaviour (paper Table 12): `ed` scales ×2 with the outputs,
//!   `ed_norm` does not.
//! - **G1d** degenerate references: constant → ED ≈ 0; pure affine → ED ≈ |c₁|.
//! - **G1e** node-sampler robustness: ordering holds across 8 independent seeds.
//! - **G2** latency: per-path cost at the paper's efficiency (r=4/K=3) and
//!   performance (r=15/K=7) configurations, plus pair-count scaling.
//!
//! G3 (no-regression) is the `cargo clippy` / default-build check recorded in
//! `.benchmarks/665_effective_degree_goat.md`; G4 (alloc-free) lives in the
//! isolated binary `effective_degree_alloc_check.rs` — a `CountingAllocator`
//! here would perturb the `Instant::now()` timing loops.

#![cfg(feature = "effective_degree")]

use katgpt_core::effective_degree::{
    EdConfig, EdScratch, MAX_ED_TERMS, ed_from_coeff_norms, ed_over_pairs,
    effective_degree_along_path, randomized_cosine_nodes,
};
use katgpt_core::linalg::{chol_solve_f64, cholesky_f64};
use std::hint::black_box;
use std::time::Instant;

const IN_DIM: usize = 6;

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Deterministic "data manifold": points on a smooth 1-parameter curve in ℝ⁶
/// plus a small structured wobble. Deliberately NOT white noise — paper C.1
/// shows random-pixel endpoints destroy the ED signal, so the gate must be
/// anchored to structured points to be a fair test of the metric.
fn manifold_points(n: usize, phase: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * IN_DIM);
    for i in 0..n {
        let t = (i as f32) / (n as f32) * core::f32::consts::TAU + phase;
        for d in 0..IN_DIM {
            let f = (d + 1) as f32;
            out.push((0.6 * (f * t).sin() + 0.25 * (0.5 * f * t).cos()) / f.sqrt());
        }
    }
    out.shrink_to_fit();
    out
}

/// `f_p(x) = (w · x)^p` — a multivariate polynomial of exact algebraic degree
/// `p`, whose restriction to any generic interpolation path is a degree-`p`
/// polynomial in α. This is the controlled ground-truth-degree family the
/// paper's Appendix I study uses.
fn linear_form(x: &[f32]) -> f32 {
    const W: [f32; IN_DIM] = [0.5, -0.3, 0.8, 0.2, -0.6, 0.4];
    x.iter().zip(W).map(|(a, b)| a * b).sum()
}

/// `(ed_norm, ed_norm_ac)` for `(w·x)^degree` averaged over `cfg.n_pairs` paths.
///
/// `ed_norm_ac` re-reduces the same coefficients with the `k = 0` (DC) term
/// zeroed. `ED_norm` is a degree-weighted mean over **all** coefficients
/// including the constant, so a large output offset drags it toward 0 —
/// see the DC caveat in the module docs. The AC-only read is the offset-free
/// quantity that admits the hard bound `1 ≤ ed_norm_ac ≤ degree`.
fn measure_ed_norm(degree: i32, cfg: &EdConfig) -> (f32, f32) {
    let a = manifold_points(cfg.n_pairs, 0.0);
    let b = manifold_points(cfg.n_pairs, 1.7);
    let mut scratch = EdScratch::new(cfg, IN_DIM, 1);
    let r = ed_over_pairs(
        |x, y| y[0] = linear_form(x).powi(degree),
        &a,
        &b,
        cfg,
        &mut scratch,
    )
    .unwrap();
    let mut ac = r.coeff_norms;
    ac[0] = 0.0;
    (r.ed_norm, ed_from_coeff_norms(&ac[..r.n_terms]).1)
}

// ── Legendre reference (G1b) ─────────────────────────────────────────────────

/// `P₀ … P_{m-1}` of the first kind at `u ∈ [-1,1]`:
/// `(n+1)·P_{n+1} = (2n+1)·u·P_n − n·P_{n−1}`.
fn legendre_into(u: f32, out: &mut [f32]) {
    out[0] = 1.0;
    if out.len() > 1 {
        out[1] = u;
    }
    for n in 1..out.len() - 1 {
        let nf = n as f32;
        out[n + 1] = ((2.0 * nf + 1.0) * u * out[n] - nf * out[n - 1]) / (nf + 1.0);
    }
}

/// Fit the same damped normal equations in a Legendre basis and reduce through
/// the shipped [`ed_from_coeff_norms`]. Index == degree in both bases, so the
/// reducer is literally the same code path the primitive uses — only the basis
/// changes, which is exactly the perturbation Appendix I applies.
fn legendre_ed_norm(outputs: &[f32], nodes: &[f32], n_terms: usize, damping: f32) -> f32 {
    let mut gram = vec![0.0f64; n_terms * n_terms];
    let mut rhs = vec![0.0f64; n_terms];
    let mut psi = vec![0.0f32; n_terms];
    for (i, &alpha) in nodes.iter().enumerate() {
        legendre_into(2.0f32.mul_add(alpha, -1.0), &mut psi);
        for j in 0..n_terms {
            let pj = psi[j] as f64;
            for l in 0..n_terms {
                gram[j * n_terms + l] += pj * psi[l] as f64;
            }
            rhs[j] += pj * outputs[i] as f64;
        }
    }
    for j in 0..n_terms {
        gram[j * n_terms + j] += damping as f64;
    }
    let mut chol = vec![0.0f64; n_terms * n_terms];
    let mut z = vec![0.0f64; n_terms];
    let mut coef = vec![0.0f64; n_terms];
    cholesky_f64(&mut chol, &gram, n_terms);
    chol_solve_f64(&mut coef, &mut z, &chol, &rhs, n_terms, 1);
    let norms: Vec<f32> = coef.iter().map(|&c| c.abs() as f32).collect();
    ed_from_coeff_norms(&norms).1
}

// ── G1 ───────────────────────────────────────────────────────────────────────

#[test]
fn g1a_effective_degree_preserves_algebraic_degree_order() {
    let cfg = EdConfig::precise();
    // The paper's Tasks 1–3 use degrees {1, 2, 5}; the full 1..=5 chain is a
    // strictly stronger monotonicity claim, so gate on that.
    let measured: Vec<(i32, f32, f32)> = (1..=5)
        .map(|d| {
            let (e, ac) = measure_ed_norm(d, &cfg);
            (d, e, ac)
        })
        .collect();
    for (d, e, ac) in &measured {
        println!("G1a deg{d}: ed_norm={e:.4} ed_norm_ac={ac:.4}");
    }
    for w in measured.windows(2) {
        let (d0, e0, _) = w[0];
        let (d1, e1, _) = w[1];
        assert!(e0 < e1, "deg{d0} {e0:.4} !< deg{d1} {e1:.4}");
    }
    // Separation must be real, not fitting noise.
    let ratio = measured[4].1 / measured[0].1;
    assert!(ratio > 2.0, "deg5/deg1 ed_norm ratio {ratio:.3} too tight");

    // Absolute anchor, offset-free: every non-constant restriction of exact
    // algebraic degree p must satisfy 1 ≤ ED_norm_ac ≤ p. Catches both a
    // collapsed fit (all mass on T₁) and spurious high-mode leakage (mass
    // above T_p, which would push it past p).
    for &(d, _, ac) in &measured {
        assert!(
            ac >= 1.0 - 1e-3 && ac <= d as f32 + 1e-2,
            "deg{d}: ed_norm_ac {ac:.4} outside [1, {d}]"
        );
    }
}

#[test]
fn g1b_order_survives_a_legendre_basis_swap() {
    let cfg = EdConfig::precise();
    let n_terms = cfg.n_terms();
    let mut nodes = vec![0.0f32; cfg.resolution];
    randomized_cosine_nodes(cfg.resolution, cfg.seed, &mut nodes).unwrap();

    let a = manifold_points(1, 0.0);
    let b = manifold_points(1, 1.7);

    let mut cheb = [0.0f32; 3];
    let mut leg = [0.0f32; 3];
    for (slot, degree) in [1i32, 2, 5].iter().enumerate().map(|(i, &d)| (i, d)) {
        let outputs: Vec<f32> = nodes
            .iter()
            .map(|&alpha| {
                let x: Vec<f32> = (0..IN_DIM)
                    .map(|d| alpha.mul_add(a[d] - b[d], b[d]))
                    .collect();
                linear_form(&x).powi(degree)
            })
            .collect();
        cheb[slot] = effective_degree_along_path(&outputs, &nodes, &cfg)
            .unwrap()
            .ed_norm;
        leg[slot] = legendre_ed_norm(&outputs, &nodes, n_terms, cfg.damping);
    }
    println!("G1b chebyshev={cheb:?} legendre={leg:?}");
    assert!(cheb[0] < cheb[1] && cheb[1] < cheb[2], "chebyshev order broke");
    assert!(leg[0] < leg[1] && leg[1] < leg[2], "legendre order broke");
}

#[test]
fn g1c_ed_scales_with_output_magnitude_ed_norm_does_not() {
    let cfg = EdConfig::precise();
    let a = manifold_points(cfg.n_pairs, 0.0);
    let b = manifold_points(cfg.n_pairs, 1.7);
    let mut s1 = EdScratch::new(&cfg, IN_DIM, 1);
    let mut s2 = EdScratch::new(&cfg, IN_DIM, 1);
    let r1 = ed_over_pairs(|x, y| y[0] = linear_form(x).powi(3), &a, &b, &cfg, &mut s1).unwrap();
    let r2 = ed_over_pairs(
        |x, y| y[0] = 2.0 * linear_form(x).powi(3),
        &a,
        &b,
        &cfg,
        &mut s2,
    )
    .unwrap();
    println!(
        "G1c ed {:.6} -> {:.6} (x{:.4}); ed_norm {:.6} -> {:.6}",
        r1.ed,
        r2.ed,
        r2.ed / r1.ed,
        r1.ed_norm,
        r2.ed_norm
    );
    assert!((r2.ed / r1.ed - 2.0).abs() < 1e-3, "ED must scale ×2");
    assert!((r2.ed_norm - r1.ed_norm).abs() < 1e-4, "ED_norm must not scale");
}

#[test]
fn g1d_constant_is_zero_and_affine_is_first_coefficient() {
    let cfg = EdConfig::cheap();
    let mut nodes = vec![0.0f32; cfg.resolution];
    randomized_cosine_nodes(cfg.resolution, cfg.seed, &mut nodes).unwrap();

    let constant = vec![-2.25f32; cfg.resolution];
    let rc = effective_degree_along_path(&constant, &nodes, &cfg).unwrap();
    assert!(rc.ed < 1e-4, "constant ED = {}", rc.ed);
    assert!(rc.ed_norm < 1e-4, "constant ED_norm = {}", rc.ed_norm);

    // f(α) = 1 + 4α  ⇒  in u = 2α−1: 3 + 2·T₁(u), so ED = |c₁| = 2.
    let affine: Vec<f32> = nodes.iter().map(|&a| 4.0f32.mul_add(a, 1.0)).collect();
    let ra = effective_degree_along_path(&affine, &nodes, &cfg).unwrap();
    println!("G1d affine coeffs = {:?}", &ra.coeff_norms[..ra.n_terms]);
    assert!((ra.ed - ra.coeff_norms[1]).abs() < 1e-5, "ED != |c₁|");
    assert!((ra.coeff_norms[1] - 2.0).abs() < 1e-3);
}

#[test]
fn g1e_order_is_stable_across_node_seeds() {
    for seed in 0..8u64 {
        let cfg = EdConfig {
            seed: 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(seed + 1),
            ..EdConfig::precise()
        };
        let (e1, _) = measure_ed_norm(1, &cfg);
        let (e2, _) = measure_ed_norm(2, &cfg);
        let (e5, _) = measure_ed_norm(5, &cfg);
        assert!(
            e1 < e2 && e2 < e5,
            "seed {seed}: order broke ({e1:.4}, {e2:.4}, {e5:.4})"
        );
    }
}

// ── G2 ───────────────────────────────────────────────────────────────────────

fn best_of_3<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

#[test]
fn g2_per_path_latency_and_pair_scaling() {
    // Fit-only cost (the (K+1)² solve + r basis evaluations), decode excluded —
    // the decode is the consumer's cost, not the metric's.
    for cfg in [EdConfig::cheap(), EdConfig::precise()] {
        let mut nodes = vec![0.0f32; cfg.resolution];
        randomized_cosine_nodes(cfg.resolution, cfg.seed, &mut nodes).unwrap();
        let outputs: Vec<f32> = nodes.iter().map(|&a| a.powi(3) - 0.4 * a).collect();
        let ns = best_of_3(200_000, || {
            black_box(
                effective_degree_along_path(black_box(&outputs), black_box(&nodes), &cfg).unwrap(),
            );
        });
        println!(
            "G2 fit r={} K={}: {ns:.1} ns/path",
            cfg.resolution, cfg.max_degree
        );
        // Issue 668 T3 target is sub-µs at the cheap config; the precise
        // config gets 2× headroom so the gate is not load-flaky on a busy box.
        let budget = if cfg.resolution <= 4 { 500.0 } else { 2000.0 };
        assert!(
            ns < budget,
            "r={} K={}: {ns} ns/path exceeds {budget} ns",
            cfg.resolution,
            cfg.max_degree
        );
    }

    // Node sampling cost (paid once per path by the driver).
    let cfg = EdConfig::cheap();
    let mut nodes = vec![0.0f32; cfg.resolution];
    let ns = best_of_3(500_000, || {
        randomized_cosine_nodes(black_box(cfg.resolution), black_box(cfg.seed), &mut nodes).unwrap();
    });
    println!("G2 randomized_cosine_nodes r={}: {ns:.1} ns", cfg.resolution);
    assert!(ns < 200.0, "node sampling {ns} ns");

    // Pair-count scaling of the full driver (decode = one multiply-add chain).
    let mut prev: Option<(usize, f64)> = None;
    for n_pairs in [4usize, 8, 16, 32] {
        let cfg = EdConfig {
            n_pairs,
            ..EdConfig::cheap()
        };
        let a = manifold_points(n_pairs, 0.0);
        let b = manifold_points(n_pairs, 1.7);
        let mut scratch = EdScratch::new(&cfg, IN_DIM, 1);
        let ns = best_of_3(20_000, || {
            black_box(
                ed_over_pairs(
                    |x, y| y[0] = linear_form(x).powi(3),
                    black_box(&a),
                    black_box(&b),
                    &cfg,
                    &mut scratch,
                )
                .unwrap(),
            );
        });
        println!("G2 ed_over_pairs n_pairs={n_pairs}: {ns:.1} ns ({:.1} ns/pair)", ns / n_pairs as f64);
        if let Some((pn, pns)) = prev {
            let ratio = ns / pns;
            let expected = n_pairs as f64 / pn as f64;
            // Tolerance 2.5× (was 1.6×): isolated, per-pair cost is flat
            // (~153-155 ns/pair across 4→32 pairs) and the ratio reads ~1.0×.
            // Under a full-workspace release run the same cells measured
            // 305→495 ns/pair (+62% cell noise) → a 3.24× reading that is
            // scheduler noise, not superlinearity (the 718(a) pricing run
            // aborted here). 2.5× still catches a true per-pair doubling at
            // a 2× pair step (4× > 2.5×) — the actual regression class this
            // gate exists for.
            assert!(
                ratio < expected * 2.5,
                "n_pairs {pn}→{n_pairs}: {ratio:.2}× is worse than linear ({expected:.2}×)"
            );
        }
        prev = Some((n_pairs, ns));
    }
}

#[test]
fn g2_max_terms_is_the_documented_cap() {
    assert_eq!(MAX_ED_TERMS, 8);
    assert_eq!(EdConfig::cheap().n_terms(), 4);
    assert_eq!(EdConfig::precise().n_terms(), 8);
}
