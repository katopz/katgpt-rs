//! In-module tests for slice_tca (Plan 596 T1.6 / T1.7 + Phase 2 support).
//!
//! Fixtures are seeded `fastrand` plants: a class-σ plant is `Σ a_r ⊗ M_r`
//! with generic random unit-norm slice matrices `M_r` and unit-norm loadings
//! — the class signature lives in the slice matrices' genericity. Noise is
//! additive uniform scaled to a target energy ratio.
//!
//! # Fixture shapes (load-bearing)
//!
//! Classifier-dependent tests use balanced shapes (`[48, 48, 48]`): a pure
//! class-σ plant's COMPLEMENTARY-axis share is the Marchenko–Pastur top-R
//! concentration of its generic slice Grams, ≈ `R·(1+√aspect)²/min_dim` —
//! ~0.17 at R=2 on [48³], safely under `ROUTE_THETA = 0.25`. Thin-axis shapes
//! (e.g. k=8) concentrate cross-axis shares ABOVE θ and false-positive route
//! — a documented calibration boundary (Bench 714 caveats), not exercised by
//! these tests.

use super::svd::covariability_shares_into;
use super::types::*;
use super::{
    fit_slice_into, fit_with_ranks_into, reallocate_class, relative_loss, route, route_default,
};

/// Balanced shape for classifier-dependent tests.
const BAL: [usize; 3] = [48, 48, 48];

// ─── Fixture generators ─────────────────────────────────────────────────────

struct Rng(fastrand::Rng);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(fastrand::Rng::with_seed(seed))
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.0.f32() * 2.0 - 1.0
    }
    fn unit_vec(&mut self, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| self.sym()).collect();
        let n = frob_norm(&v).max(1e-12);
        for e in &mut v {
            *e /= n;
        }
        v
    }
    fn unit_matrix(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        let mut m: Vec<f32> = (0..rows * cols).map(|_| self.sym()).collect();
        let n = frob_norm(&m).max(1e-12);
        for e in &mut m {
            *e /= n;
        }
        m
    }
}

fn frob_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Add a class-σ plant: `count` components `a ⊗ M` (unit loadings, generic
/// unit-norm slices), accumulating into `x`.
fn add_plant(x: &mut Tensor3, class: SliceClass, count: usize, rng: &mut Rng) {
    let [n, t, k] = x.shape;
    for _ in 0..count {
        let d = [n, t, k][class.axis()];
        let a = rng.unit_vec(d);
        let (m, _rows, _cols) = match class {
            SliceClass::Entity => (rng.unit_matrix(t, k), t, k),
            SliceClass::Time => (rng.unit_matrix(n, k), n, k),
            SliceClass::Episode => (rng.unit_matrix(n, t), n, t),
        };
        add_rank1_slice(&mut x.data, x.shape, class, &a, &m, 1.0);
    }
}

/// Add uniform noise scaled so its energy is `ratio` of the current signal
/// energy (deterministic given the rng state).
fn add_noise(x: &mut Tensor3, ratio: f32, rng: &mut Rng) {
    let sig = frob_norm(&x.data);
    let noise: Vec<f32> = (0..x.data.len()).map(|_| rng.sym()).collect();
    let nn = frob_norm(&noise).max(1e-12);
    let scale = sig * ratio / nn;
    for (dst, &src) in x.data.iter_mut().zip(noise.iter()) {
        *dst += scale * src;
    }
}

fn mixed_fixture(shape: [usize; 3], noise: f32, seed: u64) -> Tensor3 {
    let mut x = Tensor3::zeros(shape);
    let mut rng = Rng::new(seed);
    add_plant(&mut x, SliceClass::Entity, 2, &mut rng);
    add_plant(&mut x, SliceClass::Time, 2, &mut rng);
    if noise > 0.0 {
        add_noise(&mut x, noise, &mut rng);
    }
    x
}

fn pure_fixture(shape: [usize; 3], class: SliceClass, comps: usize, noise: f32, seed: u64) -> Tensor3 {
    let mut x = Tensor3::zeros(shape);
    let mut rng = Rng::new(seed);
    add_plant(&mut x, class, comps, &mut rng);
    if noise > 0.0 {
        add_noise(&mut x, noise, &mut rng);
    }
    x
}

// ─── T1.3 classifier ────────────────────────────────────────────────────────

fn shares_of(x: &Tensor3, shape: [usize; 3], r: usize) -> [f32; 3] {
    super::covariability_shares(&x.data, shape, r)
}

#[test]
fn shares_separate_pure_classes() {
    for class in SliceClass::ALL {
        let x = pure_fixture(BAL, class, 2, 0.05, 42 + class.axis() as u64);
        let shares = shares_of(&x, BAL, 2);
        let own = shares[class.axis()];
        for (other, &s) in shares.iter().enumerate() {
            if other != class.axis() {
                assert!(
                    own > s + 0.2,
                    "class {class:?}: own share {own:.3} vs axis {other} share {s:.3}"
                );
            }
        }
    }
}

#[test]
fn shares_route_mixed_two_class() {
    let x = mixed_fixture(BAL, 0.05, 7);
    let shares = shares_of(&x, BAL, 2);
    let routes = route_default(shares);
    assert!(routes[0] > 0.5, "entity class routed OFF: {routes:?}");
    assert!(routes[1] > 0.5, "time class routed OFF: {routes:?}");
    assert!(routes[2] < 0.5, "episode class routed ON: {routes:?}");
}

#[test]
fn route_is_sigmoid_not_softmax() {
    // Both classes can be ~fully on simultaneously (non-exclusive).
    let r = route([0.9, 0.9, 0.0], 0.25, 20.0);
    assert!(r[0] > 0.99 && r[1] > 0.99);
    assert!(r[2] < 0.01);
    // Monotone in each share independently.
    let lo = route([0.2, 0.0, 0.0], 0.25, 20.0);
    let hi = route([0.3, 0.0, 0.0], 0.25, 20.0);
    assert!(hi[0] > lo[0]);
}

#[test]
fn zero_tensor_shares_zero_and_empty_fit() {
    let shape = [6, 5, 4];
    let x = Tensor3::zeros(shape);
    let shares = shares_of(&x, shape, 2);
    assert_eq!(shares, [0.0; 3]);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    fit_slice_into(&x.data, shape, &SliceTcaConfig::default(), &mut scratch, &mut d).unwrap();
    assert_eq!(d.n_components, 0);
    let mut hat = Tensor3::zeros(shape);
    d.reconstruct_into(&mut hat).unwrap();
    assert!(hat.data.iter().all(|&v| v == 0.0));
}

// ─── T1.2 single-class fit ──────────────────────────────────────────────────

#[test]
fn single_class_recovers_plant() {
    let shape = [24, 32, 12];
    let x = pure_fixture(shape, SliceClass::Entity, 2, 0.0, 99);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    super::fit_single_class_into(&x.data, shape, 0, 2, ENERGY_FLOOR_TAU, &mut scratch, &mut d)
        .unwrap();
    assert_eq!(d.n_components, 2);
    assert_eq!(d.ranks, [2, 0, 0]);
    // Pure generic-M plant at the true rank: near-exact recovery
    // (Eckart–Young at the unfolding level; the slice model represents it
    // exactly).
    let loss = relative_loss(&x, &d);
    assert!(loss < 1e-4, "single-class loss {loss}");
}

#[test]
fn single_class_rank_bounds_enforced() {
    let shape = [4, 6, 8];
    let x = Tensor3::zeros(shape);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    let err = super::fit_single_class_into(
        &x.data,
        shape,
        0,
        5,
        ENERGY_FLOOR_TAU,
        &mut scratch,
        &mut d,
    );
    // min(4, 48) = 4 < 5.
    assert!(matches!(err, Err(SliceTcaError::RankTooLarge { axis: 0, rank: 5, bound: 4 })));
}

// ─── T1.4 / T1.7 ALS ────────────────────────────────────────────────────────

#[test]
fn als_loss_monotone_non_increasing() {
    let shape = [24, 32, 12];
    let x = mixed_fixture(shape, 0.05, 7);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    let cfg = SliceTcaConfig::default();
    fit_with_ranks_into(&x.data, shape, [2, 2, 0], &cfg, &mut scratch, &mut d).unwrap();
    let losses = scratch.last_sweep_losses();
    assert_eq!(losses.len(), cfg.als_sweeps + 1, "init + sweep losses recorded");
    // Monotone within f32 slack (the exact-math sequence is non-increasing).
    let tol = 1e-5;
    for w in 1..losses.len() {
        assert!(
            losses[w] <= losses[w - 1] + tol,
            "loss rose at sweep {w}: {losses:?}"
        );
    }
}

#[test]
fn als_improves_or_matches_svd_init() {
    let shape = [24, 32, 12];
    let x = mixed_fixture(shape, 0.05, 7);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    let cfg = SliceTcaConfig::default();
    fit_with_ranks_into(&x.data, shape, [2, 2, 0], &cfg, &mut scratch, &mut d).unwrap();
    let first = scratch.last_sweep_losses()[0];
    let last = *scratch.last_sweep_losses().last().unwrap();
    assert!(last <= first + 1e-6, "ALS worsened: first {first}, last {last}");
}

#[test]
fn mixed_fit_beats_single_class_fits() {
    let shape = BAL;
    let x = mixed_fixture(shape, 0.0, 7);
    let mut scratch = SliceTcaScratch::with_capacity(shape);

    // Joint fit.
    let mut joint = SliceDecomposition::empty(shape);
    let cfg = SliceTcaConfig::default();
    fit_with_ranks_into(&x.data, shape, [2, 2, 0], &cfg, &mut scratch, &mut joint).unwrap();
    let joint_loss = relative_loss(&x, &joint);

    // Floor: best single-class fit at the SAME total budget (4).
    let mut best_floor = f32::INFINITY;
    for axis in 0..3 {
        let mut d = SliceDecomposition::empty(shape);
        super::fit_single_class_into(&x.data, shape, axis, 4, ENERGY_FLOOR_TAU, &mut scratch, &mut d)
            .unwrap();
        best_floor = best_floor.min(relative_loss(&x, &d));
    }
    assert!(
        joint_loss < best_floor,
        "joint {joint_loss} vs single-class floor {best_floor}"
    );
}

#[test]
fn full_pipeline_reconstruction_round_trip() {
    let shape = BAL;
    let x = mixed_fixture(shape, 0.05, 7);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    fit_slice_into(&x.data, shape, &SliceTcaConfig::default(), &mut scratch, &mut d).unwrap();
    let loss = relative_loss(&x, &d);
    // Noise floor ~5% energy + classifier margins: a well-fit mixed signal
    // must sit well below the noise-free budget.
    assert!(loss < 0.25, "round-trip loss {loss}");
    // Canonical order: weights descending.
    for i in 1..d.n_components {
        assert!(
            d.weights[i - 1] >= d.weights[i],
            "weights not sorted: {:?}",
            &d.weights[..d.n_components]
        );
    }
    // Sign rule: largest-|·| loading entry positive.
    for i in 0..d.n_components {
        let l = d.loading(i);
        let mut best = 0.0f32;
        let mut best_abs = -1.0f32;
        for &v in l {
            if v.abs() > best_abs {
                best_abs = v.abs();
                best = v;
            }
        }
        if d.weights[i] > NORM_EPS {
            assert!(best > 0.0, "component {i} sign rule violated");
        }
    }
}

// ─── T1.5 canonicalization + reallocation ───────────────────────────────────

#[test]
fn class_pass_canonical_output_unchanged() {
    // The same rank-1 tensor u(0) ⊗ v(1) ⊗ w(2), expressed as an Entity
    // component and as a Time component. Canonical weights must match and
    // the reconstructions agree (the plan's "passed between classes"
    // invariance; only the class label differs).
    let shape = [6, 8, 4];
    let mut rng = Rng::new(3);
    let u = rng.unit_vec(shape[0]);
    let v = rng.unit_vec(shape[1]);
    let w = rng.unit_vec(shape[2]);
    let amp = 3.0f32;

    let mut d1 = SliceDecomposition::empty(shape);
    // Entity: loading = amp·u, slice = v ⊗ w (t × k).
    let mut m1 = vec![0.0; shape[1] * shape[2]];
    for p in 0..shape[1] {
        for l in 0..shape[2] {
            m1[p * shape[2] + l] = v[p] * w[l];
        }
    }
    d1.push_component(
        SliceClass::Entity,
        &u.iter().map(|&x| amp * x).collect::<Vec<_>>(),
        &m1,
        shape[1],
        shape[2],
        amp,
    )
    .unwrap();

    let mut d2 = SliceDecomposition::empty(shape);
    // Time: loading = amp·v, slice = u ⊗ w (n × k).
    let mut m2 = vec![0.0; shape[0] * shape[2]];
    for i in 0..shape[0] {
        for l in 0..shape[2] {
            m2[i * shape[2] + l] = u[i] * w[l];
        }
    }
    d2.push_component(
        SliceClass::Time,
        &v.iter().map(|&x| amp * x).collect::<Vec<_>>(),
        &m2,
        shape[0],
        shape[2],
        amp,
    )
    .unwrap();

    d1.canonicalize(0.0);
    d2.canonicalize(0.0);
    assert!((d1.weights[0] - d2.weights[0]).abs() < 1e-5);
    let mut hat1 = Tensor3::zeros(shape);
    let mut hat2 = Tensor3::zeros(shape);
    d1.reconstruct_into(&mut hat1).unwrap();
    d2.reconstruct_into(&mut hat2).unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in hat1.data.iter().zip(hat2.data.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(max_diff < 1e-5, "recon diff {max_diff}");
}

#[test]
fn reallocate_exact_for_rank1_slice() {
    // A plant with an exactly rank-1 slice: X = a ⊗ (b ⊗ c). The fitted
    // slice is rank-1 up to f32 noise, so reallocation is (near-)lossless.
    let shape = [8, 6, 4];
    let mut rng = Rng::new(11);
    let a = rng.unit_vec(shape[0]);
    let b = rng.unit_vec(shape[1]);
    let c = rng.unit_vec(shape[2]);
    let mut x = Tensor3::zeros(shape);
    // slice m = b ⊗ c (t × k), unit Frobenius norm by construction.
    let m: Vec<f32> = (0..shape[1] * shape[2])
        .map(|e| {
            let (p, l) = (e / shape[2], e % shape[2]);
            b[p] * c[l]
        })
        .collect();
    add_rank1_slice(&mut x.data, shape, SliceClass::Entity, &a, &m, 1.0);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    super::fit_single_class_into(&x.data, shape, 0, 1, ENERGY_FLOOR_TAU, &mut scratch, &mut d)
        .unwrap();
    assert_eq!(d.n_components, 1);
    let mut before = Tensor3::zeros(shape);
    d.reconstruct_into(&mut before).unwrap();

    let ratio = reallocate_class(&mut d, 0, SliceClass::Time, &mut scratch).unwrap();
    assert!(ratio < 1e-3, "reallocation ratio {ratio}");
    assert_eq!(d.classes[0], SliceClass::Time);
    let mut after = Tensor3::zeros(shape);
    d.reconstruct_into(&mut after).unwrap();
    let mut diff = 0.0f32;
    for (a, b) in before.data.iter().zip(after.data.iter()) {
        diff = diff.max((a - b).abs());
    }
    assert!(diff < 1e-2, "reallocation changed the tensor by {diff}");
}

#[test]
fn reallocate_degenerate_reports_ratio() {
    let shape = [8, 6, 4];
    let x = pure_fixture(shape, SliceClass::Entity, 2, 0.0, 5);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    super::fit_single_class_into(&x.data, shape, 0, 2, ENERGY_FLOOR_TAU, &mut scratch, &mut d)
        .unwrap();
    // A generic (t × k) slice is far from rank-1: the documented-degenerate
    // case returns a positive ratio instead of pretending to be lossless.
    let ratio = reallocate_class(&mut d, 0, SliceClass::Episode, &mut scratch).unwrap();
    assert!(ratio > 0.1, "expected degenerate ratio > 0.1, got {ratio}");
}

// ─── T1.4 HOSVD consumption ─────────────────────────────────────────────────

#[test]
fn hosvd_init_matches_gram_init() {
    // Small shape inside Tucker's SVD_MAX_RANK=16 bound: both init paths
    // must recover the same singular amplitudes and the pure plant exactly.
    let shape = [8, 6, 4];
    let x = pure_fixture(shape, SliceClass::Entity, 2, 0.0, 21);
    let cfg_gram = SliceTcaConfig {
        init: InitMode::GramSvd,
        ..SliceTcaConfig::default()
    };
    let cfg_hosvd = SliceTcaConfig {
        init: InitMode::Hosvd,
        ..SliceTcaConfig::default()
    };
    let mut s1 = SliceTcaScratch::with_capacity(shape);
    let mut d1 = SliceDecomposition::empty(shape);
    fit_with_ranks_into(&x.data, shape, [2, 0, 0], &cfg_gram, &mut s1, &mut d1).unwrap();
    let mut s2 = SliceTcaScratch::with_capacity(shape);
    let mut d2 = SliceDecomposition::empty(shape);
    fit_with_ranks_into(&x.data, shape, [2, 0, 0], &cfg_hosvd, &mut s2, &mut d2).unwrap();
    assert_eq!(d1.n_components, d2.n_components);
    for i in 0..d1.n_components {
        let rel = ((d1.weights[i] - d2.weights[i]) / d1.weights[i]).abs();
        assert!(
            rel < 1e-3,
            "comp {i}: gram {} vs hosvd {}",
            d1.weights[i],
            d2.weights[i]
        );
    }
    let (l1, l2) = (relative_loss(&x, &d1), relative_loss(&x, &d2));
    assert!(l1 < 1e-3 && l2 < 1e-3, "gram {l1} hosvd {l2}");
}

#[test]
fn hosvd_auto_falls_back_on_large_shapes() {
    // [24, 32, 8]: mode-1 unfolding min-dim is 32 > Tucker's SVD_MAX_RANK
    // (16) → Auto must fall back to GramSvd; explicit Hosvd must refuse.
    let shape = [24, 32, 8];
    let x = mixed_fixture(shape, 0.0, 33);
    let cfg = SliceTcaConfig {
        init: InitMode::Auto,
        ..SliceTcaConfig::default()
    };
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    fit_with_ranks_into(&x.data, shape, [2, 2, 0], &cfg, &mut scratch, &mut d).unwrap();
    assert!(d.n_components >= 3);

    let cfg_forced = SliceTcaConfig {
        init: InitMode::Hosvd,
        ..SliceTcaConfig::default()
    };
    let err = fit_with_ranks_into(&x.data, shape, [2, 2, 0], &cfg_forced, &mut scratch, &mut d);
    assert!(matches!(err, Err(SliceTcaError::HosvdShapeUnsupported)));
}

// ─── T1.6 determinism ───────────────────────────────────────────────────────

#[test]
fn determinism_blake3_across_calls_and_rebuild() {
    let shape = [24, 32, 12];
    let x = mixed_fixture(shape, 0.05, 77);
    let cfg = SliceTcaConfig::default();

    let mut warm = SliceTcaScratch::with_capacity(shape);
    let mut warm_d = SliceDecomposition::empty(shape);
    fit_slice_into(&x.data, shape, &cfg, &mut warm, &mut warm_d).unwrap();
    let reference = warm_d.canonical_hash();

    // 16 calls: a fresh scratch + decomp every 4th call (calls 0, 4, 8, 12),
    // the warm pair otherwise — all must reproduce the reference hash.
    for call in 0..16u32 {
        let h = if call % 4 == 0 {
            let mut scratch = SliceTcaScratch::with_capacity(shape);
            let mut d = SliceDecomposition::empty(shape);
            fit_slice_into(&x.data, shape, &cfg, &mut scratch, &mut d).unwrap();
            d.canonical_hash()
        } else {
            let mut d = SliceDecomposition::empty(shape);
            fit_slice_into(&x.data, shape, &cfg, &mut warm, &mut d).unwrap();
            d.canonical_hash()
        };
        assert_eq!(h, reference, "call {call} diverged");
    }

    // Drop-and-rebuild (the warm pair destroyed, everything re-created).
    drop(warm);
    drop(warm_d);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    fit_slice_into(&x.data, shape, &cfg, &mut scratch, &mut d).unwrap();
    assert_eq!(d.canonical_hash(), reference, "rebuild diverged");
}

// ─── T2.1 rank selection ────────────────────────────────────────────────────

#[test]
fn knee_recovers_pure_plant_ranks() {
    let shape = BAL;
    let x = pure_fixture(shape, SliceClass::Entity, 2, 0.02, 13);
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let shares = covariability_shares_into(&x.data, shape, 2, &mut scratch);
    let routes = route(shares, ROUTE_THETA, ROUTE_ALPHA);
    let ranks = super::select_ranks_knee(
        scratch.spectra(),
        scratch.norm_sq(),
        routes,
        &SliceTcaConfig::default(),
    );
    assert_eq!(ranks, [2, 0, 0], "ranks {ranks:?}");
}

#[test]
fn knee_and_blocked_cv_agree_on_clean_mixture() {
    let shape = BAL;
    let x = mixed_fixture(shape, 0.0, 5);
    let cfg = SliceTcaConfig::default();
    let mut scratch = SliceTcaScratch::with_capacity(shape);

    let shares = covariability_shares_into(&x.data, shape, cfg.share_rank, &mut scratch);
    let routes = route(shares, cfg.route_theta, cfg.route_alpha);
    let knee = super::select_ranks_knee(scratch.spectra(), scratch.norm_sq(), routes, &cfg);

    let cv = super::select_ranks_blocked_cv(&x.data, shape, &cfg, &mut scratch, 4, 3).unwrap();
    assert_eq!(knee, cv, "knee {knee:?} vs cv {cv:?}");
}

// ─── T2.4 alloc discipline (TrackingAllocator) ──────────────────────────────
// Gated on the SAME predicate as `crate::alloc` itself (Issue 758): the test
// runs in dev (debug_assertions) AND under `--release --features
// alloc_tracking` — the Issue 741 configuration the alloc gates are meant to
// be read in (verified against the optimised code that ships). In a release
// build WITHOUT the feature the substrate does not exist, so the test
// compiles away rather than breaking the whole lib test harness (E0432).
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[test]
fn zero_alloc_reconstruction_and_classifier() {
    use crate::alloc::{get_alloc_stats, reset_alloc_stats};
    let shape = BAL;
    let x = mixed_fixture(shape, 0.05, 7);
    let cfg = SliceTcaConfig::default();
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    fit_slice_into(&x.data, shape, &cfg, &mut scratch, &mut d).unwrap();
    let mut hat = Tensor3::zeros(shape);
    d.reconstruct_into(&mut hat).unwrap();
    let mut slab = vec![0.0f32; shape[1] * shape[2]];

    // Warm every measured path once, then measure 32 steady-state rounds.
    let _ = covariability_shares_into(&x.data, shape, 2, &mut scratch);
    d.entity_slice_into(3, &mut slab).unwrap();
    reset_alloc_stats();
    for _ in 0..32 {
        let shares = covariability_shares_into(&x.data, shape, 2, &mut scratch);
        let _routes = route_default(shares);
        d.reconstruct_into(&mut hat).unwrap();
        d.entity_slice_into(3, &mut slab).unwrap();
    }
    let (allocs, _bytes) = get_alloc_stats();
    assert_eq!(allocs, 0, "{allocs} allocations in the steady-state hot path");
}

#[test]
fn entity_slice_matches_full_reconstruction() {
    let shape = BAL;
    let x = mixed_fixture(shape, 0.05, 7);
    let cfg = SliceTcaConfig::default();
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    let mut d = SliceDecomposition::empty(shape);
    fit_slice_into(&x.data, shape, &cfg, &mut scratch, &mut d).unwrap();
    let mut hat = Tensor3::zeros(shape);
    d.reconstruct_into(&mut hat).unwrap();
    let mut slab = vec![0.0f32; shape[1] * shape[2]];
    for e in [0, 24, 47] {
        d.entity_slice_into(e, &mut slab).unwrap();
        let expect = &hat.data[e * shape[1] * shape[2]..(e + 1) * shape[1] * shape[2]];
        let mut max_diff = 0.0f32;
        for (a, b) in slab.iter().zip(expect.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff < 1e-4, "entity {e}: diff {max_diff}");
    }
}
