//! Grid sizer — Plan 619 T1.5 (support-radius law) + Issue 938 T1/T2
//! (measured-erosion capacity, arity-free law).
//!
//! `LatticeMemory::sized` turns (expected items, near-miss radius) into a
//! grid + a fitted occupancy warp. The law, and why it is shaped this way:
//!
//! **The support radius drives the grid.** The class claim (graded
//! near-miss recall) lives in the 3×3 bump neighborhood — a query perturbed
//! by less than the support radius must recover through neighbor weight.
//! Under the CDF warp the coordinate slope at the median is
//! `0.399·L/σ_fit` cells per raw unit, so the ±1.5-cell support spans
//! `3.76·σ_fit/L` raw units: the grid side is `L = ⌈3.76/R⌉` for a desired
//! radius `R` (in fitted-σ units) — PER AXIS, at any arity (Issue 938 T2.2:
//! cells = `L^AXES`, and the capacity envelope generalizes to
//! `N_max ≈ 0.35·d_k·(3.76/R)^AXES`). Sizing from OCCUPANCY instead (the
//! first cut, `items/λ`) decoupled the two — at 10⁶ items it chose
//! L ≈ 700 and a 0.005σ support, the G5 gate measured the cliff (recall
//! 0.997 → 0.06 at ε = 0.02σ), and the law below is the repair.
//!
//! **Occupancy floats; the delta rule carries the crowding.** With L fixed
//! by the support, the per-cell mean is `λ = items/L^AXES` — allowed to
//! exceed 1 because a cell is a `d_k × d_v` associative memory, not one
//! slot. The capacity gate: same-cell keys are ~N(0, 1/d_k) correlated, so
//! each write erodes older associations by `E[cos²] = 1/d_k`; survival ≥ ½
//! after λ overwrites gives `λ ≤ ln2·d_k`, halved for the cross-term noise
//! floor → the first-order capacity `0.35·d_k` (gated, not assumed — the
//! GOAT gates measure the real recall at the chosen operating point).
//! Issue 938 T1 adds the measured-erosion parameterization: a corpus whose
//! same-cell keys erode FASTER than `1/d_k` (correlated/vocabulary-style
//! keys — μLattice §5.1 measured the hazard on learned keys) tightens the
//! refusal surface through [`capacity_per_cell_eroded`]
//! / [`LatticeMemory::sized_eroded`], estimated from the fit corpus itself.
//!
//! **The CDF warp stays load-bearing.** It equalizes per-axis marginals
//! (uniform occupancy predictions hold — the histogram gate below) AND
//! decouples the grid from the projected range: tails clamp into the
//! boundary cells, so `L` is free of the ±4σ coverage arithmetic entirely.

use super::LatticeMemoryG;
use super::address::{LshAxes, split_coordinate};
use super::cdf::CdfWarp;
use super::types::{BumpKernel, LatticeConfigError, LatticeConfigG};

/// Cells per support radius: the median CDF slope `0.399·L/σ` inverted —
/// a ±1.5-cell support spans `3.76σ/L` raw units. The sizer's one constant.
const CELLS_PER_SUPPORT: f32 = 3.76;

/// The default target recall of the capacity law: survival ≥ ½ at the
/// capacity load (the T1.1 pin).
const DEFAULT_TARGET_RECALL: f32 = 0.5;

/// First-order per-cell capacity at the measured erosion — the T1.1
/// generalization of `0.35·d_k`.
///
/// Derivation: a same-cell association survives λ later overwrites with
/// probability ≈ `(1 − e_cos2)^λ`; survival ≥ `target_recall` gives
/// `λ ≤ ln(1/target_recall)/e_cos2`, halved for the cross-term noise floor.
/// The halving is CALIBRATED to the shipped pin: at the default erosion
/// `e = 1/d_k` and `target = ½` the law must return exactly `0.35·d_k` —
/// the pin is returned directly (Issue 938 T1.4: the G3 law
/// "defaults bit-identical" holds by construction, not by hoping two float
/// roundings cancel), and the general arm moves by the survival ratio.
///
/// # Panics
/// On `e_cos2 ≤ 0`/non-finite, or `target_recall` outside `(0, 1)`.
#[must_use]
pub fn capacity_per_cell_eroded(d_k: usize, e_cos2: f32, target_recall: f32) -> f32 {
    assert!(
        e_cos2 > 0.0 && e_cos2.is_finite(),
        "e_cos2 must be finite > 0"
    );
    assert!(
        target_recall > 0.0 && target_recall < 1.0,
        "target_recall must be in (0, 1), got {target_recall}"
    );
    // The shipped pin — bit-for-bit (T1.1).
    if e_cos2 == 1.0 / d_k as f32 && target_recall == DEFAULT_TARGET_RECALL {
        return 0.35 * d_k as f32;
    }
    // The cross-term noise-floor factor, calibrated to the pin:
    // 0.35·d_k = (0.35/ln2)·ln2·d_k.
    const NOISE_FLOOR: f32 = 0.35 / core::f32::consts::LN_2;
    NOISE_FLOOR * target_recall.recip().ln() / e_cos2
}

/// First-order per-cell capacity: `0.35·d_k` associations (see module docs
/// for the erosion derivation; the GOAT gates verify it empirically). The
/// thin wrapper over [`capacity_per_cell_eroded`] at the pinned defaults —
/// Plan 619 callers never churn (Issue 938 T1.1).
#[must_use]
pub fn capacity_per_cell(d_k: usize) -> f32 {
    capacity_per_cell_eroded(d_k, 1.0 / d_k as f32, DEFAULT_TARGET_RECALL)
}

/// The support-radius law's grid side: `L = ⌈3.76/R⌉` per axis (T2.2's
/// arity-free form of the inline expression). `near_miss_sigma` is the
/// caller-validated radius in `(0, 10]` σ — a non-positive or non-finite
/// radius clamps to a side of 1 rather than panicking here (the sizer
/// validates before calling).
#[must_use]
pub fn grid_side_for(near_miss_sigma: f32) -> u32 {
    (CELLS_PER_SUPPORT / near_miss_sigma).ceil().max(1.0) as u32
}

/// The support radius a built lattice's grid actually delivers, in
/// fitted-σ units (the inverse of the sizing law — the honest readout for
/// callers that let the budget clamp their grid).
#[must_use]
pub fn support_radius(grid_side: u32) -> f32 {
    CELLS_PER_SUPPORT / grid_side as f32
}

/// Which erosion assumption the capacity gate consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Erosion {
    /// `e = 1/d_k` — the Plan 619 random-key assumption (the shipped
    /// `sized` posture, bit-identical refusal surface).
    RandomKeyAssumption,
    /// Estimate the same-cell `E[cos²]` from the fit corpus and use
    /// `max(e_hat, 1/d_k)` (Issue 938 T1.2 — tightens the refusal surface
    /// when the corpus' keys are correlated).
    MeasureFromFitCorpus,
}

/// The per-group / total pair caps of the erosion estimator: a deterministic
/// bounded scan (fit-time, but still O(1) per cell beyond the caps).
const EROSION_MAX_GROUP_SCAN: usize = 24;
const EROSION_MAX_PAIRS: u64 = 200_000;

/// Estimate the same-cell `E[cos²]` of the fit corpus under the fitted grid
/// (Issue 938 T1.2): pairs of samples landing in the same primary cell,
/// their key-cosine squared over the first `d_k` components (the delta-rule
/// key width), averaged in f64.
///
/// Fit-time ONLY (the hot path never runs this — the G4 law is untouched).
/// Deterministic: a sorted scan with the caps above. `None` when the corpus
/// produced no same-cell pairs (a spread corpus at a fine grid — the
/// random-key assumption stands).
fn estimate_same_cell_e_cos2<const A: usize>(
    axes: &LshAxes<A>,
    warp: Option<&[CdfWarp; A]>,
    samples: &[&[f32]],
    side: u32,
    d_k: usize,
) -> Option<f32> {
    let side_usize = side as usize;
    // Primary cell per sample (the same fold the lattice's locate uses).
    let mut cell_of: Vec<u32> = Vec::with_capacity(samples.len());
    for s in samples {
        let c = axes.coordinates(s);
        let mut idx = 0_usize;
        for axis in 0..A {
            let cc = match warp {
                Some(w) => w[axis].apply(c[axis]),
                None => c[axis],
            };
            let (i, f) = split_coordinate(cc);
            let m = (i + i64::from(f > 0.5)).clamp(0, i64::from(side) - 1) as usize;
            idx = idx * side_usize + m;
        }
        cell_of.push(idx as u32);
    }
    // Group by cell: a sorted index permutation, then a linear group scan.
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by_key(|&i| cell_of[i]);
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    let mut gi = 0;
    while gi < order.len() && count < EROSION_MAX_PAIRS {
        let cell = cell_of[order[gi]];
        let mut gj = gi + 1;
        while gj < order.len() && cell_of[order[gj]] == cell && count < EROSION_MAX_PAIRS {
            gj += 1;
        }
        let group = &order[gi..gj];
        let take = group.len().min(EROSION_MAX_GROUP_SCAN);
        for a in 0..take {
            for b in (a + 1)..take {
                let (ka, kb) = (samples[group[a]], samples[group[b]]);
                let width = d_k.min(ka.len()).min(kb.len());
                let mut dot = 0.0_f32;
                let mut na = 0.0_f32;
                let mut nb = 0.0_f32;
                for i in 0..width {
                    let x = ka[i];
                    let y = kb[i];
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                }
                let denom = na.sqrt() * nb.sqrt();
                if denom > 1e-12 {
                    let c = dot / denom;
                    sum += f64::from(c * c);
                    count += 1;
                }
            }
        }
        gi = gj;
    }
    if count == 0 {
        None
    } else {
        Some((sum / count as f64) as f32)
    }
}

impl<const A: usize> LatticeMemoryG<A> {
    /// Construct a lattice sized so that a query perturbed by less than
    /// `near_miss_sigma` (in fitted-σ units of the projected corpus) stays
    /// inside the graded 3^A support. `samples` is the fit corpus
    /// (`d`-wide, ≥ 1 required).
    ///
    /// Refusals are the honest answers: [`LatticeConfigError::OverCapacity`]
    /// names the `d_k` the requested (items, radius) pair needs — widen the
    /// key width, shrink the corpus, or widen the radius — and the
    /// constructor's `byte_budget` still binds on the full slab.
    ///
    /// # Errors
    /// [`LatticeConfigError::BadNearMissRadius`] outside (0, 10],
    /// [`LatticeConfigError::EmptySample`] on no samples,
    /// [`LatticeConfigError::OverCapacity`] when the requested operating
    /// point exceeds `capacity_per_cell(d_k)`, then everything `new` refuses.
    // The plan (619 T1.5) names this constructor's signature; a params
    // struct would obscure the one-shot sizing call. The scalars are the
    // config, the sample slice is the fit corpus.
    #[allow(clippy::too_many_arguments)]
    pub fn sized(
        d: usize,
        d_k: usize,
        d_v: usize,
        seed: u64,
        samples: &[&[f32]],
        expected_items: usize,
        near_miss_sigma: f32,
        byte_budget: Option<usize>,
    ) -> Result<Self, LatticeConfigError> {
        Self::sized_impl(
            d,
            d_k,
            d_v,
            seed,
            samples,
            expected_items,
            near_miss_sigma,
            byte_budget,
            Erosion::RandomKeyAssumption,
        )
    }

    /// [`sized`](Self::sized) with the MEASURED-erosion capacity gate
    /// (Issue 938 T1.2): the same-cell `E[cos²]` estimated from the fit
    /// corpus tightens the refusal surface when the corpus' keys are more
    /// correlated than the `1/d_k` random-key assumption. Everything else —
    /// grid law, warp, config — is `sized`'s.
    ///
    /// # Errors
    /// Everything `sized` refuses, plus [`LatticeConfigError::OverCapacity`]
    /// where `sized` would have accepted (the point of the gate).
    #[allow(clippy::too_many_arguments)]
    pub fn sized_eroded(
        d: usize,
        d_k: usize,
        d_v: usize,
        seed: u64,
        samples: &[&[f32]],
        expected_items: usize,
        near_miss_sigma: f32,
        byte_budget: Option<usize>,
    ) -> Result<Self, LatticeConfigError> {
        Self::sized_impl(
            d,
            d_k,
            d_v,
            seed,
            samples,
            expected_items,
            near_miss_sigma,
            byte_budget,
            Erosion::MeasureFromFitCorpus,
        )
    }

    /// The shared sizing body (Plan 619 T1.5 + Issue 938 T1/T2).
    #[allow(clippy::too_many_arguments)]
    fn sized_impl(
        d: usize,
        d_k: usize,
        d_v: usize,
        seed: u64,
        samples: &[&[f32]],
        expected_items: usize,
        near_miss_sigma: f32,
        byte_budget: Option<usize>,
        erosion: Erosion,
    ) -> Result<Self, LatticeConfigError> {
        if !(near_miss_sigma > 0.0 && near_miss_sigma <= 10.0) {
            return Err(LatticeConfigError::BadNearMissRadius(near_miss_sigma));
        }
        if samples.is_empty() {
            return Err(LatticeConfigError::EmptySample);
        }
        // T2.2's arity-free law: L per axis, cells = L^A. The checked pow is
        // the adversarial-input guard (a tiny radius at A = 3 overflows a
        // 64-bit cell count long before it allocates).
        let side = grid_side_for(near_miss_sigma);
        let cells = match (side as usize).checked_pow(A as u32) {
            Some(c) => c,
            None => {
                return Err(LatticeConfigError::GridTooLarge {
                    side_per_axis: side,
                    axes: A,
                });
            }
        };
        let lambda = expected_items as f32 / cells as f32;
        // Axes + projections + warp: ONE construction shared by the erosion
        // estimate and the config (construction-time, never the hot path; a
        // capacity refusal below costs microseconds of wasted fit — the DRY
        // price of the measured-erosion gate reading the same corpus).
        let axes = LshAxes::<A>::new(d, [1.0; A], seed, &{
            let mut parts = Vec::with_capacity(A + 2);
            for _ in 0..A {
                parts.push(side as u64);
            }
            parts.push(d_k as u64);
            parts.push(d_v as u64);
            parts
        });
        let mut proj_rows: [Vec<f32>; A] =
            std::array::from_fn(|_| Vec::with_capacity(samples.len()));
        for s in samples {
            let c = axes.coordinates(s);
            for (axis, row) in proj_rows.iter_mut().enumerate() {
                row.push(c[axis]);
            }
        }
        let warp: [CdfWarp; A] = std::array::from_fn(|axis| CdfWarp::fit(&proj_rows[axis], side));
        let e_random = 1.0 / d_k as f32;
        let e = match erosion {
            Erosion::RandomKeyAssumption => e_random,
            Erosion::MeasureFromFitCorpus => {
                estimate_same_cell_e_cos2(&axes, Some(&warp), samples, side, d_k)
                    .unwrap_or(e_random)
                    .max(e_random)
            }
        };
        let capacity = capacity_per_cell_eroded(d_k, e, DEFAULT_TARGET_RECALL);
        if lambda > capacity {
            // The remedy is arithmetic: λ ≤ 0.35·d_k needs
            // d_k ≥ λ/0.35 — named, not gestured at.
            let required_d_k = (lambda / 0.35).ceil().max(1.0) as usize;
            return Err(LatticeConfigError::OverCapacity {
                items: expected_items,
                cells,
                capacity_per_cell: capacity,
                required_d_k,
            });
        }
        let config = LatticeConfigG {
            d,
            d_k,
            d_v,
            grid: [side; A],
            w: [1.0; A],
            seed,
            byte_budget,
            warp: Some(warp),
            bump_kernel: BumpKernel::default(),
        };
        LatticeMemoryG::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::super::LatticeMemory;
    use super::super::types::{BumpKernel, LatticeConfigError};
    use super::{capacity_per_cell, capacity_per_cell_eroded, grid_side_for, support_radius};

    /// Deterministic standard Gaussian embeddings (Box–Muller over a seeded
    /// fastrand stream — the `address.rs` draw, test-side).
    fn gaussian_samples(d: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(d);
            for _ in 0..d {
                let u1 = rng.f32().max(f32::EPSILON);
                let u2 = rng.f32();
                let mag = (-2.0 * u1.ln()).sqrt();
                v.push(mag * (core::f32::consts::TAU * u2).cos());
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn sizer_refuses_bad_radius_empty_samples_and_over_capacity() {
        let samples = gaussian_samples(8, 16, 1);
        let refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
        for bad in [0.0_f32, -0.5, f32::NAN, f32::INFINITY, 11.0] {
            assert!(
                matches!(
                    LatticeMemory::sized(8, 4, 4, 7, &refs, 100, bad, None),
                    Err(LatticeConfigError::BadNearMissRadius(_))
                ),
                "radius {bad} must be refused"
            );
        }
        assert!(matches!(
            LatticeMemory::sized(8, 4, 4, 7, &[], 100, 0.1, None),
            Err(LatticeConfigError::EmptySample)
        ));
        // Over capacity: 1000 items into a coarse 0.5σ-radius grid (8×8 =
        // 64 cells → λ = 15.6) against d_k = 4 (capacity 1.4) — refused,
        // naming the d_k the point needs.
        let big = gaussian_samples(8, 1000, 2);
        let big_refs: Vec<&[f32]> = big.iter().map(|s| s.as_slice()).collect();
        match LatticeMemory::sized(8, 4, 4, 7, &big_refs, 1000, 0.5, None) {
            Err(LatticeConfigError::OverCapacity { required_d_k, .. }) => {
                assert!(required_d_k >= 40, "remedy d_k {required_d_k} too small");
            }
            Err(other) => panic!("expected OverCapacity, got {other}"),
            Ok(_) => panic!("expected OverCapacity, but the point was accepted"),
        }
    }

    #[test]
    fn occupancy_histogram_matches_the_cdf_corrected_prediction() {
        // The T1.5 gate: write the expected items, then the measured
        // per-cell histogram must match the (CDF-corrected) Poisson
        // prediction at the SUPPORT-driven grid's λ — uniform marginals are
        // the warp's load-bearing property.
        let n = 4000;
        let samples = gaussian_samples(16, n, 0x619_5101);
        let refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
        let mut lat = LatticeMemory::sized(16, 4, 4, 42, &refs, n, 0.05, None).unwrap();
        assert_eq!(lat.config().bump_kernel, BumpKernel::Tent);
        assert!(lat.config().warp.is_some());
        // The sizing law, both directions: ⌈3.76/0.05⌉ = 76, and the
        // delivered radius is never overpromised (ceil can only tighten).
        let side = lat.config().grid[0];
        assert_eq!(side, 76); // ⌈3.76/0.05⌉
        let delivered = support_radius(side);
        assert!(
            delivered <= 0.05 && delivered > 0.05 * 0.98,
            "delivered {delivered}"
        );
        for s in &samples {
            lat.write_value(s, &[1.0; 4]);
        }
        // Per-cell occupancy over the written grid.
        let s_usize = side as usize;
        let cells = lat.config().cell_count();
        let mut occupied = vec![0_u32; cells];
        for s in &samples {
            let [r, c] = lat.primary_cell(s);
            occupied[r * s_usize + c] += 1;
        }
        let lambda = n as f32 / cells as f32;
        let predicted_ge2 = 1.0 - (-lambda).exp() * (1.0 + lambda);
        let ge2 = occupied.iter().filter(|&&o| o >= 2).count();
        let share = ge2 as f32 / cells as f32;
        assert!(
            (share - predicted_ge2).abs() <= 0.04,
            "P(≥2 items/cell) = {share:.4} vs predicted {predicted_ge2:.4} (λ = {lambda:.3})"
        );
        let max = *occupied.iter().max().unwrap();
        assert!(
            (max as f32) <= lambda + 6.0 * lambda.max(1.0).sqrt(),
            "a wrapped cell holds {max} items — past the Poisson tail at λ = {lambda:.3}"
        );
        // Everything claimed exactly once per written cell; occupancy count
        // matches the histogram's nonzero cells.
        let nonzero = occupied.iter().filter(|&&o| o > 0).count();
        assert_eq!(lat.written_cells(), nonzero);
        // And the read stays graded on the sized lattice (warp didn't break
        // the near-miss property) — at the support radius the lattice was
        // SIZED for, not an arbitrary one.
        let mut q = samples[0].clone();
        let radius = support_radius(side) * 0.5;
        let mut rng = fastrand::Rng::with_seed(0x619_0619);
        for v in q.iter_mut() {
            *v += radius * (rng.f32() * 2.0 - 1.0);
        }
        let mut out = [0.0_f32; 4];
        lat.read_blend(&q, &mut out);
        assert!(
            out.iter().sum::<f32>() > 0.0,
            "sized-lattice near-miss read a cliff"
        );
    }

    #[test]
    fn unwarped_grid_overloads_the_center() {
        // The contrast that proves the correction does something: the SAME
        // embeddings on the SAME grid without the warp pile into the central
        // columns — the overload the Gaussian-projection geometry predicts.
        let n = 4000;
        let samples = gaussian_samples(16, n, 0x619_5101);
        let refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
        let warped = LatticeMemory::sized(16, 4, 4, 42, &refs, n, 0.05, None).unwrap();
        let mut cfg = warped.config().clone();
        cfg.warp = None;
        let mut raw = LatticeMemory::new(cfg).unwrap();
        for s in &samples {
            raw.write_value(s, &[1.0; 4]);
        }
        let count = |lat: &LatticeMemory| {
            let s = lat.config().grid[0] as usize;
            let mut occ = vec![0_u32; lat.config().cell_count()];
            for sample in &samples {
                let [r, c] = lat.primary_cell(sample);
                occ[r * s + c] += 1;
            }
            *occ.iter().max().unwrap()
        };
        let (raw_max, wrapped_max) = (count(&raw), count(&warped));
        assert!(
            raw_max >= 3 * wrapped_max.max(1),
            "unwrapped peak {raw_max} must dwarf the wrapped peak {wrapped_max}"
        );
        assert!(
            raw_max >= 12,
            "unwrapped peak {raw_max} below the overload floor"
        );
    }

    #[test]
    fn capacity_law_matches_the_erosion_derivation() {
        // The pinned first-order estimate: d_k = 32 → 11 associations per
        // cell (ln2·32 = 22, halved for the cross-term noise floor).
        assert!((capacity_per_cell(32) - 11.2).abs() < 1e-6);
        assert!((capacity_per_cell(4) - 1.4).abs() < 1e-6);
    }

    // ── Issue 938 T1 — the measured-erosion parameterization ─────────────

    #[test]
    fn eroded_capacity_defaults_are_the_pinned_law_bit_for_bit() {
        // T1.1's contract: at (e = 1/d_k, target = ½) the parameterized law
        // IS the shipped `0.35·d_k`, bit-for-bit, at every realistic width.
        for d_k in [1_usize, 2, 3, 4, 7, 8, 16, 31, 32, 64, 100, 128] {
            let e = 1.0 / d_k as f32;
            assert_eq!(
                capacity_per_cell_eroded(d_k, e, 0.5).to_bits(),
                capacity_per_cell(d_k).to_bits(),
                "d_k={d_k}: the default arm must be the pin bit-for-bit"
            );
        }
        // The erosion axis moves capacity by the survival ratio. At
        // e = 0.083 (the μLattice two-point figure — a HYPOTHESIS input for
        // this pin, never adopted: T1.3 measures OUR fixtures), d_k = 32:
        // 0.35/0.083 ≈ 4.2 associations, not 11.2.
        let cap = capacity_per_cell_eroded(32, 0.083, 0.5);
        assert!(
            (cap - 4.216).abs() < 5e-3,
            "eroded capacity {cap} off the law"
        );
        // Target recall moves it the other way: demanding 0.9 survival
        // tightens the load (ln(1/0.9) < ln 2).
        let strict = capacity_per_cell_eroded(32, 1.0 / 32.0, 0.9);
        assert!(strict < capacity_per_cell(32) && strict > 0.0);
    }

    #[test]
    fn grid_side_law_matches_the_sizer_and_arity() {
        // T2.2's extracted law, both directions.
        assert_eq!(grid_side_for(0.05), 76); // ⌈3.76/0.05⌉
        assert_eq!(grid_side_for(10.0), 1);
        let side = grid_side_for(0.05);
        let delivered = support_radius(side);
        assert!(delivered <= 0.05 && delivered > 0.05 * 0.98);
    }

    #[test]
    fn eroded_sizer_tightens_on_correlated_keys_only() {
        // T1.2 end-to-end: a CLUSTERED corpus (same-cell keys far more
        // correlated than 1/d_k) is refused by `sized_eroded` where plain
        // `sized` accepts; a RANDOM corpus gets the identical verdict from
        // both (the estimate stays under the assumption — no behavior
        // change without measured cause).
        let d = 16;
        let d_k = 8;
        let mut rng = fastrand::Rng::with_seed(0x938_0e10);
        // Clustered: one shared center + small per-key noise ⇒ same-cell
        // cosines ≈ 0.5–0.9 ⇒ cos² ≫ 1/8.
        let center: Vec<f32> = (0..d).map(|_| gaussian(&mut rng)).collect();
        let mut clustered: Vec<Vec<f32>> = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut k: Vec<f32> = center.iter().map(|&c| c * 3.0).collect();
            for k_i in k.iter_mut() {
                *k_i += rng.f32() * 2.0 - 1.0;
            }
            let n: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut k {
                *x /= n;
            }
            clustered.push(k);
        }
        let clustered_refs: Vec<&[f32]> = clustered.iter().map(|s| s.as_slice()).collect();
        // 64 items into a 0.5σ grid (side 8 → 64 cells → λ = 1) at d_k = 8:
        // capacity 2.8 — `sized` accepts.
        let plain = LatticeMemory::sized(d, d_k, 4, 7, &clustered_refs, 64, 0.5, None);
        assert!(
            plain.is_ok(),
            "random-key law must accept the clustered point"
        );
        // `sized_eroded` measures the correlation and refuses.
        match LatticeMemory::sized_eroded(d, d_k, 4, 7, &clustered_refs, 64, 0.5, None) {
            Err(LatticeConfigError::OverCapacity {
                capacity_per_cell: cap,
                ..
            }) => {
                assert!(
                    cap < 0.35 * d_k as f32,
                    "measured erosion must TIGHTEN the capacity: {cap}"
                );
            }
            Err(other) => panic!("expected OverCapacity, got {other}"),
            Ok(_) => panic!("eroded sizer accepted a correlated corpus — the estimate is blind"),
        }
        // Random corpus: both postures agree (the estimate ≤ 1/d_k). 128
        // items over the 64-cell grid → λ = 2.0, under the d_k = 8 capacity
        // of 2.8 — inside BOTH gates' acceptance.
        let random = gaussian_samples(d, 128, 0x938_0e11);
        let random_refs: Vec<&[f32]> = random.iter().map(|s| s.as_slice()).collect();
        assert!(LatticeMemory::sized(d, d_k, 4, 7, &random_refs, 128, 0.5, None).is_ok());
        assert!(
            LatticeMemory::sized_eroded(d, d_k, 4, 7, &random_refs, 128, 0.5, None).is_ok(),
            "random keys must not trip the eroded gate"
        );
    }

    /// One Gaussian draw (the `address.rs` shape, test-side).
    fn gaussian(rng: &mut fastrand::Rng) -> f32 {
        let u1 = rng.f32().max(f32::EPSILON);
        let u2 = rng.f32();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}
