//! Modelless codebook construction via Lloyd's k-means algorithm.
//!
//! This module implements the offline (per-fit, not per-transition) codebook
//! construction path for the OTF-LAM primitive (Plan 375 Phase 2):
//!
//! 1. [`fit_codebook_kmeans_into`] — deterministic Lloyd's algorithm with
//!    k-means++ initialization, no gradient descent.
//! 2. [`EffectCodebook::from_observed_transitions`] — patchify observed
//!    transitions and fit a codebook.
//!
//! # Modelless contract
//!
//! K-means is a **deterministic iterative refinement** algorithm
//! (Lloyd 1957), NOT gradient descent. It converges to a local optimum
//! from a fixed seed. This satisfies AGENTS.md's "modelless-first mandate"
//! (the only weight mutations allowed at runtime are freeze/thaw,
//! raw/lora hot-swap, and latent-space updates — k-means falls under
//! "freeze/thaw" because the codebook is frozen after fit).
//!
//! # Allocation discipline
//!
//! The fit path is **offline** (runs once per codebook construction, not
//! per transition), so allocation is acceptable here. The inference hot
//! path (`kernel.rs`) is the zero-allocation half. This module allocates
//! `Vec`s for centroid accumulators and the assignment scratch — that's
//! the right trade-off (allocation cost amortized over `max_iters`).

use super::types::EffectCodebook;

/// A simple deterministic PRNG for k-means initialization.
///
/// Implementation of SplitMix64 — same algorithm as fastrand's underlying
/// generator when seeded. We implement it here to keep this module
/// dependency-free (no fastrand dep required for k-means alone).
///
/// **Why not use `fastrand::Rng`?** The crate already depends on fastrand
/// (see `Cargo.toml` line 13), so callers who prefer the crate RNG can
/// pass their own `seed`-seeded `fastrand::Rng`. This built-in PRNG keeps
/// the k-means code path deterministic and isolated.
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for full f64 precision.
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Squared Euclidean distance between two D-dim points.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

/// Fit a codebook of `k` centroids from a set of D-dim patches using
/// Lloyd's algorithm with k-means++ initialization.
///
/// Writes the resulting centroids (row-major `k × D`) into
/// `out.centroids[..k*D]`. The caller must ensure `k ≤ K` (the codebook's
/// const generic size); excess rows `[k*D..K*D]` are left untouched
/// (callers should `zeroed()` the codebook first if they want clean
/// unused rows).
///
/// # Algorithm
///
/// 1. **k-means++ initialization** — pick the first centroid uniformly at
///    random; pick each subsequent centroid with probability proportional
///    to D²(distance) from the nearest already-picked centroid. This
///    spreads initial centroids across the data and dramatically improves
///    final-cluster quality vs random init.
/// 2. **Lloyd iteration** — repeat until convergence (no assignment
///    changes) or `max_iters`:
///    - Assign each patch to nearest centroid.
///    - Recompute centroids as the mean of assigned patches.
///    - If a centroid has zero assigned patches, reinitialize it to a
///      random patch (prevents dead centroids).
///
/// Deterministic from `seed`. No gradient descent.
///
/// # Panics
///
/// Debug-mode panic if `k == 0`, `k > K`, `patches` is empty, or any
/// patch length differs from the first patch's length. The first patch's
/// length determines `D` (must match the codebook's `D` const generic).
pub fn fit_codebook_kmeans_into<const K: usize, const D: usize>(
    patches: &[&[f32]],
    k: usize,
    seed: u64,
    max_iters: usize,
    out: &mut EffectCodebook<K, D>,
) {
    debug_assert!(k > 0, "k must be > 0");
    debug_assert!(k <= K, "k ({k}) must be ≤ K ({K})");
    debug_assert!(!patches.is_empty(), "patches must be non-empty");
    debug_assert!(max_iters > 0, "max_iters must be > 0");

    let n = patches.len();
    let d = patches[0].len();
    debug_assert_eq!(d, D, "patch dim {d} != codebook D={D}");
    for p in patches {
        debug_assert_eq!(p.len(), d, "all patches must have the same dim");
    }

    let mut rng = SplitMix64::new(seed);

    // ── k-means++ initialization ───────────────────────────────────────
    // Pick the first centroid uniformly at random.
    let first_idx = (rng.next_u64() as usize) % n;
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut c0 = vec![0.0f32; D];
    c0.copy_from_slice(&patches[first_idx][..D]);
    centroids.push(c0);

    // Squared distance from each patch to its nearest chosen centroid.
    let mut d2_nearest: Vec<f64> = (0..n)
        .map(|i| sq_dist(patches[i], &centroids[0]) as f64)
        .collect();

    for _ in 1..k {
        let total: f64 = d2_nearest.iter().sum();
        if total <= 0.0 {
            // All patches coincide with chosen centroids — pick random.
            let idx = (rng.next_u64() as usize) % n;
            let mut c = vec![0.0f32; D];
            c.copy_from_slice(&patches[idx][..D]);
            centroids.push(c);
        } else {
            // Sample proportional to D².
            let r = rng.next_f64() * total;
            let mut acc = 0.0f64;
            let mut chosen = n - 1;
            for (i, &d2) in d2_nearest.iter().enumerate() {
                acc += d2;
                if acc >= r {
                    chosen = i;
                    break;
                }
            }
            let mut c = vec![0.0f32; D];
            c.copy_from_slice(&patches[chosen][..D]);
            centroids.push(c);
        }
        // Update d2_nearest with the new centroid.
        let new_c = centroids.last().unwrap();
        for i in 0..n {
            let d2 = sq_dist(patches[i], new_c) as f64;
            if d2 < d2_nearest[i] {
                d2_nearest[i] = d2;
            }
        }
    }

    debug_assert_eq!(centroids.len(), k);

    // ── Lloyd iteration ────────────────────────────────────────────────
    let mut assignments: Vec<usize> = vec![0; n];
    let mut sums: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; D]).collect();
    let mut counts: Vec<usize> = vec![0; k];

    for _iter in 0..max_iters {
        let mut changed = false;

        // Assign step.
        for i in 0..n {
            let mut best_k = 0usize;
            let mut best_d2 = f32::INFINITY;
            for (kk, centroid) in centroids.iter().enumerate().take(k) {
                let d2 = sq_dist(patches[i], centroid);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_k = kk;
                }
            }
            if assignments[i] != best_k {
                assignments[i] = best_k;
                changed = true;
            }
        }

        // Update step.
        for s in sums.iter_mut() {
            s.fill(0.0);
        }
        counts.fill(0);
        for i in 0..n {
            let kk = assignments[i];
            counts[kk] += 1;
            for d in 0..D {
                sums[kk][d] += patches[i][d] as f64;
            }
        }

        // Recompute centroids; reinit dead centroids to the patch with
        // the largest distance from any current centroid (k-means||-style).
        // This avoids the random-reinit failure mode where a dead centroid
        // gets placed inside an existing cluster.
        if counts.contains(&0) {
            // Recompute nearest-centroid distance for each patch.
            for i in 0..n {
                let mut best_d2 = f32::INFINITY;
                for kk in 0..k {
                    if counts[kk] == 0 {
                        continue;
                    }
                    let d2 = sq_dist(patches[i], &centroids[kk]);
                    if d2 < best_d2 {
                        best_d2 = d2;
                    }
                }
                d2_nearest[i] = best_d2 as f64;
            }
        }
        for kk in 0..k {
            if counts[kk] == 0 {
                // Pick the patch with the largest nearest-centroid distance.
                let (far_idx, _far_d2) = d2_nearest
                    .iter()
                    .enumerate()
                    // cmp_for_max: NaN can never win the farthest-code selection.
                    .max_by(|(_, a), (_, b)| crate::float_order::cmp_for_max_f64(**a, **b))
                    .map_or((0, 0.0), |(i, &d)| (i, d));
                centroids[kk].copy_from_slice(&patches[far_idx][..D]);
                // Update d2_nearest: this patch is now at distance 0.
                d2_nearest[far_idx] = 0.0;
                changed = true;
            } else {
                let inv = 1.0f64 / counts[kk] as f64;
                for d in 0..D {
                    centroids[kk][d] = sums[kk][d] as f32 * inv as f32;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // ── Copy into the output codebook ─────────────────────────────
    for (dst, src) in out.centroids.iter_mut().zip(centroids.iter()).take(k) {
        for (dst_slot, &src_val) in dst.iter_mut().zip(src.iter()).take(D) {
            *dst_slot = src_val;
        }
    }
}

/// Patchify a 1D motion signal `o_t` of length `L` into non-overlapping
/// `patch_size`-element patches. Returns a `Vec<&[f32]>` of length `L / patch_size`.
///
/// Panics (debug) if `L % patch_size != 0`.
pub fn patchify_1d(o_t: &[f32], patch_size: usize) -> Vec<&[f32]> {
    debug_assert!(patch_size > 0, "patch_size must be > 0");
    debug_assert_eq!(
        o_t.len() % patch_size,
        0,
        "signal length {} not divisible by patch_size {}",
        o_t.len(),
        patch_size
    );
    let n_patches = o_t.len() / patch_size;
    let mut out = Vec::with_capacity(n_patches);
    for i in 0..n_patches {
        out.push(&o_t[i * patch_size..(i + 1) * patch_size]);
    }
    out
}

/// Compute the motion input `o_t` from a transition pair `(x_t, x_{t+1})`.
///
/// Per Research 374 §10 code verification, the paper's default motion input
/// is **acceleration** (second-order), but velocity is also supported.
/// We use velocity here for the simpler modelless baseline:
/// `o_t = x_{t+1} − x_t`.
///
/// Writes into `out` (caller-allocated). Zero allocation.
///
/// # Panics
///
/// Debug-mode panic if `x_t.len() != x_next.len()` or `out.len() < x_t.len()`.
pub fn motion_input_velocity_into(x_t: &[f32], x_next: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x_t.len(), x_next.len());
    debug_assert!(out.len() >= x_t.len());
    let n = x_t.len();
    let mut i = 0;
    while i + 4 <= n {
        out[i] = x_next[i] - x_t[i];
        out[i + 1] = x_next[i + 1] - x_t[i + 1];
        out[i + 2] = x_next[i + 2] - x_t[i + 2];
        out[i + 3] = x_next[i + 3] - x_t[i + 3];
        i += 4;
    }
    while i < n {
        out[i] = x_next[i] - x_t[i];
        i += 1;
    }
}

impl<const K: usize, const D: usize> EffectCodebook<K, D> {
    /// Construct a frozen codebook from observed transitions.
    ///
    /// For each `(x_t, x_{t+1})` pair:
    /// 1. Compute motion input `o_t = x_{t+1} − x_t` (velocity).
    /// 2. Patchify `o_t` into `patch_size`-element blocks.
    /// 3. Collect all patches and run k-means with `k` clusters.
    ///
    /// Returns the frozen codebook. Deterministic from `seed`.
    ///
    /// # Allocation
    ///
    /// This is an offline construction path — allocations are expected
    /// here. The hot path (`kernel.rs`) is the zero-allocation half.
    ///
    /// # Panics
    ///
    /// Debug-mode panic if `patch_size != D` (each patch must fit a
    /// codebook row), or if any transition's length isn't divisible by
    /// `patch_size`, or `k > K`.
    pub fn from_observed_transitions(
        transitions: &[(Vec<f32>, Vec<f32>)],
        patch_size: usize,
        k: usize,
        seed: u64,
        max_iters: usize,
    ) -> Self {
        debug_assert_eq!(patch_size, D, "patch_size {patch_size} != codebook D={D}");
        debug_assert!(k <= K, "k ({k}) > K ({K})");

        // Pre-pass: count total patches.
        let total_patches: usize = transitions
            .iter()
            .map(|(a, b)| {
                debug_assert_eq!(a.len(), b.len());
                debug_assert_eq!(a.len() % patch_size, 0);
                a.len() / patch_size
            })
            .sum();

        // Collect patches into a flat buffer + slice table.
        let mut flat: Vec<f32> = Vec::with_capacity(total_patches * D);
        let mut slices: Vec<&[f32]> = Vec::with_capacity(total_patches);
        let mut scratch_o = vec![0.0f32; transitions.first().map_or(0, |(a, _)| a.len())];
        for (x_t, x_next) in transitions.iter() {
            motion_input_velocity_into(x_t, x_next, &mut scratch_o);
            for i in 0..(x_t.len() / patch_size) {
                let start = flat.len();
                flat.extend_from_slice(&scratch_o[i * patch_size..(i + 1) * patch_size]);
                // SAFETY: we just extended flat by patch_size elements; the
                // slice start..start+patch_size is valid for the lifetime of
                // `flat` (which outlives this closure).
                slices.push(unsafe {
                    let p = flat.as_ptr().add(start);
                    std::slice::from_raw_parts(p, patch_size)
                });
            }
        }

        let mut out = Self::zeroed();
        fit_codebook_kmeans_into(&slices, k, seed, max_iters, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T2.3 — k-means on synthetic 2D transitions. Verify:
    /// - All K centroids are distinct (no collapse).
    /// - Deterministic (same seed → same centroids).
    /// - Reconstruction MSE < identity baseline (predict `o_t = 0`).
    #[test]
    fn kmeans_synthetic_clusters_distinct_and_deterministic() {
        // 100 transitions, each patch is 4-dim. 3 well-separated clusters.
        let mut rng = SplitMix64::new(42);
        let mut transitions: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(100);
        for _ in 0..100 {
            let cluster = (rng.next_u64() as usize) % 3;
            let center: [f32; 4] = match cluster {
                0 => [10.0, 10.0, 10.0, 10.0],
                1 => [-10.0, 10.0, -10.0, 10.0],
                _ => [10.0, -10.0, 10.0, -10.0],
            };
            let x_t: Vec<f32> = (0..4).map(|_| rng.next_f64() as f32 * 0.5).collect();
            let x_next: Vec<f32> = x_t
                .iter()
                .enumerate()
                .map(|(i, &x)| x + center[i] + (rng.next_f64() as f32 - 0.5) * 0.5)
                .collect();
            transitions.push((x_t, x_next));
        }

        // Fit K=3, D=4 codebook (3 centroids on 3 true clusters —
        // well-separated, no over-clustering).
        let cb_a: EffectCodebook<3, 4> =
            EffectCodebook::from_observed_transitions(&transitions, 4, 3, 12345, 50);
        let cb_b: EffectCodebook<3, 4> =
            EffectCodebook::from_observed_transitions(&transitions, 4, 3, 12345, 50);

        // Determinism: same seed → bit-identical centroids.
        assert_eq!(cb_a.centroids, cb_b.centroids, "k-means not deterministic");

        // All 3 centroids must be distinct (no two identical rows).
        for i in 0..3 {
            for j in (i + 1)..3 {
                let a = cb_a.centroid(i);
                let b = cb_a.centroid(j);
                let dist = sq_dist(a, b).sqrt();
                assert!(dist > 0.5, "centroids {i} and {j} collapsed: dist={dist}");
            }
        }

        // Reconstruction MSE < identity baseline (predict o_t = 0).
        let mut id_mse = 0.0f64;
        let mut id_count = 0u64;
        let mut rec_mse = 0.0f64;
        let mut rec_count = 0u64;
        let mut scratch_o = vec![0.0f32; 4];
        for (x_t, x_next) in transitions.iter() {
            motion_input_velocity_into(x_t, x_next, &mut scratch_o);
            // Identity: predict o_t = 0.
            for v in scratch_o.iter() {
                id_mse += (*v as f64) * (*v as f64);
                id_count += 1;
            }
            // Reconstruction: nearest-centroid quantization error.
            let mut best_d2 = f32::INFINITY;
            for k in 0..3 {
                let d2 = sq_dist(&scratch_o, cb_a.centroid(k));
                if d2 < best_d2 {
                    best_d2 = d2;
                }
            }
            rec_mse += best_d2 as f64;
            rec_count += 1;
        }
        let id_avg = id_mse / id_count as f64;
        let rec_avg = rec_mse / rec_count as f64;
        assert!(
            rec_avg < id_avg,
            "reconstruction MSE {rec_avg:.4} should be < identity baseline {id_avg:.4}"
        );
    }

    /// k-means++ initialization is deterministic from seed.
    #[test]
    fn kmeans_deterministic_from_seed() {
        let patches: Vec<&[f32]> = vec![
            &[1.0, 1.0],
            &[1.1, 0.9],
            &[0.9, 1.1],
            &[-1.0, -1.0],
            &[-1.1, -0.9],
            &[-0.9, -1.1],
        ];
        let mut a: EffectCodebook<2, 2> = EffectCodebook::zeroed();
        let mut b: EffectCodebook<2, 2> = EffectCodebook::zeroed();
        fit_codebook_kmeans_into(&patches, 2, 999, 20, &mut a);
        fit_codebook_kmeans_into(&patches, 2, 999, 20, &mut b);
        assert_eq!(a.centroids, b.centroids, "k-means should be deterministic");

        // Different seed → likely different (could be the same if both find
        // the same optimum, but for this clean 2-cluster case both should
        // converge to the same answer regardless of init).
        let mut c: EffectCodebook<2, 2> = EffectCodebook::zeroed();
        fit_codebook_kmeans_into(&patches, 2, 1, 20, &mut c);
        // Either identical or symmetric — both acceptable for a clean
        // 2-cluster problem. We just check it ran without panicking.
        let _ = c;
    }

    /// `patchify_1d` produces correct non-overlapping patches.
    #[test]
    fn patchify_1d_correct() {
        let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let patches = patchify_1d(&signal, 2);
        assert_eq!(patches.len(), 3);
        assert_eq!(patches[0], &[1.0, 2.0]);
        assert_eq!(patches[1], &[3.0, 4.0]);
        assert_eq!(patches[2], &[5.0, 6.0]);
    }

    /// `motion_input_velocity_into` computes `x_next - x_t`.
    #[test]
    fn motion_input_velocity_correct() {
        let x_t = [1.0f32, 2.0, 3.0];
        let x_next = [1.5f32, 2.5, 2.0];
        let mut out = [0.0f32; 3];
        motion_input_velocity_into(&x_t, &x_next, &mut out);
        assert_eq!(out, [0.5, 0.5, -1.0]);
    }

    /// T2.4 — cross-carrier transfer smoke test.
    ///
    /// Fit codebook on one motion pattern, evaluate reconstruction MSE
    /// on a different motion pattern. Verify the codebook produces
    /// reasonable (bounded) reconstruction on out-of-distribution data —
    /// not a paper-faithful transfer experiment (the G3 gate in
    /// bench_375 will compare factorized vs monolithic on the paper's
    /// actual transfer setup).
    #[test]
    fn cross_carrier_transfer_is_bounded() {
        // "Carrier A": motion centered around [+5, +5, +5, +5].
        let mut rng_a = SplitMix64::new(7);
        let train: Vec<(Vec<f32>, Vec<f32>)> = (0..50)
            .map(|_| {
                let x_t: Vec<f32> = (0..4).map(|_| rng_a.next_f64() as f32).collect();
                let x_next: Vec<f32> = x_t
                    .iter()
                    .map(|&x| x + 5.0 + (rng_a.next_f64() as f32 - 0.5))
                    .collect();
                (x_t, x_next)
            })
            .collect();

        // "Carrier B": motion centered around [+4.8, +5.1, +4.9, +5.2] —
        // similar magnitude, slightly perturbed direction. This is the
        // realistic transfer scenario (same domain, slightly different
        // parameters), not the adversarial case of radically different
        // directions.
        let mut rng_b = SplitMix64::new(11);
        let test_transitions: Vec<(Vec<f32>, Vec<f32>)> = (0..50)
            .map(|_| {
                let x_t: Vec<f32> = (0..4).map(|_| rng_b.next_f64() as f32).collect();
                let delta = [4.8f32, 5.1, 4.9, 5.2];
                let x_next: Vec<f32> = x_t
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x + delta[i] + (rng_b.next_f64() as f32 - 0.5))
                    .collect();
                (x_t, x_next)
            })
            .collect();

        // Fit codebook on train.
        let cb: EffectCodebook<4, 4> =
            EffectCodebook::from_observed_transitions(&train, 4, 4, 42, 30);

        // Source MSE: reconstruction error on train.
        let source_mse = mean_recon_mse(&cb, &train, 4);
        // Target MSE: reconstruction error on test.
        let target_mse = mean_recon_mse(&cb, &test_transitions, 4);

        // For similar-motion carriers, transfer should be near-lossless.
        // The modelless k-means codebook generalizes within a motion
        // family — this is the paper's Table 1 finding (simple motion
        // inputs transfer well across morphologies within the same
        // motion class).
        assert!(
            target_mse < 1.0,
            "target MSE {target_mse:.4} should be < 1.0 for similar-motion transfer"
        );
        // And the source MSE should be small (k-means fit the train data).
        assert!(
            source_mse < 1.0,
            "source MSE {source_mse:.4} should be < 1.0 after k-means fit"
        );
    }

    fn mean_recon_mse<const K: usize, const D: usize>(
        cb: &EffectCodebook<K, D>,
        transitions: &[(Vec<f32>, Vec<f32>)],
        k_active: usize,
    ) -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0u64;
        let mut scratch_o = vec![0.0f32; D];
        for (x_t, x_next) in transitions.iter() {
            motion_input_velocity_into(x_t, x_next, &mut scratch_o);
            let mut best_d2 = f32::INFINITY;
            for k in 0..k_active.min(K) {
                let d2 = sq_dist(&scratch_o, cb.centroid(k));
                if d2 < best_d2 {
                    best_d2 = d2;
                }
            }
            sum += best_d2 as f64;
            count += 1;
        }
        sum / count as f64
    }

    /// `SplitMix64` is deterministic.
    #[test]
    fn splitmix64_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
