//! TEMP — Perturbed-Loss-Vector Diversity Fingerprint (Plan 341, Research 323).
//!
//! Distilled from Jin et al., *"Reasoning Quality Emerges Early: Data Curation
//! for Reasoning Models"* ([arXiv:2606.26797](https://arxiv.org/abs/2606.26797),
//! ICML 2026). Research note:
//! [`katgpt-rs/.research/323_TEMP_Perturbed_Loss_Vector_Fingerprint.md`].
//!
//! # The primitive
//!
//! Given two latent-state snapshots `S_0, S_1` and a candidate experience set,
//! compute a **perturbed-loss-vector diversity fingerprint** per candidate and
//! select the K-subset with maximal spread. Theorem 3.1 (modellessly reframed):
//! similar loss vectors across K extrapolated checkpoints ⇒ similar gradients
//! along `v = S_1 − S_0` during the next weight-mutation cycle (freeze/thaw
//! swap, consolidation tick, LoRA hot-swap).
//!
//! # Modelless invariant (AGENTS.md)
//!
//! No training, no gradients, no backprop. The checkpoints are committed
//! shards; the extrapolated snapshots are deterministic linear combinations;
//! the loss is a per-step NLL on a short prefix; the bound is pure arithmetic.
//!
//! # Latent vs raw boundary
//!
//! Loss vectors are local latent (per-candidate, not synced). The selected
//! index set and the aggregate Lipschitz bound are deterministic raw scalars
//! safe to sync/replay/quorum-commit (Research 323 §5).
//!
//! # Sigmoid, never softmax
//!
//! Per `AGENTS.md`: the [`LossKernel`] implementations use sigmoid-gated
//! dot projections (see the test-fixture kernel and the downstream
//! `RavenSlotLossKernel` in riir-neuron-db Plan 005). No softmax over the
//! loss vectors — they are ranking signals, not a probability distribution.

#![allow(clippy::needless_range_loop)]

// ──────────────────────────────────────────────────────────────────────────
// Tunables
// ──────────────────────────────────────────────────────────────────────────

/// 1/√2 — the irreducible factor in the Theorem 3.1 bound.
/// `(2δ/λ + G)(1/√2 + τ) + C_H·ε`.
const INV_SQRT_2: f32 = 0.7071067811865476_f32;

// ──────────────────────────────────────────────────────────────────────────
// LossKernel trait (T1.2)
// ──────────────────────────────────────────────────────────────────────────

/// Per-step negative-log-probability kernel at a given parameter snapshot.
///
/// Implementors compose existing infrastructure:
/// - `ac_prefix::ConditionalLogprob` (Plan 313) — token-level NLL for text traces.
/// - HLA surprise wrapper (`sense::reconstruction`) — per-tick HLA surprise.
/// - Functor-coherence wrapper, KARC residual wrapper — future composition
///   (Plan 341 Phase 3, all deferred).
/// - `RavenSlotLossKernel` (riir-neuron-db Plan 005) — shard-style dot surprise.
///
/// `theta` is the flattened parameter snapshot (e.g. `style_weights[64]`).
/// `z_prefix` is the first N steps of the candidate experience. The kernel
/// returns `L_z(theta) = sum_{t<N} -log p(z_prefix[t] | z_prefix[<t], theta)`.
pub trait LossKernel {
    /// Compute the short-prefix loss of candidate `z` at snapshot `theta`.
    fn short_prefix_loss(&self, theta: &[f32], z_prefix: &[f32]) -> f32;
}

// ──────────────────────────────────────────────────────────────────────────
// Extrapolated snapshot schedule (T1.3)
// ──────────────────────────────────────────────────────────────────────────

/// Directionally-extrapolated snapshot schedule (deterministic, BLAKE3-reproducible).
///
/// Produces K snapshots `theta_j = S_0 + lambda_j * (1 + xi_j) * v` where
/// `v = S_1 - S_0`, `lambda_j` is the caller-provided schedule, and `xi_j`
/// is deterministic zero-mean uniform noise in `[-noise_sigma, +noise_sigma]`
/// derived from a BLAKE3 hash of `noise_seeds[j]` (paper Eq. 5 modelless reframe).
///
/// `noise_sigma = 0.0` disables noise → pure linear extrapolation along `v`.
///
/// # Allocation discipline (G4)
///
/// Writes into caller-provided `out: &mut [Vec<f32>]` (len == k). Each inner
/// `Vec` is resized to `s0.len()` if needed — alloc-free if pre-capacitized.
/// Callers on the hot path should pre-allocate: `out = vec![vec![0.0; d]; k]`.
///
/// # Panics
///
/// Asserts `s0.len() == s1.len()`, `lambda_schedule.len() == out.len()`,
/// `noise_seeds.len() == out.len()`.
pub fn extrapolated_snapshot_schedule(
    s0: &[f32],
    s1: &[f32],
    lambda_schedule: &[f32],
    noise_seeds: &[u64],
    noise_sigma: f32,
    out: &mut [Vec<f32>],
) {
    assert_eq!(s0.len(), s1.len(), "s0 and s1 must have same dimension");
    assert_eq!(
        lambda_schedule.len(),
        out.len(),
        "lambda_schedule and out must have length k"
    );
    assert_eq!(
        noise_seeds.len(),
        out.len(),
        "noise_seeds and out must have length k"
    );

    let d = s0.len();

    for (j, theta_j) in out.iter_mut().enumerate() {
        if theta_j.len() != d {
            theta_j.resize(d, 0.0);
        }
        let xi_j = if noise_sigma == 0.0 {
            0.0
        } else {
            blake3_noise(noise_seeds[j], noise_sigma)
        };
        let coeff = lambda_schedule[j] * (1.0 + xi_j);
        let theta = theta_j.as_mut_slice();
        for i in 0..d {
            // v_i = s1[i] - s0[i]; theta[i] = s0[i] + coeff * v_i
            theta[i] = s0[i] + coeff * (s1[i] - s0[i]);
        }
    }
}

/// Deterministic zero-mean uniform noise in `[-sigma, +sigma]` from a BLAKE3
/// hash of `seed`. Same `(seed, sigma)` ⇒ same output, bit-identical across
/// platforms and runs (quorum-reproducibility, G4).
#[inline]
fn blake3_noise(seed: u64, sigma: f32) -> f32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    let u = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    // Map u32 uniformly to [-1, 1], then scale by sigma.
    let normalized = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
    normalized * sigma
}

// ──────────────────────────────────────────────────────────────────────────
// Perturbed loss vector (T1.4)
// ──────────────────────────────────────────────────────────────────────────

/// Compute the perturbed-loss vector `L_z` for candidate `z_prefix` across the
/// K extrapolated snapshots.
///
/// Calls `kernel.short_prefix_loss(theta_j, z_prefix)` for each `j` and writes
/// the result into `out: &mut [f32]` (len == k). Zero-allocation.
///
/// # Panics
///
/// Asserts `theta_schedule.len() == out.len()`.
pub fn perturbed_loss_vector<L: LossKernel + ?Sized>(
    kernel: &L,
    theta_schedule: &[Vec<f32>],
    z_prefix: &[f32],
    out: &mut [f32],
) {
    assert_eq!(
        theta_schedule.len(),
        out.len(),
        "theta_schedule and out must have length k"
    );
    for (j, theta_j) in theta_schedule.iter().enumerate() {
        out[j] = kernel.short_prefix_loss(theta_j.as_slice(), z_prefix);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Lipschitz gradient bound (T1.5) — Theorem 3.1
// ──────────────────────────────────────────────────────────────────────────

/// Lipschitz bound from Theorem 3.1 (modelless gradient-diversity proxy).
///
/// Given `delta = ||L_{z1} - L_{z2}||_inf` across K checkpoints, returns the
/// upper bound on `|<grad L_{z1} - grad L_{z2}, v>|` during the next
/// weight-mutation cycle along `v = S_1 - S_0`:
///
/// ```text
/// bound = (2*delta/lambda + G) * (1/sqrt(2) + tau) + C_H * epsilon
/// ```
///
/// where `lambda` is the snapshot step size, `G` is the per-checkpoint
/// gradient norm bound, `tau` is the directional-noise tolerance, `C_H` is
/// the Hessian Lipschitz constant, and `epsilon` is the curvature slack.
///
/// # The irreducible floor
///
/// At `delta = 0` (identical loss vectors), the bound reduces to
/// `G*(1/sqrt(2) + tau) + C_H*epsilon` — the floor set by gradient norm,
/// noise tolerance, and curvature. Two candidates with identical fingerprints
/// would still induce gradients differing by up to this floor along `v`.
#[inline]
pub fn lipschitz_gradient_bound(
    delta: f32,
    lambda: f32,
    g: f32,
    tau: f32,
    c_h: f32,
    epsilon: f32,
) -> f32 {
    debug_assert!(lambda > 0.0, "lambda must be positive");
    (2.0 * delta / lambda + g) * (INV_SQRT_2 + tau) + c_h * epsilon
}

// ──────────────────────────────────────────────────────────────────────────
// Pairwise bound matrix (T1.6)
// ──────────────────────────────────────────────────────────────────────────

/// Pairwise Lipschitz bound matrix over `n` candidates.
///
/// For each pair `(i, j)`, computes `delta_ij = ||L_i - L_j||_inf` and writes
/// `lipschitz_gradient_bound(delta_ij, ...)` into `out[i*n + j]`. Diagonal
/// entries (i == j) have `delta = 0` → the irreducible floor.
///
/// Zero-allocation. The output is symmetric (`out[i*n + j] == out[j*n + i]`).
/// For `n > 64` this is rayon-parallelizable; Phase 1 ships sequential.
///
/// # Panics
///
/// Asserts `out.len() == n*n` where `n = loss_vectors.len()`, and that all
/// loss vectors have equal length (== k, the schedule size).
pub fn pairwise_bound(
    loss_vectors: &[&[f32]],
    lambda: f32,
    g: f32,
    tau: f32,
    c_h: f32,
    epsilon: f32,
    out: &mut [f32],
) {
    let n = loss_vectors.len();
    assert_eq!(out.len(), n * n, "out must be n*n");
    for i in 0..n {
        for j in 0..n {
            let delta_ij = if i == j {
                0.0
            } else {
                l_inf_distance(loss_vectors[i], loss_vectors[j])
            };
            out[i * n + j] = lipschitz_gradient_bound(delta_ij, lambda, g, tau, c_h, epsilon);
        }
    }
}

/// L-infinity distance `||a - b||_inf` = max elementwise absolute difference.
#[inline]
fn l_inf_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0_f32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    max_abs
}

// ──────────────────────────────────────────────────────────────────────────
// Diversity selection (T1.7) — greedy max-min
// ──────────────────────────────────────────────────────────────────────────

/// Greedy max-min diversity selection on perturbed-loss vectors.
///
/// From `n` candidates (each with a k-dim loss vector), pick the `k_subset`
/// candidates whose loss vectors have maximal spread. Algorithm (the modelless
/// analog of TEMP §3.2 Algorithm 1):
/// 1. Seed with the pair `(i, j)` of maximal L_inf distance.
/// 2. Iteratively add the candidate that maximizes the minimum L_inf distance
///    to the current subset (max-min facility location).
///
/// # Allocation discipline (G4)
///
/// `scratch` (len >= `k_subset`) is used as the working selected-set buffer;
/// the return value is a `Vec<usize>` copy of `scratch[..k_subset]`. The
/// greedy logic itself is allocation-free (only the return Vec allocates).
///
/// # Panics
///
/// Asserts `1 <= k_subset <= n` and `scratch.len() >= k_subset`.
pub fn select_diverse_subset(
    loss_vectors: &[&[f32]],
    k_subset: usize,
    scratch: &mut [usize],
) -> Vec<usize> {
    let n = loss_vectors.len();
    assert!(k_subset >= 1 && k_subset <= n, "k_subset must be in [1, n]");
    assert!(scratch.len() >= k_subset, "scratch must hold >= k_subset indices");

    let selected = &mut scratch[..k_subset];

    if k_subset == 1 {
        // Trivial: any single candidate. Pick index 0 by convention.
        selected[0] = 0;
    } else {
        // Seed with the max-distance pair.
        let (i, j) = argmax_pair(loss_vectors);
        selected[0] = i;
        selected[1] = j;
        let mut count = 2;
        while count < k_subset {
            // Find the candidate not yet selected that maximizes the minimum
            // L_inf distance to the current selected set.
            let mut best_c = 0_usize;
            let mut best_min = -1.0_f32;
            for c in 0..n {
                if selected[..count].contains(&c) {
                    continue;
                }
                let mut min_d = f32::INFINITY;
                for &s in &selected[..count] {
                    let d = l_inf_distance(loss_vectors[c], loss_vectors[s]);
                    if d < min_d {
                        min_d = d;
                    }
                }
                if min_d > best_min {
                    best_min = min_d;
                    best_c = c;
                }
            }
            selected[count] = best_c;
            count += 1;
        }
    }

    selected.to_vec()
}

/// Find the pair `(i, j)` with maximal L_inf distance among all candidates.
/// O(n²); used once per `select_diverse_subset` call to seed the greedy loop.
fn argmax_pair(loss_vectors: &[&[f32]]) -> (usize, usize) {
    let n = loss_vectors.len();
    debug_assert!(n >= 2);
    let mut best = (0_usize, 1_usize);
    let mut best_dist = l_inf_distance(loss_vectors[0], loss_vectors[1]);
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = l_inf_distance(loss_vectors[i], loss_vectors[j]);
            if dist > best_dist {
                best_dist = dist;
                best = (i, j);
            }
        }
    }
    best
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (Phase 1 exit: ≥8 unit tests)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test kernel: L = softplus(-dot(theta, z_prefix)) = -log(sigmoid(dot)).
    /// Matches `RavenSlotLossKernel` in riir-neuron-db Plan 005 T1.2.
    /// Monotone decreasing in dot product; numerically stable via softplus.
    struct DotSurpriseKernel;

    impl LossKernel for DotSurpriseKernel {
        fn short_prefix_loss(&self, theta: &[f32], z_prefix: &[f32]) -> f32 {
            let len = theta.len().min(z_prefix.len());
            let dot = simd_dot(theta, z_prefix, len);
            softplus(-dot)
        }
    }

    /// Inline dot product (keeps tests independent of the simd dispatch layer;
    /// plain scalar accumulation is deterministic and good enough for fixtures).
    #[inline]
    fn simd_dot(a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut s = 0.0_f32;
        for i in 0..len {
            s += a[i] * b[i];
        }
        s
    }

    /// Numerically stable softplus: log(1 + exp(x)). Equals -log(sigmoid(x))
    /// for the loss interpretation. Clamped beyond ±40 (sigmoid saturates).
    #[inline]
    fn softplus(x: f32) -> f32 {
        if x > 40.0 {
            x
        } else if x < -40.0 {
            0.0
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    // ── T1.3 extrapolated_snapshot_schedule ──────────────────────────────

    #[test]
    fn extrapolated_no_noise_linear_interpolation() {
        // k=4, noise_sigma=0 → 4 evenly-spaced points on [s0, s1].
        let s0 = vec![0.0_f32, 0.0];
        let s1 = vec![10.0_f32, 20.0];
        let lambda = [0.0_f32, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        let seeds = [0u64; 4];
        let mut out = vec![Vec::with_capacity(2); 4];
        extrapolated_snapshot_schedule(&s0, &s1, &lambda, &seeds, 0.0, &mut out);
        let expected = [
            [0.0_f32, 0.0],
            [10.0 / 3.0, 20.0 / 3.0],
            [20.0 / 3.0, 40.0 / 3.0],
            [10.0, 20.0],
        ];
        for (j, exp_j) in expected.iter().enumerate() {
            for i in 0..2 {
                assert!(
                    (out[j][i] - exp_j[i]).abs() < 1e-5,
                    "out[{}][{}]={} expected={}",
                    j,
                    i,
                    out[j][i],
                    exp_j[i]
                );
            }
        }
    }

    #[test]
    fn extrapolated_noise_within_bounds() {
        // With noise_sigma=0.1, coeff_j = lambda_j*(1+xi_j) where |xi_j| <= 0.1.
        let s0 = vec![0.0_f32];
        let s1 = vec![1.0_f32];
        let lambda = [0.5_f32];
        let seeds = [42u64];
        let mut out = vec![Vec::with_capacity(1)];
        extrapolated_snapshot_schedule(&s0, &s1, &lambda, &seeds, 0.1, &mut out);
        // theta_0 = 0 + 0.5*(1+xi)*1 where |xi| <= 0.1 → theta_0 in [0.45, 0.55]
        assert!(
            out[0][0] >= 0.45 && out[0][0] <= 0.55,
            "theta_0={} should be in [0.45, 0.55]",
            out[0][0]
        );
    }

    #[test]
    fn extrapolated_blake3_deterministic() {
        // Same seeds → bit-identical output (G4 quorum-reproducibility).
        let s0 = vec![1.0_f32, 2.0, 3.0];
        let s1 = vec![4.0_f32, 5.0, 6.0];
        let lambda = [0.25_f32, 0.5, 0.75];
        let seeds = [1u64, 2, 3];
        let mut out1 = vec![Vec::with_capacity(3); 3];
        let mut out2 = vec![Vec::with_capacity(3); 3];
        extrapolated_snapshot_schedule(&s0, &s1, &lambda, &seeds, 0.05, &mut out1);
        extrapolated_snapshot_schedule(&s0, &s1, &lambda, &seeds, 0.05, &mut out2);
        for j in 0..3 {
            assert_eq!(out1[j], out2[j], "run 1 != run 2 at j={}", j);
        }
    }

    // ── T1.4 perturbed_loss_vector ───────────────────────────────────────

    #[test]
    fn perturbed_loss_writes_k_values() {
        let kernel = DotSurpriseKernel;
        let theta_schedule = vec![vec![1.0_f32], vec![2.0], vec![3.0]];
        let z_prefix = vec![0.5_f32];
        let mut out = [0.0_f32; 3];
        perturbed_loss_vector(&kernel, &theta_schedule, &z_prefix, &mut out);
        // loss is monotone decreasing in dot(theta, z) — dot grows 0.5 → 1.0 → 1.5
        assert!(
            out[0] > out[1],
            "loss should decrease as dot increases: {} vs {}",
            out[0],
            out[1]
        );
        assert!(
            out[1] > out[2],
            "loss should decrease as dot increases: {} vs {}",
            out[1],
            out[2]
        );
    }

    // ── T1.5 lipschitz_gradient_bound ────────────────────────────────────

    #[test]
    fn lipschitz_bound_delta_zero_floor() {
        // delta=0 → bound = g*(1/sqrt(2)+tau) + c_h*epsilon
        let lambda = 1.0_f32;
        let g = 0.5_f32;
        let tau = 0.1_f32;
        let c_h = 0.01_f32;
        let epsilon = 0.001_f32;
        let bound = lipschitz_gradient_bound(0.0, lambda, g, tau, c_h, epsilon);
        let expected = g * (INV_SQRT_2 + tau) + c_h * epsilon;
        assert!(
            (bound - expected).abs() < 1e-6,
            "bound={} expected={}",
            bound,
            expected
        );
    }

    #[test]
    fn lipschitz_bound_monotone_in_delta() {
        let lambda = 1.0_f32;
        let g = 0.5_f32;
        let tau = 0.1_f32;
        let c_h = 0.01_f32;
        let epsilon = 0.001_f32;
        let b0 = lipschitz_gradient_bound(0.0, lambda, g, tau, c_h, epsilon);
        let b1 = lipschitz_gradient_bound(0.1, lambda, g, tau, c_h, epsilon);
        let b2 = lipschitz_gradient_bound(1.0, lambda, g, tau, c_h, epsilon);
        assert!(b0 < b1, "b0={} should be < b1={}", b0, b1);
        assert!(b1 < b2, "b1={} should be < b2={}", b1, b2);
    }

    // ── T1.6 pairwise_bound ──────────────────────────────────────────────

    #[test]
    fn pairwise_bound_symmetric_and_diagonal_floor() {
        let l1 = [1.0_f32, 2.0];
        let l2 = [3.0_f32, 0.0];
        let lvs: Vec<&[f32]> = vec![&l1, &l2];
        let mut out = [0.0_f32; 4];
        pairwise_bound(&lvs, 1.0, 0.0, 0.0, 0.0, 0.0, &mut out);
        // delta_12 = max(|1-3|, |2-0|) = 2; delta_11 = delta_22 = 0.
        // bound(i,j) = (2*delta/1 + 0)*(1/sqrt(2)+0) + 0 = 2*delta/sqrt(2)
        let off = 2.0 * 2.0 * INV_SQRT_2; // delta=2
        let diag = 0.0; // delta=0, g=0, c_h=0
        assert!(
            (out[0] - diag).abs() < 1e-6,
            "diag(0,0)={} expected {}",
            out[0],
            diag
        );
        assert!(
            (out[3] - diag).abs() < 1e-6,
            "diag(1,1)={} expected {}",
            out[3],
            diag
        );
        assert!(
            (out[1] - off).abs() < 1e-6,
            "off(0,1)={} expected {}",
            out[1],
            off
        );
        assert!(
            (out[2] - off).abs() < 1e-6,
            "off(1,0)={} expected {}",
            out[2],
            off
        );
    }

    // ── T1.7 select_diverse_subset ───────────────────────────────────────

    #[test]
    fn select_diverse_picks_spread_subset() {
        // 4 candidates on a line: 0, 1, 2, 10. Pick k=2 → must be (0, 3)
        // since (0,3) is the first pair achieving max distance 10.
        let l0 = [0.0_f32];
        let l1 = [1.0_f32];
        let l2 = [2.0_f32];
        let l3 = [10.0_f32];
        let lvs: Vec<&[f32]> = vec![&l0, &l1, &l2, &l3];
        let mut scratch = [0_usize; 2];
        let picked = select_diverse_subset(&lvs, 2, &mut scratch);
        assert_eq!(picked.len(), 2);
        assert!(picked.contains(&3), "must include index 3 (the far one)");
        assert_eq!(picked, vec![0, 3]);
    }

    #[test]
    fn select_diverse_greedy_grows_spread() {
        // 5 candidates in 2D; verify the greedy third pick maximizes min-dist.
        let l0 = [0.0_f32, 0.0];
        let l1 = [0.1_f32, 0.1]; // near l0
        let l2 = [5.0_f32, 0.0]; // far from {l0, l1}
        let l3 = [0.0_f32, 5.0]; // far from {l0, l1, l2}
        let l4 = [5.0_f32, 5.0];
        let lvs: Vec<&[f32]> = vec![&l0, &l1, &l2, &l3, &l4];
        let mut scratch = [0_usize; 3];
        let picked = select_diverse_subset(&lvs, 3, &mut scratch);
        assert_eq!(picked.len(), 3);
        // Seed = first max-distance pair. (0,1)=0.1, (0,2)=5 (first hit at 5).
        assert_eq!(picked[0], 0);
        assert_eq!(picked[1], 2);
        // Third pick: max min-dist to {0, 2}.
        //   c=1: min(0.1, 4.9) = 0.1
        //   c=3: min(5.0, 5.0) = 5.0
        //   c=4: min(5.0, 5.0) = 5.0   (tie; first wins → c=3)
        assert_eq!(picked[2], 3);
    }

    #[test]
    fn select_diverse_single_candidate() {
        let l0 = [1.0_f32];
        let lvs: Vec<&[f32]> = vec![&l0];
        let mut scratch = [0_usize; 1];
        let picked = select_diverse_subset(&lvs, 1, &mut scratch);
        assert_eq!(picked, vec![0]);
    }
}
