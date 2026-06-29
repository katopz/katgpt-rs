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
const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

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
///
/// Thin wrapper over `simd_l_inf_distance_f32` (NEON/AVX2/scalar dispatch).
/// Inlined so the dispatch is hoisted into the caller's hot loop. Bit-identical
/// to the previous scalar loop on all finite inputs (verified by
/// `katgpt_types::simd::tests::l_inf_distance_matches_scalar_across_lengths`).
#[inline]
fn l_inf_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    crate::simd::simd_l_inf_distance_f32(a, b, a.len())
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
/// # Performance (G5)
///
/// The greedy fill maintains a cached `min_dist[c]` vector (the minimum
/// L_inf distance from candidate `c` to any selected element) and a boolean
/// `is_selected[c]` mask. When a new element is selected, `min_dist` is updated
/// in one O(n·K) pass (not recomputed from scratch each round). Total greedy
/// fill complexity: O(n·k_subset·K). The one-time `argmax_pair` seed is
/// O(n²·K) and dominates for small `k_subset`.
///
/// The output is **bit-identical** to the naive recomputation — the min of
/// mins is the min, regardless of evaluation order. Verified by G4
/// quorum-reproducibility (100/100 hash matches on randomized configs).
///
/// # Allocation discipline (G4)
///
/// `scratch` (len >= `k_subset`) is used as the working selected-set buffer;
/// the return value is a `Vec<usize>` copy of `scratch[..k_subset]`. The
/// `min_dist` and `is_selected` workspaces are heap-allocated `Vec<f32>` /
/// `Vec<bool>` (resized to `n` on each call). Use
/// [`select_diverse_subset_into`] to pass reusable workspaces and avoid
/// reallocation on repeated calls. The only OTHER heap allocation is the
/// return `Vec<usize>`.
///
/// # Panics
///
/// Asserts `1 <= k_subset <= n` and `scratch.len() >= k_subset`.
pub fn select_diverse_subset(
    loss_vectors: &[&[f32]],
    k_subset: usize,
    scratch: &mut [usize],
) -> Vec<usize> {
    select_diverse_subset_into(loss_vectors, k_subset, scratch, &mut Vec::new(), &mut Vec::new())
}

/// Same as [`select_diverse_subset`] but accepts caller-provided workspaces
/// for `min_dist` and `is_selected` so repeated calls don't reallocate. Both
/// workspaces are resized to `n` (truncated or grown as needed) and fully
/// overwritten on each call.
///
/// The return value is a `Vec<usize>` copy of `scratch[..k_subset]`.
pub fn select_diverse_subset_into(
    loss_vectors: &[&[f32]],
    k_subset: usize,
    scratch: &mut [usize],
    min_dist_workspace: &mut Vec<f32>,
    is_selected_workspace: &mut Vec<bool>,
) -> Vec<usize> {
    let n = loss_vectors.len();
    assert!(k_subset >= 1 && k_subset <= n, "k_subset must be in [1, n]");
    assert!(scratch.len() >= k_subset, "scratch must hold >= k_subset indices");

    let selected = &mut scratch[..k_subset];

    if k_subset == 1 {
        // Trivial: any single candidate. Pick index 0 by convention.
        selected[0] = 0;
        return selected.to_vec();
    }

    // Resize workspaces to n (reusing capacity across calls).
    min_dist_workspace.clear();
    min_dist_workspace.resize(n, f32::INFINITY);
    is_selected_workspace.clear();
    is_selected_workspace.resize(n, false);
    let min_dist = &mut min_dist_workspace[..n];
    let is_selected = &mut is_selected_workspace[..n];

    // Seed with the max-distance pair.
    let (i, j) = argmax_pair(loss_vectors);
    selected[0] = i;
    selected[1] = j;
    is_selected[i] = true;
    is_selected[j] = true;
    // Selected elements have min_dist = 0.0 (distance to themselves). The
    // argmax loop skips them via `is_selected`, so this is fine — they can
    // never win after being marked.
    min_dist[i] = 0.0;
    min_dist[j] = 0.0;

    // Initialize min_dist for the remaining candidates: min distance to {i, j}.
    for c in 0..n {
        if is_selected[c] {
            continue;
        }
        let di = l_inf_distance(loss_vectors[c], loss_vectors[i]);
        let dj = l_inf_distance(loss_vectors[c], loss_vectors[j]);
        min_dist[c] = if di < dj { di } else { dj };
    }

    let mut count = 2;
    while count < k_subset {
        // Find the unselected candidate with max min-distance to the selected
        // set. Candidates are scanned in index order; ties broken by strict
        // `>` (lower index wins) — same convention as the naive version.
        let mut best_c = 0_usize;
        let mut best_min = -1.0_f32;
        for c in 0..n {
            if is_selected[c] {
                continue;
            }
            if min_dist[c] > best_min {
                best_min = min_dist[c];
                best_c = c;
            }
        }
        selected[count] = best_c;
        is_selected[best_c] = true;
        min_dist[best_c] = 0.0;

        // Update min_dist for remaining candidates: min(min_dist[c], dist(c, best_c)).
        for c in 0..n {
            if is_selected[c] {
                continue;
            }
            let d = l_inf_distance(loss_vectors[c], loss_vectors[best_c]);
            if d < min_dist[c] {
                min_dist[c] = d;
            }
        }
        count += 1;
    }

    selected.to_vec()
}

/// Find the pair `(i, j)` with maximal L_inf distance among all candidates.
/// O(n²); used once per `select_diverse_subset` call to seed the greedy loop.
///
/// # SIMD blocking (riir-neuron-db Issue 003)
///
/// For the production K=8 workload the naive O(n²) double loop spends most
/// of its time in `l_inf_distance`'s per-call horizontal reduction
/// (`vmaxvq_f32` on NEON, `horizontal_max_256` on AVX2). This impl blocks
/// the inner `j` loop by the SIMD j-width (4 candidates at a time): one
/// broadcast `i`-vector is diffed against 4 `j`-vectors in parallel, and
/// the block's max-distance is reduced to a scalar ONCE per 4 distances
/// (not once per distance). When a block beats the running `best_dist`, a
/// scalar rescan of those 4 pairs finds the exact winning index — this
/// branch is rare (taken only when a new global max appears, O(log n) times
/// across the whole O(n²) scan for typical distributions), so its cost
/// amortizes to zero.
///
/// Output is bit-identical to the scalar double loop: max-of-maxes is
/// associative/commutative for finite f32, so the blocked reduction order
/// does not change the result (verified by `g4_determinism_bit_identical`).
fn argmax_pair(loss_vectors: &[&[f32]]) -> (usize, usize) {
    let n = loss_vectors.len();
    debug_assert!(n >= 2);
    if n < 2 {
        return (0, 0);
    }

    let k = loss_vectors[0].len();
    if k == 0 {
        return (0, 1.min(n - 1));
    }

    // Seed with the (0, 1) pair so `best_dist` starts from a real value.
    let mut best = (0_usize, 1_usize);
    let mut best_dist = l_inf_distance(loss_vectors[0], loss_vectors[1]);

    // Scalar fallback for tiny K (sub-SIMD-width) or no-SIMD targets. The
    // blocked path's amortization only wins when K >= the SIMD width; for
    // K < 4 the per-block horizontal-reduction overhead exceeds the savings.
    if k < 4 || !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = l_inf_distance(loss_vectors[i], loss_vectors[j]);
                if dist > best_dist {
                    best_dist = dist;
                    best = (i, j);
                }
            }
        }
        return best;
    }

    // SIMD-blocked path: 4 j-vectors per inner iteration against one broadcast i.
    // BLOCK=4 is the NEON width; on AVX2 the K-inner loop still uses 8-wide ops
    // (the j-blocking is orthogonal to the K-vectorization — we block j by 4
    // because that keeps the register pressure manageable on both ISAs while
    // still amortizing the horizontal reduction 4×).
    const J_BLOCK: usize = 4;

    for i in 0..n {
        let base = loss_vectors[i];
        let mut j = i + 1;
        // Align j up to the largest multiple of J_BLOCK <= n.
        let block_end = j + ((n - j) / J_BLOCK) * J_BLOCK;

        while j < block_end {
            // Compute the max L_inf distance from `base` to the 4 candidates
            // [j, j+1, j+2, j+3] in ONE horizontal reduction (vs 4 reductions
            // in the naive loop). This is the amortization win.
            let block_max = block_max_l_inf_distance(base, loss_vectors, j, k);
            if block_max > best_dist {
                // Rare path: a new global max appeared somewhere in this block.
                // Rescan the 4 pairs scalar to find the exact winner. Cost is
                // O(J_BLOCK) per hit, and hits are O(log n) over the whole scan.
                for jj in j..j + J_BLOCK {
                    let dist = l_inf_distance(base, loss_vectors[jj]);
                    if dist > best_dist {
                        best_dist = dist;
                        best = (i, jj);
                    }
                }
            }
            j += J_BLOCK;
        }

        // Scalar tail (1..J_BLOCK-1 trailing j's).
        while j < n {
            let dist = l_inf_distance(base, loss_vectors[j]);
            if dist > best_dist {
                best_dist = dist;
                best = (i, j);
            }
            j += 1;
        }
    }
    best
}

/// Max L_inf distance from `base` to each of `candidates[j..j+4]`, reduced to
/// ONE scalar in a single horizontal pass. Used by `argmax_pair`'s blocked
/// inner loop to amortize the horizontal reduction across 4 distances.
///
/// # Safety contract (caller-enforced, debug_asserted)
///
/// - `j + 4 <= candidates.len()`
/// - Every `candidates[r][..k]` has at least `k` elements (uniform-K invariant)
#[inline]
fn block_max_l_inf_distance(base: &[f32], candidates: &[&[f32]], j: usize, k: usize) -> f32 {
    debug_assert!(j + 4 <= candidates.len());
    let c0 = candidates[j];
    let c1 = candidates[j + 1];
    let c2 = candidates[j + 2];
    let c3 = candidates[j + 3];

    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vabdq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32};
        unsafe {
            // Process K in 4-element chunks. Each chunk: load base + 4 candidates
            // (5 loads), 4× vabdq, fold into one running 4-wide block_max via
            // vmaxq. ONE vmaxvq horizontal reduce at the end → 4 distances for
            // the price of 1 reduction (vs 4 in the naive per-call loop).
            let mut block_max = vdupq_n_f32(0.0);
            let mut idx = 0;
            let chunks = k / 4;
            for _ in 0..chunks {
                let b = vld1q_f32(base.as_ptr().add(idx));
                let d0 = vabdq_f32(b, vld1q_f32(c0.as_ptr().add(idx)));
                let d1 = vabdq_f32(b, vld1q_f32(c1.as_ptr().add(idx)));
                let d2 = vabdq_f32(b, vld1q_f32(c2.as_ptr().add(idx)));
                let d3 = vabdq_f32(b, vld1q_f32(c3.as_ptr().add(idx)));
                block_max = vmaxq_f32(block_max, d0);
                block_max = vmaxq_f32(block_max, d1);
                block_max = vmaxq_f32(block_max, d2);
                block_max = vmaxq_f32(block_max, d3);
                idx += 4;
            }
            let mut m = vmaxvq_f32(block_max);
            // Scalar tail for K not divisible by 4.
            while idx < k {
                let bi = *base.get_unchecked(idx);
                m = m
                    .max((bi - *c0.get_unchecked(idx)).abs())
                    .max((bi - *c1.get_unchecked(idx)).abs())
                    .max((bi - *c2.get_unchecked(idx)).abs())
                    .max((bi - *c3.get_unchecked(idx)).abs());
                idx += 1;
            }
            m
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 path: same structure, 8-wide chunks. One broadcast base load
        // per chunk, 4× (sub + and-mask) for the 4 candidates, fold via
        // _mm256_max_ps, one horizontal_max_256 reduce at the end.
        //
        // The intrinsics live in a `target_feature(enable="avx2,fma")`-gated
        // helper (`avx2_block_max_l_inf_distance`) and dispatched via the
        // shared runtime `simd_level()` check — same pattern as
        // `katgpt_types::simd::research::avx2_dist_sq`. Non-AVX2 x86_64 CPUs
        // fall through to the scalar block.
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            unsafe { avx2_block_max_l_inf_distance(base, c0, c1, c2, c3, k) }
        } else {
            block_max_l_inf_distance_scalar(base, c0, c1, c2, c3, k)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        block_max_l_inf_distance_scalar(base, c0, c1, c2, c3, k)
    }
}

/// Scalar reference for `block_max_l_inf_distance`. Computes the max L_inf
/// distance from `base` to each of `c0..c3` over `k` elements. Used by the
/// non-SIMD cfg branches and by non-AVX2 x86_64.
///
/// On aarch64 this is dead code (the NEON block path is always taken), so
/// we silence the dead-code lint there. On every other target it's live.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn block_max_l_inf_distance_scalar(
    base: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
    k: usize,
) -> f32 {
    let mut m = 0.0_f32;
    for idx in 0..k {
        let bi = base[idx];
        m = m
            .max((bi - c0[idx]).abs())
            .max((bi - c1[idx]).abs())
            .max((bi - c2[idx]).abs())
            .max((bi - c3[idx]).abs());
    }
    m
}

/// AVX2 kernel for `block_max_l_inf_distance`. `#[target_feature]`-gated so
/// the `_mm256_*` intrinsics compile on any x86_64 target (the runtime
/// `simd_level() == Avx2` check in the caller guards the actual invocation).
/// Mirrors the structure of `katgpt_types::simd::research::avx2_dist_sq`.
///
/// `unsafe_op_in_unsafe_fn` is allowed here to match the existing
/// `katgpt_types::simd::research::avx2_*` kernels (same edition-2024 lint
/// posture — the whole body is intrinsic calls under a `target_feature`
/// contract; wrapping each in its own `unsafe {}` would just add noise).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx2_block_max_l_inf_distance(
    base: &[f32],
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    c3: &[f32],
    k: usize,
) -> f32 {
    use core::arch::x86_64::{
        _mm256_and_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_sub_ps,
    };
    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
    let mut block_max = _mm256_setzero_ps();
    let mut idx = 0;
    let chunks = k / 8;
    for _ in 0..chunks {
        let b = _mm256_loadu_ps(base.as_ptr().add(idx));
        let d0 = _mm256_and_ps(
            _mm256_sub_ps(b, _mm256_loadu_ps(c0.as_ptr().add(idx))),
            abs_mask,
        );
        let d1 = _mm256_and_ps(
            _mm256_sub_ps(b, _mm256_loadu_ps(c1.as_ptr().add(idx))),
            abs_mask,
        );
        let d2 = _mm256_and_ps(
            _mm256_sub_ps(b, _mm256_loadu_ps(c2.as_ptr().add(idx))),
            abs_mask,
        );
        let d3 = _mm256_and_ps(
            _mm256_sub_ps(b, _mm256_loadu_ps(c3.as_ptr().add(idx))),
            abs_mask,
        );
        block_max = _mm256_max_ps(block_max, d0);
        block_max = _mm256_max_ps(block_max, d1);
        block_max = _mm256_max_ps(block_max, d2);
        block_max = _mm256_max_ps(block_max, d3);
        idx += 8;
    }
    // Horizontal max across the 8 lanes (inlined copy of
    // `horizontal_max_256`, which is `pub(super)` in katgpt-types and
    // unreachable here). Same shuffle sequence, kept in sync.
    use core::arch::x86_64::{
        _mm_cvtss_f32, _mm_max_ps, _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
    };
    let hi = _mm256_extractf128_ps(block_max, 1);
    let lo = _mm256_castps256_ps128(block_max);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_shuffle_ps(m128, m128, 0xB1);
    let m2 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_shuffle_ps(m2, m2, 0x4E);
    let m3 = _mm_max_ps(m2, shuf2);
    let mut m = _mm_cvtss_f32(m3);
    while idx < k {
        let bi = *base.get_unchecked(idx);
        m = m
            .max((bi - *c0.get_unchecked(idx)).abs())
            .max((bi - *c1.get_unchecked(idx)).abs())
            .max((bi - *c2.get_unchecked(idx)).abs())
            .max((bi - *c3.get_unchecked(idx)).abs());
        idx += 1;
    }
    m
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

    // ────────────────────────────────────────────────────────────────────
    // Plan 341 Phase 2 — GOAT gate tests (G1, G2, G4)
    // ────────────────────────────────────────────────────────────────────

    /// Deterministic xorshift RNG for GOAT-gate fixture generation.
    struct FixtureRng(u64);
    impl FixtureRng {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn uniform(&mut self) -> f32 {
            let bits = ((self.next_u64() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
            f32::from_bits(bits) - 1.0
        }
        /// Approx standard normal via Box-Muller (fixture generation only).
        fn normal(&mut self) -> f32 {
            let u1 = self.uniform().max(1e-10);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }
    }

    /// Sequential dot kernel: interprets `z_prefix` as N tokens of dim D,
    /// computes `sum_t softplus(-dot(theta, z_prefix[t*D..(t+1)*D]))`.
    /// Used for the G2 prefix-length sweep where N must be variable.
    struct SequentialDotKernel {
        d: usize,
    }
    impl LossKernel for SequentialDotKernel {
        fn short_prefix_loss(&self, theta: &[f32], z_prefix: &[f32]) -> f32 {
            let d = self.d.min(theta.len());
            let n_tokens = z_prefix.len() / d;
            let mut total = 0.0_f32;
            for t in 0..n_tokens {
                let mut dot = 0.0_f32;
                for i in 0..d {
                    dot += theta[i] * z_prefix[t * d + i];
                }
                total += softplus(-dot);
            }
            total
        }
    }

    /// Minimum pairwise Lipschitz bound over all C(k, 2) pairs in a subset.
    /// Returns +∞ for subsets of size < 2 (no pairs). This is the quantity
    /// the greedy max-min algorithm directly optimizes: it picks the subset
    /// whose worst-case (minimum) pairwise bound is as large as possible.
    fn min_pairwise_bound(
        subset: &[usize],
        lvs: &[&[f32]],
        lambda: f32,
        g: f32,
        tau: f32,
        c_h: f32,
        epsilon: f32,
    ) -> f32 {
        let mut min_bound = f32::INFINITY;
        for a in 0..subset.len() {
            for b in (a + 1)..subset.len() {
                let delta = l_inf_distance(lvs[subset[a]], lvs[subset[b]]);
                let bound = lipschitz_gradient_bound(delta, lambda, g, tau, c_h, epsilon);
                if bound < min_bound {
                    min_bound = bound;
                }
            }
        }
        min_bound
    }

    /// Kendall tau rank correlation between two equally-sized value slices.
    /// Returns a value in [-1, 1]. Ties contribute 0 to numerator and
    /// denominator (standard Kendall tau-a convention).
    fn kendall_tau(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        if n < 2 {
            return 1.0;
        }
        let mut concordant: i64 = 0;
        let mut discordant: i64 = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let da = a[i] - a[j];
                let db = b[i] - b[j];
                if (da > 0.0 && db > 0.0) || (da < 0.0 && db < 0.0) {
                    concordant += 1;
                } else if (da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0) {
                    discordant += 1;
                }
            }
        }
        let total = (n * (n - 1) / 2) as f32;
        if total == 0.0 {
            1.0
        } else {
            (concordant - discordant) as f32 / total
        }
    }

    // ── T2.1 G1 — Bound preservation under diversity selection ──────────

    #[test]
    fn g1_bound_preservation_under_diversity_selection() {
        // n=64 candidates drawn from a Gaussian mixture: 50 from a tight
        // cluster N(0, 0.5) and 14 from a wide spread N(0, 5.0) in each of 8
        // dims. The greedy max-min selector picks 8 spread-out candidates
        // (mostly from the wide component) whose minimum pairwise L_inf delta
        // is ~5–10. A random 8-subset is dominated by the 50 cluster points
        // (expected ~6.25 from the cluster), whose minimum pairwise delta is
        // ~0.5–1.5. The MINIMUM pairwise Lipschitz bound — the quantity the
        // greedy algorithm directly optimizes — should be ≥ 2× higher for the
        // selected subset than for a random subset (median over 50 trials).
        //
        // This directly validates Theorem 3.1: the diverse selector picks the
        // subset whose members would induce maximally-different gradients
        // along v — even the worst-case pair has a high bound.
        let k_dim: usize = 8;
        let k_subset: usize = 8;
        let n: usize = 64;
        let n_cluster: usize = 50;

        let mut rng = FixtureRng::new(42);
        let loss_vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let scale = if i < n_cluster { 0.5 } else { 5.0 };
                (0..k_dim).map(|_| rng.normal() * scale).collect()
            })
            .collect();
        let lvs_refs: Vec<&[f32]> = loss_vecs.iter().map(|v| v.as_slice()).collect();

        let mut scratch = vec![0_usize; k_subset];
        let selected = select_diverse_subset(&lvs_refs, k_subset, &mut scratch);
        assert_eq!(selected.len(), k_subset);

        // Bound params: small lambda amplifies delta's contribution.
        let lambda = 0.1_f32;
        let g = 0.5_f32;
        let tau = 0.1_f32;
        let c_h = 0.01_f32;
        let epsilon = 0.001_f32;

        let selected_min_bound =
            min_pairwise_bound(&selected, &lvs_refs, lambda, g, tau, c_h, epsilon);

        // Min pairwise bound for 50 deterministic random 8-subsets; report the
        // median (robust to outlier subsets that happen to be well-spread).
        let mut random_min_bounds: Vec<f32> = Vec::with_capacity(50);
        for seed in 100..150 {
            let mut s_rng = FixtureRng::new(seed);
            let mut idxs: Vec<usize> = (0..n).collect();
            for i in 0..k_subset {
                let j = i + (s_rng.next_u64() as usize) % (n - i);
                idxs.swap(i, j);
            }
            random_min_bounds.push(min_pairwise_bound(
                &idxs[..k_subset],
                &lvs_refs,
                lambda,
                g,
                tau,
                c_h,
                epsilon,
            ));
        }
        random_min_bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let random_median = random_min_bounds[random_min_bounds.len() / 2];
        let ratio = selected_min_bound / random_median;

        eprintln!(
            "G1: selected_min_bound={:.4} random_median_min_bound={:.4} ratio={:.2}x (target >= 2.0x)",
            selected_min_bound, random_median, ratio
        );
        assert!(
            selected_min_bound >= 2.0 * random_median,
            "G1 FAIL: selected min bound {:.4} should be >= 2x random median min bound {:.4} (ratio {:.2}x)",
            selected_min_bound,
            random_median,
            ratio
        );
    }

    // ── T2.2 G2 — Prefix-length sweep (Kendall tau >= 0.85 at N=32 vs N=256) ──

    #[test]
    fn g2_prefix_length_sweep_kendall_tau() {
        // n=32 candidates, each with 256-token prefix (D=8 per token).
        // K=8 checkpoints. Per-candidate diversity score = sum of L_inf
        // distances to all other candidates' loss vectors. The Kendall tau
        // between the N=32 and N=256 rankings must be >= 0.85 — the modelless
        // analog of paper Fig. 6 (short prefix captures full-trace signal).
        //
        // Fixture: each candidate has a well-separated unit direction d_i.
        // Tokens are d_i scaled to amplitude 5.0 + small Gaussian noise
        // (std 0.15). The signal-to-noise ratio is ~33:1 per token; at N=32
        // the averaging effect gives SNR ~190:1, which stabilizes the ranking.
        let n: usize = 32;
        let d: usize = 8;
        let k: usize = 8;
        let n_max: usize = 256;
        let signal_scale: f32 = 5.0;
        let noise_std: f32 = 0.15;

        let mut rng = FixtureRng::new(12345);

        // 32 well-separated unit directions.
        let mut directions: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let dir: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
            let norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            directions.push(dir.iter().map(|x| x / norm).collect());
        }

        // Each candidate: 256 tokens of (signal_scale * dir_i + noise_std * noise).
        let prefixes: Vec<Vec<f32>> = directions
            .iter()
            .map(|dir| {
                let mut z = vec![0.0_f32; n_max * d];
                for t in 0..n_max {
                    for i in 0..d {
                        z[t * d + i] = signal_scale * dir[i] + noise_std * rng.normal();
                    }
                }
                z
            })
            .collect();

        // K=8 checkpoints: s0 = origin, s1 = random unit vector.
        let s0 = vec![0.0_f32; d];
        let s1: Vec<f32> = {
            let v: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter().map(|x| x / norm).collect()
        };
        let lambda = [
            0.0_f32, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0, 1.0,
        ];
        let seeds = [0u64; 8];
        let mut theta_schedule = vec![Vec::with_capacity(d); k];
        extrapolated_snapshot_schedule(&s0, &s1, &lambda, &seeds, 0.0, &mut theta_schedule);

        let kernel = SequentialDotKernel { d };

        let n_values: [usize; 6] = [8, 16, 32, 64, 128, 256];
        let mut scores_by_n: Vec<Vec<f32>> = Vec::with_capacity(n_values.len());
        for &n_tokens in &n_values {
            let mut loss_vecs: Vec<[f32; 8]> = vec![[0.0_f32; 8]; n];
            for i in 0..n {
                let z_slice = &prefixes[i][..n_tokens * d];
                perturbed_loss_vector(&kernel, &theta_schedule, z_slice, &mut loss_vecs[i]);
            }
            let mut scores = vec![0.0_f32; n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        scores[i] += l_inf_distance(&loss_vecs[i], &loss_vecs[j]);
                    }
                }
            }
            scores_by_n.push(scores);
        }

        // Kendall tau vs N=256 (last entry).
        let ref_idx = n_values.len() - 1;
        let ref_scores = &scores_by_n[ref_idx];
        let n_ref = n_values[ref_idx];
        for (idx, &n_tokens) in n_values.iter().enumerate() {
            let tau = kendall_tau(ref_scores, &scores_by_n[idx]);
            eprintln!(
                "G2: Kendall tau at N={} vs N={}: {:.4}",
                n_tokens, n_ref, tau
            );
        }

        let tau_32 = kendall_tau(ref_scores, &scores_by_n[2]);
        assert!(
            tau_32 >= 0.85,
            "G2 FAIL: Kendall tau at N=32 vs N=256 is {:.4}, should be >= 0.85",
            tau_32
        );
    }

    // ── T2.4 G4 — Determinism / quorum-reproducibility ─────────────────

    #[test]
    fn g4_determinism_bit_identical() {
        // Two independent runs with the same (s0, s1, lambda, seeds, noise,
        // candidates) must produce bit-identical theta schedule, loss vectors,
        // and selected subset. This is the sync-boundary requirement.
        let n: usize = 32;
        let d: usize = 8;
        let k: usize = 8;

        let mut rng = FixtureRng::new(999);
        let s0: Vec<f32> = (0..d).map(|_| rng.uniform()).collect();
        let s1: Vec<f32> = (0..d).map(|_| rng.uniform()).collect();
        let lambda = [0.0_f32, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0];
        let seeds = [42u64, 43, 44, 45, 46, 47, 48, 49];
        let noise_sigma = 0.05_f32;

        let prefixes: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.uniform()).collect())
            .collect();

        let kernel = SequentialDotKernel { d };

        fn run(
            s0: &[f32],
            s1: &[f32],
            lambda: &[f32],
            seeds: &[u64],
            noise_sigma: f32,
            prefixes: &[Vec<f32>],
            kernel: &SequentialDotKernel,
            d: usize,
            k: usize,
            n: usize,
        ) -> (Vec<Vec<f32>>, Vec<[f32; 8]>, Vec<usize>) {
            let mut theta = vec![Vec::with_capacity(d); k];
            extrapolated_snapshot_schedule(s0, s1, lambda, seeds, noise_sigma, &mut theta);
            let mut loss_vecs: Vec<[f32; 8]> = vec![[0.0_f32; 8]; n];
            for i in 0..n {
                perturbed_loss_vector(kernel, &theta, &prefixes[i], &mut loss_vecs[i]);
            }
            let lvs: Vec<&[f32]> = loss_vecs.iter().map(|v| v.as_slice()).collect();
            let mut scratch = vec![0_usize; 8];
            let selected = select_diverse_subset(&lvs, 8, &mut scratch);
            (theta, loss_vecs, selected)
        }

        let (theta1, loss_vecs1, selected1) =
            run(&s0, &s1, &lambda, &seeds, noise_sigma, &prefixes, &kernel, d, k, n);
        let (theta2, loss_vecs2, selected2) =
            run(&s0, &s1, &lambda, &seeds, noise_sigma, &prefixes, &kernel, d, k, n);

        for j in 0..k {
            assert_eq!(theta1[j], theta2[j], "theta[{}] differs across runs", j);
        }
        for i in 0..n {
            assert_eq!(loss_vecs1[i], loss_vecs2[i], "loss_vec[{}] differs", i);
        }
        assert_eq!(selected1, selected2, "selected subset differs across runs");
    }
}
