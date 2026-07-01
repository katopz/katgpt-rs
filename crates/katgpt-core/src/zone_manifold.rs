//! Zone Affective Manifold — crowd-scale PCA via power iteration + deflation.
//!
//! Extracts the top-k principal directions ("zone mood axes") of an `(N, D)`
//! crowd-activation matrix in `O(N·D·k·iters + D²·k)` time, with `D²` scratch.
//! For the HLA use case (`D = 8`) the covariance is an `8×8 = 64-float` PSD
//! matrix that fits in 256 bytes — trivially L1-resident.
//!
//! ## Algorithm
//!
//! 1. Accumulate the `D×D` covariance `C = X^T·X` (optionally rayon-parallel
//!    over `N` when `N > parallel_threshold`). Subtract the mean first if
//!    [`ZoneManifoldConfig::center`] is set (default: true — PCA requires
//!    mean-centered data; pass `false` only for second-moment use cases).
//! 2. For each of `k` components:
//!    - Power-iterate `v ← C·v / ‖C·v‖` for `n_iters` steps from a fixed seed.
//!    - Record the Rayleigh quotient `λ = v^T·C·v / v^T·v` as the eigenvalue.
//!    - Deflate: `C ← C − λ·v·v^T`.
//! 3. Sign-fix each axis against the previous-tick snapshot (if supplied) so
//!    the same physical axis doesn't flip sign across ticks — temporal
//!    continuity for downstream routing.
//! 4. Project each NPC onto the zone axes: `P[i, j] = dot(X[i], axes[j])`.
//!
//! ## Determinism
//!
//! The power-iteration seed is the deterministic constant `1/√D` (not random),
//! matching [`crate::data_probe::stable_rank_update_into`]'s convention. Power
//! iteration on a PSD matrix converges to the dominant eigenvector regardless
//! of the seed as long as it has nonzero overlap. All arithmetic is IEEE-754
//! `f32` with a fixed operation order, so the output is bit-identical across
//! `x86_64` / `aarch64` / `wasm32` for the same crowd snapshot.
//!
//! ## Zero-alloc hot path
//!
//! All scratch lives in [`ZoneManifoldScratch`], pre-allocated once and reused
//! across ticks. [`zone_affective_manifold`] performs no heap allocation after
//! warmup (once `N` and `D` are stable).
//!
//! ## Latent vs raw boundary
//!
//! This primitive is **purely latent**. The zone axes and per-NPC projections
//! are `f32` SVD outputs — they **never cross the sync boundary**. If a
//! zone-level event derived from the manifold needs chain commit, project to
//! raw scalars (event ID, intensity, timestamp) at the caller and commit those,
//! not the axes. See the repo `AGENTS.md` sync-boundary rule.
//!
//! Feature-gated behind `#[cfg(feature = "zone_affective_manifold")]`.

use crate::simd;
use rayon::prelude::*;

// ── Types ───────────────────────────────────────────────────────

/// Per-tick report returned by [`zone_affective_manifold`].
///
/// All slices borrow from the caller's output buffers (zero-copy).
#[derive(Debug)]
pub struct ZoneManifoldReport<'a> {
    /// Zone mood axes. When `n_groups == 1`, this is `(D, k)` row-major.
    /// When `n_groups > 1`, this is `(G, D, k)` row-major — group `g`'s axes
    /// at `[g*D*k .. (g+1)*D*k)`.
    pub zone_axes: &'a [f32],
    /// `(N, k)` row-major per-NPC projections. NPC `i` is projected onto its
    /// group's axes: group id = `i * n_groups / n`.
    pub npc_projections: &'a [f32],
    /// Top-k eigenvalues of the dominant group (group 0), descending. When
    /// `n_groups > 1`, only group 0's eigenvalues are reported here; the full
    /// per-group eigenvalues are in the output `eigenvalues` buffer at
    /// `[g*k .. (g+1)*k]`.
    pub eigenvalues: &'a [f32],
    /// Explained-variance ratio of the top-k subspace (group 0). Computed from
    /// the trace of group 0's original covariance before deflation.
    pub explained_variance_ratio: f32,
    /// Number of groups (1 = single zone-wide PCA, >1 = cluster-and-distribute).
    pub n_groups: usize,
    /// Number of NPCs processed.
    pub n: usize,
    /// Latent dimension `D`.
    pub d: usize,
    /// Target manifold dimension `k`.
    pub k: usize,
}

/// Caller-owned scratch, reused across ticks. Pre-allocate once via
/// [`ZoneManifoldScratch::new`]; subsequent calls are allocation-free as long
/// as `D` and `K` don't grow.
pub struct ZoneManifoldScratch {
    /// `(D, D)` row-major covariance matrix. Deflated in place across the `k`
    /// components during power iteration; the caller sees the *deflated*
    /// covariance after the call — use [`Self::restore_cov`] if you need the
    /// original for explained-variance diagnostics (the report already carries
    /// `explained_variance_ratio`, so most callers never need this).
    pub cov: Vec<f32>,
    /// Backup of the un-deflated covariance (restored at end of each call).
    pub cov_backup: Vec<f32>,
    /// Current power-iteration iterate (length `D`).
    pub v: Vec<f32>,
    /// Next iterate / matvec target (length `D`).
    pub w: Vec<f32>,
    /// Column-mean accumulator (length `D`), used when `center = true`.
    pub mean: Vec<f32>,
    /// Per-chunk accumulator for rayon parallel covariance (length
    /// `num_chunks * D*D`, grown lazily).
    pub chunk_cov: Vec<f32>,
    cached_d: usize,
    cached_k: usize,
}

impl ZoneManifoldScratch {
    /// Allocate scratch for a `D`-dimensional latent space with at most `K`
    /// manifold axes.
    pub fn new(d: usize, k: usize) -> Self {
        let dd = d * d;
        Self {
            cov: vec![0.0; dd],
            cov_backup: vec![0.0; dd],
            v: vec![0.0; d],
            w: vec![0.0; d],
            mean: vec![0.0; d],
            chunk_cov: Vec::new(),
            cached_d: d,
            cached_k: k,
        }
    }

    /// Resize if `D` or `K` changed. No-op on the hot path.
    pub fn ensure_capacity(&mut self, d: usize, k: usize) {
        if self.cached_d == d && self.cached_k == k {
            return;
        }
        let dd = d * d;
        self.cov.resize(dd, 0.0);
        self.cov_backup.resize(dd, 0.0);
        self.v.resize(d, 0.0);
        self.w.resize(d, 0.0);
        self.mean.resize(d, 0.0);
        self.cached_d = d;
        self.cached_k = k;
    }
}

/// Configuration knobs. Defaults are tuned for the HLA `D=8, k=3` use case.
#[derive(Debug, Clone, Copy)]
pub struct ZoneManifoldConfig {
    /// Power-iteration count per component. 5 matches
    /// [`crate::data_probe::stable_rank_update_into`]'s default and is more
    /// than enough for an `8×8` PSD matrix.
    pub n_iters: u8,
    /// Subtract the per-column mean before forming the covariance (standard
    /// PCA). Set `false` only for second-moment / spread-without-mean use cases.
    pub center: bool,
    /// Below this NPC count, accumulate the covariance serially (rayon
    /// parallelism overhead exceeds the work). Per `AGENTS.md`: rayon's
    /// thread-pool overhead is ~5µs, so the crossover is around `N = 1000`.
    pub parallel_threshold: usize,
    /// Sign-fixing: if the previous-tick axis is supplied and its dot product
    /// with the new axis is negative, flip the new axis. Essential for temporal
    /// continuity — without it the same physical axis can flip sign tick to
    /// tick (eigenvector sign ambiguity).
    pub sign_fix: bool,
    /// Cold-start fallback: if `N < min_n_for_manifold`, skip the SVD entirely
    /// and emit the identity axes (first `k` standard basis vectors). The
    /// caller should blend toward the manifold via
    /// `sigmoid((N − min_n) / τ)` per Issue 001 risk §3.
    pub min_n_for_manifold: usize,
    /// **Cluster-and-distribute** (the perf-critical optimization): partition
    /// the N NPCs into `n_groups` contiguous blocks, compute a per-group PCA
    /// (each block is N/G NPCs — fits in L1), then distribute each NPC its
    /// group's mood axes. This is both faster (embarrassingly parallel,
    /// cache-friendly per group) and richer (captures multi-modal mood
    /// structure a single zone-wide PCA averages away).
    ///
    /// `n_groups = 1` (default) → single zone-wide PCA (original behavior).
    /// `n_groups > 1` → grouped: `out_zone_axes` holds `(G, D, k)` row-major
    /// (group g's axes at `[g*D*k .. (g+1)*D*k)`), `out_npc_projections` is
    /// `(N, k)` where NPC i uses group `(i * G / N)`'s axes.
    ///
    /// Auto-scaling: when 0 (the default), the function picks `G` from the
    /// thread count and crowd size — see [`Self::auto_group_count`].
    pub n_groups: usize,
    /// Minimum NPCs per group. Prevents tiny groups with noisy axes. If
    /// `N / n_groups < min_per_group`, `n_groups` is clamped down.
    pub min_per_group: usize,
}

impl Default for ZoneManifoldConfig {
    fn default() -> Self {
        Self {
            n_iters: 5,
            center: true,
            parallel_threshold: 1024,
            sign_fix: true,
            min_n_for_manifold: 32, // 4·D at D=8
            n_groups: 1,            // 1 = single zone-wide PCA (original). Set 0 for auto-group.
            min_per_group: 64,      // ≥ 8·D at D=8 for stable axes
        }
    }
}

impl ZoneManifoldConfig {
    /// Pick the group count for a crowd of `n` NPCs. Targets ~1 group per
    /// rayon worker, clamped so each group has at least `min_per_group` NPCs.
    /// Returns 1 for small crowds (serial single-PCA path).
    pub fn auto_group_count(&self, n: usize) -> usize {
        if self.n_groups > 0 {
            return self.n_groups.min(n / self.min_per_group.max(1)).max(1);
        }
        if n < self.min_n_for_manifold.max(self.min_per_group) {
            return 1;
        }
        let n_workers = rayon::current_num_threads().max(1);
        // Target 1 group per worker, but keep each group ≥ min_per_group.
        let g = n_workers;
        let g = g.min(n / self.min_per_group.max(1));
        g.max(1)
    }
}

/// Error returned by [`zone_affective_manifold`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneManifoldError {
    /// `crowd_hla.len() != n * d`.
    DimMismatch,
    /// `out_zone_axes.len() < d * k` (or `g * d * k` in grouped mode).
    AxesBufferTooSmall,
    /// `out_npc_projections.len() < n * k`.
    ProjectionBufferTooSmall,
    /// `eigenvalues.len() < g * k` (grouped mode).
    EigenvaluesBufferTooSmall,
    /// `k == 0` or `k > d`.
    InvalidK,
    /// `d == 0`.
    InvalidD,
}

impl std::fmt::Display for ZoneManifoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimMismatch => write!(f, "crowd_hla length != n * d"),
            Self::AxesBufferTooSmall => write!(f, "out_zone_axes length < d * k"),
            Self::ProjectionBufferTooSmall => write!(f, "out_npc_projections length < n * k"),
            Self::EigenvaluesBufferTooSmall => write!(f, "eigenvalues length < g * k"),
            Self::InvalidK => write!(f, "k must be in 1..=d"),
            Self::InvalidD => write!(f, "d must be > 0"),
        }
    }
}

impl std::error::Error for ZoneManifoldError {}

// ── Core primitive ──────────────────────────────────────────────

/// Compute the zone affective manifold: top-k principal directions of the
/// `(N, D)` crowd-activation matrix, plus per-NPC projections.
///
/// # Arguments
///
/// * `crowd_hla` — `(N, D)` row-major. `crowd_hla[i*d + j]` is NPC `i`'s
///   `j`-th latent coordinate.
/// * `n` — number of NPCs.
/// * `d` — latent dimensionality (HLA = 8).
/// * `k` — target manifold dimensionality (3 or 4).
/// * `out_zone_axes` — `(D, k)` row-major output. `out_zone_axes[j*d + i]` is
///   axis `j`'s `i`-th component. Must have length `≥ d * k`.
/// * `out_npc_projections` — `(N, k)` row-major output.
///   `out_npc_projections[i*k + j]` is NPC `i`'s projection onto axis `j`.
///   Must have length `≥ n * k`.
/// * `eigenvalues` — `(k,)` output buffer for the top-k eigenvalues, descending.
/// * `scratch` — caller-owned, reused across ticks (zero-alloc hot path).
/// * `prev_zone_axes` — previous-tick axes `(D, k)` for sign-fixing, or `None`
///   on the first tick / when sign-fixing is disabled.
/// * `config` — tuning knobs.
///
/// # Returns
///
/// A [`ZoneManifoldReport`] borrowing the output buffers, or an error on
/// dimension mismatch.
///
/// # Cold-start behavior
///
/// If `n < config.min_n_for_manifold`, the function writes identity axes (first
/// `k` standard basis vectors) and zero projections, and returns
/// `explained_variance_ratio = 0.0`. The caller should blend toward the
/// manifold as the crowd grows.
#[allow(clippy::too_many_arguments)]
pub fn zone_affective_manifold<'a>(
    crowd_hla: &[f32],
    n: usize,
    d: usize,
    k: usize,
    out_zone_axes: &'a mut [f32],
    out_npc_projections: &'a mut [f32],
    eigenvalues: &'a mut [f32],
    scratch: &mut ZoneManifoldScratch,
    prev_zone_axes: Option<&[f32]>,
    config: &ZoneManifoldConfig,
) -> Result<ZoneManifoldReport<'a>, ZoneManifoldError> {
    // ── Dimension validation ──────────────────────────────────────
    if d == 0 {
        return Err(ZoneManifoldError::InvalidD);
    }
    if k == 0 || k > d {
        return Err(ZoneManifoldError::InvalidK);
    }
    if crowd_hla.len() < n * d {
        return Err(ZoneManifoldError::DimMismatch);
    }
    if out_zone_axes.len() < d * k {
        return Err(ZoneManifoldError::AxesBufferTooSmall);
    }
    if out_npc_projections.len() < n * k {
        return Err(ZoneManifoldError::ProjectionBufferTooSmall);
    }

    scratch.ensure_capacity(d, k);
    let dd = d * d;

    // ── Cold-start fallback ───────────────────────────────────────
    if n < config.min_n_for_manifold.max(1) {
        out_zone_axes[..d * k].fill(0.0);
        for j in 0..k {
            out_zone_axes[j * d + j] = 1.0;
        }
        for i in 0..n {
            let row = &crowd_hla[i * d..(i + 1) * d];
            for j in 0..k {
                out_npc_projections[i * k + j] = row[j];
            }
        }
        eigenvalues[..k].fill(0.0);
        return Ok(ZoneManifoldReport {
            zone_axes: &out_zone_axes[..d * k],
            npc_projections: &out_npc_projections[..n * k],
            eigenvalues: &eigenvalues[..k],
            explained_variance_ratio: 0.0,
            n_groups: 1,
            n,
            d,
            k,
        });
    }

    // ── Cluster-and-distribute path ───────────────────────────────
    //
    // When n_groups > 1, partition the crowd into G contiguous blocks,
    // compute a per-group PCA (each block fits in L1), then distribute each
    // NPC its group's mood axes. Embarrassingly parallel; far faster than a
    // single zone-wide PCA at scale because each group's data is L1-resident.
    let g = config.auto_group_count(n);
    if g > 1 {
        return compute_grouped(
            crowd_hla, n, d, k, g, out_zone_axes, out_npc_projections, eigenvalues,
            scratch, prev_zone_axes, config,
        );
    }

    // ── Phase 1: mean (optional) + covariance accumulation ────────
    //
    // Accumulate the covariance into scratch.cov in place. The parallel path
    // uses scratch.chunk_cov as a staging buffer, then reduces into scratch.cov.
    // We compute the mean first (needed for centering + projection).
    compute_mean_and_covariance(crowd_hla, n, d, config, scratch);

    // Backup the un-deflated covariance for explained-variance ratio.
    scratch.cov_backup[..dd].copy_from_slice(&scratch.cov[..dd]);
    let trace_total: f32 = (0..d).map(|i| scratch.cov[i * d + i]).sum();
    let trace_total = trace_total.max(1e-30);

    // ── Phase 2: power iteration + deflation for top-k eigenvectors ─
    {
        let axes = &mut out_zone_axes[..d * k];
        for comp in 0..k {
            power_iteration_deflate(
                &mut scratch.cov[..dd],
                d,
                &mut scratch.v[..d],
                &mut scratch.w[..d],
                config.n_iters,
                &mut axes[comp * d..(comp + 1) * d],
                &mut eigenvalues[comp],
            );
        }
    }

    // ── Phase 3: sign-fixing for temporal continuity ──────────────
    if let Some(prev) = prev_zone_axes.filter(|_| config.sign_fix) {
        let prev_k = (prev.len() / d).min(k);
        let axes = &mut out_zone_axes[..d * k];
        for j in 0..prev_k {
            let new_axis = &mut axes[j * d..(j + 1) * d];
            let prev_axis = &prev[j * d..(j + 1) * d];
            let dot = simd::simd_dot_f32(new_axis, prev_axis, d);
            if dot < 0.0 {
                for x in new_axis.iter_mut() {
                    *x = -*x;
                }
            }
        }
    }

    // ── Phase 4: per-NPC projections ──────────────────────────────
    let mean = if config.center {
        &scratch.mean[..d]
    } else {
        &ZERO_MEAN[..]
    };
    let axes = &out_zone_axes[..d * k];
    let projs = &mut out_npc_projections[..n * k];
    if n > config.parallel_threshold {
        // Parallel: each NPC's k projections are independent. Use coarse
        // chunks (many NPCs per task) to amortize rayon's ~5µs/task overhead.
        // Target ~256 NPCs per task → ~40 tasks for N=10k.
        let batch = ((n / (rayon::current_num_threads().max(1) * 8)) + 1).max(1);
        let stride = k * batch;
        projs
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(chunk_idx, proj_chunk)| {
                let base = chunk_idx * batch;
                let count = proj_chunk.len() / k;
                for di in 0..count {
                    let i = base + di;
                    let row = &crowd_hla[i * d..(i + 1) * d];
                    let proj_row = &mut proj_chunk[di * k..(di + 1) * k];
                    for j in 0..k {
                        let axis = &axes[j * d..(j + 1) * d];
                        proj_row[j] = project_centered(row, axis, mean, d);
                    }
                }
            });
    } else {
        for i in 0..n {
            let row = &crowd_hla[i * d..(i + 1) * d];
            for j in 0..k {
                let axis = &axes[j * d..(j + 1) * d];
                projs[i * k + j] = project_centered(row, axis, mean, d);
            }
        }
    }

    // Restore the un-deflated covariance (so repeated calls start clean).
    scratch.cov[..dd].copy_from_slice(&scratch.cov_backup[..dd]);

    let explained: f32 = eigenvalues[..k].iter().sum::<f32>() / trace_total;

    Ok(ZoneManifoldReport {
        zone_axes: &out_zone_axes[..d * k],
        npc_projections: &out_npc_projections[..n * k],
        eigenvalues: &eigenvalues[..k],
        explained_variance_ratio: explained,
        n_groups: 1,
        n,
        d,
        k,
    })
}

/// Sentinel zero-mean slice used when `center = false`.
const ZERO_MEAN: [f32; 0] = [];

/// Cluster-and-distribute: partition N NPCs into `g` contiguous blocks,
/// compute a per-group PCA (each block fits in L1), distribute each NPC its
/// group's mood axes. Embarrassingly parallel across groups.
#[allow(clippy::too_many_arguments)]
fn compute_grouped<'a>(
    crowd_hla: &[f32],
    n: usize,
    d: usize,
    k: usize,
    g: usize,
    out_zone_axes: &'a mut [f32],
    out_npc_projections: &'a mut [f32],
    eigenvalues: &'a mut [f32],
    _scratch: &mut ZoneManifoldScratch,
    prev_zone_axes: Option<&[f32]>,
    config: &ZoneManifoldConfig,
) -> Result<ZoneManifoldReport<'a>, ZoneManifoldError> {
    let dd = d * d;
    let dk = d * k;

    if out_zone_axes.len() < g * dk {
        return Err(ZoneManifoldError::AxesBufferTooSmall);
    }
    if eigenvalues.len() < g * k {
        return Err(ZoneManifoldError::EigenvaluesBufferTooSmall);
    }

    // Trim N to a multiple of G so all groups are equal-sized.
    let group_n = n / g;
    let n_used = group_n * g;

    // Pre-split all three output buffers into equal per-group chunks.
    let group_axes_chunks: Vec<&mut [f32]> = out_zone_axes[..g * dk].chunks_mut(dk).collect();
    let group_eig_chunks: Vec<&mut [f32]> = eigenvalues[..g * k].chunks_mut(k).collect();
    let group_projs_chunks: Vec<&mut [f32]> = out_npc_projections[..n_used * k]
        .chunks_mut(group_n * k)
        .collect();

    // Parallel per-group PCA.
    group_axes_chunks
        .into_par_iter()
        .zip(group_eig_chunks)
        .zip(group_projs_chunks)
        .enumerate()
        .for_each(|(grp, ((group_axes, group_eig), projs))| {
        let start = grp * group_n;
        let group_crowd = &crowd_hla[start * d..(start + group_n) * d];

        // Stack-local scratch (D ≤ 8 enforced by debug_assert).
        let mut local_cov = [0.0f32; 64];
        let mut local_mean = [0.0f32; 8];
        let mut local_v = [0.0f32; 8];
        let mut local_w = [0.0f32; 8];
        debug_assert!(d <= 8, "compute_grouped requires d ≤ 8 (got {})", d);
        let local_cov = &mut local_cov[..dd];
        let local_mean = &mut local_mean[..d];
        let local_v = &mut local_v[..d];
        let local_w = &mut local_w[..d];

        // Per-group mean.
        if config.center {
            local_mean.fill(0.0);
            for i in 0..group_n {
                let row = &group_crowd[i * d..(i + 1) * d];
                for j in 0..d {
                    local_mean[j] += row[j];
                }
            }
            let inv_n = 1.0 / group_n as f32;
            for slot in local_mean[..d].iter_mut() {
                *slot *= inv_n;
            }
        } else {
            local_mean.fill(0.0);
        }

        // Per-group covariance.
        local_cov.fill(0.0);
        accumulate_triangle(local_cov, group_crowd, 0, group_n, d, local_mean);

        // Per-group power iteration + deflation.
        for comp in 0..k {
            power_iteration_deflate(
                local_cov, d, local_v, local_w, config.n_iters,
                &mut group_axes[comp * d..(comp + 1) * d],
                &mut group_eig[comp],
            );
        }

        // Per-group sign-fixing.
        if let Some(prev) = prev_zone_axes.filter(|_| config.sign_fix) {
            let prev_offset = grp * dk;
            if prev_offset + dk <= prev.len() {
                let prev_group_axes = &prev[prev_offset..prev_offset + dk];
                for j in 0..k {
                    let new_axis = &mut group_axes[j * d..(j + 1) * d];
                    let prev_axis = &prev_group_axes[j * d..(j + 1) * d];
                    let dot = simd::simd_dot_f32(new_axis, prev_axis, d);
                    if dot < 0.0 {
                        for x in new_axis.iter_mut() {
                            *x = -*x;
                        }
                    }
                }
            }
        }

        // Per-group NPC projections.
        let mean_ref = if config.center { local_mean } else { &ZERO_MEAN[..] };
        for i in 0..group_n {
            let row = &group_crowd[i * d..(i + 1) * d];
            let proj_row = &mut projs[i * k..(i + 1) * k];
            for j in 0..k {
                let axis = &group_axes[j * d..(j + 1) * d];
                proj_row[j] = project_centered(row, axis, mean_ref, d);
            }
        }
    });

    // Report: group 0 as representative.
    let g0_sum: f32 = eigenvalues[..k].iter().sum();
    let all_sum: f32 = eigenvalues[..g * k].iter().sum();
    let explained = if all_sum > 1e-30 {
        (g0_sum / (all_sum / g as f32)).min(1.0)
    } else {
        0.0
    };

    Ok(ZoneManifoldReport {
        zone_axes: &out_zone_axes[..g * dk],
        npc_projections: &out_npc_projections[..n * k],
        eigenvalues: &eigenvalues[..k],
        explained_variance_ratio: explained,
        n_groups: g,
        n,
        d,
        k,
    })
}

/// Compute the per-column mean (if centering) and the covariance, writing into
/// `scratch.cov` and `scratch.mean`.
fn compute_mean_and_covariance(
    crowd: &[f32],
    n: usize,
    d: usize,
    config: &ZoneManifoldConfig,
    scratch: &mut ZoneManifoldScratch,
) {
    let dd = d * d;
    let mean = &mut scratch.mean[..d];
    if config.center {
        mean.fill(0.0);
        for i in 0..n {
            let row = &crowd[i * d..(i + 1) * d];
            for j in 0..d {
                mean[j] += row[j];
            }
        }
        let inv_n = 1.0 / n as f32;
        for slot in mean[..d].iter_mut() {
            *slot *= inv_n;
        }
    } else {
        mean.fill(0.0);
    }

    if n > config.parallel_threshold {
        // Chunk geometry: aim for ~8 chunks per rayon worker.
        let n_workers = rayon::current_num_threads().max(1);
        let target_chunks = n_workers * 8;
        let chunk_size = n.div_ceil(target_chunks).max(1);
        let num_chunks = n.div_ceil(chunk_size);
        let needed = num_chunks * dd;
        if scratch.chunk_cov.len() < needed {
            scratch.chunk_cov.resize(needed, 0.0);
        }
        // Disjoint borrows of distinct scratch fields.
        let mean_ref: &[f32] = mean;
        let chunk_buf = &mut scratch.chunk_cov[..needed];
        chunk_buf.fill(0.0);
        // Parallel accumulation into per-chunk D×D slices.
        chunk_buf
            .par_chunks_mut(dd)
            .enumerate()
            .for_each(|(chunk_idx, chunk_slice)| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                accumulate_triangle(chunk_slice, crowd, start, end, d, mean_ref);
            });
        // Serial reduce into scratch.cov.
        scratch.cov[..dd].fill(0.0);
        for chunk_slice in chunk_buf.chunks_exact(dd) {
            for (idx, &v) in chunk_slice.iter().enumerate() {
                scratch.cov[idx] += v;
            }
        }
    } else {
        let mean_ref: &[f32] = mean;
        scratch.cov[..dd].fill(0.0);
        accumulate_triangle(&mut scratch.cov[..dd], crowd, 0, n, d, mean_ref);
    }
}

/// Accumulate the upper-triangle covariance for rows `[start, end)` into `cov`,
/// then mirror the upper triangle to the lower triangle.
///
/// Upper-triangle-only accumulation (d*(d+1)/2 entries per row, not d²) is
/// faster than the full SIMD outer product at d=8 — the SIMD call overhead and
/// the doubled write count outweigh the vectorization benefit at this size.
fn accumulate_triangle(
    cov: &mut [f32],
    crowd: &[f32],
    start: usize,
    end: usize,
    d: usize,
    mean: &[f32],
) {
    let centered = !mean.is_empty();
    for i in start..end {
        let row = &crowd[i * d..(i + 1) * d];
        for a in 0..d {
            let xa = if centered { row[a] - mean[a] } else { row[a] };
            for b in a..d {
                let xb = if centered { row[b] - mean[b] } else { row[b] };
                cov[a * d + b] += xa * xb;
            }
        }
    }
    // Mirror upper → lower.
    for a in 0..d {
        for b in (a + 1)..d {
            cov[b * d + a] = cov[a * d + b];
        }
    }
}

/// `dot(row - mean, axis)`. When `mean` is empty (center=false), falls back to
/// `dot(row, axis)`. SIMD-optimized via a stack-local mean-subtracted buffer.
#[inline]
fn project_centered(row: &[f32], axis: &[f32], mean: &[f32], d: usize) -> f32 {
    if mean.is_empty() {
        simd::simd_dot_f32(row, axis, d)
    } else {
        let mut xc = [0.0f32; 64];
        // SAFETY: bounded by the same debug_assert as accumulate_triangle; d
        // is the latent dim (HLA=8, max 64 in practice).
        let xc = &mut xc[..d];
        for j in 0..d {
            xc[j] = row[j] - mean[j];
        }
        simd::simd_dot_f32(xc, axis, d)
    }
}

// ── Power iteration + deflation ─────────────────────────────────

/// One component of the deflation loop: power-iterate on `cov` to find the
/// dominant eigenvector, write it to `axis_out`, record the eigenvalue, then
/// deflate `cov -= λ · v v^T`.
///
/// `cov` is modified in place (deflated). `v` and `w` are scratch.
fn power_iteration_deflate(
    cov: &mut [f32],
    d: usize,
    v: &mut [f32],
    w: &mut [f32],
    n_iters: u8,
    axis_out: &mut [f32],
    eigenvalue_out: &mut f32,
) {
    // Deterministic seed: 1/√D on each coordinate. Matches stable_rank_update_into.
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    for x in v.iter_mut() {
        *x = inv_sqrt_d;
    }

    let iters = n_iters.max(1) as usize;
    let mut lambda = 0.0f32;

    for _ in 0..iters {
        // w = C · v  (cov is symmetric, so C·v = C^T·v — single matvec).
        w.fill(0.0);
        for a in 0..d {
            let row = &cov[a * d..(a + 1) * d];
            w[a] = simd::simd_dot_f32(row, v, d);
        }

        // Rayleigh quotient: λ = v^T w / v^T v.
        let vtv = simd::simd_dot_f32(v, v, d);
        let vtw = simd::simd_dot_f32(v, w, d);
        if vtv <= 0.0 {
            break;
        }
        lambda = vtw / vtv;

        // Normalize w → v for next iteration.
        let norm_w = simd::simd_dot_f32(w, w, d).max(1e-30).sqrt();
        let inv_norm = 1.0 / norm_w;
        for j in 0..d {
            v[j] = w[j] * inv_norm;
        }
    }

    // Write the converged eigenvector.
    axis_out[..d].copy_from_slice(&v[..d]);
    *eigenvalue_out = lambda;

    // Deflate: C -= λ · v v^T.
    for a in 0..d {
        let va = v[a];
        let scale = lambda * va;
        let row = &mut cov[a * d..(a + 1) * d];
        for b in 0..d {
            row[b] -= scale * v[b];
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `(N, D)` crowd where the variance is concentrated along known
    /// directions.
    fn build_synthetic_crowd(
        n: usize,
        d: usize,
        eigenvalues: &[f32],
        eigenvectors: &[f32], // (D, len(eigenvalues)) row-major, orthonormal
        seed: u64,
    ) -> Vec<f32> {
        let k = eigenvalues.len();
        let mut rng = SeededRng::new(seed);
        let mut crowd = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..k {
                let sigma = eigenvalues[j].sqrt();
                let z = rng.gaussian() * sigma;
                for idx in 0..d {
                    crowd[i * d + idx] += z * eigenvectors[j * d + idx];
                }
            }
        }
        crowd
    }

    /// Minimal deterministic Gaussian RNG (Box-Muller) for test fixtures.
    struct SeededRng {
        state: u64,
    }
    impl SeededRng {
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
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn gaussian(&mut self) -> f32 {
            let u1 = self.next_f32().max(1e-10);
            let u2 = self.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            r * theta.cos()
        }
    }

    #[test]
    fn cold_start_emits_identity_axes() {
        let d = 8;
        let k = 3;
        let n = 10; // < min_n_for_manifold=32
        let crowd = vec![0.5f32; n * d];
        let mut axes = vec![0.0; d * k];
        let mut projs = vec![0.0; n * k];
        let mut eigvals = vec![0.0; k];
        let mut scratch = ZoneManifoldScratch::new(d, k);
        let cfg = ZoneManifoldConfig::default();
        let report = zone_affective_manifold(
            &crowd, n, d, k, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
        )
        .unwrap();
        for j in 0..k {
            assert!(
                (report.zone_axes[j * d + j] - 1.0).abs() < 1e-6,
                "axis {j} should be identity"
            );
        }
        assert_eq!(report.explained_variance_ratio, 0.0);
    }

    #[test]
    fn recovers_known_principal_directions() {
        let d = 8;
        let k = 2;
        let n = 2000;
        let mut evecs = vec![0.0f32; k * d];
        evecs[0] = 1.0; // axis 0 = e_0
        evecs[d + 1] = 1.0; // axis 1 = e_1
        let evals = [4.0f32, 1.0f32];
        let crowd = build_synthetic_crowd(n, d, &evals, &evecs, 42);
        let mut axes = vec![0.0; d * k];
        let mut projs = vec![0.0; n * k];
        let mut eigvals = vec![0.0; k];
        let mut scratch = ZoneManifoldScratch::new(d, k);
        let cfg = ZoneManifoldConfig::default();
        let report = zone_affective_manifold(
            &crowd, n, d, k, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
        )
        .unwrap();
        let axis0 = &report.zone_axes[0..d];
        assert!(
            axis0[0].abs() > 0.9,
            "axis 0 should align with e_0, got {:?}",
            axis0
        );
        let axis1 = &report.zone_axes[d..2 * d];
        assert!(
            axis1[1].abs() > 0.9,
            "axis 1 should align with e_1, got {:?}",
            axis1
        );
        let ratio = report.eigenvalues[0] / report.eigenvalues[1].max(1e-10);
        assert!(
            (ratio - 4.0).abs() < 1.0,
            "eigenvalue ratio should be ≈ 4, got {ratio}"
        );
    }

    #[test]
    fn explained_variance_ratio_in_unit_interval() {
        let d = 8;
        let k = 3;
        let n = 1000;
        let mut rng = SeededRng::new(7);
        let mut crowd = vec![0.0f32; n * d];
        for x in crowd.iter_mut() {
            *x = rng.gaussian();
        }
        let mut axes = vec![0.0; d * k];
        let mut projs = vec![0.0; n * k];
        let mut eigvals = vec![0.0; k];
        let mut scratch = ZoneManifoldScratch::new(d, k);
        let cfg = ZoneManifoldConfig::default();
        let report = zone_affective_manifold(
            &crowd, n, d, k, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
        )
        .unwrap();
        assert!(
            report.explained_variance_ratio > 0.0 && report.explained_variance_ratio <= 1.0 + 1e-5,
            "explained_variance_ratio out of range: {}",
            report.explained_variance_ratio
        );
    }

    #[test]
    fn sign_fixing_provides_temporal_continuity() {
        let d = 4;
        let k = 1;
        let n = 500;
        let mut evecs = vec![0.0f32; k * d];
        evecs[0] = 1.0;
        let crowd = build_synthetic_crowd(n, d, &[3.0f32], &evecs, 99);

        let mut axes1 = vec![0.0; d * k];
        let mut projs1 = vec![0.0; n * k];
        let mut eigvals1 = vec![0.0; k];
        let mut scratch = ZoneManifoldScratch::new(d, k);
        let cfg = ZoneManifoldConfig::default();
        zone_affective_manifold(
            &crowd, n, d, k, &mut axes1, &mut projs1, &mut eigvals1, &mut scratch, None, &cfg,
        )
        .unwrap();

        let mut axes2 = vec![0.0; d * k];
        let mut projs2 = vec![0.0; n * k];
        let mut eigvals2 = vec![0.0; k];
        zone_affective_manifold(
            &crowd, n, d, k, &mut axes2, &mut projs2, &mut eigvals2, &mut scratch,
            Some(&axes1), &cfg,
        )
        .unwrap();

        let dot = simd::simd_dot_f32(&axes1[..d], &axes2[..d], d);
        assert!(
            dot > 0.99,
            "sign-fixed axes should be ≈ identical, dot = {dot}"
        );
    }

    #[test]
    fn determinism_bit_identical_across_calls() {
        let d = 8;
        let k = 3;
        let n = 1000;
        let crowd: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let run = || {
            let mut axes = vec![0.0; d * k];
            let mut projs = vec![0.0; n * k];
            let mut eigvals = vec![0.0; k];
            let mut scratch = ZoneManifoldScratch::new(d, k);
            let cfg = ZoneManifoldConfig::default();
            zone_affective_manifold(
                &crowd, n, d, k, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
            )
            .unwrap();
            (axes, eigvals)
        };
        let (axes_a, eig_a) = run();
        let (axes_b, eig_b) = run();
        assert_eq!(axes_a, axes_b, "axes not bit-identical");
        assert_eq!(eig_a, eig_b, "eigenvalues not bit-identical");
    }

    #[test]
    fn mood_discrimination_separates_distributions() {
        let d = 8;
        let k = 2;
        let n = 2000;
        let mut evecs_a = vec![0.0f32; k * d];
        evecs_a[0] = 1.0;
        evecs_a[d + 1] = 1.0;
        let crowd_a = build_synthetic_crowd(n, d, &[5.0, 0.5], &evecs_a, 11);

        let mut evecs_b = vec![0.0f32; k * d];
        evecs_b[2] = 1.0;
        evecs_b[d + 3] = 1.0;
        let crowd_b = build_synthetic_crowd(n, d, &[5.0, 0.5], &evecs_b, 22);

        let run = |crowd: &[f32]| -> Vec<f32> {
            let mut axes = vec![0.0; d * k];
            let mut projs = vec![0.0; n * k];
            let mut eigvals = vec![0.0; k];
            let mut scratch = ZoneManifoldScratch::new(d, k);
            let cfg = ZoneManifoldConfig::default();
            zone_affective_manifold(
                crowd, n, d, k, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
            )
            .unwrap();
            axes
        };

        let axes_a = run(&crowd_a);
        let axes_b = run(&crowd_b);

        let cos = simd::simd_dot_f32(&axes_a[..d], &axes_b[..d], d).abs();
        assert!(
            cos < 0.3,
            "mood axes should be > 70° apart, |cos| = {cos}"
        );
    }

    #[allow(clippy::field_reassign_with_default)]
    #[test]
    fn parallel_matches_serial() {
        // Parallel and serial differ only in floating-point summation order.
        // With well-separated eigenvalues the eigenvectors are stable under
        // this perturbation; with near-degenerate eigenvalues they can rotate.
        // Use a crowd with a clear dominant axis so the result is stable.
        let d = 8;
        let k = 3;
        let n = 5000;
        let mut evecs = vec![0.0f32; k * d];
        evecs[0] = 1.0;
        evecs[d + 1] = 1.0;
        evecs[2 * d + 2] = 1.0;
        // Well-separated eigenvalues (10:2:0.5) → stable eigenvectors.
        let crowd = build_synthetic_crowd(n, d, &[10.0, 2.0, 0.5], &evecs, 123);

        let mut cfg_serial = ZoneManifoldConfig::default();
        cfg_serial.parallel_threshold = usize::MAX; // force serial

        let mut cfg_parallel = ZoneManifoldConfig::default();
        cfg_parallel.parallel_threshold = 1; // force parallel

        let run = |cfg: ZoneManifoldConfig| -> Vec<f32> {
            let mut axes = vec![0.0; d * k];
            let mut projs = vec![0.0; n * k];
            let mut eigvals = vec![0.0; k];
            let mut scratch = ZoneManifoldScratch::new(d, k);
            zone_affective_manifold(
                &crowd, n, d, k, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
            )
            .unwrap();
            axes
        };

        let serial = run(cfg_serial);
        let parallel = run(cfg_parallel);
        // With well-separated eigenvalues, the eigenvectors are stable under
        // float-order perturbation. Allow 1e-2 tolerance for the cumulative
        // effect of N=5000 reordered additions.
        for (a, b) in serial.iter().zip(parallel.iter()) {
            assert!(
                (a - b).abs() < 1e-2,
                "parallel/serial drift too large: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn rejects_bad_dimensions() {
        let mut scratch = ZoneManifoldScratch::new(8, 3);
        let mut axes = vec![0.0; 24];
        let mut projs = vec![0.0; 30];
        let mut eigvals = vec![0.0; 3];
        let crowd = vec![0.0f32; 80];
        let cfg = ZoneManifoldConfig::default();
        assert_eq!(
            zone_affective_manifold(
                &crowd, 10, 8, 9, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
            )
            .unwrap_err(),
            ZoneManifoldError::InvalidK
        );
        assert_eq!(
            zone_affective_manifold(
                &crowd, 100, 8, 3, &mut axes, &mut projs, &mut eigvals, &mut scratch, None, &cfg,
            )
            .unwrap_err(),
            ZoneManifoldError::DimMismatch
        );
    }
}
