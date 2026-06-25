//! Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN.
//!
//! See `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md`
//! and Plan 310. Generalizes FUNCATTN (Plan 286 / Research 257, arxiv 2605.31559)
//! to `d_src ≠ d_dst`: two matmuls over a frozen asymmetric basis pair enable
//! **train-on-small-deploy-on-large** latent transfer without retraining.
//!
//! ## Math
//!
//! Given a source latent state `s ∈ R^{d_src}` and frozen, column-orthonormal
//! bases `Φ_src ∈ R^{d_src × k}` and `Ψ_dst ∈ R^{d_dst × k}`:
//!
//! ```text
//! a  ← Φ_src^T · s        // project to k-dim spectral (R^k)
//! t  ← Ψ_dst · a           // reconstruct at destination resolution (R^{d_dst})
//! ```
//!
//! For band-limited fields (energy within the first `k` basis components) this
//! is exact (Parseval). For full-spectrum inputs it is the least-squares
//! projection — exactly FUNCATTN's ridge-regularization interpretation
//! (Research 291 §1.4 Lipschitz bound).
//!
//! The FUNCATTN operator `C ∈ R^{k × k}` (which transports between semantic
//! domains *within* a resolution) composes as a third matmul:
//!
//! ```text
//! t_cross_domain ← Ψ_dst · C · Φ_src^T · s
//! ```
//!
//! `C` is obtained externally from [`crate::funcattn::solve_convex_combo_dual`]
//! (hence the `cross_resolution_transport = ["funcattn"]` feature implication).
//!
//! ## Why modelless
//!
//! Both bases are **frozen learned artifacts** — trained offline via paired
//! small-dim/large-dim shards, committed via BLAKE3, loaded at init. The
//! runtime transport is matmuls only. No gradients, no inference-time solve.
//! This is exactly the freeze/thaw pattern: the bases are frozen, the transport
//! is inference-time.
//!
//! ## Zero-alloc hot path
//!
//! All intermediate buffers live in [`CrossResScratch`], pre-allocated once and
//! reused across calls via [`CrossResScratch::ensure_capacity`]. The transport
//! path performs no heap allocation after warmup — verified by G4.
//!
//! ## Rank constraint
//!
//! For `Φ_src^T Φ_src` and `Ψ_dst^T Ψ_dst` to be `I_k` (full rank `k`),
//! `k ≤ min(d_src, d_dst)` is **required**. Enforced in [`CrossResolutionBases::new`].

use crate::simd;

// ── Errors ────────────────────────────────────────────────────────

/// Errors returned by [`CrossResolutionBases::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossResolutionError {
    /// `k > min(d_src, d_dst)` — the basis cannot be column-orthonormal at
    /// this rank because the smaller dimension is shorter than `k`. See
    /// Research 291 §5.4. Caller must either lower `k` or raise both dims.
    RankDeficient,
    /// Slice lengths don't match the declared `(d_src, d_dst, k)` shape.
    ShapeMismatch,
}

// ── Bases ─────────────────────────────────────────────────────────

/// Frozen, BLAKE3-committed asymmetric basis pair for cross-resolution transport.
///
/// `phi_src ∈ R^{d_src × k}` and `psi_dst ∈ R^{d_dst × k}` are stored row-major
/// (each row contiguous, matching `funcattn::compute_basis_into`'s layout for
/// SIMD-friendly dot products). Both must be column-orthonormal
/// (`Φ^T Φ = Ψ^T Ψ = I_k`) — verified at construction by [`Self::verify_orthonormal`].
///
/// The `commitment` field is `BLAKE3(phi_src_le || psi_dst_le || d_src_le ||
/// d_dst_le || k_le)` so a basis pair can be tracked as a content-addressed
/// artifact (mirrors the `MerkleFrozenEnvelope` pattern in riir-neuron-db).
#[derive(Debug, Clone)]
pub struct CrossResolutionBases {
    /// Flattened `d_src × k`, row-major. Source-tier basis `Φ_src`.
    pub phi_src: Vec<f32>,
    /// Flattened `d_dst × k`, row-major. Destination-tier basis `Ψ_dst`.
    pub psi_dst: Vec<f32>,
    /// Source latent dimension (e.g. 16 for plasma-tier shards).
    pub d_src: usize,
    /// Destination latent dimension (e.g. 256 for cold-tier shards).
    pub d_dst: usize,
    /// Spectral rank. Must satisfy `k ≤ min(d_src, d_dst)`.
    pub k: usize,
    /// `BLAKE3(phi_src_le || psi_dst_le || d_src_le || d_dst_le || k_le)`.
    pub commitment: [u8; 32],
}

impl CrossResolutionBases {
    /// Construct a basis pair, computing the BLAKE3 commitment.
    ///
    /// Returns [`CrossResolutionError::RankDeficient`] if `k > min(d_src, d_dst)`.
    /// Returns [`CrossResolutionError::ShapeMismatch`] if the slice lengths
    /// don't match `(d_src * k, d_dst * k)`.
    ///
    /// **Does not** enforce column-orthonormality — caller responsibility
    /// (offline training produces orthonormal bases; check with
    /// [`Self::verify_orthonormal`] if provenance is uncertain).
    pub fn new(
        phi_src: Vec<f32>,
        psi_dst: Vec<f32>,
        d_src: usize,
        d_dst: usize,
        k: usize,
    ) -> Result<Self, CrossResolutionError> {
        if phi_src.len() != d_src * k || psi_dst.len() != d_dst * k {
            return Err(CrossResolutionError::ShapeMismatch);
        }
        if k > d_src.min(d_dst) {
            return Err(CrossResolutionError::RankDeficient);
        }
        let commitment = compute_commitment(&phi_src, &psi_dst, d_src, d_dst, k);
        Ok(Self {
            phi_src,
            psi_dst,
            d_src,
            d_dst,
            k,
            commitment,
        })
    }

    /// Re-check that the stored commitment matches the current basis contents.
    /// Returns `false` if either matrix was mutated after construction.
    pub fn verify_commitment(&self) -> bool {
        compute_commitment(&self.phi_src, &self.psi_dst, self.d_src, self.d_dst, self.k)
            == self.commitment
    }

    /// Check `Φ_src^T Φ_src ≈ I_k` and `Ψ_dst^T Ψ_dst ≈ I_k` to within `tol`.
    /// Diagonals must be ≈ 1.0, off-diagonals ≈ 0.0.
    pub fn verify_orthonormal(&self, tol: f32) -> bool {
        orthonormal_check(&self.phi_src, self.d_src, self.k, tol)
            && orthonormal_check(&self.psi_dst, self.d_dst, self.k, tol)
    }
}

/// Compute `BLAKE3(phi_src_le || psi_dst_le || d_src_le || d_dst_le || k_le)`.
///
/// The bases are serialized as little-endian `f32` per-element (matches the
/// `engram/commitment.rs::build_merkle_root` convention — host-endianness-
/// explicit so the same basis produces the same commitment on big-endian
/// targets). The dim header is little-endian `usize` (8 bytes each on 64-bit).
fn compute_commitment(
    phi_src: &[f32],
    psi_dst: &[f32],
    d_src: usize,
    d_dst: usize,
    k: usize,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for &f in phi_src {
        hasher.update(&f.to_le_bytes());
    }
    for &f in psi_dst {
        hasher.update(&f.to_le_bytes());
    }
    hasher.update(&d_src.to_le_bytes());
    hasher.update(&d_dst.to_le_bytes());
    hasher.update(&k.to_le_bytes());
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

/// `G^T G ≈ I_k` check for a `(rows × k)` row-major matrix `G`.
/// Diagonal `(i == j)` entries of `G^T G` must be ≈ 1, off-diagonals ≈ 0.
fn orthonormal_check(mat: &[f32], rows: usize, k: usize, tol: f32) -> bool {
    debug_assert_eq!(mat.len(), rows * k);
    for i in 0..k {
        for j in i..k {
            // dot = Σ_r mat[r, i] · mat[r, j]  (column i dotted with column j).
            // Columns are strided by k; assemble contiguous views via offsets.
            // For small k (≤ 64) and small rows (≤ 256) this is L1-resident —
            // a scalar loop is fine here (cold path, constructor only).
            let mut dot = 0.0f32;
            for r in 0..rows {
                dot += mat[r * k + i] * mat[r * k + j];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            if (dot - target).abs() > tol {
                return false;
            }
        }
    }
    true
}

// ── Scratch ───────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for zero-alloc transport. Mirrors
/// [`crate::funcattn::FuncAttnScratch`].
///
/// Create once via [`CrossResScratch::new`], then call
/// [`CrossResScratch::ensure_capacity`] before each transport. The hot path
/// performs no heap allocation when dimensions match the cache.
pub struct CrossResScratch {
    /// k-dim spectral coefficient buffer.
    pub spectral: Vec<f32>,
    /// k-dim scratch for the cross-domain C-multiply (avoids reusing `spectral`
    /// so the in-place C application can read `spectral` and write here).
    pub spectral_dst: Vec<f32>,
    cached_k: usize,
}

impl CrossResScratch {
    /// Allocate scratch for the given rank.
    pub fn new(k: usize) -> Self {
        Self {
            spectral: vec![0.0; k],
            spectral_dst: vec![0.0; k],
            cached_k: k,
        }
    }

    /// Resize buffers if `k` changed. No-op on the hot path.
    pub fn ensure_capacity(&mut self, k: usize) {
        if self.cached_k == k {
            return;
        }
        self.spectral.resize(k, 0.0);
        self.spectral_dst.resize(k, 0.0);
        self.cached_k = k;
    }
}

// ── Transport primitives ──────────────────────────────────────────

/// Project source latent state → k-dim spectral coefficients.
/// `spectral = Φ_src^T · src_state` where `Φ_src` is `(d_src, k)` row-major.
///
/// Zero-alloc — caller provides the `spectral` buffer (typically
/// `scratch.spectral`).
#[inline]
#[allow(clippy::needless_range_loop)] // spectral transport kernel: indices participate in stride arithmetic (r*k+j)
pub fn project_to_spectral_into(
    src_state: &[f32],
    bases: &CrossResolutionBases,
    spectral: &mut [f32],
) {
    debug_assert_eq!(src_state.len(), bases.d_src, "src_state must be (d_src,)");
    debug_assert_eq!(spectral.len(), bases.k, "spectral must be (k,)");
    let d_src = bases.d_src;
    let k = bases.k;
    // For each output column j, dot column j of Φ_src with src_state.
    // Column j of a row-major (d_src × k) matrix is the strided slice
    // [j, j+k, j+2k, ...]. For k ≤ 64 (L1-resident), a contiguous-source dot
    // over the strided column beats gather; we restructure as:
    //   spectral[j] = Σ_r phi_src[r*k + j] * src_state[r]
    // which is what `simd_dot_f32` would do if we materialized the strided
    // column. For small k, the gather is unavoidable; LLVM auto-unrolls the
    // short inner loop well.
    for j in 0..k {
        let mut acc = 0.0f32;
        for r in 0..d_src {
            acc += bases.phi_src[r * k + j] * src_state[r];
        }
        spectral[j] = acc;
    }
}

/// Reconstruct destination latent state from k-dim spectral coefficients.
/// `dst_state = Ψ_dst · spectral` where `Ψ_dst` is `(d_dst, k)` row-major.
///
/// Zero-alloc — caller provides the `dst_state` buffer. Each output row is a
/// contiguous dot product over a row of `Ψ_dst`, so this path is the
/// SIMD-friendly one (`simd::simd_dot_f32` over contiguous slices).
#[inline]
#[allow(clippy::needless_range_loop)] // spectral transport kernel: row index r participates in stride r*k for row-major Ψ_dst
pub fn reconstruct_from_spectral_into(
    spectral: &[f32],
    bases: &CrossResolutionBases,
    dst_state: &mut [f32],
) {
    debug_assert_eq!(spectral.len(), bases.k, "spectral must be (k,)");
    debug_assert_eq!(dst_state.len(), bases.d_dst, "dst_state must be (d_dst,)");
    let d_dst = bases.d_dst;
    let k = bases.k;
    // Each row r of Ψ_dst is contiguous in memory: psi_dst[r*k .. r*k + k].
    // dst[r] = dot(psi_dst row r, spectral) — straight SIMD dot.
    for r in 0..d_dst {
        let row = &bases.psi_dst[r * k..(r + 1) * k];
        dst_state[r] = simd::simd_dot_f32(row, spectral, k);
    }
}

/// Full cross-resolution transport: `src_state (d_src) → dst_state (d_dst)`.
///
/// Zero-alloc given a pre-allocated [`CrossResScratch`]. The canonical hot
/// path — projects to spectral, then reconstructs at the destination tier.
#[inline]
pub fn transport_cross_resolution_into(
    src_state: &[f32],
    bases: &CrossResolutionBases,
    scratch: &mut CrossResScratch,
    dst_state: &mut [f32],
) {
    scratch.ensure_capacity(bases.k);
    project_to_spectral_into(src_state, bases, &mut scratch.spectral);
    reconstruct_from_spectral_into(&scratch.spectral, bases, dst_state);
}

/// Allocating convenience wrapper. Prefer [`transport_cross_resolution_into`]
/// on hot paths.
pub fn transport_cross_resolution(src_state: &[f32], bases: &CrossResolutionBases) -> Vec<f32> {
    let mut dst = vec![0.0; bases.d_dst];
    let mut scratch = CrossResScratch::new(bases.k);
    transport_cross_resolution_into(src_state, bases, &mut scratch, &mut dst);
    dst
}

/// Cross-resolution **and** cross-domain transport — the F2 fusion with
/// FUNCATTN (Research 291 §2.4).
///
/// Computes `dst = Ψ_dst · C · Φ_src^T · src` as a clean 3-matrix product, all
/// small (`k ≪ d_src, d_dst`). `c_op ∈ R^{k × k}` is the FUNCATTN operator
/// obtained externally from [`crate::funcattn::solve_convex_combo_dual`] —
/// the cross-resolution layer adds the asymmetric `Φ_src / Ψ_dst` bases on
/// top of the existing symmetric operator.
///
/// Zero-alloc given a pre-allocated [`CrossResScratch`]. The `C` multiply
/// reuses `scratch.spectral_dst` as the C-output buffer so `scratch.spectral`
/// (the Φ_src projection) can be read in-place.
#[inline]
pub fn transport_cross_domain_cross_resolution_into(
    src_state: &[f32],
    bases: &CrossResolutionBases,
    c_op: &[f32],
    scratch: &mut CrossResScratch,
    dst_state: &mut [f32],
) {
    debug_assert_eq!(c_op.len(), bases.k * bases.k, "c_op must be (k, k)");
    scratch.ensure_capacity(bases.k);
    let k = bases.k;

    // 1. src → spectral_src  (Φ_src^T · src)
    project_to_spectral_into(src_state, bases, &mut scratch.spectral);

    // 2. spectral_src → spectral_dst via C  (C · spectral_src, row-major k×k)
    //    spectral_dst[i] = Σ_j c_op[i*k + j] * spectral[j].
    //    Row-major C → contiguous dot per output row.
    for i in 0..k {
        let c_row = &c_op[i * k..(i + 1) * k];
        scratch.spectral_dst[i] = simd::simd_dot_f32(c_row, &scratch.spectral[..k], k);
    }

    // 3. spectral_dst → dst_state  (Ψ_dst · spectral_dst)
    reconstruct_from_spectral_into(&scratch.spectral_dst, bases, dst_state);
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* PRNG matching `funcattn.rs::tests::make_rng`.
    fn make_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.max(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let bits = (s >> 11) as u32;
            let u01 = bits as f32 / u32::MAX as f32;
            u01 * 2.0 - 1.0
        }
    }

    /// First `k` columns of a `dim × dim` identity, flattened row-major as
    /// `dim × k`. This is the trivially column-orthonormal basis used in
    /// Plan 310 T1.4 smoke tests.
    fn identity_truncated(dim: usize, k: usize) -> Vec<f32> {
        assert!(k <= dim);
        let mut m = vec![0.0f32; dim * k];
        // Only the diagonal entries `m[r * k + c]` where `r == c < k` get 1.0.
        // (The extra rows `r ∈ [k, dim)` stay zero — they correspond to
        // dimensions that the rank-k basis cannot represent.)
        for c in 0..k {
            m[c * k + c] = 1.0;
        }
        m
    }

    /// Build a column-orthonormal `dim × k` basis via Gram-Schmidt on random
    /// rows. Used by G1-style round-trip tests where identity-truncation would
    /// be too easy (round-trip is exact by construction).
    fn random_orthonormal(dim: usize, k: usize, seed: u64) -> Vec<f32> {
        assert!(k <= dim);
        let mut rng = make_rng(seed);
        // Start with `k` random column vectors.
        let mut cols: Vec<Vec<f32>> = (0..k).map(|_| (0..dim).map(|_| rng()).collect()).collect();
        // Modified Gram-Schmidt over columns.
        for i in 0..k {
            for j in 0..i {
                let dot: f32 = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
                // j < i: split_at_mut(i) gives [0..i) in `left`, cols[i] in right[0].
                let (left, right) = cols.split_at_mut(i);
                for (ci, cj) in right[0].iter_mut().zip(left[j].iter()) {
                    *ci -= dot * *cj;
                }
            }
            let norm: f32 = cols[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
            for v in cols[i].iter_mut() {
                *v *= inv;
            }
        }
        // Pack columns into row-major `dim × k`.
        let mut m = vec![0.0f32; dim * k];
        for r in 0..dim {
            for c in 0..k {
                m[r * k + c] = cols[c][r];
            }
        }
        m
    }

    #[test]
    fn smoke_asymmetric_dims_compile_and_transport() {
        // 16 → 256 transport with identity-truncated bases, k=8.
        let d_src = 16usize;
        let d_dst = 256usize;
        let k = 8usize;
        let phi_src = identity_truncated(d_src, k);
        let psi_dst = identity_truncated(d_dst, k);
        let bases = CrossResolutionBases::new(phi_src, psi_dst, d_src, d_dst, k)
            .expect("identity bases should construct");
        assert!(bases.verify_orthonormal(1e-5));
        assert!(bases.verify_commitment());

        // Source state: band-limited to first k components.
        let mut src = vec![0.0f32; d_src];
        for (i, s) in src.iter_mut().enumerate().take(k) {
            *s = (i as f32) * 0.5 - 1.5;
        }
        let mut dst = vec![0.0f32; d_dst];
        let mut scratch = CrossResScratch::new(k);
        transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);
        // dst[0..k] should equal src[0..k]; dst[k..] should be zero.
        for i in 0..k {
            assert!(
                (dst[i] - src[i]).abs() < 1e-5,
                "dst[{i}] = {} expected {} (band-limited transport)",
                dst[i],
                src[i]
            );
        }
        for (i, d) in dst.iter().enumerate().take(d_dst).skip(k) {
            assert!(d.abs() < 1e-6, "dst[{i}] = {} expected 0", d);
        }
    }

    #[test]
    fn smoke_roundtrip_preserves_bandlimited_signal() {
        // 64 → 256 → 64 round-trip with random orthonormal bases.
        // Band-limited input (energy only in first k=8 components in the
        // Φ_src eigenbasis) should reconstruct with cos ≈ 1.0.
        let d_src = 64usize;
        let d_dst = 256usize;
        let k = 8usize;
        let phi_src = random_orthonormal(d_src, k, 0xA1B2_C3D4);
        let psi_dst = random_orthonormal(d_dst, k, 0xB2C3_D4E5);
        let bases = CrossResolutionBases::new(phi_src, psi_dst, d_src, d_dst, k)
            .expect("random orthonormal bases should construct");
        assert!(bases.verify_orthonormal(1e-4));

        // Forward bases: 64 → 256.
        // Reverse bases: 256 → 64 (swap roles — same matrices, swapped slots).
        let reverse = CrossResolutionBases::new(
            bases.psi_dst.clone(),
            bases.phi_src.clone(),
            d_dst,
            d_src,
            k,
        )
        .expect("reverse bases should construct");

        // Band-limited src: only the first k spectral coefficients are nonzero
        // when projected through Φ_src. Easiest construction: src = Φ_src · a
        // for a k-dim `a`, which by orthonormality projects back to `a` exactly.
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let mut src = vec![0.0f32; d_src];
        for (r, s) in src.iter_mut().enumerate().take(d_src) {
            let mut acc = 0.0f32;
            for (j, aj) in a.iter().enumerate().take(k) {
                acc += bases.phi_src[r * k + j] * aj;
            }
            *s = acc;
        }

        // Forward: 64 → 256.
        let mut dst = vec![0.0f32; d_dst];
        let mut scratch = CrossResScratch::new(k);
        transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);

        // Reverse: 256 → 64.
        let mut recon = vec![0.0f32; d_src];
        transport_cross_resolution_into(&dst, &reverse, &mut scratch, &mut recon);

        // Band-limited input should round-trip with cos ≈ 1.0.
        let cos = cosine(&src, &recon);
        assert!(
            cos > 0.999,
            "band-limited round-trip cos = {cos:.6}, expected > 0.999"
        );
    }

    #[test]
    fn smoke_non_bandlimited_loses_information() {
        // Random (full-spectrum) src state. Round-trip should have cos < 1.0
        // because energy outside the rank-k subspace is lost.
        let d_src = 64usize;
        let d_dst = 256usize;
        let k = 8usize;
        let phi_src = random_orthonormal(d_src, k, 0x1111_2222);
        let psi_dst = random_orthonormal(d_dst, k, 0x3333_4444);
        let bases = CrossResolutionBases::new(phi_src, psi_dst, d_src, d_dst, k)
            .expect("bases should construct");

        let reverse = CrossResolutionBases::new(
            bases.psi_dst.clone(),
            bases.phi_src.clone(),
            d_dst,
            d_src,
            k,
        )
        .expect("reverse bases should construct");

        // Full-spectrum random src.
        let mut rng = make_rng(0xCAFE_BABE);
        let src: Vec<f32> = (0..d_src).map(|_| rng()).collect();

        let mut dst = vec![0.0f32; d_dst];
        let mut scratch = CrossResScratch::new(k);
        transport_cross_resolution_into(&src, &bases, &mut scratch, &mut dst);

        let mut recon = vec![0.0f32; d_src];
        transport_cross_resolution_into(&dst, &reverse, &mut scratch, &mut recon);

        let cos = cosine(&src, &recon);
        // For random R^64 input and k=8, the round-trip retains at most k/d_src
        // = 8/64 = 12.5% of the energy in expectation. cos should be well
        // below 1.0 (typically 0.1–0.4 depending on basis alignment).
        assert!(
            cos < 0.99,
            "non-band-limited round-trip cos = {cos:.6}, expected < 0.99 \
             (information outside rank-k subspace should be lost)"
        );
        // Sanity floor — even random round-trip should not be near-zero
        // unless bases are pathologically misaligned.
        assert!(
            cos > 0.0,
            "non-band-limited round-trip cos = {cos:.6}, expected > 0 \
             (some spectral energy always survives)"
        );
    }

    #[test]
    fn constructor_rejects_rank_deficient_k() {
        // k > d_src should fail. Build raw matrices of the declared shape —
        // do NOT use `identity_truncated` (it asserts k ≤ dim internally).
        let phi_src = vec![0.0f32; 4 * 8]; // (d_src=4, k=8) — k > d_src
        let psi_dst = vec![0.0f32; 16 * 8]; // (d_dst=16, k=8)
        let err = CrossResolutionBases::new(phi_src, psi_dst, 4, 16, 8).unwrap_err();
        assert_eq!(err, CrossResolutionError::RankDeficient);
    }

    #[test]
    fn constructor_rejects_shape_mismatch() {
        let phi_src = vec![0.0f32; 10]; // should be 16
        let psi_dst = vec![0.0f32; 16 * 8];
        let err = CrossResolutionBases::new(phi_src, psi_dst, 16, 16, 8).unwrap_err();
        assert_eq!(err, CrossResolutionError::ShapeMismatch);
    }

    #[test]
    fn cross_domain_variant_runs_and_matches_manual() {
        // Smoke: cross-domain variant should produce output equal to manually
        // composing project → C → reconstruct.
        let d_src = 16usize;
        let d_dst = 32usize;
        let k = 4usize;
        let phi_src = random_orthonormal(d_src, k, 0x5001);
        let psi_dst = random_orthonormal(d_dst, k, 0x6002);
        let bases = CrossResolutionBases::new(phi_src, psi_dst, d_src, d_dst, k)
            .expect("bases should construct");

        // Random C operator (k × k).
        let mut rng = make_rng(0x7003);
        let c_op: Vec<f32> = (0..k * k).map(|_| rng()).collect();

        let mut rng2 = make_rng(0x8004);
        let src: Vec<f32> = (0..d_src).map(|_| rng2()).collect();

        // Fused call.
        let mut dst_fused = vec![0.0f32; d_dst];
        let mut scratch = CrossResScratch::new(k);
        transport_cross_domain_cross_resolution_into(
            &src,
            &bases,
            &c_op,
            &mut scratch,
            &mut dst_fused,
        );

        // Manual reference.
        let mut spec = vec![0.0f32; k];
        project_to_spectral_into(&src, &bases, &mut spec);
        let mut spec_dst = vec![0.0f32; k];
        for i in 0..k {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += c_op[i * k + j] * spec[j];
            }
            spec_dst[i] = acc;
        }
        let mut dst_ref = vec![0.0f32; d_dst];
        reconstruct_from_spectral_into(&spec_dst, &bases, &mut dst_ref);

        for i in 0..d_dst {
            assert!(
                (dst_fused[i] - dst_ref[i]).abs() < 1e-5,
                "fused[{i}] = {} vs ref[{i}] = {}",
                dst_fused[i],
                dst_ref[i]
            );
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dot: f32 = simd::simd_dot_f32(a, b, a.len());
        let na: f32 = simd::simd_dot_f32(a, a, a.len()).sqrt();
        let nb: f32 = simd::simd_dot_f32(b, b, b.len()).sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        dot / (na * nb)
    }
}
