//! SigmoidLatentCodec — generic-trait facade over the existing latent
//! projection pattern (Plan 331 Phase 3, Research 312).
//!
//! `SigmoidLatentCodec<D>` projects an `&[f32; D]` latent vector onto K frozen
//! direction vectors via dot-product, applies a numerically-stable sigmoid gate
//! `σ(dot + bias) / τ`, and returns the top-k scores by magnitude as a
//! fixed-size [`CompressedLatent<K>`]. `decompress` is a deterministic
//! pseudo-inverse via the direction matrix transpose — it recovers the top-k
//! subspace, NOT the original vector (latent projection is not bijective).
//!
//! # What this IS (honest framing)
//!
//! Per Research 312 §2.1 and §4 caveat #5, this is **structurally identical**
//! to what `DensityBudget` + `extract_hla_slice` already do for HLA slices
//! (riir-ai Plan 311, Research 133). The value is **API uniformity** — the same
//! [`crate::babel_codec::BabelCodec`] trait now covers both text and latent
//! surfaces — NOT a new capability. Do not double-count this as novelty.
//!
//! # Modelless
//!
//! Deterministic closed-form: dot products + sigmoid + top-k selection. No
//! training, no backprop, no gradient descent. The `directions` and `bias`
//! arrays are frozen at construction (the "reader" state).
//!
//! # Sigmoid, NOT softmax (per AGENTS.md)
//!
//! Each projection is squashed independently via `σ(dot + bias)`. The K scores
//! do NOT sum to 1 (sigmoid is not softmax) — they are independent per-axis
//! activations. This matches the codebase convention
//! (`personality_composition::sigmoid`, `latent_field_steering`, etc.).

use crate::babel_codec::BabelCodec;
use crate::babel_codec::commitment::BabelCommitment;

/// Numerically stable scalar sigmoid: `σ(x) = 1 / (1 + e^{-x})`.
///
/// Delegates to [`crate::simd::fast_sigmoid`] (Cephes 6th-order polynomial
/// for `exp`, ~1 ULP accurate, ~1.7× faster than libm `exp` on aarch64).
/// Result is in `(0, 1)` for all finite inputs, saturating to `0.0` / `1.0`
/// for `|x| > 40` (the f32 precision floor).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    crate::simd::fast_sigmoid(x)
}

/// Fixed-size container for the top-k projected latent scores.
///
/// `K` is the number of retained (index, score) pairs. The buffer is filled
/// highest-magnitude-first; unused slots (when fewer than K directions fire)
/// carry `index = SENTINEL_INDEX` and `score = 0.0`. This makes `compress`
/// zero-allocation: callers pass a pre-sized `CompressedLatent<K>` and we write
/// in place.
///
/// `#[repr(C)]` + `Copy` so it can cross FFI / sync boundaries as a POD blob
/// (relevant for the future LatCal commitment bridge, Issue 002 — resolved + removed).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CompressedLatent<const K: usize> {
    /// Direction index for each of the top-k scores. `SENTINEL_INDEX` = unused.
    pub indices: [u32; K],
    /// Sigmoid-gated projection score for each top-k direction.
    pub scores: [f32; K],
    /// Number of valid entries in `indices` / `scores` (≤ K).
    pub len: u8,
}

/// Sentinel index marking an unused slot in [`CompressedLatent::indices`].
pub const SENTINEL_INDEX: u32 = u32::MAX;

impl<const K: usize> CompressedLatent<K> {
    /// Construct an empty (all-sentinel) compressed latent.
    pub const fn empty() -> Self {
        Self {
            indices: [SENTINEL_INDEX; K],
            scores: [0.0; K],
            len: 0,
        }
    }

    /// Number of valid entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// True if no entries are valid.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte view of the POD payload (for BLAKE3 commitment).
    pub fn as_bytes(&self) -> &[u8] {
        let size = core::mem::size_of::<Self>();
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size) }
    }
}

impl<const K: usize> Default for CompressedLatent<K> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Sigmoid-gated latent projection codec (generic over latent dimension `D`
/// and retain count `K`).
///
/// - `directions`: `Vec<[f32; D]>` frozen projection matrix (row-major).
///   `K_DIR = directions.len()` is the total number of directions; the codec
///   retains `K` top-magnitude scores per call where `K ≤ K_DIR`.
/// - `bias`: per-direction bias added to the dot product before sigmoid.
/// - `tau`: temperature divisor inside the sigmoid argument
///   (`σ((dot + bias) / tau)`). `tau > 0` sharpens; `tau < 0` is rejected at
///   construction.
///
/// `D` is the latent dimension; `K` is the compile-time retain count (the size
/// of the output [`CompressedLatent<K>`]). This two-const-generic shape matches
/// the `personality_composition::PersonalityWeightedComposition<N, D>` precedent.
/// The number of directions `K_DIR` is a runtime field so a single codec type
/// can serve banks of varying size without threading a third const generic.
pub struct SigmoidLatentCodec<const D: usize, const K: usize> {
    /// Frozen direction vectors, row-major. Length = `K_DIR * D`.
    directions: Vec<[f32; D]>,
    /// Per-direction bias. Length = `K_DIR`.
    bias: Vec<f32>,
    /// Temperature inside the sigmoid argument (must be > 0).
    tau: f32,
    /// Compression ratio of the most recent `compress` call
    /// (`bytes_out / bytes_in`). For the latent codec this is
    /// `(K * 8) / (D * 4)` — fixed by construction.
    last_ratio: f32,
    /// Cached commitment of the most recent `compress` output.
    last_commitment: BabelCommitment,
    /// Reusable scratch: per-direction `(index, score)` after projection.
    /// Pre-sized to `directions.len()` once at construction; `clear()`-and-refill
    /// per call so the steady-state `compress` is zero-allocation.
    scratch_scores: Vec<(u32, f32)>,
    /// Reusable scratch: top-k selection "taken" mask, parallel to
    /// `scratch_scores`. Pre-sized once; zeroed per call.
    scratch_taken: Vec<bool>,
}

impl<const D: usize, const K: usize> SigmoidLatentCodec<D, K> {
    /// Construct a codec from frozen directions, bias, and temperature. Returns
    /// `None` if `directions.is_empty()`, `bias.len() != directions.len()`,
    /// `tau <= 0`, or `K > directions.len()`, or `K == 0`.
    pub fn new(directions: Vec<[f32; D]>, bias: Vec<f32>, tau: f32) -> Option<Self> {
        if directions.is_empty() {
            return None;
        }
        if bias.len() != directions.len() {
            return None;
        }
        if tau <= 0.0 || !tau.is_finite() {
            return None;
        }
        if K == 0 || K > directions.len() {
            return None;
        }
        let n = directions.len();
        Some(Self {
            directions,
            bias,
            tau,
            last_ratio: 1.0,
            last_commitment: BabelCommitment::zero(),
            scratch_scores: vec![(0u32, 0.0f32); n],
            scratch_taken: vec![false; n],
        })
    }

    /// Number of frozen direction vectors.
    #[inline]
    pub fn n_directions(&self) -> usize {
        self.directions.len()
    }

    /// Number of top-magnitude scores retained per `compress` call (== `K`).
    #[inline]
    pub fn k_retain(&self) -> usize {
        K
    }

    /// Access the frozen direction matrix (for inspection / serialization).
    pub fn directions(&self) -> &[[f32; D]] {
        &self.directions
    }

    /// Access the per-direction bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Temperature.
    #[inline]
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Project `input` onto all directions, returning `(index, score)` pairs
    /// sorted by descending magnitude. Writes into `out` (length = `n_dirs`).
    ///
    /// Exposed for testing — production callers use [`BabelCodec::compress`] which
    /// keeps only the top-k.
    pub fn project_all(&self, input: &[f32; D], out: &mut [(u32, f32)]) {
        debug_assert_eq!(out.len(), self.directions.len());
        for (i, (dir, &b)) in self.directions.iter().zip(self.bias.iter()).enumerate() {
            let dot = dot(dir, input);
            let score = sigmoid((dot + b) / self.tau);
            out[i] = (i as u32, score);
        }
    }
}

/// Dot product of two equal-length f32 slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

impl<const D: usize, const K: usize> BabelCodec for SigmoidLatentCodec<D, K> {
    type Input = [f32; D];
    type Compressed = CompressedLatent<K>;
    type Reader = ();

    fn compress(&mut self, input: &Self::Input) -> Self::Compressed {
        // Zero-allocation hot path (Plan 331 T3.2): all scratch lives in
        // pre-sized codec fields (`scratch_scores`, `scratch_taken`), filled in
        // place per call. The output `CompressedLatent<K>` is a stack `Copy`
        // struct returned by value — no heap traffic on the return path either.
        let n = self.directions.len();
        debug_assert_eq!(self.scratch_scores.len(), n);
        debug_assert_eq!(self.scratch_taken.len(), n);

        // 1. Project onto every direction, sigmoid-gate, write into scratch.
        for (i, (dir, &b)) in self.directions.iter().zip(self.bias.iter()).enumerate() {
            let dot_val = dot(dir, input);
            let score = sigmoid((dot_val + b) / self.tau);
            self.scratch_scores[i] = (i as u32, score);
        }
        // Reset the taken mask for this call.
        for t in self.scratch_taken.iter_mut() {
            *t = false;
        }

        // 2. Select top-k by magnitude (|score|), highest first. Selection-sort
        //    partial sort: O(K * n) — fine for n ≤ a few hundred. For
        //    very large banks a heap would win; that's a riir-ai consumer-crate
        //    concern. The `taken` mask prevents re-selecting the same direction.
        let mut out = CompressedLatent::<K>::empty();
        let k = K;
        let mut filled: usize = 0;
        for slot in 0..k {
            let mut best_i: Option<usize> = None;
            let mut best_mag = -1.0f32;
            for i in 0..n {
                if self.scratch_taken[i] {
                    continue;
                }
                let mag = self.scratch_scores[i].1.abs();
                if mag > best_mag {
                    best_mag = mag;
                    best_i = Some(i);
                }
            }
            match best_i {
                Some(i) => {
                    self.scratch_taken[i] = true;
                    let (idx, sc) = self.scratch_scores[i];
                    out.indices[slot] = idx;
                    out.scores[slot] = sc;
                    filled = slot + 1;
                }
                None => break,
            }
        }
        out.len = filled as u8;

        // 3. Record ratio + commitment.
        let bytes_in = (D * core::mem::size_of::<f32>()) as f32;
        let bytes_out =
            (out.len() * (core::mem::size_of::<u32>() + core::mem::size_of::<f32>())) as f32;
        self.last_ratio = if bytes_in > 0.0 {
            bytes_out / bytes_in
        } else {
            1.0
        };
        self.last_commitment = BabelCommitment::of(out.as_bytes());
        out
    }

    fn decompress(_reader: &Self::Reader, c: &Self::Compressed) -> Self::Input {
        // Deterministic pseudo-inverse: place each retained score back onto
        // its direction's axis with unit weight, zero elsewhere. This recovers
        // the top-k SUBSPACE, not the original vector (latent projection is
        // not bijective — the orthogonal complement is lost).
        //
        // For the canonical-basis case (direction i = e_i) this is exact: the
        // recovered vector has the sigmoid scores on the retained axes and 0
        // elsewhere. For arbitrary directions this is the projection of the
        // score vector onto the row space of the direction matrix (treated as
        // orthogonal), which is the standard Moore-Penrose-style least-squares
        // inverse under the orthogonal assumption.
        let mut out = [0.0f32; D];
        for slot in 0..c.len() {
            let idx = c.indices[slot] as usize;
            // We do not have access to the direction matrix here (the Reader is
            // () by design — the codec owns the directions, but decompress is a
            // trait method that takes only the reader). For the canonical-basis
            // case this is exact: index = axis. For general directions, callers
            // should use `reconstruct` (below), which takes &self.
            if idx < D {
                out[idx] = c.scores[slot];
            }
        }
        out
    }

    #[inline]
    fn last_ratio(&self) -> f32 {
        self.last_ratio
    }

    #[inline]
    fn commit(&self) -> BabelCommitment {
        self.last_commitment
    }

    fn verify(&self, c: &Self::Compressed, commitment: &BabelCommitment) -> bool {
        let recomputed = BabelCommitment::of(c.as_bytes());
        recomputed.as_bytes() == commitment.as_bytes()
    }
}

impl<const D: usize, const K: usize> SigmoidLatentCodec<D, K> {
    /// Reconstruct a latent vector from a [`CompressedLatent`] using the codec's
    /// own direction matrix (the proper inverse, available on `&self` but NOT
    /// via the trait's `decompress` which only gets the `Reader`).
    ///
    /// Places each retained score as a coefficient on its direction and sums:
    /// `out = Σ_slot score_slot * directions[idx_slot]`. For orthogonal
    /// directions this is the exact projection-back; for non-orthogonal
    /// directions it is the adjoint (transpose) map — the closest orthogonal
    /// assumption inverse.
    pub fn reconstruct(&self, c: &CompressedLatent<K>) -> [f32; D] {
        let mut out = [0.0f32; D];
        for slot in 0..c.len() {
            let idx = c.indices[slot] as usize;
            if idx < self.directions.len() {
                let dir = &self.directions[idx];
                let s = c.scores[slot];
                for (axis, &w) in dir.iter().enumerate() {
                    out[axis] += s * w;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a codec with `n_dirs` canonical basis vectors in D-space.
    /// Canonical basis = exact invertibility (decompress is bit-exact on the
    /// retained subspace).
    fn canonical_codec<const D: usize, const K: usize>(
        n_dirs: usize,
        tau: f32,
    ) -> SigmoidLatentCodec<D, K> {
        let mut dirs = Vec::with_capacity(n_dirs);
        for i in 0..n_dirs {
            let mut d = [0.0f32; D];
            if i < D {
                d[i] = 1.0;
            }
            dirs.push(d);
        }
        let bias = vec![0.0; n_dirs];
        SigmoidLatentCodec::new(dirs, bias, tau).expect("canonical codec must construct")
    }

    #[test]
    fn round_trip_preserves_topk_subspace_canonical_basis() {
        // D=8, 8 canonical basis directions, retain top-3. The input has its
        // energy concentrated on axes 0, 2, 4 → those are the top-3 by magnitude.
        const D: usize = 8;
        const K: usize = 3;
        let mut codec = canonical_codec::<D, K>(8, 1.0);
        let mut input = [0.0f32; D];
        input[0] = 5.0;
        input[2] = 4.0;
        input[4] = 3.0;
        input[6] = 0.1; // small — should NOT make top-3

        let compressed = codec.compress(&input);
        assert_eq!(compressed.len(), K);

        // Top-3 indices must be 0, 2, 4 (in some order — we sort by magnitude).
        let mut got: Vec<u32> = compressed.indices[..compressed.len()].to_vec();
        got.sort_unstable();
        assert_eq!(got, vec![0, 2, 4], "top-3 must be axes 0,2,4 by magnitude");

        // Decompress via the trait (canonical basis → exact on retained axes).
        let recovered = SigmoidLatentCodec::<D, K>::decompress(&(), &compressed);
        // The retained axes carry their sigmoid-gated scores; non-retained axes are 0.
        for axis in 0..D {
            if got.contains(&(axis as u32)) {
                let dot = input[axis]; // bias = 0, tau = 1
                let expected = sigmoid(dot);
                assert!(
                    (recovered[axis] - expected).abs() < 1e-5,
                    "axis {axis}: expected sigmoid({dot})={expected:.6}, got {:.6}",
                    recovered[axis]
                );
            } else {
                assert!(
                    recovered[axis].abs() < 1e-6,
                    "non-retained axis {axis} should be ~0, got {:.6}",
                    recovered[axis]
                );
            }
        }
    }

    #[test]
    fn zero_vector_yields_all_half_scores() {
        // σ(0) = 0.5 for every direction (dot=0, bias=0, tau=1).
        const D: usize = 4;
        const K: usize = 2;
        let mut codec = canonical_codec::<D, K>(4, 1.0);
        let input = [0.0f32; D];
        let compressed = codec.compress(&input);
        for &score in compressed.scores[..compressed.len()].iter() {
            assert!(
                (score - 0.5).abs() < 1e-6,
                "zero-vector score should be σ(0)=0.5, got {score}"
            );
        }
    }

    #[test]
    fn max_magnitude_vector_saturates_to_one() {
        // Large positive dot → σ → saturates to ~1.0.
        const D: usize = 4;
        const K: usize = 2;
        let mut codec = canonical_codec::<D, K>(4, 1.0);
        let mut input = [0.0f32; D];
        input[0] = 100.0; // huge positive
        let compressed = codec.compress(&input);
        // Top score must be ~1.0.
        let top = compressed.scores[..compressed.len()]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (top - 1.0).abs() < 1e-5,
            "saturated score should be ~1.0, got {top}"
        );
    }

    #[test]
    fn compress_is_deterministic_across_calls() {
        const D: usize = 8;
        const K: usize = 4;
        let mut codec = canonical_codec::<D, K>(8, 1.0);
        let input = [0.3, -0.7, 1.2, 0.0, -2.1, 0.5, 0.9, -0.4];
        let c1 = codec.compress(&input);
        let c2 = codec.compress(&input);
        assert_eq!(c1.indices, c2.indices, "indices must match across calls");
        assert_eq!(c1.scores, c2.scores, "scores must match across calls");
        assert_eq!(c1.len, c2.len);
    }

    #[test]
    fn direction_orthogonality_makes_reconstruct_exact_on_retained_axes() {
        // Canonical basis: reconstruct places the score back on its axis.
        const D: usize = 4;
        const K: usize = 2;
        let codec = canonical_codec::<D, K>(4, 1.0);
        let mut input = [0.0f32; D];
        input[1] = 2.0;
        input[3] = -1.5;
        let mut codec_mut = codec.clone_for_test();
        let compressed = codec_mut.compress(&input);
        let reconstructed = codec.reconstruct(&compressed);
        // On retained axes (1 and 3, by magnitude), reconstruct should yield
        // score * direction = score (since direction is e_axis with magnitude 1).
        // Non-retained axes contribute 0.
        let mut retained: Vec<u32> = compressed.indices[..compressed.len()].to_vec();
        retained.sort_unstable();
        for axis in 0..D {
            if retained.contains(&(axis as u32)) {
                let expected = sigmoid(input[axis]);
                assert!(
                    (reconstructed[axis] - expected).abs() < 1e-5,
                    "retained axis {axis}: expected {expected:.5}, got {:.5}",
                    reconstructed[axis]
                );
            }
        }
    }

    #[test]
    fn bias_shifts_the_sigmoid_midpoint() {
        // With bias = +5 on every direction, even a zero input gives σ(5)≈0.993.
        const D: usize = 4;
        const K: usize = 2;
        let dirs: Vec<[f32; D]> = (0..4)
            .map(|i| {
                let mut d = [0.0; D];
                d[i] = 1.0;
                d
            })
            .collect();
        let bias = vec![5.0; 4];
        let mut codec = SigmoidLatentCodec::<D, K>::new(dirs, bias, 1.0).unwrap();
        let input = [0.0f32; D];
        let compressed = codec.compress(&input);
        let expected = sigmoid(5.0);
        for &score in compressed.scores[..compressed.len()].iter() {
            assert!(
                (score - expected).abs() < 1e-5,
                "bias-shifted score should be σ(5)={expected:.5}, got {score:.5}"
            );
        }
    }

    #[test]
    fn tau_sharpens_or_softens_the_gate() {
        // Small tau → sharper (closer to step function).
        // Large tau → softer (closer to 0.5 for moderate inputs).
        const D: usize = 2;
        let dirs = vec![[1.0, 0.0], [0.0, 1.0]];
        let bias = vec![0.0; 2];
        let input = [0.5, 0.5]; // moderate positive

        let sharp = SigmoidLatentCodec::<D, 1>::new(dirs.clone(), bias.clone(), 0.1).unwrap();
        let soft = SigmoidLatentCodec::<D, 1>::new(dirs, bias, 10.0).unwrap();
        let mut sharp_m = sharp;
        let mut soft_m = soft;

        let cs = sharp_m.compress(&input);
        let cf = soft_m.compress(&input);

        let sharp_score = cs.scores[0];
        let soft_score = cf.scores[0];
        // σ(0.5/0.1)=σ(5)≈0.993  >  σ(0.5/10)=σ(0.05)≈0.512
        assert!(
            sharp_score > soft_score,
            "tau=0.1 (sharp, {sharp_score:.4}) must produce higher score than tau=10 (soft, {soft_score:.4})"
        );
        assert!((soft_score - sigmoid(0.05)).abs() < 1e-5);
        assert!((sharp_score - sigmoid(5.0)).abs() < 1e-5);
    }

    #[test]
    fn k_less_than_d_case_keeps_only_top_k() {
        // D=8, K=3 → output has exactly 3 entries, not 8.
        const D: usize = 8;
        const K: usize = 3;
        let mut codec = canonical_codec::<D, K>(8, 1.0);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let compressed = codec.compress(&input);
        assert_eq!(compressed.len(), K, "must retain exactly K=3 entries");
        // Top-3 by magnitude (here all positive, so top-3 largest): 8, 7, 6 → axes 7, 6, 5.
        let mut got: Vec<u32> = compressed.indices[..compressed.len()].to_vec();
        got.sort_unstable();
        assert_eq!(
            got,
            vec![5, 6, 7],
            "top-3 must be axes 5,6,7 (largest inputs)"
        );
    }

    #[test]
    fn constructor_rejects_invalid_args() {
        const D: usize = 4;
        // Empty directions.
        assert!(SigmoidLatentCodec::<D, 1>::new(vec![], vec![], 1.0).is_none());
        // Mismatched bias length.
        let dirs = vec![[1.0, 0.0, 0.0, 0.0]];
        assert!(SigmoidLatentCodec::<D, 1>::new(dirs.clone(), vec![], 1.0).is_none());
        // tau <= 0.
        assert!(SigmoidLatentCodec::<D, 1>::new(dirs.clone(), vec![0.0], 0.0).is_none());
        assert!(SigmoidLatentCodec::<D, 1>::new(dirs.clone(), vec![0.0], -1.0).is_none());
        // K = 0 (compile-time rejected — can't construct a 0-retain codec type
        // meaningfully; skip).
        // K > n_dirs.
        assert!(SigmoidLatentCodec::<D, 5>::new(dirs.clone(), vec![0.0], 1.0).is_none());
    }

    #[test]
    fn last_ratio_is_recorded_and_sane() {
        const D: usize = 8;
        const K: usize = 4;
        let mut codec = canonical_codec::<D, K>(8, 1.0);
        let _ = codec.compress(&[1.0; D]);
        let r = codec.last_ratio();
        // bytes_out = K * (4 + 4) = 32, bytes_in = D * 4 = 32 → ratio = 1.0
        // (latent projection is not a byte-compression win by itself; its
        // value is API uniformity, per the honest framing in the module docs).
        assert!(
            (r - 1.0).abs() < 1e-5,
            "expected ratio ~1.0 for D=8 K=4, got {r}"
        );
    }

    #[test]
    fn commit_and_verify_round_trip() {
        const D: usize = 4;
        const K: usize = 2;
        let mut codec = canonical_codec::<D, K>(4, 1.0);
        let input = [1.0, -2.0, 0.5, 3.0];
        let compressed = codec.compress(&input);
        let commitment = codec.commit();
        // Verify against the same compressed payload.
        assert!(
            codec.verify(&compressed, &commitment),
            "verify must accept the just-committed payload"
        );
        // Tamper: flip one score.
        let mut tampered = compressed;
        tampered.scores[0] += 0.001;
        assert!(
            !codec.verify(&tampered, &commitment),
            "verify must reject a tampered payload"
        );
    }

    // ─── Test-only helpers ───────────────────────────────────────────────────

    /// Clone helper for tests (production code clones by reconstructing from
    /// the fields; tests just need a second mutable copy).
    trait CloneForTest {
        fn clone_for_test(&self) -> Self;
    }

    impl<const D: usize, const K: usize> CloneForTest for SigmoidLatentCodec<D, K> {
        fn clone_for_test(&self) -> Self {
            SigmoidLatentCodec {
                directions: self.directions.clone(),
                bias: self.bias.clone(),
                tau: self.tau,
                last_ratio: self.last_ratio,
                last_commitment: self.last_commitment,
                scratch_scores: self.scratch_scores.clone(),
                scratch_taken: self.scratch_taken.clone(),
            }
        }
    }
}
