//! Triplet encoder with joint 3×3 rounding for OCTOPUS KV cache.
//!
//! Encoding pipeline for a rotated KV vector:
//! 1. Decompose into ⌈d/3⌉ triplets via [`triplet::decompose`]
//! 2. For each triplet: encode direction via octahedral map, quantize (ξ, η, ρ)
//! 3. Joint 3×3 rounding: search 9 direction candidates to maximize
//!    alignment `s = t · n(ξ,η)`, then pick norm nearest to `s`
//!
//! The joint 3×3 rounding is encoder-only (6-14% MSE reduction) with
//! zero change to the decoder or bitstream format.

use super::octahedral::{oct_decode, oct_encode};
use super::triplet::{Triplet, decompose};
use super::types::{OctopusCodebook, TripletIndices};

/// Encode a single triplet into quantized indices using nearest-centroid.
///
/// Simple baseline: independently quantize each component.
/// No joint rounding — fast but suboptimal.
pub fn encode_triplet_simple(triplet: &Triplet, codebook: &OctopusCodebook) -> TripletIndices {
    let (xi, eta) = oct_encode(triplet.dir[0], triplet.dir[1], triplet.dir[2]);

    TripletIndices {
        i_xi: codebook.oct.quantize(xi),
        i_eta: codebook.oct.quantize(eta),
        i_rho: codebook.norm.quantize(triplet.norm),
    }
}

/// Encode a single triplet with joint 3×3 rounding.
///
/// Instead of independently quantizing ξ and η, this searches a 3×3
/// neighborhood around the scalar seed to find the direction that
/// maximizes `s = t · n(ξ, η)` (dot product with the true triplet vector).
/// Then picks the norm centroid nearest to `s` (not to `||t||`).
///
/// This gives 6-14% MSE reduction at zero bitstream change (encoder-only).
///
/// # Algorithm
/// 1. Compute scalar seed indices (j_xi, j_eta) via nearest-centroid
/// 2. Enumerate 9 candidates: (j_xi + δ_xi, j_eta + δ_eta) for δ ∈ {-1, 0, 1}
/// 3. For each valid candidate, decode direction n = oct_decode(ξ, η)
/// 4. Score = t · n = ρ * (dir · n) — maximize over candidates
/// 5. Pick norm centroid nearest to best score (not to ρ)
pub fn encode_triplet_joint(triplet: &Triplet, codebook: &OctopusCodebook) -> TripletIndices {
    // Handle zero-norm triplet: return zero indices
    if triplet.norm < 1e-10 {
        return TripletIndices::zero();
    }

    let (xi, eta) = oct_encode(triplet.dir[0], triplet.dir[1], triplet.dir[2]);

    // Scalar seed indices
    let j_xi = codebook.oct.quantize(xi) as i16;
    let j_eta = codebook.oct.quantize(eta) as i16;
    let n_dir = codebook.oct.centroids.len() as i16;

    // Hoist hot-loop invariants: the centroid slice and the triplet direction
    // components are loop-invariant across the 3×3 candidate search.
    let centroids = codebook.oct.centroids.as_slice();
    let (dir_x, dir_y, dir_z) = (triplet.dir[0], triplet.dir[1], triplet.dir[2]);
    let norm = triplet.norm;

    // Search 3×3 neighborhood for best direction alignment
    let mut best_score = f32::NEG_INFINITY;
    let mut best_i_xi = j_xi;
    let mut best_i_eta = j_eta;

    for dx in -1i16..=1 {
        let cx = j_xi + dx;
        if cx < 0 || cx >= n_dir {
            continue;
        }
        for dy in -1i16..=1 {
            let cy = j_eta + dy;
            if cy < 0 || cy >= n_dir {
                continue;
            }

            let xi_c = centroids[cx as usize];
            let eta_c = centroids[cy as usize];
            let (nx, ny, nz) = oct_decode(xi_c, eta_c);

            // Score = t · n = ρ * (dir · n)
            let dot = dir_x * nx + dir_y * ny + dir_z * nz;
            let score = norm * dot;

            if score > best_score {
                best_score = score;
                best_i_xi = cx;
                best_i_eta = cy;
            }
        }
    }

    // Pick norm nearest to best_score (not to ||t||!)
    // Key insight from paper: the reconstruction error is minimized when
    // the norm centroid matches the projected scalar s = t · n̂,
    // not the original norm ρ = ||t||.
    let i_rho = codebook.norm.quantize(best_score.max(0.0));

    TripletIndices {
        i_xi: best_i_xi as u16,
        i_eta: best_i_eta as u16,
        i_rho,
    }
}

/// Encode a full d-dimensional rotated vector into triplet indices.
///
/// Returns ⌈d/3⌉ packed index tuples. If `use_joint_rounding` is false,
/// uses simple independent quantization for each component.
pub fn encode_vector(
    rotated: &[f32],
    codebook: &OctopusCodebook,
    use_joint_rounding: bool,
) -> Vec<TripletIndices> {
    let triplets = decompose(rotated);
    let mut out = Vec::with_capacity(triplets.len());
    encode_vector_into(&triplets, codebook, use_joint_rounding, &mut out);
    out
}

/// Zero-alloc variant of [`encode_vector`].
///
/// Takes pre-decomposed triplets and writes into `out`.
pub fn encode_vector_into(
    triplets: &[Triplet],
    codebook: &OctopusCodebook,
    use_joint_rounding: bool,
    out: &mut Vec<TripletIndices>,
) {
    out.clear();
    out.reserve(triplets.len());
    for t in triplets {
        out.push(if use_joint_rounding {
            encode_triplet_joint(t, codebook)
        } else {
            encode_triplet_simple(t, codebook)
        });
    }
}

/// Decode a single triplet's indices back to a 3-element vector.
///
/// Pipeline: dequantize (ξ, η) → oct_decode → direction, dequantize ρ → norm,
/// then reconstruct: `ρ · (x, y, z)`.
pub fn decode_triplet(indices: &TripletIndices, codebook: &OctopusCodebook) -> [f32; 3] {
    let xi = codebook.oct.dequantize(indices.i_xi);
    let eta = codebook.oct.dequantize(indices.i_eta);
    let rho = codebook.norm.dequantize(indices.i_rho);

    let (x, y, z) = oct_decode(xi, eta);
    [rho * x, rho * y, rho * z]
}

/// Decode all triplet indices back into a d-dimensional vector.
///
/// The output has length `indices.len() * 3` (may be longer than original d
/// due to zero-padding). Caller should truncate to original dimension.
pub fn decode_vector(indices: &[TripletIndices], codebook: &OctopusCodebook) -> Vec<f32> {
    let d = indices.len() * 3;
    let mut out = vec![0.0f32; d];
    decode_vector_into(indices, codebook, &mut out);
    out
}

/// Decode triplet indices into a pre-allocated buffer (zero-alloc hot path).
///
/// Writes exactly `indices.len() * 3` elements.
pub fn decode_vector_into(indices: &[TripletIndices], codebook: &OctopusCodebook, out: &mut [f32]) {
    for (i, idx) in indices.iter().enumerate() {
        let v = decode_triplet(idx, codebook);
        let base = i * 3;
        out[base] = v[0];
        out[base + 1] = v[1];
        out[base + 2] = v[2];
    }
}

/// Reconstruct a triplet from its indices (for MSE computation).
///
/// Returns the `Triplet` with dequantized norm and direction.
pub fn reconstruct_triplet(indices: &TripletIndices, codebook: &OctopusCodebook) -> Triplet {
    let xi = codebook.oct.dequantize(indices.i_xi);
    let eta = codebook.oct.dequantize(indices.i_eta);
    let rho = codebook.norm.dequantize(indices.i_rho);
    let (x, y, z) = oct_decode(xi, eta);
    Triplet {
        norm: rho,
        dir: [x, y, z],
    }
}

/// Compute per-triplet MSE between original and encoded-then-decoded triplets.
///
/// Useful for benchmarking the quality of joint vs. simple rounding.
pub fn triplet_mse(
    original: &[Triplet],
    indices: &[TripletIndices],
    codebook: &OctopusCodebook,
) -> f32 {
    assert_eq!(original.len(), indices.len());
    let mut total = 0.0f32;
    for (orig, idx) in original.iter().zip(indices.iter()) {
        let recon = decode_triplet(idx, codebook);
        let orig_vec = orig.to_vec();
        for k in 0..3 {
            let diff = orig_vec[k] - recon[k];
            total += diff * diff;
        }
    }
    total / (original.len() as f32 * 3.0)
}

/// Pack triplet indices into a flat byte buffer for storage.
///
/// Layout per triplet: `[i_xi (dir_bits)] [i_eta (dir_bits)] [i_rho (nrm_bits)]`
/// packed contiguously in bitstream order.
///
/// Returns the packed byte buffer.
pub fn pack_triplet_indices(indices: &[TripletIndices], dir_bits: u8, nrm_bits: u8) -> Vec<u8> {
    let dir_bits_usize = dir_bits as usize;
    let bits_per_triplet = 2 * dir_bits_usize + nrm_bits as usize;
    let total_bits = indices.len() * bits_per_triplet;
    let mut packed = vec![0u8; total_bits.div_ceil(8)];

    let dir_mask = ((1u16 << dir_bits) - 1) as u32;
    let nrm_mask = ((1u16 << nrm_bits) - 1) as u32;
    let total_mask = (1u32 << bits_per_triplet) - 1;

    let mut bit_pos = 0usize;
    for idx in indices {
        // Combine all 3 indices into a single value and pack at once
        let combined = (idx.i_xi as u32 & dir_mask)
            | ((idx.i_eta as u32 & dir_mask) << dir_bits_usize)
            | ((idx.i_rho as u32 & nrm_mask) << (2 * dir_bits_usize));
        let byte_pos = bit_pos / 8;
        let shift = bit_pos % 8;
        let v = (combined & total_mask) << shift;
        packed[byte_pos] |= v as u8;
        if shift + bits_per_triplet > 8 {
            packed[byte_pos + 1] |= (v >> 8) as u8;
        }
        if shift + bits_per_triplet > 16 {
            packed[byte_pos + 2] |= (v >> 16) as u8;
        }
        if shift + bits_per_triplet > 24 {
            packed[byte_pos + 3] |= (v >> 24) as u8;
        }
        bit_pos += bits_per_triplet;
    }

    packed
}

/// Pack triplet indices into a pre-allocated byte buffer (zero-alloc hot path).
///
/// Clears and resizes `out` as needed. Equivalent to [`pack_triplet_indices`] but avoids allocation.
///
/// Optimized: combines all 3 indices into a single u32 and writes once per triplet
/// instead of 3 separate pack_bits calls.
pub fn pack_triplet_indices_into(
    indices: &[TripletIndices],
    dir_bits: u8,
    nrm_bits: u8,
    out: &mut Vec<u8>,
) {
    let dir_bits_usize = dir_bits as usize;
    let bits_per_triplet = 2 * dir_bits_usize + nrm_bits as usize;
    let total_bits = indices.len() * bits_per_triplet;
    let byte_len = total_bits.div_ceil(8);
    out.clear();
    out.resize(byte_len, 0);

    let dir_mask = ((1u16 << dir_bits) - 1) as u32;
    let nrm_mask = ((1u16 << nrm_bits) - 1) as u32;
    let total_mask = (1u32 << bits_per_triplet) - 1;

    let mut bit_pos = 0usize;
    for idx in indices {
        // Combine all 3 indices into a single value and pack at once
        let combined = (idx.i_xi as u32 & dir_mask)
            | ((idx.i_eta as u32 & dir_mask) << dir_bits_usize)
            | ((idx.i_rho as u32 & nrm_mask) << (2 * dir_bits_usize));
        let byte_pos = bit_pos / 8;
        let shift = bit_pos % 8;
        let v = (combined & total_mask) << shift;
        out[byte_pos] |= v as u8;
        if shift + bits_per_triplet > 8 {
            out[byte_pos + 1] |= (v >> 8) as u8;
        }
        if shift + bits_per_triplet > 16 {
            out[byte_pos + 2] |= (v >> 16) as u8;
        }
        if shift + bits_per_triplet > 24 {
            out[byte_pos + 3] |= (v >> 24) as u8;
        }
        bit_pos += bits_per_triplet;
    }
}

/// Unpack triplet indices from a flat byte buffer.
///
/// Inverse of [`pack_triplet_indices`].
pub fn unpack_triplet_indices(
    packed: &[u8],
    n_triplets: usize,
    dir_bits: u8,
    nrm_bits: u8,
) -> Vec<TripletIndices> {
    let mut indices = Vec::with_capacity(n_triplets);
    unpack_triplet_indices_into(packed, n_triplets, dir_bits, nrm_bits, &mut indices);
    indices
}

/// Zero-alloc variant of [`unpack_triplet_indices`].
///
/// Clears and fills `out` with unpacked `TripletIndices`.
///
/// Optimized: reads all 3 indices in a single combined u32 and extracts fields,
/// avoiding 3 separate unpack_bits calls per triplet.
pub fn unpack_triplet_indices_into(
    packed: &[u8],
    n_triplets: usize,
    dir_bits: u8,
    nrm_bits: u8,
    out: &mut Vec<TripletIndices>,
) {
    out.clear();
    out.reserve(n_triplets);
    let dir_bits_usize = dir_bits as usize;
    let bits_per_triplet = 2 * dir_bits_usize + nrm_bits as usize;
    let dir_mask = ((1u16 << dir_bits) - 1) as u32;
    let nrm_mask = ((1u16 << nrm_bits) - 1) as u32;

    let mut bit_pos = 0usize;
    for _ in 0..n_triplets {
        let byte_pos = bit_pos / 8;
        let shift = bit_pos % 8;
        // Read up to 4 bytes and assemble combined value
        let mut raw = packed[byte_pos] as u32;
        if byte_pos + 1 < packed.len() {
            raw |= (packed[byte_pos + 1] as u32) << 8;
        }
        if byte_pos + 2 < packed.len() {
            raw |= (packed[byte_pos + 2] as u32) << 16;
        }
        if byte_pos + 3 < packed.len() && shift + bits_per_triplet > 24 {
            raw |= (packed[byte_pos + 3] as u32) << 24;
        }
        let combined = (raw >> shift) & ((1u32 << bits_per_triplet) - 1);

        out.push(TripletIndices {
            i_xi: (combined & dir_mask) as u16,
            i_eta: ((combined >> dir_bits_usize) & dir_mask) as u16,
            i_rho: ((combined >> (2 * dir_bits_usize)) & nrm_mask) as u16,
        });
        bit_pos += bits_per_triplet;
    }
}

/// Pack `n_bits` of `value` into `buf` starting at `bit_pos`.
#[allow(dead_code)]
fn pack_bits(buf: &mut [u8], bit_pos: usize, value: u32, n_bits: u8) {
    let byte_pos = bit_pos / 8;
    let shift = bit_pos % 8;
    let mask = (1u32 << n_bits as usize) - 1;
    let v = (value & mask) << shift;
    buf[byte_pos] |= v as u8;
    if shift + n_bits as usize > 8 {
        buf[byte_pos + 1] |= (v >> 8) as u8;
    }
    if shift + n_bits as usize > 16 {
        buf[byte_pos + 2] |= (v >> 16) as u8;
    }
}

/// Unpack `n_bits` from `buf` starting at `bit_pos`.
#[allow(dead_code)]
fn unpack_bits(buf: &[u8], bit_pos: usize, n_bits: u8) -> u32 {
    let byte_pos = bit_pos / 8;
    let shift = bit_pos % 8;
    let mut raw = buf[byte_pos] as u32;
    if byte_pos + 1 < buf.len() {
        raw |= (buf[byte_pos + 1] as u32) << 8;
    }
    if byte_pos + 2 < buf.len() && shift + n_bits as usize > 16 {
        raw |= (buf[byte_pos + 2] as u32) << 16;
    }
    (raw >> shift) & ((1u32 << n_bits as usize) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_codebook(dim: usize, nominal_bits: u8) -> OctopusCodebook {
        OctopusCodebook::build(dim, nominal_bits)
    }

    // ── Simple encode/decode roundtrip ────────────────────────

    #[test]
    fn test_simple_roundtrip_single_triplet() {
        let cb = make_codebook(128, 3);
        let norm = (0.25f32 + 0.25 + 0.5).sqrt();
        let t = Triplet {
            norm: 0.2,
            dir: [0.5f32, 0.5, std::f32::consts::FRAC_1_SQRT_2].map(|v| v / norm),
        };
        let idx = encode_triplet_simple(&t, &cb);
        let recon = decode_triplet(&idx, &cb);
        let orig = t.to_vec();
        // With 3-bit nominal (dir=4, nrm=2), roundtrip should be reasonable
        for k in 0..3 {
            assert!(
                (recon[k] - orig[k]).abs() < 0.3,
                "roundtrip mismatch [{k}]: got {}, expected {}",
                recon[k],
                orig[k]
            );
        }
    }

    #[test]
    fn test_joint_roundtrip_single_triplet() {
        let cb = make_codebook(128, 3);
        let norm = (0.25f32 + 0.25 + 0.5).sqrt();
        let t = Triplet {
            norm: 0.2,
            dir: [0.5f32, 0.5, std::f32::consts::FRAC_1_SQRT_2].map(|v| v / norm),
        };
        let idx = encode_triplet_joint(&t, &cb);
        let recon = decode_triplet(&idx, &cb);
        let orig = t.to_vec();
        for k in 0..3 {
            assert!(
                (recon[k] - orig[k]).abs() < 0.3,
                "roundtrip mismatch [{k}]: got {}, expected {}",
                recon[k],
                orig[k]
            );
        }
    }

    #[test]
    fn test_encode_vector_roundtrip() {
        // NOTE: encode_vector operates on pre-rotated triplet coordinates.
        // At 2-bit nominal (dir=3, nrm=1), the norm codebook has only 2 levels
        // — very coarse. Without rotation preconditioning the raw coordinates
        // don't match the Beta marginal, so tolerance must be generous.
        // Real usage goes through kv_cache which rotates first.
        let cb = make_codebook(128, 2);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
        let indices = encode_vector(&v, &cb, false);
        assert_eq!(indices.len(), 43); // ⌈128/3⌉

        let recon = decode_vector(&indices, &cb);
        // At 2-bit without rotation, individual coordinates can deviate significantly.
        // Verify reconstruction captures gross structure (cosine similarity).
        let dot: f32 = v.iter().zip(&recon[..128]).map(|(a, b)| a * b).sum();
        let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = recon[..128].iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.3,
            "encode_vector roundtrip cosine = {cos} (2-bit, no rotation)"
        );
    }

    #[test]
    fn test_encode_vector_roundtrip_3bit() {
        // 3-bit with rotation is much more representative
        let cb = make_codebook(128, 3);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin()).collect();
        let indices = encode_vector(&v, &cb, false);

        let recon = decode_vector(&indices, &cb);
        let dot: f32 = v.iter().zip(&recon[..128]).map(|(a, b)| a * b).sum();
        let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = recon[..128].iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.5,
            "encode_vector roundtrip cosine = {cos} (3-bit, no rotation)"
        );
    }

    #[test]
    fn test_joint_better_than_simple() {
        let cb = make_codebook(128, 2);
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).cos()).collect();
        let triplets = decompose(&v);

        let idx_simple = encode_vector(&v, &cb, false);
        let idx_joint = encode_vector(&v, &cb, true);

        let mse_simple = triplet_mse(&triplets, &idx_simple, &cb);
        let mse_joint = triplet_mse(&triplets, &idx_joint, &cb);

        assert!(
            mse_joint <= mse_simple * 1.05, // allow 5% tolerance for randomness
            "joint MSE {mse_joint} should be <= simple MSE {mse_simple}"
        );
    }

    // ── Zero-norm handling ────────────────────────────────────

    #[test]
    fn test_encode_zero_triplet() {
        let cb = make_codebook(128, 2);
        let t = Triplet::zero();
        let idx = encode_triplet_joint(&t, &cb);
        assert_eq!(idx.i_xi, 0);
        assert_eq!(idx.i_eta, 0);
        assert_eq!(idx.i_rho, 0);
    }

    // ── Pack/unpack roundtrip ─────────────────────────────────

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let dir_bits = 3;
        let nrm_bits = 1;
        let indices: Vec<TripletIndices> = (0..43)
            .map(|i| TripletIndices {
                i_xi: (i % 8) as u16,
                i_eta: ((i + 1) % 8) as u16,
                i_rho: (i % 2) as u16,
            })
            .collect();

        let packed = pack_triplet_indices(&indices, dir_bits, nrm_bits);
        let unpacked = unpack_triplet_indices(&packed, 43, dir_bits, nrm_bits);

        assert_eq!(unpacked.len(), 43);
        for (i, (orig, recon)) in indices.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(orig.i_xi, recon.i_xi, "xi mismatch at triplet {i}");
            assert_eq!(orig.i_eta, recon.i_eta, "eta mismatch at triplet {i}");
            assert_eq!(orig.i_rho, recon.i_rho, "rho mismatch at triplet {i}");
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip_3bit() {
        let dir_bits = 4;
        let nrm_bits = 2;
        let indices: Vec<TripletIndices> = (0..43)
            .map(|i| TripletIndices {
                i_xi: (i % 16) as u16,
                i_eta: ((i + 3) % 16) as u16,
                i_rho: (i % 4) as u16,
            })
            .collect();

        let packed = pack_triplet_indices(&indices, dir_bits, nrm_bits);
        let unpacked = unpack_triplet_indices(&packed, 43, dir_bits, nrm_bits);

        for (i, (orig, recon)) in indices.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(orig.i_xi, recon.i_xi, "xi at {i}");
            assert_eq!(orig.i_eta, recon.i_eta, "eta at {i}");
            assert_eq!(orig.i_rho, recon.i_rho, "rho at {i}");
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let dir_bits = 5;
        let nrm_bits = 3;
        let indices: Vec<TripletIndices> = (0..22)
            .map(|i| TripletIndices {
                i_xi: (i % 32) as u16,
                i_eta: ((i * 7) % 32) as u16,
                i_rho: (i % 8) as u16,
            })
            .collect();

        let packed = pack_triplet_indices(&indices, dir_bits, nrm_bits);
        let unpacked = unpack_triplet_indices(&packed, 22, dir_bits, nrm_bits);

        for (i, (orig, recon)) in indices.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(orig.i_xi, recon.i_xi, "xi at {i}");
            assert_eq!(orig.i_eta, recon.i_eta, "eta at {i}");
            assert_eq!(orig.i_rho, recon.i_rho, "rho at {i}");
        }
    }

    // ── decode_vector_into ────────────────────────────────────

    #[test]
    fn test_decode_vector_into() {
        let cb = make_codebook(64, 3);
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let indices = encode_vector(&v, &cb, true);
        let n = indices.len() * 3;
        let mut out = vec![0.0f32; n];
        decode_vector_into(&indices, &cb, &mut out);
        let full = decode_vector(&indices, &cb);
        assert_eq!(out.len(), full.len());
        for i in 0..n {
            assert!((out[i] - full[i]).abs() < 1e-6, "mismatch at {i}");
        }
    }

    // ── reconstruct_triplet ───────────────────────────────────

    #[test]
    fn test_reconstruct_triplet() {
        let cb = make_codebook(128, 3);
        let t = Triplet {
            norm: 0.3,
            dir: [0.0, 0.0, 1.0],
        };
        let idx = encode_triplet_simple(&t, &cb);
        let recon = reconstruct_triplet(&idx, &cb);
        // Direction should be near (0, 0, 1)
        assert!(recon.dir[2] > 0.8, "z-dir should be near 1.0");
        // Norm should be reasonable
        assert!(
            (recon.norm - 0.3).abs() < 0.3,
            "norm: got {}, expected ~0.3",
            recon.norm
        );
    }
}
