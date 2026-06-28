//! MTP (Multi-Token Prediction) projection loader and target-activation projector.
//!
//! Plan 016 / Plan 055 substrate. Pure data + binary loader — no forward logic.

use katgpt_core::simd::simd_dot_f32;

/// Project target activation into the drafter's embedding space.
///
/// Two strategies:
/// 1. Learned projection — matmul against `mtp_proj` weights
/// 2. Truncate/Pad — zero-cost fallback when no projection file exists
pub fn project_target_activation(
    out_buf: &mut [f32],         // [drafter_n_embd] output buffer
    target_hidden: &[f32],       // [target_n_embd] from target's forward pass
    mtp_proj: Option<&Vec<f32>>, // optional [drafter_n_embd, target_n_embd] weights
    target_n_embd: usize,
    drafter_n_embd: usize,
    activation_threshold: usize,
) {
    // Gate: skip if target is too small for activation conditioning
    if target_n_embd < activation_threshold {
        return;
    }

    match mtp_proj {
        // Strategy 1: Learned projection — full matmul
        Some(proj_weights) => {
            // proj_weights layout: [drafter_n_embd * target_n_embd]
            // out[i] = sum_j(proj_weights[i * target_n_embd + j] * target_hidden[j])
            let out_len = out_buf.len().min(drafter_n_embd);
            for (i, out_slot) in out_buf.iter_mut().enumerate().take(out_len) {
                let row_off = i * target_n_embd;
                *out_slot = simd_dot_f32(
                    &proj_weights[row_off..row_off + target_n_embd],
                    &target_hidden[..target_n_embd],
                    target_n_embd,
                );
            }
        }
        // Strategy 2: Truncate/Pad — zero-cost fallback
        None => {
            let copy_len = drafter_n_embd.min(target_n_embd);
            out_buf[..copy_len].copy_from_slice(&target_hidden[..copy_len]);
            // Zero-pad if drafter dimension is larger (rest should already be zeroed)
            if drafter_n_embd > target_n_embd {
                out_buf[target_n_embd..drafter_n_embd].fill(0.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MTP Projection Binary Loader (Plan 016)
// ---------------------------------------------------------------------------

/// Binary format constants for MTP projection weights.
const MTP_PROJ_MAGIC: u32 = 0x4D54505A; // "MTPZ"
const MTP_PROJ_VERSION: u32 = 1;

/// Loaded MTP projection weights from compact binary (MTPZ v1).
///
/// Maps `[target_hidden; token_embed]` (in_dim = 2 × target_n_embd) → draft_n_embd.
#[derive(Debug)]
pub struct MtpProjection {
    /// Input dimension (2 × target_n_embd for `[target_hidden; token_embed]`).
    pub in_dim: usize,
    /// Output dimension (draft_n_embd).
    pub out_dim: usize,
    /// Weight matrix `[out_dim * in_dim]`, row-major.
    pub weights: Vec<f32>,
    /// Bias vector `[out_dim]`.
    pub bias: Vec<f32>,
}

/// Load MTP projection weights from compact binary format (MTPZ v1).
///
/// # Binary Layout
///
/// ```text
/// [magic: u32]     0x4D54505A ("MTPZ")
/// [version: u32]   1
/// [in_dim: u32]    input dimension
/// [out_dim: u32]   output dimension
/// [weights: f32 × out_dim × in_dim]  row-major
/// [bias: f32 × out_dim]
/// [checksum: u32]  blake3 of everything above
/// ```
///
/// # Errors
///
/// Returns an error string on: invalid magic, unsupported version, size mismatch,
/// blake3 checksum failure, or NaN/Inf in loaded data.
pub fn load_mtp_projection(path: &std::path::Path) -> Result<MtpProjection, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let file_size = data.len();

    // Header: 4 × u32 = 16 bytes
    let header_size: usize = 16;
    if file_size < header_size + 4 {
        return Err(format!("File too small: {file_size} bytes"));
    }

    // Parse header (little-endian)
    let magic = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    );
    let version = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    );
    let in_dim = u32::from_le_bytes(
        data[8..12]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    ) as usize;
    let out_dim = u32::from_le_bytes(
        data[12..16]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    ) as usize;

    if magic != MTP_PROJ_MAGIC {
        return Err(format!(
            "Invalid magic: expected {MTP_PROJ_MAGIC:#010x}, got {magic:#010x}"
        ));
    }
    if version != MTP_PROJ_VERSION {
        return Err(format!(
            "Unsupported version: expected {MTP_PROJ_VERSION}, got {version}"
        ));
    }

    // Calculate expected sizes
    let weights_bytes = out_dim * in_dim * 4; // f32 = 4 bytes
    let bias_bytes = out_dim * 4;
    let expected_size = header_size + weights_bytes + bias_bytes + 4; // +4 checksum

    if file_size != expected_size {
        return Err(format!(
            "Size mismatch: expected {expected_size} bytes, got {file_size} bytes (in_dim={in_dim}, out_dim={out_dim})"
        ));
    }

    // Verify blake3 checksum
    let payload = &data[..file_size - 4];
    let stored_checksum = u32::from_le_bytes(
        data[file_size - 4..]
            .try_into()
            .map_err(|_| "checksum parse error".to_string())?,
    );
    let computed_hash = blake3::hash(payload);
    let computed_checksum = u32::from_le_bytes(
        computed_hash.as_bytes()[..4]
            .try_into()
            .map_err(|_| "hash parse error".to_string())?,
    );

    if computed_checksum != stored_checksum {
        return Err(format!(
            "BLAKE3 checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
        ));
    }

    // Extract weights and bias as f32 (little-endian).
    // bytemuck::pod_collect_to_vec performs a single SIMD-friendly bulk copy
    // (memcpy-like) instead of N iterator yields + N TryInto checks + N panicking
    // unwraps. Size already validated by file_size == expected_size above, so
    // the assert_eq! checks are demoted to debug_assert_eq! (no runtime cost
    // in release).
    let weights_offset = header_size;
    let bias_offset = weights_offset + weights_bytes;

    let weights: Vec<f32> =
        bytemuck::pod_collect_to_vec(&data[weights_offset..bias_offset]);
    let bias: Vec<f32> =
        bytemuck::pod_collect_to_vec(&data[bias_offset..file_size - 4]);

    debug_assert_eq!(weights.len(), out_dim * in_dim, "weights count mismatch");
    debug_assert_eq!(bias.len(), out_dim, "bias count mismatch");

    // Validate no NaN/Inf
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(format!("NaN/Inf in weights at index {i}"));
        }
    }
    for (i, &b) in bias.iter().enumerate() {
        if !b.is_finite() {
            return Err(format!("NaN/Inf in bias at index {i}"));
        }
    }

    Ok(MtpProjection {
        in_dim,
        out_dim,
        weights,
        bias,
    })
}

#[cfg(test)]
mod mtp_projection_binary_tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a valid MTPZ v1 binary at a temp path.
    fn create_test_binary(in_dim: usize, out_dim: usize) -> std::path::PathBuf {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&MTP_PROJ_MAGIC.to_le_bytes());
        buf.extend_from_slice(&MTP_PROJ_VERSION.to_le_bytes());
        buf.extend_from_slice(&(in_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(out_dim as u32).to_le_bytes());

        // Weights (zeros)
        for _ in 0..(out_dim * in_dim) {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Bias (zeros)
        for _ in 0..out_dim {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Checksum (blake3 of everything above)
        let hash = blake3::hash(&buf);
        let checksum = u32::from_le_bytes(hash.as_bytes()[..4].try_into().unwrap());
        buf.extend_from_slice(&checksum.to_le_bytes());

        let path = std::env::temp_dir().join("microgpt_test_mtp_projection.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        path
    }

    #[test]
    fn test_load_mtp_projection_valid_binary() {
        let path = create_test_binary(64, 16); // 2*32=64 in, 16 out
        let proj = load_mtp_projection(&path).unwrap();

        assert_eq!(proj.in_dim, 64);
        assert_eq!(proj.out_dim, 16);
        assert_eq!(proj.weights.len(), 64 * 16);
        assert_eq!(proj.bias.len(), 16);
        assert!(proj.weights.iter().all(|&w| w == 0.0));
        assert!(proj.bias.iter().all(|&b| b == 0.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_mtp_projection_invalid_magic() {
        let path = std::env::temp_dir().join("microgpt_test_mtp_bad_magic.bin");
        let mut buf = vec![0u8; 24]; // header(16) + min data(4) + checksum(4)
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        drop(f);

        let err = load_mtp_projection(&path).unwrap_err();
        assert!(
            err.contains("Invalid magic"),
            "expected 'Invalid magic' error, got: {err}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_mtp_projection_bad_checksum() {
        let path = std::env::temp_dir().join("microgpt_test_mtp_bad_checksum.bin");
        let mut buf = Vec::new();

        buf.extend_from_slice(&MTP_PROJ_MAGIC.to_le_bytes());
        buf.extend_from_slice(&MTP_PROJ_VERSION.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // in_dim
        buf.extend_from_slice(&2u32.to_le_bytes()); // out_dim

        // Weights + bias (all zeros)
        for _ in 0..(2 * 4 + 2) {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Wrong checksum
        buf.extend_from_slice(&0xCAFEBABEu32.to_le_bytes());

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        drop(f);

        let err = load_mtp_projection(&path).unwrap_err();
        assert!(
            err.contains("checksum mismatch"),
            "expected 'checksum mismatch' error, got: {err}"
        );

        let _ = std::fs::remove_file(&path);
    }
}
