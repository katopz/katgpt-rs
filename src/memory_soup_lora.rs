//! Memory Soup LoRA Artifact Importer (Plan 253 T19 G5).
//!
//! Standalone parser for the MSP0 binary format exported by riir-gpu's
//! `MemorySoupExportBundle`. Uses ONLY `std` + `blake3` — does NOT depend
//! on riir-gpu. This proves a third-party consumer (katgpt-rs) can read
//! the artifacts produced by riir-gpu training without a heavy dependency.
//!
//! ## Format (matches riir-gpu `MemorySoupExportBundle::from_state`)
//!
//! ```text
//! [magic: 4B  = b"MSP0"]
//! [version: 4B LE u32 = 1]
//! [blake3: 32B = hash of payload]
//! [payload: f32 LE values]
//!   [n_checkpoints: f32]
//!   [gate_dim: f32]
//!   [n_embd: f32]
//!   [lora_rank: f32]
//!   [n_targets: f32]
//!   [n_layers: f32]
//!   [gate_weight: gate_dim * n_embd f32s]
//!   for each checkpoint:
//!     [repr: gate_dim f32s]
//!     [delta: params_per_cp f32s]
//! ```
//!
//! `params_per_cp = n_layers * n_targets * 2 * lora_rank * n_embd`.
//!
//! ## Usage
//!
//! ```ignore
//! use katgpt_rs::memory_soup_lora::import_memory_soup_artifact;
//!
//! let bytes = std::fs::read("soup.msp0")?;
//! let artifact = import_memory_soup_artifact(&bytes)
//!     .ok_or("invalid or corrupted MSP0 file")?;
//!
//! // Use artifact.gate_weight for inference-time query projection.
//! // Use artifact.checkpoints for parameter-space interpolation.
//! ```
//!
//! Feature-gated behind `memory_soup_lora` — off by default.

use blake3;

/// Magic bytes for the MSP0 format (must match riir-gpu `SOUP_MAGIC`).
const MSP0_MAGIC: &[u8; 4] = b"MSP0";

/// Current format version (must match riir-gpu `SOUP_VERSION`).
const MSP0_VERSION: u32 = 1;

/// A single LoRA checkpoint entry (delta + segment representation).
#[derive(Debug, Clone)]
pub struct SoupCheckpoint {
    /// Segment representation vector `[gate_dim]`.
    pub repr: Vec<f32>,
    /// Flat LoRA delta parameters `[params_per_cp]`.
    /// Layout: `[layer_0_target_0_A, layer_0_target_0_B, ...]`.
    pub delta: Vec<f32>,
}

/// Parsed Memory Soup artifact (the inference-time view).
///
/// Contains everything needed for parameter-space interpolation at
/// inference: the trained gate projection and the checkpoint bank.
/// BLAKE3-verified on import.
#[derive(Debug, Clone)]
pub struct MemorySoupArtifact {
    /// Number of checkpoints in the bank.
    pub n_checkpoints: usize,
    /// Gating projection dimension (output of W_u · query).
    pub gate_dim: usize,
    /// Input embedding dimension.
    pub n_embd: usize,
    /// LoRA rank.
    pub lora_rank: usize,
    /// Number of LoRA targets per layer.
    pub n_targets: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Gate projection weights `[gate_dim * n_embd]`, row-major.
    /// `projected[j] = sum_a weight[j * n_embd + a] * query[a]`.
    pub gate_weight: Vec<f32>,
    /// Checkpoint bank: K entries, each with repr + delta.
    pub checkpoints: Vec<SoupCheckpoint>,
    /// BLAKE3 hash of the payload (verified on import).
    pub blake3_hash: [u8; 32],
}

/// Compute the number of f32 parameters per checkpoint.
#[inline]
fn params_per_checkpoint(n_layers: usize, n_targets: usize, lora_rank: usize, n_embd: usize) -> usize {
    n_layers * n_targets * 2 * lora_rank * n_embd
}

/// Import a Memory Soup artifact from MSP0 binary data.
///
/// Verifies magic, version, and BLAKE3 integrity. Returns `None` if the
/// data is malformed, the version is unsupported, or the integrity check
/// fails.
///
/// This function is the G5 GOAT gate for Plan 253 T19: it proves katgpt-rs
/// can consume riir-gpu's exported artifacts without depending on riir-gpu.
pub fn import_memory_soup_artifact(data: &[u8]) -> Option<MemorySoupArtifact> {
    // Minimum size: magic(4) + version(4) + hash(32) + header(6*4) = 64
    if data.len() < 64 {
        return None;
    }

    // ── Magic ──────────────────────────────────────────────────────
    if &data[0..4] != MSP0_MAGIC {
        return None;
    }

    // ── Version ────────────────────────────────────────────────────
    let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
    if version != MSP0_VERSION {
        return None;
    }

    // ── BLAKE3 integrity check ─────────────────────────────────────
    let stored_hash: [u8; 32] = data[8..40].try_into().ok()?;
    let computed = blake3::hash(&data[40..]);
    if computed.as_bytes() != &stored_hash {
        return None;
    }

    // ── Parse payload as f32 LE ────────────────────────────────────
    let payload_f32: Vec<f32> = data[40..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    if payload_f32.len() < 6 {
        return None;
    }

    let k = payload_f32[0] as usize;
    let gate_dim = payload_f32[1] as usize;
    let n_embd = payload_f32[2] as usize;
    let lora_rank = payload_f32[3] as usize;
    let n_targets = payload_f32[4] as usize;
    let n_layers = payload_f32[5] as usize;

    // Sanity: avoid pathological values that could cause huge allocations.
    if k == 0 || k > 1024 || gate_dim == 0 || gate_dim > 4096 || n_embd == 0 || n_embd > 4096 {
        return None;
    }

    let gate_params = gate_dim * n_embd;
    let params_per_cp = params_per_checkpoint(n_layers, n_targets, lora_rank, n_embd);

    // ── Expected size check ────────────────────────────────────────
    let expected = 6 + gate_params + k * (gate_dim + params_per_cp);
    if payload_f32.len() < expected {
        return None;
    }

    let mut offset = 6;

    // ── Gate weights ───────────────────────────────────────────────
    let gate_weight: Vec<f32> = payload_f32[offset..offset + gate_params].to_vec();
    offset += gate_params;

    // ── Checkpoints ────────────────────────────────────────────────
    let mut checkpoints: Vec<SoupCheckpoint> = Vec::with_capacity(k);
    for _ in 0..k {
        let repr: Vec<f32> = payload_f32[offset..offset + gate_dim].to_vec();
        offset += gate_dim;
        let delta: Vec<f32> = payload_f32[offset..offset + params_per_cp].to_vec();
        offset += params_per_cp;
        checkpoints.push(SoupCheckpoint { repr, delta });
    }

    Some(MemorySoupArtifact {
        n_checkpoints: k,
        gate_dim,
        n_embd,
        lora_rank,
        n_targets,
        n_layers,
        gate_weight,
        checkpoints,
        blake3_hash: stored_hash,
    })
}

/// Sigmoid gate (clamped to prevent overflow). Matches riir-gpu's `sigmoid_gate`.
#[inline]
pub fn sigmoid_gate(score: f32) -> f32 {
    let clamped = score.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Compute γ-weighted interpolation of checkpoint deltas for a query.
///
/// This is the inference-time operation: project the query, compute
/// relevance scores via sigmoid-gated dot product, then blend the
/// checkpoint deltas. Matches riir-gpu's `interpolate` (GRM mode).
///
/// Returns `(interpolated_delta, gate_weights)`.
pub fn interpolate_query(
    query: &[f32],
    artifact: &MemorySoupArtifact,
) -> (Vec<f32>, Vec<f32>) {
    let k = artifact.checkpoints.len();
    if k == 0 || artifact.n_embd != query.len() {
        return (Vec::new(), Vec::new());
    }

    let dim = artifact.gate_dim;
    let scale = 1.0 / (dim as f32).sqrt().max(1e-8);

    // Project query: projected[j] = sum_a gate_weight[j*n_embd + a] * query[a]
    // SIMD-accelerated matvec — one dot product per output lane.
    let mut projected = vec![0.0f32; dim];
    // Allow: hot SIMD matvec — explicit row indexing keeps the dot-product lane clear.
    #[allow(clippy::needless_range_loop)]
    for j in 0..dim {
        let row_off = j * artifact.n_embd;
        projected[j] = crate::simd::simd_dot_f32(
            &artifact.gate_weight[row_off..row_off + artifact.n_embd],
            query,
            artifact.n_embd,
        );
    }

    // Score each checkpoint and apply sigmoid gate.
    let gammas: Vec<f32> = artifact
        .checkpoints
        .iter()
        .map(|cp| {
            let dot = crate::simd::simd_dot_f32(&projected, &cp.repr, dim);
            sigmoid_gate(dot * scale)
        })
        .collect();

    // Interpolate deltas via fused SAXPY: interpolated += w * cp.delta.
    // simd_fused_decay_write(dst, decay=1.0, src, scale=w) computes
    // dst = 1.0*dst + w*src in one NEON/AVX2 pass per chunk.
    let delta_len = artifact.checkpoints[0].delta.len();
    let mut interpolated = vec![0.0f32; delta_len];
    for (i, cp) in artifact.checkpoints.iter().enumerate() {
        let w = gammas[i];
        crate::simd::simd_fused_decay_write(&mut interpolated, 1.0, &cp.delta, w);
    }

    (interpolated, gammas)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid MSP0 bundle in-memory (matching the riir-gpu format spec)
    /// and verify the importer can parse it. This is the G5 GOAT gate:
    /// katgpt-rs can consume riir-gpu's artifacts without a riir-gpu dep.
    fn build_sample_bundle(
        k: usize,
        gate_dim: usize,
        n_embd: usize,
        lora_rank: usize,
        n_targets: usize,
        n_layers: usize,
    ) -> Vec<u8> {
        let _gate_params = gate_dim * n_embd;
        let params_per_cp = params_per_checkpoint(n_layers, n_targets, lora_rank, n_embd);

        // Build payload.
        let mut payload: Vec<f32> = vec![
            k as f32,
            gate_dim as f32,
            n_embd as f32,
            lora_rank as f32,
        ];
        payload.push(n_targets as f32);
        payload.push(n_layers as f32);

        // Gate weights: identity-like (W_u[j, j] = 1.0 within K×K block).
        for j in 0..gate_dim {
            for a in 0..n_embd {
                payload.push(if j == a && j < k { 10.0 } else { 0.0 });
            }
        }

        // Checkpoints: each has repr = one-hot, delta = small random-ish.
        for i in 0..k {
            // repr
            for j in 0..gate_dim {
                payload.push(if j == i { 1.0 } else { 0.0 });
            }
            // delta
            for p in 0..params_per_cp {
                payload.push(((i * 7919 + p * 13) as f32 % 100.0) * 0.001);
            }
        }

        // Serialize payload to bytes.
        let payload_bytes: Vec<u8> = payload.iter().flat_map(|f| f.to_le_bytes()).collect();
        let hash = *blake3::hash(&payload_bytes).as_bytes();

        // Build full bundle.
        let mut data = Vec::with_capacity(4 + 4 + 32 + payload_bytes.len());
        data.extend_from_slice(MSP0_MAGIC);
        data.extend_from_slice(&MSP0_VERSION.to_le_bytes());
        data.extend_from_slice(&hash);
        data.extend_from_slice(&payload_bytes);
        data
    }

    #[test]
    fn g5_import_valid_bundle() {
        let bundle = build_sample_bundle(4, 8, 8, 4, 2, 1);
        let artifact = import_memory_soup_artifact(&bundle)
            .expect("valid bundle must parse");

        assert_eq!(artifact.n_checkpoints, 4);
        assert_eq!(artifact.gate_dim, 8);
        assert_eq!(artifact.n_embd, 8);
        assert_eq!(artifact.lora_rank, 4);
        assert_eq!(artifact.n_targets, 2);
        assert_eq!(artifact.n_layers, 1);
        assert_eq!(artifact.gate_weight.len(), 8 * 8);
        assert_eq!(artifact.checkpoints.len(), 4);

        // Verify gate weight: W_u[0,0] should be 10.0 (identity block).
        assert_eq!(artifact.gate_weight[0], 10.0);
        assert_eq!(artifact.gate_weight[9], 10.0); // [1,1]

        // Verify checkpoint reprs are one-hot.
        assert_eq!(artifact.checkpoints[0].repr[0], 1.0);
        assert_eq!(artifact.checkpoints[0].repr[1], 0.0);
        assert_eq!(artifact.checkpoints[1].repr[1], 1.0);

        // Verify delta lengths.
        let expected_delta_len = 2 * 2 * 4 * 8; // n_layers * n_targets * 2 * rank * n_embd
        assert_eq!(artifact.checkpoints[0].delta.len(), expected_delta_len);
    }

    #[test]
    fn g5_import_rejects_wrong_magic() {
        let mut bundle = build_sample_bundle(2, 4, 4, 2, 1, 1);
        // Corrupt magic.
        bundle[0] = b'X';
        assert!(import_memory_soup_artifact(&bundle).is_none());
    }

    #[test]
    fn g5_import_rejects_wrong_version() {
        let mut bundle = build_sample_bundle(2, 4, 4, 2, 1, 1);
        // Corrupt version.
        bundle[4..8].copy_from_slice(&999u32.to_le_bytes());
        assert!(import_memory_soup_artifact(&bundle).is_none());
    }

    #[test]
    fn g5_import_rejects_tampered_payload() {
        let mut bundle = build_sample_bundle(2, 4, 4, 2, 1, 1);
        // Flip a byte in the payload (after the 40-byte header).
        let last = bundle.len() - 1;
        bundle[last] ^= 0xFF;
        assert!(import_memory_soup_artifact(&bundle).is_none());
    }

    #[test]
    fn g5_import_rejects_truncated_data() {
        let bundle = build_sample_bundle(2, 4, 4, 2, 1, 1);
        // Truncate to 10 bytes (way too short).
        assert!(import_memory_soup_artifact(&bundle[..10]).is_none());
    }

    #[test]
    fn g5_interpolate_query_works() {
        let bundle = build_sample_bundle(4, 8, 8, 4, 2, 1);
        let artifact = import_memory_soup_artifact(&bundle)
            .expect("valid bundle");

        // Query with dim 0 hot → should activate checkpoint 0 most.
        let query: Vec<f32> = (0..8)
            .map(|j| if j == 0 { 1.0 } else { 0.0 })
            .collect();

        let (delta, gammas) = interpolate_query(&query, &artifact);

        assert!(!delta.is_empty());
        assert_eq!(gammas.len(), 4);

        // Checkpoint 0 should have the highest gamma.
        let best = (0..4)
            .max_by(|&a, &b| gammas[a].partial_cmp(&gammas[b]).unwrap())
            .unwrap();
        assert_eq!(best, 0, "domain-0 query should select checkpoint 0");
    }

    /// Cross-format consistency: build a bundle with the SAME logical content
    /// that riir-gpu would produce (same layout, same magic, same version,
    /// same hash algorithm) and verify katgpt-rs parses it identically.
    /// This is the core G5 guarantee.
    #[test]
    fn g5_cross_format_consistency() {
        // Use Plan 253 defaults: gate_dim=64, n_embd=64, rank=32, targets=6, layers=1.
        let bundle = build_sample_bundle(4, 64, 64, 32, 6, 1);
        let artifact = import_memory_soup_artifact(&bundle)
            .expect("Plan 253 default config must parse");

        // The params_per_cp for this config:
        // 1 * 6 * 2 * 32 * 64 = 24,576
        assert_eq!(
            artifact.checkpoints[0].delta.len(),
            24_576,
            "Plan 253 default params_per_cp"
        );

        // Gate weight matrix should be 64*64 = 4096.
        assert_eq!(artifact.gate_weight.len(), 4096);

        // Inference should produce finite output.
        let query = vec![1.0f32; 64];
        let (delta, _) = interpolate_query(&query, &artifact);
        assert!(delta.iter().all(|&v| v.is_finite()));
        assert_eq!(delta.len(), 24_576);
    }
}
