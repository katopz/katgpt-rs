//! CPU-side LoRA adapter.

use super::*;
use super::domain::{read_f32_le, read_u16_le, read_u32_le};

// ---------------------------------------------------------------------------
// LoRA Adapter — CPU inference path (Plan 025)
// ---------------------------------------------------------------------------

/// CPU-side LoRA adapter for inference.
/// Loads from the same binary format as `GpuLoraAdapter` (Plan 008):
/// `[LORA(4) | version(4) | blake3(32) | payload...]`
/// where payload = `[n_adapters(4) | rank(4) | alpha(4) | adapter_data...]`
/// and adapter_data = `[in_dim(4) | out_dim(4) | a_f32s | b_f32s]`
///
/// Use [`LoraAdapter::load`] to read ALL adapters from a multi-adapter file
/// (correct for L2+ models), or [`LoraAdapter::load_first`] when only the
/// first adapter is needed (e.g., single-forward-pass heuristic players).
///
/// Zero-copy: loaded once per domain, reference-passed during inference.
///
/// Fields ordered by descending alignment to minimize padding:
/// usize/Vec (8-byte) → f32 (4-byte).
pub struct LoraAdapter {
    /// LoRA rank.
    pub rank: usize,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Down-projection: [rank × in_dim]
    pub a: Vec<f32>,
    /// Up-projection: [out_dim × rank]
    pub b: Vec<f32>,
    /// Scaling factor (alpha / rank).
    pub alpha: f32,
}

impl LoraAdapter {
    /// Load ALL adapters from a Plan 008 binary LoRA file.
    ///
    /// Multi-adapter files (e.g., L2+ with 6 adapters/layer × n_layer) return every
    /// adapter in declaration order. Single-adapter files return a 1-element Vec.
    ///
    /// Issue 299: previously this returned only the first adapter, silently
    /// discarding layers 1+ and invalidating L2+ arena benchmarks.
    pub fn load(path: &std::path::Path) -> Result<Vec<Self>, String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u32 = 1;

        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read lora file: {e}"))?;

        if file_data.len() < 44 {
            return Err("File too small for lora header".into());
        }

        if &file_data[0..4] != LORA_MAGIC {
            return Err("Invalid lora magic bytes".into());
        }

        let version = u32::from_le_bytes(
            file_data[4..8]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
        );
        if version != LORA_VERSION {
            return Err(format!("Unsupported lora version: {version}"));
        }

        let stored_checksum = &file_data[8..40];
        let payload = &file_data[40..];

        let computed = blake3::hash(payload);
        if computed.as_bytes() != stored_checksum {
            return Err("LoRA file checksum mismatch".into());
        }

        let mut offset = 0usize;
        let n_adapters = read_u32_le(payload, &mut offset)? as usize;
        let rank = read_u32_le(payload, &mut offset)? as usize;
        let alpha = read_f32_le(payload, &mut offset)?;

        if n_adapters == 0 {
            return Err("No adapters in lora file".into());
        }

        let mut adapters = Vec::with_capacity(n_adapters);
        for i in 0..n_adapters {
            let in_dim = read_u32_le(payload, &mut offset)? as usize;
            let out_dim = read_u32_le(payload, &mut offset)? as usize;

            let a_count = rank * in_dim;
            let b_count = out_dim * rank;
            let a_bytes = a_count * std::mem::size_of::<f32>();
            let b_bytes = b_count * std::mem::size_of::<f32>();

            if offset + a_bytes + b_bytes > payload.len() {
                return Err(format!("Truncated adapter {i} data"));
            }

            let a: Vec<f32> = {
                #[cfg(target_endian = "little")]
                {
                    let mut v = Vec::with_capacity(a_count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload[offset..].as_ptr(),
                            v.as_mut_ptr() as *mut u8,
                            a_bytes,
                        );
                        v.set_len(a_count);
                    }
                    v
                }
                #[cfg(not(target_endian = "little"))]
                {
                    payload[offset..offset + a_bytes]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                        .collect()
                }
            };
            offset += a_bytes;

            let b: Vec<f32> = {
                #[cfg(target_endian = "little")]
                {
                    let mut v = Vec::with_capacity(b_count);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            payload[offset..].as_ptr(),
                            v.as_mut_ptr() as *mut u8,
                            b_bytes,
                        );
                        v.set_len(b_count);
                    }
                    v
                }
                #[cfg(not(target_endian = "little"))]
                {
                    payload[offset..offset + b_bytes]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                        .collect()
                }
            };
            offset += b_bytes;

            adapters.push(Self {
                rank,
                in_dim,
                out_dim,
                alpha,
                a,
                b,
            });
        }

        if offset != payload.len() {
            return Err(format!(
                "LoRA payload trailing data: consumed {offset}, payload {}",
                payload.len()
            ));
        }

        Ok(adapters)
    }

    /// Load only the first adapter from a Plan 008 binary LoRA file.
    ///
    /// Convenience for consumers that store a single `LoraAdapter` and only run
    /// one forward pass (e.g., `LoraPlayer`, `FullHLPlayer`). Multi-adapter
    /// files (L2+) have layers 1+ silently dropped — this is explicit about
    /// that limitation so callers cannot accidentally regress on Issue 299.
    ///
    /// For correct multi-adapter evaluation, use [`load`](Self::load) and apply
    /// each adapter to its target projection during the forward pass.
    pub fn load_first(path: &std::path::Path) -> Result<Self, String> {
        let adapters = Self::load(path)?;
        adapters
            .into_iter()
            .next()
            .ok_or_else(|| "LoRA file declared zero-length adapter list".into())
    }

    /// Save adapters to a Plan 008 binary LoRA file (the inverse of [`load`](Self::load)).
    ///
    /// All adapters MUST share the same `rank` and `alpha` — the file format stores
    /// them once in the header. Per-adapter `in_dim`/`out_dim` are stored individually
    /// (they can differ across targets, e.g. Q vs K in GQA).
    ///
    /// Format: `["LORA" | version=1(u32) | blake3(payload)(32) | payload]` where
    /// payload = `[n_adapters(u32) | rank(u32) | alpha(f32) | per-adapter:
    /// in_dim(u32) | out_dim(u32) | a_f32s | b_f32s]`.
    ///
    /// This is the CPU-side counterpart to the private GPU LoRA exporter,
    /// producing byte-identical files that load via either path. Used by
    /// `CpuLoraTrainer` (Issue 018 CPU fallback) to produce arena-loadable
    /// adapters without a GPU.
    pub fn save(
        adapters: &[&Self],
        rank: usize,
        alpha: f32,
        path: &std::path::Path,
    ) -> Result<(), String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u32 = 1;

        if adapters.is_empty() {
            return Err("Cannot save zero adapters".into());
        }

        let mut payload = Vec::with_capacity(12 + adapters.len() * (8 + 16));

        // Header
        payload.extend_from_slice(&(adapters.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(rank as u32).to_le_bytes());
        payload.extend_from_slice(&alpha.to_le_bytes());

        // Adapter data
        for adapter in adapters {
            if adapter.rank != rank {
                return Err(format!(
                    "Adapter rank mismatch: header={rank}, adapter={}",
                    adapter.rank
                ));
            }
            payload.extend_from_slice(&(adapter.in_dim as u32).to_le_bytes());
            payload.extend_from_slice(&(adapter.out_dim as u32).to_le_bytes());

            let a_count = rank * adapter.in_dim;
            let b_count = adapter.out_dim * rank;
            if adapter.a.len() != a_count {
                return Err(format!(
                    "A matrix size mismatch: expected {a_count}, got {}",
                    adapter.a.len()
                ));
            }
            if adapter.b.len() != b_count {
                return Err(format!(
                    "B matrix size mismatch: expected {b_count}, got {}",
                    adapter.b.len()
                ));
            }
            // Bulk write matrix data on LE targets (avoids per-element
            // extend_from_slice overhead). f32 is plain-old-data with no
            // padding; on LE targets to_ne_bytes == to_le_bytes.
            // SAFETY: f32 has no padding and is plain-old-data; we reinterpret
            // the contiguous &[f32] as bytes for a single bulk copy.
            #[cfg(target_endian = "little")]
            {
                let a_bytes = unsafe {
                    std::slice::from_raw_parts(
                        adapter.a.as_ptr() as *const u8,
                        adapter.a.len() * std::mem::size_of::<f32>(),
                    )
                };
                payload.extend_from_slice(a_bytes);
                let b_bytes = unsafe {
                    std::slice::from_raw_parts(
                        adapter.b.as_ptr() as *const u8,
                        adapter.b.len() * std::mem::size_of::<f32>(),
                    )
                };
                payload.extend_from_slice(b_bytes);
            }
            #[cfg(not(target_endian = "little"))]
            {
                for val in &adapter.a {
                    payload.extend_from_slice(&val.to_le_bytes());
                }
                for val in &adapter.b {
                    payload.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        // Compute blake3 checksum of payload
        let checksum = blake3::hash(&payload);

        // Assemble file: magic + version + checksum + payload
        let mut file_data = Vec::with_capacity(4 + 4 + 32 + payload.len());
        file_data.extend_from_slice(LORA_MAGIC);
        file_data.extend_from_slice(&LORA_VERSION.to_le_bytes());
        file_data.extend_from_slice(checksum.as_bytes());
        file_data.extend_from_slice(&payload);

        std::fs::write(path, &file_data)
            .map_err(|e| format!("Failed to write lora file: {e}"))
    }

    /// Load LoRA adapters from a compact binary format.
    ///
    /// Format:
    /// ```text
    /// [MAGIC: "LORA" 4B]
    /// [VERSION: 1B]
    /// [RANK: 2B LE]
    /// [N_LAYERS: 2B LE]
    /// [N_TARGETS: 2B LE]
    /// [TARGET_IDS: N_TARGETS × 2B LE]  (0=q_proj, 1=k_proj, 2=v_proj, 3=o_proj,
    ///                                    4=gate_proj, 5=up_proj, 6=down_proj)
    /// [LAYER_DATA: for each (layer, target):
    ///   [A_ROWS: 2B][A_COLS: 2B][A_DATA: A_ROWS×A_COLS × 4B f32]
    ///   [B_ROWS: 2B][B_COLS: 2B][B_DATA: B_ROWS×B_COLS × 4B f32]
    /// ]
    /// [BLAKE3_HASH: 32B]  — covers everything before it
    /// ```
    ///
    /// Alpha defaults to `rank * 2`.
    pub fn load_from_bin(path: &std::path::Path) -> Result<Vec<Self>, String> {
        const LORA_MAGIC: &[u8; 4] = b"LORA";
        const LORA_VERSION: u8 = 1;

        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read lora bin file: {e}"))?;

        // Minimum: magic(4) + version(1) + rank(2) + n_layers(2) + n_targets(2) + hash(32) = 43
        if file_data.len() < 43 {
            return Err("File too small for lora bin header".into());
        }

        // Validate blake3 checksum — last 32 bytes cover everything before them
        let data_len = file_data.len() - 32;
        let stored_checksum = &file_data[data_len..];
        let computed = blake3::hash(&file_data[..data_len]);
        if computed.as_bytes() != stored_checksum {
            return Err("LoRA bin file checksum mismatch".into());
        }

        let mut offset = 0usize;

        // Magic
        if &file_data[offset..offset + 4] != LORA_MAGIC {
            return Err("Invalid lora bin magic bytes".into());
        }
        offset += 4;

        // Version
        let version = file_data[offset];
        if version != LORA_VERSION {
            return Err(format!("Unsupported lora bin version: {version}"));
        }
        offset += 1;

        // Rank
        let rank = read_u16_le(&file_data, &mut offset)? as usize;

        // N_LAYERS
        let n_layers = read_u16_le(&file_data, &mut offset)? as usize;

        // N_TARGETS
        let n_targets = read_u16_le(&file_data, &mut offset)? as usize;

        if n_layers == 0 || n_targets == 0 {
            return Err("No layers or targets in lora bin file".into());
        }

        // TARGET_IDS
        let mut target_ids = Vec::with_capacity(n_targets);
        for _ in 0..n_targets {
            let tid = read_u16_le(&file_data, &mut offset)?;
            match tid {
                0..=6 => target_ids.push(tid),
                _ => return Err(format!("Invalid target ID: {tid}")),
            }
        }

        // LAYER_DATA
        let alpha = (rank * 2) as f32;
        let mut adapters = Vec::with_capacity(n_layers * n_targets);

        for _layer in 0..n_layers {
            for &_target_id in &target_ids {
                // A matrix: [rank × in_dim]
                let a_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let a_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let a_count = a_rows * a_cols;
                let a_bytes = a_count * std::mem::size_of::<f32>();

                if offset + a_bytes > data_len {
                    return Err("Truncated A matrix data".into());
                }

                let a: Vec<f32> = {
                    #[cfg(target_endian = "little")]
                    {
                        let mut v = Vec::with_capacity(a_count);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                file_data[offset..].as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                a_bytes,
                            );
                            v.set_len(a_count);
                        }
                        v
                    }
                    #[cfg(not(target_endian = "little"))]
                    {
                        file_data[offset..offset + a_bytes]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                            .collect()
                    }
                };
                offset += a_bytes;

                // B matrix: [out_dim × rank]
                let b_rows = read_u16_le(&file_data, &mut offset)? as usize;
                let b_cols = read_u16_le(&file_data, &mut offset)? as usize;
                let b_count = b_rows * b_cols;
                let b_bytes = b_count * std::mem::size_of::<f32>();

                if offset + b_bytes > data_len {
                    return Err("Truncated B matrix data".into());
                }

                let b: Vec<f32> = {
                    #[cfg(target_endian = "little")]
                    {
                        let mut v = Vec::with_capacity(b_count);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                file_data[offset..].as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                b_bytes,
                            );
                            v.set_len(b_count);
                        }
                        v
                    }
                    #[cfg(not(target_endian = "little"))]
                    {
                        file_data[offset..offset + b_bytes]
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is 4 bytes")))
                            .collect()
                    }
                };
                offset += b_bytes;

                let in_dim = a_cols;
                let out_dim = b_rows;

                adapters.push(Self {
                    rank,
                    in_dim,
                    out_dim,
                    alpha,
                    a,
                    b,
                });
            }
        }

        if offset != data_len {
            return Err(format!(
                "Unexpected trailing data: read {offset}, expected {data_len}"
            ));
        }

        if adapters.is_empty() {
            return Err("No adapters loaded from lora bin file".into());
        }

        Ok(adapters)
    }
}

/// Apply LoRA delta in-place: `output += (alpha/rank) × B @ (A @ input)`
///
/// `lora_buf` is a pre-allocated `[rank]` intermediate buffer — zero alloc in hot path.
/// The B×hidden multiplication and scaling are fused directly into the output accumulation,
/// avoiding a separate delta buffer.
#[inline(always)]
pub fn lora_apply(output: &mut [f32], lora: &LoraAdapter, input: &[f32], lora_buf: &mut [f32]) {
    let scale = lora.alpha / lora.rank as f32;

    // 1. hidden = A @ input  (rank × in_dim) @ [in_dim] → [rank]
    matmul(lora_buf, &lora.a, input, lora.rank, lora.in_dim);

    // 2. output += scale × (B @ hidden) — SIMD-accelerated per-row dot product
    for r in 0..lora.out_dim {
        let row_off = r * lora.rank;
        let sum =
            crate::simd::simd_dot_f32(&lora.b[row_off..row_off + lora.rank], lora_buf, lora.rank);
        unsafe {
            *output.get_unchecked_mut(r) += scale * sum;
        }
    }
}

/// A loaded LoRA pair for modality-specific inference (Plan 025).
/// Reader is active during bidirectional prefill, writer during causal decode.
/// Switching is a reference swap — zero data movement.
pub struct LoraPair {
    /// LoRA active during bidirectional prefill (e.g., Python Reader).
    pub reader: Option<LoraAdapter>,
    /// LoRA active during causal decode (e.g., Rust Writer).
    pub writer: Option<LoraAdapter>,
}

impl LoraPair {
    /// Empty pair — no LoRA applied.
    pub fn none() -> Self {
        Self {
            reader: None,
            writer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_adapters() -> Vec<LoraAdapter> {
        // 6 adapters mirroring the Bomber/Generic layout: Q, K, V, O, Mlp1, Mlp2.
        // rank=4, in_dim=32 (n_embd), out_dim varies (GQA: K/V are n_kv_head*head_dim).
        vec![
            LoraAdapter {
                rank: 4,
                in_dim: 32,
                out_dim: 32,
                a: (0..4 * 32).map(|i| i as f32 * 0.01).collect(),
                b: (0..32 * 4).map(|i| i as f32 * -0.01).collect(),
                alpha: 8.0,
            },
            LoraAdapter {
                rank: 4,
                in_dim: 32,
                out_dim: 8, // kv_dim (n_kv_head=1, head_dim=8)
                a: (0..4 * 32).map(|i| i as f32).collect(),
                b: (0..8 * 4).map(|i| i as f32 * 2.0).collect(),
                alpha: 8.0,
            },
            LoraAdapter {
                rank: 4,
                in_dim: 32,
                out_dim: 8,
                a: (0..4 * 32).map(|i| i as f32 * 0.5).collect(),
                b: (0..8 * 4).map(|i| i as f32 * -2.0).collect(),
                alpha: 8.0,
            },
            LoraAdapter {
                rank: 4,
                in_dim: 32,
                out_dim: 32,
                a: (0..4 * 32).map(|i| i as f32 * 0.1).collect(),
                b: (0..32 * 4).map(|i| i as f32 * 0.3).collect(),
                alpha: 8.0,
            },
            LoraAdapter {
                rank: 4,
                in_dim: 32,
                out_dim: 32, // FFN up (mlp1)
                a: (0..4 * 32).map(|i| i as f32 * 1.5).collect(),
                b: (0..32 * 4).map(|i| i as f32 * -1.5).collect(),
                alpha: 8.0,
            },
            LoraAdapter {
                rank: 4,
                in_dim: 32,
                out_dim: 32, // FFN down (mlp2)
                a: (0..4 * 32).map(|i| i as f32 * 2.5).collect(),
                b: (0..32 * 4).map(|i| i as f32 * -2.5).collect(),
                alpha: 8.0,
            },
        ]
    }

    #[test]
    fn save_load_roundtrip_preserves_all_adapters() {
        let tmp = std::env::temp_dir().join("katgpt_lora_roundtrip_test.bin");
        let original = make_test_adapters();
        let refs: Vec<&LoraAdapter> = original.iter().collect();

        LoraAdapter::save(&refs, 4, 8.0, &tmp).expect("save should succeed");
        let loaded = LoraAdapter::load(&tmp).expect("load should succeed");

        assert_eq!(loaded.len(), original.len(), "adapter count must match");
        for (i, (orig, load)) in original.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(load.rank, orig.rank, "adapter {i} rank");
            assert_eq!(load.in_dim, orig.in_dim, "adapter {i} in_dim");
            assert_eq!(load.out_dim, orig.out_dim, "adapter {i} out_dim");
            assert_eq!(load.alpha, orig.alpha, "adapter {i} alpha");
            assert_eq!(load.a.len(), orig.a.len(), "adapter {i} A length");
            assert_eq!(load.b.len(), orig.b.len(), "adapter {i} B length");
            for (j, (a, b)) in orig.a.iter().zip(load.a.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "adapter {i} A[{j}] bit-identical");
            }
            for (j, (a, b)) in orig.b.iter().zip(load.b.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "adapter {i} B[{j}] bit-identical");
            }
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn save_rejects_empty_adapter_list() {
        let tmp = std::env::temp_dir().join("katgpt_lora_empty_test.bin");
        let empty: Vec<&LoraAdapter> = vec![];
        let result = LoraAdapter::save(&empty, 4, 8.0, &tmp);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn save_rejects_rank_mismatch() {
        let tmp = std::env::temp_dir().join("katgpt_lora_rankmismatch_test.bin");
        let adapters = make_test_adapters();
        let refs: Vec<&LoraAdapter> = adapters.iter().collect();
        // Pass rank=8 but adapters have rank=4
        let result = LoraAdapter::save(&refs, 8, 8.0, &tmp);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&tmp);
    }
}
