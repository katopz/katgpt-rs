//! ShardKV: Asymmetric KV cache compression inspired by the Shard paper (Research 109).
//!
//! K and V have fundamentally different structural properties in attention:
//! - K participates in the softmax numerator → errors amplified O(e^ε) (Research 081)
//! - V only scales the weighted sum → errors are linear O(w·ε)
//!
//! This motivates asymmetric compression:
//!
//! **K path** (precision-critical):
//!   1. Normalize → store norm
//!   2. undo_rope() — remove position-dependent phase
//!   3. PCA rotation (SpectralRotation from spectralquant)
//!   4. Water-fill bit allocation
//!   5. Lloyd-Max quantize → variable-bit pack
//!
//! **V path** (prefill — lossy but high quality via VQ):
//!   1. Normalize → store norm
//!   2. Walsh-Hadamard rotation
//!   3. K-means VQ on groups of 4 channels (256-entry codebook) → 2 bits/elem
//!
//! **V path** (decode streaming — lossless at 8 bits):
//!   1. Normalize → store norm
//!   2. Walsh-Hadamard rotation
//!   3. 8-bit Lloyd-Max quantize (TurboQuant-style, data-oblivious)
//!
//! **Sink + window**:
//!   - First `sink_tokens` positions: FP16 (attention sinks)
//!   - Last `window_tokens` positions: FP16 (recency window)
//!   - Everything in between: compressed via K/V paths above

use super::rope::RopeFreqs;
use super::types::{ShardCalibration, ShardConfig, ShardLayer};
use katgpt_core::simd::{simd_scale_inplace, simd_sum_sq};
use crate::spectralquant::spectral::{BitAllocator, LloydMaxQuantizer, waterfill_bits};
use crate::spectralquant::spectral_rotation::SpectralRotation;
use crate::spectralquant::types::LloydMaxCodebook;

/// Compressed KV cache with asymmetric K/V compression.
///
/// Sink + window tokens are stored as raw f32 (FP16-emulated via f32),
/// while interior tokens use the asymmetric codec.
pub struct ShardKVCache {
    // ── Position tracking (usize — group together for alignment) ──
    pos: usize,
    n_layers: usize,
    kv_dim: usize,
    #[allow(dead_code)]
    head_dim: usize,
    max_seq_len: usize,
    sink_tokens: usize,
    window_tokens: usize,
    /// End of prefill positions. Positions >= this use 8-bit decode streaming.
    /// 0 means prefill not finalized — all positions use VQ.
    prefill_len: usize,

    // ── Per-layer state (calibration + codebooks) ──
    pub layers: Vec<ShardLayer>,

    // ── Pre-built rotations (avoid clone per call) ──
    k_rotations: Vec<SpectralRotation>,
    /// Pre-allocated RoPE frequencies for undo/reapply (avoids Vec alloc per call).
    rope_freqs: RopeFreqs,

    // ── Compressed storage for interior tokens ──
    /// Packed K indices: [layer][position] → variable-bit packed bytes.
    /// Inner Vecs are pre-allocated with `kv_dim` capacity to avoid decode-path allocation.
    key_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position K norms.
    key_norms: Vec<Vec<f32>>,
    /// Packed V indices.
    val_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position V norms.
    val_norms: Vec<Vec<f32>>,

    // ── Sink + window (raw f32 storage) ──
    /// FP16-equivalent sink + window K storage: [layer][position][dim].
    key_raw: Vec<Vec<Vec<f32>>>,
    /// FP16-equivalent sink + window V storage.
    val_raw: Vec<Vec<Vec<f32>>>,

    // ── Scratch buffers (zero-alloc hot path) ──
    scratch_normalized: Vec<f32>,
    scratch_rotated: Vec<f32>,
    scratch_unrotated: Vec<f32>,
    scratch_indices: Vec<u8>,
    scratch_bits: Vec<u8>,
    /// Scratch for VQ group indices in prefill V path (avoids `vec![0u8; n_groups]` per call).
    scratch_vq_indices: Vec<u8>,
}

impl ShardKVCache {
    /// Build from pre-computed calibration data.
    ///
    /// Fits Lloyd-Max codebooks by generating synthetic data from the
    /// eigenvalue distribution for K, and from unit-sphere samples for V.
    pub fn from_calibration(config: &ShardConfig, k_calibrations: &[ShardCalibration]) -> Self {
        let n_layers = config.n_layers;
        let kv_dim = config.kv_dim;
        let head_dim = config.head_dim;
        let max_seq_len = config.max_seq_len;
        let sink_tokens = config.sink_tokens;
        let window_tokens = config.window_tokens;

        let mut layers: Vec<ShardLayer> = k_calibrations
            .iter()

            .map(|k_cal| {
                let d_eff = (k_cal.k_d_eff.ceil() as usize).max(1).min(head_dim);

                // K-path: bit allocation via water-fill
                let allocator = BitAllocator::new(config.min_tail_bits, config.max_bits);
                let (k_b_high, k_b_low) =
                    allocator.allocate(k_cal.k_d_eff, config.avg_bits_k, head_dim);

                // Water-fill for K semantic dims
                let k_bits_per_dim = {
                    let first_ev: Vec<f64> = k_cal
                        .k_eigenvalues
                        .iter()
                        .take(d_eff)
                        .map(|&x| x as f64)
                        .collect();
                    let total_semantic = k_b_high as usize * d_eff;
                    if total_semantic > 0 {
                        Some(waterfill_bits(
                            &first_ev,
                            total_semantic,
                            config.min_tail_bits,
                            Some(config.max_bits),
                        ))
                    } else {
                        None
                    }
                };

                // Placeholder codebooks — fitted below from synthetic data
                let k_semantic_codebook = if k_bits_per_dim.is_none() {
                    Some(LloydMaxCodebook {
                        centroids: vec![0.0; 1 << k_b_high],
                        n_bits: k_b_high,
                    })
                } else {
                    None
                };

                let k_per_dim_codebooks = k_bits_per_dim.as_ref().map(|bits| {
                    bits.iter()
                        .map(|&b| LloydMaxCodebook {
                            centroids: vec![0.0; 1 << b],
                            n_bits: b,
                        })
                        .collect()
                });

                let k_tail_codebook = LloydMaxCodebook {
                    centroids: vec![0.0; 1 << k_b_low.max(1)],
                    n_bits: k_b_low.max(1),
                };

                // V-path: placeholder VQ codebook — fitted below from synthetic data
                let gs = config.v_vq_group_size;
                let cs = config.v_vq_codebook_size;
                let v_bits_per_elem = (cs as f32 - 1.0).log2() / gs as f32;
                let v_vq_codebook = super::types::VqCodebook {
                    centroids: vec![0.0; cs * gs],
                    codebook_size: cs,
                    group_size: gs,
                };

                // Decode streaming codebook placeholder — fitted below
                let decode_stream_bits = config.decode_stream_bits;
                let decode_v_codebook = LloydMaxCodebook {
                    centroids: vec![0.0; 1 << decode_stream_bits],
                    n_bits: decode_stream_bits,
                };

                ShardLayer {
                    calibration: k_cal.clone(),
                    d_eff,
                    k_b_high,
                    k_b_low,
                    k_bits_per_dim,
                    k_per_dim_codebooks,
                    k_semantic_codebook,
                    k_tail_codebook,
                    v_vq_codebook,
                    v_bits_per_elem,
                    decode_stream_bits,
                    decode_v_codebook,
                }
            })
            .collect();

        // Fit codebooks from synthetic data matching the actual pipeline.
        // K path: random unit-norm vector → rotate by V^T → fit codebooks.
        // V path: random unit-norm vector → Hadamard → fit codebook.
        let n_synthetic = 512;
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            let d_eff = layer.d_eff;
            let eigenvectors = &layer.calibration.k_eigenvectors;
            let mut rng = crate::types::Rng::new(config.seed.wrapping_add(layer_idx as u64 * 31));

            // Generate K-path synthetic data: normalize → rotate by V^T
            let k_synthetic: Vec<Vec<f32>> = (0..n_synthetic)
                .map(|_| {
                    let mut x: Vec<f32> = (0..head_dim).map(|_| rng.normal()).collect();
                    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                    for v in x.iter_mut() {
                        *v /= norm;
                    }
                    let mut rotated = vec![0.0f32; head_dim];
                    for j in 0..head_dim {
                        let mut sum = 0.0f32;
                        for i in 0..head_dim {
                            sum += x[i] * eigenvectors[i * head_dim + j];
                        }
                        rotated[j] = sum;
                    }
                    rotated
                })
                .collect();

            // Fit K tail codebook
            let k_tail_data: Vec<f32> = k_synthetic
                .iter()
                .flat_map(|s| s.iter().skip(d_eff).copied())
                .collect();
            let mut k_tail_q = LloydMaxQuantizer::new(
                layer.k_b_low.max(1),
                200,
                config.seed.wrapping_add(layer_idx as u64 * 51 + 1),
            );
            k_tail_q.fit(&k_tail_data);
            layer.k_tail_codebook.centroids = k_tail_q.centroids().to_vec();

            // Fit K semantic codebook(s)
            if let Some(ref mut cb) = layer.k_semantic_codebook {
                let sem_data: Vec<f32> = k_synthetic
                    .iter()
                    .flat_map(|s| s.iter().take(d_eff).copied())
                    .collect();
                let mut sem_q = LloydMaxQuantizer::new(
                    layer.k_b_high,
                    200,
                    config.seed.wrapping_add(layer_idx as u64 * 51 + 2),
                );
                sem_q.fit(&sem_data);
                cb.centroids = sem_q.centroids().to_vec();
            } else if let Some(ref mut per_dim) = layer.k_per_dim_codebooks {
                let bits = layer.k_bits_per_dim.as_ref();
                for (dim, cb) in per_dim.iter_mut().enumerate() {
                    let dim_data: Vec<f32> = k_synthetic.iter().map(|s| s[dim]).collect();
                    let bits_for_dim = bits
                        .and_then(|b| b.get(dim).copied())
                        .unwrap_or(layer.k_b_high)
                        .max(1);
                    let mut q = LloydMaxQuantizer::new(
                        bits_for_dim,
                        200,
                        config.seed.wrapping_add((dim + 10) as u64),
                    );
                    q.fit(&dim_data);
                    cb.centroids = q.centroids().to_vec();
                }
            }

            // Fit V VQ codebook: generate data from normalize → Hadamard pipeline,
            // then group into groups of `gs` and run K-means.
            let gs = config.v_vq_group_size;
            let cs = config.v_vq_codebook_size;
            let v_synthetic: Vec<Vec<f32>> = (0..n_synthetic)
                .map(|_| {
                    let mut x: Vec<f32> = (0..kv_dim).map(|_| rng.normal()).collect();
                    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                    for v in x.iter_mut() {
                        *v /= norm;
                    }
                    hadamard_transform_inplace(&mut x);
                    x
                })
                .collect();

            // Build grouped data for K-means: each row is `gs` consecutive channels
            let n_groups = kv_dim / gs;
            let mut vq_data = Vec::with_capacity(n_synthetic * n_groups * gs);
            for s in &v_synthetic {
                for g in 0..n_groups {
                    let base = g * gs;
                    for d in 0..gs {
                        vq_data.push(s[base + d]);
                    }
                }
            }
            let vq_centroids = kmeans_fit(
                &vq_data,
                n_synthetic * n_groups,
                gs,
                cs,
                30,
                config.seed.wrapping_add(layer_idx as u64 * 51 + 3),
            );
            layer.v_vq_codebook.centroids = vq_centroids;

            // Fit decode streaming codebook: 8-bit Lloyd-Max on Hadamard-rotated data
            let v_data: Vec<f32> = v_synthetic.iter().flat_map(|s| s.iter().copied()).collect();
            let mut decode_q = LloydMaxQuantizer::new(
                layer.decode_stream_bits,
                200,
                config.seed.wrapping_add(layer_idx as u64 * 51 + 4),
            );
            decode_q.fit(&v_data);
            layer.decode_v_codebook.centroids = decode_q.centroids().to_vec();
        }

        // Allocate storage — inner Vecs pre-sized with `kv_dim` capacity to avoid
        // allocation on the decode hot path (decode always writes exactly `kv_dim` bytes).
        let key_indices = (0..n_layers)
            .map(|_| (0..max_seq_len).map(|_| Vec::with_capacity(kv_dim)).collect())
            .collect();
        let key_norms = (0..n_layers).map(|_| vec![0.0f32; max_seq_len]).collect();
        let val_indices = (0..n_layers)
            .map(|_| (0..max_seq_len).map(|_| Vec::with_capacity(kv_dim)).collect())
            .collect();
        let val_norms = (0..n_layers).map(|_| vec![0.0f32; max_seq_len]).collect();

        let raw_slots = sink_tokens + window_tokens;
        let key_raw = (0..n_layers)
            .map(|_| (0..raw_slots).map(|_| vec![0.0f32; kv_dim]).collect())
            .collect();
        let val_raw = (0..n_layers)
            .map(|_| (0..raw_slots).map(|_| vec![0.0f32; kv_dim]).collect())
            .collect();

        let k_rotations: Vec<SpectralRotation> = k_calibrations
            .iter()
            .map(|k_cal| SpectralRotation::new(k_cal.k_eigenvectors.clone(), k_cal.head_dim))
            .collect();

        let rope_freqs = RopeFreqs::new(head_dim);

        let n_vq_groups = kv_dim / config.v_vq_group_size;

        Self {
            pos: 0,
            n_layers,
            kv_dim,
            head_dim,
            max_seq_len,
            sink_tokens,
            window_tokens,
            prefill_len: 0,
            layers,
            k_rotations,
            rope_freqs,
            key_indices,
            key_norms,
            val_indices,
            val_norms,
            key_raw,
            val_raw,
            scratch_normalized: vec![0.0f32; kv_dim],
            scratch_rotated: vec![0.0f32; kv_dim],
            scratch_unrotated: vec![0.0f32; kv_dim],
            scratch_indices: vec![0u8; kv_dim],
            scratch_bits: vec![0u8; kv_dim],
            scratch_vq_indices: vec![0u8; n_vq_groups],
        }
    }

    /// Whether a position should be stored as raw (FP16-like) or compressed.
    fn is_raw_slot(&self, pos: usize) -> bool {
        pos < self.sink_tokens || pos + self.window_tokens >= self.max_seq_len
    }

    /// Map a position to the raw storage slot index.
    fn raw_slot_index(&self, pos: usize) -> usize {
        if pos < self.sink_tokens {
            pos
        } else {
            // Window slot: maps to [sink_tokens..sink_tokens+window_tokens)
            self.sink_tokens + (pos - (self.max_seq_len - self.window_tokens))
        }
    }

    /// Store a key vector at given layer and position.
    ///
    /// K path (prefill): normalize → undo RoPE → PCA rotate → quantize → pack.
    /// K path (decode): normalize → Hadamard rotate → 8-bit Lloyd-Max streaming.
    /// Sink/window tokens: store as raw f32.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);

        // Sink + window: raw f32 storage
        if self.is_raw_slot(pos) {
            let slot = self.raw_slot_index(pos);
            self.key_raw[layer][slot].copy_from_slice(key);
            self.key_norms[layer][pos] = simd_norm(key);
            return;
        }

        // 1. Compute and store norm
        let norm = simd_norm(key);
        if norm < 1e-8 {
            self.key_norms[layer][pos] = 0.0;
            return;
        }
        self.key_norms[layer][pos] = norm;

        // 2. Normalize
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..self.kv_dim].copy_from_slice(key);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Check if decode streaming (post-prefill)
        let is_decode = self.prefill_len > 0 && pos >= self.prefill_len;

        if is_decode {
            // Decode streaming: Hadamard + 8-bit Lloyd-Max (data-oblivious, no drift)
            hadamard_transform_inplace(&mut self.scratch_normalized);
            let cb = &self.layers[layer].decode_v_codebook; // same distribution after Hadamard
            for (i, &v) in self.scratch_normalized.iter().enumerate() {
                self.scratch_indices[i] = quantize_to_idx(v, &cb.centroids);
            }
            // Store as raw bytes (8 bits = 1 byte per index)
            // No allocation: inner Vec was pre-allocated with `kv_dim` capacity.
            let buf = &mut self.key_indices[layer][pos];
            buf.clear();
            buf.extend_from_slice(&self.scratch_indices[..self.kv_dim]);
        } else {
            // Prefill: undo RoPE → PCA → quantize → pack
            let layer_state = &self.layers[layer];
            let d_eff = layer_state.d_eff;
            let k_b_low = layer_state.k_b_low.max(1);

            // 3. Undo RoPE (uses cached RopeFreqs — no allocation)
            self.rope_freqs.apply(&mut self.scratch_normalized, pos, true);

            // 4. PCA rotation (use pre-built rotation, no clone)
            let rotation = &self.k_rotations[layer];
            rotation.rotate(&self.scratch_normalized, &mut self.scratch_rotated);

            // 5. Quantize semantic dims
            if let Some(ref per_dim) = layer_state.k_per_dim_codebooks {
                for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                    self.scratch_indices[i] =
                        quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
                }
            } else if let Some(ref cb) = layer_state.k_semantic_codebook {
                for i in 0..d_eff {
                    self.scratch_indices[i] =
                        quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
                }
            }

            // Quantize tail dims
            let tail_cb = &layer_state.k_tail_codebook;
            for (i, &v) in self.scratch_rotated.iter().enumerate().skip(d_eff) {
                self.scratch_indices[i] = quantize_to_idx(v, &tail_cb.centroids);
            }

            // 6. Build bits array and pack
            if let Some(ref bits) = layer_state.k_bits_per_dim {
                self.scratch_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
            } else {
                self.scratch_bits[..d_eff].fill(layer_state.k_b_high);
            }
            self.scratch_bits[d_eff..self.kv_dim].fill(k_b_low);

            pack_variable_bits(
                &self.scratch_indices[..self.kv_dim],
                &self.scratch_bits[..self.kv_dim],
                &mut self.key_indices[layer][pos],
            );
        }
    }

    /// Store a value vector at given layer and position.
    ///
    /// V path (prefill): normalize → Hadamard rotate → K-means VQ (groups of 4).
    /// V path (decode): normalize → Hadamard rotate → 8-bit Lloyd-Max streaming.
    /// Sink/window tokens: store as raw f32.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);

        // Sink + window: raw f32 storage
        if self.is_raw_slot(pos) {
            let slot = self.raw_slot_index(pos);
            self.val_raw[layer][slot].copy_from_slice(value);
            self.val_norms[layer][pos] = simd_norm(value);
            return;
        }

        let layer_state = &self.layers[layer];

        // 1. Compute and store norm
        let norm = simd_norm(value);
        if norm < 1e-8 {
            self.val_norms[layer][pos] = 0.0;
            return;
        }
        self.val_norms[layer][pos] = norm;

        // 2. Normalize
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..self.kv_dim].copy_from_slice(value);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // 3. Hadamard rotation
        hadamard_transform_inplace(&mut self.scratch_normalized);

        // 4. Check if decode streaming (post-prefill) or prefill VQ
        let is_decode = self.prefill_len > 0 && pos >= self.prefill_len;

        if is_decode {
            // Decode streaming: 8-bit Lloyd-Max per coordinate (guaranteed lossless)
            let cb = &layer_state.decode_v_codebook;
            for (i, &v) in self.scratch_normalized.iter().enumerate() {
                self.scratch_indices[i] = quantize_to_idx(v, &cb.centroids);
            }
            // Store as raw bytes (8 bits = 1 byte per index)
            // No allocation: inner Vec was pre-allocated with `kv_dim` capacity.
            let buf = &mut self.val_indices[layer][pos];
            buf.clear();
            buf.extend_from_slice(&self.scratch_indices[..self.kv_dim]);
        } else {
            // Prefill VQ: K-means VQ on groups of `group_size` channels
            let cb = &layer_state.v_vq_codebook;
            let gs = cb.group_size;
            let n_groups = self.kv_dim / gs;

            for g in 0..n_groups {
                let base = g * gs;
                let mut best_dist = f32::MAX;
                let mut best_idx = 0u8;
                for c in 0..cb.codebook_size {
                    let mut dist = 0.0f32;
                    for d in 0..gs {
                        let diff = self.scratch_normalized[base + d] - cb.centroids[c * gs + d];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = c as u8;
                    }
                }
                self.scratch_vq_indices[g] = best_idx;
            }

            // No allocation: inner Vec was pre-allocated with `kv_dim` capacity.
            let buf = &mut self.val_indices[layer][pos];
            buf.clear();
            buf.extend_from_slice(&self.scratch_vq_indices[..n_groups]);
        }
    }

    /// Dequantize a key at given layer and position into `out`.
    ///
    /// K path (prefill): unpack → dequantize → inverse PCA → reapply RoPE → scale by norm.
    /// K path (decode): Lloyd-Max lookup → inverse Hadamard → scale by norm.
    /// Sink/window tokens: copy from raw storage.
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);

        // Sink + window: raw f32 copy
        if self.is_raw_slot(pos) {
            let slot = self.raw_slot_index(pos);
            out.copy_from_slice(&self.key_raw[layer][slot]);
            return;
        }

        let norm = self.key_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let is_decode = self.prefill_len > 0 && pos >= self.prefill_len;

        if is_decode {
            // Decode streaming dequant: byte indices → Lloyd-Max lookup → inverse Hadamard
            let cb = &self.layers[layer].decode_v_codebook;
            let indices = &self.key_indices[layer][pos];
            for (i, byte) in indices.iter().enumerate().take(self.kv_dim) {
                self.scratch_rotated[i] = dequantize_idx(*byte, &cb.centroids);
            }
            // Inverse Hadamard
            hadamard_transform_inplace(&mut self.scratch_rotated);
            // Scale by norm → output
            out.copy_from_slice(&self.scratch_rotated);
            simd_scale_inplace(out, norm);
        } else {
            // Prefill: unpack → dequantize → inverse PCA → reapply RoPE
            let layer_state = &self.layers[layer];
            let d_eff = layer_state.d_eff;
            let k_b_low = layer_state.k_b_low.max(1);

            // Build bits array
            if let Some(ref bits) = layer_state.k_bits_per_dim {
                self.scratch_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
            } else {
                self.scratch_bits[..d_eff].fill(layer_state.k_b_high);
            }
            self.scratch_bits[d_eff..self.kv_dim].fill(k_b_low);

            // Unpack
            unpack_variable_bits(
                &self.key_indices[layer][pos],
                &self.scratch_bits[..self.kv_dim],
                self.kv_dim,
                &mut self.scratch_indices,
            );

            // Dequantize into scratch_rotated
            if let Some(ref per_dim) = layer_state.k_per_dim_codebooks {
                for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                    self.scratch_rotated[i] =
                        dequantize_idx(self.scratch_indices[i], &cb.centroids);
                }
            } else if let Some(ref cb) = layer_state.k_semantic_codebook {
                for i in 0..d_eff {
                    self.scratch_rotated[i] =
                        dequantize_idx(self.scratch_indices[i], &cb.centroids);
                }
            }
            let tail_cb = &layer_state.k_tail_codebook;
            for i in d_eff..self.kv_dim {
                self.scratch_rotated[i] =
                    dequantize_idx(self.scratch_indices[i], &tail_cb.centroids);
            }

            // Inverse PCA rotation (use pre-built rotation, no clone)
            let rotation = &self.k_rotations[layer];
            rotation.unrotate(&self.scratch_rotated, &mut self.scratch_unrotated);

            // Reapply RoPE (uses cached RopeFreqs — no allocation)
            self.rope_freqs.apply(&mut self.scratch_unrotated, pos, false);

            // Scale by norm → output
            out.copy_from_slice(&self.scratch_unrotated);
            simd_scale_inplace(out, norm);
        }
    }

    /// Dequantize a value at given layer and position into `out`.
    ///
    /// V path (prefill VQ): VQ lookup → inverse Hadamard → scale by norm.
    /// V path (decode streaming): Lloyd-Max lookup → inverse Hadamard → scale by norm.
    /// Sink/window tokens: copy from raw storage.
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);

        // Sink + window: raw f32 copy
        if self.is_raw_slot(pos) {
            let slot = self.raw_slot_index(pos);
            out.copy_from_slice(&self.val_raw[layer][slot]);
            return;
        }

        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let is_decode = self.prefill_len > 0 && pos >= self.prefill_len;

        if is_decode {
            // Decode streaming dequant: byte indices → Lloyd-Max lookup
            let cb = &layer_state.decode_v_codebook;
            let indices = &self.val_indices[layer][pos];
            for (i, byte) in indices.iter().enumerate().take(self.kv_dim) {
                self.scratch_rotated[i] = dequantize_idx(*byte, &cb.centroids);
            }
        } else {
            // Prefill VQ dequant: VQ group indices → centroid lookup
            let cb = &layer_state.v_vq_codebook;
            let gs = cb.group_size;
            let indices = &self.val_indices[layer][pos];
            let n_groups = indices.len().min(self.kv_dim / gs);
            // stride math: `g` drives both `indices[g]` and `base = g * gs`
            #[allow(clippy::needless_range_loop)]
            for g in 0..n_groups {
                let idx = indices[g] as usize;
                let base = g * gs;
                // memcpy of `gs` consecutive centroids beats scalar loop.
                self.scratch_rotated[base..base + gs]
                    .copy_from_slice(&cb.centroids[idx * gs..idx * gs + gs]);
            }
        }

        // Inverse Hadamard
        hadamard_transform_inplace(&mut self.scratch_rotated);

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_rotated);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for a new sequence.
    pub fn reset(&mut self) {
        self.pos = 0;
        for layer in 0..self.n_layers {
            for p in 0..self.max_seq_len {
                self.key_indices[layer][p].clear();
                self.key_norms[layer][p] = 0.0;
                self.val_indices[layer][p].clear();
                self.val_norms[layer][p] = 0.0;
            }
            for slot in 0..self.sink_tokens + self.window_tokens {
                self.key_raw[layer][slot].fill(0.0);
                self.val_raw[layer][slot].fill(0.0);
            }
        }
    }

    /// Mark that prefill has completed at this position.
    ///
    /// All subsequent `store_value` calls will use 8-bit decode streaming
    /// (guaranteed lossless) instead of VQ (lossy but compact).
    pub fn mark_prefill_done(&mut self, pos: usize) {
        self.prefill_len = pos;
    }

    /// Current write position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set the current write position.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Compression ratio vs f32 (32 bits per coordinate).
    ///
    /// Accounts for both K and V paths and the sink/window overhead.
    pub fn compression_ratio(&self) -> f32 {
        if self.layers.is_empty() {
            return 1.0;
        }
        let layer0 = &self.layers[0];
        let d_eff = layer0.d_eff;

        // K bits per token
        let k_semantic_bits: f32 = if let Some(ref bits) = layer0.k_bits_per_dim {
            bits.iter().take(d_eff).map(|&b| b as f32).sum()
        } else {
            d_eff as f32 * layer0.k_b_high as f32
        };
        let k_tail_bits = (self.kv_dim - d_eff) as f32 * layer0.k_b_low.max(1) as f32;
        let k_bits = k_semantic_bits + k_tail_bits + 32.0; // +32 for norm

        // V bits per token: VQ prefill path
        // Each group of `group_size` channels uses log2(codebook_size) bits
        let v_bits = self.kv_dim as f32 * layer0.v_bits_per_elem + 32.0; // +32 for norm

        // Total compressed bits per (K,V) pair
        let compressed_bits = k_bits + v_bits;
        let original_bits = 2.0 * self.kv_dim as f32 * 32.0;

        if compressed_bits < 1.0 {
            return 1.0;
        }
        original_bits / compressed_bits
    }
}

impl crate::types::QuantizedKVCache for ShardKVCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.store_key(layer, pos, key);
    }
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.store_value(layer, pos, value);
    }
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_key_into(layer, pos, out);
    }
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_value_into(layer, pos, out);
    }
    fn reset(&mut self) {
        self.reset();
    }
    fn pos(&self) -> usize {
        self.pos()
    }
    fn set_pos(&mut self, pos: usize) {
        self.set_pos(pos);
    }
}

// ── Internal helpers ──────────────────────────────────────────────────

fn simd_norm(v: &[f32]) -> f32 {
    simd_sum_sq(v, v.len()).sqrt()
}

/// In-place Walsh-Hadamard transform.
///
/// Recursively applies H₂ = [[1,1],[1,-1]]/√2 to power-of-2 length vectors.
/// For non-power-of-2 lengths, falls back to identity (no rotation).
fn hadamard_transform_inplace(x: &mut [f32]) {
    let n = x.len();
    if n == 0 || n == 1 {
        return;
    }

    // Check power-of-2
    if !n.is_power_of_two() {
        // Fallback: no rotation for non-power-of-2 dims
        return;
    }

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let mut step = 2;
    while step <= n {
        let half = step / 2;
        for block_start in (0..n).step_by(step) {
            for i in 0..half {
                let a = x[block_start + i];
                let b = x[block_start + half + i];
                x[block_start + i] = (a + b) * inv_sqrt2;
                x[block_start + half + i] = (a - b) * inv_sqrt2;
            }
        }
        step *= 2;
    }
}

fn quantize_to_idx(value: f32, centroids: &[f32]) -> u8 {
    let n = centroids.len();
    debug_assert!(n <= 256, "codebook too large for u8 index");
    if n <= 1 {
        return 0;
    }
    // Lloyd-Max codebook centroids are sorted ascending (see
    // `LloydMaxQuantizer::fit` — initialized via sorted quantile placement).
    // Binary search gives O(log n) instead of O(n) on the per-token-per-channel
    // quantize hot path. Matches `spectral_kv_cache::quantize_to_idx`.
    if value <= centroids[0] {
        return 0;
    }
    if value >= centroids[n - 1] {
        return (n - 1) as u8;
    }
    // Find the bracket [lo, lo+1].
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if centroids[mid] <= value {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Pick the closer of the two brackets.
    let d_lo = value - centroids[lo];
    let d_hi = centroids[hi] - value;
    if d_lo <= d_hi {
        lo as u8
    } else {
        hi as u8
    }
}

fn dequantize_idx(idx: u8, centroids: &[f32]) -> f32 {
    centroids.get(idx as usize).copied().unwrap_or(0.0)
}

/// K-means clustering for VQ codebook fitting.
///
/// Returns `k × dims` centroids (flattened row-major).
/// Data is `n_points × dims`, flattened row-major.
fn kmeans_fit(
    data: &[f32],
    n_points: usize,
    dims: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = crate::types::Rng::new(seed);

    // Initialize centroids by randomly selecting k data points
    let mut centroids = vec![0.0f32; k * dims];
    for c in 0..k {
        let idx = (rng.next() as usize) % n_points;
        centroids[c * dims..(c + 1) * dims].copy_from_slice(&data[idx * dims..(idx + 1) * dims]);
    }

    let mut assignments = vec![0usize; n_points];
    let mut sums = vec![0.0f32; k * dims];
    let mut counts = vec![0usize; k];

    for _ in 0..max_iter {
        // Assign each point to nearest centroid
        let mut changed = false;
        for p in 0..n_points {
            let mut best_dist = f32::MAX;
            let mut best_c = 0;
            for c in 0..k {
                let mut dist = 0.0f32;
                for d in 0..dims {
                    let diff = data[p * dims + d] - centroids[c * dims + d];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[p] != best_c {
                changed = true;
                assignments[p] = best_c;
            }
        }
        if !changed {
            break;
        }

        // Recompute centroids (reuse pre-allocated buffers)
        sums.fill(0.0);
        counts.fill(0);
        for p in 0..n_points {
            let c = assignments[p];
            counts[c] += 1;
            for d in 0..dims {
                sums[c * dims + d] += data[p * dims + d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv_count = 1.0 / counts[c] as f32;
                for d in 0..dims {
                    centroids[c * dims + d] = sums[c * dims + d] * inv_count;
                }
            }
        }
    }

    centroids
}

/// Pack variable-bit indices into bytes (LSB-first).
/// `bits_per_dim.len()` must be `>= indices.len()` (callers always pass a full-length array).
fn pack_variable_bits(indices: &[u8], bits_per_dim: &[u8], out: &mut Vec<u8>) {
    out.clear();
    // Pre-reserve to avoid reallocation in the push loop. Inner Vecs were pre-sized
    // with `kv_dim` capacity at construction; reserve keeps us within that capacity.
    let total_bits: usize = bits_per_dim
        .iter()
        .take(indices.len())
        .map(|&b| b as usize)
        .sum();
    out.reserve(total_bits.div_ceil(8));
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0u32;

    for (i, &idx) in indices.iter().enumerate() {
        let bits = bits_per_dim[i] as u32;
        bit_buffer |= (idx as u64) << bits_in_buffer;
        bits_in_buffer += bits;

        while bits_in_buffer >= 8 {
            out.push((bit_buffer & 0xFF) as u8);
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }
    if bits_in_buffer > 0 {
        out.push((bit_buffer & 0xFF) as u8);
    }
}

/// Unpack variable-bit indices from bytes (LSB-first).
/// `bits_per_dim.len()` must be `>= n_dims` (callers always pass a full-length array).
fn unpack_variable_bits(packed: &[u8], bits_per_dim: &[u8], n_dims: usize, out: &mut [u8]) {
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0u32;
    let mut byte_idx = 0;

    for (i, o) in out.iter_mut().enumerate().take(n_dims) {
        let bits = bits_per_dim[i] as u32;
        while bits_in_buffer < bits && byte_idx < packed.len() {
            bit_buffer |= (packed[byte_idx] as u64) << bits_in_buffer;
            bits_in_buffer += 8;
            byte_idx += 1;
        }
        let mask = (1u64 << bits) - 1;
        *o = (bit_buffer & mask) as u8;
        bit_buffer >>= bits;
        bits_in_buffer -= bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectralquant::spectral::participation_ratio;

    fn make_test_calibration(head_dim: usize) -> ShardCalibration {
        let mut eigenvectors = vec![0.0f32; head_dim * head_dim];
        for i in 0..head_dim {
            eigenvectors[i * head_dim + i] = 1.0;
        }
        // Exponential decay eigenvalues — realistic spectral profile
        let eigenvalues: Vec<f32> = (0..head_dim)
            .map(|i| 10.0 * 0.8f32.powi(i as i32))
            .collect();
        let d_eff = participation_ratio(&eigenvalues);
        ShardCalibration {
            k_eigenvectors: eigenvectors,
            k_eigenvalues: eigenvalues,
            k_d_eff: d_eff,
            head_dim,
        }
    }

    fn make_test_config(head_dim: usize, max_seq_len: usize) -> ShardConfig {
        ShardConfig {
            avg_bits_k: 4.0,
            avg_bits_v: 2.0,
            min_tail_bits: 1,
            max_bits: 8,
            n_layers: 1,
            kv_dim: head_dim,
            head_dim,
            max_seq_len,
            sink_tokens: 4,
            window_tokens: 4,
            seed: 42,
            v_vq_group_size: 4,
            v_vq_codebook_size: 256,
            decode_stream_bits: 8,
        }
    }

    #[test]
    fn test_kv_cache_roundtrip() {
        let head_dim = 128;
        let max_seq_len = 32;
        let config = make_test_config(head_dim, max_seq_len);
        let cal = make_test_calibration(head_dim);
        let mut cache = ShardKVCache::from_calibration(&config, &[cal]);

        let layer = 0;
        let pos = 10; // Interior position (not sink/window)
        let key: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 + 1.0).sin() * 0.5)
            .collect();
        let value: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 + 1.0).cos() * 0.3)
            .collect();

        cache.store_key(layer, pos, &key);
        cache.store_value(layer, pos, &value);

        let mut key_out = vec![0.0f32; head_dim];
        let mut val_out = vec![0.0f32; head_dim];
        cache.dequantize_key_into(layer, pos, &mut key_out);
        cache.dequantize_value_into(layer, pos, &mut val_out);

        // Cosine similarity should be high
        let k_cos = cosine_similarity(&key, &key_out);
        let v_cos = cosine_similarity(&value, &val_out);
        assert!(k_cos > 0.95, "K cosine similarity too low: {k_cos}");
        assert!(v_cos > 0.70, "V cosine similarity too low: {v_cos}");
    }

    #[test]
    fn test_zero_vector_handling() {
        let head_dim = 64;
        let max_seq_len = 16;
        let config = make_test_config(head_dim, max_seq_len);
        let cal = make_test_calibration(head_dim);
        let mut cache = ShardKVCache::from_calibration(&config, &[cal]);

        let layer = 0;
        let pos = 5;
        let zeros = vec![0.0f32; head_dim];

        cache.store_key(layer, pos, &zeros);
        cache.store_value(layer, pos, &zeros);

        let mut key_out = vec![0.0f32; head_dim];
        let mut val_out = vec![0.0f32; head_dim];
        cache.dequantize_key_into(layer, pos, &mut key_out);
        cache.dequantize_value_into(layer, pos, &mut val_out);

        assert!(
            key_out.iter().all(|&x| x.abs() < 1e-6),
            "zero K should dequantize to zero"
        );
        assert!(
            val_out.iter().all(|&x| x.abs() < 1e-6),
            "zero V should dequantize to zero"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let head_dim = 128;
        let max_seq_len = 16;
        let config = make_test_config(head_dim, max_seq_len);
        let cal = make_test_calibration(head_dim);
        let cache = ShardKVCache::from_calibration(&config, &[cal]);

        let ratio = cache.compression_ratio();
        // At avg_bits_k=4, avg_bits_v=2, should compress significantly
        assert!(ratio > 2.0, "compression ratio should be > 2x, got {ratio}");
        assert!(
            ratio < 50.0,
            "compression ratio should be < 50x, got {ratio}"
        );
    }

    #[test]
    fn test_sink_window_exact_roundtrip() {
        let head_dim = 64;
        let max_seq_len = 16;
        let config = make_test_config(head_dim, max_seq_len);
        let cal = make_test_calibration(head_dim);
        let mut cache = ShardKVCache::from_calibration(&config, &[cal]);

        // Position 0 is a sink token → exact roundtrip
        let key: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let value: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).cos()).collect();

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let mut key_out = vec![0.0f32; head_dim];
        let mut val_out = vec![0.0f32; head_dim];
        cache.dequantize_key_into(0, 0, &mut key_out);
        cache.dequantize_value_into(0, 0, &mut val_out);

        for (i, (orig, rec)) in key.iter().zip(key_out.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "sink K should be exact at [{i}]: {orig} vs {rec}"
            );
        }
        for (i, (orig, rec)) in value.iter().zip(val_out.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "sink V should be exact at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_multi_position() {
        let head_dim = 64;
        let max_seq_len = 16;
        let config = make_test_config(head_dim, max_seq_len);
        let cal = make_test_calibration(head_dim);
        let mut cache = ShardKVCache::from_calibration(&config, &[cal]);

        let layer = 0;
        let mut rng = crate::types::Rng::new(123);

        // Store multiple positions
        let keys: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..head_dim).map(|_| rng.normal() * 0.5).collect())
            .collect();
        let values: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..head_dim).map(|_| rng.normal() * 0.3).collect())
            .collect();

        for pos in 0..10usize {
            cache.store_key(layer, pos, &keys[pos]);
            cache.store_value(layer, pos, &values[pos]);
        }

        // Verify each independently
        let mut key_out = vec![0.0f32; head_dim];
        let mut val_out = vec![0.0f32; head_dim];
        for pos in 0..10usize {
            cache.dequantize_key_into(layer, pos, &mut key_out);
            cache.dequantize_value_into(layer, pos, &mut val_out);

            let k_cos = cosine_similarity(&keys[pos], &key_out);
            let _v_cos = cosine_similarity(&values[pos], &val_out);
            // Sink tokens (pos 0-3) are exact; others are compressed
            if pos < 4 {
                assert!(k_cos > 0.999, "sink K cos at pos={pos}: {k_cos}");
            } else {
                assert!(k_cos > 0.90, "K cos at pos={pos}: {k_cos}");
            }
        }
    }

    #[test]
    fn test_reset_clears() {
        let head_dim = 64;
        let max_seq_len = 16;
        let config = make_test_config(head_dim, max_seq_len);
        let cal = make_test_calibration(head_dim);
        let mut cache = ShardKVCache::from_calibration(&config, &[cal]);

        let key: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        cache.store_key(0, 5, &key);
        assert_eq!(cache.pos(), 0);

        cache.set_pos(10);
        assert_eq!(cache.pos(), 10);

        cache.reset();
        assert_eq!(cache.pos(), 0);

        let mut key_out = vec![0.0f32; head_dim];
        cache.dequantize_key_into(0, 5, &mut key_out);
        assert!(
            key_out.iter().all(|&x| x.abs() < 1e-6),
            "reset should clear data"
        );
    }

    #[test]
    fn test_multi_layer_independence() {
        let head_dim = 64;
        let max_seq_len = 16;
        let mut config = make_test_config(head_dim, max_seq_len);
        config.n_layers = 2;
        let cal = make_test_calibration(head_dim);
        let mut cache = ShardKVCache::from_calibration(&config, &[cal.clone(), cal]);

        let key0: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let key1: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).cos()).collect();

        cache.store_key(0, 5, &key0);
        cache.store_key(1, 5, &key1);

        let mut out0 = vec![0.0f32; head_dim];
        let mut out1 = vec![0.0f32; head_dim];
        cache.dequantize_key_into(0, 5, &mut out0);
        cache.dequantize_key_into(1, 5, &mut out1);

        let cos0 = cosine_similarity(&key0, &out0);
        let cos1 = cosine_similarity(&key1, &out1);
        assert!(cos0 > 0.90, "layer 0 K cos: {cos0}");
        assert!(cos1 > 0.90, "layer 1 K cos: {cos1}");
    }

    #[test]
    fn test_hadamard_roundtrip() {
        let n = 128;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0).sin()).collect();
        let original = x.clone();

        hadamard_transform_inplace(&mut x);
        hadamard_transform_inplace(&mut x); // H² = I

        for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Hadamard roundtrip failed at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let dims = 64;
        let bits = vec![4u8; dims];
        let indices: Vec<u8> = (0..dims).map(|i| (i % 16) as u8).collect();

        let mut packed = Vec::new();
        pack_variable_bits(&indices, &bits, &mut packed);

        let mut unpacked = vec![0u8; dims];
        unpack_variable_bits(&packed, &bits, dims, &mut unpacked);

        assert_eq!(indices, unpacked);
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}
