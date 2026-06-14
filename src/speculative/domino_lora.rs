//! Domino LoRA causal correction adapter for speculative decoding (Plan 231).
//!
//! Lightweight inference-only adapter that reads `domino_lora.bin` (from riir-engine training).
//! Applies a LoRA-style correction ΔL to draft model logits, with a GRU for causal state
//! tracking across positions in a draft block.
//!
//! Binary format (DMAD): magic(4) + version(4) + n_embd(4) + vocab_size(4) + rank(4) +
//! gru_hidden(4) + embed_dim(4) = 32-byte header, then f32 weights, then blake3(32).

use std::path::Path;

use crate::simd::simd_add_inplace;
use crate::types::matmul;

// ── Binary format constants ──────────────────────────────────
const DOMINO_MAGIC: &[u8; 4] = b"DMAD";
const DOMINO_VERSION: u32 = 1;

// ── DominoGRU — inference-only GRU for causal state tracking ──

/// Minimal GRU cell (inference-only, no save/load needed).
///
/// Standard GRU with 6 weights and 6 biases:
/// - reset gate:    z = σ(Wz·[x,h] + bz)
/// - update gate:   r = σ(Wr·[x,h] + br)
/// - new gate:      n = tanh(Wn·[x, r⊙h] + bn)
/// - output:        h' = (1-z)⊙h + z⊙n
pub struct DominoGRU {
    // Reset gate weights and biases [hidden_size, input_size+hidden_size]
    pub wz: Vec<f32>,
    pub bz: Vec<f32>,
    // Update gate
    pub wr: Vec<f32>,
    pub br: Vec<f32>,
    // New gate
    pub wn: Vec<f32>,
    pub bn: Vec<f32>,
    // Output projection
    pub wo: Vec<f32>,
    pub bo: Vec<f32>,

    pub input_size: usize,
    pub hidden_size: usize,
}

impl DominoGRU {
    /// Parse GRU weights from a byte slice, advancing offset.
    pub fn from_bytes(
        data: &[u8],
        offset: &mut usize,
        input_size: usize,
        hidden_size: usize,
    ) -> Self {
        let concat_dim = input_size + hidden_size;

        let wz = read_f32_slice(data, offset, hidden_size * concat_dim);
        let bz = read_f32_slice(data, offset, hidden_size);
        let wr = read_f32_slice(data, offset, hidden_size * concat_dim);
        let br = read_f32_slice(data, offset, hidden_size);
        let wn = read_f32_slice(data, offset, hidden_size * concat_dim);
        let bn = read_f32_slice(data, offset, hidden_size);
        let wo = read_f32_slice(data, offset, hidden_size * hidden_size);
        let bo = read_f32_slice(data, offset, hidden_size);

        Self {
            wz,
            bz,
            wr,
            br,
            wn,
            bn,
            wo,
            bo,
            input_size,
            hidden_size,
        }
    }

    /// GRU forward pass: compute h_out from input x and previous hidden h_prev.
    ///
    /// Writes output into `h_out` (caller ensures len == hidden_size).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        &self,
        x: &[f32],
        h_prev: &[f32],
        h_out: &mut [f32],
        concat_buf: &mut [f32],
        gate_buf: &mut [f32],
    ) {
        let hs = self.hidden_size;
        let concat_dim = self.input_size + hs;

        // Concatenate [x, h_prev]
        concat_buf[..self.input_size].copy_from_slice(&x[..self.input_size]);
        concat_buf[self.input_size..concat_dim].copy_from_slice(&h_prev[..hs]);

        // Reset gate: z = σ(Wz·[x,h] + bz)
        matmul(&mut gate_buf[..hs], &self.wz, concat_buf, hs, concat_dim);
        for i in 0..hs {
            gate_buf[i] = sigmoid(gate_buf[i] + self.bz[i]);
        }
        let z = &gate_buf[..hs];

        // Update gate: r = σ(Wr·[x,h] + br)
        matmul(&mut h_out[..hs], &self.wr, concat_buf, hs, concat_dim);
        for i in 0..hs {
            h_out[i] = sigmoid(h_out[i] + self.br[i]);
        }

        // Element-wise multiply r ⊙ h_prev, store in concat_buf tail
        for i in 0..hs {
            concat_buf[self.input_size + i] = h_out[i] * h_prev[i];
        }

        // New gate: n = tanh(Wn·[x, r⊙h] + bn)
        matmul(&mut h_out[..hs], &self.wn, concat_buf, hs, concat_dim);
        for i in 0..hs {
            h_out[i] = (h_out[i] + self.bn[i]).tanh();
        }

        // Output: h' = (1-z)⊙h_prev + z⊙n
        for i in 0..hs {
            h_out[i] = (1.0 - z[i]) * h_prev[i] + z[i] * h_out[i];
        }
    }
}

// ── DominoLoraCorrection ──────────────────────────────────────

/// Domino LoRA correction adapter for speculative decoding.
///
/// Reads the DMAD format from riir-engine training and applies a LoRA-style
/// correction ΔL to draft model logits during speculative decoding.
pub struct DominoLoraCorrection {
    /// Down-projection weight [rank, concat_dim] where concat_dim = n_embd + gru_hidden
    pub w_down: Vec<f32>,
    /// Up-projection weight [vocab_size, rank]
    pub w_up: Vec<f32>,
    /// GRU for causal state tracking
    pub gru: DominoGRU,
    /// Adapter rank
    pub adapter_rank: usize,
    /// GRU hidden dimension
    pub gru_hidden: usize,
    /// Model embedding dimension
    pub n_embd: usize,
    /// Vocabulary size
    pub vocab_size: usize,

    // Pre-allocated scratch buffers
    concat_buf: Vec<f32>, // [n_embd + gru_hidden]
    down_buf: Vec<f32>,   // [adapter_rank]
    gru_concat: Vec<f32>, // [embed_dim + gru_hidden] for GRU forward
    gru_gate: Vec<f32>,   // [gru_hidden] for GRU gate computation
}

impl std::fmt::Debug for DominoLoraCorrection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DominoLoraCorrection")
            .field("adapter_rank", &self.adapter_rank)
            .field("gru_hidden", &self.gru_hidden)
            .field("n_embd", &self.n_embd)
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl DominoLoraCorrection {
    /// Create a new Domino LoRA correction adapter with the given dimensions.
    ///
    /// Weights are initialized with small random values for testing/benchmarking.
    /// For production, use `load()` to load trained weights from a DMAD binary file.
    pub fn new_for_test(
        n_embd: usize,
        vocab_size: usize,
        adapter_rank: usize,
        gru_hidden: usize,
        embed_dim: usize,
    ) -> Self {
        let concat_dim = n_embd + gru_hidden;
        Self {
            w_down: vec![0.1; adapter_rank * concat_dim],
            w_up: vec![0.1; vocab_size * adapter_rank],
            gru: DominoGRU {
                wz: vec![0.05; gru_hidden * (embed_dim + gru_hidden)],
                bz: vec![0.0; gru_hidden],
                wr: vec![0.05; gru_hidden * (embed_dim + gru_hidden)],
                br: vec![0.0; gru_hidden],
                wn: vec![0.05; gru_hidden * (embed_dim + gru_hidden)],
                bn: vec![0.0; gru_hidden],
                wo: vec![0.05; gru_hidden * gru_hidden],
                bo: vec![0.0; gru_hidden],
                input_size: embed_dim,
                hidden_size: gru_hidden,
            },
            adapter_rank,
            gru_hidden,
            n_embd,
            vocab_size,
            concat_buf: vec![0.0; concat_dim],
            down_buf: vec![0.0; adapter_rank],
            gru_concat: vec![0.0; embed_dim + gru_hidden],
            gru_gate: vec![0.0; gru_hidden],
        }
    }

    /// Load a Domino LoRA adapter from a DMAD binary file.
    ///
    /// The file format is:
    /// - Header (32 bytes): magic(4) + version(4) + n_embd(4) + vocab_size(4) +
    ///   rank(4) + gru_hidden(4) + embed_dim(4)
    /// - GRU weights (6 weights + 6 biases)
    /// - w_down [rank × concat_dim]
    /// - w_up [vocab_size × rank]
    /// - blake3 checksum (32 bytes)
    pub fn load(
        path: &Path,
        n_embd: usize,
        vocab_size: usize,
        adapter_rank: usize,
        gru_hidden: usize,
        embed_dim: usize,
    ) -> Result<Self, String> {
        let file_data =
            std::fs::read(path).map_err(|e| format!("Failed to read domino lora file: {e}"))?;

        // Minimum: header(32) + hash(32) = 64
        if file_data.len() < 64 {
            return Err("File too small for domino lora header".into());
        }

        // Verify magic
        if &file_data[0..4] != DOMINO_MAGIC {
            return Err("Invalid domino lora magic bytes".into());
        }

        // Verify version
        let version = u32::from_le_bytes(
            file_data[4..8]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("Version parse: {e}"))?,
        );
        if version != DOMINO_VERSION {
            return Err(format!("Unsupported domino lora version: {version}"));
        }

        // Verify blake3 checksum (last 32 bytes cover everything before them)
        let data_len = file_data.len() - 32;
        let stored_checksum = &file_data[data_len..];
        let computed = blake3::hash(&file_data[..data_len]);
        if computed.as_bytes() != stored_checksum {
            return Err("Domino lora file checksum mismatch".into());
        }

        // Parse header dimensions
        let file_n_embd = u32::from_le_bytes(
            file_data[8..12]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("n_embd parse: {e}"))?,
        ) as usize;
        let file_vocab_size = u32::from_le_bytes(
            file_data[12..16]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("vocab_size parse: {e}"))?,
        ) as usize;
        let file_rank = u32::from_le_bytes(
            file_data[16..20]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("rank parse: {e}"))?,
        ) as usize;
        let file_gru_hidden = u32::from_le_bytes(
            file_data[20..24]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("gru_hidden parse: {e}"))?,
        ) as usize;
        let _file_embed_dim = u32::from_le_bytes(
            file_data[24..28]
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| format!("embed_dim parse: {e}"))?,
        ) as usize;

        // Validate dimensions match caller expectations
        if file_n_embd != n_embd {
            return Err(format!(
                "n_embd mismatch: file={file_n_embd}, expected={n_embd}"
            ));
        }
        if file_vocab_size != vocab_size {
            return Err(format!(
                "vocab_size mismatch: file={file_vocab_size}, expected={vocab_size}"
            ));
        }
        if file_rank != adapter_rank {
            return Err(format!(
                "rank mismatch: file={file_rank}, expected={adapter_rank}"
            ));
        }
        if file_gru_hidden != gru_hidden {
            return Err(format!(
                "gru_hidden mismatch: file={file_gru_hidden}, expected={gru_hidden}"
            ));
        }

        // Parse weights after header
        let mut offset = 32;

        // GRU weights
        let gru = DominoGRU::from_bytes(&file_data, &mut offset, embed_dim, gru_hidden);

        // w_down [rank, concat_dim]
        let concat_dim = n_embd + gru_hidden;
        let w_down = read_f32_slice(&file_data, &mut offset, adapter_rank * concat_dim);

        // w_up [vocab_size, rank]
        let w_up = read_f32_slice(&file_data, &mut offset, vocab_size * adapter_rank);

        // Pre-allocate scratch buffers
        let concat_buf = vec![0.0f32; concat_dim];
        let down_buf = vec![0.0f32; adapter_rank];
        let gru_concat = vec![0.0f32; embed_dim + gru_hidden];
        let gru_gate = vec![0.0f32; gru_hidden];

        Ok(Self {
            w_down,
            w_up,
            gru,
            adapter_rank,
            gru_hidden,
            n_embd,
            vocab_size,
            concat_buf,
            down_buf,
            gru_concat,
            gru_gate,
        })
    }

    /// Apply the Domino LoRA correction to logits.
    ///
    /// Forward pass: ΔL = w_up @ (w_down @ concat(hidden, causal_state))
    /// Then adds ΔL element-wise to logits_out.
    pub fn correct(&mut self, hidden: &[f32], causal_state: &[f32], logits_out: &mut [f32]) {
        let concat_dim = self.n_embd + self.gru_hidden;

        // Concatenate [hidden, causal_state]
        self.concat_buf[..self.n_embd].copy_from_slice(&hidden[..self.n_embd]);
        self.concat_buf[self.n_embd..concat_dim].copy_from_slice(&causal_state[..self.gru_hidden]);

        // Down-projection: down_buf = w_down @ concat_buf  [rank]
        matmul(
            &mut self.down_buf,
            &self.w_down,
            &self.concat_buf,
            self.adapter_rank,
            concat_dim,
        );

        // ReLU activation on down-projection (standard LoRA pattern).
        // Branchless `max(0.0)` is auto-vectorizable; the previous `if *v < 0.0`
        // form generated a branch per element.
        for v in self.down_buf.iter_mut() {
            *v = v.max(0.0);
        }

        // Temporary delta logits buffer — compute into logits_out delta then add
        // We need a temp buffer for the up-projection output
        // Reuse concat_buf since we no longer need it
        let delta_buf = &mut self.concat_buf;
        let delta = &mut delta_buf[..self.vocab_size];

        // Up-projection: delta = w_up @ down_buf  [vocab_size]
        matmul(
            delta,
            &self.w_up,
            &self.down_buf,
            self.vocab_size,
            self.adapter_rank,
        );

        // Add ΔL to base logits
        simd_add_inplace(&mut logits_out[..self.vocab_size], &delta[..self.vocab_size]);
    }

    /// Run a GRU forward step for causal state tracking.
    ///
    /// Takes token embedding and previous hidden state, produces new hidden state.
    pub fn gru_step(&mut self, token_embed: &[f32], h_prev: &[f32], h_out: &mut [f32]) {
        self.gru.forward_into(
            token_embed,
            h_prev,
            h_out,
            &mut self.gru_concat,
            &mut self.gru_gate,
        );
    }

    /// Returns the GRU hidden dimension.
    pub fn gru_hidden_size(&self) -> usize {
        self.gru_hidden
    }

    /// Returns the adapter rank.
    pub fn adapter_rank(&self) -> usize {
        self.adapter_rank
    }
}

// ── Helpers ──────────────────────────────────────────────────

/// Read `count` f32 values from `data` starting at `offset`, advancing offset.
///
/// Uses `bytemuck::pod_collect_to_vec` which handles unaligned source data
/// correctly (the underlying file slice may not be 4-byte aligned). The
/// chunks_exact+from_le_bytes path was per-element work that the compiler
/// could only partially auto-vectorize. This path is cold (load-time only) but
/// also used by the test harness, so a noticeable speed-up on large adapters
/// is welcome. Relies on the host being little-endian (true for x86_64, aarch64).
fn read_f32_slice(data: &[u8], offset: &mut usize, count: usize) -> Vec<f32> {
    let byte_count = count * 4;
    let slice = &data[*offset..*offset + byte_count];
    *offset += byte_count;
    bytemuck::pod_collect_to_vec(slice)
}

/// Standard sigmoid function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_read_f32_slice() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut offset = 0;
        let vals = read_f32_slice(&data, &mut offset, 4);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(offset, 16);
    }

    #[test]
    fn test_domino_lora_load_invalid_magic() {
        let temp = std::env::temp_dir().join("test_domino_bad_magic.bin");
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&temp, &data).unwrap();
        let result = DominoLoraCorrection::load(&temp, 4, 8, 2, 4, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("magic"));
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_domino_lora_load_bad_checksum() {
        let temp = std::env::temp_dir().join("test_domino_bad_checksum.bin");
        // Write valid header but wrong checksum
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"DMAD");
        data[4..8].copy_from_slice(&1u32.to_le_bytes());
        // Fill rest with garbage + fake hash at end
        let hash = blake3::hash(b"not the real data");
        data[96..128].copy_from_slice(hash.as_bytes());
        std::fs::write(&temp, &data).unwrap();
        let result = DominoLoraCorrection::load(&temp, 4, 8, 2, 4, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("checksum"));
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn test_gru_forward_shape() {
        let input_size = 4;
        let hidden_size = 4;
        let gru = DominoGRU {
            wz: vec![0.1; hidden_size * (input_size + hidden_size)],
            bz: vec![0.0; hidden_size],
            wr: vec![0.1; hidden_size * (input_size + hidden_size)],
            br: vec![0.0; hidden_size],
            wn: vec![0.1; hidden_size * (input_size + hidden_size)],
            bn: vec![0.0; hidden_size],
            wo: vec![0.1; hidden_size * hidden_size],
            bo: vec![0.0; hidden_size],
            input_size,
            hidden_size,
        };

        let x = vec![1.0f32; input_size];
        let h_prev = vec![0.0f32; hidden_size];
        let mut h_out = vec![0.0f32; hidden_size];
        let mut concat_buf = vec![0.0f32; input_size + hidden_size];
        let mut gate_buf = vec![0.0f32; hidden_size];

        gru.forward_into(&x, &h_prev, &mut h_out, &mut concat_buf, &mut gate_buf);

        // Output should be non-trivial (not all zeros)
        assert!(h_out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_correct_adds_delta() {
        let n_embd = 4;
        let vocab_size = 8;
        let rank = 2;
        let gru_hidden = 4;
        let embed_dim = 4;

        let mut domino = DominoLoraCorrection {
            w_down: vec![0.1; rank * (n_embd + gru_hidden)],
            w_up: vec![0.1; vocab_size * rank],
            gru: DominoGRU {
                wz: vec![0.0; gru_hidden * (embed_dim + gru_hidden)],
                bz: vec![0.0; gru_hidden],
                wr: vec![0.0; gru_hidden * (embed_dim + gru_hidden)],
                br: vec![0.0; gru_hidden],
                wn: vec![0.0; gru_hidden * (embed_dim + gru_hidden)],
                bn: vec![0.0; gru_hidden],
                wo: vec![0.0; gru_hidden * gru_hidden],
                bo: vec![0.0; gru_hidden],
                input_size: embed_dim,
                hidden_size: gru_hidden,
            },
            adapter_rank: rank,
            gru_hidden,
            n_embd,
            vocab_size,
            concat_buf: vec![0.0; n_embd + gru_hidden],
            down_buf: vec![0.0; rank],
            gru_concat: vec![0.0; embed_dim + gru_hidden],
            gru_gate: vec![0.0; gru_hidden],
        };

        let hidden = vec![1.0f32; n_embd];
        let causal_state = vec![0.0f32; gru_hidden];
        let mut logits = vec![0.0f32; vocab_size];

        domino.correct(&hidden, &causal_state, &mut logits);

        // Logits should be modified (non-zero delta added)
        assert!(logits.iter().any(|&v| v != 0.0));
    }
}
