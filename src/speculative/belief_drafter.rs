//! NextLat Belief-State Speculative Drafter — lightweight MLP recursive hidden state prediction.
//!
//! Implements Plan 217 Phase 0: `LatentDynamicsMLP` struct with forward pass, binary I/O,
//! and random initialization. The MLP predicts `h_{t+1}` from `(h_t, emb(x_{t+1}))` using a
//! 3-layer residual architecture inspired by arXiv:2511.05963 (NextLat).
//!
//! Architecture: `h_{t+1} = h_t + FC3(GELU(FC2(GELU(FC1(LN(concat(h_t, next_emb)))))))`
//!
//! Feature-gated behind `belief_drafter` — off by default until GOAT proof.

#![cfg(feature = "belief_drafter")]

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::simd::simd_dot_f32;

// ── Magic & Version ────────────────────────────────────────────
const MAGIC: &[u8; 4] = b"NLDM";
const VERSION: u32 = 1;

// ── GELU Approximation ────────────────────────────────────────

/// Standard GELU approximation: `0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ── LayerNorm ──────────────────────────────────────────────────

/// Standard LayerNorm: normalize to zero mean / unit variance, then apply affine transform.
fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(weight.len(), n);
    assert_eq!(bias.len(), n);
    assert_eq!(output.len(), n);

    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    for i in 0..n {
        output[i] = weight[i] * (input[i] - mean) * inv_std + bias[i];
    }
}

// ── Linear (Row-Major Matmul + Bias) ──────────────────────────

/// Row-major matmul + bias: `output[i] = dot(weight[i*in_dim..(i+1)*in_dim], input) + bias[i]`
/// Uses `simd_dot_f32` for each row.
fn linear(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize, output: &mut [f32]) {
    let in_dim = input.len();
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(bias.len(), out_dim);
    assert_eq!(output.len(), out_dim);

    for i in 0..out_dim {
        let row_off = i * in_dim;
        let dot = simd_dot_f32(&weight[row_off..row_off + in_dim], input, in_dim);
        output[i] = dot + bias[i];
    }
}

// ── LatentDynamicsMLP ──────────────────────────────────────────

/// 3-layer residual MLP that predicts next hidden states from `(h_t, emb(x_{t+1}))`.
///
/// - Input: `LayerNorm(concat(h_t, next_emb))` — shape `[2 * n_embd]`
/// - FC1: `[2 * n_embd] → [n_embd]`, GELU
/// - FC2: `[n_embd] → [n_embd]`, GELU
/// - FC3: `[n_embd] → [n_embd]`
/// - Output: `h_{t+1} = h_t + FC3(...)` (residual connection)
///
/// For Config::micro (n_embd=16): ~1.5K params. For Config::bpe (n_embd=32): ~6K params.
#[derive(Debug)]
pub struct LatentDynamicsMLP {
    pub n_embd: usize,
    pub norm_weight: Vec<f32>, // [2 * n_embd]
    pub norm_bias: Vec<f32>,   // [2 * n_embd]
    pub fc1_weight: Vec<f32>,  // [n_embd, 2*n_embd] row-major
    pub fc1_bias: Vec<f32>,    // [n_embd]
    pub fc2_weight: Vec<f32>,  // [n_embd, n_embd] row-major
    pub fc2_bias: Vec<f32>,    // [n_embd]
    pub fc3_weight: Vec<f32>,  // [n_embd, n_embd] row-major
    pub fc3_bias: Vec<f32>,    // [n_embd]
}

impl LatentDynamicsMLP {
    /// Run the MLP forward pass: `h_{t+1} = h_t + FC3(GELU(FC2(GELU(FC1(LN(concat))))))`.
    ///
    /// - `h_t`: current hidden state `[n_embd]`
    /// - `next_emb`: embedding of next token `[n_embd]`
    /// - Returns: predicted next hidden state `[n_embd]`
    pub fn forward(&self, h_t: &[f32], next_emb: &[f32]) -> Vec<f32> {
        let n = self.n_embd;
        assert_eq!(h_t.len(), n, "h_t must have length n_embd");
        assert_eq!(next_emb.len(), n, "next_emb must have length n_embd");

        let concat_dim = 2 * n;

        // 1. Concatenate h_t and next_emb
        let mut concat = vec![0.0f32; concat_dim];
        concat[..n].copy_from_slice(h_t);
        concat[n..].copy_from_slice(next_emb);

        // 2. LayerNorm
        let mut normed = vec![0.0f32; concat_dim];
        layer_norm(&concat, &self.norm_weight, &self.norm_bias, &mut normed);

        // 3. FC1: [2*n_embd] → [n_embd] + GELU
        let mut fc1_out = vec![0.0f32; n];
        linear(&normed, &self.fc1_weight, &self.fc1_bias, n, &mut fc1_out);
        for v in &mut fc1_out {
            *v = gelu(*v);
        }

        // 4. FC2: [n_embd] → [n_embd] + GELU
        let mut fc2_out = vec![0.0f32; n];
        linear(&fc1_out, &self.fc2_weight, &self.fc2_bias, n, &mut fc2_out);
        for v in &mut fc2_out {
            *v = gelu(*v);
        }

        // 5. FC3: [n_embd] → [n_embd] (no activation)
        let mut fc3_out = vec![0.0f32; n];
        linear(&fc2_out, &self.fc3_weight, &self.fc3_bias, n, &mut fc3_out);

        // 6. Residual: h_{t+1} = h_t + FC3(...)
        let mut result = vec![0.0f32; n];
        for i in 0..n {
            result[i] = h_t[i] + fc3_out[i];
        }

        result
    }

    /// Load MLP weights from a binary file.
    ///
    /// Binary format:
    /// - 4 bytes: magic "NLDM"
    /// - u32: version (must be 1)
    /// - u32: n_embd
    /// - Raw f32 arrays in order: norm_weight, norm_bias, fc1_weight, fc1_bias,
    ///   fc2_weight, fc2_bias, fc3_weight, fc3_bias
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open error: {e}"))?;
        let mut rdr = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 4];
        rdr.read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        match &magic {
            MAGIC => {}
            other => return Err(format!("bad magic: expected {:?}, got {:?}", MAGIC, other)),
        }

        // Version
        let version = read_u32(&mut rdr)?;
        match version {
            VERSION => {}
            v => return Err(format!("unsupported version: {v} (expected {VERSION})")),
        }

        // n_embd
        let n_embd = read_u32(&mut rdr)? as usize;
        if n_embd == 0 {
            return Err("n_embd must be > 0".into());
        }

        let concat_dim = 2 * n_embd;
        let fc1_rows = n_embd * concat_dim;
        let fc2_rows = n_embd * n_embd;
        let fc3_rows = n_embd * n_embd;

        let norm_weight = read_f32_vec(&mut rdr, concat_dim, "norm_weight")?;
        let norm_bias = read_f32_vec(&mut rdr, concat_dim, "norm_bias")?;
        let fc1_weight = read_f32_vec(&mut rdr, fc1_rows, "fc1_weight")?;
        let fc1_bias = read_f32_vec(&mut rdr, n_embd, "fc1_bias")?;
        let fc2_weight = read_f32_vec(&mut rdr, fc2_rows, "fc2_weight")?;
        let fc2_bias = read_f32_vec(&mut rdr, n_embd, "fc2_bias")?;
        let fc3_weight = read_f32_vec(&mut rdr, fc3_rows, "fc3_weight")?;
        let fc3_bias = read_f32_vec(&mut rdr, n_embd, "fc3_bias")?;

        Ok(Self {
            n_embd,
            norm_weight,
            norm_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        })
    }

    /// Save MLP weights to a binary file (for roundtrip testing).
    pub fn save_to_bin(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("create error: {e}"))?;
        let mut wtr = BufWriter::new(file);

        wtr.write_all(MAGIC)
            .map_err(|e| format!("write magic: {e}"))?;
        write_u32(&mut wtr, VERSION)?;
        write_u32(&mut wtr, self.n_embd as u32)?;

        write_f32_slice(&mut wtr, &self.norm_weight)?;
        write_f32_slice(&mut wtr, &self.norm_bias)?;
        write_f32_slice(&mut wtr, &self.fc1_weight)?;
        write_f32_slice(&mut wtr, &self.fc1_bias)?;
        write_f32_slice(&mut wtr, &self.fc2_weight)?;
        write_f32_slice(&mut wtr, &self.fc2_bias)?;
        write_f32_slice(&mut wtr, &self.fc3_weight)?;
        write_f32_slice(&mut wtr, &self.fc3_bias)?;

        wtr.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }

    /// Initialize MLP with Xavier-like weights, zeros for biases, ones for norm_weight.
    ///
    /// Uses a simple seeded LCG RNG for reproducibility (no external rand dependency).
    pub fn random_init(n_embd: usize) -> Self {
        let concat_dim = 2 * n_embd;

        // Seeded LCG: x_{n+1} = (a * x_n + c) mod 2^32
        // Using Numerical Recipes constants
        let mut state: u32 = 42;

        let mut next_f32 = || -> f32 {
            state = state.wrapping_mul(1_106_351_524).wrapping_add(12_345);
            // Map to (-1, 1) uniformly
            let bits = state >> 1; // clear sign bit
            let f = (bits as f32) / (u32::MAX as f32 * 0.5) - 1.0;
            f
        };

        // Xavier init: scale = sqrt(2 / fan_in)
        let xavier_fc1 = (2.0 / concat_dim as f32).sqrt();
        let xavier_fc2 = (2.0 / n_embd as f32).sqrt();
        let xavier_fc3 = (2.0 / n_embd as f32).sqrt();

        // LayerNorm: ones for weight, zeros for bias
        let norm_weight = vec![1.0f32; concat_dim];
        let norm_bias = vec![0.0f32; concat_dim];

        // FC1: [n_embd, 2*n_embd]
        let fc1_weight: Vec<f32> = (0..n_embd * concat_dim)
            .map(|_| next_f32() * xavier_fc1)
            .collect();
        let fc1_bias = vec![0.0f32; n_embd];

        // FC2: [n_embd, n_embd]
        let fc2_weight: Vec<f32> = (0..n_embd * n_embd)
            .map(|_| next_f32() * xavier_fc2)
            .collect();
        let fc2_bias = vec![0.0f32; n_embd];

        // FC3: [n_embd, n_embd]
        let fc3_weight: Vec<f32> = (0..n_embd * n_embd)
            .map(|_| next_f32() * xavier_fc3)
            .collect();
        let fc3_bias = vec![0.0f32; n_embd];

        Self {
            n_embd,
            norm_weight,
            norm_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        }
    }
}

// ── Binary I/O Helpers ─────────────────────────────────────────

fn read_u32(rdr: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    rdr.read_exact(&mut buf)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u32(wtr: &mut impl Write, val: u32) -> Result<(), String> {
    wtr.write_all(&val.to_le_bytes())
        .map_err(|e| format!("write u32: {e}"))?;
    Ok(())
}

fn read_f32_vec(rdr: &mut impl Read, expected_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let byte_len = expected_len * 4;
    let mut buf = vec![0u8; byte_len];
    rdr.read_exact(&mut buf)
        .map_err(|e| format!("read {label}: {e}"))?;
    let vec: Vec<f32> = buf
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    match vec.len() == expected_len {
        true => Ok(vec),
        false => Err(format!(
            "{label}: expected {expected_len} elements, got {}",
            vec.len()
        )),
    }
}

fn write_f32_slice(wtr: &mut impl Write, data: &[f32]) -> Result<(), String> {
    for &v in data {
        wtr.write_all(&v.to_le_bytes())
            .map_err(|e| format!("write f32: {e}"))?;
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_mlp_forward_shape_micro() {
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let output = mlp.forward(&h_t, &next_emb);
        assert_eq!(output.len(), n_embd, "output must have length n_embd=16");
    }

    #[test]
    fn test_mlp_forward_shape_bpe() {
        let n_embd = 32;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.5f32; n_embd];
        let next_emb = vec![0.3f32; n_embd];
        let output = mlp.forward(&h_t, &next_emb);
        assert_eq!(output.len(), n_embd, "output must have length n_embd=32");
    }

    #[test]
    fn test_mlp_residual_connection() {
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t: Vec<f32> = (0..n_embd).map(|i| i as f32 * 0.1).collect();
        let next_emb = vec![1.0f32; n_embd];

        let output = mlp.forward(&h_t, &next_emb);

        // The output should NOT equal h_t exactly (FC3 output is nonzero for non-zero input)
        // and should NOT equal just the FC3 output (it's h_t + FC3, not just FC3)
        // Verify residual: output[i] != h_t[i] for at least some i (FC3 is nonzero)
        let any_different = output
            .iter()
            .zip(h_t.iter())
            .any(|(&o, &h)| (o - h).abs() > 1e-6);
        assert!(
            any_different,
            "residual connection must produce output != h_t"
        );
    }

    #[test]
    fn test_random_init_produces_valid_mlp() {
        for &n_embd in &[16usize, 32] {
            let mlp = LatentDynamicsMLP::random_init(n_embd);
            let h_t = vec![1.0f32; n_embd];
            let next_emb = vec![-0.5f32; n_embd];
            let output = mlp.forward(&h_t, &next_emb);

            for (i, &v) in output.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "output[{i}] is not finite (n_embd={n_embd}): {v}"
                );
            }
        }
    }

    #[test]
    fn test_load_from_bin_roundtrip() {
        let n_embd = 16;
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let h_t = vec![0.7f32; n_embd];
        let next_emb = vec![0.2f32; n_embd];
        let expected = mlp.forward(&h_t, &next_emb);

        // Write to temp file
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nextlat_test.bin");
        mlp.save_to_bin(&path).expect("save");

        // Load back
        let loaded = LatentDynamicsMLP::load_from_bin(&path).expect("load");

        // Verify dimensions match
        assert_eq!(loaded.n_embd, n_embd);

        // Forward pass must produce identical output
        let actual = loaded.forward(&h_t, &next_emb);
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "roundtrip mismatch at [{i}]: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn test_load_from_bin_bad_magic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad_magic.bin");
        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(b"XXXX").expect("write bad magic");
        file.write_all(&1u32.to_le_bytes()).expect("write version");
        file.write_all(&16u32.to_le_bytes()).expect("write n_embd");
        drop(file);

        let result = LatentDynamicsMLP::load_from_bin(&path);
        match result {
            Err(msg) if msg.contains("bad magic") => {}
            other => panic!("expected bad magic error, got: {other:?}"),
        }
    }

    #[test]
    fn test_gelu_sanity() {
        // GELU(0) ≈ 0
        assert!(gelu(0.0).abs() < 1e-6, "gelu(0) should be ~0");

        // GELU(large positive) > 0
        assert!(gelu(10.0) > 0.0, "gelu(10) should be positive");
        assert!((gelu(10.0) - 10.0).abs() < 0.1, "gelu(10) should be ~10");

        // GELU(negative) < 0 (for moderately negative values)
        assert!(gelu(-1.0) < 0.0, "gelu(-1) should be negative");

        // GELU is approximately identity for large positive
        assert!(gelu(5.0) > 4.9, "gelu(5) should be ~5");

        // GELU approaches 0 for large negative
        assert!(gelu(-10.0).abs() < 0.01, "gelu(-10) should be ~0");
    }
}
