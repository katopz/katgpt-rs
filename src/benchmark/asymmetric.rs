#[allow(dead_code)]
pub struct AsymmetricBenchResult {
    /// Configuration tested.
    pub key_bits: u8,
    pub val_bits: u8,
    /// Cosine similarity between original and dequantized key vectors.
    pub cosine_sim_key: f32,
    /// Cosine similarity between original and dequantized value vectors.
    pub cosine_sim_value: f32,
    /// Compression ratio vs fp32.
    pub compression_ratio: f32,
    /// Label for this configuration.
    pub label: String,
}

#[cfg(feature = "asymmetric_kv")]
impl AsymmetricBenchResult {
    /// Harmonic mean of key and value cosine similarities.
    pub fn combined_fidelity(&self) -> f32 {
        if self.cosine_sim_key <= 0.0 || self.cosine_sim_value <= 0.0 {
            return 0.0;
        }
        2.0 * self.cosine_sim_key * self.cosine_sim_value
            / (self.cosine_sim_key + self.cosine_sim_value)
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Asymmetric KV cross-method benchmark (Plan 123, T6)
// ---------------------------------------------------------------------------

/// Simple uniform quantization helper for asymmetric benchmarking.
///
/// Maps `[min, max]` to `2^bits` uniform levels, then dequantizes back.
/// Intentionally method-agnostic — actual TurboQuant/SpectralQuant are behind
/// their own feature gates.
#[cfg(feature = "asymmetric_kv")]
fn quantize_dequantize(data: &[f32], bits: u8) -> Vec<f32> {
    if data.is_empty() || bits == 0 {
        return data.to_vec();
    }
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < 1e-10 {
        return data.to_vec();
    }
    let n_levels = (1u32 << bits) as f32;
    data.iter()
        .map(|&v| {
            let normalized = (v - min) / range;
            let quantized = (normalized * (n_levels - 1.0)).round() / (n_levels - 1.0);
            quantized * range + min
        })
        .collect()
}

/// Cross-method asymmetric benchmark comparing symmetric vs asymmetric KV configs.
///
/// Tests multiple quantization configs: (3,3) symmetric, (4,2) aggressive,
/// (8,2) aggressive asymmetric, (8,3) recommended, (2,8) inverted.
/// Returns one [`AsymmetricBenchResult`] per config.
///
/// Uses simple uniform quantization (not actual TurboQuant/SpectralQuant — those
/// are behind other feature gates). The asymmetry signal is method-independent
/// because it stems from softmax amplification, not the quantizer.
#[cfg(feature = "asymmetric_kv")]
pub fn bench_asymmetric_cross_method(
    head_dim: usize,
    n_kv_heads: usize,
    seq_len: usize,
) -> Vec<AsymmetricBenchResult> {
    use fastrand::Rng as FastrandRng;

    let mut rng = FastrandRng::with_seed(12345);

    /// Test configs: (key_bits, val_bits, label)
    const CONFIGS: &[(u8, u8, &str)] = &[
        (3, 3, "symmetric_3_3"),
        (4, 2, "aggressive_4_2"),
        (8, 2, "aggressive_8_2"),
        (8, 3, "recommended_8_3"),
        (2, 8, "inverted_2_8"),
    ];

    let mut results = Vec::with_capacity(CONFIGS.len());

    for &(key_bits, val_bits, label) in CONFIGS {
        let mut cos_k_sum = 0.0f32;
        let mut cos_v_sum = 0.0f32;
        let samples = n_kv_heads * seq_len;

        // Pre-allocate key/value buffers outside the loop
        let mut key = vec![0.0f32; head_dim];
        let mut value = vec![0.0f32; head_dim];

        for _ in 0..samples {
            // Fill random K and V vectors in-place
            for k in key.iter_mut() {
                *k = rng.f32() * 2.0 - 1.0;
            }
            for v in value.iter_mut() {
                *v = rng.f32() * 2.0 - 1.0;
            }

            // Quantize + dequantize with respective bit widths
            let recon_k = quantize_dequantize(&key, key_bits);
            let recon_v = quantize_dequantize(&value, val_bits);

            // Measure reconstruction fidelity via cosine similarity
            cos_k_sum += cosine_similarity(&key, &recon_k);
            cos_v_sum += cosine_similarity(&value, &recon_v);
        }

        let avg_cos_k = cos_k_sum / samples as f32;
        let avg_cos_v = cos_v_sum / samples as f32;
        let avg_bits = (key_bits as f32 + val_bits as f32) / 2.0;
        let compression_ratio = 32.0 / avg_bits;

        results.push(AsymmetricBenchResult {
            key_bits,
            val_bits,
            cosine_sim_key: avg_cos_k,
            cosine_sim_value: avg_cos_v,
            compression_ratio,
            label: label.to_string(),
        });
    }

    results
}
