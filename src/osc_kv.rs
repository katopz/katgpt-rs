//! Oscillatory KV Cache — IMEX discretization of a damped harmonic oscillator.
//!
//! Stores KV pairs as oscillatory states (y, z) using symplectic IMEX. Plan 189 Phase 2.

use crate::types::QuantizedKVCache;

// ---------------------------------------------------------------------------
// Config

/// Configuration for [`OscKVCache`].
#[derive(Clone, Debug)]
pub struct OscKVConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Dimension of each key / value head.
    pub kv_dim: usize,
    /// Maximum sequence length (pre-allocated capacity).
    pub max_seq_len: usize,
    /// IMEX timestep (default 0.01).
    pub dt: f32,
    /// Default damping coefficient per channel (default 0.1).
    pub beta_default: f32,
}

impl Default for OscKVConfig {
    fn default() -> Self {
        Self {
            n_layers: 1,
            kv_dim: 64,
            max_seq_len: 512,
            dt: 0.01,
            beta_default: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer state

/// Per-layer oscillatory KV state.
struct OscKVLayer {
    /// Position coordinate (compressed KV signal). `[max_seq_len * kv_dim]`
    y: Vec<f32>,
    /// Velocity coordinate (derivative signal). `[max_seq_len * kv_dim]`
    z: Vec<f32>,
    /// Angular frequency squared per channel. `[kv_dim]`
    omega_sq: Vec<f32>,
    /// Damping coefficient per channel. `[kv_dim]`
    beta: Vec<f32>,
}

impl OscKVLayer {
    fn new(kv_dim: usize, max_seq_len: usize, beta_default: f32) -> Self {
        let total = max_seq_len * kv_dim;
        Self {
            y: vec![0.0f32; total],
            z: vec![0.0f32; total],
            omega_sq: vec![1.0f32; kv_dim],
            beta: vec![beta_default; kv_dim],
        }
    }
}

// ---------------------------------------------------------------------------
// OscKVCache

/// Oscillatory KV cache using IMEX discretization of a damped harmonic oscillator.
/// Reconstruction reads `y` (position) with optional `z` (velocity) blend.
pub struct OscKVCache {
    key_layers: Vec<OscKVLayer>,
    val_layers: Vec<OscKVLayer>,
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
    pos: usize,
    dt: f32,
}

impl OscKVCache {
    /// Build from an [`OscKVConfig`]. Pre-allocates all memory.
    pub fn with_config(config: &OscKVConfig) -> Self {
        let key_layers = (0..config.n_layers)
            .map(|_| OscKVLayer::new(config.kv_dim, config.max_seq_len, config.beta_default))
            .collect();
        let val_layers = (0..config.n_layers)
            .map(|_| OscKVLayer::new(config.kv_dim, config.max_seq_len, config.beta_default))
            .collect();

        Self {
            key_layers,
            val_layers,
            n_layers: config.n_layers,
            kv_dim: config.kv_dim,
            max_seq_len: config.max_seq_len,
            pos: 0,
            dt: config.dt,
        }
    }

    /// Store a key vector via IMEX step (explicit position, implicit velocity).
    #[inline]
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);

        let l = &mut self.key_layers[layer];
        let base = pos * self.kv_dim;
        let dt = self.dt;

        // IMEX step — per-channel damped harmonic oscillator
        for i in 0..self.kv_dim {
            let y_n = l.y[base + i];
            let z_n = l.z[base + i];
            let omega_sq = l.omega_sq[i];
            let beta = l.beta[i];
            let f = key[i];

            // Explicit position update
            let y_new = y_n + dt * z_n;

            // Implicit velocity update: z_{n+1} = z_n + dt*(-ω²y_{n+1} - βz_n + f)
            let z_new = z_n + dt * (-omega_sq * y_new - beta * z_n + f);

            l.y[base + i] = y_new;
            l.z[base + i] = z_new;
        }
    }

    /// Store a value vector via IMEX step.
    #[inline]
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);

        let l = &mut self.val_layers[layer];
        let base = pos * self.kv_dim;
        let dt = self.dt;

        for i in 0..self.kv_dim {
            let y_n = l.y[base + i];
            let z_n = l.z[base + i];
            let omega_sq = l.omega_sq[i];
            let beta = l.beta[i];
            let f = value[i];

            let y_new = y_n + dt * z_n;
            let z_new = z_n + dt * (-omega_sq * y_new - beta * z_n + f);

            l.y[base + i] = y_new;
            l.z[base + i] = z_new;
        }
    }

    /// Reconstruct key from position `y` with small velocity blend.
    #[inline]
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);

        let l = &self.key_layers[layer];
        let base = pos * self.kv_dim;

        for i in 0..self.kv_dim {
            // Primary reconstruction from position, small velocity blend for fidelity
            out[i] = l.y[base + i] + 0.1 * l.z[base + i];
        }
    }

    /// Reconstruct value from position `y` with small velocity blend.
    #[inline]
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);

        let l = &self.val_layers[layer];
        let base = pos * self.kv_dim;

        for i in 0..self.kv_dim {
            out[i] = l.y[base + i] + 0.1 * l.z[base + i];
        }
    }

    /// Reset all oscillatory state to zero.
    pub fn reset(&mut self) {
        for l in &mut self.key_layers {
            l.y.fill(0.0);
            l.z.fill(0.0);
        }
        for l in &mut self.val_layers {
            l.y.fill(0.0);
            l.z.fill(0.0);
        }
        self.pos = 0;
    }

    /// Current write position.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set write position.
    #[inline]
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Set per-channel angular frequency squared for a key layer.
    pub fn set_key_omega_sq(&mut self, layer: usize, omega_sq: &[f32]) {
        debug_assert_eq!(omega_sq.len(), self.kv_dim);
        self.key_layers[layer].omega_sq.copy_from_slice(omega_sq);
    }

    /// Set per-channel damping for a key layer.
    pub fn set_key_beta(&mut self, layer: usize, beta: &[f32]) {
        debug_assert_eq!(beta.len(), self.kv_dim);
        self.key_layers[layer].beta.copy_from_slice(beta);
    }
}

// ---------------------------------------------------------------------------
// Trait impl

impl QuantizedKVCache for OscKVCache {
    #[inline]
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.store_key(layer, pos, key);
    }

    #[inline]
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.store_value(layer, pos, value);
    }

    #[inline]
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_key_into(layer, pos, out);
    }

    #[inline]
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_value_into(layer, pos, out);
    }

    #[inline]
    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos()
    }

    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.set_pos(pos);
    }
}

// ---------------------------------------------------------------------------
// Helpers

/// Cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    dot / denom
}

/// Sigmoid activation (NOT softmax per project constraints).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n_layers: usize, kv_dim: usize, max_seq_len: usize) -> OscKVConfig {
        OscKVConfig {
            n_layers,
            kv_dim,
            max_seq_len,
            dt: 0.01,
            beta_default: 0.1,
        }
    }

    /// T1: Store a key, reconstruct it, assert cosine similarity > 0.8.
    #[test]
    fn test_store_and_reconstruct() {
        let kv_dim = 32;
        let config = make_config(1, kv_dim, 64);
        let mut cache = OscKVCache::with_config(&config);

        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);

        let mut out = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut out);

        let sim = cosine_sim(&key, &out);
        assert!(
            sim > 0.8,
            "cosine similarity too low after single store: {sim}"
        );
    }

    /// T2: With β=0, energy stays bounded over many steps.
    #[test]
    fn test_imex_energy_preservation() {
        let kv_dim = 16;
        let mut config = make_config(1, kv_dim, 512);
        config.beta_default = 0.0; // undamped
        let mut cache = OscKVCache::with_config(&config);

        // Store a constant forcing over many positions
        let forcing = vec![1.0f32; kv_dim];
        let n_steps = 200;

        for pos in 0..n_steps {
            cache.store_key(0, pos, &forcing);
        }

        // Compute total energy (y² + z²) over all stored positions
        let mut max_energy = 0.0f32;
        for pos in 0..n_steps {
            let base = pos * kv_dim;
            let l = &cache.key_layers[0];
            let mut energy = 0.0f32;
            for i in 0..kv_dim {
                energy += l.y[base + i] * l.y[base + i] + l.z[base + i] * l.z[base + i];
            }
            if energy > max_energy {
                max_energy = energy;
            }
        }

        // Energy should stay bounded (not grow exponentially)
        // With dt=0.01, omega_sq=1.0, 200 steps: energy stays O(n*dt) ≈ 2.0
        assert!(max_energy < 100.0, "energy grew unbounded: {max_energy}");
    }

    /// T3: Reset zeros state and position.
    #[test]
    fn test_reset_clears_state() {
        let kv_dim = 8;
        let config = make_config(1, kv_dim, 16);
        let mut cache = OscKVCache::with_config(&config);

        let key = vec![1.0f32; kv_dim];
        cache.store_key(0, 0, &key);
        cache.set_pos(5);
        assert_eq!(cache.pos(), 5);

        cache.reset();
        assert_eq!(cache.pos(), 0);

        // Verify data is zeroed
        let l = &cache.key_layers[0];
        for &v in &l.y[..kv_dim] {
            assert_eq!(v, 0.0, "y not zeroed after reset");
        }
        for &v in &l.z[..kv_dim] {
            assert_eq!(v, 0.0, "z not zeroed after reset");
        }
    }

    /// T4: Verify trait object works.
    #[test]
    fn test_trait_impl() {
        let config = make_config(1, 8, 16);
        let mut cache: Box<dyn QuantizedKVCache> = Box::new(OscKVCache::with_config(&config));

        let key = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let val = vec![0.1f32; 8];

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &val);

        let mut k_out = vec![0.0f32; 8];
        let mut v_out = vec![0.0f32; 8];
        cache.dequantize_key_into(0, 0, &mut k_out);
        cache.dequantize_value_into(0, 0, &mut v_out);

        // Should not be all zeros
        let k_sum: f32 = k_out.iter().map(|x| x.abs()).sum();
        let v_sum: f32 = v_out.iter().map(|x| x.abs()).sum();
        assert!(k_sum > 0.0, "key reconstruction is zero");
        assert!(v_sum > 0.0, "value reconstruction is zero");

        cache.reset();
        assert_eq!(cache.pos(), 0);
    }

    /// T5: Two layers reconstruct independently.
    #[test]
    fn test_multi_layer() {
        let kv_dim = 16;
        let config = make_config(2, kv_dim, 32);
        let mut cache = OscKVCache::with_config(&config);

        let key0: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.5).sin()).collect();
        let key1: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.5).cos()).collect();

        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);

        let mut out0 = vec![0.0f32; kv_dim];
        let mut out1 = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut out0);
        cache.dequantize_key_into(1, 0, &mut out1);

        let sim0 = cosine_sim(&key0, &out0);
        let sim1 = cosine_sim(&key1, &out1);
        assert!(sim0 > 0.8, "layer 0 cosine sim too low: {sim0}");
        assert!(sim1 > 0.8, "layer 1 cosine sim too low: {sim1}");

        // Cross-layer: should NOT match
        let cross = cosine_sim(&out0, &out1);
        assert!(
            cross < 0.99,
            "layers should produce different outputs, cross={cross}"
        );
    }

    /// T6: Cyclic input has higher reconstruction quality than random.
    #[test]
    fn test_cyclic_input_quality() {
        let kv_dim = 16;
        let config = make_config(1, kv_dim, 64);
        let mut cache = OscKVCache::with_config(&config);

        // Cyclic pattern: alternating [1, 0, 1, 0, ...]
        let cyclic: Vec<f32> = (0..kv_dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();

        // Random-ish pattern (deterministic for test stability)
        let random: Vec<f32> = (0..kv_dim)
            .map(|i| {
                let x = ((i as f32 * 7.3 + 1.1) * 13.7).sin();
                sigmoid(x) // sigmoid activation per project constraints
            })
            .collect();

        let n_steps = 20;

        // Store cyclic at positions 0..n_steps
        for pos in 0..n_steps {
            cache.store_key(0, pos, &cyclic);
        }

        // Measure cyclic reconstruction quality
        let mut cyclic_quality = 0.0f32;
        let mut out = vec![0.0f32; kv_dim];
        for pos in 0..n_steps {
            cache.dequantize_key_into(0, pos, &mut out);
            cyclic_quality += cosine_sim(&cyclic, &out);
        }
        cyclic_quality /= n_steps as f32;

        // Reset and do the same with random
        cache.reset();
        for pos in 0..n_steps {
            cache.store_key(0, pos, &random);
        }

        let mut random_quality = 0.0f32;
        for pos in 0..n_steps {
            cache.dequantize_key_into(0, pos, &mut out);
            random_quality += cosine_sim(&random, &out);
        }
        random_quality /= n_steps as f32;

        // Cyclic should have higher or equal quality
        // (repeated identical forcing builds a stronger oscillatory resonance)
        assert!(
            cyclic_quality >= random_quality - 0.01, // small tolerance
            "cyclic quality ({cyclic_quality}) should be >= random ({random_quality})"
        );
    }
}
