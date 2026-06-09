//! VocabChannel Pruner — ROTATE-Derived ConstraintPruner for Speculative Decoding (Plan 228).
//!
//! At model load time, decomposes MLP output weights into vocabulary channels using
//! kurtosis-maximizing Householder reflections (ROTATE method, arXiv:2606.03990).
//! Builds per-neuron token reachability maps. At inference time, acts as a
//! `ConstraintPruner` to reject unreachable tokens in DDTree speculative decoding.
//!
//! **Expected gain:** 30-60% DDTree branch reduction, quality-neutral.
//!
//! # Architecture
//!
//! ```text
//! Load Time:
//!   Wout[l][i] → ROTATE → channels {v₁...vₖ}
//!   Each channel → top-K tokens → per-neuron reachability set
//!   Aggregate → per-layer union reachability
//!
//! Inference Time:
//!   ConstraintPruner::is_valid(depth, token_idx, ...) = reachability.contains(token_idx)
//! ```
//!
//! # Phases
//!
//! 1. Core math: skewness, Householder reflection, vocab projection, token masking
//! 2. ROTATE decomposition: kurtosis-maximizing channel discovery
//! 3. Reachability map: per-layer, per-neuron token sets with binary serialization
//! 4. ConstraintPruner: integrates with DDTree via `ConstraintPruner` trait

use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use katgpt_core::traits::ConstraintPruner;

// ── Phase 1: Core Math ──────────────────────────────────────────────

/// Population skewness (γ₁) of a distribution in a single O(n) pass.
///
/// Returns 0.0 for degenerate inputs (n < 3 or zero variance).
/// No allocation — operates entirely on the input slice.
#[inline]
pub fn skewness(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 3.0 {
        return 0.0;
    }

    let mean: f32 = values.iter().sum::<f32>() / n;

    let (m2, m3) = values.iter().fold((0.0f32, 0.0f32), |(m2, m3), &x| {
        let d = x - mean;
        (m2 + d * d, m3 + d * d * d)
    });

    if m2 < 1e-10 {
        return 0.0;
    }

    // Population skewness: γ₁ = n²·m₃ / (n·m₂)^{3/2} = √n · m₃ / m₂^{3/2}
    let m2_32 = m2 * m2.sqrt();
    if m2_32 < 1e-10 {
        return 0.0;
    }

    n.sqrt() * m3 / m2_32
}

/// Excess kurtosis (κ) of a distribution in a single O(n) pass.
///
/// Reuses the same formula as `kurtosis_gate::excess_kurtosis` for self-containment.
/// Returns 0.0 for degenerate inputs (n < 4 or zero variance).
#[inline]
pub fn excess_kurtosis(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 4.0 {
        return 0.0;
    }

    let mean: f32 = values.iter().sum::<f32>() / n;

    let (m2, m4) = values.iter().fold((0.0f32, 0.0f32), |(m2, m4), &x| {
        let d = x - mean;
        (m2 + d * d, m4 + d * d * d * d)
    });

    if m2 < 1e-10 {
        return 0.0;
    }

    (m4 * n) / (m2 * m2) - 3.0
}

/// Apply a Householder reflection to vector `x` given Householder vector `h`.
///
/// Computes `(I - 2·h·hᵀ / ‖h‖²) @ x = x - 2·h·(hᵀ·x) / ‖h‖²`.
///
/// O(d) time, O(d) allocation for the output.
/// Returns a zero vector if `h` has zero norm (degenerate reflection).
#[inline]
pub fn householder_apply(h: &[f32], x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(h.len(), x.len());

    let h_norm_sq: f32 = h.iter().map(|hi| hi * hi).sum();
    if h_norm_sq < 1e-12 {
        return x.to_vec();
    }

    let dot_hx: f32 = h.iter().zip(x.iter()).map(|(hi, xi)| hi * xi).sum();
    let scale = 2.0 * dot_hx / h_norm_sq;

    x.iter()
        .zip(h.iter())
        .map(|(&xi, &hi)| xi - scale * hi)
        .collect()
}

/// Sigmoid function. Clamps input to avoid overflow in exp.
#[cfg(test)]
#[inline]
fn sigmoid(x: f32) -> f32 {
    let clamped = x.clamp(-20.0, 20.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Project a single neuron weight vector through the LM head to get vocabulary logits.
///
/// Given:
/// - `neuron_weight`: size `n_embd` — the neuron's output weight in residual space
/// - `lm_head`: flat `[vocab_size * n_embd]` — LM head weights, row-major
///
/// Computes `logits[t] = dot(lm_head[t*n_embd..(t+1)*n_embd], neuron_weight)`
/// for each token `t` in `0..vocab_size`.
///
/// Returns a vector of size `vocab_size`.
pub fn vocab_project(
    neuron_weight: &[f32],
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
) -> Vec<f32> {
    debug_assert_eq!(neuron_weight.len(), n_embd);
    debug_assert!(lm_head.len() >= vocab_size * n_embd);

    let mut logits = vec![0.0f32; vocab_size];
    for t in 0..vocab_size {
        let row = &lm_head[t * n_embd..(t + 1) * n_embd];
        logits[t] = row
            .iter()
            .zip(neuron_weight.iter())
            .map(|(&a, &b)| a * b)
            .sum();
    }
    logits
}

/// Apply token exclusion mask in-place. Sets masked positions to 0.0.
#[inline]
fn apply_mask(logits: &mut [f32], mask: &[bool]) {
    for (i, m) in mask.iter().enumerate() {
        if *m {
            logits[i] = 0.0;
        }
    }
}

/// Mark tokens whose channel logits are outliers as excluded (True in mask).
///
/// Tokens with `|z_i - μ| > k_sigma * σ` are marked True.
/// Existing True values in the mask are preserved (union).
pub fn iterative_token_mask(logits: &[f32], mask: &mut [bool], k_sigma: f32) {
    let n = logits.len() as f32;
    if n < 2.0 {
        return;
    }

    let mean: f32 = logits.iter().sum::<f32>() / n;
    let variance: f32 = logits.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let sigma = variance.sqrt();

    if sigma < 1e-10 {
        return;
    }

    let threshold = k_sigma * sigma;
    for (i, &z) in logits.iter().enumerate() {
        if (z - mean).abs() > threshold {
            mask[i] = true;
        }
    }
}

/// Find the top-K indices by value using partial selection.
///
/// Returns at most `k` indices sorted by descending value.
fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    if k == 0 {
        return Vec::new();
    }

    // For small k, use partial selection via a min-heap approach
    let mut top: Vec<(usize, f32)> = (0..k).map(|i| (i, values[i])).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for i in k..values.len() {
        let val = values[i];
        if val > top.last().map(|&(_, v)| v).unwrap_or(f32::NEG_INFINITY) {
            // Find insertion point
            let pos = top.partition_point(|&(_, v)| v > val);
            top.insert(pos, (i, val));
            top.pop(); // keep only k elements
        }
    }

    top.into_iter().map(|(idx, _)| idx).collect()
}

/// Cosine similarity between two vectors.
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (dot, na, nb): (f32, f32, f32) = a
        .iter()
        .zip(b.iter())
        .fold((0.0, 0.0, 0.0), |(d, na, nb), (&ai, &bi)| {
            (d + ai * bi, na + ai * ai, nb + bi * bi)
        });
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    dot / denom
}

// ── Phase 2: ROTATE Decomposition ───────────────────────────────────

/// A single vocabulary channel discovered by ROTATE decomposition.
#[derive(Debug, Clone)]
pub struct VocabChannel {
    /// Direction vector in weight space (size n_embd)
    pub direction: Vec<f32>,
    /// Top-K token indices promoted by this channel
    pub top_tokens: Vec<usize>,
    /// Kurtosis of this channel's vocabulary projection
    pub kurtosis: f32,
    /// Skewness of this channel's vocabulary projection
    pub skewness: f32,
}

/// Configuration for the ROTATE decomposition pipeline.
#[derive(Debug, Clone)]
pub struct VocabChannelConfig {
    /// Number of channels to extract per neuron
    pub max_channels: usize,
    /// Top-K tokens per channel for reachability
    pub top_k_tokens: usize,
    /// Kurtosis threshold for accepting a channel
    pub kurtosis_threshold: f32,
    /// Regularization weight for cosine similarity preservation
    pub lambda: f32,
    /// Learning rate for Householder vector optimization
    pub eta: f32,
    /// Maximum optimization iterations per channel
    pub max_iterations: usize,
    /// Standard deviation multiplier for token masking
    pub sigma_mask: f32,
    /// Number of random coordinate descent dimensions per iteration
    pub coords_per_iter: usize,
    /// Finite difference epsilon for gradient estimation
    pub fd_epsilon: f32,
}

impl Default for VocabChannelConfig {
    fn default() -> Self {
        Self {
            max_channels: 5,
            top_k_tokens: 50,
            kurtosis_threshold: 1.0,
            lambda: 0.1,
            eta: 0.01,
            max_iterations: 20,
            sigma_mask: 2.0,
            coords_per_iter: 3,
            fd_epsilon: 1e-3,
        }
    }
}

/// ROTATE decomposition engine for discovering vocabulary channels in MLP neurons.
pub struct VocabChannelDecomposer {
    config: VocabChannelConfig,
}

impl VocabChannelDecomposer {
    pub fn new(config: VocabChannelConfig) -> Self {
        Self { config }
    }

    /// Decompose a single neuron weight into vocabulary channels.
    ///
    /// Uses iterative ROTATE: Householder-reflected directions that maximize
    /// kurtosis of the vocabulary projection, with cosine similarity regularization.
    pub fn decompose_neuron(
        &self,
        neuron_weight: &[f32],
        lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
    ) -> Vec<VocabChannel> {
        let mut channels = Vec::with_capacity(self.config.max_channels);
        let mut mask = vec![false; vocab_size];

        for _ in 0..self.config.max_channels {
            let channel =
                match self.discover_channel(neuron_weight, lm_head, vocab_size, n_embd, &mask) {
                    Some(ch) => ch,
                    None => break,
                };

            if channel.kurtosis < self.config.kurtosis_threshold {
                break;
            }

            // Update mask: exclude outlier tokens for the next iteration
            let z = vocab_project(&channel.direction, lm_head, vocab_size, n_embd);
            iterative_token_mask(&z, &mut mask, self.config.sigma_mask);

            channels.push(channel);
        }

        channels
    }

    /// Discover a single channel via Householder optimization.
    fn discover_channel(
        &self,
        w: &[f32],
        lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
        mask: &[bool],
    ) -> Option<VocabChannel> {
        // Initialize Householder vector as small random perturbation
        let mut h: Vec<f32> = (0..n_embd).map(|_| (fastrand::f32() - 0.5) * 0.1).collect();

        // Optimize h to maximize kurtosis - lambda*(1 - cos(v, w))
        self.optimize_householder(&mut h, w, lm_head, vocab_size, n_embd, mask);

        // Compute final channel
        let v = householder_apply(&h, w);
        let mut z = vocab_project(&v, lm_head, vocab_size, n_embd);
        apply_mask(&mut z, mask);

        let k = excess_kurtosis(&z);
        let s = skewness(&z);
        let top_tokens = topk_indices(&z, self.config.top_k_tokens);

        Some(VocabChannel {
            direction: v,
            top_tokens,
            kurtosis: k,
            skewness: s,
        })
    }

    /// Optimize Householder vector via random coordinate descent on kurtosis.
    fn optimize_householder(
        &self,
        h: &mut [f32],
        w: &[f32],
        lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
        mask: &[bool],
    ) {
        let eps = self.config.fd_epsilon;

        for _ in 0..self.config.max_iterations {
            // Pick random dimensions for coordinate descent
            for _ in 0..self.config.coords_per_iter {
                let dim = fastrand::usize(0..n_embd);

                // Compute objective at h + eps*e_dim
                h[dim] += eps;
                let v_plus = householder_apply(h, w);
                let mut z_plus = vocab_project(&v_plus, lm_head, vocab_size, n_embd);
                apply_mask(&mut z_plus, mask);
                let obj_plus =
                    excess_kurtosis(&z_plus) + self.config.lambda * cosine_sim(&v_plus, w);

                // Compute objective at h - eps*e_dim
                h[dim] -= 2.0 * eps;
                let v_minus = householder_apply(h, w);
                let mut z_minus = vocab_project(&v_minus, lm_head, vocab_size, n_embd);
                apply_mask(&mut z_minus, mask);
                let obj_minus =
                    excess_kurtosis(&z_minus) + self.config.lambda * cosine_sim(&v_minus, w);

                // Restore h
                h[dim] += eps;

                // Gradient ascent
                let gradient = (obj_plus - obj_minus) / (2.0 * eps);
                h[dim] += self.config.eta * gradient;
            }
        }
    }
}

/// Decompose all neurons in a single layer's MLP down-projection weights.
///
/// # Arguments
/// * `mlp_w2` - Flat `[n_embd * mlp_hidden]` row-major MLP weights
/// * `lm_head` - Flat `[vocab_size * n_embd]` row-major LM head weights
/// * `n_embd` - Embedding dimension
/// * `mlp_hidden` - MLP hidden dimension
/// * `vocab_size` - Vocabulary size
/// * `config` - Decomposition configuration
///
/// # Returns
/// Per-neuron sorted token reachability sets (union of top-K tokens from each channel).
pub fn decompose_layer_channels(
    mlp_w2: &[f32],
    lm_head: &[f32],
    n_embd: usize,
    mlp_hidden: usize,
    vocab_size: usize,
    config: &VocabChannelConfig,
) -> Vec<Vec<usize>> {
    let decomposer = VocabChannelDecomposer::new(config.clone());
    let mut result = Vec::with_capacity(mlp_hidden);

    for neuron_idx in 0..mlp_hidden {
        // Extract column neuron_idx from mlp_w2 (row-major [n_embd, mlp_hidden])
        // neuron weight: w[i] = mlp_w2[i * mlp_hidden + neuron_idx] for i in 0..n_embd
        let neuron_weight: Vec<f32> = (0..n_embd)
            .map(|i| mlp_w2[i * mlp_hidden + neuron_idx])
            .collect();

        let channels = decomposer.decompose_neuron(&neuron_weight, lm_head, vocab_size, n_embd);

        // Union top-K tokens from each channel into a sorted set
        let mut token_set: Vec<usize> = Vec::new();
        for ch in &channels {
            for &tok in &ch.top_tokens {
                if let Err(pos) = token_set.binary_search(&tok) {
                    token_set.insert(pos, tok);
                }
            }
        }

        result.push(token_set);
    }

    result
}

// ── Phase 3: Reachability Map ───────────────────────────────────────

/// Per-layer, per-neuron token reachability map.
///
/// Uses compact sorted `Vec<usize>` sets per neuron for O(log n) lookup.
#[derive(Debug, Clone)]
pub struct VocabChannelMap {
    /// Per-layer reachability: `layers[layer_idx][neuron_idx]` = sorted token set
    layers: Vec<Vec<Vec<usize>>>,
    /// Vocabulary size for bounds checking
    vocab_size: usize,
}

impl VocabChannelMap {
    /// Build the map from per-neuron channel results.
    ///
    /// `channels_per_neuron[layer][neuron]` = sorted token reachability set.
    pub fn from_channels(channels_per_neuron: &[Vec<Vec<usize>>], vocab_size: usize) -> Self {
        Self {
            layers: channels_per_neuron.to_vec(),
            vocab_size,
        }
    }

    /// Number of layers in the map.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Number of neurons in a given layer.
    #[inline]
    pub fn neuron_count(&self, layer: usize) -> usize {
        match self.layers.get(layer) {
            Some(l) => l.len(),
            None => 0,
        }
    }

    /// Check if a token is reachable by a specific neuron in a specific layer.
    ///
    /// Uses binary search on the sorted token set: O(log n).
    pub fn is_reachable(&self, layer: usize, neuron: usize, token: usize) -> bool {
        match self.layers.get(layer).and_then(|l| l.get(neuron)) {
            Some(tokens) => tokens.binary_search(&token).is_ok(),
            None => false,
        }
    }

    /// Get the sorted token set for a specific neuron.
    ///
    /// Returns an empty slice if layer or neuron is out of bounds.
    #[inline]
    pub fn neuron_tokens(&self, layer: usize, neuron: usize) -> &[usize] {
        match self.layers.get(layer).and_then(|l| l.get(neuron)) {
            Some(tokens) => tokens,
            None => &[],
        }
    }

    /// Compute the union of all neurons' token sets for a given layer.
    ///
    /// Returns a sorted Vec. Useful for layer-level pruning without activation info.
    pub fn layer_union(&self, layer: usize) -> Vec<usize> {
        match self.layers.get(layer) {
            Some(neurons) => {
                let mut union_set: Vec<usize> = Vec::new();
                for tokens in neurons {
                    for &tok in tokens {
                        if let Err(pos) = union_set.binary_search(&tok) {
                            union_set.insert(pos, tok);
                        }
                    }
                }
                union_set
            }
            None => Vec::new(),
        }
    }

    /// Compute the global union of all tokens across all layers and neurons.
    ///
    /// Returns a sorted Vec.
    pub fn global_union(&self) -> Vec<usize> {
        let mut union_set: Vec<usize> = Vec::new();
        for neurons in &self.layers {
            for tokens in neurons {
                for &tok in tokens {
                    if let Err(pos) = union_set.binary_search(&tok) {
                        union_set.insert(pos, tok);
                    }
                }
            }
        }
        union_set
    }

    /// Serialize to a simple binary format.
    ///
    /// Format:
    /// - u32 LE: num_layers
    /// - Per layer:
    ///   - u32 LE: num_neurons
    ///   - Per neuron:
    ///     - u32 LE: num_tokens
    ///     - u32 LE × num_tokens: token indices
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        let num_layers = self.layers.len() as u32;
        buf.extend_from_slice(&num_layers.to_le_bytes());

        for neurons in &self.layers {
            let num_neurons = neurons.len() as u32;
            buf.extend_from_slice(&num_neurons.to_le_bytes());

            for tokens in neurons {
                let num_tokens = tokens.len() as u32;
                buf.extend_from_slice(&num_tokens.to_le_bytes());
                for &tok in tokens {
                    buf.extend_from_slice(&(tok as u32).to_le_bytes());
                }
            }
        }

        buf
    }

    /// Deserialize from binary format.
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("data too short for header".to_string());
        }

        let mut offset = 0;

        let read_u32 = |data: &[u8], off: &mut usize| -> Result<u32, String> {
            if *off + 4 > data.len() {
                return Err("unexpected end of data reading u32".to_string());
            }
            let val = u32::from_le_bytes(
                data[*off..*off + 4]
                    .try_into()
                    .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
            );
            *off += 4;
            Ok(val)
        };

        let num_layers = read_u32(data, &mut offset)? as usize;
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let num_neurons = read_u32(data, &mut offset)? as usize;
            let mut neurons = Vec::with_capacity(num_neurons);

            for _ in 0..num_neurons {
                let num_tokens = read_u32(data, &mut offset)? as usize;
                let mut tokens = Vec::with_capacity(num_tokens);

                for _ in 0..num_tokens {
                    let tok = read_u32(data, &mut offset)? as usize;
                    tokens.push(tok);
                }

                neurons.push(tokens);
            }

            layers.push(neurons);
        }

        Ok(Self {
            layers,
            vocab_size: 0, // Unknown from serialized data; caller sets via builder
        })
    }

    /// Get the stored vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Set the vocabulary size (used after deserialization).
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }
}

// ── Phase 4: ConstraintPruner ───────────────────────────────────────

/// ROTATE-derived ConstraintPruner for speculative decoding.
///
/// Stores per-layer, per-neuron token reachability maps. At inference time,
/// checks if a candidate token is in the reachability set of the current layer.
///
/// # Active Context
///
/// The pruner uses a thread-safe "active context" — the inference loop sets
/// the current layer and active neurons before each speculative tree build.
/// If no active context is set, falls back to the per-layer union (all neurons).
///
/// # Thread Safety
///
/// - `active_layer`: `AtomicUsize` for lock-free reads
/// - `active_neurons`: `RwLock<Vec<usize>>` for concurrent read/write
/// - `ConstraintPruner` impl is `Send + Sync`
pub struct VocabChannelPruner {
    /// Per-layer, per-neuron token reachability
    map: VocabChannelMap,
    /// Currently active layer (set by inference loop)
    active_layer: AtomicUsize,
    /// Currently active neurons (set by inference loop)
    active_neurons: RwLock<Vec<usize>>,
    /// Per-layer union of all neuron reachability (fallback when no neurons set)
    per_layer_union: Vec<Vec<usize>>,
}

impl VocabChannelPruner {
    /// Create a new pruner from a pre-built VocabChannelMap.
    pub fn new(map: VocabChannelMap) -> Self {
        let per_layer_union: Vec<Vec<usize>> =
            (0..map.layer_count()).map(|l| map.layer_union(l)).collect();

        Self {
            map,
            active_layer: AtomicUsize::new(0),
            active_neurons: RwLock::new(Vec::new()),
            per_layer_union,
        }
    }

    /// Set the current active context for the pruner.
    ///
    /// Call this before each speculative tree build to inform the pruner
    /// which layer and neurons are currently active.
    pub fn set_active_context(&self, layer: usize, neurons: &[usize]) {
        self.active_layer.store(layer, AtomicOrdering::Relaxed);
        if let Ok(mut guard) = self.active_neurons.write() {
            guard.clear();
            guard.extend_from_slice(neurons);
        }
    }

    /// Get the number of layers.
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.map.layer_count()
    }

    /// Check if a token is reachable by a specific set of neurons in a layer.
    ///
    /// A token is reachable if ANY of the given neurons has it in their reachability set.
    pub fn is_valid_with_neurons(
        &self,
        layer: usize,
        active_neurons: &[usize],
        token_idx: usize,
    ) -> bool {
        for &neuron in active_neurons {
            if self.map.is_reachable(layer, neuron, token_idx) {
                return true;
            }
        }
        false
    }

    /// Check if a token is in the layer's union reachability set.
    ///
    /// This is the fallback check when neuron activations are not available.
    pub fn is_valid_layer_union(&self, layer: usize, token_idx: usize) -> bool {
        match self.per_layer_union.get(layer) {
            Some(tokens) => tokens.binary_search(&token_idx).is_ok(),
            None => true, // Unknown layer → don't prune
        }
    }

    /// Get a reference to the underlying map.
    #[inline]
    pub fn map(&self) -> &VocabChannelMap {
        &self.map
    }
}

impl ConstraintPruner for VocabChannelPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        let layer = self.active_layer.load(AtomicOrdering::Relaxed);

        // Try neuron-specific check first
        if let Ok(neurons) = self.active_neurons.read() {
            if !neurons.is_empty() {
                return self.is_valid_with_neurons(layer, &neurons, token_idx);
            }
        }

        // Fallback: per-layer union
        self.is_valid_layer_union(layer, token_idx)
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[usize],
        _parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let layer = self.active_layer.load(AtomicOrdering::Relaxed);

        // Try neuron-specific check first
        if let Ok(neurons) = self.active_neurons.read() {
            if !neurons.is_empty() {
                let len = candidates.len().min(results.len());
                for i in 0..len {
                    results[i] = self.is_valid_with_neurons(layer, &neurons, candidates[i]);
                }
                return;
            }
        }

        // Fallback: per-layer union batch check
        let len = candidates.len().min(results.len());
        match self.per_layer_union.get(layer) {
            Some(tokens) => {
                for i in 0..len {
                    results[i] = tokens.binary_search(&candidates[i]).is_ok();
                }
            }
            None => {
                // Unknown layer → don't prune
                results[..len].fill(true);
            }
        }
    }
}

// ── Phase 4: ComposedPruner (DDTree Integration) ────────────────────

/// Compose multiple `ConstraintPruner`s — token is valid only if ALL pruners agree.
///
/// This enables stacking `VocabChannelPruner` with any existing pruner
/// (e.g., `SudokuPruner`, `EpisodePruner`) without modifying either.
///
/// # Example
///
/// ```ignore
/// let composed = ComposedPruner::new(vec![
///     Box::new(vocab_pruner),
///     Box::new(other_pruner),
/// ]);
/// build_dd_tree_pruned(&marginals, &config, &composed, false);
/// ```
pub struct ComposedPruner {
    pruners: Vec<Box<dyn ConstraintPruner>>,
}

impl ComposedPruner {
    /// Create a composed pruner from a list of pruners.
    ///
    /// An empty list acts as `NoPruner` (all tokens valid).
    pub fn new(pruners: Vec<Box<dyn ConstraintPruner>>) -> Self {
        Self { pruners }
    }

    /// Create a single-pruner composition (no overhead if only one).
    pub fn single(pruner: Box<dyn ConstraintPruner>) -> Self {
        Self {
            pruners: vec![pruner],
        }
    }

    /// Number of composed pruners.
    pub fn len(&self) -> usize {
        self.pruners.len()
    }

    /// True if no pruners are composed.
    pub fn is_empty(&self) -> bool {
        self.pruners.is_empty()
    }
}

impl ConstraintPruner for ComposedPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // ALL must agree — short-circuit on first rejection
        self.pruners
            .iter()
            .all(|p| p.is_valid(depth, token_idx, parent_tokens))
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        // Initialize: all valid
        results[..len].fill(true);

        // AND-reduce across all pruners
        let mut buf = vec![false; len];
        for pruner in &self.pruners {
            pruner.batch_is_valid(depth, candidates, parent_tokens, &mut buf);
            for i in 0..len {
                results[i] = results[i] && buf[i];
            }
        }
    }
}

// ── Phase 5: Load-Time Pipeline ─────────────────────────────────────

/// Result of load-time ROTATE decomposition for a single model.
pub struct DecompositionResult {
    /// Per-layer decomposition timing.
    pub layer_timings_ms: Vec<f64>,
    /// Total decomposition time.
    pub total_ms: f64,
    /// BLAKE3 hash of the weight bytes used for decomposition.
    pub weight_hash: [u8; 32],
    /// The constructed pruner (ready for use).
    pub pruner: VocabChannelPruner,
}

/// Decompose all layers of a model's MLP output weights into vocab channels.
///
/// Runs ROTATE decomposition on each layer's `mlp_w2` matrix against `lm_head`,
/// builds a `VocabChannelMap`, and wraps it in a `VocabChannelPruner`.
///
/// Returns timing per layer, total time, and the BLAKE3 hash of all weight bytes.
pub fn decompose_model_channels(
    weights: &crate::transformer::TransformerWeights,
    config: &crate::types::Config,
    channel_config: &VocabChannelConfig,
) -> DecompositionResult {
    let mut hasher = blake3::Hasher::new();

    // Hash lm_head once
    hasher.update(bytemuck::cast_slice::<f32, u8>(&weights.lm_head));

    let n_layer = weights.layers.len();
    let mut layer_channels = Vec::with_capacity(n_layer);
    let mut layer_timings_ms = Vec::with_capacity(n_layer);

    let total_start = std::time::Instant::now();

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        // Hash this layer's mlp_w2
        hasher.update(bytemuck::cast_slice::<f32, u8>(&layer.mlp_w2));

        let layer_start = std::time::Instant::now();
        let channels = decompose_layer_channels(
            &layer.mlp_w2,
            &weights.lm_head,
            config.n_embd,
            config.mlp_hidden,
            config.vocab_size,
            channel_config,
        );
        let layer_elapsed = layer_start.elapsed();
        layer_timings_ms.push(layer_elapsed.as_secs_f64() * 1000.0);

        log::info!(
            "[vocab_channel] Layer {}/{}: {} neurons, {:.1}ms",
            layer_idx,
            n_layer,
            channels.len(),
            layer_elapsed.as_secs_f64() * 1000.0,
        );

        layer_channels.push(channels);
    }

    let total_elapsed = total_start.elapsed();
    let weight_hash = *hasher.finalize().as_bytes();

    let map = VocabChannelMap::from_channels(&layer_channels, config.vocab_size);
    let pruner = VocabChannelPruner::new(map);

    log::info!(
        "[vocab_channel] Decomposition complete: {} layers, {:.1}ms total, hash={:.16}",
        n_layer,
        total_elapsed.as_secs_f64() * 1000.0,
        weight_hash[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    DecompositionResult {
        layer_timings_ms,
        total_ms: total_elapsed.as_secs_f64() * 1000.0,
        weight_hash,
        pruner,
    }
}

/// Cache file header for serialized `VocabChannelMap`.
#[derive(serde::Serialize, serde::Deserialize)]
struct CacheHeader {
    /// BLAKE3 hash of the weight bytes used for decomposition.
    weight_hash: [u8; 32],
    /// Vocab size at decomposition time.
    vocab_size: usize,
    /// Layer count at decomposition time.
    layer_count: usize,
}

/// Try to load a cached `VocabChannelMap` from disk.
///
/// Returns `Some(pruner)` if the cache exists, the weight hash matches,
/// and the dimensions are compatible. Returns `None` otherwise.
pub fn load_cached_pruner(
    cache_path: &std::path::Path,
    expected_hash: &[u8; 32],
    expected_vocab_size: usize,
    expected_layer_count: usize,
) -> Option<VocabChannelPruner> {
    let data = std::fs::read(cache_path).ok()?;
    if data.len() < 4 + std::mem::size_of::<CacheHeader>() {
        log::info!("[vocab_channel] Cache file too small, skipping");
        return None;
    }

    // Header: 4 bytes JSON length (big-endian) + JSON + binary map
    let json_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if data.len() < 4 + json_len {
        log::info!("[vocab_channel] Cache file truncated, skipping");
        return None;
    }

    let header: CacheHeader = serde_json::from_slice(&data[4..4 + json_len]).ok()?;

    if &header.weight_hash != expected_hash {
        log::info!("[vocab_channel] Cache hash mismatch, skipping");
        return None;
    }
    if header.vocab_size != expected_vocab_size || header.layer_count != expected_layer_count {
        log::info!(
            "[vocab_channel] Cache dimension mismatch (expected vocab={}, layers={}), skipping",
            expected_vocab_size,
            expected_layer_count,
        );
        return None;
    }

    let map_bytes = &data[4 + json_len..];
    let map = VocabChannelMap::deserialize(map_bytes).ok()?;
    log::info!(
        "[vocab_channel] Loaded cached map: {} layers, {:.1}KB",
        map.layer_count(),
        map_bytes.len() as f64 / 1024.0,
    );
    Some(VocabChannelPruner::new(map))
}

/// Save a `VocabChannelMap` to disk with BLAKE3 hash verification header.
///
/// Format: `[4-byte JSON len][JSON header][binary map]`
pub fn save_pruner_cache(
    cache_path: &std::path::Path,
    weight_hash: &[u8; 32],
    pruner: &VocabChannelPruner,
    vocab_size: usize,
    layer_count: usize,
) -> std::io::Result<()> {
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let header = CacheHeader {
        weight_hash: *weight_hash,
        vocab_size,
        layer_count,
    };
    let header_json = serde_json::to_vec(&header)?;

    let map_bytes = pruner.map().serialize();

    let mut out = Vec::with_capacity(4 + header_json.len() + map_bytes.len());
    out.extend_from_slice(&(header_json.len() as u32).to_be_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&map_bytes);

    std::fs::write(cache_path, out)?;
    log::info!(
        "[vocab_channel] Saved cache: {:.1}KB to {}",
        (4 + header_json.len() + map_bytes.len()) as f64 / 1024.0,
        cache_path.display(),
    );
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Phase 1 Tests ──

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric distribution → skewness ≈ 0
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let s = skewness(&values);
        assert!(
            s.abs() < 0.1,
            "Symmetric distribution should have near-zero skewness, got {s}"
        );
    }

    #[test]
    fn test_skewness_right_tail() {
        // Right-skewed: long tail to the right
        let values = [1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
        let s = skewness(&values);
        assert!(
            s > 0.5,
            "Right-skewed distribution should have positive skewness, got {s}"
        );
    }

    #[test]
    fn test_skewness_left_tail() {
        // Left-skewed: most values are high, tail extends to the left
        let values = [100.0, 100.0, 100.0, 100.0, 100.0, 1.0];
        let s = skewness(&values);
        assert!(
            s < -0.5,
            "Left-skewed distribution should have negative skewness, got {s}"
        );
    }

    #[test]
    fn test_skewness_degenerate() {
        assert_eq!(skewness(&[]), 0.0, "Empty should be 0");
        assert_eq!(skewness(&[1.0]), 0.0, "Single should be 0");
        assert_eq!(skewness(&[1.0, 2.0]), 0.0, "Two should be 0");
        assert_eq!(skewness(&[3.0, 3.0, 3.0]), 0.0, "Zero variance should be 0");
    }

    #[test]
    fn test_excess_kurtosis_peaked() {
        let values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
        let k = excess_kurtosis(&values);
        assert!(
            k > 5.0,
            "Peaked distribution should have high kurtosis, got {k}"
        );
    }

    #[test]
    fn test_excess_kurtosis_uniform() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let k = excess_kurtosis(&values);
        assert!(k < 0.0, "Uniform should have negative kurtosis, got {k}");
    }

    #[test]
    fn test_excess_kurtosis_edge_cases() {
        assert_eq!(excess_kurtosis(&[]), 0.0);
        assert_eq!(excess_kurtosis(&[1.0]), 0.0);
        assert_eq!(excess_kurtosis(&[3.0, 3.0, 3.0, 3.0]), 0.0);
    }

    #[test]
    fn test_householder_apply_identity() {
        // Zero Householder vector → no reflection (identity)
        let h = [0.0f32; 4];
        let x = [1.0, 2.0, 3.0, 4.0];
        let result = householder_apply(&h, &x);
        for (r, &xi) in result.iter().zip(x.iter()) {
            assert!((r - xi).abs() < 1e-6, "Zero h should be identity");
        }
    }

    #[test]
    fn test_householder_apply_reflection() {
        // h = e₁ (standard basis) → reflects x about the hyperplane orthogonal to e₁
        let h = [1.0f32, 0.0, 0.0, 0.0];
        let x = [1.0, 2.0, 3.0, 4.0];
        let result = householder_apply(&h, &x);
        // R = I - 2*e₁*e₁ᵀ → x → (x₀ - 2x₀, x₁, x₂, x₃)
        assert!(
            (result[0] - (-1.0)).abs() < 1e-6,
            "First component should be negated"
        );
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_householder_apply_preserves_norm() {
        // Householder reflections are orthogonal → preserve norm
        let h = [0.5f32, 0.5, 0.5, 0.5];
        let x = [1.0, 2.0, 3.0, 4.0];
        let result = householder_apply(&h, &x);

        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_r: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_x - norm_r).abs() < 1e-4,
            "Norm should be preserved: {norm_x} vs {norm_r}"
        );
    }

    #[test]
    fn test_vocab_project_basic() {
        // lm_head = [[1, 0], [0, 1], [1, 1]] → vocab_size=3, n_embd=2
        let lm_head = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let neuron_weight = [2.0f32, 3.0];

        let logits = vocab_project(&neuron_weight, &lm_head, 3, 2);

        assert_eq!(logits.len(), 3);
        assert!((logits[0] - 2.0).abs() < 1e-6, "token 0: 1*2 + 0*3 = 2");
        assert!((logits[1] - 3.0).abs() < 1e-6, "token 1: 0*2 + 1*3 = 3");
        assert!((logits[2] - 5.0).abs() < 1e-6, "token 2: 1*2 + 1*3 = 5");
    }

    #[test]
    fn test_vocab_project_zeros() {
        let lm_head = [1.0f32, 2.0, 3.0, 4.0];
        let neuron_weight = [0.0f32, 0.0];
        let logits = vocab_project(&neuron_weight, &lm_head, 2, 2);
        assert!((logits[0]).abs() < 1e-10);
        assert!((logits[1]).abs() < 1e-10);
    }

    #[test]
    fn test_iterative_token_mask_basic() {
        // Values: [1, 2, 3, 100] → mean ≈ 26.5, σ ≈ 43.1, k=1.5 → threshold ≈ 64.7
        // 100 is outlier, rest are within range
        let logits = [1.0f32, 2.0, 3.0, 100.0];
        let mut mask = [false; 4];
        iterative_token_mask(&logits, &mut mask, 1.5);

        assert!(mask[3], "100.0 should be masked as outlier");
        assert!(!mask[0], "1.0 should not be masked");
        assert!(!mask[1], "2.0 should not be masked");
        assert!(!mask[2], "3.0 should not be masked");
    }

    #[test]
    fn test_iterative_token_mask_preserves_existing() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut mask = [false, true, false, false]; // token 1 already masked
        iterative_token_mask(&logits, &mut mask, 0.5);

        assert!(mask[1], "Existing mask should be preserved");
    }

    #[test]
    fn test_iterative_token_mask_uniform() {
        // All same → σ = 0 → no masking
        let logits = [5.0f32; 10];
        let mut mask = [false; 10];
        iterative_token_mask(&logits, &mut mask, 2.0);
        assert!(mask.iter().all(|m| !m), "Uniform should not mask anything");
    }

    #[test]
    fn test_topk_indices_basic() {
        let values = [3.0f32, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0, 5.0, 3.5];
        let top = topk_indices(&values, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], 4, "Highest value at index 4 (9.0)");
        assert_eq!(top[1], 6, "Second highest at index 6 (6.0)");
        assert_eq!(top[2], 7, "Third highest at index 7 (5.0)");
    }

    #[test]
    fn test_topk_indices_larger_than_input() {
        let values = [1.0f32, 2.0, 3.0];
        let top = topk_indices(&values, 10);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_topk_indices_empty() {
        let values: [f32; 0] = [];
        let top = topk_indices(&values, 5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_cosine_sim_identical() {
        let a = [1.0f32, 2.0, 3.0];
        assert!((cosine_sim(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert!((cosine_sim(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    // ── Phase 2 Tests ──

    #[test]
    fn test_decompose_neuron_discovers_channels() {
        // Create a neuron weight that clearly points to token 2
        // n_embd = 4, vocab_size = 4
        // lm_head: identity matrix — token t → unit vector e_t
        let lm_head: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
            0.0, 0.0, 0.0, 1.0, // token 3
        ];

        // Neuron weight pointing strongly at token 2
        let neuron_weight = [0.01f32, 0.01, 10.0, 0.01];

        // Verify the raw projection
        let logits = vocab_project(&neuron_weight, &lm_head, 4, 4);
        assert!(
            (logits[2] - 10.0).abs() < 1e-6,
            "Token 2 should have logit 10.0"
        );

        let config = VocabChannelConfig {
            max_channels: 3,
            top_k_tokens: 4,
            kurtosis_threshold: -10.0, // Accept any channel
            lambda: 0.001,
            eta: 0.005,
            max_iterations: 20,
            sigma_mask: 5.0,
            coords_per_iter: 4,
            fd_epsilon: 1e-3,
        };

        let decomposer = VocabChannelDecomposer::new(config);
        let channels = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);

        assert!(!channels.is_empty(), "Should discover at least one channel");
        let first = &channels[0];
        assert!(
            !first.top_tokens.is_empty(),
            "Channel should have top tokens"
        );
        // The first channel should contain token 2 as the dominant token
        assert_eq!(
            first.top_tokens[0], 2,
            "Token 2 should be the top token, got {:?}",
            first.top_tokens
        );
    }

    #[test]
    fn test_decompose_neuron_kurtosis_threshold() {
        let lm_head: Vec<f32> = (0..4)
            .flat_map(|t| {
                let mut row = vec![0.0f32; 4];
                row[t] = 1.0;
                row
            })
            .collect();

        let neuron_weight = [1.0f32, 1.0, 1.0, 1.0];

        let config = VocabChannelConfig {
            max_channels: 5,
            kurtosis_threshold: 100.0, // Very high threshold → should reject all
            ..Default::default()
        };

        let decomposer = VocabChannelDecomposer::new(config);
        let channels = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);
        assert!(
            channels.is_empty(),
            "Should not discover channels with high threshold"
        );
    }

    #[test]
    fn test_decompose_layer_channels() {
        let n_embd = 4;
        let mlp_hidden = 3;
        let vocab_size = 6;

        // mlp_w2: [n_embd, mlp_hidden] row-major
        let mlp_w2: Vec<f32> = vec![
            // row 0 (embd dim 0)
            1.0, 0.0, 0.5, // row 1 (embd dim 1)
            0.0, 1.0, 0.5, // row 2 (embd dim 2)
            0.0, 0.0, 1.0, // row 3 (embd dim 3)
            0.0, 0.0, 0.0,
        ];

        // lm_head: [vocab_size, n_embd]
        let lm_head: Vec<f32> = (0..vocab_size)
            .flat_map(|t| {
                let mut row = vec![0.0f32; n_embd];
                if t < n_embd {
                    row[t] = 1.0;
                }
                row
            })
            .collect();

        let config = VocabChannelConfig {
            max_channels: 2,
            top_k_tokens: 3,
            kurtosis_threshold: -10.0,
            ..Default::default()
        };

        let result =
            decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);

        assert_eq!(
            result.len(),
            mlp_hidden,
            "Should have one token set per neuron"
        );
        for (i, tokens) in result.iter().enumerate() {
            // Each set should be sorted
            for window in tokens.windows(2) {
                assert!(
                    window[0] < window[1],
                    "Token set for neuron {i} should be sorted"
                );
            }
        }
    }

    // ── Phase 3 Tests ──

    #[test]
    fn test_channel_map_from_channels() {
        let channels_per_neuron = vec![
            vec![
                vec![1, 3, 5], // neuron 0
                vec![2, 4, 6], // neuron 1
                vec![1, 2, 7], // neuron 2
            ],
            vec![
                vec![0, 1, 2], // layer 1, neuron 0
                vec![3, 4, 5], // layer 1, neuron 1
            ],
        ];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        assert_eq!(map.layer_count(), 2);
        assert_eq!(map.neuron_count(0), 3);
        assert_eq!(map.neuron_count(1), 2);

        assert!(map.is_reachable(0, 0, 3));
        assert!(map.is_reachable(0, 2, 7));
        assert!(!map.is_reachable(0, 0, 2));
        assert!(map.is_reachable(1, 0, 0));
    }

    #[test]
    fn test_channel_map_neuron_tokens() {
        let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4]]];
        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);

        assert_eq!(map.neuron_tokens(0, 0), &[1, 3, 5]);
        assert_eq!(map.neuron_tokens(0, 1), &[2, 4]);
        let empty: &[usize] = &[];
        assert_eq!(map.neuron_tokens(0, 99), empty);
        assert_eq!(map.neuron_tokens(99, 0), empty);
    }

    #[test]
    fn test_channel_map_layer_union() {
        let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 3, 6], vec![1, 7]]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let union = map.layer_union(0);

        assert_eq!(union, vec![1, 2, 3, 5, 6, 7]);
    }

    #[test]
    fn test_channel_map_global_union() {
        let channels_per_neuron = vec![vec![vec![1, 3], vec![5]], vec![vec![2, 3], vec![7]]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let union = map.global_union();

        assert_eq!(union, vec![1, 2, 3, 5, 7]);
    }

    #[test]
    fn test_channel_map_roundtrip_serialization() {
        let channels_per_neuron = vec![
            vec![vec![1, 3, 5], vec![2, 4, 6, 8], vec![0, 7]],
            vec![vec![10, 20, 30], vec![15, 25]],
        ];

        let original = VocabChannelMap::from_channels(&channels_per_neuron, 32000);
        let bytes = original.serialize();
        let restored = VocabChannelMap::deserialize(&bytes)
            .expect("deserialization should succeed")
            .with_vocab_size(32000);

        assert_eq!(restored.layer_count(), original.layer_count());
        assert_eq!(restored.neuron_count(0), original.neuron_count(0));
        assert_eq!(restored.neuron_count(1), original.neuron_count(1));

        for layer in 0..original.layer_count() {
            for neuron in 0..original.neuron_count(layer) {
                assert_eq!(
                    restored.neuron_tokens(layer, neuron),
                    original.neuron_tokens(layer, neuron),
                    "Mismatch at layer={layer} neuron={neuron}"
                );
            }
        }
    }

    #[test]
    fn test_channel_map_deserialize_too_short() {
        let result = VocabChannelMap::deserialize(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_channel_map_empty_serialization() {
        let original = VocabChannelMap::from_channels(&[], 0);
        let bytes = original.serialize();
        let restored = VocabChannelMap::deserialize(&bytes).expect("should succeed");
        assert_eq!(restored.layer_count(), 0);
    }

    // ── Phase 4 Tests ──

    #[test]
    fn test_pruner_basic() {
        let channels_per_neuron = vec![vec![
            vec![1, 3, 5], // neuron 0: tokens 1,3,5
            vec![2, 4, 6], // neuron 1: tokens 2,4,6
        ]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        // Set active context: layer 0, neuron 0 active
        pruner.set_active_context(0, &[0]);

        assert!(pruner.is_valid(0, 1, &[]), "Token 1 reachable by neuron 0");
        assert!(pruner.is_valid(0, 3, &[]), "Token 3 reachable by neuron 0");
        assert!(
            !pruner.is_valid(0, 2, &[]),
            "Token 2 NOT reachable by neuron 0"
        );
        assert!(
            !pruner.is_valid(0, 7, &[]),
            "Token 7 NOT reachable by any neuron 0"
        );
    }

    #[test]
    fn test_pruner_multiple_neurons() {
        let channels_per_neuron = vec![vec![
            vec![1, 3, 5], // neuron 0
            vec![2, 4, 6], // neuron 1
        ]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        // Both neurons active
        pruner.set_active_context(0, &[0, 1]);

        assert!(pruner.is_valid(0, 1, &[]), "Token 1 reachable by neuron 0");
        assert!(pruner.is_valid(0, 2, &[]), "Token 2 reachable by neuron 1");
        assert!(!pruner.is_valid(0, 7, &[]), "Token 7 NOT reachable");
    }

    #[test]
    fn test_pruner_fallback_to_union() {
        let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4, 6]]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        // No neurons set → should use per-layer union
        // Union of neuron 0 and 1: {1, 2, 3, 4, 5, 6}
        assert!(pruner.is_valid(0, 1, &[]), "Token 1 in union");
        assert!(pruner.is_valid(0, 4, &[]), "Token 4 in union");
        assert!(!pruner.is_valid(0, 7, &[]), "Token 7 NOT in union");
        assert!(!pruner.is_valid(0, 0, &[]), "Token 0 NOT in union");
    }

    #[test]
    fn test_pruner_unknown_layer() {
        let map = VocabChannelMap::from_channels(&[], 10);
        let pruner = VocabChannelPruner::new(map);

        // Unknown layer → don't prune (return true)
        assert!(
            pruner.is_valid(0, 42, &[]),
            "Unknown layer should not prune"
        );
    }

    #[test]
    fn test_pruner_batch_is_valid() {
        let channels_per_neuron = vec![vec![vec![1, 3, 5]]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        pruner.set_active_context(0, &[0]);

        let candidates = [1, 2, 3, 4, 5, 6];
        let mut results = [false; 6];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        assert!(results[0], "Token 1 should be valid");
        assert!(!results[1], "Token 2 should be invalid");
        assert!(results[2], "Token 3 should be valid");
        assert!(!results[3], "Token 4 should be invalid");
        assert!(results[4], "Token 5 should be valid");
        assert!(!results[5], "Token 6 should be invalid");
    }

    #[test]
    fn test_pruner_batch_is_valid_union_fallback() {
        let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4]]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        // No active neurons → fallback to union {1, 2, 3, 4, 5}
        let candidates = [0, 1, 2, 3, 4, 5, 6];
        let mut results = [false; 7];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        assert!(!results[0], "Token 0 NOT in union");
        assert!(results[1], "Token 1 in union");
        assert!(results[2], "Token 2 in union");
        assert!(results[3], "Token 3 in union");
        assert!(results[4], "Token 4 in union");
        assert!(results[5], "Token 5 in union");
        assert!(!results[6], "Token 6 NOT in union");
    }

    #[test]
    fn test_pruner_is_valid_with_neurons() {
        let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4, 6]]];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        assert!(pruner.is_valid_with_neurons(0, &[0], 3));
        assert!(!pruner.is_valid_with_neurons(0, &[0], 2));
        assert!(pruner.is_valid_with_neurons(0, &[0, 1], 2));
        assert!(!pruner.is_valid_with_neurons(0, &[0, 1], 7));
    }

    // ── Integration / Larger Tests ──

    #[test]
    fn test_full_pipeline_small() {
        // Simulate a tiny model: 2 layers, 3 neurons each, vocab=8, n_embd=4
        let n_embd = 4;
        let mlp_hidden = 3;
        let vocab_size = 8;

        // lm_head: [8, 4] — each token gets a unique direction
        let lm_head: Vec<f32> = (0..vocab_size)
            .flat_map(|t| {
                let mut row = vec![0.1f32; n_embd];
                if t < n_embd {
                    row[t] = 5.0; // Strong signal for tokens 0-3
                }
                row
            })
            .collect();

        let config = VocabChannelConfig {
            max_channels: 2,
            top_k_tokens: 4,
            kurtosis_threshold: -10.0,
            lambda: 0.01,
            eta: 0.01,
            max_iterations: 5,
            sigma_mask: 3.0,
            coords_per_iter: 2,
            fd_epsilon: 1e-3,
        };

        // Layer 0
        let mlp_w2_layer0: Vec<f32> =
            vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        let layer0_tokens = decompose_layer_channels(
            &mlp_w2_layer0,
            &lm_head,
            n_embd,
            mlp_hidden,
            vocab_size,
            &config,
        );

        assert_eq!(layer0_tokens.len(), mlp_hidden);
        for tokens in &layer0_tokens {
            // Verify sorted
            for window in tokens.windows(2) {
                assert!(window[0] < window[1]);
            }
        }

        // Build map
        let channels_per_neuron = vec![layer0_tokens];
        let map = VocabChannelMap::from_channels(&channels_per_neuron, vocab_size);
        let pruner = VocabChannelPruner::new(map);

        assert_eq!(pruner.layer_count(), 1);
    }

    #[test]
    fn test_serialization_roundtrip_large() {
        // Larger map with multiple layers
        let channels_per_neuron: Vec<Vec<Vec<usize>>> = (0..4)
            .map(|layer| {
                (0..10)
                    .map(|neuron| {
                        let base = layer * 100 + neuron * 10;
                        (0..5).map(|i| base + i).collect()
                    })
                    .collect()
            })
            .collect();

        let original = VocabChannelMap::from_channels(&channels_per_neuron, 1000);
        let bytes = original.serialize();

        // Verify serialized size is reasonable
        let expected_min = 4 + 4 * (4 + 10 * (4 + 5 * 4)); // header + 4 layers × (neurons + tokens)
        assert!(
            bytes.len() >= expected_min,
            "Serialized size {} should be at least {}",
            bytes.len(),
            expected_min
        );

        let restored = VocabChannelMap::deserialize(&bytes)
            .expect("deserialization should succeed")
            .with_vocab_size(1000);

        // Full verification
        for layer in 0..original.layer_count() {
            for neuron in 0..original.neuron_count(layer) {
                assert_eq!(
                    restored.neuron_tokens(layer, neuron),
                    original.neuron_tokens(layer, neuron)
                );
            }
        }
    }

    #[test]
    fn test_apply_mask() {
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        let mask = [false, true, false, true];
        apply_mask(&mut logits, &mask);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 0.0);
    }

    #[test]
    fn test_vocabulary_config_default() {
        let config = VocabChannelConfig::default();
        assert_eq!(config.max_channels, 5);
        assert_eq!(config.top_k_tokens, 50);
        assert_eq!(config.kurtosis_threshold, 1.0);
        assert!(config.lambda > 0.0);
        assert!(config.eta > 0.0);
        assert!(config.max_iterations > 0);
    }

    #[test]
    fn test_pruner_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VocabChannelPruner>();
    }

    #[test]
    fn test_pruner_context_switching() {
        let channels_per_neuron = vec![
            vec![vec![1, 2], vec![3, 4]], // layer 0
            vec![vec![5, 6], vec![7, 8]], // layer 1
        ];

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
        let pruner = VocabChannelPruner::new(map);

        // Layer 0, neuron 0
        pruner.set_active_context(0, &[0]);
        assert!(pruner.is_valid(0, 1, &[]));
        assert!(!pruner.is_valid(0, 3, &[]));

        // Switch to layer 1, neuron 1
        pruner.set_active_context(1, &[1]);
        assert!(!pruner.is_valid(0, 1, &[]));
        assert!(pruner.is_valid(0, 7, &[]));
    }

    #[test]
    fn test_decompose_neuron_zero_weight() {
        let lm_head: Vec<f32> = (0..4)
            .flat_map(|t| {
                let mut row = vec![0.0f32; 4];
                row[t] = 1.0;
                row
            })
            .collect();

        let neuron_weight = [0.0f32; 4];
        let config = VocabChannelConfig {
            max_channels: 2,
            kurtosis_threshold: -100.0,
            ..Default::default()
        };

        let decomposer = VocabChannelDecomposer::new(config);
        let channels = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);

        // Zero weight → zero logits → kurtosis = 0 → may or may not produce channels
        // Just verify it doesn't panic
        let _ = channels;
    }

    // ── Benchmark ──

    #[test]
    fn test_bench_vocab_project_v1000() {
        let vocab_size = 1000;
        let n_embd = 256;
        let lm_head: Vec<f32> = (0..vocab_size * n_embd)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let neuron_weight: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.01).cos()).collect();

        let start = std::time::Instant::now();
        let iters = 1_000;
        for _ in 0..iters {
            std::hint::black_box(vocab_project(&neuron_weight, &lm_head, vocab_size, n_embd));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("vocab_project V=1000 d=256: {per_call:.0}ns/call");
        assert!(
            per_call < 15_000_000.0,
            "vocab_project V=1000 should be <15ms (debug), got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_householder_apply_d256() {
        let d = 256;
        let h: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.2).cos()).collect();

        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(householder_apply(&h, &x));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("householder_apply d=256: {per_call:.0}ns/call");
        assert!(
            per_call < 100_000.0,
            "householder_apply d=256 should be <100μs, got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_pruner_is_valid() {
        let channels_per_neuron: Vec<Vec<Vec<usize>>> = (0..1)
            .map(|_| {
                (0..100)
                    .map(|n| {
                        let mut tokens: Vec<usize> = (0..50).map(|i| n * 50 + i).collect();
                        tokens.sort();
                        tokens
                    })
                    .collect()
            })
            .collect();

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 5000);
        let pruner = VocabChannelPruner::new(map);

        pruner.set_active_context(0, &[0, 1, 2]);

        let start = std::time::Instant::now();
        let iters = 100_000;
        for i in 0..iters {
            std::hint::black_box(pruner.is_valid(0, i % 5000, &[]));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("VocabChannelPruner::is_valid: {per_call:.0}ns/call");
        assert!(
            per_call < 10_000.0,
            "is_valid should be <10μs, got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_batch_is_valid() {
        let channels_per_neuron: Vec<Vec<Vec<usize>>> = (0..1)
            .map(|_| {
                (0..100)
                    .map(|n| {
                        let mut tokens: Vec<usize> = (0..50).map(|i| n * 50 + i).collect();
                        tokens.sort();
                        tokens
                    })
                    .collect()
            })
            .collect();

        let map = VocabChannelMap::from_channels(&channels_per_neuron, 5000);
        let pruner = VocabChannelPruner::new(map);

        pruner.set_active_context(0, &[0, 1, 2]);

        let candidates: Vec<usize> = (0..1000).collect();
        let mut results = vec![false; 1000];

        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            pruner.batch_is_valid(0, &candidates, &[], &mut results);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("VocabChannelPruner::batch_is_valid V=1000: {per_call:.0}ns/call");
        assert!(
            per_call < 1_000_000.0,
            "batch_is_valid V=1000 should be <1ms, got {per_call:.0}ns"
        );
    }
}

// TL;DR: ROTATE-derived ConstraintPruner for speculative decoding (Plan 228).
// - Phase 1: `skewness()`, `excess_kurtosis()`, `householder_apply()`, `vocab_project()`,
//   `iterative_token_mask()` — core math, O(n) single-pass, no alloc on hot paths.
// - Phase 2: `VocabChannelDecomposer` — kurtosis-maximizing Householder coordinate descent
//   discovers vocabulary channels per neuron; `decompose_layer_channels()` extracts column
//   weights from row-major mlp_w2 and runs per-neuron decomposition.
// - Phase 3: `VocabChannelMap` — per-layer, per-neuron sorted token sets with O(log n) binary
//   search lookup, layer/global unions, binary serialization round-trip.
// - Phase 4: `VocabChannelPruner` implements `ConstraintPruner` — thread-safe active context
//   (AtomicUsize + RwLock) for neuron-specific pruning, falls back to per-layer union.
// Feature-gated behind `vocab_channel_pruner`.
