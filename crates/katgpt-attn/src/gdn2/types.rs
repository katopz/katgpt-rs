//! GDN2 (Gated DeltaNet-2) cache types.
//!
//! Implements constant-size O(d_k × d_v) recurrent state for GDN2 attention.
//! Replaces growing KV cache with fixed-size recurrent state matrix S per head,
//! with decoupled erase/write gates for fine-grained memory control.
//!
//! # Gate Configurations
//!
//! | Config | Gates | State interaction | Quality |
//! |--------|-------|-------------------|---------|
//! | **EraseOnly** | b (erase), w (scalar write) | ~90% of full gain | Good |
//! | **Full** | b (erase), w (channel write) | Full expressiveness | Best |
//! | **Kda** | β (tied scalar) | Baseline | Simple |
//!
//! Reference: "Gated DeltaNet" (2024). See `.research/70_*.md` for derivation.
//!
//! # GQA Support
//!
//! With Grouped Query Attention (`n_kv_head < n_head`):
//! - State S shared per KV group, accessed by all Q heads in same group
//!
//! KV group for Q head `h`: `kv_group = h * n_kv_head / n_head`

use katgpt_core::types::Config;

// ── Gate Configuration ────────────────────────────────────────

/// Gate configuration controlling which gates are active.
///
/// Controls the expressiveness-cost tradeoff:
/// - `EraseOnly`: channel-wise erase b + scalar write w. Recovers ~90% of full gain.
/// - `Full`: channel-wise erase b + channel-wise write w. Maximum expressiveness.
/// - `Kda`: single scalar β (tied gates). Simplest baseline.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gdn2GateConfig {
    /// Erase-only: channel-wise b, scalar w. Recovers ~90% of full gain.
    #[default]
    EraseOnly,
    /// Full GDN2: channel-wise erase b + write w gates.
    Full,
    /// KDA baseline: scalar β (tied gates).
    Kda,
}

// ── Per-Head State ─────────────────────────────────────────────

/// Per-head recurrent state for GDN2.
///
/// Maintains a single state matrix S ∈ R^{d_k × d_v} that is updated
/// via gated delta rule at each timestep.
///
/// State size per head: dk × dv floats (typically hd² for dk = dv = hd).
#[derive(Clone)]
pub struct Gdn2HeadState {
    /// Recurrent state matrix S ∈ R^{dk × dv}.
    /// Row-major: s[i * dv + j] = S_{i,j}.
    pub s: Vec<f32>,
}

impl Gdn2HeadState {
    /// Allocate zeroed state for one head.
    pub fn new(dk: usize, dv: usize) -> Self {
        Self {
            s: vec![0.0; dk * dv],
        }
    }

    /// Reset to zeroed state (reuse allocations).
    pub fn reset(&mut self) {
        self.s.fill(0.0);
    }
}

// ── Per-Layer State ────────────────────────────────────────────

/// Per-layer GDN2 state with GQA support.
///
/// GQA-aware layout:
/// - `heads[n_kv_head]`: state matrix S per KV group, shared across Q heads
///
/// Total state per layer: n_kv_head × dk × dv floats.
#[derive(Clone)]
pub struct Gdn2LayerState {
    /// Per-KV-head recurrent states. GQA: shared across Q heads in same group.
    pub heads: Vec<Gdn2HeadState>,

    // ── Pre-allocated scratch buffers (zero alloc in hot path) ──
    /// Output buffer `[head_dim]`, zeroed before each use.
    pub out_buf: Vec<f32>,
    /// Temporary buffer `[head_dim]`, zeroed before each use.
    pub temp_buf: Vec<f32>,
    /// Erase gate defaults `[head_dim]`, pre-filled with 0.5.
    pub erase_b: Vec<f32>,
    /// Per-channel decay `[head_dim]`, pre-filled with 0.99.
    pub decay_alpha: Vec<f32>,
    /// Channel-wise write gate `[head_dim]`, pre-filled with 1.0.
    pub write_w_channel: Vec<f32>,
    /// Delta buffer `[head_dim]`, zeroed before each use.
    pub delta: Vec<f32>,

    /// Gate configuration for this layer.
    pub gate_config: Gdn2GateConfig,
}

impl Gdn2LayerState {
    /// Allocate zeroed state for one layer given config.
    pub fn new(config: &Config, gate_config: Gdn2GateConfig) -> Self {
        let dk = config.head_dim;
        let dv = config.head_dim;
        let mut heads = Vec::with_capacity(config.n_kv_head);
        for _ in 0..config.n_kv_head {
            heads.push(Gdn2HeadState::new(dk, dv));
        }
        Self {
            heads,
            gate_config,
            out_buf: vec![0.0; dv],
            temp_buf: vec![0.0; dv],
            erase_b: vec![0.5; dv],
            decay_alpha: vec![0.99; dk],
            write_w_channel: vec![1.0; dv],
            delta: vec![0.0; dv],
        }
    }

    /// Reset to zeroed state (reuse allocations).
    ///
    /// Only resets the head state matrices; scratch buffers are zeroed
    /// on each use so they don't need resetting here.
    pub fn reset(&mut self) {
        for h in &mut self.heads {
            h.reset();
        }
    }

    /// KV group index for a given Q head.
    #[inline]
    pub fn kv_group(head_idx: usize, config: &Config) -> usize {
        head_idx * config.n_kv_head / config.n_head
    }
}

// ── Multi-Layer Cache ──────────────────────────────────────────

/// Multi-layer GDN2 cache for recurrent decode.
///
/// Streaming recurrence: constant O(d_k × d_v) per token per head,
/// independent of sequence length. The recurrent update is:
///
/// ```text
/// 1. Decay:  S *= Diag(α)           (row-wise scale by decay)
/// 2. Read:   r = Sᵀ(b ⊙ k)         (gated matvec)
/// 3. Update: S += k ⊗ (w⊙v − r)    (outer product delta)
/// 4. Readout: o = Sᵀ q              (matvec)
/// ```
///
/// With decoupled erase gate b and write gate w, GDN2 achieves
/// fine-grained memory control at O(1) decode cost.
#[derive(Clone)]
pub struct MultiLayerGdn2Cache {
    /// Per-layer state.
    pub layers: Vec<Gdn2LayerState>,
    /// Epsilon for numerical stability (default: 1e-6).
    pub eps: f32,
}

impl MultiLayerGdn2Cache {
    /// Allocate zeroed cache for all layers with default gate config.
    pub fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| Gdn2LayerState::new(config, Gdn2GateConfig::default()))
                .collect(),
            eps: 1e-6,
        }
    }

    /// Allocate with custom gate config for all layers.
    pub fn with_gate_config(config: &Config, gate_config: Gdn2GateConfig) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| Gdn2LayerState::new(config, gate_config))
                .collect(),
            eps: 1e-6,
        }
    }

    /// Reset all layers to zeroed state (reuse allocations).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Total cache size in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            for head in &layer.heads {
                total += head.s.len() * 4;
            }
        }
        total
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn2_head_state_new_reset() {
        let dk = 8;
        let dv = 8;
        let mut state = Gdn2HeadState::new(dk, dv);
        assert_eq!(state.s.len(), dk * dv);
        assert!(state.s.iter().all(|&x| x == 0.0));

        state.s[0] = 1.0;
        state.s[dk * dv - 1] = 2.0;
        state.reset();
        assert!(state.s.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn gdn2_layer_state_gqa_mapping() {
        let config = Config::gqa_draft(); // n_head=8, n_kv_head=2
        // Heads 0-3 → KV group 0; Heads 4-7 → KV group 1
        assert_eq!(Gdn2LayerState::kv_group(0, &config), 0);
        assert_eq!(Gdn2LayerState::kv_group(3, &config), 0);
        assert_eq!(Gdn2LayerState::kv_group(4, &config), 1);
        assert_eq!(Gdn2LayerState::kv_group(7, &config), 1);
    }

    #[test]
    fn gdn2_cache_new_has_correct_layer_count() {
        let config = Config::micro(); // n_layer=1
        let cache = MultiLayerGdn2Cache::new(&config);
        assert_eq!(cache.layers.len(), config.n_layer);
    }

    #[test]
    fn gdn2_cache_memory_bytes() {
        let config = Config::micro(); // hd=4, n_kv_head=4, n_layer=1
        let cache = MultiLayerGdn2Cache::new(&config);
        let bytes = cache.memory_bytes();
        // Per layer: 4 kv_heads × (4 × 4) = 64 floats = 256 bytes
        assert_eq!(bytes, 64 * 4);
    }

    #[test]
    fn gdn2_cache_reset() {
        let config = Config::micro();
        let mut cache = MultiLayerGdn2Cache::new(&config);
        // Mutate some state
        cache.layers[0].heads[0].s[0] = 5.0;
        cache.layers[0].heads[0].s[15] = 3.0;
        cache.reset();
        assert_eq!(cache.layers[0].heads[0].s[0], 0.0);
        assert_eq!(cache.layers[0].heads[0].s[15], 0.0);
    }

    #[test]
    fn gdn2_gate_config_default() {
        assert_eq!(Gdn2GateConfig::default(), Gdn2GateConfig::EraseOnly);
    }

    #[test]
    fn gdn2_cache_with_gate_config() {
        let config = Config::micro();
        let cache = MultiLayerGdn2Cache::with_gate_config(&config, Gdn2GateConfig::Full);
        for layer in &cache.layers {
            assert_eq!(layer.gate_config, Gdn2GateConfig::Full);
        }
    }

    #[test]
    fn gdn2_smaller_than_hla_symmetric() {
        let config = Config::micro(); // hd=4, n_head=4, n_kv_head=4, n_layer=1
        let gdn2_cache = MultiLayerGdn2Cache::new(&config);
        let gdn2_bytes = gdn2_cache.memory_bytes();
        // HLA symmetric: 224 floats = 896 bytes
        // GDN2: 64 floats = 256 bytes
        // GDN2 should be smaller
        assert!(
            gdn2_bytes < 896,
            "GDN2 ({gdn2_bytes}B) should be smaller than HLA symmetric (896B)"
        );
    }
}
