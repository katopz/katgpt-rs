//! Higher-order Linear Attention (HLA) cache types.
//!
//! Implements constant-size O(d²) inference cache for symmetric second-order HLA
//! and O(d·dv) cache for asymmetric AHLA. Replaces growing KV cache with fixed-size
//! prefix sufficient statistics that capture higher-order query-key interactions.
//!
//! Reference: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."
//! See `.research/28_Higher_order_Linear_Attention.md` for full derivation.
//!
//! # GQA Support
//!
//! With Grouped Query Attention (`n_kv_head < n_head`):
//! - Symmetric HLA: SK shared per KV group, CQV/mQ/G/h per Q head
//! - AHLA: PKV/mK shared per KV group, E/n per Q head
//!
//! KV group for Q head `h`: `kv_group = h * n_kv_head / n_head`

use crate::types::Config;

// ── Symmetric Second-Order HLA ────────────────────────────────

/// Per-Q-head state for symmetric second-order HLA.
///
/// Captures query-value interactions and causal corrections.
/// The key second moment (SK) is stored per KV group in [`HlaLayerState`].
///
/// State size per head: 3 × (hd × hd) + 2 × hd = 3hd² + 2hd floats.
#[derive(Clone)]
pub struct HlaQHeadState {
    /// Query-value cross moment: Σ q_i v_iᵀ ∈ R^{hd × hd}
    pub cqv: Vec<f32>,
    /// Query mass: Σ q_i ∈ R^{hd}
    pub mq: Vec<f32>,
    /// Causal correction numerator: Σ k_i (k_iᵀ CQV_{i-1}) ∈ R^{hd × hd}
    pub g: Vec<f32>,
    /// Causal correction denominator: Σ k_i (k_iᵀ mQ_{i-1}) ∈ R^{hd}
    pub h: Vec<f32>,
}

impl HlaQHeadState {
    /// Allocate zeroed state for one Q head.
    pub fn new(hd: usize) -> Self {
        let hd2 = hd * hd;
        Self {
            cqv: vec![0.0; hd2],
            mq: vec![0.0; hd],
            g: vec![0.0; hd2],
            h: vec![0.0; hd],
        }
    }

    /// Reset to zeroed state (reuse allocations).
    pub fn reset(&mut self) {
        self.cqv.fill(0.0);
        self.mq.fill(0.0);
        self.g.fill(0.0);
        self.h.fill(0.0);
    }
}

/// Per-layer state for symmetric second-order HLA.
///
/// GQA-aware layout:
/// - `sk[n_kv_head]`: key second moment shared per KV group
/// - `heads[n_head]`: per-Q-head cross moments (CQV, mQ, G, h)
///
/// Total state per layer: n_kv × hd² + n_head × (3hd² + 2hd).
#[derive(Clone)]
pub struct HlaLayerState {
    /// Key second moment per KV group: Σ k_i k_iᵀ ∈ R^{hd × hd}
    /// Shared across Q heads that map to the same KV group.
    pub sk: Vec<Vec<f32>>,
    /// Per-Q-head state.
    pub heads: Vec<HlaQHeadState>,
}

impl HlaLayerState {
    /// Allocate zeroed state for one layer given config.
    pub fn new(config: &Config) -> Self {
        let hd = config.head_dim;
        let hd2 = hd * hd;
        Self {
            sk: (0..config.n_kv_head).map(|_| vec![0.0; hd2]).collect(),
            heads: (0..config.n_head).map(|_| HlaQHeadState::new(hd)).collect(),
        }
    }

    /// Reset to zeroed state (reuse allocations).
    pub fn reset(&mut self) {
        for sk in &mut self.sk {
            sk.fill(0.0);
        }
        for head in &mut self.heads {
            head.reset();
        }
    }

    /// KV group index for a given Q head.
    #[inline]
    pub fn kv_group(head_idx: usize, config: &Config) -> usize {
        head_idx * config.n_kv_head / config.n_head
    }
}

/// Multi-layer cache for symmetric second-order HLA.
///
/// Streaming recurrence: constant O(d² + d·dv) per token, independent of
/// sequence length. The output is computed as:
///
/// ```text
/// o_t = q_tᵀ (SK_t · CQV_t − G_t) / (q_tᵀ (SK_t · mQ_t − h_t) + ε)
/// ```
///
/// With exponential decay γ, all accumulators are scaled: `A_t = γ·A_{t-1} + Δ`.
#[derive(Clone)]
pub struct MultiLayerHlaCache {
    /// Per-layer state.
    pub layers: Vec<HlaLayerState>,
    /// Exponential decay factor γ ∈ (0, 1]. Default: 1.0 (no decay).
    /// Controls spectral growth and adds recency bias.
    pub gamma: f32,
    /// Epsilon for normalization denominator (default: 1e-6).
    pub eps: f32,
}

impl MultiLayerHlaCache {
    /// Allocate zeroed cache for all layers.
    pub fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| HlaLayerState::new(config))
                .collect(),
            gamma: 1.0,
            eps: 1e-6,
        }
    }

    /// Allocate with custom decay.
    pub fn with_gamma(config: &Config, gamma: f32) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| HlaLayerState::new(config))
                .collect(),
            gamma,
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
            // SK: n_kv_head × hd²
            total += layer.sk.iter().map(|s| s.len() * 4).sum::<usize>();
            // Heads: n_head × (3hd² + 2hd)
            for head in &layer.heads {
                total += head.cqv.len() * 4;
                total += head.mq.len() * 4;
                total += head.g.len() * 4;
                total += head.h.len() * 4;
            }
        }
        total
    }
}

// ── Asymmetric Second-Order HLA (AHLA) ────────────────────────

/// Per-Q-head state for asymmetric second-order HLA (AHLA).
///
/// Captures routed accumulation and denominator for the left-cascaded
/// product A·A·V (as opposed to symmetric A·Aᵀ·V).
///
/// State size per head: hd² + hd = hd(hd+1) floats.
#[derive(Clone)]
pub struct AhlaQHeadState {
    /// Routed accumulation: Σ k_i (q_iᵀ PKV_i) ∈ R^{hd × hd}
    pub e: Vec<f32>,
    /// Denominator accumulator: Σ k_i (q_iᵀ mK_i) ∈ R^{hd}
    pub n: Vec<f32>,
}

impl AhlaQHeadState {
    /// Allocate zeroed state for one Q head.
    pub fn new(hd: usize) -> Self {
        Self {
            e: vec![0.0; hd * hd],
            n: vec![0.0; hd],
        }
    }

    /// Reset to zeroed state (reuse allocations).
    pub fn reset(&mut self) {
        self.e.fill(0.0);
        self.n.fill(0.0);
    }
}

/// Per-layer state for asymmetric AHLA.
///
/// GQA-aware layout:
/// - `pkv[n_kv_head]`: key-value prefix shared per KV group
/// - `mk[n_kv_head]`: key mass shared per KV group
/// - `heads[n_head]`: per-Q-head state (E, n)
///
/// Total state per layer: n_kv × (hd² + hd) + n_head × (hd² + hd).
/// Smaller than symmetric HLA when n_head > n_kv_head (typical GQA).
#[derive(Clone)]
pub struct AhlaLayerState {
    /// Key-value prefix per KV group: Σ k_j v_jᵀ ∈ R^{hd × hd}
    pub pkv: Vec<Vec<f32>>,
    /// Key mass per KV group: Σ k_j ∈ R^{hd}
    pub mk: Vec<Vec<f32>>,
    /// Per-Q-head state.
    pub heads: Vec<AhlaQHeadState>,
}

impl AhlaLayerState {
    /// Allocate zeroed state for one layer given config.
    pub fn new(config: &Config) -> Self {
        let hd = config.head_dim;
        let hd2 = hd * hd;
        Self {
            pkv: (0..config.n_kv_head).map(|_| vec![0.0; hd2]).collect(),
            mk: (0..config.n_kv_head).map(|_| vec![0.0; hd]).collect(),
            heads: (0..config.n_head)
                .map(|_| AhlaQHeadState::new(hd))
                .collect(),
        }
    }

    /// Reset to zeroed state (reuse allocations).
    pub fn reset(&mut self) {
        for pkv in &mut self.pkv {
            pkv.fill(0.0);
        }
        for mk in &mut self.mk {
            mk.fill(0.0);
        }
        for head in &mut self.heads {
            head.reset();
        }
    }

    /// KV group index for a given Q head.
    #[inline]
    pub fn kv_group(head_idx: usize, config: &Config) -> usize {
        head_idx * config.n_kv_head / config.n_head
    }
}

/// Multi-layer cache for asymmetric AHLA.
///
/// Streaming recurrence: constant O(d·dv) per token.
/// The output is computed as:
///
/// ```text
/// o_t = q_tᵀ E_t / (q_tᵀ n_t + ε)
/// ```
///
/// AHLA routes value through key index i: left-cascaded A·A·V,
/// providing second-order interactions at linear attention cost.
#[derive(Clone)]
pub struct MultiLayerAhlaCache {
    /// Per-layer state.
    pub layers: Vec<AhlaLayerState>,
    /// Exponential decay factor γ ∈ (0, 1]. Default: 1.0 (no decay).
    pub gamma: f32,
    /// Epsilon for normalization denominator (default: 1e-6).
    pub eps: f32,
}

impl MultiLayerAhlaCache {
    /// Allocate zeroed cache for all layers.
    pub fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| AhlaLayerState::new(config))
                .collect(),
            gamma: 1.0,
            eps: 1e-6,
        }
    }

    /// Allocate with custom decay.
    pub fn with_gamma(config: &Config, gamma: f32) -> Self {
        Self {
            layers: (0..config.n_layer)
                .map(|_| AhlaLayerState::new(config))
                .collect(),
            gamma,
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
            // PKV + MK: n_kv_head × (hd² + hd)
            for pkv in &layer.pkv {
                total += pkv.len() * 4;
            }
            for mk in &layer.mk {
                total += mk.len() * 4;
            }
            // Heads: n_head × (hd² + hd)
            for head in &layer.heads {
                total += head.e.len() * 4;
                total += head.n.len() * 4;
            }
        }
        total
    }
}

// ── Memory Comparison Helper ──────────────────────────────────

/// Cache variant for benchmark comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HlaVariant {
    /// Symmetric second-order: A·Aᵀ·V, state 3hd² + 2hd per Q head + hd² per KV group.
    Symmetric,
    /// Asymmetric (AHLA): A·A·V, state hd² + hd per Q head + hd² + hd per KV group.
    Asymmetric,
}

impl HlaVariant {
    /// State size in floats per Q head for this variant.
    pub fn floats_per_q_head(self, hd: usize) -> usize {
        match self {
            Self::Symmetric => 3 * hd * hd + 2 * hd,
            Self::Asymmetric => hd * hd + hd,
        }
    }

    /// State size in floats per KV group for this variant.
    pub fn floats_per_kv_group(self, hd: usize) -> usize {
        match self {
            Self::Symmetric => hd * hd,
            Self::Asymmetric => hd * hd + hd,
        }
    }

    /// Total state size in bytes for one layer.
    pub fn layer_bytes(self, config: &Config) -> usize {
        let hd = config.head_dim;
        let q_total = config.n_head * self.floats_per_q_head(hd);
        let kv_total = config.n_kv_head * self.floats_per_kv_group(hd);
        (q_total + kv_total) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hla_head_state_new_reset() {
        let hd = 8;
        let mut state = HlaQHeadState::new(hd);
        assert_eq!(state.cqv.len(), hd * hd);
        assert_eq!(state.mq.len(), hd);
        assert_eq!(state.g.len(), hd * hd);
        assert_eq!(state.h.len(), hd);
        assert!(state.cqv.iter().all(|&x| x == 0.0));

        state.cqv[0] = 1.0;
        state.reset();
        assert!(state.cqv.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ahla_head_state_new_reset() {
        let hd = 4;
        let mut state = AhlaQHeadState::new(hd);
        assert_eq!(state.e.len(), hd * hd);
        assert_eq!(state.n.len(), hd);
        assert!(state.e.iter().all(|&x| x == 0.0));

        state.e[0] = 2.0;
        state.reset();
        assert!(state.e.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hla_layer_state_gqa_mapping() {
        let config = Config::gqa_draft(); // n_head=8, n_kv_head=2
        // Heads 0-3 → KV group 0; Heads 4-7 → KV group 1
        assert_eq!(HlaLayerState::kv_group(0, &config), 0);
        assert_eq!(HlaLayerState::kv_group(3, &config), 0);
        assert_eq!(HlaLayerState::kv_group(4, &config), 1);
        assert_eq!(HlaLayerState::kv_group(7, &config), 1);
    }

    #[test]
    fn hla_cache_memory_bytes() {
        let config = Config::micro(); // hd=4, n_head=4, n_kv_head=4, n_layer=1
        let cache = MultiLayerHlaCache::new(&config);
        let bytes = cache.memory_bytes();
        // Per layer: 4 kv_groups × 16 (sk) + 4 heads × (2×16 + 2×4) = 64 + 160 = 224 floats
        assert_eq!(bytes, 224 * 4); // 896 bytes
    }

    #[test]
    fn ahla_cache_memory_bytes() {
        let config = Config::micro();
        let cache = MultiLayerAhlaCache::new(&config);
        let bytes = cache.memory_bytes();
        // Per layer: 4 kv_groups × (16+4) + 4 heads × (16+4) = 80 + 80 = 160 floats
        assert_eq!(bytes, 160 * 4);
    }

    #[test]
    fn ahla_smaller_than_symmetric() {
        let config = Config::bpe(); // hd=8
        let sym = HlaVariant::Symmetric.layer_bytes(&config);
        let asym = HlaVariant::Asymmetric.layer_bytes(&config);
        assert!(
            asym < sym,
            "AHLA ({asym}) should be smaller than symmetric ({sym})"
        );
    }

    #[test]
    fn hla_cache_reset() {
        let config = Config::micro();
        let mut cache = MultiLayerHlaCache::new(&config);
        // Mutate some state
        cache.layers[0].sk[0][0] = 5.0;
        cache.layers[0].heads[0].cqv[0] = 3.0;
        cache.reset();
        assert_eq!(cache.layers[0].sk[0][0], 0.0);
        assert_eq!(cache.layers[0].heads[0].cqv[0], 0.0);
    }

    #[test]
    fn gamma_default_no_decay() {
        let config = Config::micro();
        let cache = MultiLayerHlaCache::new(&config);
        assert_eq!(cache.gamma, 1.0);
        let cache_ahla = MultiLayerAhlaCache::new(&config);
        assert_eq!(cache_ahla.gamma, 1.0);
    }
}
