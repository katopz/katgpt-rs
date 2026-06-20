//! FreqBandit — Oscillatory State-Space Modelless Distillation (Phase 1, Plan 189)
//!
//! Uses spectral analysis of recent token streams to select speculative decode parameters.
//! Three frequency bands map to distinct draft-tree configurations:
//!
//! - **Low** (period > 16 tokens) → large draft tree, deep lookahead
//! - **Mid** (period 4–16 tokens) → balanced
//! - **High** (period < 4 tokens) → shallow draft tree, more verify iterations
//!
//! # Architecture
//!
//! - [`FrequencyProfile`] — spectral analysis result from a token window
//! - [`FrequencyBand`] — three-band classification (Low/Mid/High)
//! - [`FrequencyBandit`] — UCB1 bandit selecting bands based on reward
//! - [`SpecBandConfig`] — speculative decode parameters per band
//!
//! # Reward Signal
//!
//! `reward = acceptance_rate × latency_improvement` from speculative decode.
//! Uses **sigmoid** activation (NOT softmax) per project constraint.

use crate::trigger_gate::ComputeTier;
use crate::types::Rng;

// ── Frequency Band ─────────────────────────────────────────────

/// Temporal frequency band classification.
///
/// Determined by dominant period in the token stream's spectral profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrequencyBand {
    /// Long-range patterns (>16 token period). Maps to large draft tree.
    Low = 0,
    /// Medium patterns (4–16 token period). Balanced config.
    Mid = 1,
    /// Short-range patterns (<4 token period). Shallow draft tree.
    High = 2,
}

impl FrequencyBand {
    /// Number of arms in the frequency bandit.
    pub const NUM_ARMS: usize = 3;

    /// Convert to arm index.
    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }

    /// Convert from arm index. Returns `None` if out of range.
    #[inline]
    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Low),
            1 => Some(Self::Mid),
            2 => Some(Self::High),
            _ => None,
        }
    }

    /// Recommended compute tier for this frequency band.
    ///
    /// High-frequency decode → prefer GPU (faster verify iterations).
    /// Low-frequency decode → CPU acceptable (longer draft OK).
    pub fn recommended_tier(self) -> ComputeTier {
        match self {
            Self::Low => ComputeTier::CpuOnly,    // Deep draft OK on CPU
            Self::Mid => ComputeTier::CpuGpu,     // Balanced, GPU helps
            Self::High => ComputeTier::CpuGpuAne, // Fast verify needs all hardware
        }
    }
}

// ── Frequency Profile ──────────────────────────────────────────

/// Spectral analysis result for a token window.
///
/// Contains band energies, dominant band classification, and spectral entropy.
#[derive(Clone, Debug)]
pub struct FrequencyProfile {
    /// Dominant frequency band.
    pub dominant_band: FrequencyBand,
    /// Energy in each band: [low, mid, high].
    pub band_energies: [f32; 3],
    /// Spectral entropy (0.0 = pure tone, 1.0 = white noise).
    pub spectral_entropy: f32,
}

/// Analyze the spectral content of a token stream window.
///
/// Uses a simple DFT dot-product approach for small windows (N ≤ 128).
/// Computes energy in three frequency bands:
/// - Low: periods > 16 tokens (k = 1..N/16)
/// - Mid: periods 4–16 tokens (k = N/16..N/4)
/// - High: periods < 4 tokens (k = N/4..N/2)
///
/// Only analyzes the last `window_size` tokens from the input.
pub fn token_stream_spectrum(tokens: &[usize], window_size: usize) -> FrequencyProfile {
    let window = if tokens.len() > window_size {
        &tokens[tokens.len() - window_size..]
    } else {
        tokens
    };

    let n = window.len();
    if n < 4 {
        return FrequencyProfile {
            dominant_band: FrequencyBand::Low,
            band_energies: [1.0, 0.0, 0.0],
            spectral_entropy: 0.0,
        };
    }

    // Pre-compute DC-removed signal once (saves repeated per-iter subtraction
    // and the per-token `as f32` cast inside the DFT inner loop).
    //
    // Compute mean inline for DC removal; the mean itself is not needed
    // afterwards — only the centered signal `sig` is used in the DFT.
    let mean = window.iter().map(|&t| t as f32).sum::<f32>() / n as f32;
    let mut sig = Vec::with_capacity(n);
    sig.extend(window.iter().map(|&t| t as f32 - mean));
    let _ = mean; // captured into sig above; documented for clarity.

    // DFT coefficients — only need up to Nyquist (n/2)
    let max_k = n / 2;
    let mut magnitudes = vec![0.0f32; max_k + 1];

    // DC component (k=0) is discarded after mean removal
    magnitudes[0] = 0.0;

    // Compute |X[k]|² for k=1..max_k using the rotation recurrence instead of
    // per-element cos/sin calls.
    //
    // For fixed k, the sequence angle[i] = (2π k / n) · i increments by a
    // constant Δ = 2π k / n per index. Using the rotation identities
    //   cos(a+Δ) = cos(a)cos(Δ) − sin(a)sin(Δ)
    //   sin(a+Δ) = sin(a)cos(Δ) + cos(a)sin(Δ)
    // we only need 2 transcendentals per k (cos(Δ), sin(Δ)) instead of 2n,
    // reducing total transcendental cost from O(n²/2) to O(n/2).
    //
    // The recurrence is numerically stable for the short windows we care
    // about here (n ≤ window_size, typically ≤ 256). For very long windows
    // we'd periodically renormalize, but that's not needed at this scale.
    let two_pi_over_n = 2.0 * std::f32::consts::PI / n as f32;

    for k in 1..=max_k {
        let delta = two_pi_over_n * k as f32;
        let cos_d = delta.cos();
        let sin_d = delta.sin();

        // Start at angle 0: cos=1, sin=0. After first iteration we'll have
        // rotated by Δ, giving angle = Δ·1 as expected.
        let mut cos_a = 1.0f32;
        let mut sin_a = 0.0f32;
        let mut re = 0.0f32;
        let mut im = 0.0f32;

        for i in 0..n {
            let x = unsafe { *sig.get_unchecked(i) };
            // FMA contractions for re/im accumulation (inner loop of O(n²) DFT).
            re = x.mul_add(cos_a, re);
            im = (-x).mul_add(sin_a, im);

            // Rotate to next angle: (cos_a, sin_a) ← (cos_a·cos_d − sin_a·sin_d,
            // sin_a·cos_d + cos_a·sin_d). Two FMAs each.
            let new_cos = cos_a * cos_d - sin_a * sin_d;
            let new_sin = sin_a * cos_d + cos_a * sin_d;
            cos_a = new_cos;
            sin_a = new_sin;
        }

        unsafe {
            // FMA: re² + im² in a single rounding.
            *magnitudes.get_unchecked_mut(k) = re.mul_add(re, im * im) / (n as f32);
        }
    }

    // Band boundaries (in terms of frequency index k)
    // Low: long period (>16 tokens) → k ∈ [1, n/16]
    // Mid: medium period (4-16 tokens) → k ∈ (n/16, n/4]
    // High: short period (<4 tokens) → k ∈ (n/4, n/2]
    let low_end = (n / 16).max(1);
    let mid_end = (n / 4).max(low_end + 1);

    let mut low_energy = 0.0f32;
    let mut mid_energy = 0.0f32;
    let mut high_energy = 0.0f32;

    #[allow(clippy::needless_range_loop)]
    for k in 1..=max_k {
        let e = magnitudes[k];
        if k <= low_end {
            low_energy += e;
        } else if k <= mid_end {
            mid_energy += e;
        } else {
            high_energy += e;
        }
    }

    let total_energy = low_energy + mid_energy + high_energy;
    let band_energies = if total_energy > 1e-10 {
        [low_energy, mid_energy, high_energy]
    } else {
        // Flat/constant signal → DC → Low band
        return FrequencyProfile {
            dominant_band: FrequencyBand::Low,
            band_energies: [1.0, 0.0, 0.0],
            spectral_entropy: 0.0,
        };
    };

    // Dominant band
    let dominant_band = match (
        low_energy >= mid_energy && low_energy >= high_energy,
        mid_energy >= high_energy,
    ) {
        (true, _) => FrequencyBand::Low,
        (false, true) => FrequencyBand::Mid,
        (false, false) => FrequencyBand::High,
    };

    // Spectral entropy: H = -Σ p_k ln(p_k) / ln(K), normalized to [0,1]
    let mut entropy = 0.0f32;
    for &mag in magnitudes.iter().skip(1) {
        let p = mag / total_energy;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    let max_entropy = (max_k as f32).ln();
    let spectral_entropy = if max_entropy > 1e-10 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    FrequencyProfile {
        dominant_band,
        band_energies,
        spectral_entropy,
    }
}

// ── Sigmoid Activation ─────────────────────────────────────────

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
///
/// Used for all activation in this module. NOT softmax per project constraint.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid to band energies to get activation weights.
///
/// Each band gets an independent sigmoid gate: σ(energy_i).
/// This is NOT softmax — the weights don't sum to 1.
#[inline]
pub fn sigmoid_band_weights(band_energies: &[f32; 3]) -> [f32; 3] {
    [
        sigmoid(band_energies[0]),
        sigmoid(band_energies[1]),
        sigmoid(band_energies[2]),
    ]
}

// ── Spec Band Config ───────────────────────────────────────────

/// Speculative decode configuration mapped from a frequency band.
///
/// Each band maps to distinct draft-tree parameters optimized for that
/// temporal pattern regime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecBandConfig {
    /// Number of parallel branches at each draft-tree level.
    pub draft_tree_width: usize,
    /// Maximum depth of the draft tree (lookahead horizon).
    pub draft_tree_depth: usize,
    /// Number of verification iterations against the target model.
    pub verify_iterations: usize,
}

impl FrequencyBand {
    /// Map this band to speculative decode parameters.
    pub fn spec_config(self) -> SpecBandConfig {
        match self {
            // Long-range patterns: invest in deep speculative lookahead
            Self::Low => SpecBandConfig {
                draft_tree_width: 5,
                draft_tree_depth: 8,
                verify_iterations: 1,
            },
            // Medium patterns: balanced config
            Self::Mid => SpecBandConfig {
                draft_tree_width: 4,
                draft_tree_depth: 5,
                verify_iterations: 2,
            },
            // Short-range patterns: shallow tree, more verify passes
            Self::High => SpecBandConfig {
                draft_tree_width: 3,
                draft_tree_depth: 3,
                verify_iterations: 3,
            },
        }
    }
}

// ── Frequency Bandit ───────────────────────────────────────────

/// UCB1 frequency bandit for adaptive speculative decode parameter selection.
///
/// Arms: {Low, Mid, High} frequency bands.
/// Reward: `acceptance_rate × latency_improvement`.
///
/// Uses incremental Q-value update: `Q(a) += (r - Q(a)) / n(a)`.
/// UCB1 selection: `Q(a) + sqrt(2 * ln(N) / n(a))`.
pub struct FrequencyBandit {
    /// Q-value estimates for each arm [Low, Mid, High].
    arm_q_values: [f64; 3],
    /// Pull counts for each arm.
    arm_counts: [u32; 3],
    /// UCB1 exploration constant. Default: sqrt(2).
    exploration_c: f32,
    /// Total pulls across all arms.
    total_pulls: u32,
    /// Last selected band (for reward attribution).
    last_selected: Option<FrequencyBand>,
}

impl FrequencyBandit {
    /// Create a new frequency bandit with default exploration constant.
    pub fn new() -> Self {
        Self {
            arm_q_values: [0.0; 3],
            arm_counts: [0; 3],
            exploration_c: 2.0f32.sqrt(),
            total_pulls: 0,
            last_selected: None,
        }
    }

    /// Create with custom exploration constant.
    pub fn with_exploration(mut self, c: f32) -> Self {
        self.exploration_c = c;
        self
    }

    /// Select a frequency band using UCB1.
    ///
    /// Unvisited arms are prioritized (score = +∞).
    /// For visited arms: `Q(a) + c * sqrt(ln(N) / n(a))`.
    pub fn select_band(&mut self, _rng: &mut Rng) -> FrequencyBand {
        // First pass: any unvisited arm gets priority
        for i in 0..3 {
            if self.arm_counts[i] == 0 {
                let band = FrequencyBand::from_index(i).unwrap();
                self.last_selected = Some(band);
                return band;
            }
        }

        // All arms visited: pick highest UCB1 score.
        // Hoist invariants out of the 3-arm loop: `ln(total)` and the
        // exploration constant only depend on total pulls, not on the arm.
        let ln_total = (self.total_pulls as f64).ln();
        let c = self.exploration_c as f64;
        let coef = c * (2.0 * ln_total).sqrt();

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..3 {
            let q = self.arm_q_values[i];
            // exploration = c * sqrt(2 * ln(N) / n)
            //             = (c * sqrt(2 * ln(N))) / sqrt(n)
            // The numerator is loop-invariant; only `1/sqrt(n)` varies per arm.
            let exploration = coef / (self.arm_counts[i] as f64).sqrt();
            let score = q + exploration;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let band = FrequencyBand::from_index(best_idx).unwrap();
        self.last_selected = Some(band);
        band
    }

    /// Update Q-value for a band after observing reward.
    ///
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    pub fn update(&mut self, band: FrequencyBand, reward: f64) {
        let i = band.as_index();
        self.arm_counts[i] += 1;
        self.total_pulls += 1;
        // Pre-compute the reciprocal so we multiply instead of divide below.
        let inv_n = 1.0 / self.arm_counts[i] as f64;
        self.arm_q_values[i] += (reward - self.arm_q_values[i]) * inv_n;
    }

    /// Map a frequency band to speculative decode configuration.
    #[inline]
    pub fn map_to_spec_config(&self, band: FrequencyBand) -> SpecBandConfig {
        band.spec_config()
    }

    /// Q-value for a given band.
    #[inline]
    pub fn q_value(&self, band: FrequencyBand) -> f64 {
        self.arm_q_values[band.as_index()]
    }

    /// Pull count for a given band.
    #[inline]
    pub fn count(&self, band: FrequencyBand) -> u32 {
        self.arm_counts[band.as_index()]
    }

    /// Total pulls across all arms.
    #[inline]
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Last selected band.
    #[inline]
    pub fn last_selected(&self) -> Option<FrequencyBand> {
        self.last_selected
    }

    /// Best arm by Q-value.
    pub fn best_arm(&self) -> FrequencyBand {
        let mut best = 0;
        let mut best_q = f64::NEG_INFINITY;
        for i in 0..3 {
            if self.arm_q_values[i] > best_q {
                best_q = self.arm_q_values[i];
                best = i;
            }
        }
        FrequencyBand::from_index(best).unwrap()
    }

    /// Get compute tier recommendation based on current best arm.
    /// Uses the band with highest cumulative Q-value.
    pub fn tier_recommendation(&self) -> ComputeTier {
        self.best_arm().recommended_tier()
    }

    // ── RV Bandit Pruning (Plan 202, Phase 4) ─────────────────────

    /// Compute per-arm reward variance using Welford online statistics.
    ///
    /// Returns `[f64; 3]` variances for [Low, Mid, High] arms.
    /// Arms with < 2 samples have variance = 0.0.
    #[cfg(feature = "rv_bandit_pruning")]
    pub fn arm_variances(&self) -> [f64; 3] {
        // Welford online variance from Q-value updates.
        // Since we only track incremental mean (Q(a)), not raw samples,
        // we use Q-value spread as a proxy: variance ≈ (Q(a) - Q_mean)².
        // Preserve original semantics: exclude zero Q-values from the mean.
        let mut q_sum = 0.0_f64;
        for &q in &self.arm_q_values {
            if q != 0.0 {
                q_sum += q;
            }
        }
        let q_mean = q_sum / 3.0_f64;
        let mut variances = [0.0; 3];
        for (i, &q) in self.arm_q_values.iter().enumerate() {
            match self.arm_counts[i] {
                0 | 1 => variances[i] = 0.0,
                _ => {
                    let delta = q - q_mean;
                    variances[i] = delta * delta;
                }
            }
        }
        variances
    }

    /// Suppress arms below the `(1 - ρ)` quantile of variance.
    ///
    /// RAGEN-2 proves nucleus-style filtering > top-k > no filter.
    /// Suppressed arms are penalized (Q-value set to -∞) so UCB1 avoids them.
    ///
    /// # Arguments
    /// * `rho` — retention fraction (0.0–1.0). 1.0 = no suppression.
    #[cfg(feature = "rv_bandit_pruning")]
    pub fn suppress_low_rv_arms(&mut self, rho: f32) {
        let variances = self.arm_variances();

        // Compute threshold at (1 - ρ) quantile
        let threshold = match rho {
            rho if rho >= 1.0 => return,        // No suppression
            rho if rho <= 0.0 => f64::INFINITY, // Suppress all
            _ => {
                // Sort variances to find quantile
                let mut sorted = variances;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                // (1 - ρ) quantile index
                let idx = ((1.0 - rho) * 2.0) as usize;
                sorted[idx.min(2)]
            }
        };

        // Suppress arms below threshold
        for (i, &v) in variances.iter().enumerate() {
            if v < threshold && self.arm_counts[i] > 0 {
                self.arm_q_values[i] = f64::NEG_INFINITY;
            }
        }
    }

    /// Check if an arm is suppressed (Q = -∞).
    #[cfg(feature = "rv_bandit_pruning")]
    pub fn is_arm_suppressed(&self, band: FrequencyBand) -> bool {
        self.arm_q_values[band.as_index()] == f64::NEG_INFINITY
    }

    /// Unsuppress all arms (reset Q-values from -∞ to 0).
    #[cfg(feature = "rv_bandit_pruning")]
    pub fn unsuppress_all(&mut self) {
        for q in &mut self.arm_q_values {
            if *q == f64::NEG_INFINITY {
                *q = 0.0;
            }
        }
    }
}

impl Default for FrequencyBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── FreqTierAdapter ────────────────────────────────────────────

/// Adapter: feeds FreqBandit's spectral analysis into compute tier selection.
///
/// Analyzes token streams via spectral methods and translates the dominant
/// frequency band into a [`ComputeTier`] recommendation suitable for
/// [`InferenceRouter`](crate::inference_router::InferenceRouter).
pub struct FreqTierAdapter {
    bandit: FrequencyBandit,
}

impl FreqTierAdapter {
    /// Create a new adapter wrapping an existing [`FrequencyBandit`].
    pub fn new(bandit: FrequencyBandit) -> Self {
        Self { bandit }
    }

    /// Analyze token stream and return tier recommendation.
    pub fn recommend_tier(&mut self, tokens: &[usize], window_size: usize) -> ComputeTier {
        let profile = token_stream_spectrum(tokens, window_size);
        // Select the band and immediately query tier recommendation.
        // Note: select requires an rng for UCB1 exploration, but we only
        // need the deterministic tier mapping from the dominant band.
        profile.dominant_band.recommended_tier()
    }

    /// Update the bandit with reward signal from last decode.
    pub fn update_reward(&mut self, reward: f32) {
        let band = self.bandit.last_selected();
        if let Some(b) = band {
            self.bandit.update(b, reward as f64)
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> Rng {
        Rng::new(42)
    }

    // ── Spectral Analysis Tests ─────────────────────────────

    #[test]
    fn test_frequency_profile_cyclic() {
        // Repeated pattern (0,1,0,1,...) → High band (period = 2)
        let tokens: Vec<usize> = (0..64).map(|i| i % 2).collect();
        let profile = token_stream_spectrum(&tokens, 64);

        assert_eq!(
            profile.dominant_band,
            FrequencyBand::High,
            "Cyclic 0,1 pattern should be High band (period=2 < 4)"
        );
        // High energy should dominate
        assert!(
            profile.band_energies[2] > profile.band_energies[0],
            "High band energy should exceed Low for cyclic pattern"
        );
    }

    #[test]
    fn test_frequency_profile_flat() {
        // All same token → constant = DC → Low band
        let tokens: Vec<usize> = vec![42; 64];
        let profile = token_stream_spectrum(&tokens, 64);

        assert_eq!(
            profile.dominant_band,
            FrequencyBand::Low,
            "Constant signal should be Low band (DC only)"
        );
        // Very low spectral entropy for constant signal
        assert!(
            profile.spectral_entropy < 0.5,
            "Constant signal should have low entropy"
        );
    }

    #[test]
    fn test_frequency_profile_random() {
        // Random tokens → energy spread across bands
        let mut rng = make_rng();
        let tokens: Vec<usize> = (0..128)
            .map(|_| (rng.uniform() * 1000.0) as usize)
            .collect();
        let profile = token_stream_spectrum(&tokens, 128);

        // Random signal should have higher spectral entropy
        assert!(
            profile.spectral_entropy > 0.1,
            "Random signal should have non-trivial spectral entropy, got {}",
            profile.spectral_entropy
        );
        // All energies should be non-negative
        for &e in &profile.band_energies {
            assert!(e >= 0.0, "Band energy should be non-negative");
        }
    }

    #[test]
    fn test_frequency_profile_short_window() {
        // Less than 4 tokens → fallback to Low
        let tokens = vec![1, 2, 3];
        let profile = token_stream_spectrum(&tokens, 64);
        assert_eq!(profile.dominant_band, FrequencyBand::Low);
    }

    #[test]
    fn test_frequency_profile_mid_period() {
        // Pattern with period 8 → Mid band
        let tokens: Vec<usize> = (0..64).map(|i| i % 8).collect();
        let profile = token_stream_spectrum(&tokens, 64);

        assert_eq!(
            profile.dominant_band,
            FrequencyBand::Mid,
            "Period-8 pattern should be Mid band (4 ≤ 8 ≤ 16)"
        );
    }

    #[test]
    fn test_frequency_profile_low_period() {
        // Very slowly varying: period = 32 → Low band
        let tokens: Vec<usize> = (0..64).map(|i| i / 32).collect();
        let profile = token_stream_spectrum(&tokens, 64);

        assert_eq!(
            profile.dominant_band,
            FrequencyBand::Low,
            "Period-32+ pattern should be Low band"
        );
    }

    // ── Sigmoid Tests ───────────────────────────────────────

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-100.0) < 0.001);
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_not_softmax() {
        // Sigmoid weights are independent — they do NOT sum to 1
        let energies = [1.0, 2.0, 3.0];
        let weights = sigmoid_band_weights(&energies);

        // Each weight in (0, 1)
        for &w in &weights {
            assert!((0.0..1.0).contains(&w), "sigmoid weight should be in (0,1)");
        }

        // Sum should NOT be 1.0 (that would be softmax)
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() > 0.01,
            "sigmoid weights should NOT sum to 1.0 (that would be softmax), got sum={}",
            sum
        );

        // Verify: sigmoid is monotonic → higher input → higher output
        assert!(weights[0] < weights[1]);
        assert!(weights[1] < weights[2]);
    }

    // ── Bandit Tests ────────────────────────────────────────

    #[test]
    fn test_bandit_cold_start_explores_all() {
        let mut bandit = FrequencyBandit::new();
        let mut rng = make_rng();
        let mut visited = [false; 3];

        // UCB1 should explore all 3 arms in first 3 pulls
        for _ in 0..3 {
            let band = bandit.select_band(&mut rng);
            visited[band.as_index()] = true;
            bandit.update(band, 0.5);
        }

        assert!(
            visited.iter().all(|&v| v),
            "UCB1 should explore all arms in first 3 pulls"
        );
    }

    #[test]
    fn test_bandit_selection_convergence() {
        let mut bandit = FrequencyBandit::new();
        let mut rng = make_rng();

        // Simulate: Mid band always gets high reward, others get low
        for _ in 0..200 {
            let band = bandit.select_band(&mut rng);
            let reward = match band {
                FrequencyBand::Mid => 0.95,
                _ => 0.1,
            };
            bandit.update(band, reward);
        }

        // After convergence, best arm should be Mid
        assert_eq!(
            bandit.best_arm(),
            FrequencyBand::Mid,
            "After 200 episodes, bandit should converge to Mid (highest reward)"
        );
        assert!(
            bandit.q_value(FrequencyBand::Mid) > bandit.q_value(FrequencyBand::Low),
            "Mid Q-value should exceed Low"
        );
    }

    #[test]
    fn test_bandit_update_incremental_mean() {
        let mut bandit = FrequencyBandit::new();

        bandit.update(FrequencyBand::Low, 1.0);
        assert!((bandit.q_value(FrequencyBand::Low) - 1.0).abs() < 1e-10);

        bandit.update(FrequencyBand::Low, 0.0);
        assert!(
            (bandit.q_value(FrequencyBand::Low) - 0.5).abs() < 1e-10,
            "Q-value should be incremental mean"
        );

        bandit.update(FrequencyBand::Low, 1.0);
        assert!((bandit.q_value(FrequencyBand::Low) - 2.0 / 3.0).abs() < 1e-10);
    }

    // ── Spec Config Tests ───────────────────────────────────

    #[test]
    fn test_spec_config_mapping_distinct() {
        let low = FrequencyBand::Low.spec_config();
        let mid = FrequencyBand::Mid.spec_config();
        let high = FrequencyBand::High.spec_config();

        // Each config should be distinct
        assert_ne!(low, mid, "Low and Mid configs should differ");
        assert_ne!(mid, high, "Mid and High configs should differ");
        assert_ne!(low, high, "Low and High configs should differ");
    }

    #[test]
    fn test_spec_config_low_deep_tree() {
        let config = FrequencyBand::Low.spec_config();
        assert!(
            config.draft_tree_depth > FrequencyBand::Mid.spec_config().draft_tree_depth,
            "Low band should have deeper draft tree than Mid"
        );
        assert!(
            config.verify_iterations < FrequencyBand::High.spec_config().verify_iterations,
            "Low band should have fewer verify iterations than High"
        );
    }

    #[test]
    fn test_spec_config_high_shallow_tree() {
        let config = FrequencyBand::High.spec_config();
        assert!(
            config.draft_tree_depth < FrequencyBand::Mid.spec_config().draft_tree_depth,
            "High band should have shallower draft tree than Mid"
        );
        assert!(
            config.verify_iterations > FrequencyBand::Low.spec_config().verify_iterations,
            "High band should have more verify iterations than Low"
        );
    }

    #[test]
    fn test_bandit_map_to_spec_config() {
        let bandit = FrequencyBandit::new();
        let config = bandit.map_to_spec_config(FrequencyBand::Mid);
        assert_eq!(config, FrequencyBand::Mid.spec_config());
    }

    // ── Integration Test ────────────────────────────────────

    #[test]
    fn test_full_pipeline() {
        let mut bandit = FrequencyBandit::new();
        let mut rng = make_rng();

        // Simulate a few episodes: spectral analysis → bandit select → reward
        for ep in 0..50 {
            // Generate a token stream with known pattern
            let tokens: Vec<usize> = (0..64)
                .map(|i| {
                    if ep < 25 {
                        i % 2 // High freq for first half
                    } else {
                        i / 32 // Low freq for second half
                    }
                })
                .collect();

            let profile = token_stream_spectrum(&tokens, 64);
            let band = bandit.select_band(&mut rng);

            // Reward: higher if bandit band matches spectral band
            let reward = if band == profile.dominant_band {
                0.9
            } else {
                0.2
            };

            bandit.update(band, reward);
        }

        // Bandit should have learned something
        assert!(bandit.total_pulls() == 50);
        assert!(
            bandit.q_value(bandit.best_arm()) > 0.0,
            "Best arm should have positive Q-value"
        );
    }

    #[test]
    fn test_frequency_band_roundtrip() {
        for i in 0..3 {
            let band = FrequencyBand::from_index(i).unwrap();
            assert_eq!(band.as_index(), i);
        }
        assert!(FrequencyBand::from_index(3).is_none());
    }

    #[test]
    fn test_freq_band_recommended_tier() {
        assert_eq!(FrequencyBand::Low.recommended_tier(), ComputeTier::CpuOnly);
        assert_eq!(FrequencyBand::Mid.recommended_tier(), ComputeTier::CpuGpu);
        assert_eq!(
            FrequencyBand::High.recommended_tier(),
            ComputeTier::CpuGpuAne
        );
    }

    #[test]
    fn test_freq_tier_adapter_low_freq() {
        // Cyclic pattern with period ~32 (low frequency)
        let tokens: Vec<usize> = (0..128).map(|i| (i / 32) % 4).collect();
        let mut adapter = FreqTierAdapter::new(FrequencyBandit::new());
        let tier = adapter.recommend_tier(&tokens, 128);
        assert_eq!(tier, ComputeTier::CpuOnly);
    }

    #[test]
    fn test_freq_tier_adapter_high_freq() {
        // Cyclic pattern with period 2 (high frequency)
        let tokens: Vec<usize> = (0..128).map(|i| i % 2).collect();
        let mut adapter = FreqTierAdapter::new(FrequencyBandit::new());
        let tier = adapter.recommend_tier(&tokens, 128);
        assert_eq!(tier, ComputeTier::CpuGpuAne);
    }
}

// TL;DR: FreqBandit (Plan 189 Phase 1) — spectral token-stream analysis → 3-arm UCB1 bandit → speculative decode config.
// DFT dot-product for small windows, sigmoid activation (NOT softmax), maps Low/Mid/High bands to draft tree parameters.
// FreqTierAdapter bridges FrequencyBandit → ComputeTier for InferenceRouter integration.
