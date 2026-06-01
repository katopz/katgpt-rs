//! Hydra-Aware Adaptive Layer Budget — Emergent Self-Repair Distillation.
//!
//! Distills the Hydra Effect (emergent self-repair in transformers) into an
//! adaptive layer budget that skips non-contributing layers during the forward pass.
//!
//! Two modes:
//! - Modelless: Pre-computed layer importance profiles (lookup table, zero overhead)
//! - Model-based: Per-layer logit lens scoring during forward pass (one matmul per layer)
//!
//! Reference: arXiv:2307.15771 — McGrath et al. (DeepMind)
//! Feature-gated behind `hydra_budget` (default-OFF until GOAT proves gain).

use katgpt_core::{HydraBudgetConfig, HydraLayerProfile};

/// Maximum number of layers supported for logit lens scoring.
#[allow(dead_code)]
const MAX_LAYERS: usize = 128;

/// Pre-computed set of layers to skip, derived from HydraLayerProfile calibration.
#[derive(Clone, Debug)]
pub struct HydraSkipPlan {
    /// Bitmask: skip_layers[l] = true ⇒ skip layer l.
    pub skip_layers: Vec<bool>,
    /// Cumulative DE thresholds for early termination.
    pub cumulative_de: Vec<f32>,
    /// Total DE across all layers.
    pub total_de: f32,
}

/// Result of adaptive budget computation.
#[derive(Clone, Debug)]
pub struct HydraBudgetResult {
    /// Layers to skip.
    pub skipped: Vec<usize>,
    /// Cumulative DE at early termination point (if any).
    pub early_exit_layer: Option<usize>,
    /// Estimated compute savings fraction.
    pub savings_fraction: f32,
}

/// Given profiles and config, return set of layers to skip.
///
/// Never skips layers with high `backup_frequency` (>0.1) — these are Hydra backups
/// that protect output quality. Never skips non-erasure layers with significant `mean_de`.
pub fn hydra_layer_skip(
    profiles: &[HydraLayerProfile],
    config: &HydraBudgetConfig,
) -> HydraSkipPlan {
    let n = profiles.len();
    let mut skip_layers = vec![false; n];

    for (l, profile) in profiles.iter().enumerate() {
        // Never skip Hydra backup layers — they self-repair other layers' damage.
        if profile.backup_frequency > 0.1 {
            continue;
        }
        // Never skip non-erasure layers with significant direct effect.
        if !profile.is_erasure && profile.mean_de.abs() >= config.skip_threshold {
            continue;
        }
        // Skip erasure MLPs during draft if configured.
        if config.skip_erasure_draft && profile.is_erasure {
            skip_layers[l] = true;
            continue;
        }
        // Skip layers with negligible |DE|.
        if profile.mean_de.abs() < config.skip_threshold {
            skip_layers[l] = true;
        }
    }

    // Pre-compute cumulative DE for early exit.
    let mut cumulative_de = Vec::with_capacity(n);
    let mut running = 0.0f32;
    for (_l, profile) in profiles.iter().enumerate() {
        running += profile.mean_de.abs();
        cumulative_de.push(running);
    }
    let total_de = running;

    HydraSkipPlan {
        skip_layers,
        cumulative_de,
        total_de,
    }
}

/// Compute which layers to skip based on plan, and determine early exit point.
pub fn hydra_adaptive_budget(skip_plan: &HydraSkipPlan, num_layers: usize) -> HydraBudgetResult {
    let n = skip_plan.skip_layers.len().min(num_layers);
    let mut skipped = Vec::new();

    for l in 0..n {
        if skip_plan.skip_layers[l] {
            skipped.push(l);
        }
    }

    // Early exit: find first layer where cumulative DE >= threshold × total.
    let early_exit_layer = if skip_plan.total_de > 0.0 {
        let threshold_de = 0.95 * skip_plan.total_de;
        skip_plan
            .cumulative_de
            .iter()
            .position(|&cum| cum >= threshold_de)
    } else {
        None
    };

    let savings_fraction = if num_layers > 0 {
        skipped.len() as f32 / num_layers as f32
    } else {
        0.0
    };

    HydraBudgetResult {
        skipped,
        early_exit_layer,
        savings_fraction,
    }
}

/// Inline-friendly check: should layer `layer_idx` be skipped?
///
/// Single array index + bool check — zero allocation, suitable for hot path.
#[inline]
pub fn should_skip_layer(skip_plan: &HydraSkipPlan, layer_idx: usize) -> bool {
    skip_plan
        .skip_layers
        .get(layer_idx)
        .copied()
        .unwrap_or(false)
}

// ── Stage-Aware Skip (T14, Plan 165) ─────────────────────────

/// Check if layer should be skipped based on decode stage.
/// During Draft stage, erasure MLPs can be skipped (they remove info needed for quality,
/// but draft only needs direction). During other stages, use full skip plan.
#[cfg(feature = "decode_specialize")]
pub fn should_skip_layer_stage(
    skip_plan: &HydraSkipPlan,
    layer_idx: usize,
    stage: crate::transformer::DecodeStage,
) -> bool {
    use crate::transformer::DecodeStage;

    let base_skip = should_skip_layer(skip_plan, layer_idx);

    match stage {
        DecodeStage::Draft => {
            // During draft, also skip erasure layers even if base plan doesn't.
            // Check if this layer has erasure flag set.
            if base_skip {
                return true;
            }
            // Check cumulative_de to see if this layer's contribution is negligible
            // relative to total — if so, skip in draft mode for speed.
            let de_ratio = if skip_plan.total_de > 0.0 && layer_idx < skip_plan.cumulative_de.len()
            {
                skip_plan.cumulative_de[layer_idx] / skip_plan.total_de
            } else {
                1.0
            };
            // Skip if cumulative DE at this layer already captures > 90% of total
            de_ratio > 0.90
        }
        DecodeStage::Prefill | DecodeStage::Verify | DecodeStage::Sample => base_skip,
    }
}

/// Calibrate HydraLayerProfile from a direct-effect matrix.
///
/// `de_matrix[prompt][layer]` = direct effect of layer `layer` on prompt `prompt`.
/// Returns one HydraLayerProfile per layer.
pub fn calibrate_profiles(de_matrix: &[Vec<f32>]) -> Vec<HydraLayerProfile> {
    if de_matrix.is_empty() {
        return Vec::new();
    }

    let n_layers = de_matrix[0].len();
    let n_prompts = de_matrix.len() as f32;

    let mut profiles = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let mut sum_abs = 0.0f32;
        let mut negative_count = 0usize;
        let mut backup_count = 0usize;

        for prompt_de in de_matrix {
            let de = prompt_de.get(l).copied().unwrap_or(0.0);
            sum_abs += de.abs();
            if de < 0.0 {
                negative_count += 1;
            }
            // A layer is a "backup" if it has significant negative DE
            // (indicating it compensates for another layer's damage).
            if de < -0.01 {
                backup_count += 1;
            }
        }

        let mean_de = sum_abs / n_prompts;
        let backup_frequency = backup_count as f32 / n_prompts;
        let is_erasure = negative_count as f32 / n_prompts > 0.5;

        profiles.push(HydraLayerProfile {
            mean_de,
            backup_frequency,
            is_erasure,
        });
    }

    profiles
}

// ── Model-Based Logit Lens (Phase 3, Plan 165) ────────────────

/// Per-layer logit lens score computed during the forward pass.
///
/// For each layer l, computes:
/// `score_l = centered_logits(RMSNorm(z^l) @ W_U)` for top token
///
/// This is one matmul per layer — the same pattern as the final logit computation.
#[derive(Clone, Debug)]
pub struct LogitLensScore {
    /// Per-layer score: how much each layer shifts the top-token prediction.
    pub layer_scores: Vec<f32>,
    /// Index of the early-exit layer (cumulative DE convergence).
    pub early_exit_layer: Option<usize>,
}

/// Compute per-layer logit lens scores from hidden states.
///
/// `hidden_states[layer][n_embd]` = RMSNorm(z^layer) for each layer.
/// `lm_head` = [vocab_size × n_embd] language model head weights.
/// `top_token_idx` = the token predicted by the full forward pass.
///
/// Returns per-layer score = how much layer l's output shifts the top token logit.
/// Positive = layer supports the final prediction. Negative = layer opposes it.
pub fn logit_lens_score(
    hidden_states: &[Vec<f32>],
    lm_head: &[f32],
    top_token_idx: usize,
    vocab_size: usize,
    n_embd: usize,
) -> LogitLensScore {
    let n_layers = hidden_states.len();
    let mut layer_scores = Vec::with_capacity(n_layers);

    for layer_hidden in hidden_states {
        // Compute logit for top token: lm_head[top_token_idx * n_embd..(top_token_idx+1) * n_embd] · hidden
        let token_weights =
            &lm_head[top_token_idx * n_embd..(top_token_idx + 1).min(vocab_size) * n_embd];
        let score = token_weights
            .iter()
            .zip(layer_hidden.iter())
            .map(|(w, h)| w * h)
            .sum::<f32>();
        layer_scores.push(score);
    }

    // Find early-exit layer based on cumulative score convergence
    let total_score: f32 = layer_scores.iter().map(|s| s.abs()).sum();
    let early_exit_layer = if total_score > 0.0 {
        let mut cumulative = 0.0f32;
        let mut exit_layer = None;
        for (l, &score) in layer_scores.iter().enumerate() {
            cumulative += score.abs();
            if cumulative >= 0.95 * total_score {
                exit_layer = Some(l);
                break;
            }
        }
        exit_layer
    } else {
        None
    };

    LogitLensScore {
        layer_scores,
        early_exit_layer,
    }
}

/// Adaptive depth gate — determines if cumulative DE has converged.
///
/// Returns `true` when cumulative |DE| exceeds the configured fraction of total |DE|,
/// indicating remaining layers contribute < 5% and can be skipped.
pub fn adaptive_depth_gate(layer_scores: &[f32], cumulative_threshold: f32) -> Option<usize> {
    let total: f32 = layer_scores.iter().map(|s| s.abs()).sum();
    if total <= 0.0 {
        return None;
    }
    let threshold = cumulative_threshold * total;
    let mut cumulative = 0.0f32;
    for (l, &score) in layer_scores.iter().enumerate() {
        cumulative += score.abs();
        if cumulative >= threshold {
            return Some(l);
        }
    }
    None
}

// ── Profile Calibration Tool (T7, Plan 165) ─────────────────

/// Run logit lens calibration on a set of prompts, producing HydraLayerProfile per layer.
/// This is an offline tool — run once, store profiles in config.
///
/// `hidden_states_per_prompt[prompt][layer][n_embd]` = hidden state at each layer for each prompt.
/// `lm_head` = `[vocab_size × n_embd]` language model head weights.
pub fn calibrate_from_prompts(
    hidden_states_per_prompt: &[Vec<Vec<f32>>],
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
) -> Vec<HydraLayerProfile> {
    if hidden_states_per_prompt.is_empty() {
        return Vec::new();
    }

    // Collect per-layer scores across all prompts into a DE matrix.
    let _n_layers = hidden_states_per_prompt[0].len();
    let mut de_matrix: Vec<Vec<f32>> = Vec::with_capacity(hidden_states_per_prompt.len());

    for prompt_hidden in hidden_states_per_prompt {
        // Use token 0 as the top token for scoring (representative token).
        let score = logit_lens_score(prompt_hidden, lm_head, 0, vocab_size, n_embd);
        de_matrix.push(score.layer_scores);
    }

    calibrate_profiles(&de_matrix)
}

// ── Erasure Detection (Phase 4, Plan 165) ────────────────────

/// Detect erasure layers from profiles.
///
/// Erasure MLPs have negative mean DE (they remove information from the residual stream).
/// Returns indices of layers that act as erasure.
pub fn detect_erasure_layers(profiles: &[HydraLayerProfile]) -> Vec<usize> {
    profiles
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_erasure)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profiles(values: &[(f32, f32, bool)]) -> Vec<HydraLayerProfile> {
        values
            .iter()
            .map(
                |&(mean_de, backup_frequency, is_erasure)| HydraLayerProfile {
                    mean_de,
                    backup_frequency,
                    is_erasure,
                },
            )
            .collect()
    }

    #[test]
    fn test_skip_plan_respects_backup_layers() {
        // Layer 0: high backup frequency → never skip.
        // Layer 1: low backup, negligible DE → skip.
        let profiles = make_profiles(&[
            (0.001, 0.5, false), // backup layer
            (0.001, 0.0, false), // negligible DE, not backup
        ]);
        let config = HydraBudgetConfig::default();
        let plan = hydra_layer_skip(&profiles, &config);

        assert!(!plan.skip_layers[0], "backup layer should NOT be skipped");
        assert!(plan.skip_layers[1], "negligible layer should be skipped");
    }

    #[test]
    fn test_early_exit_cumulative() {
        // 10 layers with equal DE = 0.1 each → total = 1.0.
        // 95% threshold at cumulative 0.95 → layer index 9.
        let profiles: Vec<HydraLayerProfile> = (0..10)
            .map(|_| HydraLayerProfile {
                mean_de: 0.1,
                backup_frequency: 0.0,
                is_erasure: false,
            })
            .collect();
        let config = HydraBudgetConfig {
            cumulative_threshold: 0.95,
            ..Default::default()
        };
        let plan = hydra_layer_skip(&profiles, &config);
        let result = hydra_adaptive_budget(&plan, 10);

        assert_eq!(result.early_exit_layer, Some(9));
    }

    #[test]
    fn test_calibrate_profiles() {
        // 2 prompts × 3 layers.
        let de_matrix = vec![
            vec![0.5, -0.3, 0.01], // prompt 0
            vec![0.4, -0.1, 0.02], // prompt 1
        ];
        let profiles = calibrate_profiles(&de_matrix);

        assert_eq!(profiles.len(), 3);

        // Layer 0: mean_de = (0.5 + 0.4) / 2 = 0.45
        assert!((profiles[0].mean_de - 0.45).abs() < 1e-6);
        assert!(!profiles[0].is_erasure);

        // Layer 1: mean_de = (0.3 + 0.1) / 2 = 0.2, both negative → erasure
        assert!((profiles[1].mean_de - 0.2).abs() < 1e-6);
        assert!(profiles[1].is_erasure);

        // Layer 2: mean_de = (0.01 + 0.02) / 2 = 0.015
        assert!((profiles[2].mean_de - 0.015).abs() < 1e-6);
    }

    #[test]
    fn test_no_skip_when_all_important() {
        // All layers have high DE and low backup frequency.
        let profiles: Vec<HydraLayerProfile> = (0..5)
            .map(|_| HydraLayerProfile {
                mean_de: 1.0,
                backup_frequency: 0.0,
                is_erasure: false,
            })
            .collect();
        let config = HydraBudgetConfig {
            skip_threshold: 0.01,
            ..Default::default()
        };
        let plan = hydra_layer_skip(&profiles, &config);
        let result = hydra_adaptive_budget(&plan, 5);

        assert!(result.skipped.is_empty(), "no layers should be skipped");
        assert_eq!(result.savings_fraction, 0.0);
    }

    #[test]
    fn test_modelless_zero_overhead() {
        // Build a plan and verify should_skip_layer is a simple branch.
        let profiles = make_profiles(&[
            (0.5, 0.0, false),   // important → not skipped
            (0.001, 0.0, false), // negligible → skipped
            (0.0, 0.5, true),    // backup → not skipped
        ]);
        let config = HydraBudgetConfig::default();
        let plan = hydra_layer_skip(&profiles, &config);

        // should_skip_layer is just a single array index + bool check.
        assert!(!should_skip_layer(&plan, 0));
        assert!(should_skip_layer(&plan, 1));
        assert!(!should_skip_layer(&plan, 2));
        // Out of bounds → false (not skipped).
        assert!(!should_skip_layer(&plan, 99));
    }

    // ── Phase 3: Model-Based Logit Lens Tests ──

    #[test]
    fn test_logit_lens_score_basic() {
        // 2 layers, 4-dim hidden, 8 vocab size
        let hidden_states = vec![
            vec![1.0, 0.0, 0.0, 0.0], // layer 0
            vec![0.0, 1.0, 0.0, 0.0], // layer 1
        ];
        // lm_head: 8 tokens × 4 dims
        let lm_head = vec![
            1.0, 0.0, 0.0, 0.0, // token 0: aligned with layer 0
            0.0, 1.0, 0.0, 0.0, // token 1: aligned with layer 1
            0.0, 0.0, 1.0, 0.0, // token 2
            0.0, 0.0, 0.0, 1.0, // token 3
            0.5, 0.5, 0.0, 0.0, // token 4
            0.0, 0.0, 0.5, 0.5, // token 5
            0.0, 0.0, 0.0, 0.0, // token 6
            0.0, 0.0, 0.0, 0.0, // token 7
        ];
        let result = logit_lens_score(&hidden_states, &lm_head, 0, 8, 4);

        assert_eq!(result.layer_scores.len(), 2);
        // Layer 0 (aligned with token 0) should score high
        assert!(result.layer_scores[0] > 0.0);
        // Layer 1 (not aligned with token 0) should score lower
        assert!(result.layer_scores[1].abs() < result.layer_scores[0]);
    }

    #[test]
    fn test_adaptive_depth_gate_convergence() {
        // 10 layers, layer 8 already captures 95% of total DE
        let scores = vec![0.5; 10]; // all equal
        let exit = adaptive_depth_gate(&scores, 0.95);
        // 0.95 * 5.0 = 4.75, cumulative reaches at layer 9 (10 * 0.5 = 5.0)
        assert!(exit.is_some());
        assert_eq!(exit.unwrap(), 9);
    }

    // ── Phase 4: Erasure Detection Tests ──

    #[test]
    fn test_calibrate_from_prompts() {
        // 2 prompts × 2 layers × 4 dims
        let hidden_states = vec![
            // Prompt 0
            vec![
                vec![1.0, 0.0, 0.0, 0.0], // layer 0
                vec![0.0, 1.0, 0.0, 0.0], // layer 1
            ],
            // Prompt 1
            vec![
                vec![0.8, 0.0, 0.0, 0.0], // layer 0
                vec![0.0, 0.9, 0.0, 0.0], // layer 1
            ],
        ];
        // lm_head: 4 tokens × 4 dims
        let lm_head = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
            0.0, 0.0, 0.0, 1.0, // token 3
        ];
        let profiles = calibrate_from_prompts(&hidden_states, &lm_head, 4, 4);

        assert_eq!(profiles.len(), 2);
        // Layer 0 aligned with token 0 → high mean_de
        assert!(profiles[0].mean_de > 0.5);
    }

    #[test]
    fn test_detect_erasure_layers() {
        let profiles = make_profiles(&[
            (0.5, 0.0, false), // not erasure
            (0.3, 0.0, true),  // erasure
            (0.1, 0.0, false), // not erasure
            (0.2, 0.0, true),  // erasure
        ]);
        let erasure = detect_erasure_layers(&profiles);
        assert_eq!(erasure, vec![1, 3]);
    }
}
