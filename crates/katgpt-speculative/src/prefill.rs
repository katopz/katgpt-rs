//! Speculative prefill: PFlash-inspired prompt compression via importance scoring.
//!
//! Uses the draft model's attention scores to estimate per-token importance,
//! then compresses the prompt to top-`keep_ratio` spans before target prefill.
//! Inspired by [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631).
//!
//! ## Plan 390 (2026-07-05) — trait-impl split
//!
//! This module hosts the **pure substrate** half of prefill: the trait
//! `PrefillScorer`, the substrate scorers (`RandomScorer`, `UniformScorer`),
//! the pure compression/selection functions (`compress_prompt`, `block_select`,
//! `block_select_grid`, `compress_prompt_blocks`, `should_compress`, etc.), and
//! the top-level orchestrators (`speculative_prefill*`) that compose the trait
//! + pure helpers.
//!
//! The forward-coupled impls (`AttentionScorer`, `BlockAttentionScorer`) and
//! the entmax-coupled `block_select_entmax` stay in the root crate
//! (`katgpt-rs/src/speculative/prefill.rs`) because they need
//! `crate::transformer::forward` / `crate::speculative::types::SpeculativeContext`
//! (root-only types) or `crate::dash_attn::{entmax_1p5, entmax_support}`
//! (re-export from katgpt-attn's heavy `dash_attn` feature chain).
//!
//! Back-compat is preserved via the root's `pub use prefill::*` re-export.

// `FlashPrefillConfig` + `PrefillMode` + `ScoreReduction` come from
// `katgpt_core::speculative::types` (re-exported via `pub use katgpt_core::speculative::types::*`
// at the crate root).
use katgpt_core::speculative::types::{FlashPrefillConfig, PrefillMode};
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;

// ── Scoring Trait ──────────────────────────────────────────────

/// Strategy for scoring per-token importance over a prompt.
pub trait PrefillScorer: Send + Sync {
    /// Returns per-token importance scores, same length as `prompt_tokens`.
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32>;

    /// Zero-alloc variant: writes scores into pre-allocated buffer.
    /// `scores` must be `>= prompt_tokens.len()`. Written up to `prompt_tokens.len()`.
    fn score_into(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        // Default: allocate and copy
        let result = self.score(draft_weights, draft_config, prompt_tokens);
        let len = result.len().min(scores.len());
        scores[..len].copy_from_slice(&result[..len]);
    }
}

// ── Substrate Scorer Implementations (pure, no forward) ───────

/// Random importance scorer (baseline).
/// Assigns random importance for ablation studies.
pub struct RandomScorer {
    pub seed: u64,
}

impl PrefillScorer for RandomScorer {
    fn score(
        &self,
        _draft_weights: &TransformerWeights,
        _draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        let mut rng = katgpt_types::Rng::new(self.seed);
        prompt_tokens.iter().map(|_| rng.uniform()).collect()
    }
}

/// Uniform importance scorer (baseline).
/// All tokens get equal importance — equivalent to stride-based subsampling.
pub struct UniformScorer;

impl PrefillScorer for UniformScorer {
    fn score(
        &self,
        _draft_weights: &TransformerWeights,
        _draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        vec![1.0; prompt_tokens.len()]
    }
}

// ── Compression ────────────────────────────────────────────────

/// Compress a prompt by selecting top-importance tokens.
///
/// Always keeps:
/// - First `prefix_len` tokens (system prompt / instruction prefix)
/// - Last `suffix_len` tokens (immediate context)
/// - Top-scoring tokens from the middle by importance
///
/// Returns indices of selected tokens in original order.
pub fn compress_prompt(
    importance_scores: &[f32],
    keep_ratio: f32,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    let total = importance_scores.len();
    if total == 0 {
        return Vec::new();
    }

    // Always keep prefix and suffix
    let prefix_len = prefix_len.min(total);
    let suffix_len = suffix_len.min(total.saturating_sub(prefix_len));

    let mandatory = prefix_len + suffix_len;
    let budget = ((total as f32 * keep_ratio).ceil() as usize).max(mandatory);

    if budget >= total {
        // Keep everything
        return (0..total).collect();
    }

    // Middle tokens: score-sorted selection
    let middle_start = prefix_len;
    let middle_end = total.saturating_sub(suffix_len);
    let middle_budget = budget.saturating_sub(mandatory);

    let mut middle_indices: Vec<(usize, f32)> = (middle_start..middle_end)
        .map(|i| (i, importance_scores[i]))
        .collect();

    // Sort by score descending, take top middle_budget. float_order::desc:
    // a NaN importance score sorts last, never first.
    middle_indices.sort_by(|a, b| katgpt_core::float_order::desc(a.1, b.1));
    middle_indices.truncate(middle_budget);

    // Re-sort surviving indices by original position (in-place — avoids a second Vec alloc).
    middle_indices.sort_by_key(|(i, _)| *i);

    let mut selected: Vec<usize> = Vec::with_capacity(budget);

    // Prefix
    selected.extend(0..prefix_len);

    // Middle (already position-sorted above)
    selected.extend(middle_indices.into_iter().map(|(i, _)| i));

    // Suffix
    selected.extend(total.saturating_sub(suffix_len)..total);

    selected
}

// ── Top-level API ──────────────────────────────────────────────

/// Run speculative prefill: score tokens, compress, return selected indices.
///
/// This is the main entry point. After compression, the caller runs the
/// target model forward on only the selected tokens.
pub fn speculative_prefill(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    keep_ratio: f32,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let scores = scorer.score(draft_weights, draft_config, prompt_tokens);
    compress_prompt(&scores, keep_ratio, prefix_len, suffix_len)
}

// ── PFlash Block-Sparse Selection (Plan 044) ──────────────────

/// Block selection: turns per-block scores into selected block indices.
///
/// Rules (from FlashPrefill / PFlash):
///   - sink:       k_block < attention_sink -> always include
///   - window:     |q_block - k_block| < window -> include (recent context)
///   - last_full:  q_block >= num_blocks - last_n_full -> include all blocks
///   - alpha:      score >= max_score * alpha -> include (importance threshold)
///   - causal:     k_block <= q_block
///
/// For prefill scoring, q_block = last block (scoring from generation-start position).
pub fn block_select(block_scores: &[f32], cfg: &FlashPrefillConfig) -> Vec<usize> {
    let num_blocks = block_scores.len();
    if num_blocks == 0 {
        return Vec::new();
    }

    let q_block = num_blocks - 1;
    let max_score = block_scores.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_score * cfg.alpha;

    let mut selected: Vec<usize> = Vec::with_capacity(num_blocks);

    for (k_block, &score) in block_scores.iter().enumerate() {
        if k_block > q_block {
            continue;
        }

        let keep = k_block < cfg.attention_sink
            || q_block.abs_diff(k_block) < cfg.window
            || q_block >= num_blocks.saturating_sub(cfg.last_n_full)
            || score >= threshold;

        if keep {
            selected.push(k_block);
        }
    }

    // Iteration is monotonically increasing in k_block, so `selected` is
    // already sorted and unique — no sort/dedup needed.
    selected.shrink_to_fit();
    selected
}

/// Compute compression ratio from block scores: fraction of blocks passing alpha threshold.
///
/// This is the ratio r used by `adaptive_tree_budget()` to scale DDTree budget.
/// It's a free byproduct of the same scoring that `block_select` already does —
/// no additional compute required.
///
/// # Returns
/// r ∈ (0, 1] where 1.0 = all blocks pass (complex prompt), low = simple prompt.
pub fn block_compression_ratio(block_scores: &[f32], alpha: f32) -> f32 {
    let num_blocks = block_scores.len();
    if num_blocks == 0 {
        return 1.0;
    }
    let max_score = block_scores.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_score * alpha;
    let passing = block_scores.iter().filter(|&&s| s >= threshold).count();
    (passing as f32) / (num_blocks as f32)
}

// NOTE: `block_select_entmax` (Plan 106 T20) stays in root's prefill.rs because
// it consumes `crate::dash_attn::{entmax_1p5, entmax_support}`, which is a
// re-export of katgpt-attn's `dash_attn` feature. That feature pulls a heavy
// chain (katgpt-forward, katgpt-pruners/bandit, katgpt-kv, katgpt-transformer,
// serde) — pulling it into katgpt-speculative for one function would be
// disproportionate. See root `src/speculative/prefill.rs`.

/// Full block-selection with per-(q_block, k_block, head) score grid.
///
/// `score`: `[M][N][H]` row-major (M=q_blocks, N=k_blocks, H=heads).
/// Returns selected indices per (q_block, head) and counts.
pub fn block_select_grid(
    score: &[f32],
    num_q_blocks: usize,
    num_k_blocks: usize,
    num_heads: usize,
    cfg: &FlashPrefillConfig,
) -> (Vec<i32>, Vec<i32>) {
    let m = num_q_blocks;
    let n = num_k_blocks;
    let h = num_heads;

    let mut idx_out = vec![-1i32; m * n * h];
    let mut cnt_out = vec![0i32; m * h];

    // Pre-allocate scratch buffer once; `clear()` reuses capacity across (q, head)
    // iterations instead of allocating a new Vec per inner loop.
    let mut selected_buf: Vec<i32> = Vec::with_capacity(n);

    for q in 0..m {
        let last_full = q >= m.saturating_sub(cfg.last_n_full);

        for head in 0..h {
            let mut max_score: f32 = -f32::INFINITY;
            for k in 0..=q.min(n - 1) {
                let v = score[q * n * h + k * h + head];
                max_score = max_score.max(v);
            }
            let thresh = max_score * cfg.alpha;

            selected_buf.clear();
            for k in 0..=q.min(n - 1) {
                let keep = k < cfg.attention_sink
                    || q.abs_diff(k) < cfg.window
                    || last_full
                    || score[q * n * h + k * h + head] >= thresh;

                if keep {
                    selected_buf.push(k as i32);
                }
            }

            // selected_buf is already sorted (monotonic k).
            let idx_row = &mut idx_out[q * n * h + head..];
            for (i, &sel) in selected_buf.iter().enumerate() {
                idx_row[i * h] = sel;
            }
            cnt_out[q * h + head] = selected_buf.len() as i32;
        }
    }

    (idx_out, cnt_out)
}

// ── MaxSim Block Scoring (Research 45, Plan 080 T6) ────────────

/// Score block pairs using MaxSim late-interaction instead of mean-K dot.
///
/// Given query block Q ∈ [Lq, dim] and key block K ∈ [Lk, dim],
/// computes `Σ_i max_j dot(Q[i], K[j])` — the MaxSim score.
/// This replaces the standard mean dot-product block score with a
/// score that captures the maximum activation per query token.
///
/// For PFlash block selection, this means blocks with spiky attention
/// patterns (a few highly-activated key tokens) score higher than
/// blocks with uniform but moderate activation — better needle detection.
///
/// # Feature flag
/// `maxsim` — Plan 080
#[cfg(feature = "maxsim")]
#[inline]
pub fn block_score_maxsim(
    q_block: &[f32],
    k_block: &[f32],
    block_len_q: usize,
    block_len_k: usize,
    dim: usize,
) -> f32 {
    katgpt_core::simd::maxsim_score(q_block, k_block, block_len_q, block_len_k, dim)
}

/// Compress prompt using block-sparse selection (PFlash algorithm).
///
/// 1. Aggregate per-token scores to per-block scores (max of block)
/// 2. Select blocks via `block_select` rules
/// 3. Flatten selected blocks to token indices
/// 4. Always include prefix and suffix tokens
pub fn compress_prompt_blocks(
    importance_scores: &[f32],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    let total = importance_scores.len();
    if total == 0 {
        return Vec::new();
    }

    let block_size = cfg.block_size;
    let num_blocks = total.div_ceil(block_size);

    let mut block_scores = vec![0.0f32; num_blocks];
    for (i, &score) in importance_scores.iter().enumerate() {
        let block_idx = i / block_size;
        block_scores[block_idx] = block_scores[block_idx].max(score);
    }

    let selected_blocks = block_select(&block_scores, cfg);

    // Upper bound on selected tokens: never more than the original prompt length.
    let mut selected_tokens: Vec<usize> = Vec::with_capacity(total);
    let prefix_end = prefix_len.min(total);
    selected_tokens.extend(0..prefix_end);

    for &block_idx in &selected_blocks {
        let block_start = block_idx * block_size;
        let block_end = ((block_idx + 1) * block_size).min(total);
        for token_idx in block_start..block_end {
            if token_idx >= prefix_end && token_idx < total.saturating_sub(suffix_len) {
                selected_tokens.push(token_idx);
            }
        }
    }

    let suffix_start = total.saturating_sub(suffix_len);
    selected_tokens.extend(suffix_start..total);

    selected_tokens.sort();
    selected_tokens.dedup();

    selected_tokens.shrink_to_fit();
    selected_tokens
}

/// Whether to apply compression for the given prompt length and mode.
pub fn should_compress(mode: PrefillMode, prompt_len: usize, threshold: usize) -> bool {
    match mode {
        PrefillMode::Off => false,
        PrefillMode::Always => true,
        PrefillMode::Auto => prompt_len >= threshold,
    }
}

/// PFlash compression — CPU path (fallback when no GPU).
pub fn speculative_prefill_block(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if prompt_tokens.is_empty() {
        return Vec::new();
    }
    let scores = scorer.score(draft_weights, draft_config, prompt_tokens);
    compress_prompt_blocks(&scores, cfg, prefix_len, suffix_len)
}

/// PFlash compression with adaptive threshold.
/// Picks CPU path (GPU path available via feature flag).
#[allow(clippy::too_many_arguments)]
pub fn speculative_prefill_adaptive(
    scorer: &dyn PrefillScorer,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    prompt_tokens: &[usize],
    mode: PrefillMode,
    threshold: usize,
    cfg: &FlashPrefillConfig,
    prefix_len: usize,
    suffix_len: usize,
) -> Vec<usize> {
    if !should_compress(mode, prompt_tokens.len(), threshold) {
        return (0..prompt_tokens.len()).collect();
    }
    speculative_prefill_block(
        scorer,
        draft_weights,
        draft_config,
        prompt_tokens,
        cfg,
        prefix_len,
        suffix_len,
    )
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_transformer::TransformerWeights;
    use katgpt_types::Rng;

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_compress_preserves_prefix_and_suffix() {
        let scores = vec![0.1; 20];
        let selected = compress_prompt(&scores, 0.5, 3, 2);

        // First 3 must be present
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(selected.contains(&2));
        // Last 2 must be present
        assert!(selected.contains(&18));
        assert!(selected.contains(&19));
    }

    #[test]
    fn test_compress_ratio_approximate() {
        let scores = vec![0.5; 100];
        let selected = compress_prompt(&scores, 0.1, 2, 2);
        // 100 * 0.1 = 10, plus mandatory prefix(2) + suffix(2) = 4
        // Budget = max(10, 4) = 10, middle_budget = 10 - 4 = 6
        assert!(
            selected.len() <= 14,
            "should keep roughly keep_ratio tokens"
        );
    }

    #[test]
    fn test_compress_empty_prompt() {
        let selected = compress_prompt(&[], 0.5, 2, 2);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_compress_single_token() {
        let scores = vec![0.5];
        let selected = compress_prompt(&scores, 0.5, 0, 0);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn test_compress_keeps_all_when_budget_exceeds() {
        let scores = vec![0.5; 5];
        let selected = compress_prompt(&scores, 1.0, 0, 0);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_uniform_scorer_all_equal() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = vec![0; 8];
        let scorer = UniformScorer;
        let scores = scorer.score(&weights, &config, &tokens);
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_random_scorer_varies() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = vec![0; 8];
        let s1 = RandomScorer { seed: 1 };
        let s2 = RandomScorer { seed: 2 };
        let scores1 = s1.score(&weights, &config, &tokens);
        let scores2 = s2.score(&weights, &config, &tokens);
        assert_ne!(scores1, scores2, "different seeds should differ");
    }

    #[test]
    fn test_speculative_prefill_end_to_end() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = (0..8).collect();
        let selected = speculative_prefill(&UniformScorer, &weights, &config, &tokens, 0.5, 1, 1);
        // Should keep first, last, and some middle
        assert!(selected.contains(&0), "should keep first token");
        assert!(selected.contains(&7), "should keep last token");
        assert!(selected.len() < tokens.len(), "should compress");
    }

    /// NIAH-style test: needle-in-haystack retrieval after compression.
    /// Generates [hay×N] + [needle_marker, secret] + [hay×N], then verifies
    /// the needle tokens survive compression when importance scores are high.
    #[test]
    fn test_niah_needle_preserved_after_compression() {
        // Prompt layout: [hay×5] + [needle_marker, secret] + [hay×5] = 12 tokens
        // vocab_size=27: a-z (0-25) + bos (26). Use 'a'=0 as hay, 'z'=25 as marker, 'y'=24 as secret.
        let hay_token: usize = 0; // 'a'
        let needle_marker: usize = 25; // 'z'
        let secret: usize = 24; // 'y'

        let prompt_tokens: Vec<usize> = [
            vec![hay_token; 5],
            vec![needle_marker, secret],
            vec![hay_token; 5],
        ]
        .concat();
        assert_eq!(prompt_tokens.len(), 12);

        // Craft importance scores: needle and secret get high scores, hay gets low
        let mut scores = vec![0.1f32; 12];
        scores[5] = 1.0; // needle marker position — highest importance
        scores[6] = 0.9; // secret position — second highest

        // Case 1: keep_ratio=0.3 → budget = ceil(12*0.3) = 4 tokens
        // Needle (score=1.0) and secret (score=0.9) should both survive
        let selected_30 = compress_prompt(&scores, 0.3, 1, 1);
        assert!(
            selected_30.contains(&5),
            "needle marker (pos 5) should survive at keep_ratio=0.3"
        );
        assert!(
            selected_30.contains(&6),
            "secret (pos 6) should survive at keep_ratio=0.3"
        );

        // Case 2: keep_ratio=0.1 → budget = ceil(12*0.1) = 2 tokens
        // Only top-2 by score survive: needle (1.0) and secret (0.9)
        // After mandatory prefix(1) + suffix(1) = 2, middle_budget = 0
        // So needle/secret must win via prefix/suffix OR score selection
        let selected_10 = compress_prompt(&scores, 0.1, 1, 1);
        // With prefix_len=1 and suffix_len=1, mandatory=2, budget=2
        // Middle budget = 0, so only prefix[0] and suffix[11] are kept.
        // Needle at pos 5 and secret at pos 6 are in the middle and won't be kept.
        assert!(
            selected_10.contains(&0),
            "prefix token should survive at keep_ratio=0.1"
        );
        assert!(
            selected_10.contains(&11),
            "suffix token should survive at keep_ratio=0.1"
        );
        // Middle tokens (including needle) are expected to be dropped at this budget
        assert_eq!(
            selected_10.len(),
            2,
            "only prefix+suffix survive when middle_budget=0"
        );

        // Instead, test with no prefix/suffix mandatory so needle wins by score
        let selected_10_nofix = compress_prompt(&scores, 0.1, 0, 0);
        // budget = 2, middle_budget = 2, top-2 scores: pos 5 (1.0) and pos 6 (0.9)
        assert!(
            selected_10_nofix.contains(&5),
            "needle marker should survive when no mandatory prefix/suffix"
        );
        assert!(
            selected_10_nofix.contains(&6),
            "secret should survive when no mandatory prefix/suffix"
        );

        // Case 3: uniform scores (no importance discrimination) — needle is lost
        let uniform_scores = vec![0.5f32; 12];
        let selected_uniform = compress_prompt(&uniform_scores, 0.1, 0, 0);
        // budget = 2, all scores equal, so arbitrary 2 tokens selected
        // Needle is NOT guaranteed to survive without importance discrimination
        assert_eq!(
            selected_uniform.len(),
            2,
            "uniform scoring should select exactly budget tokens"
        );
        // Demonstrate that importance scoring is what preserves the needle:
        // With uniform scores, needle may or may not be in the result
        let needle_survives_uniform =
            selected_uniform.contains(&5) && selected_uniform.contains(&6);
        // This is expected to usually be false — proving why scoring matters
        // We don't assert it's false (could be coincidentally true), but log the result
        if needle_survives_uniform {
            eprintln!("  [NIAH] Needle coincidentally survived with uniform scores");
        } else {
            eprintln!("  [NIAH] Needle lost with uniform scores (expected — scoring matters)");
        }
    }

    // ── Plan 044: PFlash Block-Sparse Prefill Tests ─────────────

    #[test]
    fn test_block_select_sink_rule() {
        let cfg = FlashPrefillConfig::default();
        let scores = vec![0.0, 0.5, 0.8, 0.3, 0.6]; // 5 blocks
        let selected = block_select(&scores, &cfg);
        // Block 0 should always be selected (sink rule)
        assert!(selected.contains(&0));
    }

    #[test]
    fn test_block_select_window_rule() {
        let cfg = FlashPrefillConfig::default();
        let scores = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // 5 blocks
        let selected = block_select(&scores, &cfg);
        // Last 2 blocks should be selected (window around q_block=4)
        assert!(selected.contains(&4));
        assert!(selected.contains(&3));
    }

    #[test]
    fn test_block_select_alpha_rule() {
        let cfg = FlashPrefillConfig {
            alpha: 0.5,
            attention_sink: 0,
            window: 0,
            last_n_full: 0,
            ..Default::default()
        };
        let scores = vec![0.1, 0.9, 0.2, 0.8, 1.0]; // 5 blocks, q=4
        let selected = block_select(&scores, &cfg);
        // Should select blocks with score >= 0.5 (0.5 * max=1.0)
        assert!(selected.contains(&1)); // 0.9 >= 0.5
        assert!(selected.contains(&3)); // 0.8 >= 0.5
        assert!(selected.contains(&4)); // 1.0 >= 0.5 (q_block)
        assert!(!selected.contains(&0)); // 0.1 < 0.5
    }

    #[test]
    fn test_block_select_empty() {
        let cfg = FlashPrefillConfig::default();
        let selected = block_select(&[], &cfg);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_compress_prompt_blocks_preserves_prefix_suffix() {
        let cfg = FlashPrefillConfig::default();
        let scores = vec![0.5; 100];
        let selected = compress_prompt_blocks(&scores, &cfg, 5, 3);
        // First 5 tokens always included
        assert!(selected.contains(&0));
        assert!(selected.contains(&4));
        // Last 3 tokens always included
        assert!(selected.contains(&97));
        assert!(selected.contains(&99));
    }

    #[test]
    fn test_should_compress_modes() {
        assert!(!should_compress(PrefillMode::Off, 1000, 100));
        assert!(should_compress(PrefillMode::Always, 10, 100));
        assert!(should_compress(PrefillMode::Auto, 200, 100));
        assert!(!should_compress(PrefillMode::Auto, 50, 100));
    }
}
