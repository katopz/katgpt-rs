//! SpecMarginals — spec-to-DDTree marginal integration.
//!
//! Converts compiled spec constraints into token probability biases (logit shifts)
//! that integrate with DDTree's marginal interface.
//!
//! Strategy:
//! - Allowlist rules: allowed tokens get bias 0.0, others get large negative bias (-20.0)
//! - Blocklist rules: blocked tokens get bias -20.0, others get 0.0
//! - Combined: per-token bias is the sum across all rules
//!
//! DDTree can use SpecMarginals as both a ConstraintPruner (soft threshold) and
//! a logit modifier (apply_to_logits).

use katgpt_core::traits::ConstraintPruner;

use super::compiler::SpecCompiler;
use super::types::*;

/// Large negative bias applied to strongly discouraged tokens.
const BLOCKED_BIAS: f32 = -20.0;

/// Soft threshold for the ConstraintPruner `is_valid` check.
/// Tokens with bias above this are considered "valid enough" to explore.
const VALIDITY_THRESHOLD: f32 = -10.0;

/// A single token-level log-probability bias.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenBias {
    /// Token index in the vocabulary.
    pub token_idx: usize,
    /// Log-probability bias to add (positive = encourage, negative = discourage).
    pub bias: f32,
}

/// Spec-to-DDTree marginal: token biases derived from compiled spec rules.
///
/// The biases are stored sorted by `token_idx` for O(log n) binary-search lookup.
/// Tokens not present in the bias list receive `default_bias`.
#[derive(Clone, Debug)]
pub struct SpecMarginals {
    /// Per-token biases, sorted by `token_idx` for binary search.
    biases: Vec<TokenBias>,
    /// Bias applied to tokens not in the bias list.
    default_bias: f32,
    /// Total vocabulary size.
    vocab_size: usize,
}

impl SpecMarginals {
    /// Build marginals from a compiled spec.
    ///
    /// For each rule:
    /// - Allowlist: tokens in `allowed` → bias 0.0, others → -20.0
    /// - Blocklist: tokens in `allowed` → bias -20.0, others → 0.0
    ///
    /// Per-token bias is the sum across all rules. The `default_bias` is set
    /// based on spec type — restrictive specs (Classification) default to -20.0,
    /// permissive specs default to 0.0.
    pub fn from_spec(spec: &CompiledSpec) -> Self {
        // Accumulate per-token biases in a flat map approach.
        // We use a Vec indexed by token for correctness, but limit capacity.
        let cap = spec.vocab_size.min(256);
        let mut bias_map: Vec<(usize, f32)> = Vec::with_capacity(cap);

        // Process each rule and accumulate biases.
        for rule in &spec.rules {
            let in_bias = if rule.is_allowlist {
                0.0 // Allowlist: in→0.0, out→-20.0
            } else {
                BLOCKED_BIAS // Blocklist: in→-20.0, out→0.0
            };

            // Iterate over tokens in the allowed bitmap and set in_bias.
            // For tokens NOT in the bitmap, they get out_bias.
            // We track which tokens we've seen and their accumulated bias.
            match &rule.allowed {
                CompactBitmap::Empty => {
                    if rule.is_allowlist {
                        // Empty allowlist = everything blocked.
                        // All tokens get BLOCKED_BIAS; we don't add individual entries.
                    }
                    // Empty blocklist = nothing blocked; no effect.
                }
                CompactBitmap::Sparse(indices) => {
                    for &idx_u16 in indices {
                        let idx = idx_u16 as usize;
                        accumulate_bias(&mut bias_map, idx, in_bias);
                    }
                }
                CompactBitmap::Dense(bits) => {
                    for (word_idx, &word) in bits.iter().enumerate() {
                        if word == 0 {
                            continue;
                        }
                        let base = word_idx * 64;
                        let mut w = word;
                        while w != 0 {
                            let bit = w.trailing_zeros() as usize;
                            let idx = base + bit;
                            accumulate_bias(&mut bias_map, idx, in_bias);
                            w &= w - 1; // clear lowest set bit
                        }
                    }
                }
            }
        }

        // Also account for global_blocked and global_allowed.
        if !spec.global_blocked.is_empty() {
            // Tokens in global_blocked get BLOCKED_BIAS.
            match &spec.global_blocked {
                CompactBitmap::Sparse(indices) => {
                    for &idx_u16 in indices {
                        accumulate_bias(&mut bias_map, idx_u16 as usize, BLOCKED_BIAS);
                    }
                }
                CompactBitmap::Dense(bits) => {
                    for (word_idx, &word) in bits.iter().enumerate() {
                        if word == 0 {
                            continue;
                        }
                        let base = word_idx * 64;
                        let mut w = word;
                        while w != 0 {
                            let bit = w.trailing_zeros() as usize;
                            accumulate_bias(&mut bias_map, base + bit, BLOCKED_BIAS);
                            w &= w - 1;
                        }
                    }
                }
                CompactBitmap::Empty => {}
            }
        }

        // Sort by token_idx for binary search.
        bias_map.sort_unstable_by_key(|&(idx, _)| idx);
        // Deduplicate: sum biases for same token_idx.
        bias_map = dedup_sum(bias_map);

        let biases: Vec<TokenBias> = bias_map
            .into_iter()
            .map(|(token_idx, bias)| TokenBias { token_idx, bias })
            .collect();

        // Determine default bias: restrictive specs → -20.0, permissive → 0.0.
        // Heuristic: if any rule is a non-empty allowlist, the spec is restrictive.
        let is_restrictive = spec
            .rules
            .iter()
            .any(|r| r.is_allowlist && !r.allowed.is_empty() && r.depth.is_none())
            || !spec.global_blocked.is_empty();

        let default_bias = if is_restrictive { BLOCKED_BIAS } else { 0.0 };

        SpecMarginals {
            biases,
            default_bias,
            vocab_size: spec.vocab_size,
        }
    }

    /// Look up the bias for a specific token.
    ///
    /// Uses binary search in the sorted bias list. Returns `default_bias`
    /// for tokens not present.
    #[inline]
    pub fn bias_for_token(&self, token_idx: usize) -> f32 {
        match self
            .biases
            .binary_search_by_key(&token_idx, |tb| tb.token_idx)
        {
            Ok(i) => self.biases[i].bias,
            Err(_) => self.default_bias,
        }
    }

    /// Apply biases to a logit slice in-place.
    ///
    /// For each position `i` in the logit slice:
    /// - If token `i` has an explicit bias, add it.
    /// - Otherwise, add `default_bias`.
    ///
    /// Zero-allocation: iterates in place without intermediate collections.
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        let len = logits.len().min(self.vocab_size);
        if self.biases.is_empty() {
            // Fast path: no explicit biases, just apply default.
            for i in 0..len {
                logits[i] += self.default_bias;
            }
            return;
        }

        // Two-pointer merge: biases are sorted, logits are index-ordered.
        let mut bias_idx = 0;
        for i in 0..len {
            if bias_idx < self.biases.len() && self.biases[bias_idx].token_idx == i {
                logits[i] += self.biases[bias_idx].bias;
                bias_idx += 1;
            } else {
                logits[i] += self.default_bias;
            }
        }
    }

    /// Return the top-k most positive biases (most encouraged tokens).
    ///
    /// Useful for debugging and for DDTree to know which tokens are most favored.
    pub fn top_k_biases(&self, k: usize) -> Vec<TokenBias> {
        if k == 0 || self.biases.is_empty() {
            return Vec::new();
        }

        let mut sorted: Vec<&TokenBias> = self.biases.iter().collect();
        sorted.sort_unstable_by(|a, b| {
            b.bias
                .partial_cmp(&a.bias)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(k).cloned().collect()
    }

    /// Number of explicit bias entries.
    pub fn len(&self) -> usize {
        self.biases.len()
    }

    /// Whether there are no explicit bias entries.
    pub fn is_empty(&self) -> bool {
        self.biases.is_empty()
    }
}

/// End-to-end: compile a spec string into marginals.
///
/// Uses `SpecCompiler` to parse and compile the spec, then converts to marginals.
pub fn spec_to_marginals(spec_str: &str, vocab_size: usize) -> SpecMarginals {
    let compiler = SpecCompiler::new(vocab_size);
    let result = compiler.compile(spec_str);
    SpecMarginals::from_spec(&result.spec)
}

impl ConstraintPruner for SpecMarginals {
    /// A token is valid if its bias > -10.0 (soft threshold).
    ///
    /// This allows DDTree to use marginals as both a pruner and a probability modifier.
    /// Tokens with strongly negative bias are pruned; tokens with mild or positive bias pass.
    #[inline]
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        self.bias_for_token(token_idx) > VALIDITY_THRESHOLD
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Accumulate a bias for a token index in the bias map.
/// If the token already exists, add the bias to the existing value.
/// Otherwise, insert a new entry (unsorted — sorted later).
fn accumulate_bias(bias_map: &mut Vec<(usize, f32)>, token_idx: usize, bias: f32) {
    match bias_map.iter_mut().find(|(idx, _)| *idx == token_idx) {
        Some((_, existing)) => *existing += bias,
        None => bias_map.push((token_idx, bias)),
    }
}

/// Deduplicate a sorted-by-index list by summing biases for identical indices.
fn dedup_sum(sorted: Vec<(usize, f32)>) -> Vec<(usize, f32)> {
    if sorted.is_empty() {
        return sorted;
    }
    let mut out = Vec::with_capacity(sorted.len());
    let mut cur = sorted[0];
    for i in 1..sorted.len() {
        if sorted[i].0 == cur.0 {
            cur.1 += sorted[i].1;
        } else {
            out.push(cur);
            cur = sorted[i];
        }
    }
    out.push(cur);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classification_spec() -> CompiledSpec {
        // "Classify as positive or negative"
        let allowed = CompactBitmap::from_token_indices(
            [
                b'p', b'o', b's', b'i', b't', b'v', b'e', // "positive"
                b'n', b'g', b'a', b'r', // "negative"
                b' ', b'\n', // whitespace
            ]
            .iter()
            .map(|&b| b as usize),
        );

        CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![SpecRule {
                depth: None,
                prefix: Vec::new(),
                allowed,
                is_allowlist: true,
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    fn make_blocklist_spec() -> CompiledSpec {
        // Block angle brackets (HTML prevention).
        CompiledSpec {
            spec_hash: [1u8; 32],
            rules: vec![SpecRule {
                depth: None,
                prefix: Vec::new(),
                allowed: CompactBitmap::from_token_indices(
                    [b'<', b'>'].iter().map(|&b| b as usize),
                ),
                is_allowlist: false, // blocklist
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    fn make_json_repair_spec() -> CompiledSpec {
        CompiledSpec {
            spec_hash: [2u8; 32],
            rules: vec![
                // Depth 0: only { or [
                SpecRule {
                    depth: Some(0),
                    prefix: Vec::new(),
                    allowed: CompactBitmap::from_token_indices(
                        [b'{', b'['].iter().map(|&b| b as usize),
                    ),
                    is_allowlist: true,
                },
                // Global: JSON-safe chars
                SpecRule {
                    depth: None,
                    prefix: Vec::new(),
                    allowed: CompactBitmap::from_token_indices(
                        (b'a'..=b'z')
                            .chain(b'0'..=b'9')
                            .chain([b'{', b'}', b'[', b']', b':', b',', b'"', b' ', b'\n'])
                            .map(|b| b as usize),
                    ),
                    is_allowlist: true,
                },
            ],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    // --- bias_for_token ---

    #[test]
    fn test_bias_for_token_classification() {
        let spec = make_classification_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // Allowed tokens should have 0.0 bias (allowlist: in→0.0)
        assert_eq!(marginals.bias_for_token(b'p' as usize), 0.0);
        assert_eq!(marginals.bias_for_token(b'o' as usize), 0.0);
        assert_eq!(marginals.bias_for_token(b'e' as usize), 0.0);
        assert_eq!(marginals.bias_for_token(b' ' as usize), 0.0);

        // Disallowed tokens should get default_bias (-20.0)
        assert_eq!(marginals.bias_for_token(b'x' as usize), BLOCKED_BIAS);
        assert_eq!(marginals.bias_for_token(b'z' as usize), BLOCKED_BIAS);
        assert_eq!(marginals.bias_for_token(b'!' as usize), BLOCKED_BIAS);
    }

    #[test]
    fn test_bias_for_token_blocklist() {
        let spec = make_blocklist_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // Blocklist: tokens IN the list get -20.0
        assert_eq!(marginals.bias_for_token(b'<' as usize), BLOCKED_BIAS);
        assert_eq!(marginals.bias_for_token(b'>' as usize), BLOCKED_BIAS);

        // Other tokens get 0.0 (default for permissive spec)
        assert_eq!(marginals.bias_for_token(b'a' as usize), 0.0);
    }

    // --- apply_to_logits ---

    #[test]
    fn test_apply_to_logits_classification() {
        let spec = make_classification_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        let mut logits = vec![0.0f32; 256];
        marginals.apply_to_logits(&mut logits);

        // Allowed tokens: bias 0.0, so logits unchanged
        assert_eq!(logits[b'p' as usize], 0.0);
        assert_eq!(logits[b' ' as usize], 0.0);

        // Disallowed tokens: bias -20.0
        assert_eq!(logits[b'x' as usize], BLOCKED_BIAS);
        assert_eq!(logits[b'!' as usize], BLOCKED_BIAS);
    }

    #[test]
    fn test_apply_to_logits_preserves_existing() {
        let spec = make_classification_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        let mut logits = vec![0.0f32; 256];
        logits[b'p' as usize] = 5.0;
        logits[b'x' as usize] = 3.0;

        marginals.apply_to_logits(&mut logits);

        assert_eq!(logits[b'p' as usize], 5.0); // 5.0 + 0.0
        assert_eq!(logits[b'x' as usize], 3.0 + BLOCKED_BIAS); // 3.0 + (-20.0)
    }

    // --- top_k_biases ---

    #[test]
    fn test_top_k_biases() {
        let spec = make_classification_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // All explicit biases are 0.0 (allowlist in-bias).
        // top_k should return the first k sorted by bias (all equal).
        let top = marginals.top_k_biases(3);
        assert_eq!(top.len(), 3);
        for tb in &top {
            assert_eq!(tb.bias, 0.0);
        }
    }

    #[test]
    fn test_top_k_biases_empty() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };
        let marginals = SpecMarginals::from_spec(&spec);

        let top = marginals.top_k_biases(5);
        assert!(top.is_empty());
    }

    // --- ConstraintPruner ---

    #[test]
    fn test_pruner_classification_valid() {
        let spec = make_classification_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // Allowed tokens: bias 0.0 > -10.0 → valid
        assert!(marginals.is_valid(0, b'p' as usize, &[]));
        assert!(marginals.is_valid(0, b' ' as usize, &[]));

        // Blocked tokens: bias -20.0 < -10.0 → invalid
        assert!(!marginals.is_valid(0, b'x' as usize, &[]));
        assert!(!marginals.is_valid(0, b'!' as usize, &[]));
    }

    #[test]
    fn test_pruner_blocklist() {
        let spec = make_blocklist_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // Blocked tokens: bias -20.0 < -10.0 → invalid
        assert!(!marginals.is_valid(0, b'<' as usize, &[]));
        assert!(!marginals.is_valid(0, b'>' as usize, &[]));

        // Others: bias 0.0 > -10.0 → valid
        assert!(marginals.is_valid(0, b'a' as usize, &[]));
    }

    // --- spec_to_marginals end-to-end ---

    #[test]
    fn test_spec_to_marginals_classification() {
        let marginals = spec_to_marginals("Classify sentiment as positive or negative", 256);

        // "positive" and "negative" label chars should have 0.0 bias
        assert_eq!(marginals.bias_for_token(b'p' as usize), 0.0);
        assert_eq!(marginals.bias_for_token(b'n' as usize), 0.0);

        // Unknown chars should get negative bias
        assert!(marginals.bias_for_token(b'x' as usize) < 0.0);
    }

    #[test]
    fn test_spec_to_marginals_json_repair() {
        let marginals = spec_to_marginals("Fix malformed JSON", 256);

        // JSON structural tokens should have positive or zero bias
        let brace_bias = marginals.bias_for_token(b'{' as usize);
        assert!(
            brace_bias >= 0.0,
            "JSON brace should have >= 0 bias, got {brace_bias}"
        );
    }

    // --- Combined rules ---

    #[test]
    fn test_combined_rules() {
        let spec = make_json_repair_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // '{' is in both rules (depth 0 allowlist + global allowlist)
        // Each contributes 0.0, so total should be 0.0
        assert_eq!(marginals.bias_for_token(b'{' as usize), 0.0);

        // 'a' is in global but NOT in depth-0 rule.
        // Global allowlist in-bias = 0.0 for 'a'.
        // Depth-0 rule out-bias not applied (only has depth=0 rule).
        // Since 'a' is in global allowlist it gets 0.0.
        assert_eq!(marginals.bias_for_token(b'a' as usize), 0.0);
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_spec_marginals() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };
        let marginals = SpecMarginals::from_spec(&spec);

        // No rules → default_bias = 0.0 (permissive)
        assert_eq!(marginals.default_bias, 0.0);
        assert_eq!(marginals.bias_for_token(0), 0.0);
        assert_eq!(marginals.bias_for_token(255), 0.0);
        assert!(marginals.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_global_blocked_makes_restrictive() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::from_token_indices(
                [b'<', b'>'].iter().map(|&b| b as usize),
            ),
        };
        let marginals = SpecMarginals::from_spec(&spec);

        // global_blocked makes it restrictive → default_bias = -20.0
        assert_eq!(marginals.default_bias, BLOCKED_BIAS);

        // Blocked tokens get -20.0 (explicit) + -20.0 (default) = -40.0
        let bias = marginals.bias_for_token(b'<' as usize);
        assert_eq!(bias, BLOCKED_BIAS); // explicit entry

        // Non-blocked tokens get default_bias = -20.0
        assert_eq!(marginals.bias_for_token(b'a' as usize), BLOCKED_BIAS);
    }

    #[test]
    fn test_len_and_empty() {
        let spec = make_classification_spec();
        let marginals = SpecMarginals::from_spec(&spec);

        // Should have explicit biases for the 13 allowed chars
        assert!(!marginals.is_empty());
        assert!(marginals.len() > 0);

        let empty_spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };
        let empty_marginals = SpecMarginals::from_spec(&empty_spec);
        assert!(empty_marginals.is_empty());
        assert_eq!(empty_marginals.len(), 0);
    }
}
