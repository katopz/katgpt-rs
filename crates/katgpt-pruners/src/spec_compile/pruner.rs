//! ConstraintPruner implementation for CompiledSpec.
//!
//! Implements the core `ConstraintPruner` trait for spec-compiled rules.
//! O(1) per-token validation via bitmap lookup. Zero neural forward pass.

#![allow(clippy::needless_range_loop)]

use katgpt_core::traits::ConstraintPruner;

use super::types::*;

impl ConstraintPruner for CompiledSpec {
    /// Check if `token_idx` at `depth` is valid according to compiled rules.
    ///
    /// Evaluation order (most specific first):
    /// 1. Depth-specific rules with matching prefix → use their bitmap
    /// 2. Depth-agnostic (global) rules → use their bitmap
    /// 3. Global blocked set → reject
    /// 4. Global allowed set → accept
    /// 5. Default: allow (no matching rule)
    #[inline]
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Phase 1: Check depth-specific rules with prefix matching
        for rule in &self.rules {
            // Skip if depth doesn't match
            if let Some(rule_depth) = rule.depth
                && rule_depth != depth
            {
                continue;
            }

            // Check prefix constraint
            if !prefix_matches(&rule.prefix, parent_tokens) {
                continue;
            }

            // Apply the rule
            let in_bitmap = rule.allowed.contains(token_idx);
            return if rule.is_allowlist {
                in_bitmap // Allowlist: must be in the set
            } else {
                !in_bitmap // Blocklist: must NOT be in the set
            };
        }

        // Phase 2: Global blocked set
        if self.global_blocked.contains(token_idx) {
            return false;
        }

        // Phase 3: Global allowed set (if non-empty, must be in it)
        if !self.global_allowed.is_empty() {
            return self.global_allowed.contains(token_idx);
        }

        // Default: allow
        true
    }

    /// Batch validation: check multiple candidates at the same depth.
    ///
    /// Optimized to evaluate rules once per batch instead of per-token.
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        // Find the applicable rule for this depth + prefix
        let applicable_rule = self.find_rule(depth, parent_tokens);

        match applicable_rule {
            Some((bitmap, is_allowlist)) => {
                let len = candidates.len().min(results.len());
                match bitmap {
                    CompactBitmap::Empty => {
                        for i in 0..len {
                            results[i] = !is_allowlist; // Empty allowlist = all blocked
                        }
                    }
                    CompactBitmap::Sparse(a) => {
                        // Sparse: binary search per candidate
                        for i in 0..len {
                            let idx = candidates[i];
                            let in_bitmap = if idx <= u16::MAX as usize {
                                a.binary_search(&(idx as u16)).is_ok()
                            } else {
                                false
                            };
                            results[i] = if is_allowlist { in_bitmap } else { !in_bitmap };
                        }
                    }
                    CompactBitmap::Dense(bits) => {
                        // Dense: direct bit check — O(1) per candidate
                        for i in 0..len {
                            let idx = candidates[i];
                            let word = idx / 64;
                            let bit = idx % 64;
                            let in_bitmap = if word < 1024 {
                                (bits[word] >> bit) & 1 == 1
                            } else {
                                false
                            };
                            results[i] = if is_allowlist { in_bitmap } else { !in_bitmap };
                        }
                    }
                }
            }
            None => {
                // No applicable rule — check global sets, then default allow
                let len = candidates.len().min(results.len());
                for i in 0..len {
                    let idx = candidates[i];
                    if self.global_blocked.contains(idx) {
                        results[i] = false;
                    } else if !self.global_allowed.is_empty() {
                        results[i] = self.global_allowed.contains(idx);
                    } else {
                        results[i] = true;
                    }
                }
            }
        }
    }
}

impl CompiledSpec {
    /// Find the applicable rule for a given depth and prefix.
    /// Returns the bitmap and whether it's an allowlist.
    #[inline]
    fn find_rule(&self, depth: usize, parent_tokens: &[usize]) -> Option<(&CompactBitmap, bool)> {
        for rule in &self.rules {
            if let Some(rule_depth) = rule.depth
                && rule_depth != depth
            {
                continue;
            }
            if !prefix_matches(&rule.prefix, parent_tokens) {
                continue;
            }
            return Some((&rule.allowed, rule.is_allowlist));
        }
        None
    }
}

/// Check if a prefix matches the current path.
#[inline]
fn prefix_matches(prefix: &[PrefixEntry], parent_tokens: &[usize]) -> bool {
    if prefix.is_empty() {
        return true; // No prefix constraint → always matches
    }

    for entry in prefix {
        let actual = parent_tokens.get(entry.depth);
        match actual {
            Some(&token) if token == entry.token_idx => continue,
            _ => return false,
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classification_spec() -> CompiledSpec {
        // Simulate: "Classify as positive or negative"
        // Allow: p, o, s, i, t, v, e, n, g, a, r, space, newline
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

    fn make_json_depth0_spec() -> CompiledSpec {
        CompiledSpec {
            spec_hash: [1u8; 32],
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

    #[test]
    fn test_classification_allowlist() {
        let spec = make_classification_spec();

        // Characters in "positive" and "negative" should be allowed
        assert!(spec.is_valid(0, b'p' as usize, &[]));
        assert!(spec.is_valid(0, b'o' as usize, &[]));
        assert!(spec.is_valid(1, b's' as usize, &[b'p' as usize]));
        assert!(spec.is_valid(0, b' ' as usize, &[]));

        // Characters NOT in the allowlist should be blocked
        assert!(!spec.is_valid(0, b'x' as usize, &[]));
        assert!(!spec.is_valid(0, b'z' as usize, &[]));
        assert!(!spec.is_valid(0, b'!' as usize, &[]));
    }

    #[test]
    fn test_json_depth0_rule() {
        let spec = make_json_depth0_spec();

        // Depth 0: only { or [ allowed
        assert!(spec.is_valid(0, b'{' as usize, &[]));
        assert!(spec.is_valid(0, b'[' as usize, &[]));
        assert!(!spec.is_valid(0, b'a' as usize, &[]));
        assert!(!spec.is_valid(0, b'"' as usize, &[]));

        // Depth 1+: JSON-safe chars allowed
        assert!(spec.is_valid(1, b'a' as usize, &[b'{' as usize]));
        assert!(spec.is_valid(1, b'"' as usize, &[b'{' as usize]));
        assert!(spec.is_valid(1, b'0' as usize, &[b'{' as usize]));
    }

    #[test]
    fn test_batch_validation() {
        let spec = make_classification_spec();
        let candidates: Vec<usize> = vec![
            b'p' as usize,
            b'o' as usize,
            b'x' as usize,
            b'z' as usize,
            b' ' as usize,
        ];
        let mut results = vec![false; 5];

        spec.batch_is_valid(0, &candidates, &[], &mut results);

        assert!(results[0]); // 'p' allowed
        assert!(results[1]); // 'o' allowed
        assert!(!results[2]); // 'x' blocked
        assert!(!results[3]); // 'z' blocked
        assert!(results[4]); // ' ' allowed
    }

    #[test]
    fn test_prefix_matching() {
        // Rule that activates only after seeing '{' at depth 0
        let spec = CompiledSpec {
            spec_hash: [2u8; 32],
            rules: vec![SpecRule {
                depth: Some(1),
                prefix: vec![PrefixEntry {
                    depth: 0,
                    token_idx: b'{' as usize,
                }],
                allowed: CompactBitmap::from_token_indices(
                    [b'"', b'}'].iter().map(|&b| b as usize),
                ),
                is_allowlist: true,
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };

        // After '{' at depth 0, depth 1 only allows '"' or '}'
        assert!(spec.is_valid(1, b'"' as usize, &[b'{' as usize]));
        assert!(spec.is_valid(1, b'}' as usize, &[b'{' as usize]));
        assert!(!spec.is_valid(1, b'a' as usize, &[b'{' as usize]));

        // Without matching prefix, no rule applies → default allow
        assert!(spec.is_valid(1, b'a' as usize, &[b'[' as usize]));
    }

    #[test]
    fn test_empty_spec_allows_everything() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: Vec::new(),
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };

        // Empty spec: everything allowed
        assert!(spec.is_valid(0, 0, &[]));
        assert!(spec.is_valid(0, 255, &[]));
    }

    #[test]
    fn test_global_blocked() {
        let spec = CompiledSpec {
            spec_hash: [0u8; 32],
            rules: Vec::new(),
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::from_token_indices(
                [b'<', b'>'].iter().map(|&b| b as usize),
            ),
        };

        assert!(!spec.is_valid(0, b'<' as usize, &[]));
        assert!(!spec.is_valid(0, b'>' as usize, &[]));
        assert!(spec.is_valid(0, b'a' as usize, &[]));
    }
}
