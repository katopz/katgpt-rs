//! SpecChain — AND/OR composition of multiple CompiledSpec pruners.
//!
//! Chains specs together using logical operators:
//! - **AND**: token valid only if ALL specs agree (intersection of allowed sets)
//! - **OR**: token valid if ANY spec allows it (union of allowed sets)
//!
//! Feature gate: `spec_compile` (depends on `spec_pruner`)

use katgpt_core::traits::{ConstraintPruner, ScreeningPruner};

use super::types::*;

// ── ChainOp ─────────────────────────────────────────────────────

/// Logical operator connecting two adjacent specs in a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChainOp {
    /// Both specs must agree — intersection of allowed tokens.
    And = 0,
    /// Either spec is sufficient — union of allowed tokens.
    Or = 1,
}

// ── SpecChain ───────────────────────────────────────────────────

/// A chain of compiled specs composed with AND/OR operators.
///
/// `ops[i]` connects `specs[i]` and `specs[i+1]`.
/// Evaluation is left-associative: `((s0 op0 s1) op1 s2) op2 s3 ...`
#[derive(Clone, Debug)]
pub struct SpecChain {
    /// The compiled specs in chain order.
    pub specs: Vec<CompiledSpec>,

    /// Operators connecting adjacent specs. `ops.len() == specs.len() - 1`.
    pub ops: Vec<ChainOp>,

    /// BLAKE3 hash of all spec hashes concatenated with ops.
    pub chain_hash: [u8; 32],
}

impl SpecChain {
    /// Build a new chain from specs and connecting operators.
    ///
    /// # Panics
    ///
    /// Panics if `ops.len() != specs.len() - 1` or if `specs` is empty.
    pub fn new(specs: Vec<CompiledSpec>, ops: Vec<ChainOp>) -> Self {
        assert!(!specs.is_empty(), "SpecChain requires at least one spec");
        assert_eq!(
            ops.len(),
            specs.len() - 1,
            "ops.len() must equal specs.len() - 1"
        );

        let chain_hash = compute_chain_hash(&specs, &ops);

        Self {
            specs,
            ops,
            chain_hash,
        }
    }

    /// Combine global bitmaps across the chain according to AND/OR semantics.
    ///
    /// Returns `(combined_allowed, combined_blocked)`.
    ///
    /// - **AND**: intersection of all `global_allowed`, union of all `global_blocked`
    /// - **OR**: union of all `global_allowed`, intersection of all `global_blocked`
    pub fn combine_bitmaps(&self) -> (CompactBitmap, CompactBitmap) {
        if self.specs.len() == 1 {
            return (
                self.specs[0].global_allowed.clone(),
                self.specs[0].global_blocked.clone(),
            );
        }

        let mut allowed = self.specs[0].global_allowed.clone();
        let mut blocked = self.specs[0].global_blocked.clone();

        for (i, spec) in self.specs.iter().skip(1).enumerate() {
            match self.ops[i] {
                ChainOp::And => {
                    allowed = bitmap_intersect(&allowed, &spec.global_allowed);
                    blocked.union_with(&spec.global_blocked);
                }
                ChainOp::Or => {
                    allowed.union_with(&spec.global_allowed);
                    blocked = bitmap_intersect(&blocked, &spec.global_blocked);
                }
            }
        }

        (allowed, blocked)
    }
}

// ── ConstraintPruner ────────────────────────────────────────────

impl ConstraintPruner for SpecChain {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let mut result = self.specs[0].is_valid(depth, token_idx, parent_tokens);

        for (i, spec) in self.specs.iter().skip(1).enumerate() {
            let spec_valid = spec.is_valid(depth, token_idx, parent_tokens);
            result = match self.ops[i] {
                ChainOp::And => result && spec_valid,
                ChainOp::Or => result || spec_valid,
            };
        }

        result
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        if len == 0 {
            return;
        }

        // Initialize with first spec
        self.specs[0].batch_is_valid(depth, candidates, parent_tokens, results);

        // Combine each subsequent spec
        let mut buf = vec![false; len];
        for (i, spec) in self.specs.iter().skip(1).enumerate() {
            spec.batch_is_valid(depth, candidates, parent_tokens, &mut buf);
            match self.ops[i] {
                ChainOp::And => {
                    for j in 0..len {
                        results[j] = results[j] && buf[j];
                    }
                }
                ChainOp::Or => {
                    for j in 0..len {
                        results[j] = results[j] || buf[j];
                    }
                }
            }
        }
    }
}

// ── ScreeningPruner ─────────────────────────────────────────────

impl ScreeningPruner for SpecChain {
    /// AND: minimum relevance across specs.
    /// OR: maximum relevance across specs.
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Compute each spec's relevance via BinaryScreeningPruner semantics
        // (CompiledSpec implements ConstraintPruner → binary relevance).
        let mut rel = spec_relevance(&self.specs[0], depth, token_idx, parent_tokens);

        for (i, spec) in self.specs.iter().skip(1).enumerate() {
            let r = spec_relevance(spec, depth, token_idx, parent_tokens);
            rel = match self.ops[i] {
                ChainOp::And => rel.min(r),
                ChainOp::Or => rel.max(r),
            };
        }

        rel
    }
}

/// Get relevance score from a CompiledSpec (binary: 1.0 if valid, 0.0 if not).
#[inline]
fn spec_relevance(
    spec: &CompiledSpec,
    depth: usize,
    token_idx: usize,
    parent_tokens: &[usize],
) -> f32 {
    if spec.is_valid(depth, token_idx, parent_tokens) {
        1.0
    } else {
        0.0
    }
}

// ── Bitmap intersection helper ──────────────────────────────────

/// Compute the intersection of two bitmaps: only bits set in BOTH survive.
fn bitmap_intersect(a: &CompactBitmap, b: &CompactBitmap) -> CompactBitmap {
    match (a, b) {
        (CompactBitmap::Empty, _) | (_, CompactBitmap::Empty) => CompactBitmap::Empty,

        (CompactBitmap::Sparse(a_arr), CompactBitmap::Sparse(b_arr)) => {
            let mut result = Vec::with_capacity(a_arr.len().min(b_arr.len()));
            let mut ai = 0;
            let mut bi = 0;
            while ai < a_arr.len() && bi < b_arr.len() {
                match a_arr[ai].cmp(&b_arr[bi]) {
                    std::cmp::Ordering::Less => ai += 1,
                    std::cmp::Ordering::Greater => bi += 1,
                    std::cmp::Ordering::Equal => {
                        result.push(a_arr[ai]);
                        ai += 1;
                        bi += 1;
                    }
                }
            }
            CompactBitmap::from_sorted_indices(result)
        }

        (CompactBitmap::Dense(a_bits), CompactBitmap::Dense(b_bits)) => {
            let mut result = Box::new([0u64; 1024]);
            for i in 0..1024 {
                result[i] = a_bits[i] & b_bits[i];
            }
            // Check if all zero → Empty
            if result.iter().all(|&w| w == 0) {
                CompactBitmap::Empty
            } else {
                CompactBitmap::Dense(result)
            }
        }

        (CompactBitmap::Dense(bits), CompactBitmap::Sparse(arr))
        | (CompactBitmap::Sparse(arr), CompactBitmap::Dense(bits)) => {
            let mut result = Vec::with_capacity(arr.len());
            for &lo in arr {
                let idx = lo as usize;
                let word = idx / 64;
                let bit = idx % 64;
                if word < 1024 && (bits[word] >> bit) & 1 == 1 {
                    result.push(lo);
                }
            }
            CompactBitmap::from_sorted_indices(result)
        }
    }
}

// ── Chain hash computation ──────────────────────────────────────

/// Compute BLAKE3 hash of all spec hashes concatenated with ops.
fn compute_chain_hash(specs: &[CompiledSpec], ops: &[ChainOp]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for spec in specs {
        hasher.update(&spec.spec_hash);
    }
    for op in ops {
        hasher.update(&[*op as u8]);
    }
    *hasher.finalize().as_bytes()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec that allows only tokens in `indices`.
    fn allowlist_spec(hash: [u8; 32], indices: &[usize]) -> CompiledSpec {
        CompiledSpec {
            spec_hash: hash,
            rules: vec![SpecRule {
                depth: None,
                prefix: Vec::new(),
                allowed: CompactBitmap::from_token_indices(indices.iter().copied()),
                is_allowlist: true,
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    /// Spec that blocks only tokens in `indices`.
    fn blocklist_spec(hash: [u8; 32], indices: &[usize]) -> CompiledSpec {
        CompiledSpec {
            spec_hash: hash,
            rules: vec![SpecRule {
                depth: None,
                prefix: Vec::new(),
                allowed: CompactBitmap::from_token_indices(indices.iter().copied()),
                is_allowlist: false,
            }],
            vocab_size: 256,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        }
    }

    // ── AND chain: both specs restrict → only intersection passes ──

    #[test]
    fn test_and_chain_intersection() {
        // Spec A: allow {1, 2, 3, 4}
        let spec_a = allowlist_spec([1u8; 32], &[1, 2, 3, 4]);
        // Spec B: allow {3, 4, 5, 6}
        let spec_b = allowlist_spec([2u8; 32], &[3, 4, 5, 6]);

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::And]);

        // Only intersection {3, 4} passes
        assert!(!chain.is_valid(0, 1, &[]));
        assert!(!chain.is_valid(0, 2, &[]));
        assert!(chain.is_valid(0, 3, &[]));
        assert!(chain.is_valid(0, 4, &[]));
        assert!(!chain.is_valid(0, 5, &[]));
        assert!(!chain.is_valid(0, 6, &[]));
    }

    // ── OR chain: either spec allows → token passes ─────────────

    #[test]
    fn test_or_chain_union() {
        // Spec A: allow {1, 2}
        let spec_a = allowlist_spec([3u8; 32], &[1, 2]);
        // Spec B: allow {2, 3}
        let spec_b = allowlist_spec([4u8; 32], &[2, 3]);

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::Or]);

        // Union {1, 2, 3} passes
        assert!(chain.is_valid(0, 1, &[]));
        assert!(chain.is_valid(0, 2, &[]));
        assert!(chain.is_valid(0, 3, &[]));
        assert!(!chain.is_valid(0, 4, &[]));
    }

    // ── chain hash is deterministic ──────────────────────────────

    #[test]
    fn test_chain_hash_deterministic() {
        let spec_a = allowlist_spec([0xAA; 32], &[1, 2]);
        let spec_b = allowlist_spec([0xBB; 32], &[3, 4]);

        let chain1 = SpecChain::new(vec![spec_a.clone(), spec_b.clone()], vec![ChainOp::And]);
        let chain2 = SpecChain::new(vec![spec_a.clone(), spec_b.clone()], vec![ChainOp::And]);

        assert_eq!(chain1.chain_hash, chain2.chain_hash);
    }

    #[test]
    fn test_chain_hash_differs_for_different_ops() {
        let spec_a = allowlist_spec([0xAA; 32], &[1, 2]);
        let spec_b = allowlist_spec([0xBB; 32], &[3, 4]);

        let chain_and = SpecChain::new(vec![spec_a.clone(), spec_b.clone()], vec![ChainOp::And]);
        let chain_or = SpecChain::new(vec![spec_a.clone(), spec_b.clone()], vec![ChainOp::Or]);

        assert_ne!(chain_and.chain_hash, chain_or.chain_hash);
    }

    // ── empty chain (single spec) behaves correctly ──────────────

    #[test]
    fn test_single_spec_chain() {
        let spec = allowlist_spec([5u8; 32], &[10, 20, 30]);
        let chain = SpecChain::new(vec![spec], vec![]);

        assert!(chain.is_valid(0, 10, &[]));
        assert!(chain.is_valid(0, 20, &[]));
        assert!(chain.is_valid(0, 30, &[]));
        assert!(!chain.is_valid(0, 40, &[]));
    }

    #[test]
    #[should_panic(expected = "SpecChain requires at least one spec")]
    fn test_empty_specs_panics() {
        let _: SpecChain = SpecChain::new(vec![], vec![]);
    }

    #[test]
    #[should_panic(expected = "ops.len() must equal specs.len() - 1")]
    fn test_mismatched_ops_panics() {
        let spec_a = allowlist_spec([1u8; 32], &[1]);
        let spec_b = allowlist_spec([2u8; 32], &[2]);
        let _: SpecChain = SpecChain::new(
            vec![spec_a, spec_b],
            vec![ChainOp::And, ChainOp::Or], // too many ops
        );
    }

    // ── chained screening relevance (AND: min, OR: max) ──────────

    #[test]
    fn test_screening_and_min_relevance() {
        let spec_a = allowlist_spec([1u8; 32], &[1, 2]);
        let spec_b = blocklist_spec([2u8; 32], &[2]); // blocks 2, allows everything else

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::And]);

        // Token 1: spec_a allows → 1.0, spec_b doesn't block → 1.0 → min = 1.0
        assert_eq!(chain.relevance(0, 1, &[]), 1.0);

        // Token 2: spec_a allows → 1.0, spec_b blocks → 0.0 → min = 0.0
        assert_eq!(chain.relevance(0, 2, &[]), 0.0);

        // Token 3: spec_a doesn't allow → 0.0, spec_b allows → 1.0 → min = 0.0
        assert_eq!(chain.relevance(0, 3, &[]), 0.0);
    }

    #[test]
    fn test_screening_or_max_relevance() {
        let spec_a = allowlist_spec([1u8; 32], &[1, 2]);
        let spec_b = allowlist_spec([2u8; 32], &[2, 3]);

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::Or]);

        // Token 1: spec_a allows → 1.0, spec_b doesn't → 0.0 → max = 1.0
        assert_eq!(chain.relevance(0, 1, &[]), 1.0);

        // Token 2: both allow → max = 1.0
        assert_eq!(chain.relevance(0, 2, &[]), 1.0);

        // Token 3: spec_a doesn't → 0.0, spec_b allows → 1.0 → max = 1.0
        assert_eq!(chain.relevance(0, 3, &[]), 1.0);

        // Token 4: neither allows → max = 0.0
        assert_eq!(chain.relevance(0, 4, &[]), 0.0);
    }

    // ── batch_is_valid ───────────────────────────────────────────

    #[test]
    fn test_batch_and_chain() {
        let spec_a = allowlist_spec([1u8; 32], &[1, 2, 3, 4]);
        let spec_b = allowlist_spec([2u8; 32], &[3, 4, 5, 6]);

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::And]);

        let candidates: Vec<usize> = vec![1, 2, 3, 4, 5, 6];
        let mut results = vec![false; 6];

        chain.batch_is_valid(0, &candidates, &[], &mut results);

        assert!(!results[0]); // 1: only in A
        assert!(!results[1]); // 2: only in A
        assert!(results[2]); // 3: in both
        assert!(results[3]); // 4: in both
        assert!(!results[4]); // 5: only in B
        assert!(!results[5]); // 6: only in B
    }

    // ── combine_bitmaps ──────────────────────────────────────────

    #[test]
    fn test_combine_bitmaps_and() {
        let spec_a = CompiledSpec {
            spec_hash: [1u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::from_token_indices([1, 2, 3, 4].iter().copied()),
            global_blocked: CompactBitmap::from_token_indices([10].iter().copied()),
        };
        let spec_b = CompiledSpec {
            spec_hash: [2u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::from_token_indices([3, 4, 5, 6].iter().copied()),
            global_blocked: CompactBitmap::from_token_indices([11].iter().copied()),
        };

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::And]);
        let (allowed, blocked) = chain.combine_bitmaps();

        // AND: intersection of allowed = {3, 4}
        assert!(allowed.contains(3));
        assert!(allowed.contains(4));
        assert!(!allowed.contains(1));
        assert!(!allowed.contains(5));

        // AND: union of blocked = {10, 11}
        assert!(blocked.contains(10));
        assert!(blocked.contains(11));
    }

    #[test]
    fn test_combine_bitmaps_or() {
        let spec_a = CompiledSpec {
            spec_hash: [1u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::from_token_indices([1, 2].iter().copied()),
            global_blocked: CompactBitmap::from_token_indices([10, 20].iter().copied()),
        };
        let spec_b = CompiledSpec {
            spec_hash: [2u8; 32],
            rules: vec![],
            vocab_size: 256,
            global_allowed: CompactBitmap::from_token_indices([2, 3].iter().copied()),
            global_blocked: CompactBitmap::from_token_indices([20, 30].iter().copied()),
        };

        let chain = SpecChain::new(vec![spec_a, spec_b], vec![ChainOp::Or]);
        let (allowed, blocked) = chain.combine_bitmaps();

        // OR: union of allowed = {1, 2, 3}
        assert!(allowed.contains(1));
        assert!(allowed.contains(2));
        assert!(allowed.contains(3));

        // OR: intersection of blocked = {20}
        assert!(blocked.contains(20));
        assert!(!blocked.contains(10));
        assert!(!blocked.contains(30));
    }

    // ── multi-op chain ───────────────────────────────────────────

    #[test]
    fn test_three_spec_chain_and_or() {
        // ((A AND B) OR C)
        // A: allow {1, 2}
        // B: allow {2, 3}
        // C: allow {4, 5}
        // A AND B → {2}
        // {2} OR C → {2, 4, 5}
        let spec_a = allowlist_spec([1u8; 32], &[1, 2]);
        let spec_b = allowlist_spec([2u8; 32], &[2, 3]);
        let spec_c = allowlist_spec([3u8; 32], &[4, 5]);

        let chain = SpecChain::new(
            vec![spec_a, spec_b, spec_c],
            vec![ChainOp::And, ChainOp::Or],
        );

        assert!(!chain.is_valid(0, 1, &[])); // only in A, not in B, not in C
        assert!(chain.is_valid(0, 2, &[])); // in A AND B
        assert!(!chain.is_valid(0, 3, &[])); // only in B
        assert!(chain.is_valid(0, 4, &[])); // in C
        assert!(chain.is_valid(0, 5, &[])); // in C
        assert!(!chain.is_valid(0, 6, &[])); // nowhere
    }
}
