//! SpecAsPruner types — token-level constraint rules compiled from NL specs.
//!
//! Research 229: PAW compiles specs → neural weights (~22MB LoRA).
//! We compile specs → symbolic constraint rules (~1KB bitmaps).
//! O(1) per-token validation, zero training, zero neural forward pass.

use std::fmt;

/// A single compiled rule: at a given position in the output,
/// which tokens are allowed or blocked.
///
/// Uses compact bitmap encoding (same pattern as `roaring_membership`):
/// - Sparse (<4096 bits) → sorted `Vec<u16>` array container
/// - Dense (≥4096 bits) → `Box<[u64; 1024]>` bit container
#[derive(Clone, Debug)]
pub struct SpecRule {
    /// Depth (token position) this rule applies to.
    /// `None` means "apply at all depths" (global rule).
    pub depth: Option<usize>,

    /// Prefix constraint: the rule activates when these tokens appear
    /// in the parent path at the given depths.
    pub prefix: Vec<PrefixEntry>,

    /// Token membership bitmap (allowed set).
    pub allowed: CompactBitmap,

    /// Whether `allowed` is an allowlist (true) or blocklist (false).
    /// Allowlist: only tokens in `allowed` pass.
    /// Blocklist: all tokens EXCEPT those in `allowed` pass.
    pub is_allowlist: bool,
}

/// A prefix entry: token + depth that must match for a rule to activate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixEntry {
    /// The depth in the output sequence.
    pub depth: usize,
    /// The token index that must appear at this depth.
    pub token_idx: usize,
}

/// Compiled spec: a collection of rules that implement a ConstraintPruner.
#[derive(Clone, Debug)]
pub struct CompiledSpec {
    /// BLAKE3 hash of the original spec string.
    pub spec_hash: [u8; 32],

    /// The compiled rules, ordered by specificity (more specific first).
    pub rules: Vec<SpecRule>,

    /// Total vocabulary size (determines bitmap range).
    pub vocab_size: usize,

    /// Global allowlist: tokens always allowed at any depth.
    /// If empty, all tokens are allowed by default.
    pub global_allowed: CompactBitmap,

    /// Global blocklist: tokens always blocked at any depth.
    pub global_blocked: CompactBitmap,
}

/// Compact bitmap mimicking Roaring two-level structure.
/// Same pattern as `roaring_membership::CompactBitmap` but standalone.
#[derive(Clone, Debug)]
pub enum CompactBitmap {
    /// Empty bitmap — no bits set.
    Empty,
    /// Sparse: sorted array of set-bit positions.
    Sparse(Vec<u16>),
    /// Dense: 1024 × 64 = 65536 bits.
    Dense(Box<[u64; 1024]>),
}

/// Threshold for switching between sparse and dense.
const SPARSE_MAX_CARDINALITY: usize = 4096;

impl CompactBitmap {
    /// Create an empty bitmap.
    #[inline]
    pub const fn empty() -> Self {
        CompactBitmap::Empty
    }

    /// Create a bitmap from a sorted list of u16 indices.
    pub fn from_sorted_indices(indices: Vec<u16>) -> Self {
        if indices.is_empty() {
            CompactBitmap::Empty
        } else if indices.len() <= SPARSE_MAX_CARDINALITY {
            CompactBitmap::Sparse(indices)
        } else {
            let mut bits = Box::new([0u64; 1024]);
            for &idx in &indices {
                let word = idx as usize / 64;
                let bit = idx as usize % 64;
                bits[word] |= 1u64 << bit;
            }
            CompactBitmap::Dense(bits)
        }
    }

    /// Create a bitmap from token index iterators (u32 token IDs).
    /// Filters to u16 range [0, 65535].
    pub fn from_token_indices(indices: impl Iterator<Item = usize>) -> Self {
        let mut sorted: Vec<u16> = indices
            .filter(|&i| i <= u16::MAX as usize)
            .map(|i| i as u16)
            .collect();
        sorted.sort_unstable();
        sorted.dedup();
        Self::from_sorted_indices(sorted)
    }

    /// Create a dense bitmap with all bits set in range [0, count).
    pub fn all_set(count: usize) -> Self {
        if count == 0 {
            return CompactBitmap::Empty;
        }
        if count <= SPARSE_MAX_CARDINALITY {
            return CompactBitmap::Sparse((0..count as u16).collect());
        }
        let mut bits = Box::new([0u64; 1024]);
        let full_words = count / 64;
        let remainder = count % 64;
        for i in 0..full_words.min(1024) {
            bits[i] = u64::MAX;
        }
        if remainder > 0 && full_words < 1024 {
            bits[full_words] = (1u64 << remainder) - 1;
        }
        CompactBitmap::Dense(bits)
    }

    /// Check if a given index is set.
    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        match self {
            CompactBitmap::Empty => false,
            CompactBitmap::Sparse(a) => {
                if idx > u16::MAX as usize {
                    return false;
                }
                a.binary_search(&(idx as u16)).is_ok()
            }
            CompactBitmap::Dense(bits) => {
                let word = idx / 64;
                let bit = idx % 64;
                if word >= 1024 {
                    return false;
                }
                (bits[word] >> bit) & 1 == 1
            }
        }
    }

    /// Number of set bits (cardinality).
    pub fn len(&self) -> usize {
        match self {
            CompactBitmap::Empty => 0,
            CompactBitmap::Sparse(a) => a.len(),
            CompactBitmap::Dense(bits) => bits.iter().map(|w| w.count_ones() as usize).sum(),
        }
    }

    /// Is the bitmap empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            CompactBitmap::Empty => true,
            CompactBitmap::Sparse(a) => a.is_empty(),
            CompactBitmap::Dense(_) => false,
        }
    }

    /// Insert an index into the bitmap.
    pub fn insert(&mut self, idx: usize) {
        if idx > u16::MAX as usize {
            return;
        }
        let lo = idx as u16;
        match self {
            CompactBitmap::Empty => {
                *self = CompactBitmap::Sparse(vec![lo]);
            }
            CompactBitmap::Sparse(a) => {
                if let Err(pos) = a.binary_search(&lo) {
                    a.insert(pos, lo);
                    if a.len() > SPARSE_MAX_CARDINALITY {
                        let old = std::mem::take(a);
                        *self = Self::from_sorted_indices(old);
                    }
                }
            }
            CompactBitmap::Dense(bits) => {
                let word = idx / 64;
                let bit = idx % 64;
                if word < 1024 {
                    bits[word] |= 1u64 << bit;
                }
            }
        }
    }

    /// Merge another bitmap into this one (union).
    pub fn union_with(&mut self, other: &CompactBitmap) {
        match (&mut *self, other) {
            (CompactBitmap::Empty, _) => {
                *self = other.clone();
            }
            (_, CompactBitmap::Empty) => {}
            (CompactBitmap::Dense(a), CompactBitmap::Dense(b)) => {
                for i in 0..1024 {
                    a[i] |= b[i];
                }
            }
            (CompactBitmap::Dense(a), CompactBitmap::Sparse(b)) => {
                for &lo in b {
                    let word = lo as usize / 64;
                    let bit = lo as usize % 64;
                    if word < 1024 {
                        a[word] |= 1u64 << bit;
                    }
                }
            }
            (CompactBitmap::Sparse(a), CompactBitmap::Dense(b)) => {
                let mut bits = Box::new([0u64; 1024]);
                for &lo in a.iter() {
                    let word = lo as usize / 64;
                    let bit = lo as usize % 64;
                    bits[word] |= 1u64 << bit;
                }
                for i in 0..1024 {
                    bits[i] |= b[i];
                }
                *self = CompactBitmap::Dense(bits);
            }
            (CompactBitmap::Sparse(a), CompactBitmap::Sparse(b)) => {
                for &lo in b {
                    if let Err(pos) = a.binary_search(&lo) {
                        a.insert(pos, lo);
                    }
                }
                if a.len() > SPARSE_MAX_CARDINALITY {
                    let old = std::mem::take(a);
                    *self = Self::from_sorted_indices(old);
                }
            }
        }
    }
}

impl fmt::Display for CompactBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompactBitmap::Empty => write!(f, "∅"),
            CompactBitmap::Sparse(a) => write!(f, "sparse({})", a.len()),
            CompactBitmap::Dense(bits) => {
                let count: usize = bits.iter().map(|w| w.count_ones() as usize).sum();
                write!(f, "dense({})", count)
            }
        }
    }
}

/// Spec type classification — determines compilation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SpecType {
    /// Classification spec: output is one of N fixed labels.
    /// E.g., "Classify sentiment as positive or negative"
    /// Strategy: allowlist of label tokens, block everything else.
    Classification = 0,

    /// Extraction spec: extract structured data from input.
    /// E.g., "Extract email addresses"
    /// Strategy: allow characters valid in emails, block invalid.
    Extraction = 1,

    /// Format repair spec: fix malformed input.
    /// E.g., "Fix malformed JSON"
    /// Strategy: boost structural tokens, suppress invalid patterns.
    FormatRepair = 2,

    /// Intent routing: map input to one of N routes.
    /// E.g., "Route to: search, create, delete, other"
    /// Strategy: same as Classification but with fuzzy matching.
    IntentRouting = 3,

    /// Unknown / complex spec — fallback to ternary adapter.
    Unknown = 4,
}

/// Result of spec compilation.
#[derive(Clone, Debug)]
pub struct CompilationResult {
    /// The compiled spec.
    pub spec: CompiledSpec,
    /// Detected spec type.
    pub spec_type: SpecType,
    /// Number of rules compiled.
    pub rule_count: usize,
    /// Estimated size in bytes.
    pub size_bytes: usize,
    /// Whether the compilation is exact (all outputs provably valid).
    pub is_exact: bool,
}
