//! Token taggers (Plan 335 Phase 1 T1.9).
//!
//! The diagnostic needs a per-position class label to do tag-stratified
//! aggregation. The tagger is pluggable — prose consumers use POS, code
//! consumers use source-level categories, game-runtime consumers use game-
//! state-derived labels (open-class content vs closed-class function vs
//! copy/n-gram). This module ships the trait + one trivial impl
//! ([`CopyNGramTagger`]) that marks positions completing a repeated n-gram
//! (the paper's COPY_k feature). Richer taggers (POS, bracket detection,
//! game-state-derived) are consumer-side.

use crate::paired_loss::types::TokenClass;

/// Classifies each token position into a [`TokenClass`].
///
/// The tagger is called ONCE per position to build the `classes: &[TokenClass]`
/// array, which is then reused across all filter queries. Tagger latency is
/// NOT hot-path-critical — it's amortized over many `filtered_mean` calls.
///
/// **Convention:** `prefix` is the full token sequence INCLUDING the token at
/// `position` (i.e., `prefix.len() > position` and `prefix[position] ==
/// token_id` when `position < prefix.len()`). Taggers that need the prefix
/// for context (e.g., n-gram detection) read it directly; taggers that only
/// need the token id (e.g., a vocab-lookup tagger) ignore `prefix`.
pub trait TokenTagger {
    /// Classify the token at `position` (id `token_id`) given the full
    /// `prefix` sequence (which includes the current token at `position`).
    fn classify(&self, token_id: u32, position: usize, prefix: &[u32]) -> TokenClass;
}

/// Marks positions completing a repeated n-gram of length `n` (paper's
/// COPY_k feature, §4 Pattern iii).
///
/// A position `i` is `CopyN(n)` if the n-gram `prefix[i-n+1 ..= i]` appears
/// at least once earlier in `prefix` (starting at some `j < i-n+1`, so the
/// previous occurrence ends strictly before the current position — no self-
/// match). Otherwise the position is [`TokenClass::Other`] (this tagger only
/// knows copy status; a richer tagger would fall through to a base classifier
/// for non-copy positions).
///
/// **Complexity:** O(L · n) per position, O(L² · n) total for a length-L
/// sequence. Fine for Phase 1 (the tagger is called once per eval, not per
/// forward-pass token). A hash-based O(L) version is a Phase 2 optimization
/// if the tagger ever lands on a hot path.
///
/// # Example
///
/// `prefix = [10, 20, 10, 20, 10]`, `n = 2`:
/// - position 0: not enough context → Other
/// - position 1: n-gram `[10, 20]`, no earlier occurrence → Other
/// - position 2: n-gram `[20, 10]`, search `j ∈ 0..1`: `[10,20]` ≠ `[20,10]` → Other
/// - position 3: n-gram `[10, 20]`, search `j ∈ 0..2`: `j=0` → `[10,20]` ✓ → **CopyN(2)**
/// - position 4: n-gram `[20, 10]`, search `j ∈ 0..3`: `j=1` → `[20,10]` ✓ → **CopyN(2)**
#[derive(Clone, Copy, Debug)]
pub struct CopyNGramTagger {
    /// The n-gram length to detect (`n ≥ 2`; `n = 1` matches every repeat
    /// and is rarely useful).
    pub n: usize,
}

impl CopyNGramTagger {
    /// Construct with n-gram length `n`.
    #[inline]
    pub const fn new(n: usize) -> Self {
        Self { n }
    }
}

impl TokenTagger for CopyNGramTagger {
    #[inline]
    fn classify(&self, _token_id: u32, position: usize, prefix: &[u32]) -> TokenClass {
        let n = self.n;
        // Need at least n tokens up to and including position.
        // position + 1 is the count of tokens [0..=position].
        if n == 0 || position + 1 < n || prefix.len() < position + 1 {
            return TokenClass::Other;
        }
        // The n-gram ending at `position`: prefix[position-n+1 ..= position].
        let cur_start = position + 1 - n;
        let cur = &prefix[cur_start..=position];
        // Search for a previous occurrence starting at j < cur_start.
        // The previous occurrence prefix[j..j+n] ends at j+n-1 < cur_start+n-1
        // = position, so it ends strictly before the current position — no
        // self-match, no overlap with the current occurrence's start.
        // Upper bound on j: prefix.len() - n (so prefix[j..j+n] is in bounds).
        let j_max = prefix.len().saturating_sub(n);
        for j in 0..j_max.min(cur_start) {
            // j < cur_start ensures no self-match; prefix[j..j+n] is valid
            // because j+n <= cur_start <= position < prefix.len().
            if &prefix[j..j + n] == cur {
                return TokenClass::CopyN(n);
            }
        }
        TokenClass::Other
    }
}
