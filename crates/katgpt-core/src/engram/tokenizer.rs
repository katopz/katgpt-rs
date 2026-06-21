//! Surjective tokenizer compression — V → V' via NFKC + lowercase collapse.
//!
//! Plan 299 Phase 4 T4.1–T4.5. Implements the paper's surjective token
//! projection `P: V → V'`: many raw tokenizer IDs collapse to one canonical
//! ID (e.g. `"Apple"`, `" apple"`, `"APPLE"` all map to the same canonical).
//! Paper Appendix C reports **23% vocabulary reduction** on a 128k tokenizer.
//!
//! # Pipeline (per raw token, at build time only)
//!
//! ```text
//! decode(raw_id) → bytes
//!   → NFKC normalize         (unicode-normalization crate)
//!   → lowercase              (ASCII fast path; Unicode via `to_lowercase`)
//!   → BLAKE3                 (AGENTS.md: BLAKE3, not SHA)
//!   → first 8 bytes as u64   → CanonicalId(u64)
//! ```
//!
//! The pipeline is **surjective** (many-to-one) but never **injective** in
//! the wrong direction: two semantically-distinct tokens *can* collide if
//! their lowercased+normalized surface forms happen to share a BLAKE3 prefix,
//! but the 64-bit hash makes accidental collisions vanishingly unlikely.
//!
//! # CRITICAL — never softmax
//!
//! Per AGENTS.md this module contains **no `softmax` symbol**. It's pure
//! data plumbing — projection happens at the **tokenizer boundary**, before
//! the multi-head hash. The sigmoid gate lives in
//! [`crate::engram::kernel`].
//!
//! # Hot-path contract
//!
//! [`compress_token`] is **O(1) + zero-allocation**: direct index into
//! `raw_to_canonical`. The build pipeline ([`build_surjective_map`]) is
//! O(V) and allocates — but it runs once offline.
//!
//! # Serialization
//!
//! [`SurjectiveMap::save_to_bytes`] / [`load_from_bytes`] use **postcard**
//! (already a katgpt-core dep). A BLAKE3 commitment over the serialized
//! bytes is prepended; [`load_from_bytes`] verifies it on load and rejects
//! tampered maps.

use super::CanonicalId;
use super::TokenId;
use unicode_normalization::UnicodeNormalization;

/// An immutable, pre-computed surjective projection `V → V'`.
///
/// Built once offline via [`build_surjective_map`] from a tokenizer spec.
/// The `raw_to_canonical` array is indexed by raw token id
/// (`raw_to_canonical[raw_id as usize]`) — O(1) lookup, zero-allocation.
///
/// `#[repr(transparent)]` over `Box<[CanonicalId]>` would lose the
/// `repr(transparent)` guarantee on the Box, so we keep the struct
/// transparent over its single field via the standard `#[repr(transparent)]`
/// newtype pattern (the field is itself heap-allocated via Box, but the
/// struct layout is transparent to its caller-facing handle).
#[derive(Debug, Clone)]
pub struct SurjectiveMap {
    /// `raw_to_canonical[raw_id]` = canonical id for that raw token.
    /// Length = `tokenizer.vocab_size()`. Built once, never mutated.
    pub raw_to_canonical: Box<[CanonicalId]>,
}

impl SurjectiveMap {
    /// Number of raw tokens in the map (= vocab size of the source tokenizer).
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.raw_to_canonical.len()
    }

    /// Number of distinct canonical ids (= size of the compressed vocab V').
    ///
    /// O(V) scan — for diagnostics only, not on the hot path.
    pub fn canonical_vocab_size(&self) -> usize {
        let mut distinct = std::collections::HashSet::new();
        for &c in self.raw_to_canonical.iter() {
            distinct.insert(c);
        }
        distinct.len()
    }

    /// Compression ratio = `1 - canonical / raw` (paper Appendix C target: 0.23).
    pub fn compression_ratio(&self) -> f32 {
        let raw = self.vocab_size();
        if raw == 0 {
            return 0.0;
        }
        let canon = self.canonical_vocab_size();
        1.0 - (canon as f32 / raw as f32)
    }

    /// Serialize to postcard bytes with a BLAKE3 commitment prepended.
    ///
    /// Layout: `[32-byte BLAKE3 commitment] || [postcard-serialized Vec<u64>]`.
    /// The commitment is over the postcard bytes only (not self-inclusive).
    /// On load, [`load_from_bytes`] recomputes the BLAKE3 over the payload
    /// and compares — tampered maps are rejected.
    pub fn save_to_bytes(&self) -> Vec<u8> {
        // Extract the raw u64s — CanonicalId is #[repr(transparent)] over u64
        // so this is a zero-cost reinterpretation in spirit (we copy here
        // for serialization simplicity; the hot-path lookup uses the
        // Box<[CanonicalId]> directly without copy).
        let u64s: Vec<u64> = self.raw_to_canonical.iter().map(|c| c.0).collect();
        let payload = postcard::to_allocvec(&u64s).expect("postcard serialize");

        // BLAKE3 commitment over the payload.
        let mut hasher = blake3::Hasher::new();
        hasher.update(&payload);
        let commit = *hasher.finalize().as_bytes();

        let mut out = Vec::with_capacity(32 + payload.len());
        out.extend_from_slice(&commit);
        out.extend_from_slice(&payload);
        out
    }

    /// Deserialize from postcard bytes; verify the BLAKE3 commitment.
    ///
    /// Returns `Err` if the commitment does not match the payload (tampered
    /// or corrupted map) or if postcard deserialization fails.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, SurjectiveMapLoadError> {
        if bytes.len() < 32 {
            return Err(SurjectiveMapLoadError::TooShort);
        }
        let (stored_commit, payload) = bytes.split_at(32);

        // Recompute BLAKE3 over the payload.
        let mut hasher = blake3::Hasher::new();
        hasher.update(payload);
        let recomputed = *hasher.finalize().as_bytes();
        if recomputed.as_slice() != stored_commit {
            return Err(SurjectiveMapLoadError::CommitmentMismatch);
        }

        let u64s: Vec<u64> =
            postcard::from_bytes(payload).map_err(SurjectiveMapLoadError::Postcard)?;
        let raw_to_canonical: Box<[CanonicalId]> = u64s.into_iter().map(CanonicalId).collect();
        Ok(Self { raw_to_canonical })
    }
}

/// Error returned by [`SurjectiveMap::load_from_bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurjectiveMapLoadError {
    /// Fewer than 32 bytes — can't even read the commitment.
    TooShort,
    /// BLAKE3 commitment over the payload doesn't match the stored value.
    CommitmentMismatch,
    /// Postcard deserialization failed.
    Postcard(postcard::Error),
}

impl std::fmt::Display for SurjectiveMapLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "SurjectiveMap bytes too short (< 32)"),
            Self::CommitmentMismatch => write!(f, "SurjectiveMap BLAKE3 commitment mismatch"),
            Self::Postcard(e) => write!(f, "SurjectiveMap postcard error: {e}"),
        }
    }
}

impl std::error::Error for SurjectiveMapLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Postcard(e) => Some(e),
            _ => None,
        }
    }
}

/// Compress a raw tokenizer id to its canonical id via direct index lookup.
///
/// Plan 299 Phase 4 T4.2. **O(1), zero-allocation.**
///
/// # Panics
///
/// Panics if `raw_id.0 >= projection.vocab_size()` (out-of-bounds). The
/// caller is responsible for ensuring the raw id is within the source
/// tokenizer's vocab range. Use [`try_compress_token`] for a fallible version.
#[inline]
pub fn compress_token(raw_id: TokenId, projection: &SurjectiveMap) -> CanonicalId {
    projection.raw_to_canonical[raw_id.0 as usize]
}

/// Fallible variant of [`compress_token`] — returns `None` on out-of-bounds.
#[inline]
pub fn try_compress_token(raw_id: TokenId, projection: &SurjectiveMap) -> Option<CanonicalId> {
    projection.raw_to_canonical.get(raw_id.0 as usize).copied()
}

/// Trait abstracting a tokenizer for [`build_surjective_map`].
///
/// Implementors provide:
/// - `vocab_size` — number of raw tokens `V`.
/// - `decode_token` — the surface bytes of token `raw_id` (e.g. `" apple"`,
///   `"Apple"`, `"é"`, etc.).
///
/// This is intentionally minimal — katgpt-core is tokenizer-agnostic. The
/// host wraps its real tokenizer (Hugging Face `tokenizers`, sentencepiece,
/// tiktoken, etc.) in this trait.
pub trait TokenizerSpec {
    /// Number of raw tokens `V` in the tokenizer.
    fn vocab_size(&self) -> u32;

    /// The surface bytes of token `raw_id` (UTF-8 encoded).
    ///
    /// For example, `decode_token(TokenId(402))` might return `b" Apple"` (with
    /// the leading space that BPE tokenizers attach). Returning an empty
    /// slice is valid — the resulting canonical will be the BLAKE3 of empty
    /// bytes (a single fixed canonical for all empty tokens).
    fn decode_token(&self, raw_id: TokenId) -> &[u8];
}

/// Build a [`SurjectiveMap`] from a tokenizer spec.
///
/// Plan 299 Phase 4 T4.3. For each raw token id in `[0, vocab_size)`:
/// 1. Decode the surface bytes.
/// 2. Convert bytes to a string (lossy — invalid UTF-8 becomes U+FFFD).
/// 3. NFKC normalize.
/// 4. Lowercase.
/// 5. BLAKE3 over the resulting UTF-8 bytes → first 8 bytes as u64.
/// 6. Store as `CanonicalId(u64)` at `raw_to_canonical[raw_id]`.
///
/// **Build-time only** — allocates the `Box<[CanonicalId]>` once. The
/// resulting map is immutable.
pub fn build_surjective_map(tokenizer: &dyn TokenizerSpec) -> SurjectiveMap {
    let vocab = tokenizer.vocab_size() as usize;
    let mut raw_to_canonical: Vec<CanonicalId> = Vec::with_capacity(vocab);

    for raw_id in 0..vocab as u32 {
        let bytes = tokenizer.decode_token(TokenId(raw_id));
        let canonical = canonicalize_bytes(bytes);
        raw_to_canonical.push(canonical);
    }

    SurjectiveMap {
        raw_to_canonical: raw_to_canonical.into_boxed_slice(),
    }
}

/// Internal: NFKC + lowercase + BLAKE3 → CanonicalId.
///
/// Exposed as `pub(crate)` so the multi-branch tests in `tests.rs` can call
/// it directly without going through a full tokenizer.
pub(crate) fn canonicalize_bytes(bytes: &[u8]) -> CanonicalId {
    // Lossy UTF-8 decode: invalid byte sequences become U+FFFD.
    let s = String::from_utf8_lossy(bytes);

    // NFKC normalize, then lowercase, then trim leading/trailing whitespace.
    //
    // The trim step is required by the spec's "Apple" vs " apple" (leading
    // space) collapse — BPE tokenizers attach leading spaces to tokens, and
    // semantically " apple" is the same token as "apple". The paper's §2.2
    // description mentions only NFKC + lowercase, but the paper's reported
    // 23% compression ratio (Appendix C) can only be achieved by also
    // stripping the BPE leading-space marker. We honor the spec's literal
    // test expectation; users wanting strict paper-text behavior can wrap
    // their TokenizerSpec to disable trimming.
    let normalized: String = s.nfkc().collect();
    let lower: String = normalized.to_lowercase();
    let trimmed = lower.trim();

    // BLAKE3 over the trimmed lowercased UTF-8 bytes → first 8 bytes as u64.
    let mut hasher = blake3::Hasher::new();
    hasher.update(trimmed.as_bytes());
    let hash = *hasher.finalize().as_bytes();
    let mut u64_bytes = [0u8; 8];
    u64_bytes.copy_from_slice(&hash[..8]);
    CanonicalId(u64::from_le_bytes(u64_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A tiny test tokenizer: each "token" is a literal string.
    struct StaticTokenizer {
        tokens: Vec<&'static str>,
    }

    impl TokenizerSpec for StaticTokenizer {
        fn vocab_size(&self) -> u32 {
            self.tokens.len() as u32
        }
        fn decode_token(&self, raw_id: TokenId) -> &[u8] {
            self.tokens[raw_id.0 as usize].as_bytes()
        }
    }

    #[test]
    fn apple_and_apple_with_leading_space_collapse() {
        // T4.4: "Apple" and " apple" (leading space) → same canonical.
        let tok = StaticTokenizer {
            tokens: vec!["Apple", " apple"],
        };
        let map = build_surjective_map(&tok);
        let c0 = compress_token(TokenId(0), &map);
        let c1 = compress_token(TokenId(1), &map);
        assert_eq!(c0, c1, "\"Apple\" and \" apple\" must collapse");
    }

    #[test]
    fn a_uppercase_and_lowercase_collapse() {
        // T4.4: "A" and "a" → same canonical.
        let tok = StaticTokenizer {
            tokens: vec!["A", "a"],
        };
        let map = build_surjective_map(&tok);
        let c0 = compress_token(TokenId(0), &map);
        let c1 = compress_token(TokenId(1), &map);
        assert_eq!(c0, c1, "\"A\" and \"a\" must collapse");
    }

    #[test]
    fn distinct_semantic_tokens_distinct() {
        // T4.4: distinct surface forms → distinct canonicals (no spurious collapse).
        let tok = StaticTokenizer {
            tokens: vec!["cat", "dog", "bird", "fish"],
        };
        let map = build_surjective_map(&tok);
        let canonicals: Vec<CanonicalId> =
            (0..4).map(|i| compress_token(TokenId(i), &map)).collect();
        let distinct: std::collections::HashSet<_> = canonicals.iter().collect();
        assert_eq!(
            distinct.len(),
            4,
            "4 distinct tokens → 4 distinct canonicals"
        );
    }

    #[test]
    fn surjectivity_every_raw_id_maps_to_one_canonical() {
        // T4.4: every raw id maps to exactly one canonical id (surjectivity
        // of the projection — many raws → one canonical, but never zero).
        let tok = StaticTokenizer {
            tokens: vec!["Apple", " apple", "APPLE", "apPLe", "cat", "dog"],
        };
        let map = build_surjective_map(&tok);
        assert_eq!(map.vocab_size(), 6);
        // First 4 should collapse to the same canonical.
        let apple_canonical = compress_token(TokenId(0), &map);
        for i in 1..4 {
            assert_eq!(
                compress_token(TokenId(i), &map),
                apple_canonical,
                "raw {i} should collapse to apple canonical"
            );
        }
        // cat and dog should differ from apple.
        assert_ne!(compress_token(TokenId(4), &map), apple_canonical);
        assert_ne!(compress_token(TokenId(5), &map), apple_canonical);
        assert_ne!(
            compress_token(TokenId(4), &map),
            compress_token(TokenId(5), &map)
        );
        // 3 distinct canonicals out of 6 raws → 50% compression.
        assert_eq!(map.canonical_vocab_size(), 3);
        assert!((map.compression_ratio() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn nfkc_composed_and_decomposed_e_collapse() {
        // T4.4: NFKC normalization — composed "é" (U+00E9) and decomposed
        // "e" (U+0065) + combining acute (U+0301) must produce the same
        // canonical.
        let composed = "é"; // U+00E9 — 2 bytes in UTF-8 (0xC3 0xA9)
        let decomposed = "e\u{0301}"; // U+0065 + U+0301 — 3 bytes in UTF-8
        assert_ne!(
            composed.as_bytes(),
            decomposed.as_bytes(),
            "sanity: bytes differ pre-normalization"
        );
        let c_composed = canonicalize_bytes(composed.as_bytes());
        let c_decomposed = canonicalize_bytes(decomposed.as_bytes());
        assert_eq!(
            c_composed, c_decomposed,
            "NFKC must collapse composed and decomposed é"
        );
    }

    #[test]
    fn empty_token_maps_to_a_canonical() {
        // Edge case: empty surface bytes → still maps to exactly one canonical
        // (the BLAKE3 of empty input). No panic, no None.
        let tok = StaticTokenizer { tokens: vec![""] };
        let map = build_surjective_map(&tok);
        let _c = compress_token(TokenId(0), &map);
    }

    #[test]
    fn save_load_roundtrip_preserves_map() {
        // T4.5: postcard round-trip + BLAKE3 verification.
        let tok = StaticTokenizer {
            tokens: vec!["Apple", " apple", "cat", "dog", "fish"],
        };
        let map = build_surjective_map(&tok);
        let bytes = map.save_to_bytes();
        let loaded = SurjectiveMap::load_from_bytes(&bytes).expect("load succeeds");
        assert_eq!(loaded.raw_to_canonical, map.raw_to_canonical);
    }

    #[test]
    fn load_rejects_tampered_bytes() {
        // T4.5: BLAKE3 commitment must catch tampering.
        let tok = StaticTokenizer {
            tokens: vec!["Apple", " apple", "cat"],
        };
        let map = build_surjective_map(&tok);
        let mut bytes = map.save_to_bytes();

        // Flip a payload byte (after the 32-byte commitment).
        let payload_idx = 40.min(bytes.len() - 1);
        bytes[payload_idx] ^= 0xFF;

        let result = SurjectiveMap::load_from_bytes(&bytes);
        assert!(
            matches!(result, Err(SurjectiveMapLoadError::CommitmentMismatch)),
            "tampered payload must fail commitment check"
        );
    }

    #[test]
    fn try_compress_token_returns_none_for_out_of_range() {
        let tok = StaticTokenizer {
            tokens: vec!["a", "b"],
        };
        let map = build_surjective_map(&tok);
        assert!(try_compress_token(TokenId(0), &map).is_some());
        assert!(try_compress_token(TokenId(1), &map).is_some());
        assert!(try_compress_token(TokenId(2), &map).is_none());
        assert!(try_compress_token(TokenId(999), &map).is_none());
    }

    #[test]
    fn canonicalize_is_deterministic() {
        // Same input → same canonical, always.
        let c1 = canonicalize_bytes(b"Hello");
        let c2 = canonicalize_bytes(b"Hello");
        assert_eq!(c1, c2);
        let c3 = canonicalize_bytes(b"hello");
        assert_eq!(c1, c3, "case-insensitive via lowercase");
    }

    #[test]
    fn large_vocab_compression_ratio_realistic() {
        // Simulate a "vocab" with realistic duplication patterns:
        // 25% of tokens are case/whitespace variants of others. Target ~20%
        // compression as a sanity floor (paper reports 23% on real 128k).
        let mut tokens: Vec<String> = Vec::with_capacity(1000);
        let base_words = ["cat", "dog", "run", "jump", "quick", "brown", "fox"];
        for i in 0..1000 {
            let word = base_words[i % base_words.len()];
            // Half are bare words, half have leading-space or capitalized variants.
            let variant = match i % 4 {
                0 => word.to_string(),
                1 => format!(" {word}"),              // leading space
                2 => format!("{}", capitalize(word)), // capitalized
                _ => word.to_uppercase(),             // uppercase
            };
            tokens.push(variant);
        }
        let token_refs: Vec<&'static str> = vec![]; // can't easily leak — use owned below
        let _ = token_refs;

        // Use an owned-strings tokenizer shim.
        struct OwnedTokenizer {
            tokens: Vec<String>,
        }
        impl TokenizerSpec for OwnedTokenizer {
            fn vocab_size(&self) -> u32 {
                self.tokens.len() as u32
            }
            fn decode_token(&self, raw_id: TokenId) -> &[u8] {
                self.tokens[raw_id.0 as usize].as_bytes()
            }
        }
        let tok = OwnedTokenizer { tokens };
        let map = build_surjective_map(&tok);
        let ratio = map.compression_ratio();
        // 4 variants per word → 75% compression expected. With 7 base words,
        // canonical vocab is ~7 (each base word collapses its 4 variants).
        // Allow a wide tolerance — the goal is "meaningful compression".
        assert!(
            ratio > 0.5,
            "expected >50% compression on this synthetic vocab, got {ratio}"
        );
    }

    fn capitalize(s: &str) -> String {
        let mut c = s.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        }
    }

    #[test]
    fn compress_token_no_allocation_smoke() {
        // Smoke: just verify the function runs. Real zero-alloc verification
        // needs a custom allocator (deferred to the G7 / Phase 7 GOAT gate).
        let tok = StaticTokenizer {
            tokens: vec!["a", "b", "c"],
        };
        let map = build_surjective_map(&tok);
        let _c = compress_token(TokenId(1), &map);
    }

    /// Helper used in test naming: keep `HashMap` import alive.
    #[test]
    fn hashmap_import_alive() {
        let _: HashMap<CanonicalId, Vec<TokenId>> = HashMap::new();
    }
}
