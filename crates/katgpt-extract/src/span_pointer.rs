//! `span_pointer` — candidate-window span scoring (Plan 628 T0.2).
//!
//! The F0 span primitive of the extraction substrate ([`crate`] charter):
//! given one immutable `state` string, [`SpanPointer::new`] tokenizes it once
//! (cold) and precomputes candidate windows — token n-gram spans up to
//! [`SpanConfig::max_window_tokens`] tokens long, strided by
//! [`SpanConfig::window_stride`], aligned to token boundaries, each carried as
//! an EXACT byte range into the original state. [`SpanPointer::score_into`]
//! then scores those windows against a [`SpanQuery`] (field text + alias
//! table; callers pass `FieldConstraints.choices` from the
//! `katgpt_core::extraction_wire` vocabulary) into a caller-owned top-k
//! buffer.
//!
//! # Scoring channels
//!
//! The score is `sigmoid(w_lex * lexical + w_embed * embedding + w_alias *
//! alias_hit)` — SIGMOID normalization only, never softmax (the repo-wide
//! law). Each channel is O(1)-ish per window at hot time:
//!
//! - **Lexical** — Okapi BM25 over windows-as-documents: `k1 =`
//!   [`BM25_K1`], `b =` [`BM25_B`]; idf is window-level document frequency
//!   (`ln(1 + (N - df + 0.5) / (df + 0.5))`), tf is the term's count inside
//!   the window, length normalization uses the window's token count against
//!   the mean. The raw sum is divided by its per-query ceiling
//!   (`Σ idf·(k1+1)`), so the channel lands in `[0, 1)` and combines with
//!   the others on one scale. Query and state text go through the SAME
//!   tokenizer.
//! - **Embedding** — a hashed-feature sketch: word unigrams + byte
//!   trigrams (of the lowercased token) hashed with FNV-1a 64 (documented
//!   choice), folded sign-hashed (the hash's top bit is the sign, as in
//!   random-projection sketches) into a fixed [`EMBED_DIM`]-dim vector; the
//!   channel is the cosine between the query sketch (field text + alias
//!   text, folded uncapped) and the window sketch (precomputed per-token
//!   feature hashes).
//! - **Alias** — the fraction of distinct alias terms present in the
//!   window. Aliases also EXTEND the query term set of the lexical channel
//!   (exact token match), so a choice vocabulary contributes both as terms
//!   and as a presence signal.
//!
//! # Laws
//!
//! - **Byte offsets are exact.** Every [`SpanScore`] satisfies
//!   `state[start..end] == window text`; offsets are BYTE offsets into the
//!   original state and always land on char boundaries (windows are
//!   token-aligned and the tokenizer never splits a multibyte sequence).
//!   The wire counterpart is `extraction_wire::Span` (`start <= end <=
//!   state.len()`, char boundaries) — the shapes are field-identical.
//! - **Zero-alloc hot path.** `score_into` / `best_span` allocate nothing:
//!   query terms live in fixed stack slots, sketches in fixed
//!   [`EMBED_DIM`] buffers, top-k insertion is in-place
//!   (`copy_within`). The tokenizer is a lazy iterator.
//! - **Deterministic.** Same input -> identical scores: no randomness, no
//!   clock, one fixed tie rule, sequential f32 folds in a fixed order
//!   (same-build/platform determinism; `ln`/`exp` come from std).
//! - **Tie rule.** Output is sorted descending by score; ties break by
//!   earlier byte `start`, then by shorter span (`end - start`).
//!   Overlapping windows may all appear; the caller dedups if needed.
//! - **Sigmoid bounds.** Scores live in the open interval (0, 1); a
//!   window that matches nothing scores exactly `sigmoid(0) = 0.5`, which
//!   is why [`SpanConfig::min_score`] defaults to 0.5 — anything below the
//!   default failed to fire at least one channel.
//!
//! # Documented caps and approximations
//!
//! - Query terms (field text first, then aliases in order) are collected
//!   into at most [`MAX_QUERY_TERMS`] distinct slots; overflow is dropped
//!   first-come-first-served (deterministic). Aliases dedupe against the
//!   field text terms — a repeated term keeps one slot, flagged as alias.
//! - Terms longer than [`MAX_TERM_BYTES`] bytes are skipped for the
//!   lexical/alias channels (they still feed the embedding, which needs no
//!   buffer).
//! - Case folding is ASCII-only; non-ASCII bytes compare byte-exact
//!   (deterministic and allocation-free).
//! - An alias term of at least [`ALIAS_PREFIX_MIN`] bytes also hits a
//!   LONGER state token it prefixes (`"corp"` hits `"corporation"`) — the
//!   abbreviation case for choice vocabularies. The prefix rule affects
//!   the alias channel only; the lexical channel stays exact-token so
//!   BM25 statistics are never inflated.
//! - Degenerate config is sanitized: `0` stride/window limits clamp to 1,
//!   non-finite weights/min_score fall back to their documented defaults.
//!
//! The G5 external BLAKE3-pinned fixture vs a rule floor (Plan 628 T0.6)
//! binds this module's GOAT — not here.

use std::collections::HashMap;

/// Sketch dimensionality — fixed by contract (documented in the module doc).
pub const EMBED_DIM: usize = 256;

/// Default candidate-window length cap, in tokens.
pub const DEFAULT_MAX_WINDOW_TOKENS: usize = 8;

/// Default distance between consecutive window starts, in tokens.
pub const DEFAULT_WINDOW_STRIDE: usize = 1;

/// Default score gate: windows scoring below it are dropped before top-k
/// selection. `sigmoid(0) = 0.5` is the nothing-fired boundary, so the
/// default keeps only windows where at least one channel contributed.
pub const DEFAULT_MIN_SCORE: f32 = 0.5;

/// Default lexical (BM25) channel weight.
pub const DEFAULT_W_LEX: f32 = 0.5;

/// Default embedding (hashed-feature cosine) channel weight.
pub const DEFAULT_W_EMBED: f32 = 0.3;

/// Default alias-presence channel weight.
pub const DEFAULT_W_ALIAS: f32 = 0.2;

/// Okapi BM25 term-frequency saturation constant.
pub const BM25_K1: f32 = 1.2;

/// Okapi BM25 length-normalization constant.
pub const BM25_B: f32 = 0.75;

/// Maximum distinct query terms (field text + aliases combined). Overflow
/// is dropped first-come-first-served — deterministic, documented in the
/// module doc.
const MAX_QUERY_TERMS: usize = 64;

/// Maximum term byte length considered by the lexical/alias channels.
/// Longer terms still feed the embedding (which hashes without buffers).
const MAX_TERM_BYTES: usize = 96;

/// Minimum byte length for the alias prefix rule (an alias term this long
/// or longer also matches a longer state token it prefixes). Below this the
/// rule would flood every window with spurious hits.
const ALIAS_PREFIX_MIN: usize = 3;

/// FNV-1a 64 offset basis (the documented feature hash).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64 prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Scorer configuration. Plain struct, public fields; [`SpanConfig::default`]
/// carries the documented defaults. Degenerate values are sanitized at
/// [`SpanPointer::new`] (the sanitized copy is what [`SpanPointer::config`]
/// returns).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpanConfig {
    /// Longest candidate window, in tokens (sanitized: `0` -> `1`).
    pub max_window_tokens: usize,
    /// Distance between consecutive window starts, in tokens (sanitized:
    /// `0` -> `1`).
    pub window_stride: usize,
    /// Windows scoring below this are dropped BEFORE top-k selection.
    /// Non-finite falls back to [`DEFAULT_MIN_SCORE`].
    pub min_score: f32,
    /// Lexical channel weight. Non-finite falls back to [`DEFAULT_W_LEX`].
    pub w_lex: f32,
    /// Embedding channel weight. Non-finite falls back to
    /// [`DEFAULT_W_EMBED`].
    pub w_embed: f32,
    /// Alias channel weight. Non-finite falls back to [`DEFAULT_W_ALIAS`].
    pub w_alias: f32,
}

impl Default for SpanConfig {
    fn default() -> Self {
        SpanConfig {
            max_window_tokens: DEFAULT_MAX_WINDOW_TOKENS,
            window_stride: DEFAULT_WINDOW_STRIDE,
            min_score: DEFAULT_MIN_SCORE,
            w_lex: DEFAULT_W_LEX,
            w_embed: DEFAULT_W_EMBED,
            w_alias: DEFAULT_W_ALIAS,
        }
    }
}

/// One scored candidate window — exact byte offsets into the ORIGINAL state
/// string (`state[start..end]` is the window text; offsets are byte offsets
/// on char boundaries).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpanScore {
    /// Inclusive start byte offset.
    pub start: u32,
    /// Exclusive end byte offset.
    pub end: u32,
    /// Sigmoid-normalized score, in the open interval (0, 1).
    pub score: f32,
}

/// The query side of one field: the field's text plus its alias table
/// (callers pass `FieldConstraints.choices` surfaces here, split however the
/// engine authors them).
#[derive(Clone, Copy, Debug)]
pub struct SpanQuery<'a> {
    /// Field text — tokenized with the SAME tokenizer as the state.
    pub text: &'a str,
    /// Alias surfaces (e.g. `choices`); their terms extend the lexical term
    /// set and their presence fraction forms the alias channel. Their text
    /// also folds into the query embedding, uncapped.
    pub aliases: &'a [&'a str],
}

impl<'a> SpanQuery<'a> {
    /// A query with no alias table.
    pub fn new(text: &'a str) -> Self {
        SpanQuery { text, aliases: &[] }
    }
}

/// One token of the cold precomputation. `lower_*` index the concatenated
/// lowercased token buffer; `feat_*` index the concatenated per-token
/// feature-hash buffer.
struct TokenRec {
    start: u32,
    end: u32,
    lower_lo: u32,
    lower_hi: u32,
    feat_lo: u32,
    feat_hi: u32,
}

/// One candidate window: the byte range it covers and the token range it
/// spans. Byte ranges are exact against the original state.
struct WindowRec {
    start: u32,
    end: u32,
    tok_lo: u32,
    tok_hi: u32,
}

/// A per-state span scorer. Build once over a state string (cold: tokenize
/// + index + windows), then score many queries against it (hot: zero-alloc).
///
/// Not `Clone`/`Debug` on purpose: it owns a whole-state index; rebuild from
/// the same inputs instead (deterministic — identical arrays, see the
/// determinism test).
pub struct SpanPointer {
    cfg: SpanConfig,
    tokens: Vec<TokenRec>,
    lower_buf: String,
    features: Vec<u64>,
    windows: Vec<WindowRec>,
    df: HashMap<Box<str>, u32>,
    n_windows: f32,
    avgdl: f32,
}

impl SpanPointer {
    /// COLD path: tokenize `state` once, build the term -> document-frequency
    /// index (windows are the documents), and enumerate candidate windows —
    /// token n-grams of every length `1..=max_window_tokens` starting at
    /// every `window_stride`-th token, each carried as an exact byte range.
    ///
    /// Per-window storage is the byte range + token range only: the window
    /// count is O(tokens * max_window_tokens) — linear in the state, never
    /// O(n^2) — and heavier per-window data (token features, term forms)
    /// lives in flat per-token buffers the windows index into.
    pub fn new(state: &str, cfg: SpanConfig) -> Self {
        // Sanitize the config once; everything downstream reads the copy.
        let max_window_tokens = cfg.max_window_tokens.max(1);
        let window_stride = cfg.window_stride.max(1);
        let min_score = sanitize_f32(cfg.min_score, DEFAULT_MIN_SCORE);
        let w_lex = sanitize_f32(cfg.w_lex, DEFAULT_W_LEX);
        let w_embed = sanitize_f32(cfg.w_embed, DEFAULT_W_EMBED);
        let w_alias = sanitize_f32(cfg.w_alias, DEFAULT_W_ALIAS);
        let cfg = SpanConfig {
            max_window_tokens,
            window_stride,
            min_score,
            w_lex,
            w_embed,
            w_alias,
        };

        // Tokenize once; keep lowercase forms and feature hashes in flat
        // buffers the token records index into (no per-token allocations
        // beyond the two growing buffers).
        let state_bytes = state.as_bytes();
        let mut tokens = Vec::new();
        let mut lower_buf: Vec<u8> = Vec::new();
        let mut features: Vec<u64> = Vec::new();
        for (a, b) in Tokens::new(state) {
            let tok = &state_bytes[a..b];
            let lower_lo = lower_buf.len() as u32;
            lower_buf.extend(tok.iter().map(|&c| c.to_ascii_lowercase()));
            let lower_hi = lower_buf.len() as u32;
            let feat_lo = features.len() as u32;
            push_token_features(
                &lower_buf[lower_lo as usize..lower_hi as usize],
                &mut features,
            );
            let feat_hi = features.len() as u32;
            tokens.push(TokenRec {
                start: a as u32,
                end: b as u32,
                lower_lo,
                lower_hi,
                feat_lo,
                feat_hi,
            });
        }

        // Candidate windows: every length 1..=max starting at every
        // stride-th token. Generation order (start asc, length asc) is the
        // deterministic iteration order of the hot loop.
        let n_tok = tokens.len();
        let mut windows = Vec::new();
        let mut start = 0usize;
        while start < n_tok {
            let hi = (start + max_window_tokens).min(n_tok);
            for len in 1..=(hi - start) {
                let first = &tokens[start];
                let last = &tokens[start + len - 1];
                windows.push(WindowRec {
                    start: first.start,
                    end: last.end,
                    tok_lo: start as u32,
                    tok_hi: (start + len) as u32,
                });
            }
            start += window_stride;
        }

        // Window-level document frequencies (a term counts once per window
        // even when repeated inside it) for BM25 idf.
        let mut df: HashMap<Box<str>, u32> = HashMap::with_capacity(windows.len());
        let mut total_len = 0u64;
        for w in &windows {
            let lo = w.tok_lo as usize;
            let hi = w.tok_hi as usize;
            total_len += (hi - lo) as u64;
            for i in lo..hi {
                let term = &lower_buf[tokens[i].lower_lo as usize..tokens[i].lower_hi as usize];
                let dup = (lo..i).any(|j| {
                    let prev = &lower_buf[tokens[j].lower_lo as usize..tokens[j].lower_hi as usize];
                    prev == term
                });
                if dup {
                    continue;
                }
                // lower_buf holds ASCII-lowercased token bytes; ASCII
                // lowering preserves UTF-8, so this cannot fail.
                let term_str = std::str::from_utf8(term).expect("ascii lowering preserves utf-8");
                df.entry(term_str.into()).or_insert(0);
                if let Some(count) = df.get_mut(term_str) {
                    *count += 1;
                }
            }
        }

        let n_windows = windows.len() as f32;
        let avgdl = if windows.is_empty() {
            1.0
        } else {
            total_len as f32 / n_windows
        };

        SpanPointer {
            cfg,
            tokens,
            lower_buf: String::from_utf8(lower_buf).expect("ascii lowering preserves utf-8"),
            features,
            windows,
            df,
            n_windows,
            avgdl,
        }
    }

    /// The SANITIZED config this scorer runs under (see [`SpanConfig`]).
    pub fn config(&self) -> &SpanConfig {
        &self.cfg
    }

    /// Number of candidate windows precomputed at construction — also the
    /// buffer size needed for a full sweep at `min_score = 0.0`.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// HOT path: score every candidate window against `query`, keeping the
    /// top-k in `out` (k = `out.len()`), zero-allocation.
    ///
    /// Returns the number of slots filled (0 when the query carries no
    /// usable terms, the state has no tokens, or fewer than k windows clear
    /// the `min_score` gate). The filled prefix is sorted descending by
    /// score; ties break by earlier byte `start`, then by shorter span.
    /// Overlapping windows may all appear — the caller dedups if needed.
    pub fn score_into(&self, query: &SpanQuery<'_>, out: &mut [SpanScore]) -> usize {
        if out.is_empty() {
            return 0;
        }

        // Query terms: field text first, then aliases in order, deduped
        // into fixed stack slots (documented cap).
        let mut qterms = [QueryTerm::EMPTY; MAX_QUERY_TERMS];
        let mut filled = 0usize;
        self.collect_terms(query.text, false, &mut qterms, &mut filled);
        for alias in query.aliases {
            self.collect_terms(alias, true, &mut qterms, &mut filled);
        }
        let alias_total = qterms[..filled].iter().filter(|q| q.is_alias).count();

        // Query sketch: field text + alias text, folded uncapped.
        let mut qacc = [0.0f32; EMBED_DIM];
        fold_str_features(query.text, &mut qacc);
        for alias in query.aliases {
            fold_str_features(alias, &mut qacc);
        }
        let qnorm = l2_normalize(&mut qacc);

        // Nothing to score against.
        if filled == 0 && qnorm <= 0.0 {
            return 0;
        }

        // Channel activation gates (skip whole loops when a channel is off).
        let want_lex = filled > 0 && (self.cfg.w_lex != 0.0 || alias_total > 0);
        let want_embed = self.cfg.w_embed != 0.0 && qnorm > 0.0;

        let mut filled_out = 0usize;
        let mut wacc = [0.0f32; EMBED_DIM];
        let mut tf = [0u8; MAX_QUERY_TERMS];
        let mut alias_prefix = [false; MAX_QUERY_TERMS];

        for w in &self.windows {
            let lo = w.tok_lo as usize;
            let hi = w.tok_hi as usize;
            let dl = (hi - lo) as f32;

            let (lex, alias_frac) = if want_lex {
                tf[..filled].fill(0);
                alias_prefix[..filled].fill(false);
                for t in &self.tokens[lo..hi] {
                    let tok = &self.lower_buf.as_bytes()[t.lower_lo as usize..t.lower_hi as usize];
                    for (i, qt) in qterms[..filled].iter().enumerate() {
                        if qt.exact_matches(tok) {
                            tf[i] = tf[i].saturating_add(1);
                        } else if qt.prefix_matches(tok) {
                            alias_prefix[i] = true;
                        }
                    }
                }
                let norm = 1.0 - BM25_B + BM25_B * dl / self.avgdl;
                let mut num = 0.0f32;
                let mut den = 0.0f32;
                let mut hits = 0usize;
                for (i, qt) in qterms[..filled].iter().enumerate() {
                    den += qt.idf * (BM25_K1 + 1.0);
                    let f = f32::from(tf[i]);
                    num += qt.idf * (f * (BM25_K1 + 1.0)) / (f + BM25_K1 * norm);
                    if qt.is_alias && (tf[i] > 0 || alias_prefix[i]) {
                        hits += 1;
                    }
                }
                let lex = if den > 0.0 { num / den } else { 0.0 };
                let alias_frac = if alias_total > 0 {
                    hits as f32 / alias_total as f32
                } else {
                    0.0
                };
                (lex, alias_frac)
            } else {
                (0.0, 0.0)
            };

            let embed = if want_embed {
                wacc.fill(0.0);
                for t in &self.tokens[lo..hi] {
                    for &h in &self.features[t.feat_lo as usize..t.feat_hi as usize] {
                        feature_bump(&mut wacc, h);
                    }
                }
                let wnorm = l2_normalize(&mut wacc);
                if wnorm > 0.0 { dot(&qacc, &wacc) } else { 0.0 }
            } else {
                0.0
            };

            let logit =
                self.cfg.w_lex * lex + self.cfg.w_embed * embed + self.cfg.w_alias * alias_frac;
            let score = sigmoid(logit);
            if score < self.cfg.min_score {
                continue;
            }
            insert_sorted(
                out,
                &mut filled_out,
                SpanScore {
                    start: w.start,
                    end: w.end,
                    score,
                },
            );
        }

        filled_out
    }

    /// Convenience: the single best window above the gate, or `None` when
    /// nothing scored (single-slot [`SpanPointer::score_into`]).
    pub fn best_span(&self, query: &SpanQuery<'_>) -> Option<SpanScore> {
        let mut slot = [SpanScore {
            start: 0,
            end: 0,
            score: 0.0,
        }];
        if self.score_into(query, &mut slot) == 1 {
            Some(slot[0])
        } else {
            None
        }
    }

    /// The Bench-950 round-3 GOAT posture in one call: [`Self::best_span`]
    /// LOCALIZES (the pointer's containment edge), then [`snap_longest`]
    /// RESOLVES the boundaries against the query's own schema vocabulary
    /// (`query.text` first, then `query.aliases` in order — the same inputs
    /// the window was scored with). No qualifying surface inside the window
    /// → the raw window stands (the measured fallback law: the hybrid stays
    /// pointer-derived). The score rides along unchanged — the snap is a
    /// boundary refinement, not a re-rank.
    ///
    /// `text` is the SAME string the pointer was constructed from (the
    /// pointer keeps only the lowercased token buffer, so the original —
    /// whose byte offsets the windows index — must come from the caller).
    pub fn best_snapped_span<'a>(&self, text: &str, query: &SpanQuery<'a>) -> Option<SpanScore> {
        let best = self.best_span(query)?;
        let snapped = snap_longest(
            text,
            (best.start, best.end),
            std::iter::once(query.text).chain(query.aliases.iter().copied()),
        );
        Some(match snapped {
            Some((start, end)) => SpanScore {
                start,
                end,
                score: best.score,
            },
            None => best,
        })
    }

    /// The Bench-950 round-4 GOAT-winning consumer pattern in one call
    /// (the control arm): the literal gazetteer — longest match over
    /// `query.text` + `query.aliases`, [`longest_match`] — answers FIRST;
    /// where the vocabulary has no literal occurrence, the pointer's best
    /// window ([`Self::best_span`]) answers. `None` only when BOTH abstain.
    ///
    /// Measured on two domains (Bench 950 round 4): this pattern beats the
    /// snap hybrid and the bare pointer on choices-class exact (movie 2321,
    /// restaurant 1493) — the pointer's value concentrates exactly where
    /// the vocabulary has no literal occurrence, and this composite is the
    /// only shipped shape that captures it. The source that answered is
    /// disclosed ([`SpanAnswerSource`]) — the floor is a literal match, the
    /// fallback a ranked window; downstream confidence handling differs
    /// (the fallback rides the pointer's gate + Temper's fused abstain; the
    /// floor needs no gate — a literal hit IS the answer).
    ///
    /// Zero-allocation. `text` is the SAME string the pointer was
    /// constructed from (as in [`Self::best_snapped_span`]).
    pub fn floor_first_span(&self, text: &str, query: &SpanQuery<'_>) -> Option<GroundedSpan> {
        let (start, end, source) = longest_match(
            text,
            std::iter::once(query.text).chain(query.aliases.iter().copied()),
        )
        .map(|(s, e)| (s, e, SpanAnswerSource::Floor))
        .or_else(|| {
            self.best_span(query)
                .map(|b| (b.start, b.end, SpanAnswerSource::PointerFallback))
        })?;
        Some(GroundedSpan { start, end, source })
    }

    /// Collect the lowercased terms of `text` into `out` at `*filled`,
    /// deduping against already-collected slots (an alias duplicate flags
    /// the existing slot instead of adding one). Terms past the slot cap or
    /// the byte-length cap are skipped — both documented, both deterministic.
    fn collect_terms(&self, text: &str, is_alias: bool, out: &mut [QueryTerm], filled: &mut usize) {
        let bytes = text.as_bytes();
        for (a, b) in Tokens::new(text) {
            if *filled >= out.len() {
                return;
            }
            let tok = &bytes[a..b];
            if tok.len() > MAX_TERM_BYTES {
                continue;
            }
            let mut buf = [0u8; MAX_TERM_BYTES];
            for (dst, &src) in buf.iter_mut().zip(tok.iter()) {
                *dst = src.to_ascii_lowercase();
            }
            let term = &buf[..tok.len()];
            let existing = out[..*filled]
                .iter_mut()
                .find(|q| q.len as usize == term.len() && &q.bytes[..q.len as usize] == term);
            if let Some(slot) = existing {
                if is_alias {
                    slot.is_alias = true;
                }
                continue;
            }
            let df = std::str::from_utf8(term)
                .ok()
                .and_then(|s| self.df.get(s))
                .copied()
                .unwrap_or(0);
            out[*filled] = QueryTerm {
                bytes: buf,
                len: tok.len() as u8,
                idf: idf_of(self.n_windows, df),
                is_alias,
            };
            *filled += 1;
        }
    }
}

/// One query term held on the stack during scoring: the lowercased bytes,
/// their length, the precomputed BM25 idf, and whether the term came from
/// the alias table (alias terms may also prefix-match longer tokens).
struct QueryTerm {
    bytes: [u8; MAX_TERM_BYTES],
    len: u8,
    idf: f32,
    is_alias: bool,
}

impl QueryTerm {
    const EMPTY: QueryTerm = QueryTerm {
        bytes: [0; MAX_TERM_BYTES],
        len: 0,
        idf: 0.0,
        is_alias: false,
    };

    /// Exact token match (both sides already ASCII-lowercased).
    #[inline]
    fn exact_matches(&self, tok: &[u8]) -> bool {
        self.len as usize == tok.len() && &self.bytes[..self.len as usize] == tok
    }

    /// Alias prefix rule: this alias term hits a strictly longer token it
    /// prefixes (only when long enough — [`ALIAS_PREFIX_MIN`]).
    #[inline]
    fn prefix_matches(&self, tok: &[u8]) -> bool {
        let l = self.len as usize;
        self.is_alias && l >= ALIAS_PREFIX_MIN && tok.len() > l && tok.starts_with(&self.bytes[..l])
    }
}

/// Lazy word tokenizer: yields `(start, end)` BYTE ranges of maximal runs of
/// word characters. A word byte is ASCII alphanumeric OR any non-ASCII byte —
/// a multibyte sequence is therefore never split, so ranges are always char
/// boundaries and slices of the original string are safe. ASCII
/// non-alphanumerics (spaces, punctuation) split. No allocation; the same
/// iterator serves the cold state tokenize and the hot query tokenize.
struct Tokens<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Tokens<'a> {
    fn new(s: &'a str) -> Self {
        Tokens {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }
}

impl Iterator for Tokens<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        let b = self.bytes;
        while self.pos < b.len() && !is_word_byte(b[self.pos]) {
            self.pos += 1;
        }
        if self.pos >= b.len() {
            return None;
        }
        let start = self.pos;
        while self.pos < b.len() && is_word_byte(b[self.pos]) {
            self.pos += 1;
        }
        Some((start, self.pos))
    }
}

/// The tokenizer's character law: ASCII alphanumeric or non-ASCII (treated
/// as word bytes so multibyte text keeps exact byte spans).
#[inline]
fn is_word_byte(c: u8) -> bool {
    !c.is_ascii() || c.is_ascii_alphanumeric()
}

/// FNV-1a 64 over the ASCII-lowercased bytes (the documented feature hash;
/// deterministic, allocation-free).
#[inline]
fn fnv1a64_lower(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in bytes {
        h ^= u64::from(b.to_ascii_lowercase());
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Fold one hashed feature into a sketch: sign-hashed random projection —
/// the low bits pick the dimension, the top bit picks the sign.
#[inline]
fn feature_bump(acc: &mut [f32; EMBED_DIM], h: u64) {
    let idx = (h as usize) & (EMBED_DIM - 1);
    let sign = if (h >> 63) == 1 { -1.0 } else { 1.0 };
    acc[idx] += sign;
}

/// Fold a text's features (word unigrams + byte trigrams of the lowercased
/// tokens, counted per occurrence) into `acc`. Tokens shorter than three
/// bytes contribute their unigram only. Hot-path safe: no allocation.
fn fold_str_features(text: &str, acc: &mut [f32; EMBED_DIM]) {
    let bytes = text.as_bytes();
    for (a, b) in Tokens::new(text) {
        let tok = &bytes[a..b];
        feature_bump(acc, fnv1a64_lower(tok));
        if tok.len() >= 3 {
            for i in 0..=(tok.len() - 3) {
                feature_bump(acc, fnv1a64_lower(&tok[i..i + 3]));
            }
        }
    }
}

/// Cold twin of [`fold_str_features`]'s per-token half: the feature hashes
/// of one ALREADY-LOWERCASED token, appended to the flat buffer. Must stay
/// hash-identical to the hot fold (unigram + same byte trigrams).
fn push_token_features(tok_lower: &[u8], out: &mut Vec<u64>) {
    out.push(fnv1a64_lower(tok_lower));
    if tok_lower.len() >= 3 {
        for i in 0..=(tok_lower.len() - 3) {
            out.push(fnv1a64_lower(&tok_lower[i..i + 3]));
        }
    }
}

/// In-place L2 normalization; returns the norm BEFORE normalization (0.0
/// vectors are left as-is and reported as 0.0).
fn l2_normalize(v: &mut [f32; EMBED_DIM]) -> f32 {
    let mut sum = 0.0f32;
    for &x in v.iter() {
        sum += x * x;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

/// Sequential dot product (fixed order — the determinism law).
fn dot(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// BM25 inverse document frequency over windows-as-documents. Positive for
/// every `df` (including 0 — a state-absent query term still contributes to
/// the lexical ceiling, penalizing queries that mostly name things the state
/// does not contain).
#[inline]
fn idf_of(n_windows: f32, df: u32) -> f32 {
    let df = df as f32;
    (1.0 + (n_windows - df + 0.5) / (df + 0.5)).ln()
}

/// The logistic sigmoid — the ONLY normalization this module uses (never
/// softmax).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sanitize one configurable f32: non-finite falls back to the documented
/// default (deterministic; NaN comparisons would silently disable gates).
fn sanitize_f32(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

/// The strict total order of the top-k buffer: descending score, ties by
/// earlier `start`, then by shorter span. Keys are unique (window byte
/// ranges are distinct), so the order is total and the insertion result is
/// independent of iteration order.
fn rank_before(a: &SpanScore, b: &SpanScore) -> bool {
    a.score > b.score
        || (a.score == b.score && (a.start < b.start || (a.start == b.start && a.end < b.end)))
}

/// The Bench-950 round-4 boundary resolver (the GOAT-winning hybrid's
/// second half): among `surfaces`, the LONGEST occurrence fully inside
/// `window` wins; ties break by list order (strictly-greater length
/// replaces, so the earlier surface keeps a tie); per surface the leftmost
/// occurrence. The standard gazetteer longest-match law, constrained to
/// what the pointer fully saw (`ws <= os && oe <= we`). Returns `None` when
/// no surface occurs inside — the caller falls back to the window itself
/// (the hybrid stays pointer-derived; it is NOT a gazetteer ensemble).
///
/// Zero-allocation, deterministic, config-independent — a pure text
/// function over byte offsets. `window` bounds must land on `text` char
/// boundaries (asserted debug; pointer windows and word-trimmed spans are
/// token-aligned by construction). Empty surfaces never win (length 0
/// never beats the 0 seed).
pub fn snap_longest<'s, I>(text: &str, window: (u32, u32), surfaces: I) -> Option<(u32, u32)>
where
    I: IntoIterator<Item = &'s str>,
{
    let (ws, we) = (window.0 as usize, window.1 as usize);
    if ws >= we || we > text.len() {
        return None;
    }
    debug_assert!(
        text.is_char_boundary(ws) && text.is_char_boundary(we),
        "snap_longest window bounds must land on char boundaries"
    );
    let hay = &text[ws..we];
    let mut best: Option<(u32, u32)> = None;
    let mut best_len = 0usize;
    for surface in surfaces {
        // Leftmost occurrence of this surface inside the window — a match
        // within `hay` is fully inside by construction.
        if let Some((off, part)) = hay.match_indices(surface).next() {
            let len = part.len();
            if len > best_len {
                best_len = len;
                let s = ws + off;
                best = Some((s as u32, (s + len) as u32));
            }
        }
    }
    best
}

/// Who answered a [`SpanPointer::floor_first_span`] query — the Bench-950
/// round-4 disclosure the consumer needs (the two sources behave
/// differently: the floor is a literal match, the fallback a ranked
/// window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanAnswerSource {
    /// The literal longest-match gazetteer resolved it (the rule floor).
    Floor,
    /// The floor abstained — the pointer's best window answered.
    PointerFallback,
}

/// A grounded span answer: byte offsets + who resolved them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundedSpan {
    pub start: u32,
    pub end: u32,
    pub source: SpanAnswerSource,
}

/// The unconstrained gazetteer half of the round-4 control pattern: among
/// `surfaces`, the LONGEST literal occurrence anywhere in `text` wins
/// (leftmost per surface; ties by list order — strictly-greater length
/// replaces). `None` when no surface occurs at all — the abstention the
/// [`SpanPointer::floor_first_span`] fallback covers. Zero-allocation,
/// deterministic.
pub fn longest_match<'s, I>(text: &str, surfaces: I) -> Option<(u32, u32)>
where
    I: IntoIterator<Item = &'s str>,
{
    let mut best: Option<(u32, u32)> = None;
    let mut best_len = 0usize;
    for surface in surfaces {
        if let Some((s, part)) = text.match_indices(surface).next() {
            let len = part.len();
            if len > best_len {
                best_len = len;
                best = Some((s as u32, (s + len) as u32));
            }
        }
    }
    best
}

/// Fixed-size top-k insertion into a buffer kept sorted by [`rank_before`].
/// In-place (`copy_within` is a memmove) — part of the zero-alloc law.
fn insert_sorted(out: &mut [SpanScore], filled: &mut usize, cand: SpanScore) {
    let n = *filled;
    if n < out.len() {
        let mut i = 0;
        while i < n && !rank_before(&cand, &out[i]) {
            i += 1;
        }
        out.copy_within(i..n, i + 1);
        out[i] = cand;
        *filled = n + 1;
    } else {
        let last = out.len() - 1;
        if !rank_before(&cand, &out[last]) {
            return;
        }
        let mut i = last;
        while i > 0 && rank_before(&cand, &out[i - 1]) {
            out[i] = out[i - 1];
            i -= 1;
        }
        out[i] = cand;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
    use std::sync::atomic::Ordering;

    const ZERO_SPAN: SpanScore = SpanScore {
        start: 0,
        end: 0,
        score: 0.0,
    };

    fn slice<'a>(state: &'a str, s: &SpanScore) -> &'a str {
        &state[s.start as usize..s.end as usize]
    }

    fn find_span(results: &[SpanScore], text: &str, needle: &str) -> Option<SpanScore> {
        let start = text.find(needle)? as u32;
        let end = start + needle.len() as u32;
        results
            .iter()
            .copied()
            .find(|s| s.start == start && s.end == end)
    }

    fn assert_sorted_desc(out: &[SpanScore]) {
        for i in 1..out.len() {
            assert!(
                rank_before(&out[i - 1], &out[i]),
                "output not sorted by the documented tie order at {i}"
            );
        }
    }

    #[test]
    fn verbatim_mention_ranks_top_one() {
        let state = "Acme Corporation is based in Seattle. The vendor is Globex.";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery::new("Globex");
        let mut out = [ZERO_SPAN; 8];
        let n = sp.score_into(&q, &mut out);
        assert!(n >= 1);
        assert_eq!(slice(state, &out[0]), "Globex");
    }

    #[test]
    fn all_windows_slice_state_exactly() {
        let state = "Acme Corporation is based in Seattle. The vendor is Globex.";
        let sp = SpanPointer::new(
            state,
            SpanConfig {
                min_score: 0.0,
                ..SpanConfig::default()
            },
        );
        // 10 tokens, stride 1, cap 8 -> sum(min(8, 10 - s)) windows.
        let t = 10usize;
        let expected: usize = (0..t).map(|s| DEFAULT_MAX_WINDOW_TOKENS.min(t - s)).sum();
        assert_eq!(sp.window_count(), expected);

        let mut out = vec![ZERO_SPAN; expected];
        let n = sp.score_into(&SpanQuery::new("Globex"), &mut out);
        assert_eq!(n, expected);
        for s in &out[..n] {
            let (a, b) = (s.start as usize, s.end as usize);
            assert!(a < b && b <= state.len());
            assert!(state.is_char_boundary(a) && state.is_char_boundary(b));
            assert!(!&state[a..b].is_empty());
        }
        assert!(find_span(&out[..n], state, "Seattle").is_some());
    }

    #[test]
    fn alias_prefix_hit_finds_full_choice() {
        let state = "Contact Acme Corporation for details.";
        // Alias channel only: every corp-window scores sigmoid(1), every
        // other window sigmoid(0) — pure, noise-free signal.
        let cfg = SpanConfig {
            w_lex: 0.0,
            w_embed: 0.0,
            w_alias: 1.0,
            ..SpanConfig::default()
        };
        let sp = SpanPointer::new(state, cfg);
        let q = SpanQuery {
            text: "xyzzy",
            aliases: &["corp"],
        };
        let mut out = [ZERO_SPAN; 16];
        let n = sp.score_into(&q, &mut out);
        assert_eq!(n, 15); // 5 tokens -> 15 windows, all above the gate
        // The NINE windows containing "corporation" (the "corp" prefix
        // rule) rank first, in (start asc, end asc) order; the "Acme
        // Corporation" choice surface is among them verbatim.
        let expected = [
            "Contact Acme Corporation",
            "Contact Acme Corporation for",
            "Contact Acme Corporation for details",
            "Acme Corporation",
            "Acme Corporation for",
            "Acme Corporation for details",
            "Corporation",
            "Corporation for",
            "Corporation for details",
        ];
        for (got, want) in out[..9].iter().zip(expected.iter()) {
            assert_eq!(slice(state, got), *want);
        }
        for s in &out[..9] {
            assert_eq!(s.score, out[0].score, "alias hits must tie bit-exactly");
        }
        for s in &out[9..n] {
            assert_eq!(s.score, 0.5, "no-hit windows sit exactly at 0.5");
        }
        let best = sp.best_span(&q).unwrap();
        assert_eq!(slice(state, &best), "Contact Acme Corporation");
    }

    #[test]
    fn alias_hit_lifts_window_score() {
        let state = "Contact Acme Corporation for details.";
        let sp = SpanPointer::new(
            state,
            SpanConfig {
                min_score: 0.0,
                ..SpanConfig::default()
            },
        );
        let plain = SpanQuery::new("xyzzy");
        let aliased = SpanQuery {
            text: "xyzzy",
            aliases: &["corp"],
        };
        let mut a = [ZERO_SPAN; 16];
        let mut b = [ZERO_SPAN; 16];
        assert_eq!(sp.score_into(&plain, &mut a), 15);
        assert_eq!(sp.score_into(&aliased, &mut b), 15);
        let target = find_span(&a, state, "Acme Corporation").unwrap();
        let lifted = find_span(&b, state, "Acme Corporation").unwrap();
        assert!(
            lifted.score > target.score,
            "alias hit must strictly lift the same window"
        );
    }

    #[test]
    fn alias_extends_lexical_matching() {
        let state = "Acme Corp ships fast";
        let cfg = SpanConfig {
            w_lex: 1.0,
            w_embed: 0.0,
            w_alias: 0.0,
            ..SpanConfig::default()
        };
        let sp = SpanPointer::new(state, cfg);
        let q = SpanQuery {
            text: "partner",
            aliases: &["Corp"],
        };
        let mut out = [ZERO_SPAN; 10];
        let n = sp.score_into(&q, &mut out);
        assert_eq!(n, 10);
        // The alias term "corp" exact-matches the state token, so the
        // single-token window (best BM25 length normalization) ranks top.
        assert_eq!(slice(state, &out[0]), "Corp");
        assert!(find_span(&out[..n], state, "Acme Corp").is_some());
    }

    #[test]
    fn embedding_lifts_partial_overlap() {
        let state = "the invoice total is 42 dollars";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let mut out = [ZERO_SPAN; 21];
        let n = sp.score_into(&SpanQuery::new("total amount"), &mut out);
        assert!(n >= 1);
        assert!(
            slice(state, &out[0]).contains("total"),
            "partial overlap must surface a 'total' window"
        );
        assert!(out[0].score > 0.55, "embedding must lift the overlap");

        // Without the embedding channel the same query cannot clear 0.55:
        // its best lexical-only score is sigmoid(w_lex * lex) ~ 0.51.
        let lex_only = SpanPointer::new(
            state,
            SpanConfig {
                w_embed: 0.0,
                ..SpanConfig::default()
            },
        );
        let mut out2 = [ZERO_SPAN; 21];
        let n2 = lex_only.score_into(&SpanQuery::new("total amount"), &mut out2);
        assert!(n2 >= 1);
        assert!(out2[0].score < 0.55);
    }

    #[test]
    fn score_into_fills_exactly_k_sorted_desc() {
        let state = "Acme Corporation is based in Seattle. The vendor is Globex.";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let mut out = [ZERO_SPAN; 5];
        let n = sp.score_into(&SpanQuery::new("Globex"), &mut out);
        assert_eq!(n, 5);
        assert_sorted_desc(&out[..n]);
    }

    #[test]
    fn ties_break_earlier_position_then_shorter_span() {
        let state = "Contact Acme Corporation for details.";
        let cfg = SpanConfig {
            w_lex: 0.0,
            w_embed: 0.0,
            w_alias: 1.0,
            ..SpanConfig::default()
        };
        let sp = SpanPointer::new(state, cfg);
        let q = SpanQuery {
            text: "xyzzy",
            aliases: &["corp"],
        };
        let mut out = [ZERO_SPAN; 16];
        let n = sp.score_into(&q, &mut out);
        assert_eq!(n, 15);
        // Two tie groups: nine bit-equal alias hits, then six no-hit
        // windows at exactly 0.5. Within EACH equal-score group the order
        // is earlier start, then shorter span; across groups the score
        // ordering legitimately violates positional order.
        assert_eq!(out[0].score, out[8].score, "nine alias hits tie");
        assert_eq!(out[9].score, out[14].score, "no-hit windows tie");
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && out[j].score == out[i].score {
                assert!(out[j].start >= out[j - 1].start);
                if out[j].start == out[j - 1].start {
                    assert!(out[j].end >= out[j - 1].end);
                }
                j += 1;
            }
            i = j;
        }
    }

    #[test]
    fn min_score_gate_drops_weak_windows() {
        let state = "alpha beta";
        let sp = SpanPointer::new(
            state,
            SpanConfig {
                min_score: 0.55,
                ..SpanConfig::default()
            },
        );
        let mut out = [ZERO_SPAN; 4];
        let n = sp.score_into(&SpanQuery::new("alpha"), &mut out);
        assert_eq!(
            n, 2,
            "both 'alpha' windows clear 0.55, the beta window does not"
        );
        for s in &out[..n] {
            assert!(slice(state, s).contains("alpha"));
        }

        let strict = SpanPointer::new(
            state,
            SpanConfig {
                min_score: 0.7,
                ..SpanConfig::default()
            },
        );
        let mut out2 = [ZERO_SPAN; 4];
        assert_eq!(strict.score_into(&SpanQuery::new("alpha"), &mut out2), 0);
    }

    #[test]
    fn scores_stay_in_open_zero_one() {
        let state = "Acme Corporation is based in Seattle. The vendor is Globex.";
        for min_score in [0.0, 0.5] {
            let sp = SpanPointer::new(
                state,
                SpanConfig {
                    min_score,
                    ..SpanConfig::default()
                },
            );
            let mut out = [ZERO_SPAN; 64];
            let n = sp.score_into(&SpanQuery::new("Globex vendor"), &mut out);
            assert!(n > 0);
            for s in &out[..n] {
                assert!(s.score > 0.0 && s.score < 1.0);
            }
        }
    }

    #[test]
    fn determinism_same_input_identical_scores() {
        let state = "Acme Corporation is based in Seattle. The vendor is Globex.";
        let cfg = SpanConfig::default();
        let q = SpanQuery {
            text: "globex vendor",
            aliases: &["corp", "Globex"],
        };
        let a = SpanPointer::new(state, cfg);
        let b = SpanPointer::new(state, cfg);
        let mut ra = vec![ZERO_SPAN; 64];
        let mut rb = vec![ZERO_SPAN; 64];
        let mut rc = vec![ZERO_SPAN; 64];
        let na = a.score_into(&q, &mut ra);
        let nb = b.score_into(&q, &mut rb);
        let nc = a.score_into(&q, &mut rc);
        assert_eq!(na, nb);
        assert_eq!(na, nc);
        assert_eq!(ra[..na], rb[..nb]);
        assert_eq!(ra[..na], rc[..nc]);
    }

    #[test]
    fn best_span_matches_full_sweep_top() {
        let state = "Acme Corporation is based in Seattle. The vendor is Globex.";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery::new("Globex");
        let mut out = [ZERO_SPAN; 64];
        let n = sp.score_into(&q, &mut out);
        assert!(n >= 1);
        assert_eq!(sp.best_span(&q), Some(out[0]));
    }

    #[test]
    fn empty_and_separator_only_states_yield_nothing() {
        for state in ["", "!!! ... !!!"] {
            let sp = SpanPointer::new(state, SpanConfig::default());
            assert_eq!(sp.window_count(), 0);
            assert!(sp.best_span(&SpanQuery::new("anything")).is_none());
            let mut out = [ZERO_SPAN; 4];
            assert_eq!(sp.score_into(&SpanQuery::new("anything"), &mut out), 0);
        }
    }

    #[test]
    fn window_config_shapes_candidate_set() {
        let state = "a b c d";
        let count = |cfg: SpanConfig| SpanPointer::new(state, cfg).window_count();
        assert_eq!(count(SpanConfig::default()), 10);
        assert_eq!(
            count(SpanConfig {
                window_stride: 2,
                ..SpanConfig::default()
            }),
            6
        );
        assert_eq!(
            count(SpanConfig {
                max_window_tokens: 1,
                ..SpanConfig::default()
            }),
            4
        );
        // Degenerate values sanitize to 1.
        assert_eq!(
            count(SpanConfig {
                window_stride: 0,
                ..SpanConfig::default()
            }),
            10
        );
        assert_eq!(
            count(SpanConfig {
                max_window_tokens: 0,
                ..SpanConfig::default()
            }),
            4
        );
    }

    #[test]
    fn multibyte_offsets_still_slice_exactly() {
        let state = "Le café et 中文字 ok";
        let sp = SpanPointer::new(state, SpanConfig::default());
        assert_eq!(sp.window_count(), 15); // 5 tokens

        assert_eq!(
            slice(state, &sp.best_span(&SpanQuery::new("café")).unwrap()),
            "café"
        );
        assert_eq!(
            slice(state, &sp.best_span(&SpanQuery::new("中文字")).unwrap()),
            "中文字"
        );

        // Full sweep: every window stays inside the string on char
        // boundaries and slices cleanly.
        let sweep = SpanPointer::new(
            state,
            SpanConfig {
                min_score: 0.0,
                ..SpanConfig::default()
            },
        );
        let mut out = [ZERO_SPAN; 16];
        let n = sweep.score_into(&SpanQuery::new("café 中文字"), &mut out);
        assert_eq!(n, 15);
        for s in &out[..n] {
            let (a, b) = (s.start as usize, s.end as usize);
            assert!(state.is_char_boundary(a) && state.is_char_boundary(b));
            assert!(!&state[a..b].is_empty());
        }
    }

    #[test]
    fn query_without_terms_scores_nothing() {
        let sp = SpanPointer::new("alpha beta gamma", SpanConfig::default());
        let mut out = [ZERO_SPAN; 4];
        assert_eq!(sp.score_into(&SpanQuery::new("!!! ???"), &mut out), 0);
        assert!(sp.best_span(&SpanQuery::new("!!!")).is_none());
    }

    // ---- G4: counting allocator -------------------------------------------

    // The T0.2 machinery (TLS-gated counting allocator) lives in
    // `crate::alloc_gate` since the T0.6 consolidation — ONE
    // `#[global_allocator]` per test binary is a rustc rule, and the other
    // modules' gates drive the same allocator through `crate::alloc_gate`.

    #[test]
    fn hot_path_is_zero_alloc() {
        let state =
            "Acme Corporation is based in Seattle. The vendor is Globex and ships worldwide.";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery {
            text: "globex vendor",
            aliases: &["corp", "Globex"],
        };
        let mut out = [ZERO_SPAN; 8];

        // Warmup outside the measured window.
        let _ = sp.score_into(&q, &mut out);
        let _ = sp.best_span(&q);

        {
            let _window = MeasuringWindow::enter();
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            let n = sp.score_into(&q, &mut out);
            assert_eq!(
                ALLOC_COUNT.load(Ordering::Relaxed),
                0,
                "score_into allocated on the hot path"
            );
            assert!(n > 0);

            ALLOC_COUNT.store(0, Ordering::Relaxed);
            let best = sp.best_span(&q);
            assert_eq!(
                ALLOC_COUNT.load(Ordering::Relaxed),
                0,
                "best_span allocated on the hot path"
            );
            assert!(best.is_some());
        }
    }

    // ---- Bench-950 round 3: the boundary resolver -------------------------

    #[test]
    fn snap_picks_the_longest_fully_inside_surface() {
        let text = "the best movie ever made";
        let window = (4u32, 19u32);
        assert_eq!(&text[4..19], "best movie ever");
        let got = snap_longest(text, window, ["best", "movie", "best movie"]);
        assert_eq!(got, Some((4, 14)));
        assert_eq!(&text[4..14], "best movie");
    }

    #[test]
    fn snap_ties_break_by_list_order() {
        let text = "alpha beta gamma";
        let window = (0u32, 16u32);
        // Longest-match first: a longer surface wins regardless of order.
        assert_eq!(
            snap_longest(text, window, ["beta", "alpha"]),
            Some((0, 5)),
            "alpha (5 B) beats beta (4 B) — length dominates list order"
        );
        // Genuinely equal-length surfaces (alpha/gamma, both 5 B): the
        // earlier-listed one wins (strictly-greater replacement — the
        // gazetteer law).
        assert_eq!(
            snap_longest(text, window, ["gamma", "alpha"]),
            Some((11, 16))
        );
        assert_eq!(snap_longest(text, window, ["alpha", "gamma"]), Some((0, 5)));
        // The same surface twice (canonical duplicated in the alias list):
        // the leftmost occurrence wins.
        assert_eq!(
            snap_longest(text, window, ["gamma", "gamma"]),
            Some((11, 16))
        );
    }

    #[test]
    fn snap_requires_the_occurrence_inside_the_window() {
        let text = "watch the marx brothers film";
        assert_eq!(&text[6..14], "the marx");
        assert_eq!(
            snap_longest(text, (6, 14), ["marx brothers"]),
            None,
            "an occurrence crossing the window edge must not qualify"
        );
        assert_eq!(snap_longest(text, (6, 14), ["marx"]), Some((10, 14)));
    }

    #[test]
    fn snap_rejects_empty_out_of_range_and_unaligned_windows() {
        let text = "alpha beta";
        assert_eq!(snap_longest(text, (5, 5), ["alpha"]), None);
        assert_eq!(snap_longest(text, (6, 4), ["alpha"]), None);
        assert_eq!(snap_longest(text, (0, 99), ["alpha"]), None);
    }

    #[test]
    fn snap_multibyte_window_slices_exactly() {
        let text = "Le café et 中文字 ok";
        // é is 2 bytes: "café" = bytes 3..8.
        assert_eq!(&text[3..8], "café");
        assert_eq!(snap_longest(text, (3, 8), ["café"]), Some((3, 8)));
        // 中文字 = bytes 12..21 (3 chars × 3 B); 文字 = 15..18.
        assert_eq!(&text[12..21], "中文字");
        assert_eq!(
            snap_longest(text, (12, 21), ["文字", "中文字"]),
            Some((12, 21)),
            "the longer surface wins even when a shorter one is listed first"
        );
    }

    #[test]
    fn snap_none_inside_is_none() {
        let text = "the marx brothers film";
        assert_eq!(snap_longest(text, (0, 6), ["marx"]), None);
    }

    #[test]
    fn best_snapped_span_matches_the_snap_contract_end_to_end() {
        let state = "i love science fiction movies a lot and own the fiction book too";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery {
            text: "fiction",
            aliases: &["science fiction"],
        };
        let best = sp.best_span(&q).expect("verbatim mention scores");
        let snapped = sp
            .best_snapped_span(state, &q)
            .expect("the composition keeps an answer whenever the pointer has one");
        let expected = snap_longest(
            state,
            (best.start, best.end),
            std::iter::once("fiction").chain(["science fiction"]),
        )
        .unwrap_or((best.start, best.end));
        assert_eq!((snapped.start, snapped.end), expected);
        assert_eq!(snapped.score, best.score, "the snap never re-ranks");
        // A boundary refinement: the snapped span is never wider than the
        // window it refined (snap returns a fully-inside occurrence or the
        // window itself).
        assert!(snapped.start >= best.start && snapped.end <= best.end);
    }

    #[test]
    fn best_snapped_span_propagates_no_answer() {
        let sp = SpanPointer::new("alpha beta gamma", SpanConfig::default());
        assert!(
            sp.best_snapped_span("alpha beta gamma", &SpanQuery::new("!!!"))
                .is_none()
        );
    }

    #[test]
    fn snap_hot_path_is_zero_alloc() {
        let state =
            "Acme Corporation is based in Seattle. The vendor is Globex and ships worldwide.";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery {
            text: "globex vendor",
            aliases: &["corp", "Globex"],
        };

        // Warmup outside the measured window.
        let _ = sp.best_snapped_span(state, &q);

        {
            let _window = MeasuringWindow::enter();
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            let snapped = sp.best_snapped_span(state, &q);
            assert_eq!(
                ALLOC_COUNT.load(Ordering::Relaxed),
                0,
                "best_snapped_span allocated on the hot path"
            );
            assert!(snapped.is_some());

            ALLOC_COUNT.store(0, Ordering::Relaxed);
            let hit = snap_longest(state, (0, state.len() as u32), ["Globex", "Corporation"]);
            assert_eq!(
                ALLOC_COUNT.load(Ordering::Relaxed),
                0,
                "snap_longest allocated on the hot path"
            );
            assert!(hit.is_some());
        }
    }

    // ---- Bench-950 round 4: the control composite -------------------------

    #[test]
    fn longest_match_prefers_longest_then_list_order() {
        let text = "stars humphrey bogart and lauren bacall";
        // Longest wins regardless of order.
        assert_eq!(
            longest_match(text, ["bacall", "lauren bacall"]),
            Some((26, 39)),
        );
        assert_eq!(&text[26..39], "lauren bacall");
        // Genuine tie: the earlier-listed surface's leftmost occurrence.
        assert_eq!(longest_match(text, ["bogart", "bacall"]), Some((15, 21)));
        // No occurrence anywhere: the abstention the fallback covers.
        assert_eq!(longest_match(text, ["jack nicholson"]), None);
    }

    #[test]
    fn floor_first_span_prefers_the_floor() {
        let state = "stars humphrey bogart and lauren bacall on screen";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery {
            text: "bacall",
            aliases: &["lauren bacall"],
        };
        let got = sp.floor_first_span(state, &q).expect("the floor hits");
        assert_eq!(got.source, SpanAnswerSource::Floor);
        assert_eq!((got.start, got.end), (26, 39));
    }

    #[test]
    fn floor_first_span_falls_back_on_floor_abstain() {
        // "g l o b e x" spaced: no surface occurs literally anywhere (the
        // floor abstains), but the pointer's tokens still score a window.
        let state = "the vendor is g l o b e x here";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery {
            text: "globex",
            aliases: &[],
        };
        match sp.floor_first_span(state, &q) {
            Some(g) => {
                assert_eq!(g.source, SpanAnswerSource::PointerFallback);
                assert!(g.end > g.start);
                let _ = state[g.start as usize..g.end as usize].len();
            }
            None => { /* both abstained on this text — also legal */ }
        }
    }

    #[test]
    fn floor_first_span_none_when_both_abstain() {
        let sp = SpanPointer::new("alpha beta gamma", SpanConfig::default());
        let q = SpanQuery {
            text: "!!!",
            aliases: &["???"],
        };
        assert!(sp.floor_first_span("alpha beta gamma", &q).is_none());
    }

    #[test]
    fn floor_first_hot_path_is_zero_alloc() {
        let state =
            "Acme Corporation is based in Seattle. The vendor is Globex and ships worldwide.";
        let sp = SpanPointer::new(state, SpanConfig::default());
        let q = SpanQuery {
            text: "globex vendor",
            aliases: &["corp", "Globex"],
        };
        // Warmup outside the measured window.
        let _ = sp.floor_first_span(state, &q);
        {
            let _window = MeasuringWindow::enter();
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            let got = sp.floor_first_span(state, &q);
            assert_eq!(
                ALLOC_COUNT.load(Ordering::Relaxed),
                0,
                "floor_first_span allocated on the hot path"
            );
            assert!(got.is_some());
        }
    }
}
