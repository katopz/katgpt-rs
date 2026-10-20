//! `parsers` — deterministic typed parsers (Plan 628 T0.3).
//!
//! Grammar-first extraction over the wire's field kinds: number,
//! date/time/datetime, duration, and list-of-typed. Each parser scans the
//! state byte-exactly and appends RANKED CANDIDATES best-first into a
//! caller-owned buffer — the engine picks the head candidate or abstains.
//! A candidate that cannot satisfy the field's constraints is NEVER
//! produced (constraints by construction: the engine abstains rather than
//! clamps).
//!
//! Grammar design translated from published grammar SHAPES, not code:
//! - Meta **Duckling** (Apache-2.0) — the dimension structure (number /
//!   time / duration with compositional rules), relative-date anchoring to
//!   a reference time.
//! - Microsoft **Recognizers-Text** (MIT) — the English number/currency/
//!   percent/unit surface forms, comma-grouping rules, and the M/D/Y
//!   ambiguity policy.
//!
//! ## Supported subset (honest scope)
//!
//! This is a TRANSLATED SUBSET, not full coverage of either library.
//! Per-dimension scope is listed in each module doc. Deliberately NOT
//! supported (the F0 honest-limit list): number words ("five"), written
//! ordinals, timezone names/offsets, holiday names, locale date orders
//! beyond the documented US/D/M/Y rule, set/until ranges, and any
//! non-English surface form.
//!
//! ## Determinism & allocation laws
//!
//! - No clock, no RNG, no HashMap iteration order: `reference_date` is the
//!   ONLY time source. A relative expression with no anchor produces NO
//!   candidate — the parser never reads the wall clock.
//! - Scanning is zero-alloc: scanners emit `(start, end)` spans plus
//!   numeric/civil components (`RawValue` — no `String`s). `Text`/`Date`/
//!   `Time`/`DateTime` strings and duration-unit spellings are materialized
//!   at CANDIDATE-CONSTRUCTION time (the cold edge, [`materialize_raw`]).
//!   `parse_list` additionally uses one bounded scratch Vec of item ranges
//!   plus one reused scratch Vec of element candidates — both grow only
//!   when a candidate is actually constructed. The scan loops themselves
//!   allocate nothing.
//! - Ranks are ordinals (sigmoid-shaped preference), never probabilities.
//!   `rank 0` is best; lower wins; ties break by earlier span start, then
//!   shorter span. Longer/more-specific candidates carry LOWER ranks than
//!   the shorter candidates they overlap, so best-first order prefers them.
//!
//! ## Span law
//!
//! Every span is an exact byte-offset slice of the input:
//! `&state[span.start..span.end]` is the matched text, byte for byte
//! (assert-friendly by construction). Spans land on char boundaries.

pub(crate) mod words;

pub mod datetime;
pub mod duration;
pub mod list;
pub mod number;

use katgpt_core::extraction_wire::{DurationValue, FieldKind, FieldSpec, FieldValue, Span};

pub use datetime::parse_datetime_kind;
pub use duration::parse_duration;
pub use list::parse_list;
pub use number::parse_number;

/// One located, typed candidate. `rank 0` is best (lower wins); ties break
/// by earlier [`Span::start`], then shorter span (smaller
/// [`Span::end`]). Candidates arrive best-first in the caller's buffer.
/// The best candidate ALWAYS carries rank 0 (the rank-0 law, issue 953);
/// higher ranks only mark fallbacks underneath a better overlap.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedValue {
    pub span: Span,
    pub value: FieldValue,
    pub rank: u8,
}

/// Dispatch on the field kind and APPEND candidates best-first. The caller
/// clears `out`; the parser sorts only the tail it appends (existing
/// entries are untouched).
///
/// [`FieldKind::Text`] is the span scorer's dimension (not a typed grammar)
/// and [`FieldKind::Estimate`]/[`FieldKind::Forecast`] are computed heads —
/// all three append nothing here.
pub fn parse_field(
    state: &str,
    field: &FieldSpec,
    reference_date: Option<&str>,
    out: &mut Vec<ParsedValue>,
) {
    if state.len() > u32::MAX as usize {
        return;
    }
    match field.kind {
        FieldKind::Number => parse_number(state, &field.constraints, out),
        FieldKind::Date => parse_datetime_kind(state, FieldKind::Date, reference_date, out),
        FieldKind::Time => parse_datetime_kind(state, FieldKind::Time, reference_date, out),
        FieldKind::DateTime => parse_datetime_kind(state, FieldKind::DateTime, reference_date, out),
        FieldKind::Duration => parse_duration(state, &field.constraints, out),
        // The wire defaults a List field's element kind to Text (the same
        // `unwrap_or(Text)` the wire's own list check uses).
        FieldKind::List => {
            parse_list(
                state,
                field.constraints.element_kind.unwrap_or(FieldKind::Text),
                &field.constraints,
                reference_date,
                out,
            );
        }
        FieldKind::Text | FieldKind::Estimate | FieldKind::Forecast => {}
    }
}

/// A scanned value before materialization — no `String`s. Strings are built
/// only when a candidate is CONSTRUCTED (the documented cold edge).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RawValue {
    Number(f64),
    /// Civil date components; materialized to `YYYY-MM-DD`.
    Date {
        y: i64,
        m: u32,
        d: u32,
    },
    /// Wall-clock components; materialized to `HH:MM[:SS]`.
    Time {
        h: u32,
        mi: u32,
        s: Option<u32>,
    },
    /// Date + time components; materialized to `YYYY-MM-DDTHH:MM[:SS]`.
    DateTime {
        y: i64,
        mo: u32,
        d: u32,
        h: u32,
        mi: u32,
        s: Option<u32>,
    },
    /// Duration with a canonical `&'static str` unit (`DURATION_UNITS`).
    Duration {
        unit: &'static str,
        value: f64,
    },
    /// Duration converted to the caller's constraint unit (owned spelling —
    /// the wire requires the exact constraint string).
    DurationOwned {
        unit: String,
        value: f64,
    },
    /// A materialized list (elements are built during candidate
    /// construction; the scan/split itself is alloc-free).
    List(Vec<FieldValue>),
}

/// A raw candidate flowing through the internal out-vec protocol.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RawCandidate {
    pub start: u32,
    pub end: u32,
    pub rank: u8,
    /// Composition depth — the qualified-form tie-break (issue 953 ask 3):
    /// among equal-rank candidates the more-composed match wins rank 0
    /// (`$ 254 million` — currency + multiplier — over an earlier `27 %`).
    /// Sort key AFTER rank, BEFORE position.
    pub depth: u8,
    pub raw: RawValue,
}

/// Sort the scanned candidates best-first (rank, then start, then end),
/// drop exact duplicates, materialize, and APPEND to the caller's buffer.
///
/// The rank-0 law (issue 953): the BEST candidate always carries rank 0 —
/// a lone plain candidate is promoted (a bare `192` must answer a rank-0
/// read, exactly like a `$12` does), while overlapping qualified matches
/// keep their precedence. The promotion never reorders (the promoted
/// candidate is already first).
pub(crate) fn finish(mut raws: Vec<RawCandidate>, state: &str, out: &mut Vec<ParsedValue>) {
    raws.sort_by(|a, b| {
        a.rank
            .cmp(&b.rank)
            .then(b.depth.cmp(&a.depth))
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
    });
    raws.dedup_by(|a, b| {
        a.start == b.start && a.end == b.end && a.rank == b.rank && a.raw == b.raw
    });
    // After dedup: the best candidate carries rank 0 (the promotion must
    // not split a rank-equal duplicate pair the dedup just merged).
    if let Some(first) = raws.first_mut() {
        first.rank = 0;
    }
    for r in raws {
        debug_assert!(
            state.is_char_boundary(r.start as usize) && state.is_char_boundary(r.end as usize)
        );
        if let Some(value) = materialize_raw(r.raw) {
            out.push(ParsedValue {
                span: Span {
                    start: r.start,
                    end: r.end,
                },
                value,
                rank: r.rank,
            });
        }
    }
}

/// The cold edge: turn a scanned value into its wire representation.
pub(crate) fn materialize_raw(raw: RawValue) -> Option<FieldValue> {
    match raw {
        RawValue::Number(n) => Some(FieldValue::Number(n)),
        RawValue::Date { y, m, d } => Some(FieldValue::Date(format!("{y:04}-{m:02}-{d:02}"))),
        RawValue::Time { h, mi, s } => match s {
            Some(sec) => Some(FieldValue::Time(format!("{h:02}:{mi:02}:{sec:02}"))),
            None => Some(FieldValue::Time(format!("{h:02}:{mi:02}"))),
        },
        RawValue::DateTime { y, mo, d, h, mi, s } => {
            let clock = match s {
                Some(sec) => format!("{h:02}:{mi:02}:{sec:02}"),
                None => format!("{h:02}:{mi:02}"),
            };
            Some(FieldValue::DateTime(format!(
                "{y:04}-{mo:02}-{d:02}T{clock}"
            )))
        }
        RawValue::Duration { unit, value } => Some(FieldValue::Duration(DurationValue {
            unit: unit.to_string(),
            value,
        })),
        RawValue::DurationOwned { unit, value } => {
            Some(FieldValue::Duration(DurationValue { unit, value }))
        }
        RawValue::List(items) => Some(FieldValue::List(items)),
    }
}

// ---------------------------------------------------------------------------
// Shared byte-scan helpers (the tokenizer half of this module). All operate
// on byte slices with explicit window bounds; no allocation.
// ---------------------------------------------------------------------------

/// A "word" byte: ASCII alphanumeric, or any non-ASCII byte (treated
/// conservatively as word content so tokens never split inside multibyte
/// chars).
#[inline]
pub(crate) fn is_word_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c >= 0x80
}

#[inline]
pub(crate) fn is_ws_byte(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n')
}

/// Advance to the next char boundary at or after `i` (skips UTF-8
/// continuation bytes).
#[inline]
pub(crate) fn next_boundary(b: &[u8], mut i: usize, we: usize) -> usize {
    while i < we && b[i] & 0xC0 == 0x80 {
        i += 1;
    }
    i
}

/// Skip ASCII whitespace forward.
#[inline]
pub(crate) fn skip_ascii_ws(b: &[u8], mut i: usize, we: usize) -> usize {
    while i < we && is_ws_byte(b[i]) {
        i += 1;
    }
    i
}

/// Index of the previous char's first byte within the window, or `None` at
/// the window edge (which counts as a token boundary).
pub(crate) fn prev_char_index(b: &[u8], ws: usize, i: usize) -> Option<usize> {
    if i == 0 || i <= ws {
        return None;
    }
    let mut j = i - 1;
    while j > ws && b[j] & 0xC0 == 0x80 {
        j -= 1;
    }
    Some(j)
}

/// Case-insensitive ASCII word match with a hard word boundary after.
/// Returns the end offset on success.
pub(crate) fn ascii_word_at(b: &[u8], we: usize, i: usize, word: &str) -> Option<usize> {
    let w = word.as_bytes();
    if i >= we || i + w.len() > we {
        return None;
    }
    if !b[i..i + w.len()].eq_ignore_ascii_case(w) {
        return None;
    }
    let e = i + w.len();
    if e < we && is_word_byte(b[e]) {
        return None;
    }
    Some(e)
}

#[cfg(test)]
use katgpt_core::extraction_wire::FieldConstraints;

#[cfg(test)]
pub(crate) fn span_text<'a>(state: &'a str, pv: &ParsedValue) -> &'a str {
    &state[pv.span.start as usize..pv.span.end as usize]
}

#[cfg(test)]
pub(crate) fn spec(kind: FieldKind) -> FieldSpec {
    FieldSpec {
        id: "f".to_string(),
        kind,
        description: None,
        constraints: FieldConstraints::default(),
        instructions: None,
    }
}

#[cfg(test)]
pub(crate) fn cons(
    integer: bool,
    decimals: Option<u8>,
    unit: Option<&str>,
    min: Option<f64>,
    max: Option<f64>,
) -> FieldConstraints {
    FieldConstraints {
        integer,
        decimals,
        unit: unit.map(str::to_string),
        min,
        max,
        ..FieldConstraints::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds() -> [FieldKind; 9] {
        [
            FieldKind::Text,
            FieldKind::Number,
            FieldKind::Date,
            FieldKind::Time,
            FieldKind::DateTime,
            FieldKind::Duration,
            FieldKind::List,
            FieldKind::Estimate,
            FieldKind::Forecast,
        ]
    }

    #[test]
    fn parse_field_dispatches_number() {
        let field = spec(FieldKind::Number);
        let mut out = Vec::new();
        parse_field("total 42", &field, None, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].value, FieldValue::Number(42.0));
        assert_eq!(span_text("total 42", &out[0]), "42");
    }

    #[test]
    fn parse_field_text_and_computed_kinds_yield_nothing() {
        for kind in [FieldKind::Text, FieldKind::Estimate, FieldKind::Forecast] {
            let field = spec(kind);
            let mut out = Vec::new();
            parse_field("42 dollars", &field, None, &mut out);
            assert!(out.is_empty(), "{kind:?} must not produce typed candidates");
        }
    }

    #[test]
    fn parse_field_covers_every_kind_without_panic() {
        for kind in kinds() {
            let field = spec(kind);
            let mut out = Vec::new();
            parse_field(
                "1,234 and tomorrow at 5pm",
                &field,
                Some("2026-10-20"),
                &mut out,
            );
        }
    }

    #[test]
    fn candidates_are_ordered_best_first_rank_then_start_then_end() {
        let field = spec(FieldKind::Number);
        let mut out = Vec::new();
        parse_field("$1,234.50", &field, None, &mut out);
        assert!(out.len() >= 2);
        for pair in out.windows(2) {
            let a = (pair[0].rank, pair[0].span.start, pair[0].span.end);
            let b = (pair[1].rank, pair[1].span.start, pair[1].span.end);
            assert!(
                a <= b,
                "candidates must be best-first sorted: {a:?} then {b:?}"
            );
        }
        // The qualified full match leads; the plain shorter overlaps trail.
        assert_eq!(span_text("$1,234.50", &out[0]), "$1,234.50");
        assert_eq!(out[0].rank, 0);
        assert!(out[1].rank > out[0].rank);
    }

    #[test]
    fn parse_appends_and_sorts_only_the_appended_tail() {
        let field = spec(FieldKind::Number);
        let sentinel = ParsedValue {
            span: Span { start: 0, end: 0 },
            value: FieldValue::Number(0.0),
            rank: 9,
        };
        let mut out = vec![sentinel.clone()];
        parse_field("5", &field, None, &mut out);
        assert_eq!(out[0], sentinel, "existing entries are untouched");
        assert!(out.len() >= 2);
    }

    #[test]
    fn spans_are_exact_across_multibyte_prefixes() {
        // é = 2 bytes; the number starts at byte 3.
        let state = "é 5 km";
        let mut out = Vec::new();
        parse_number(state, &cons(false, None, Some("km"), None, None), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].span.start, out[0].span.end), (3, 7));
        assert_eq!(span_text(state, &out[0]), "5 km");

        // 中 = 3 bytes; the token starts at byte 4 (word-adjacent digits
        // are blocked, same law as "abc123").
        let state = "中 12kg";
        let mut out = Vec::new();
        parse_number(state, &cons(false, None, Some("kg"), None, None), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].span.start, out[0].span.end), (4, 8));
        assert_eq!(span_text(state, &out[0]), "12kg");

        // A multibyte char directly before a date (space-separated — a
        // word-adjacent date is blocked by design, see number.rs).
        let state = "é 2026-10-20";
        let mut out = Vec::new();
        parse_datetime_kind(state, FieldKind::Date, None, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(span_text(state, &out[0]), "2026-10-20");
    }

    #[test]
    fn parse_twice_is_deterministic() {
        let field = spec(FieldKind::Number);
        let mut a = Vec::new();
        let mut b = Vec::new();
        parse_field("total $1,234.50 and 5k units", &field, None, &mut a);
        parse_field("total $1,234.50 and 5k units", &field, None, &mut b);
        assert_eq!(a, b);

        let field = spec(FieldKind::DateTime);
        let mut a = Vec::new();
        let mut b = Vec::new();
        parse_field("tomorrow at 5pm", &field, Some("2026-10-20"), &mut a);
        parse_field("tomorrow at 5pm", &field, Some("2026-10-20"), &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn list_kind_dispatches_with_element_kind_default_text() {
        let field = spec(FieldKind::List);
        let mut out = Vec::new();
        parse_field("bread, butter", &field, None, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].value,
            FieldValue::List(vec![
                FieldValue::Text("bread".to_string()),
                FieldValue::Text("butter".to_string())
            ])
        );

        // No element_kind -> Text elements (the wire's own default).
        let mut out = Vec::new();
        parse_field("1, 2", &spec(FieldKind::List), None, &mut out);
        assert_eq!(
            out[0].value,
            FieldValue::List(vec![
                FieldValue::Text("1".to_string()),
                FieldValue::Text("2".to_string())
            ])
        );
    }
}
