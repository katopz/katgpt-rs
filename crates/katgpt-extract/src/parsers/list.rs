//! `list` — the List dimension (Plan 628 T0.3): split, then parse each
//! element with its own dimension parser.
//!
//! Splitting is SCAN-AND-SPLIT, never a naive split: the splitter
//! understands the element grammar's own tokens, so `1,234` survives as
//! ONE element. Design translated from Meta **Duckling** (Apache-2.0)
//! list/sequencing rules and Microsoft **Recognizers-Text** (MIT) number
//! tokenization.
//!
//! ## Supported subset (documented rules)
//!
//! - Separators: top-level commas plus the word `and` — with two
//!   per-element-kind exceptions. `Duration` elements split on COMMAS ONLY
//!   (`and` belongs to the chain grammar: `1 hour and 20 minutes` is one
//!   element). `Text` elements split on COMMAS ONLY (`and` is prose).
//!   Consecutive separators collapse (Oxford commas, `1, , 2`), so empty
//!   items never arise; a trailing separator is dropped.
//! - Number-token protection: for `Number`/`Duration` elements the walk
//!   consumes comma-grouped numeric cores, so `1,234, 5` is two elements,
//!   not three. `Date`/`Time`/`DateTime` tokens contain no commas in this
//!   subset (month-day-year with its internal comma is ambiguous inside a
//!   list and is therefore NOT split-protected: such a list yields no
//!   candidate — use `20 October 2026` or ISO inside lists).
//! - Elements: each item is parsed by the element kind's parser over the
//!   item's own byte window, and a candidate must span the item EXACTLY
//!   (best rank wins) — partial matches never become elements.
//! - Input order is preserved; the List candidate's span covers the list
//!   text (first item start to last item end). Depth is 1: elements are
//!   scalars (`List`/`Estimate`/`Forecast` element kinds produce no
//!   candidate — the wire rejects them too).
//! - ALL items must parse: one non-conforming item aborts the whole
//!   candidate set (never a partial list). Fewer than two items is not a
//!   list.
//! - `Text` elements are the trimmed item text itself (the wire's default
//!   element kind). Element constraints (`integer`/`decimals`/`unit`/
//!   `min`/`max`) are enforced per item by the element parser.

use super::{
    ParsedValue, RawCandidate, RawValue, ascii_word_at, finish, is_ws_byte, materialize_raw,
    next_boundary, skip_ascii_ws,
};
use crate::parsers::datetime::{Ref, parse_reference, scan_datetimes};
use crate::parsers::duration::scan_durations;
use crate::parsers::number::{core_at, scan_numbers};
use katgpt_core::extraction_wire::{FieldConstraints, FieldKind, FieldValue};

/// Rank 0: the full-split list is the only candidate shape this dimension
/// emits.
const RANK_LIST: u8 = 0;

/// Parse every List candidate in `state` and append them best-first.
pub fn parse_list(
    state: &str,
    element_kind: FieldKind,
    element_constraints: &FieldConstraints,
    reference_date: Option<&str>,
    out: &mut Vec<ParsedValue>,
) {
    if state.len() > u32::MAX as usize {
        return;
    }
    let reference = parse_reference(reference_date);
    let mut scratch = Vec::new();
    scan_lists(
        state,
        0..state.len(),
        element_kind,
        element_constraints,
        reference,
        &mut scratch,
    );
    finish(scratch, state, out);
}

/// The list scanner: split walk (zero-alloc over Strings), then element
/// resolution (candidate construction — the documented cold edge, including
/// one bounded scratch Vec reused across items).
pub(crate) fn scan_lists(
    state: &str,
    win: core::ops::Range<usize>,
    element_kind: FieldKind,
    econ: &FieldConstraints,
    reference: Option<Ref>,
    sink: &mut Vec<RawCandidate>,
) {
    if !matches!(
        element_kind,
        FieldKind::Text
            | FieldKind::Number
            | FieldKind::Date
            | FieldKind::Time
            | FieldKind::DateTime
            | FieldKind::Duration
    ) {
        return; // List/Estimate/Forecast elements: schema errors, not answers
    }
    let b = state.as_bytes();
    let (ws, we) = (win.start, win.end);
    let and_separates = matches!(
        element_kind,
        FieldKind::Number | FieldKind::Date | FieldKind::Time | FieldKind::DateTime
    );
    let protect_number_commas = matches!(element_kind, FieldKind::Number | FieldKind::Duration);

    let mut items: Vec<(u32, u32)> = Vec::new();
    let mut item_start: usize = usize::MAX;
    let mut i = ws;
    while i < we {
        let j = skip_ascii_ws(b, i, we);
        if j < we && b[j] == b',' {
            close_item(b, &mut items, &mut item_start, j);
            i = collapse_separators(b, we, j + 1, and_separates);
            continue;
        }
        if and_separates
            && j < we
            && (j == 0 || is_ws_byte(b[j - 1]))
            && let Some(ae) = ascii_word_at(b, we, j, "and")
        {
            close_item(b, &mut items, &mut item_start, j);
            i = ae;
            continue;
        }
        // Content.
        if item_start == usize::MAX {
            item_start = j;
        }
        if protect_number_commas && b[j].is_ascii_digit() {
            // Consume the grouped numeric core so "1,234" never splits.
            if let Some(c) = core_at(b, we, j) {
                i = c.end;
                continue;
            }
        }
        i = next_boundary(b, j + 1, we);
    }
    close_item(b, &mut items, &mut item_start, we);
    if items.len() < 2 {
        return;
    }

    // Element resolution: every item must parse exactly, or the whole set
    // is aborted (no partial lists).
    let mut values: Vec<FieldValue> = Vec::with_capacity(items.len());
    let mut scratch: Vec<RawCandidate> = Vec::new();
    for &(s, e) in &items {
        let (ts, te) = (s as usize, e as usize);
        if ts >= te {
            return;
        }
        let value = match element_kind {
            FieldKind::Text => Some(FieldValue::Text(state[ts..te].to_string())),
            _ => element_value(state, ts, te, element_kind, econ, reference, &mut scratch),
        };
        match value {
            Some(v) => values.push(v),
            None => return,
        }
    }
    let last = items[items.len() - 1];
    sink.push(RawCandidate {
        start: items[0].0,
        end: last.1,
        rank: RANK_LIST,
        depth: 0,
        raw: RawValue::List(values),
    });
}

/// Close the open item at `sep`: trim trailing whitespace off the item and
/// record its range. The split walk guarantees the range is already
/// left-trimmed.
fn close_item(b: &[u8], items: &mut Vec<(u32, u32)>, item_start: &mut usize, sep: usize) {
    if *item_start == usize::MAX {
        return;
    }
    let mut e = sep;
    while e > *item_start && is_ws_byte(b[e - 1]) {
        e -= 1;
    }
    items.push((*item_start as u32, e as u32));
    *item_start = usize::MAX;
}

/// Collapse a run of separators (whitespace, commas, and — where the
/// element grammar allows — the word "and") into one break.
fn collapse_separators(b: &[u8], we: usize, mut i: usize, and_separates: bool) -> usize {
    loop {
        let k = skip_ascii_ws(b, i, we);
        if k < we && b[k] == b',' {
            i = k + 1;
            continue;
        }
        if and_separates
            && k < we
            && (k == 0 || is_ws_byte(b[k - 1]))
            && ascii_word_at(b, we, k, "and").is_some()
        {
            i = k + 3; // "and".len()
            continue;
        }
        return k;
    }
}

/// Parse one element over its byte window; only a candidate spanning the
/// item EXACTLY qualifies (lowest rank wins).
fn element_value(
    state: &str,
    ts: usize,
    te: usize,
    kind: FieldKind,
    econ: &FieldConstraints,
    reference: Option<Ref>,
    scratch: &mut Vec<RawCandidate>,
) -> Option<FieldValue> {
    scratch.clear();
    match kind {
        FieldKind::Number => scan_numbers(state, ts..te, econ, scratch),
        FieldKind::Duration => scan_durations(state, ts..te, econ, scratch),
        FieldKind::Date | FieldKind::Time | FieldKind::DateTime => {
            scan_datetimes(state, ts..te, kind, reference, scratch);
        }
        _ => return None,
    }
    let mut best: Option<&RawCandidate> = None;
    for c in scratch.iter() {
        if c.start == ts as u32 && c.end == te as u32 {
            best = match best {
                Some(b) if b.rank <= c.rank => Some(b),
                _ => Some(c),
            };
        }
    }
    let raw = best?.raw.clone();
    materialize_raw(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::{cons, span_text};
    use katgpt_core::extraction_wire::{DurationValue, Span};

    const REF: &str = "2026-10-20";

    fn list_of(state: &str, kind: FieldKind, econ: &FieldConstraints) -> Vec<FieldValue> {
        let mut out = Vec::new();
        parse_list(state, kind, econ, Some(REF), &mut out);
        assert_eq!(
            out.len(),
            1,
            "expected one list candidate in {state:?}, got {:?}",
            out.len()
        );
        match &out[0].value {
            FieldValue::List(items) => items.clone(),
            ref other => panic!("not a list: {other:?}"),
        }
    }

    fn no_list(state: &str, kind: FieldKind, econ: &FieldConstraints) {
        let mut out = Vec::new();
        parse_list(state, kind, econ, Some(REF), &mut out);
        assert!(
            out.is_empty(),
            "expected no list candidate in {state:?}, got {:?}",
            out.first()
        );
    }

    fn num(v: f64) -> FieldValue {
        FieldValue::Number(v)
    }

    fn txt(s: &str) -> FieldValue {
        FieldValue::Text(s.to_string())
    }

    #[test]
    fn number_list_order_preserved() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            list_of("3, 1, 2", FieldKind::Number, &c),
            vec![num(3.0), num(1.0), num(2.0)]
        );
        assert_eq!(
            list_of("1, 2 and 3", FieldKind::Number, &c),
            vec![num(1.0), num(2.0), num(3.0)]
        );
        assert_eq!(
            list_of("1 and 2 and 3", FieldKind::Number, &c),
            vec![num(1.0), num(2.0), num(3.0)]
        );
    }

    #[test]
    fn comma_groups_are_not_separators() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            list_of("1,234, 5", FieldKind::Number, &c),
            vec![num(1234.0), num(5.0)]
        );
        assert_eq!(
            list_of("1,234,567, 2", FieldKind::Number, &c),
            vec![num(1_234_567.0), num(2.0)]
        );
        // A grouped number alone is one element -> not a list.
        no_list("1,234", FieldKind::Number, &c);
    }

    #[test]
    fn element_kind_mismatch_aborts_the_set() {
        let c = cons(false, None, None, None, None);
        // "two" now parses as a word number (issue 953) — use a genuinely
        // unparseable element for the mismatch abort.
        no_list("1, x, 3", FieldKind::Number, &c);
        no_list(
            "1, 2.5, 3",
            FieldKind::Number,
            &cons(true, None, None, None, None),
        );
    }

    #[test]
    fn date_list() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            list_of("2026-10-20, 2026-10-21", FieldKind::Date, &c),
            vec![
                FieldValue::Date("2026-10-20".to_string()),
                FieldValue::Date("2026-10-21".to_string())
            ]
        );
        // Relative dates anchor per element.
        assert_eq!(
            list_of("tomorrow, in 3 days", FieldKind::Date, &c),
            vec![
                FieldValue::Date("2026-10-21".to_string()),
                FieldValue::Date("2026-10-23".to_string())
            ]
        );
        // A bare weekday is not a date in this subset: the item fails and
        // aborts the set.
        no_list("tomorrow, friday", FieldKind::Date, &c);
    }

    #[test]
    fn duration_list_and_is_chain_grammar() {
        let c = cons(false, None, None, None, None);
        let items = list_of("1 hour and 20 minutes, 2 days", FieldKind::Duration, &c);
        assert_eq!(items.len(), 2);
        match (&items[0], &items[1]) {
            (FieldValue::Duration(a), FieldValue::Duration(b)) => {
                assert_eq!(a.unit, "h");
                assert!((a.value - (1.0 + 20.0 / 60.0)).abs() < 1e-9);
                assert_eq!((b.unit.as_str(), b.value), ("d", 2.0));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(list_of("45s, 90 minutes", FieldKind::Duration, &c).len(), 2);
    }

    #[test]
    fn text_list_splits_commas_only() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            list_of("bread, butter", FieldKind::Text, &c),
            vec![txt("bread"), txt("butter")]
        );
        // "and" is prose for text: one item -> not a list.
        no_list("bread and butter", FieldKind::Text, &c);
        assert_eq!(
            list_of("a, b and c", FieldKind::Text, &c),
            vec![txt("a"), txt("b and c")]
        );
    }

    #[test]
    fn oxford_comma_collapses() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            list_of("1, 2, and 3", FieldKind::Number, &c),
            vec![num(1.0), num(2.0), num(3.0)]
        );
        assert_eq!(
            list_of("1, ,2", FieldKind::Number, &c),
            vec![num(1.0), num(2.0)]
        );
        assert_eq!(
            list_of("1, 2,", FieldKind::Number, &c),
            vec![num(1.0), num(2.0)]
        );
    }

    #[test]
    fn single_value_is_not_a_list() {
        let c = cons(false, None, None, None, None);
        no_list("5", FieldKind::Number, &c);
        no_list("tomorrow", FieldKind::Date, &c);
        no_list("90 minutes", FieldKind::Duration, &c);
    }

    #[test]
    fn failing_item_aborts_whole_set() {
        let c = cons(false, None, None, None, None);
        no_list("1, 2, xyz", FieldKind::Number, &c);
        // Month-day-year commas are ambiguous in lists (documented) — the
        // stray "2026" item fails and aborts the set.
        no_list("October 20, 2026, November 5", FieldKind::Date, &c);
        // The first item starts at the first content char, so leading prose
        // contaminates it: not a clean list (scan-and-split, documented).
        no_list("items 1, 2 and 3 end", FieldKind::Number, &c);
    }

    #[test]
    fn element_constraints_enforced() {
        assert_eq!(
            list_of(
                "1, 2",
                FieldKind::Number,
                &cons(true, None, None, None, None)
            ),
            vec![num(1.0), num(2.0)]
        );
        no_list(
            "1, 2.5",
            FieldKind::Number,
            &cons(true, None, None, None, None),
        );
        no_list(
            "1, 11",
            FieldKind::Number,
            &cons(false, None, None, Some(10.0), None),
        );
        assert_eq!(
            list_of(
                "5 km, 3 km",
                FieldKind::Number,
                &cons(false, None, Some("km"), None, None)
            ),
            vec![num(5.0), num(3.0)]
        );
        no_list(
            "5 km, 3",
            FieldKind::Number,
            &cons(false, None, Some("km"), None, None),
        );
        // Duration element constraints convert per item (Duration values,
        // in the constraint unit).
        assert_eq!(
            list_of(
                "1 hour, 90 minutes",
                FieldKind::Duration,
                &cons(false, None, Some("min"), None, None)
            ),
            vec![
                FieldValue::Duration(DurationValue {
                    unit: "min".to_string(),
                    value: 60.0
                }),
                FieldValue::Duration(DurationValue {
                    unit: "min".to_string(),
                    value: 90.0
                }),
            ]
        );
    }

    #[test]
    fn list_span_covers_list_text() {
        let state = "1, 2 and 3";
        let mut out = Vec::new();
        parse_list(
            state,
            FieldKind::Number,
            &cons(false, None, None, None, None),
            Some(REF),
            &mut out,
        );
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].span.start, out[0].span.end), (0, 10));
        assert_eq!(span_text(state, &out[0]), "1, 2 and 3");
        assert_eq!(out[0].rank, 0);
    }

    #[test]
    fn datetime_elements_compose() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            list_of(
                "tomorrow at 5pm, in 3 days at noon",
                FieldKind::DateTime,
                &c
            ),
            vec![
                FieldValue::DateTime("2026-10-21T17:00".to_string()),
                FieldValue::DateTime("2026-10-23T12:00".to_string())
            ]
        );
    }

    #[test]
    fn unsupported_element_kinds_produce_nothing() {
        let c = cons(false, None, None, None, None);
        no_list("1, 2", FieldKind::List, &c);
        no_list("1, 2", FieldKind::Estimate, &c);
        no_list("1, 2", FieldKind::Forecast, &c);
    }

    #[test]
    fn parse_twice_is_deterministic() {
        let c = cons(false, None, None, None, None);
        let state = "1,234, 5 km, tomorrow";
        let mut a = Vec::new();
        let mut b = Vec::new();
        parse_list(state, FieldKind::Number, &c, Some(REF), &mut a);
        parse_list(state, FieldKind::Number, &c, Some(REF), &mut b);
        assert_eq!(a, b); // "tomorrow" fails as a Number element -> both empty
        assert!(a.is_empty());
    }

    #[test]
    fn multibyte_items_keep_exact_spans() {
        // Multibyte content is safe inside Text elements (raw spans).
        let state = "café, bar"; // é = 2 bytes; total 10
        let mut out = Vec::new();
        parse_list(
            state,
            FieldKind::Text,
            &cons(false, None, None, None, None),
            None,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(span_text(state, &out[0]), "café, bar");
        let Span { start, end } = out[0].span;
        assert_eq!((start, end), (0, 10));
        assert_eq!(
            out[0].value,
            FieldValue::List(vec![txt("café"), txt("bar")])
        );
    }
}
