//! `duration` — the Duration dimension (Plan 628 T0.3).
//!
//! English duration grammar translated from Meta **Duckling** (Apache-2.0)
//! and Microsoft **Recognizers-Text** (MIT) chain shapes.
//!
//! ## Supported subset
//!
//! - Compound chains: `3 hours 20 minutes`, `1 hour and 20 minutes`
//!   (whitespace or ` and ` between terms), `2h 30m` (attached spellings
//!   mix freely with spaced ones).
//! - Units: `s|sec|secs|second|seconds`, `m|min|mins|minute|minutes`,
//!   `h|hr|hrs|hour|hours`, `d|day|days`, `w|week|weeks`,
//!   `mo|month|months`, `y|yr|yrs|year|years` — the wire's canonical
//!   [`DURATION_UNITS`] spellings — plus the scaled words `decade(s)`
//!   (×10 years) and `fortnight(s)` (×2 weeks).
//! - Single terms: `90 minutes`, `3 days`, `45s`, `2 weeks`, `1.5 hours`.
//!   A bare number (`90`) is NOT a duration; a bare unit (`hours`) is not
//!   either. Terms are unsigned (no negative durations).
//! - Word-number terms (issue 953 lever 5): `two weeks`, `twenty min`,
//!   `twenty and four hours` (24), `two and a half hours` (2.5),
//!   `two decades` (20 y).
//! - Quantifier terms: `(all|full|whole) [.-] unit` = 1 (`all day`,
//!   `full-week`), `(a|an|another) unit` = 1 (`an hour`, `another week`),
//!   `(a) couple of unit` = 2 (`a couple of days`), `a fortnight` = 2 w,
//!   `half (an) unit` = 0.5 (`half an year`) — and the chain tail
//!   `<term> and [a] half` (`an hour and a half` = 1.5 h).
//!
//! ## Unit-choice and exactness laws (documented)
//!
//! - Without a unit constraint the emitted unit is the LARGEST unit present
//!   in the input chain (`2h 30m` -> unit `h`, value 2.5). The value may be
//!   non-terminating (`3 hours 20 minutes` -> `h`, 3.333...) — it is an
//!   f64 representation, not an exactness claim.
//! - WITH `constraints.unit`, only candidates expressible EXACTLY in that
//!   unit pass: the chain is converted over exact integer milliseconds and
//!   must divide evenly (`90 minutes` in `h` abstains; `1.5 hours` in
//!   `min` is 90). The emitted unit spelling is the constraint string
//!   itself (the wire compares it verbatim).
//! - `mo` and `y` are incommensurable with second-based units (no magic
//!   month/year lengths): a chain mixing them with time units has NO
//!   candidate, and a `mo`/`y` chain converts exactly only within its own
//!   class.
//! - Sanity caps: term values < 1e9, <= 9 fraction digits, <= 16 terms.
//! - `integer`/`decimals`/`min`/`max` constraints are the Number field's
//!   and are not applied here (the wire applies none of them to
//!   durations either).

use super::{
    ParsedValue, RawCandidate, RawValue, ascii_word_at, finish, is_word_byte, next_boundary,
    prev_char_index, skip_ascii_ws, words::word_number_at,
};
use crate::parsers::number::core_at;
use katgpt_core::extraction_wire::{DURATION_UNITS, FieldConstraints};

/// Rank 0: duration chains are their dimension's most specific shape and
/// do not overlap each other.
const RANK_CHAIN: u8 = 0;

/// Milliseconds per canonical unit (index-aligned with [`DURATION_UNITS`]);
/// `mo`/`y` carry no exact millisecond factor by design.
const MS: [i64; 7] = [1_000, 60_000, 3_600_000, 86_400_000, 604_800_000, 0, 0];

/// Unit spellings longest-first: (spelling, canonical index).
const UNITS: &[(&str, u8)] = &[
    ("seconds", 0),
    ("minutes", 1),
    ("months", 5),
    ("second", 0),
    ("minute", 1),
    ("weeks", 4),
    ("hours", 2),
    ("month", 5),
    ("years", 6),
    ("days", 3),
    ("week", 4),
    ("hour", 2),
    ("year", 6),
    ("yrs", 6),
    ("mins", 1),
    ("hrs", 2),
    ("secs", 0),
    ("sec", 0),
    ("min", 1),
    ("day", 3),
    ("mo", 5),
    ("hr", 2),
    ("yr", 6),
    ("s", 0),
    ("m", 1),
    ("h", 2),
    ("d", 3),
    ("w", 4),
    ("y", 6),
];

/// Term caps.
const MAX_TERM_VALUE: f64 = 1e9;
const MAX_FRAC_DIGITS: u32 = 9;
const MAX_TERMS: usize = 16;

/// Parse every Duration candidate in `state` and append them best-first.
pub fn parse_duration(state: &str, constraints: &FieldConstraints, out: &mut Vec<ParsedValue>) {
    if state.len() > u32::MAX as usize {
        return;
    }
    let mut scratch = Vec::new();
    scan_durations(state, 0..state.len(), constraints, &mut scratch);
    finish(scratch, state, out);
}

/// The duration scanner. Zero-alloc: pushes [`RawCandidate`]s only.
pub(crate) fn scan_durations(
    state: &str,
    win: core::ops::Range<usize>,
    cons: &FieldConstraints,
    sink: &mut Vec<RawCandidate>,
) {
    let b = state.as_bytes();
    let (ws, we) = (win.start, win.end);
    let mut i = ws;
    while i < we {
        i = next_boundary(b, i, we);
        if i >= we {
            break;
        }
        if duration_start_ok(b, ws, i) {
            match chain_at(b, we, i, cons) {
                ChainOutcome::Ok(end, raw) => {
                    sink.push(RawCandidate {
                        start: i as u32,
                        end: end as u32,
                        rank: RANK_CHAIN,
                        depth: 0,
                        raw,
                    });
                    i = end;
                    continue;
                }
                // A chain that parsed but failed resolution suppresses the
                // sub-term fallbacks inside it: "1 month 3 days" yields NO
                // candidate, not its tail "3 days" (documented subset
                // behavior).
                ChainOutcome::Failed(end) => {
                    i = end;
                    continue;
                }
                ChainOutcome::None => {}
            }
            match quantifier_chain_at(b, we, i, cons) {
                ChainOutcome::Ok(end, raw) => {
                    sink.push(RawCandidate {
                        start: i as u32,
                        end: end as u32,
                        rank: RANK_CHAIN,
                        depth: 0,
                        raw,
                    });
                    i = end;
                    continue;
                }
                ChainOutcome::Failed(end) => {
                    i = end;
                    continue;
                }
                ChainOutcome::None => {}
            }
        }
        i += 1;
    }
}

/// A chain start must not continue a word or token (`1.5 h` must not yield
/// `5 h`).
fn duration_start_ok(b: &[u8], ws: usize, i: usize) -> bool {
    match prev_char_index(b, ws, i) {
        None => true,
        Some(j) => {
            let c = b[j];
            !(is_word_byte(c) || matches!(c, b'.' | b':'))
        }
    }
}

/// One chain term: numeric core + unit (attached/hyphen/spaced), or a
/// word-number term (`two weeks`, `two and a half hours`, `two decades`).
fn term_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8, i128, i64)> {
    if let Some(core) = core_at(b, we, i) {
        if core.frac_digits > MAX_FRAC_DIGITS || core.value >= MAX_TERM_VALUE {
            return None;
        }
        let (end, unit, factor) = unit_flexible_at(b, we, core.end)?;
        // Scaled words carry their factor in the numerator (`2 decades` =
        // 20/1 y).
        return Some((end, unit, core.num * i128::from(factor), core.den));
    }
    word_term_at(b, we, i)
}

/// Scaled unit words: (spelling, canonical unit, factor) — checked before
/// [`UNITS`] everywhere (boundary-guarded, so order within is free).
const SCALED_UNITS: &[(&str, u8, i64)] = &[
    ("decades", 6, 10),
    ("decade", 6, 10),
    ("fortnights", 4, 2),
    ("fortnight", 4, 2),
];

/// Unit table at ONE exact position: scaled words first, then [`UNITS`].
fn unit_tbl_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8, i64)> {
    for (w, u, f) in SCALED_UNITS {
        if let Some(e) = ascii_word_at(b, we, i, w) {
            return Some((e, *u, *f));
        }
    }
    unit_at(b, we, i).map(|(e, u)| (e, u, 1))
}

/// Unit after a value: attached (`3h`), hyphen-joined (`3-min`), or
/// whitespace-separated (`3 min`); returns the canonical unit and the
/// spelling's value factor.
fn unit_flexible_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8, i64)> {
    if let Some(hit) = unit_tbl_at(b, we, i) {
        return Some(hit);
    }
    let mut j = i;
    if j < we && b[j] == b'-' {
        j += 1;
    }
    let k = skip_ascii_ws(b, j, we);
    if k == i {
        return None; // no hyphen and no whitespace: nothing to join with
    }
    unit_tbl_at(b, we, k)
}

/// Unit spellings with their value factor (decade = 10 years, fortnight =
/// 2 weeks), attached/hyphen/spaced.
fn unit_word_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8, i64)> {
    unit_flexible_at(b, we, i)
}

/// `and half` / `and a half` at `i` — the chain tail. Returns the end.
fn and_half_at(b: &[u8], we: usize, i: usize) -> Option<usize> {
    let j = skip_ascii_ws(b, i, we);
    if j == i {
        return None;
    }
    let ae = ascii_word_at(b, we, j, "and")?;
    let mut k = skip_ascii_ws(b, ae, we);
    if k == ae {
        return None;
    }
    if let Some(art) = ascii_word_at(b, we, k, "a") {
        let k2 = skip_ascii_ws(b, art, we);
        if k2 > art {
            k = k2;
        }
    }
    ascii_word_at(b, we, k, "half")
}

/// A word-number term: `two weeks`, `twenty and four hours` (24),
/// `two and a half hours` (2.5), `two decades` (20 y).
fn word_term_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8, i128, i64)> {
    let (mut end, v) = word_number_at(b, we, i)?;
    if !(0.0..MAX_TERM_VALUE).contains(&v) || v.fract() != 0.0 {
        return None;
    }
    // Optional "and (a )?half" modifier (two and a half hours).
    let mut extra_half = false;
    if let Some(he) = and_half_at(b, we, end) {
        extra_half = true;
        end = he;
    }
    let (ue, unit, factor) = unit_word_at(b, we, end)?;
    let base = v as i128;
    let (n, d) = if extra_half {
        ((2 * base + 1) * i128::from(factor), 2)
    } else {
        (base * i128::from(factor), 1)
    };
    Some((ue, unit, n, d))
}

fn unit_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8)> {
    for (spelling, unit) in UNITS {
        if let Some(e) = ascii_word_at(b, we, i, spelling) {
            return Some((e, *unit));
        }
    }
    None
}

/// What a chain attempt at one position produced.
enum ChainOutcome {
    /// No chain starts here.
    None,
    /// A chain parsed but failed constraint/class resolution; its end is
    /// returned so the scanner can skip past it (no fallbacks inside).
    Failed(usize),
    /// A resolved chain and its end.
    Ok(usize, RawValue),
}

/// Parse a maximal chain at `i` and resolve it against the constraints.
fn chain_at(b: &[u8], we: usize, i: usize, cons: &FieldConstraints) -> ChainOutcome {
    let Some((mut end, u, n, d)) = term_at(b, we, i) else {
        return ChainOutcome::None;
    };
    // Fixed stack slots, not a Vec (the G4 scan gate caught the per-chain
    // allocation this replaced): MAX_TERMS + 1 — the cap check runs AFTER
    // the push, so the bail case holds MAX_TERMS + 1 entries. Scanning
    // stays allocation-free (the parsers module law).
    let mut terms = [(0u8, 0i128, 0i64); MAX_TERMS + 1];
    let mut terms_len = 1usize;
    terms[0] = (u, n, d);
    loop {
        let j = skip_ascii_ws(b, end, we);
        if j == end {
            break;
        }
        let mut k = j;
        if let Some(ae) = ascii_word_at(b, we, j, "and") {
            let j2 = skip_ascii_ws(b, ae, we);
            if j2 > ae {
                k = j2;
            } else {
                break; // a dangling "and" ends the chain before itself
            }
        }
        match term_at(b, we, k) {
            Some((te, u, n, d)) => {
                terms[terms_len] = (u, n, d);
                terms_len += 1;
                end = te;
                if terms_len > MAX_TERMS {
                    return ChainOutcome::Failed(end);
                }
            }
            None => {
                // The "and [a] half" tail ("an hour and a half"): a
                // half-term of the LAST term's unit.
                if let Some(he) = and_half_at(b, we, end) {
                    terms[terms_len] = (terms[terms_len - 1].0, 1, 2);
                    terms_len += 1;
                    end = he;
                    if terms_len > MAX_TERMS {
                        return ChainOutcome::Failed(end);
                    }
                    continue;
                }
                break;
            }
        }
    }
    match resolve(cons, &terms[..terms_len]) {
        Some(raw) => ChainOutcome::Ok(end, raw),
        None => ChainOutcome::Failed(end),
    }
}

/// A quantifier-led chain: `(all|full|whole) [- ]unit`, `(a|an|another)
/// unit`, `(a) couple of unit`, `a fortnight`, `half (an) unit` — value 1,
/// 2, or 1/2 (issue 953 lever 5). Single-term chains.
fn quantifier_chain_at(b: &[u8], we: usize, i: usize, cons: &FieldConstraints) -> ChainOutcome {
    let Some((mut end, unit, n, d)) = quantifier_term_at(b, we, i) else {
        return ChainOutcome::None;
    };
    let mut terms = [(unit, n, d), (unit, 0, 1)];
    let mut terms_len = 1usize;
    // Optional "and [a] half" tail in the same unit.
    if let Some(he) = and_half_at(b, we, end) {
        terms[1] = (unit, 1, 2);
        terms_len = 2;
        end = he;
    }
    match resolve(cons, &terms[..terms_len]) {
        Some(raw) => ChainOutcome::Ok(end, raw),
        None => ChainOutcome::Failed(end),
    }
}

/// The quantifier term grammar. Returns `(end, unit, num, den)`.
fn quantifier_term_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u8, i128, i64)> {
    // half (an) unit -> 1/2 (the unit factor rides the numerator)
    if let Some(he) = ascii_word_at(b, we, i, "half") {
        let j = skip_ascii_ws(b, he, we);
        let mut k = j;
        if k > he
            && let Some(ae) = ascii_word_at(b, we, k, "an")
        {
            let j2 = skip_ascii_ws(b, ae, we);
            if j2 > ae {
                k = j2;
            }
        }
        if k > he
            && let Some((ue, unit, f)) = unit_word_at(b, we, k)
        {
            return Some((ue, unit, i128::from(f), 2));
        }
    }
    // (all|full|whole) [- ]unit -> 1 (a hyphen needs no space after it)
    for w in ["all", "full", "whole"] {
        if let Some(e) = ascii_word_at(b, we, i, w) {
            let mut j = e;
            if j < we && b[j] == b'-' {
                j += 1;
            }
            let k = skip_ascii_ws(b, j, we);
            if k > e
                && let Some((ue, unit, f)) = unit_word_at(b, we, k)
            {
                return Some((ue, unit, i128::from(f), 1));
            }
        }
    }
    // another unit -> 1
    if let Some(e) = ascii_word_at(b, we, i, "another") {
        let j = skip_ascii_ws(b, e, we);
        if j > e
            && let Some((ue, unit, f)) = unit_word_at(b, we, j)
        {
            return Some((ue, unit, i128::from(f), 1));
        }
    }
    // (a) couple of unit -> 2
    {
        let mut j = i;
        if let Some(ae) = ascii_word_at(b, we, j, "a") {
            let k = skip_ascii_ws(b, ae, we);
            if k > ae {
                j = k;
            }
        }
        if let Some(ce) = ascii_word_at(b, we, j, "couple") {
            let k = skip_ascii_ws(b, ce, we);
            if k > ce
                && let Some(oe) = ascii_word_at(b, we, k, "of")
            {
                let k2 = skip_ascii_ws(b, oe, we);
                if k2 > oe
                    && let Some((ue, unit, f)) = unit_word_at(b, we, k2)
                {
                    return Some((ue, unit, 2 * i128::from(f), 1));
                }
            }
        }
    }
    // (a|an) unit -> 1 (the most generic form, last)
    if let Some(e) = ascii_word_at(b, we, i, "an").or_else(|| ascii_word_at(b, we, i, "a")) {
        let j = skip_ascii_ws(b, e, we);
        if j > e
            && let Some((ue, unit, f)) = unit_word_at(b, we, j)
        {
            return Some((ue, unit, i128::from(f), 1));
        }
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Time,
    Month,
    Year,
}

fn class_of(unit: u8) -> Class {
    match unit {
        0..=4 => Class::Time,
        5 => Class::Month,
        _ => Class::Year,
    }
}

/// Resolve a chain against the constraints (the documented unit-choice and
/// exactness laws). Returns `None` on any violation — never a clamp.
fn resolve(cons: &FieldConstraints, terms: &[(u8, i128, i64)]) -> Option<RawValue> {
    let class = class_of(terms[0].0);
    if terms.iter().any(|t| class_of(t.0) != class) {
        return None; // mo/y never mix with time units
    }
    let d_max: i64 = terms.iter().map(|t| t.2).max()?;
    match cons.unit.as_deref() {
        Some(target) => {
            let ti = u8::try_from(DURATION_UNITS.iter().position(|&u| u == target)?).ok()?;
            if class_of(ti) != class {
                return None;
            }
            match class {
                Class::Time => {
                    let factor = i128::from(MS[ti as usize]);
                    let denom = i128::from(d_max) * factor;
                    let total: i128 = terms
                        .iter()
                        .map(|&(u, n, d)| n * i128::from(d_max / d) * i128::from(MS[u as usize]))
                        .sum();
                    if denom == 0 || total % denom != 0 {
                        return None;
                    }
                    let q = total / denom;
                    if q > 1 << 53 {
                        return None;
                    }
                    Some(RawValue::DurationOwned {
                        unit: target.to_string(),
                        value: q as f64,
                    })
                }
                Class::Month | Class::Year => {
                    let total: i128 = terms
                        .iter()
                        .map(|&(_, n, d)| n * i128::from(d_max / d))
                        .sum();
                    if total % i128::from(d_max) != 0 {
                        return None;
                    }
                    Some(RawValue::DurationOwned {
                        unit: target.to_string(),
                        value: (total / i128::from(d_max)) as f64,
                    })
                }
            }
        }
        None => match class {
            Class::Time => {
                // The emitted unit is the LARGEST unit present in the chain
                // (canonical index order == magnitude order within Time).
                let largest = terms.iter().map(|t| t.0).max()?;
                let value: f64 = terms
                    .iter()
                    .map(|&(u, n, d)| {
                        (n as f64 / d as f64)
                            * (MS[u as usize] as f64 / MS[largest as usize] as f64)
                    })
                    .sum();
                Some(RawValue::Duration {
                    unit: DURATION_UNITS[largest as usize],
                    value,
                })
            }
            Class::Month => {
                let value: f64 = terms.iter().map(|&(_, n, d)| n as f64 / d as f64).sum();
                Some(RawValue::Duration {
                    unit: DURATION_UNITS[5],
                    value,
                })
            }
            Class::Year => {
                let value: f64 = terms.iter().map(|&(_, n, d)| n as f64 / d as f64).sum();
                Some(RawValue::Duration {
                    unit: DURATION_UNITS[6],
                    value,
                })
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::{cons, span_text};
    use katgpt_core::extraction_wire::FieldValue;

    fn durations(state: &str, c: &FieldConstraints) -> Vec<(String, String, f64, u8)> {
        let mut out = Vec::new();
        parse_duration(state, c, &mut out);
        out.into_iter()
            .map(|pv| {
                (
                    span_text(state, &pv).to_string(),
                    match &pv.value {
                        FieldValue::Duration(d) => d.unit.clone(),
                        ref other => panic!("unexpected {other:?}"),
                    },
                    match &pv.value {
                        FieldValue::Duration(d) => d.value,
                        _ => unreachable!(),
                    },
                    pv.rank,
                )
            })
            .collect()
    }

    fn one(state: &str, c: &FieldConstraints) -> (String, f64) {
        let v = durations(state, c);
        assert_eq!(v.len(), 1, "expected one candidate in {state:?}, got {v:?}");
        (v[0].1.clone(), v[0].2)
    }

    fn none(state: &str, c: &FieldConstraints) {
        let v = durations(state, c);
        assert!(
            v.is_empty(),
            "expected no candidate in {state:?}, got {v:?}"
        );
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn single_unit_durations() {
        let c = cons(false, None, None, None, None);
        assert_eq!(one("90 minutes", &c), ("min".to_string(), 90.0));
        assert_eq!(one("3 days", &c), ("d".to_string(), 3.0));
        assert_eq!(one("45s", &c), ("s".to_string(), 45.0));
        assert_eq!(one("2 weeks", &c), ("w".to_string(), 2.0));
        assert!(close(one("1.5 hours", &c).1, 1.5));
        assert_eq!(one("1.5 hours", &c).0, "h");
        assert_eq!(one("1 year", &c), ("y".to_string(), 1.0));
        assert_eq!(one("3 months", &c), ("mo".to_string(), 3.0));
    }

    #[test]
    fn attached_unit_spellings() {
        let c = cons(false, None, None, None, None);
        assert_eq!(one("2h", &c), ("h".to_string(), 2.0));
        assert_eq!(one("30m", &c), ("min".to_string(), 30.0));
        assert_eq!(one("3d", &c), ("d".to_string(), 3.0));
        assert_eq!(one("2mo", &c), ("mo".to_string(), 2.0));
        assert_eq!(one("1y", &c), ("y".to_string(), 1.0));
    }

    #[test]
    fn compound_chains_largest_unit() {
        let c = cons(false, None, None, None, None);
        assert_eq!(one("2h 30m", &c), ("h".to_string(), 2.5));
        let (unit, value) = one("3 hours 20 minutes", &c);
        assert_eq!(unit, "h");
        assert!(close(value, 3.0 + 20.0 / 60.0));
        let (unit, value) = one("2 weeks 3 days", &c);
        assert_eq!(unit, "w");
        assert!(close(value, 2.0 + 3.0 / 7.0));
        // Mixed attached/spaced terms.
        assert_eq!(one("2h 30 minutes", &c), ("h".to_string(), 2.5));
    }

    #[test]
    fn and_joined_chains() {
        let c = cons(false, None, None, None, None);
        let (unit, value) = one("1 hour and 20 minutes", &c);
        assert_eq!(unit, "h");
        assert!(close(value, 1.0 + 20.0 / 60.0));
        // A dangling "and" ends the chain before itself.
        assert_eq!(one("3 hours and", &c), ("h".to_string(), 3.0));
    }

    #[test]
    fn constraint_unit_exact_conversion() {
        assert_eq!(
            one("2h 30m", &cons(false, None, Some("min"), None, None)),
            ("min".to_string(), 150.0)
        );
        assert_eq!(
            one("1.5 hours", &cons(false, None, Some("min"), None, None)),
            ("min".to_string(), 90.0)
        );
        assert_eq!(
            one("3 days", &cons(false, None, Some("h"), None, None)),
            ("h".to_string(), 72.0)
        );
        assert_eq!(
            one("2 weeks", &cons(false, None, Some("d"), None, None)),
            ("d".to_string(), 14.0)
        );
        assert_eq!(
            one("1 hour", &cons(false, None, Some("min"), None, None)),
            ("min".to_string(), 60.0)
        );
        assert_eq!(
            one("45s", &cons(false, None, Some("s"), None, None)),
            ("s".to_string(), 45.0)
        );
    }

    #[test]
    fn constraint_unit_non_exact_abstains() {
        // 90 minutes is not a whole number of hours.
        none("90 minutes", &cons(false, None, Some("h"), None, None));
        // 45 seconds is not a whole number of minutes.
        none("45s", &cons(false, None, Some("min"), None, None));
        // 2.5 hours is not a whole number of days.
        none("2h 30m", &cons(false, None, Some("d"), None, None));
    }

    #[test]
    fn month_year_class_laws() {
        let c = cons(false, None, None, None, None);
        assert_eq!(
            one("3 months", &cons(false, None, Some("mo"), None, None)),
            ("mo".to_string(), 3.0)
        );
        assert_eq!(
            one("1 year", &cons(false, None, Some("y"), None, None)),
            ("y".to_string(), 1.0)
        );
        // Incommensurable with second-based units: no candidate.
        none("3 months", &cons(false, None, Some("d"), None, None));
        none("1 year", &cons(false, None, Some("mo"), None, None));
        none("2 weeks", &cons(false, None, Some("mo"), None, None));
        // Mixed-class chains have no candidate at all.
        none("1 month 3 days", &c);
        none("1 year 2 months", &c);
    }

    #[test]
    fn non_durations_produce_nothing() {
        let c = cons(false, None, None, None, None);
        none("5", &c); // a bare number is not a duration
        none("hours", &c); // a bare unit is not a duration
        none("abc", &c);
        none("9999999999 hours", &c); // term value >= 1e9 cap
        // A unitless tail ends the chain before itself.
        assert_eq!(one("3 hours 20", &c), ("h".to_string(), 3.0));
    }

    #[test]
    fn embedded_chain_span_exact() {
        let state = "wait 2h 30m ok";
        let c = cons(false, None, None, None, None);
        let v = durations(state, &c);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].0, "2h 30m");
        assert_eq!((v[0].3, v[0].1.as_str()), (0, "h"));
        let pv = {
            let mut out = Vec::new();
            parse_duration(state, &c, &mut out);
            out.swap_remove(0)
        };
        assert_eq!((pv.span.start, pv.span.end), (5, 11));
        assert_eq!(span_text(state, &pv), "2h 30m");
    }

    #[test]
    fn parse_twice_is_deterministic() {
        let c = cons(false, None, Some("min"), None, None);
        let a = durations("3 hours 20 minutes and 5s", &c);
        let b = durations("3 hours 20 minutes and 5s", &c);
        assert_eq!(a, b);
    }

    // ---- G4: allocation gate (crate::alloc_gate; see its module doc) -------

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn scan_durations_is_zero_alloc() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        // SCAN phase only: materialization (`DurationValue`'s unit String,
        // the owned constraint spelling) is the documented cold edge of
        // `parsers::finish` — the scanners themselves are push-only.
        let state = "terms net 30 days; late fees accrue after 2 weeks and 3 hours";
        let c = cons(false, None, None, None, None);
        // Dry run sizes the scratch; the measured run reuses it (clear keeps
        // capacity, so the same candidate count cannot grow the buffer).
        let mut scratch = Vec::new();
        scan_durations(state, 0..state.len(), &c, &mut scratch);
        let expected = scratch.len();
        assert!(expected > 0, "dry run must produce candidates");
        scratch.clear();

        let _window = MeasuringWindow::enter();
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        scan_durations(state, 0..state.len(), &c, &mut scratch);
        assert_eq!(
            ALLOC_COUNT.load(Ordering::Relaxed),
            0,
            "scan_durations allocated inside the measured window"
        );
        drop(_window);
        assert_eq!(scratch.len(), expected);
    }
}
