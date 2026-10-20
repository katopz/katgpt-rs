//! `number` — the Number dimension (Plan 628 T0.3).
//!
//! English number grammar translated from Microsoft **Recognizers-Text**
//! (MIT) surface forms and Meta **Duckling** (Apache-2.0) number semantics;
//! this module translates the published grammar shapes, never the code.
//!
//! ## Supported subset
//!
//! - Integers: `42`, `1,234` (comma groups of EXACTLY three digits;
//!   `1,23` and `1,2345` are not numbers), plain digit runs (`1234567`,
//!   capped at 18 digits).
//! - Decimals: `3.25` (integer part required; `.5` alone is not a number;
//!   up to 15 fraction digits).
//! - Negatives: `-5` (attached) and `minus 5` (word + whitespace).
//! - Currency: prefix symbols `$5`, `€3.25` (attached or one space away —
//!   `$ 14 million`, issue 953); suffix words `5 dollars`, `5 euros`,
//!   `5 usd`, `5 eur`, `5 yen`, `5 francs`, `5 cents`, `5 pence`. Value is
//!   numeric; the canonical unit (`usd`/`eur`/…) is used only for
//!   constraint matching — [`FieldValue::Number`] carries no unit.
//! - Percent: `12%`, `12 %`, `12 percent`, `12 percentage`, `12 percents`,
//!   and the prefix forms `percent of 10`, `per cent of twenty-two`. The
//!   value is the LITERAL (not a fraction); the canonical unit is
//!   `percent`.
//! - Unit suffixes: `5 km`, `12kg` (attached or spaced) over a documented
//!   length/mass/volume set (see [`NUMBER_UNITS`]). The preposition unit
//!   `in` is deliberately excluded (unbounded collision with English
//!   "in").
//! - Multipliers: attached `5k`, `1.5M`, `2B`; words `2 thousand`,
//!   `2 million`, `2 billion`, `a hundred`, `3 dozens` (×12), `25 lakh`
//!   (×100000), `a trillionth` (×1e-12) — chained (`a hundred trillionths`
//!   = 1e-10). Lowercase attached `m`/`b` are NOT multipliers (`5m` = 5
//!   meters). Multiplier + suffix combos compose (`2 million dollars`).
//! - Word numbers: the magnitude grammar of [`super::words`] — `sixteen`,
//!   `twenty percent`, `one hundred and sixteen`, `one mile`, `one penny`,
//!   `a dozen`.
//! - Fractions: `1/4`, `-4/5` (attached `/den`) and the mixed form
//!   `10 1/2` = 10.5, `- 1 4/5` = -1.8 (spaced sign allowed).
//!
//! ## Constraint enforcement (by construction — a violation yields NO
//! candidate, never a clamped one)
//!
//! - `integer`: value with a fractional part is rejected (`3.0` passes —
//!   the wire's own check is `fract() != 0`).
//! - `decimals`: caps the LITERAL fraction digit count (`3.14159` under
//!   `decimals = 2` is rejected; the value is never rounded).
//! - `min`/`max`: inclusive bounds.
//! - `unit`: a bare number abstains when a unit is required (the parser
//!   cannot confirm the unit); a unit-qualified number passes only when its
//!   canonical unit matches the constraint case-insensitively.
//! - Magnitude: |value| ≥ 1e15 is rejected (f64 exactness floor at 2^53).

use super::{
    ParsedValue, RawCandidate, RawValue, ascii_word_at, finish, is_word_byte, next_boundary,
    prev_char_index, skip_ascii_ws,
    words::{article_one_at, word_number_at},
};
use katgpt_core::extraction_wire::FieldConstraints;

/// Number candidates: rank 0 = unit/currency/percent-qualified, 1 =
/// multiplier-qualified, 2 = plain full core, 3 = integer part of a
/// decimal, 4 = first comma group. Longer matches rank better where they
/// overlap.
const RANK_QUALIFIED: u8 = 0;
const RANK_MULTIPLIED: u8 = 1;
const RANK_PLAIN: u8 = 2;
const RANK_PREFRAC: u8 = 3;
const RANK_FIRST_GROUP: u8 = 4;

/// Reject values at or above this magnitude: f64 is exact only up to 2^53
/// (≈ 9.0e15), so the parser refuses the lossy regime outright.
const VALUE_CAP: f64 = 1e15;

/// Unit-suffixed surface forms: (spelling, canonical). Used for constraint
/// matching only.
const NUMBER_UNITS: &[(&str, &str)] = &[
    ("centimeters", "cm"),
    ("millimeters", "mm"),
    ("milliliters", "ml"),
    ("millilitres", "ml"),
    ("kilometers", "km"),
    ("centimeter", "cm"),
    ("millimeter", "mm"),
    ("milliliter", "ml"),
    ("millilitre", "ml"),
    ("kilometers", "km"),
    ("kilograms", "kg"),
    ("kilogram", "kg"),
    ("meters", "m"),
    ("miles", "mi"),
    ("pounds", "lb"),
    ("gallons", "gal"),
    ("liters", "l"),
    ("litres", "l"),
    ("ounces", "oz"),
    ("quarts", "qt"),
    ("tonnes", "t"),
    ("pints", "pt"),
    ("meter", "m"),
    ("mile", "mi"),
    ("gram", "g"),
    ("grams", "g"),
    ("gallon", "gal"),
    ("pound", "lb"),
    ("feet", "ft"),
    ("foot", "ft"),
    ("liter", "l"),
    ("litre", "l"),
    ("ounce", "oz"),
    ("quart", "qt"),
    ("tonne", "t"),
    ("ton", "t"),
    ("tons", "t"),
    ("pint", "pt"),
    ("kilo", "kg"),
    ("kilos", "kg"),
    ("km", "km"),
    ("cm", "cm"),
    ("mm", "mm"),
    ("mi", "mi"),
    ("ft", "ft"),
    ("kg", "kg"),
    ("lbs", "lb"),
    ("lb", "lb"),
    ("ml", "ml"),
    ("mls", "ml"),
    ("m", "m"),
    ("g", "g"),
];

const CURRENCY_WORDS: &[(&str, &str)] = &[
    ("dollars", "usd"),
    ("euros", "eur"),
    ("francs", "franc"),
    ("pennies", "penny"),
    ("dollar", "usd"),
    ("euro", "eur"),
    ("pence", "pence"),
    ("penny", "penny"),
    ("cents", "cents"),
    ("franc", "franc"),
    ("usd", "usd"),
    ("eur", "eur"),
    ("yen", "yen"),
    ("cent", "cents"),
];

/// Spelled-out percent suffixes (the wire's canonical `percent`).
const PERCENT_WORDS: &[(&str, &str)] = &[
    ("percentage", "percent"),
    ("percents", "percent"),
    ("percent", "percent"),
];

/// Word multipliers, chained (`a hundred trillionths` = 1e-10); the
/// fraction words divide (`two thirds` = 2/3).
const MULTIPLIER_WORDS: &[(&str, f64)] = &[
    ("trillionths", 1e-12),
    ("trillionth", 1e-12),
    ("thirds", 1.0 / 3.0),
    ("third", 1.0 / 3.0),
    ("thousand", 1e3),
    ("hundred", 1e2),
    ("million", 1e6),
    ("billion", 1e9),
    ("dozens", 12.0),
    ("dozen", 12.0),
    ("lakh", 1e5),
];

/// Prefix-percent surface forms (`per cent of twenty-two` = 22 percent).
const PERCENT_PREFIXES: &[&[&str]] = &[
    &["percentage", "of"],
    &["percents", "of"],
    &["percent", "of"],
    &["per", "cent", "of"],
];

/// Parse every Number candidate in `state` and append them best-first.
pub fn parse_number(state: &str, constraints: &FieldConstraints, out: &mut Vec<ParsedValue>) {
    if state.len() > u32::MAX as usize {
        return;
    }
    // Fixed scratch floor (T0.6 G4): a typical state's candidate count fits
    // without growth — `Field::Number` materialization carries no `String`,
    // so a parse with reserved scratch + reserved `out` is zero-alloc (the
    // G4 gate in this module's tests pins it). Growth beyond the floor is
    // the documented realloc edge, never a correctness one.
    let mut scratch = Vec::with_capacity(8);
    scan_numbers(state, 0..state.len(), constraints, &mut scratch);
    finish(scratch, state, out);
}

/// The number scanner. Zero-alloc: pushes [`RawCandidate`]s only.
pub(crate) fn scan_numbers(
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
        if number_start_ok(b, ws, i) {
            try_number_at(b, we, i, cons, sink);
        }
        i += 1;
    }
}

/// A candidate start must not continue a word, must not sit on the tail of
/// a decimal (`1.50` must not yield `50`), and must not sit inside a
/// thousands group (`1,234` must not yield `234`) — the group test is
/// "previous char is a comma AND a digit before it AND exactly three
/// digits after it".
fn number_start_ok(b: &[u8], ws: usize, i: usize) -> bool {
    let Some(j) = prev_char_index(b, ws, i) else {
        return true;
    };
    let c = b[j];
    if is_word_byte(c) || matches!(c, b'.' | b'+') {
        return false;
    }
    if c == b',' {
        let digit_before = j > 0 && b[j - 1].is_ascii_digit();
        let three_after = j + 4 <= b.len()
            && b[j + 1].is_ascii_digit()
            && b[j + 2].is_ascii_digit()
            && b[j + 3].is_ascii_digit();
        if digit_before && three_after {
            return false;
        }
    }
    true
}

/// One numeric core: grouped/plain integer part + optional decimal
/// fraction. No sign, currency, multiplier, or suffix — those live in the
/// scanner. Shared with the duration and list scanners.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoreOut {
    /// Byte end of the full core (fraction included).
    pub end: usize,
    /// Numeric value (sign NOT applied).
    pub value: f64,
    /// Fraction digit count in the literal.
    pub frac_digits: u32,
    /// Exact rational numerator/denominator (`value == num / den`), for
    /// exact duration conversions.
    pub num: i128,
    pub den: i64,
    /// Plain-core fallback candidate ends: `[0]` = full core, `[1]` =
    /// integer part before the fraction (end 0 = absent), `[2]` = first
    /// comma group (end 0 = absent).
    pub stops: [(usize, f64); 3],
}

/// Parse a numeric core at `i`. Returns `None` unless `b[i]` starts a
/// number.
pub(crate) fn core_at(b: &[u8], we: usize, i: usize) -> Option<CoreOut> {
    if i >= we || !b[i].is_ascii_digit() {
        return None;
    }
    // First group: up to 3 digits, then decide the mode.
    let mut e = i;
    while e < we && e - i < 3 && b[e].is_ascii_digit() {
        e += 1;
    }
    let first_end = e;
    let plain_mode = e < we && b[e].is_ascii_digit();
    let mut grouped = false;
    if plain_mode {
        // A 4th digit means no comma grouping: consume the whole run.
        while e < we && b[e].is_ascii_digit() && e - i < 18 {
            e += 1;
        }
        if e < we && b[e].is_ascii_digit() {
            return None; // > 18 digits
        }
    } else {
        // Grouping mode: (',' + exactly three digits + non-digit)*.
        while e + 4 <= we
            && b[e] == b','
            && b[e + 1].is_ascii_digit()
            && b[e + 2].is_ascii_digit()
            && b[e + 3].is_ascii_digit()
            && (e + 4 >= we || !b[e + 4].is_ascii_digit())
        {
            e += 4;
            grouped = true;
            if e - i > 18 {
                return None;
            }
        }
    }
    let grouping = grouped;
    let mut int_value: u128 = 0;
    for &c in &b[i..e] {
        if c.is_ascii_digit() {
            int_value = int_value * 10 + u128::from(c - b'0');
        }
    }
    let first_value = {
        let mut v: u128 = 0;
        for &c in &b[i..first_end] {
            v = v * 10 + u128::from(c - b'0');
        }
        v
    };
    // Optional fraction.
    let prefrac_end = e;
    let mut frac_digits: u32 = 0;
    let mut frac_value: u128 = 0;
    let has_frac = e + 1 < we && b[e] == b'.' && b[e + 1].is_ascii_digit();
    if has_frac {
        e += 1;
        while e < we && b[e].is_ascii_digit() && frac_digits < 15 {
            frac_value = frac_value * 10 + u128::from(b[e] - b'0');
            frac_digits += 1;
            e += 1;
        }
        if e < we && b[e].is_ascii_digit() {
            return None; // > 15 fraction digits
        }
    }
    let scale = 10f64.powi(frac_digits as i32);
    let value = int_value as f64 + frac_value as f64 / scale;
    let den = 10i64.pow(frac_digits);
    let num = int_value as i128 * i128::from(den) + frac_value as i128;
    let full = (e, value);
    let s1 = if has_frac {
        (prefrac_end, int_value as f64)
    } else {
        (0, 0.0)
    };
    let s2 = if grouping {
        (first_end, first_value as f64)
    } else {
        (0, 0.0)
    };
    Some(CoreOut {
        end: e,
        value,
        frac_digits,
        num,
        den,
        stops: [full, s1, s2],
    })
}

/// The word-number core: the magnitude grammar, or `a`/`an` (=1) directly
/// before a magnitude/dozen word (`a dozen`).
fn word_core_at(b: &[u8], we: usize, i: usize) -> Option<(usize, f64)> {
    if let Some(hit) = word_number_at(b, we, i) {
        return Some(hit);
    }
    article_one_at(b, we, i).map(|e| (e, 1.0))
}

/// Does a word-number core start at `i`? (the sign paths' lookahead)
fn word_core_starts(b: &[u8], we: usize, i: usize) -> bool {
    word_core_at(b, we, i).is_some()
}

/// 1-2 digit run (the fraction numerator/denominator).
fn one_two_digits_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    if i >= we || !b[i].is_ascii_digit() {
        return None;
    }
    let mut v = u32::from(b[i] - b'0');
    let mut e = i + 1;
    if e < we && b[e].is_ascii_digit() {
        v = v * 10 + u32::from(b[e] - b'0');
        e += 1;
    }
    Some((e, v))
}

/// `/den` at `i` — 1-2 digit non-zero denominator, hard boundary after
/// (a trailing `/` belongs to a date-like shape, not this fraction).
fn slash_den_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    if i >= we || b[i] != b'/' {
        return None;
    }
    let (de, den) = one_two_digits_at(b, we, i + 1)?;
    if den == 0 {
        return None;
    }
    if de < we && (is_word_byte(b[de]) || b[de] == b'/') {
        return None;
    }
    Some((de, den))
}

/// Fraction tail after a digit core: a whitespace-separated `num/den`
/// (the `10 1/2` shape — an ADDEND). The attached `/den` form is handled
/// at the call site (there the core is the NUMERATOR: `1/4` = 1/4).
fn fraction_tail_at(b: &[u8], we: usize, after: usize) -> Option<(usize, f64, usize)> {
    let j = skip_ascii_ws(b, after, we);
    if j == after {
        return None;
    }
    let (ne, num) = one_two_digits_at(b, we, j)?;
    let (de, den) = slash_den_at(b, we, ne)?;
    Some((de, f64::from(num) / f64::from(den), after))
}

/// A prefix-percent form at `i` (`per cent of 210`). Returns the offset
/// just past the prefix.
fn percent_prefix_at(b: &[u8], we: usize, i: usize) -> Option<usize> {
    for words in PERCENT_PREFIXES {
        let mut j = i;
        let mut ok = true;
        for (n, w) in words.iter().enumerate() {
            let k = if n == 0 { j } else { skip_ascii_ws(b, j, we) };
            if n > 0 && k == j {
                ok = false;
                break;
            }
            match ascii_word_at(b, we, k, w) {
                Some(e) => j = e,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(j);
        }
    }
    None
}

fn try_number_at(
    b: &[u8],
    we: usize,
    s: usize,
    cons: &FieldConstraints,
    sink: &mut Vec<RawCandidate>,
) {
    let mut i = s;
    // Prefix-percent forms: "per cent of twenty-two" -> 22 percent.
    let mut percent_ctx = false;
    if let Some(pe) = percent_prefix_at(b, we, i) {
        percent_ctx = true;
        i = skip_ascii_ws(b, pe, we);
        if i >= we {
            return;
        }
    }
    // Sign: attached '-' (whitespace optional) or the word "minus" + ws.
    let mut neg = false;
    if b[i] == b'-' {
        let j = skip_ascii_ws(b, i + 1, we);
        if j < we && (b[j].is_ascii_digit() || word_core_starts(b, we, j)) {
            neg = true;
            i = j;
        } else {
            return;
        }
    } else if let Some(e) = ascii_word_at(b, we, i, "minus") {
        let j = skip_ascii_ws(b, e, we);
        if j > e && j < we && (b[j].is_ascii_digit() || word_core_starts(b, we, j)) {
            neg = true;
            i = j;
        } else {
            return;
        }
    }
    // Currency prefix symbols, attached to the core or ONE space away
    // ("$ 14 million" — issue 953 lever 2).
    let mut currency: Option<&'static str> = None;
    if b[i] == b'$' {
        let j = skip_ascii_ws(b, i + 1, we);
        if j < we && b[j].is_ascii_digit() {
            currency = Some("usd");
            i = j;
        }
    } else if b[i] == 0xE2 && i + 3 < we && b[i + 1] == 0x82 && b[i + 2] == 0xAC {
        let j = skip_ascii_ws(b, i + 3, we);
        if j < we && b[j].is_ascii_digit() {
            currency = Some("eur");
            i = j;
        }
    }
    // The core: digits (fraction tail + fallback stops) or words.
    let core_start = i;
    let mut is_word_core = false;
    let mut core_end;
    let mut value;
    let mut frac_digits: u32 = 0;
    let mut stops = [(0usize, 0.0f64); 3];
    let mut stops_frac = [0u32; 3];
    if let Some(core) = core_at(b, we, i) {
        core_end = core.end;
        value = core.value;
        frac_digits = core.frac_digits;
        stops = core.stops;
        stops_frac[0] = frac_digits;
        // Attached `/den`: the core IS the numerator — `1/4` = 0.25.
        if let Some((de, den)) = slash_den_at(b, we, core_end) {
            value = core.value / f64::from(den);
            stops[0] = (de, value);
            stops_frac[0] = 0;
            if stops[1].0 == 0 {
                stops[1] = (core.end, core.value);
            }
            core_end = de;
            frac_digits = 0;
        } else if let Some((fe, addend, prefrac)) = fraction_tail_at(b, we, core_end) {
            // Spaced `num/den` is an ADDEND — `10 1/2` = 10.5.
            value += addend;
            stops[0] = (fe, value);
            stops_frac[0] = 0;
            if stops[1].0 == 0 {
                stops[1] = (prefrac, core.stops[0].1);
            }
            core_end = fe;
            frac_digits = 0;
        }
    } else if let Some((w_end, w_val)) = word_core_at(b, we, i) {
        is_word_core = true;
        core_end = w_end;
        value = w_val;
    } else {
        return;
    }
    let mut e = core_end;
    // Multipliers, chained: attached letters (digit cores only), then
    // word multipliers ("2 million", "a hundred trillionths").
    let mut mult: Option<(f64, usize)> = None;
    let mut mult_value: f64 = 1.0;
    loop {
        let mut consumed = false;
        if !is_word_core && e < we {
            let factor = match b[e] {
                b'k' | b'K' => Some(1e3),
                b'M' => Some(1e6),
                b'B' => Some(1e9),
                _ => None,
            };
            if let Some(factor) = factor {
                let ne = e + 1;
                if ne >= we || !is_word_byte(b[ne]) {
                    mult = Some((factor, ne));
                    mult_value *= factor;
                    e = ne;
                    consumed = true;
                }
            }
        }
        if !consumed {
            let j = skip_ascii_ws(b, e, we);
            if j > e {
                for (word, factor) in MULTIPLIER_WORDS {
                    if let Some(we2) = ascii_word_at(b, we, j, word) {
                        mult = Some((*factor, we2));
                        mult_value *= *factor;
                        e = we2;
                        consumed = true;
                        break;
                    }
                }
            }
        }
        if !consumed {
            break;
        }
    }
    // Suffixes: attached percent, attached unit (digit cores, no
    // multiplier), then spaced percent/unit/currency word.
    let mut unit: Option<&'static str> = None;
    if e < we && b[e] == b'%' {
        unit = Some("percent");
        e += 1;
    } else if !is_word_core
        && mult.is_none()
        && let Some(hit) = table_word_at(b, we, e, NUMBER_UNITS)
    {
        unit = Some(hit.0);
        e = hit.1;
    }
    if unit.is_none() {
        let j = skip_ascii_ws(b, e, we);
        if j > e && j < we {
            if b[j] == b'%' {
                unit = Some("percent");
                e = j + 1;
            } else if let Some(hit) = suffix_word_at(b, we, j) {
                unit = Some(hit.0);
                e = hit.1;
            }
        }
    }
    if percent_ctx {
        unit = Some("percent");
    }
    // The whole match must end at a token boundary.
    if e < we && (is_word_byte(b[e]) || b[e] == b'%' || b[e] == b'$') {
        return;
    }
    let qualified = currency.is_some() || unit.is_some();
    let scaled = value * mult_value;
    let full_value = if neg { -scaled } else { scaled };
    // Composition depth — the qualified tie-break (see RawCandidate).
    let depth =
        currency.is_some() as u8 + mult.is_some() as u8 + (unit.is_some() || percent_ctx) as u8;
    // The full match (sign/currency/core/multipliers/suffix).
    if accepts(cons, full_value, frac_digits, currency.or(unit)) {
        let rank = if qualified {
            RANK_QUALIFIED
        } else if mult.is_some() {
            RANK_MULTIPLIED
        } else {
            RANK_PLAIN
        };
        sink.push(RawCandidate {
            start: s as u32,
            end: e as u32,
            rank,
            depth,
            raw: RawValue::Number(full_value),
        });
    }
    // Under a qualified match, the multiplied core without the suffix is a
    // mid fallback ("2 million dollars" -> "2 million").
    if qualified && let Some((_, mult_end)) = mult {
        let v = value * mult_value;
        let v = if neg { -v } else { v };
        if accepts(cons, v, frac_digits, None) {
            sink.push(RawCandidate {
                start: core_start as u32,
                end: mult_end as u32,
                rank: RANK_MULTIPLIED,
                depth: 1,
                raw: RawValue::Number(v),
            });
        }
    }
    // Plain-core fallback stops (the plain number inside "$1,234.50").
    let ladder = [RANK_PLAIN, RANK_PREFRAC, RANK_FIRST_GROUP];
    for (slot, rank) in ladder.into_iter().enumerate() {
        let (stop_end, stop_value) = stops[slot];
        if stop_end == 0 || stop_end == core_start {
            continue;
        }
        let v = if neg { -stop_value } else { stop_value };
        if accepts(cons, v, stops_frac[slot], None) {
            sink.push(RawCandidate {
                start: core_start as u32,
                end: stop_end as u32,
                rank,
                depth: 0,
                raw: RawValue::Number(v),
            });
        }
    }
}

/// Constraint gate: a violation yields NO candidate.
fn accepts(cons: &FieldConstraints, value: f64, frac_digits: u32, unit: Option<&str>) -> bool {
    if !value.is_finite() || value.abs() >= VALUE_CAP {
        return false;
    }
    if cons.integer && value.fract() != 0.0 {
        return false;
    }
    if let Some(d) = cons.decimals
        && frac_digits > u32::from(d)
    {
        return false;
    }
    if let Some(lo) = cons.min
        && value < lo
    {
        return false;
    }
    if let Some(hi) = cons.max
        && value > hi
    {
        return false;
    }
    if let Some(want) = cons.unit.as_deref() {
        match unit {
            Some(u) if u.eq_ignore_ascii_case(want) => {}
            _ => return false,
        }
    }
    true
}

fn table_word_at(
    b: &[u8],
    we: usize,
    i: usize,
    table: &[(&'static str, &'static str)],
) -> Option<(&'static str, usize)> {
    for (spelling, canon) in table {
        if let Some(e) = ascii_word_at(b, we, i, spelling) {
            return Some((canon, e));
        }
    }
    None
}

fn suffix_word_at(b: &[u8], we: usize, i: usize) -> Option<(&'static str, usize)> {
    if let Some(hit) = table_word_at(b, we, i, NUMBER_UNITS) {
        return Some(hit);
    }
    if let Some(hit) = table_word_at(b, we, i, CURRENCY_WORDS) {
        return Some(hit);
    }
    table_word_at(b, we, i, PERCENT_WORDS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::{cons, span_text};
    use katgpt_core::extraction_wire::FieldValue;

    fn one(state: &str, cons: &FieldConstraints) -> ParsedValue {
        let mut out = Vec::new();
        parse_number(state, cons, &mut out);
        assert!(!out.is_empty(), "expected a candidate in {state:?}");
        out.swap_remove(0)
    }

    fn none(state: &str, cons: &FieldConstraints) {
        let mut out = Vec::new();
        parse_number(state, cons, &mut out);
        assert!(
            out.is_empty(),
            "expected no candidate in {state:?}, got {:?}",
            out.first().map(|p| &p.value)
        );
    }

    fn val(pv: &ParsedValue) -> f64 {
        match pv.value {
            FieldValue::Number(n) => n,
            ref other => panic!("not a number: {other:?}"),
        }
    }

    #[test]
    fn plain_integers_and_comma_groups() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("42", &c)), 42.0);
        let pv = one("1,234", &c);
        assert_eq!(val(&pv), 1234.0);
        assert_eq!(span_text("1,234", &pv), "1,234");
        // "1,234" is ONE number — the head candidate — with the first-group
        // "1" trailing as a lower-ranked fallback.
        let mut out = Vec::new();
        parse_number("1,234", &c, &mut out);
        assert_eq!(span_text("1,234", &out[0]), "1,234");
        assert!(out.iter().any(|p| span_text("1,234", p) == "1"));
        assert!(!out.iter().any(|p| span_text("1,234", p) == "234"));
        assert_eq!(val(&one("1234567", &c)), 1_234_567.0);
    }

    #[test]
    fn decimals_parse() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("3.25", &c)), 3.25);
        assert_eq!(val(&one("0.5", &c)), 0.5);
    }

    #[test]
    fn negative_sign_and_minus_word() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("-5", &c)), -5.0);
        let pv = one("minus 5", &c);
        assert_eq!(val(&pv), -5.0);
        assert_eq!(span_text("minus 5", &pv), "minus 5");
        assert_eq!(val(&one("minus 2.5", &c)), -2.5);
    }

    #[test]
    fn currency_prefix_and_suffix_words() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("$5", &c)), 5.0);
        assert_eq!(val(&one("€3.25", &c)), 3.25);
        assert_eq!(val(&one("5 dollars", &c)), 5.0);
        assert_eq!(val(&one("5 euros", &c)), 5.0);
        // Span includes the currency marker.
        let pv = one("cost: $1,234.50 today", &c);
        assert_eq!(span_text("cost: $1,234.50 today", &pv), "$1,234.50");
    }

    #[test]
    fn currency_respects_unit_constraint() {
        assert_eq!(
            val(&one("$5", &cons(false, None, Some("usd"), None, None))),
            5.0
        );
        assert_eq!(
            val(&one(
                "5 dollars",
                &cons(false, None, Some("USD"), None, None)
            )),
            5.0
        );
        none("$5", &cons(false, None, Some("km"), None, None));
        none("5 dollars", &cons(false, None, Some("eur"), None, None));
    }

    #[test]
    fn percent_forms() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("12%", &c)), 12.0);
        assert_eq!(val(&one("12 percent", &c)), 12.0);
        assert_eq!(val(&one("12 %", &c)), 12.0);
        assert_eq!(
            val(&one("12%", &cons(false, None, Some("percent"), None, None))),
            12.0
        );
        none("12%", &cons(false, None, Some("km"), None, None));
    }

    #[test]
    fn unit_suffixed_with_and_without_space() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("5 km", &c)), 5.0);
        assert_eq!(val(&one("12kg", &c)), 12.0);
        assert_eq!(val(&one("5 miles", &c)), 5.0);
        assert_eq!(
            val(&one("5 km", &cons(false, None, Some("km"), None, None))),
            5.0
        );
        assert_eq!(
            val(&one("12kg", &cons(false, None, Some("KG"), None, None))),
            12.0
        );
        // Unit mismatch: NO candidate.
        none("5 km", &cons(false, None, Some("mi"), None, None));
        none("12kg", &cons(false, None, Some("g"), None, None));
        // A bare number abstains when a unit is required.
        none("5", &cons(false, None, Some("km"), None, None));
    }

    #[test]
    fn multipliers_expand() {
        let c = cons(false, None, None, None, None);
        assert_eq!(val(&one("5k", &c)), 5_000.0);
        assert_eq!(val(&one("1.5M", &c)), 1_500_000.0);
        assert_eq!(val(&one("2 million", &c)), 2_000_000.0);
        assert_eq!(val(&one("2 billion", &c)), 2_000_000_000.0);
        assert_eq!(val(&one("2B", &c)), 2_000_000_000.0);
        // Multiplier + currency word compose.
        assert_eq!(val(&one("2 million dollars", &c)), 2_000_000.0);
        // Lowercase attached m/b are NOT multipliers.
        assert_eq!(
            val(&one("5m", &cons(false, None, Some("m"), None, None))),
            5.0
        );
    }

    #[test]
    fn integer_constraint_rejects_fractions() {
        let c = cons(true, None, None, None, None);
        assert_eq!(val(&one("42", &c)), 42.0);
        assert_eq!(val(&one("3.0", &c)), 3.0); // integral value passes
        // No candidate may span a fractional literal; the plain
        // integer-part stop ("3") is a legal integral fallback.
        let mut out = Vec::new();
        parse_number("3.5", &c, &mut out);
        assert!(!out.iter().any(|p| span_text("3.5", p) == "3.5"));
        assert!(out.iter().all(|p| val(p).fract() == 0.0));
        let mut out = Vec::new();
        parse_number("1,234.50", &c, &mut out);
        assert!(!out.iter().any(|p| val(p) == 1234.5));
        assert!(out.iter().any(|p| val(p) == 1234.0)); // the integer-part stop
    }

    #[test]
    fn decimals_cap_rejects_extra_places() {
        let c = cons(false, Some(2), None, None, None);
        assert_eq!(val(&one("3.25", &c)), 3.25);
        assert_eq!(val(&one("3", &c)), 3.0);
        // The over-cap literal never becomes a candidate (never rounded).
        let mut out = Vec::new();
        parse_number("3.14159", &c, &mut out);
        assert!(!out.iter().any(|p| span_text("3.14159", p) == "3.14159"));
        let mut out = Vec::new();
        parse_number("1.50", &cons(false, Some(1), None, None, None), &mut out); // 2 fraction digits > cap 1
        assert!(!out.iter().any(|p| span_text("1.50", p) == "1.50"));
    }

    #[test]
    fn min_max_bounds_are_inclusive_filters() {
        none("5", &cons(false, None, None, Some(10.0), None));
        none("15", &cons(false, None, None, None, Some(10.0)));
        assert_eq!(
            val(&one("10", &cons(false, None, None, Some(10.0), Some(10.0)))),
            10.0
        );
        assert_eq!(
            val(&one("10", &cons(false, None, None, Some(5.0), None))),
            10.0
        );
        assert_eq!(
            val(&one("-5", &cons(false, None, None, Some(-5.0), None))),
            -5.0
        );
    }

    #[test]
    fn embedded_in_prose_with_exact_span() {
        let state = "the total is 1,234 dollars today";
        let mut out = Vec::new();
        parse_number(state, &cons(false, None, None, None, None), &mut out);
        let pv = out
            .iter()
            .find(|p| span_text(state, p) == "1,234 dollars")
            .expect("currency-qualified candidate");
        assert_eq!(pv.rank, 0);
        assert_eq!(val(pv), 1234.0);
    }

    #[test]
    fn malformed_comma_groups_are_not_numbers() {
        let c = cons(false, None, None, None, None);
        // "1,23" -> only "1" matches; no candidate spans "1,23".
        let mut out = Vec::new();
        parse_number("1,23", &c, &mut out);
        assert!(!out.iter().any(|p| span_text("1,23", p) == "1,23"));
        assert!(out.iter().any(|p| span_text("1,23", p) == "1"));
        // "1,2345" -> only "1".
        let mut out = Vec::new();
        parse_number("1,2345", &c, &mut out);
        assert!(!out.iter().any(|p| span_text("1,2345", p) == "1,2345"));
        assert!(out.iter().any(|p| span_text("1,2345", p) == "1"));
    }

    #[test]
    fn word_adjacent_digits_are_rejected() {
        let c = cons(false, None, None, None, None);
        none("abc123", &c);
        none("123abc", &c);
        // Range-like text still yields the two plain sides.
        let mut out = Vec::new();
        parse_number("1-5", &c, &mut out);
        assert!(out.iter().any(|p| val(p) == 1.0) && out.iter().any(|p| val(p) == 5.0));
    }

    #[test]
    fn rank_orders_qualified_over_plain_overlaps() {
        let c = cons(false, None, None, None, None);
        let mut out = Vec::new();
        parse_number("$1,234.50", &c, &mut out);
        let texts: Vec<&str> = out.iter().map(|p| span_text("$1,234.50", p)).collect();
        assert_eq!(texts.first(), Some(&"$1,234.50"));
        assert_eq!(out[0].rank, 0);
        // The plain ladder follows: full core, integer part, first group.
        assert!(texts.contains(&"1,234.50") && texts.contains(&"1,234") && texts.contains(&"1"));
        for pair in out.windows(2) {
            assert!(pair[0].rank <= pair[1].rank);
        }
    }

    #[test]
    fn numeric_cap_rejects_huge_literals() {
        let c = cons(false, None, None, None, None);
        none("99999999999999999", &c); // 17 digits >= 1e15 cap
        none("123456789012345678901", &c); // > 18 digits: not a number at all
        assert_eq!(val(&one("99999999999999", &c)), 99_999_999_999_999.0);
    }

    #[test]
    fn unsupported_surface_forms_yield_nothing() {
        let c = cons(false, None, None, None, None);
        none(".5", &c); // leading-dot decimals out of scope
        none("5$", &c); // trailing currency symbol out of scope
        none("+5", &c); // explicit plus out of scope
        none("eleventy", &c); // the number-word table is finite (issue 953)
    }

    // ---- G4: allocation gate (crate::alloc_gate; see its module doc) ------

    // FULL-parse zero-alloc is deliberately NOT gated: the scratch Vec
    // lives inside `parse_number` (the with_capacity(8) floor allocates it
    // once per call) and `finish`'s stable sort carries its own std scratch
    // — removing both needs the scratch threaded through the public API, a
    // real refactor out of scope per plan 628 T0.6. What the floor fix BOUGHT:
    // the per-call cost is one FIXED block (no geometric growth), and
    // materialization is `FieldValue::Number(f64)` — no String. The zero-alloc
    // law is pinned at the scan phase below (the scanner is the reusable hot
    // half; callers that drive it directly — the engine's sweep — run it
    // alloc-free).
    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn scan_numbers_is_zero_alloc() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        let state = "invoice total $1,234.50 and 42 units minus 5 at 18 percent";
        let c = cons(false, None, None, None, None);
        // Dry run sizes the scratch; the measured run reuses it (clear keeps
        // capacity, so the same candidate count cannot grow the buffer).
        let mut scratch = Vec::new();
        scan_numbers(state, 0..state.len(), &c, &mut scratch);
        let expected = scratch.len();
        assert!(expected > 0, "dry run must produce candidates");
        scratch.clear();

        let _window = MeasuringWindow::enter();
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        scan_numbers(state, 0..state.len(), &c, &mut scratch);
        assert_eq!(
            ALLOC_COUNT.load(Ordering::Relaxed),
            0,
            "scan_numbers allocated inside the measured window"
        );
        drop(_window);
        assert_eq!(scratch.len(), expected);
    }
}
