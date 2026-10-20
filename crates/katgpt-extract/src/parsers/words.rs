//! `words` — English number-WORD grammar (Plan 628 / issue 953 lever 1).
//!
//! Surface forms translated from Microsoft **Recognizers-Text** (MIT) and
//! Meta **Duckling** (Apache-2.0); the grammar shapes, never the code.
//! Zero-alloc byte scanning, house parser style (no allocation, hard word
//! boundaries via [`ascii_word_at`]).
//!
//! ## Supported subset
//!
//! - [`word_group_at`]: one 0-99 group — `forty five`, `twenty-one`,
//!   `seventeen`. The hour/minute/day building block.
//! - [`word_number_at`]: the magnitude grammar — `twenty and four` (24),
//!   `one hundred and three` (103), `two thousand three hundred` (2300),
//!   `hundred thousand` (100000), `three million`. Additive groups compose
//!   only at strictly decreasing magnitude (`one two` is NOT a number);
//!   magnitude words multiply (`two thousand` = 2000). `a`/`an` count as 1
//!   only directly before a magnitude or dozen word (see
//!   [`article_one_at`]) — a bare article is never a number.
//! - [`word_year_at`]: written years 1000..=9999 — the thousand form
//!   (`two thousand`, `two thousand and eighteen`, `nineteen hundred`) or
//!   the digit-smash form (`nineteen seventy eight` = 1978,
//!   `nineteen zero one` = 1901).
//! - [`word_ordinal_day_at`]: ordinal day words `first`..`thirty-first`
//!   (1..=31).

use super::{ascii_word_at, skip_ascii_ws};

/// Ones 0-9.
const ONES: &[(&str, i64)] = &[
    ("zero", 0),
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
];

/// Teens 10-19.
const TEENS: &[(&str, i64)] = &[
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
];

/// Round tens 20-90.
const TENS: &[(&str, i64)] = &[
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
];

/// Magnitude words (multipliers). Bare `hundred`/`thousand` are legal
/// (`hundred thousand` = 100000).
const MAGNITUDES: &[(&str, i64)] = &[
    ("hundred", 100),
    ("thousand", 1_000),
    ("million", 1_000_000),
    ("billion", 1_000_000_000),
];

/// Dozen words (the x12 multiplier; `a dozen`, `3 dozens`).
pub(crate) const DOZENS: &[(&str, i64)] = &[("dozens", 12), ("dozen", 12)];

/// Tiny-magnitude spellings — legal article followers (`a trillionth`)
/// whose value handling lives in the number parser's multiplier chain.
const SMALL_MAGNITUDES: &[&str] = &["trillionths", "trillionth"];

/// Ordinal day words 1..=31 (`tenth`, `twenty-first`, `thirtieth`).
const ORDINAL_DAYS: &[(&str, u32)] = &[
    ("twenty-first", 21),
    ("twenty-second", 22),
    ("twenty-third", 23),
    ("twenty-fourth", 24),
    ("twenty-fifth", 25),
    ("twenty-sixth", 26),
    ("twenty-seventh", 27),
    ("twenty-eighth", 28),
    ("twenty-ninth", 29),
    ("thirty-first", 31),
    ("thirtieth", 30),
    ("thirteenth", 13),
    ("fourteenth", 14),
    ("fifteenth", 15),
    ("sixteenth", 16),
    ("seventeenth", 17),
    ("eighteenth", 18),
    ("nineteenth", 19),
    ("twentieth", 20),
    ("eleventh", 11),
    ("twelfth", 12),
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
    ("eighth", 8),
    ("ninth", 9),
    ("tenth", 10),
];

/// One 0-99 group: tens + optional ones (hyphen or space joined), a teen,
/// or a one. `ascii_word_at`'s hard boundary makes table order safe.
pub(crate) fn word_group_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    for (w, v) in TENS {
        if let Some(e) = ascii_word_at(b, we, i, w) {
            // Optional (-|ws)+ ones.
            let mut j = e;
            if j < we && b[j] == b'-' {
                j += 1;
            }
            let k = skip_ascii_ws(b, j, we);
            let probe = if k > j { k } else { j };
            for (ow, ov) in ONES {
                if let Some(oe) = ascii_word_at(b, we, probe, ow) {
                    return Some((oe, v + ov));
                }
            }
            return Some((e, *v));
        }
    }
    for table in [TEENS, ONES] {
        for (w, v) in table {
            if let Some(e) = ascii_word_at(b, we, i, w) {
                return Some((e, *v));
            }
        }
    }
    None
}

/// `a`/`an` = 1, but ONLY when a magnitude or dozen word follows — a bare
/// article is never a number. Returns the article's end offset.
pub(crate) fn article_one_at(b: &[u8], we: usize, i: usize) -> Option<usize> {
    let e = ascii_word_at(b, we, i, "a").or_else(|| ascii_word_at(b, we, i, "an"))?;
    let j = skip_ascii_ws(b, e, we);
    if j == e {
        return None;
    }
    if magnitude_or_dozen_at(b, we, j).is_some() {
        Some(e)
    } else {
        None
    }
}

/// A magnitude, dozen, or tiny-magnitude word at `i` (the article's legal
/// follower).
pub(crate) fn magnitude_or_dozen_at(b: &[u8], we: usize, i: usize) -> Option<usize> {
    MAGNITUDES
        .iter()
        .chain(DOZENS)
        .find_map(|(w, _)| ascii_word_at(b, we, i, w))
        .or_else(|| {
            SMALL_MAGNITUDES
                .iter()
                .find_map(|w| ascii_word_at(b, we, i, w))
        })
}

/// A compound tens + ordinal-one day ("twenty second" = 22).
pub(crate) fn word_tens_ordinal_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    for (w, v) in TENS {
        if let Some(te) = ascii_word_at(b, we, i, w) {
            let j = skip_ascii_ws(b, te, we);
            if j == te {
                return None;
            }
            if let Some((oe, ov)) = word_ordinal_day_at(b, we, j)
                && (1..=9).contains(&ov)
            {
                return Some((oe, *v as u32 + ov));
            }
            return None;
        }
    }
    None
}

/// The magnitude grammar (module doc). Returns the end offset and value.
pub(crate) fn word_number_at(b: &[u8], we: usize, i: usize) -> Option<(usize, f64)> {
    let mut saw_magnitude = false;
    word_number_inner(b, we, i, &mut saw_magnitude)
}

/// The scanner proper; reports whether any magnitude word was consumed
/// (the written-year thousand form's discriminator).
pub(crate) fn word_number_inner(
    b: &[u8],
    we: usize,
    i: usize,
    saw_magnitude: &mut bool,
) -> Option<(usize, f64)> {
    let mut total: f64 = 0.0;
    // The pending additive sum, folded into magnitudes as they arrive.
    let mut cur: f64 = 0.0;
    // An additive group is accepted only when `fresh` (right after a
    // magnitude) or strictly smaller than the previous additive group
    // (`one two` is not a number; `twenty and four` is 24).
    let mut last_add: f64 = 0.0;
    let mut fresh = true;
    let mut end = i;
    let mut probe = i;
    let mut any = false;
    loop {
        // Between groups: whitespace, with an optional "and". The probe
        // stays local — `end` only moves when a token actually consumes.
        if any {
            let k = skip_ascii_ws(b, end, we);
            if k == end {
                break;
            }
            let mut j = k;
            if let Some(ae) = ascii_word_at(b, we, k, "and") {
                let k2 = skip_ascii_ws(b, ae, we);
                if k2 == ae {
                    break; // a dangling "and" ends the number before itself
                }
                j = k2;
            }
            probe = j;
        }
        // Magnitude word: `hundred` multiplies the pending sum (it stays
        // pending — `hundred thousand` = 100 * 1000); `thousand` and above
        // fold the pending sum into the total.
        let mut consumed = false;
        for (w, f) in MAGNITUDES {
            if let Some(me) = ascii_word_at(b, we, probe, w) {
                if *f == 100 {
                    cur = cur.max(1.0) * 100.0;
                } else {
                    total += cur.max(1.0) * *f as f64;
                    cur = 0.0;
                }
                *saw_magnitude = true;
                fresh = true;
                end = me;
                any = true;
                consumed = true;
                break;
            }
        }
        if consumed {
            if end >= we {
                break;
            }
            continue;
        }
        // Additive group (strictly decreasing while on one scale).
        match word_group_at(b, we, probe) {
            Some((ge, v)) => {
                let v = v as f64;
                if !fresh && v >= last_add {
                    break;
                }
                cur += v;
                last_add = v;
                fresh = false;
                end = ge;
                any = true;
            }
            None => break,
        }
        if end >= we {
            break;
        }
    }
    if !any {
        return None;
    }
    Some((end, total + cur))
}

/// Written years 1000..=9999: the thousand form (a magnitude number in
/// range — `two thousand`, `two thousand and eighteen`, `nineteen hundred`)
/// or the digit-smash form (`nineteen seventy eight` = 1978,
/// `nineteen zero one` = 1901 — groups' decimal digits concatenated).
pub(crate) fn word_year_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    let mut saw_magnitude = false;
    if let Some((e, v)) = word_number_inner(b, we, i, &mut saw_magnitude)
        && saw_magnitude
        && (1000.0..=9999.0).contains(&v)
        && v.fract() == 0.0
    {
        return Some((e, v as i64));
    }
    // Smash form: first group 10-99, then more groups 0-99; the decimal
    // digits concatenate (19 | 0 | 1 -> 1901). Capped at 4 digits.
    let (mut end, g) = word_group_at(b, we, i)?;
    if !(10..=99).contains(&g) {
        return None;
    }
    let mut year: i64 = g;
    let mut digits = decimal_digits(g);
    loop {
        let j = skip_ascii_ws(b, end, we);
        if j == end {
            break;
        }
        let Some((ge, gv)) = word_group_at(b, we, j) else {
            break;
        };
        let d = decimal_digits(gv);
        if digits + d > 4 {
            break;
        }
        for _ in 0..d {
            year *= 10;
        }
        year += gv;
        digits += d;
        end = ge;
    }
    if digits == 4 && (1000..=9999).contains(&year) {
        Some((end, year))
    } else {
        None
    }
}

/// Decimal digit count of 0..=99.
fn decimal_digits(v: i64) -> u32 {
    if v >= 10 { 2 } else { 1 }
}

/// Ordinal day word (1..=31) — `first`..`thirty-first`.
pub(crate) fn word_ordinal_day_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    for (w, day) in ORDINAL_DAYS {
        if let Some(e) = ascii_word_at(b, we, i, w) {
            return Some((e, *day));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(s: &str) -> Option<(usize, f64)> {
        let b = s.as_bytes();
        word_number_at(b, b.len(), 0)
    }

    fn num_at(s: &str, at: usize) -> Option<(usize, f64)> {
        let b = s.as_bytes();
        word_number_at(b, b.len(), at)
    }

    #[test]
    fn groups_compose_with_hyphen_and_space() {
        let b = b"twenty-one hours";
        assert_eq!(word_group_at(b, b.len(), 0), Some((10, 21)));
        let b = b"forty five min";
        assert_eq!(word_group_at(b, b.len(), 0), Some((10, 45)));
        assert_eq!(word_group_at(b"seventeen", 9, 0), Some((9, 17)));
        assert_eq!(word_group_at(b"zero", 4, 0), Some((4, 0)));
        // Bare tens stand alone.
        assert_eq!(word_group_at(b"twenty min", 10, 0), Some((6, 20)));
    }

    #[test]
    fn magnitude_grammar() {
        assert_eq!(num("twenty"), Some((6, 20.0)));
        assert_eq!(num("sixteen"), Some((7, 16.0)));
        assert_eq!(num("twenty and four"), Some((15, 24.0)));
        assert_eq!(num("one hundred and three"), Some((21, 103.0)));
        assert_eq!(num("one hundred and sixteen"), Some((23, 116.0)));
        assert_eq!(num("one hundred and sixty-one"), Some((25, 161.0)));
        assert_eq!(num("hundred thousand"), Some((16, 100_000.0)));
        assert_eq!(num("two thousand three hundred"), Some((26, 2300.0)));
        assert_eq!(num("three million"), Some((13, 3_000_000.0)));
        assert_eq!(num("two billion"), Some((11, 2e9)));
    }

    #[test]
    fn additive_groups_must_decrease() {
        // "one two" is not one number — the scan yields just "one".
        assert_eq!(num("one two"), Some((3, 1.0)));
        // Resuming after the rejected group restarts fresh: "two" scans at
        // its own offset (its own candidate, not part of "one").
        assert_eq!(num_at("one two three", 4), Some((7, 2.0)));
    }

    #[test]
    fn article_one_needs_a_magnitude() {
        let b = b"a dozen";
        assert_eq!(article_one_at(b, b.len(), 0), Some(1));
        let b = b"a hundred trillionths";
        assert_eq!(article_one_at(b, b.len(), 0), Some(1));
        // A bare article before a non-magnitude is not a number seed.
        let b = b"a pen";
        assert_eq!(article_one_at(b, b.len(), 0), None);
    }

    #[test]
    fn written_years() {
        let year = |s: &str| {
            let b = s.as_bytes();
            word_year_at(b, b.len(), 0)
        };
        assert_eq!(year("two thousand"), Some((12, 2000)));
        assert_eq!(year("two thousand and eighteen"), Some((25, 2018)));
        assert_eq!(year("nineteen hundred"), Some((16, 1900)));
        assert_eq!(year("nineteen seventy eight"), Some((22, 1978)));
        assert_eq!(year("nineteen zero one"), Some((17, 1901)));
        // Not a year: a magnitude number outside 1000..=9999.
        assert_eq!(year("three million"), None);
        // The smash form needs a 10-99 lead and exactly 4 digits.
        assert_eq!(year("five"), None);
        assert_eq!(year("ten ten"), Some((7, 1010)));
    }

    #[test]
    fn ordinal_days() {
        let b = b"the tenth of September";
        assert_eq!(word_ordinal_day_at(b, b.len(), 4), Some((9, 10)));
        assert_eq!(word_ordinal_day_at(b"twenty-second", 13, 0), Some((13, 22)));
        assert_eq!(word_ordinal_day_at(b"thirtieth", 9, 0), Some((9, 30)));
    }
}
