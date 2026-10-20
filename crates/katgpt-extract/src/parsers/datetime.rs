//! `datetime` — the Date / Time / DateTime dimensions (Plan 628 T0.3).
//!
//! Calendar math is Howard Hinnant's `days_from_civil` / `civil_from_days`
//! (public-domain `chrono<->civil` algorithm), implemented locally — no
//! chrono, no libc, no clock reads (the determinism law). Grammar shapes
//! translated from Meta **Duckling** (Apache-2.0) and Microsoft
//! **Recognizers-Text** (MIT).
//!
//! ## Supported subset
//!
//! - ISO: `YYYY-MM-DD`; `HH:MM`, `HH:MM:SS`; datetime with `T` or a single
//!   space separator. 4-digit years 1000-9999 only.
//! - Relatives, anchored to `reference_date` (an ISO date string; a
//!   malformed anchor counts as absent): `today`, `tomorrow`, `yesterday`,
//!   `next|last <weekday>` (full names + `mon..sun` abbreviations),
//!   `in <n> days|weeks|months`, `<n> days|weeks|months ago`.
//!   NO anchor -> NO candidate — the parser never reads the clock.
//! - Month-day: `October 20`, `20 October`, abbreviated months (`Oct 20`,
//!   `20 Oct`). The year comes from the anchor; without one, NO candidate.
//! - Month-day-year: `October 20, 2026`, `20 Oct 2026`, `October 20 2026`
//!   — absolute, no anchor needed.
//! - Numeric: `10/20/2026` and `10/20` (anchored). Ambiguity rule
//!   (documented): the FIRST component > 12 makes it D/M/Y, otherwise the
//!   US M/D/Y reading wins the ambiguous forms. Separators `/ - .`
//!   (`21-04-2016`, `13.5.2015`), the year gap also `,` (`08/12,2015`),
//!   year-first (`2015/08/12`, `2019-sep-1`, `2019/sep/01`), and TWO-DIGIT
//!   years windowed < 50 -> 2000+, else 1900+ (`21/04/16`, `05/05/89`).
//! - Dotted/dashed month names: `16. Nov. 2016`, `05-Aug-2016`,
//!   `Dec. 31 , 1994`, attached `Dec31 , 1994`.
//! - Ordinal day suffixes: `January 12th`, `04th Jan 2019` (any of
//!   st/nd/rd/th, as Recognizers accepts `02nd`).
//! - Weekday-prefixed absolutes: `monday january 12th, 2016`,
//!   `friday 2018-7-6`.
//! - Word dates: `jan twenty two, 2018`, `jan twenty one two thousand and
//!   eighteen`, `September the twenty-first nineteen seventy eight`,
//!   `September 10, nineteen zero one`, `the tenth of September, two
//!   thousand`, `twenty second of June 2017` (word days + written years,
//!   issue 953 lever 10).
//! - Times: `5pm`, `5:30pm`, `5:30:15pm`, `9 am` (space optional),
//!   `17:30`, `17:30:45`, `05:30`, bare `H:MM` (24h), `noon`, `midnight`,
//!   and the WRITTEN clock `<hour words> [<minute words>] [meridiem]` —
//!   `eight forty` = 08:40, `ten forty five`, `fifteen fifteen p m`,
//!   `seven thirty pm`; the meridiem takes mixed dots/space (`p. m.`,
//!   `p . m .`). A bare written hour with neither minutes nor meridiem is
//!   not a time. `12am` = `00:00`, `12pm` = `12:00`. An hour alone (`5`)
//!   is not a time.
//! - Composed datetime (DateTime kind): date + (`T` | `"at"` | single
//!   space) + time, e.g. `tomorrow at 5pm`, `October 20 at 14:00`,
//!   `2026-10-21T17:00`. A bare date is NOT a DateTime candidate.
//!
//! ## Documented rules
//!
//! - Impossible calendar dates (`2026-02-30`, `2023-02-29`) produce NO
//!   candidate; month arithmetic CLAMPS to the target month's last day
//!   (`2026-01-31` + 1 month -> `2026-02-28`).
//! - Results must land in years 1..=9999, else NO candidate.
//! - Month-day-year with an internal comma is ambiguous inside LIST items
//!   (the list comma); use `20 October 2026` or ISO inside lists.

use super::words::{
    word_group_at, word_number_at, word_ordinal_day_at, word_tens_ordinal_at, word_year_at,
};
use super::{
    ParsedValue, RawCandidate, RawValue, ascii_word_at, finish, is_word_byte, is_ws_byte,
    next_boundary, prev_char_index, skip_ascii_ws,
};
use katgpt_core::extraction_wire::FieldKind;

/// A parsed calendar date (civil components, all validated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ref {
    pub y: i64,
    pub m: u32,
    pub d: u32,
}

/// Rank ladder (Date kind): absolute-with-year beats anchored-month-day
/// beats relative. Time and DateTime kinds use a single rank 0.
const RANK_ABSOLUTE: u8 = 0;
const RANK_ANCHORED: u8 = 1;
const RANK_RELATIVE: u8 = 2;

const MONTHS: &[(&str, u32)] = &[
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
    ("january", 1),
    ("february", 2),
    ("august", 8),
    ("april", 4),
    ("march", 3),
    ("july", 7),
    ("june", 6),
    ("jan", 1),
    ("feb", 2),
    ("mar", 3),
    ("apr", 4),
    ("aug", 8),
    ("sep", 9),
    ("oct", 10),
    ("nov", 11),
    ("dec", 12),
    ("may", 5),
];

/// `(spelling, weekday)` with 0 = Sunday; longest spellings first.
const WEEKDAYS: &[(&str, u32)] = &[
    ("wednesday", 3),
    ("thursday", 4),
    ("saturday", 6),
    ("tuesday", 2),
    ("sunday", 0),
    ("monday", 1),
    ("friday", 5),
    ("wed", 3),
    ("thu", 4),
    ("tue", 2),
    ("sun", 0),
    ("mon", 1),
    ("fri", 5),
    ("sat", 6),
];

/// `(spelling, unit)` with 0 = days, 1 = weeks, 2 = months.
const REL_UNITS: &[(&str, u32)] = &[
    ("months", 2),
    ("weeks", 1),
    ("month", 2),
    ("week", 1),
    ("days", 0),
    ("day", 0),
];

/// Parse every candidate of `kind` (Date | Time | DateTime) and append them
/// best-first. Other kinds append nothing.
pub fn parse_datetime_kind(
    state: &str,
    kind: FieldKind,
    reference_date: Option<&str>,
    out: &mut Vec<ParsedValue>,
) {
    if state.len() > u32::MAX as usize {
        return;
    }
    let reference = parse_reference(reference_date);
    let mut scratch = Vec::new();
    scan_datetimes(state, 0..state.len(), kind, reference, &mut scratch);
    finish(scratch, state, out);
}

/// Strict reference parser: a 10-byte ISO date that is also a REAL calendar
/// date. A malformed anchor counts as absent.
pub(crate) fn parse_reference(s: Option<&str>) -> Option<Ref> {
    let s = s?;
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let two = |r: &[u8]| -> Option<u32> {
        if r.len() == 2 && r.iter().all(u8::is_ascii_digit) {
            Some(u32::from(r[0] - b'0') * 10 + u32::from(r[1] - b'0'))
        } else {
            None
        }
    };
    if !b[0..4].iter().all(u8::is_ascii_digit) {
        return None;
    }
    let y = {
        let mut v: i64 = 0;
        for &c in &b[0..4] {
            v = v * 10 + i64::from(c - b'0');
        }
        v
    };
    let m = two(&b[5..7])?;
    let d = two(&b[8..10])?;
    if valid_ymd(y, m, d) {
        Some(Ref { y, m, d })
    } else {
        None
    }
}

/// The scanner. Zero-alloc: pushes [`RawCandidate`]s only.
pub(crate) fn scan_datetimes(
    state: &str,
    win: core::ops::Range<usize>,
    kind: FieldKind,
    reference: Option<Ref>,
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
        if datetime_start_ok(b, ws, i) {
            match kind {
                FieldKind::Date => scan_date_at(b, we, i, reference, sink),
                FieldKind::Time => {
                    if let Some(t) = time_at(b, we, i) {
                        push_time(sink, i, t);
                    }
                }
                FieldKind::DateTime => scan_composed_at(b, we, i, reference, sink),
                _ => {}
            }
        }
        i += 1;
    }
}

/// Candidate starts must not continue a word or a token (`12:30` must not
/// yield `30`, `10/20` must not yield `20`).
fn datetime_start_ok(b: &[u8], ws: usize, i: usize) -> bool {
    match prev_char_index(b, ws, i) {
        None => true,
        Some(j) => {
            let c = b[j];
            !(is_word_byte(c) || matches!(c, b':' | b'/'))
        }
    }
}

struct DateHit {
    end: usize,
    y: i64,
    m: u32,
    d: u32,
    absolute: bool,
}

struct TimeHit {
    end: usize,
    h: u32,
    mi: u32,
    s: Option<u32>,
}

fn scan_date_at(
    b: &[u8],
    we: usize,
    i: usize,
    reference: Option<Ref>,
    sink: &mut Vec<RawCandidate>,
) {
    date_forms_at(b, we, i, i, reference, sink);
    // Weekday-prefixed absolute dates ("monday january 12th, 2016",
    // "friday 2018-7-6"): the weekday names the day; the date follows.
    // The pushed span starts at the WEEKDAY (the full phrase).
    if let Some((dwe, _)) = weekday_at(b, we, i) {
        let mut j = skip_ascii_ws(b, dwe, we);
        if j < we && b[j] == b',' {
            let k = skip_ascii_ws(b, j + 1, we);
            if k > j + 1 {
                j = k;
            }
        }
        if j > dwe {
            date_forms_at(b, we, j, i, reference, sink);
        }
    }
}

/// Every absolute/anchored date form at `i`; candidates push with span
/// start `start` (usually `i`, the weekday start under a prefix).
fn date_forms_at(
    b: &[u8],
    we: usize,
    i: usize,
    start: usize,
    reference: Option<Ref>,
    sink: &mut Vec<RawCandidate>,
) {
    if let Some((e, y, m, d)) = iso_date_at(b, we, i) {
        push_date(sink, start, e, y, m, d, RANK_ABSOLUTE);
    }
    if let Some(h) = numeric_date_at(b, we, i, reference) {
        push_date(
            sink,
            start,
            h.end,
            h.y,
            h.m,
            h.d,
            if h.absolute {
                RANK_ABSOLUTE
            } else {
                RANK_ANCHORED
            },
        );
    }
    if let Some((e, y, m, d)) = year_month_name_day_at(b, we, i) {
        push_date(sink, start, e, y, m, d, RANK_ABSOLUTE);
    }
    if let Some((e, y, m, d)) = day_month_name_year_at(b, we, i) {
        push_date(sink, start, e, y, m, d, RANK_ABSOLUTE);
    }
    if let Some((abs, anch)) = month_day_at(b, we, i) {
        if let Some(a) = abs {
            push_date(sink, start, a.end, a.y, a.m, a.d, RANK_ABSOLUTE);
        }
        if start == i
            && let Some((ae, m, d)) = anch
            && let Some(r) = reference
            && let Some(y) = anchored_year(r.y, m, d)
        {
            push_date(sink, i, ae, y, m, d, RANK_ANCHORED);
        }
    }
    // `[the] <word-day> of <month>[, year]` — the span starts at the day
    // word ("the" stays outside, matching the upstream span).
    if let Some((ds, e, y, m, d, absolute)) = word_day_of_month_at(b, we, i) {
        let rank = if absolute {
            RANK_ABSOLUTE
        } else {
            RANK_ANCHORED
        };
        if absolute {
            push_date(sink, ds, e, y, m, d, rank);
        } else if start == i
            && let Some(r) = reference
            && let Some(y) = anchored_year(r.y, m, d)
        {
            push_date(sink, ds, e, y, m, d, rank);
        }
    }
    if start == i
        && let Some((e, y, m, d)) = relative_at(b, we, i, reference)
    {
        push_date(sink, i, e, y, m, d, RANK_RELATIVE);
    }
}

fn push_date(
    sink: &mut Vec<RawCandidate>,
    start: usize,
    end: usize,
    y: i64,
    m: u32,
    d: u32,
    rank: u8,
) {
    if (1..=9999).contains(&y) {
        sink.push(RawCandidate {
            start: start as u32,
            end: end as u32,
            rank,
            depth: 0,
            raw: RawValue::Date { y, m, d },
        });
    }
}

fn push_time(sink: &mut Vec<RawCandidate>, start: usize, t: TimeHit) {
    sink.push(RawCandidate {
        start: start as u32,
        end: t.end as u32,
        rank: RANK_ABSOLUTE,
        depth: 0,
        raw: RawValue::Time {
            h: t.h,
            mi: t.mi,
            s: t.s,
        },
    });
}

/// One date resolution for a composed datetime (prefers the absolute
/// with-year reading over the anchored one).
fn compose_date_at(b: &[u8], we: usize, i: usize, reference: Option<Ref>) -> Option<DateHit> {
    if let Some((e, y, m, d)) = iso_date_at(b, we, i) {
        return Some(DateHit {
            end: e,
            y,
            m,
            d,
            absolute: true,
        });
    }
    if let Some(h) = numeric_date_at(b, we, i, reference) {
        return Some(h);
    }
    if let Some((e, y, m, d)) = year_month_name_day_at(b, we, i) {
        return Some(DateHit {
            end: e,
            y,
            m,
            d,
            absolute: true,
        });
    }
    if let Some((e, y, m, d)) = day_month_name_year_at(b, we, i) {
        return Some(DateHit {
            end: e,
            y,
            m,
            d,
            absolute: true,
        });
    }
    if let Some((abs, anch)) = month_day_at(b, we, i) {
        if let Some(a) = abs {
            return Some(a);
        }
        if let Some((ae, m, d)) = anch
            && let Some(r) = reference
            && let Some(y) = anchored_year(r.y, m, d)
        {
            return Some(DateHit {
                end: ae,
                y,
                m,
                d,
                absolute: false,
            });
        }
        return None;
    }
    if let Some((ds, e, y, m, d, absolute)) = word_day_of_month_at(b, we, i) {
        let _ = ds;
        if absolute {
            return Some(DateHit {
                end: e,
                y,
                m,
                d,
                absolute: true,
            });
        }
        if let Some(r) = reference
            && let Some(y) = anchored_year(r.y, m, d)
        {
            return Some(DateHit {
                end: e,
                y,
                m,
                d,
                absolute: false,
            });
        }
        return None;
    }
    let (e, y, m, d) = relative_at(b, we, i, reference)?;
    Some(DateHit {
        end: e,
        y,
        m,
        d,
        absolute: false,
    })
}

fn scan_composed_at(
    b: &[u8],
    we: usize,
    i: usize,
    reference: Option<Ref>,
    sink: &mut Vec<RawCandidate>,
) {
    composed_date_first_at(b, we, i, reference, sink);
    composed_time_first_at(b, we, i, reference, sink);
}

fn composed_date_first_at(
    b: &[u8],
    we: usize,
    i: usize,
    reference: Option<Ref>,
    sink: &mut Vec<RawCandidate>,
) {
    let Some(hit) = compose_date_at(b, we, i, reference) else {
        return;
    };
    let time = if hit.end < we && b[hit.end] == b'T' {
        time_at(b, we, hit.end + 1)
    } else {
        // Separator: a directly-attached `,`/`-`/`@` ("04/21/2016,
        // 8:00pm", "5/3/18 @ 17:49"), whitespace, or whitespace +
        // (`,`|`-`|`@`|`at`). The attached check comes FIRST — a comma may
        // sit against the date with no space before it.
        let mut j = hit.end;
        if j < we && matches!(b[j], b',' | b'-' | b'@') {
            let k = skip_ascii_ws(b, j + 1, we);
            if k == j + 1 {
                return;
            }
            j = k;
        } else {
            j = skip_ascii_ws(b, j, we);
            if j == hit.end {
                return;
            }
            if j < we && matches!(b[j], b',' | b'-' | b'@') {
                let k = skip_ascii_ws(b, j + 1, we);
                if k == j + 1 {
                    return;
                }
                j = k;
            } else if let Some(ae) = ascii_word_at(b, we, j, "at") {
                let k = skip_ascii_ws(b, ae, we);
                if k == ae {
                    return;
                }
                j = k;
            }
        }
        time_at(b, we, j)
    };
    if let Some(t) = time {
        sink.push(RawCandidate {
            start: i as u32,
            end: t.end as u32,
            rank: RANK_ABSOLUTE,
            depth: 0,
            raw: RawValue::DateTime {
                y: hit.y,
                mo: hit.m,
                d: hit.d,
                h: t.h,
                mi: t.mi,
                s: t.s,
            },
        });
    }
}

/// Time-first composition: `19:00, 2016-12-22` (issue 953 lever 7).
fn composed_time_first_at(
    b: &[u8],
    we: usize,
    i: usize,
    reference: Option<Ref>,
    sink: &mut Vec<RawCandidate>,
) {
    let Some(t) = time_at(b, we, i) else {
        return;
    };
    // A separator is required: a directly-attached `,`/`-`/`@`
    // ("19:00, 2016-12-22"), whitespace, or whitespace + separator.
    let mut k = t.end;
    let mut had_sep = false;
    if k < we && matches!(b[k], b',' | b'-' | b'@') {
        let k2 = skip_ascii_ws(b, k + 1, we);
        if k2 > k + 1 {
            k = k2;
            had_sep = true;
        }
    }
    if !had_sep {
        k = skip_ascii_ws(b, t.end, we);
        if k > t.end && k < we && matches!(b[k], b',' | b'-' | b'@') {
            let k2 = skip_ascii_ws(b, k + 1, we);
            if k2 > k + 1 {
                k = k2;
            }
        }
        had_sep = k > t.end;
    }
    if !had_sep || k >= we {
        return;
    }
    let Some(hit) = compose_date_at(b, we, k, reference) else {
        return;
    };
    sink.push(RawCandidate {
        start: i as u32,
        end: hit.end as u32,
        rank: RANK_ABSOLUTE,
        depth: 0,
        raw: RawValue::DateTime {
            y: hit.y,
            mo: hit.m,
            d: hit.d,
            h: t.h,
            mi: t.mi,
            s: t.s,
        },
    });
}

// ---------------------------------------------------------------------------
// Calendar math — Howard Hinnant's civil algorithms (public domain),
// implemented locally. No external time sources anywhere in this module.
// ---------------------------------------------------------------------------

/// Days since 1970-01-01 for a civil date (valid for the whole proleptic
/// range; callers restrict output to years 1..=9999).
pub(crate) fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from(if m > 2 { m - 3 } else { m + 9 });
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Civil date from days since 1970-01-01.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub(crate) fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

pub(crate) fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => 0,
    }
}

pub(crate) fn valid_ymd(y: i64, m: u32, d: u32) -> bool {
    (1..=9999).contains(&y) && (1..=12).contains(&m) && (1..=days_in_month(y, m)).contains(&d)
}

/// Weekday with 0 = Sunday. 1970-01-01 was a Thursday.
fn weekday_from_days(days: i64) -> u32 {
    (days.rem_euclid(7) + 4) as u32 % 7
}

fn shift_days(r: Ref, delta: i64) -> Option<(i64, u32, u32)> {
    let (y, m, d) = civil_from_days(days_from_civil(r.y, r.m, r.d) + delta);
    if (1..=9999).contains(&y) {
        Some((y, m, d))
    } else {
        None
    }
}

/// Month arithmetic with end-of-month clamp: `2026-01-31` + 1 month ->
/// `2026-02-28` (a real calendar date is guaranteed).
fn shift_months(r: Ref, delta: i64) -> Option<(i64, u32, u32)> {
    let total = r.y * 12 + i64::from(r.m - 1) + delta;
    let ny = total.div_euclid(12);
    let nm = (total.rem_euclid(12) + 1) as u32;
    let nd = r.d.min(days_in_month(ny, nm));
    if (1..=9999).contains(&ny) {
        Some((ny, nm, nd))
    } else {
        None
    }
}

/// The next (strictly after) or previous (strictly before) occurrence of
/// `target` weekday relative to `r`.
fn shift_weekday(r: Ref, target: u32, forward: bool) -> Option<(i64, u32, u32)> {
    let cur = i64::from(weekday_from_days(days_from_civil(r.y, r.m, r.d)));
    let delta = if forward {
        (i64::from(target) - cur - 1).rem_euclid(7) + 1
    } else {
        -((cur - i64::from(target) - 1).rem_euclid(7) + 1)
    };
    shift_days(r, delta)
}

/// The anchor year for a month-day, if the day is real in that year.
fn anchored_year(y: i64, m: u32, d: u32) -> Option<i64> {
    if d <= days_in_month(y, m) {
        Some(y)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Date matchers
// ---------------------------------------------------------------------------

/// ISO `YYYY-MM-DD` (real calendar), with the ISO separator exception: the
/// end boundary may be `T` (a datetime in progress) so a Date-kind scan
/// still locates the date half of `2026-10-21T17:00`.
fn iso_date_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64, u32, u32)> {
    if i + 10 > we {
        return None;
    }
    let r = &b[i..i + 10];
    if !(r[0..4].iter().all(u8::is_ascii_digit)
        && r[4] == b'-'
        && r[7] == b'-'
        && r[5..7].iter().all(u8::is_ascii_digit)
        && r[8..10].iter().all(u8::is_ascii_digit))
    {
        return None;
    }
    let num = |r: &[u8]| -> i64 {
        let mut v: i64 = 0;
        for &c in r {
            v = v * 10 + i64::from(c - b'0');
        }
        v
    };
    let y = num(&r[0..4]);
    let m = num(&r[5..7]) as u32;
    let d = num(&r[8..10]) as u32;
    if !valid_ymd(y, m, d) {
        return None;
    }
    let e = i + 10;
    if e < we {
        let c = b[e];
        let ok_end = c == b'T' || !(c.is_ascii_alphanumeric() || matches!(c, b'-' | b'/' | b':'));
        if !ok_end {
            return None;
        }
    }
    Some((e, y, m, d))
}

fn one_two_digits(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
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

/// Exactly four ASCII digits; the value must be a 4-digit year (1000..=9999
/// after the 1..=9999 sanity pass — `0000` is refused).
fn year4_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    if i + 4 > we || !b[i..i + 4].iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut v: i64 = 0;
    for &c in &b[i..i + 4] {
        v = v * 10 + i64::from(c - b'0');
    }
    if v < 1 {
        return None;
    }
    Some((i + 4, v))
}

fn day_digits_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    let (mut e, v) = one_two_digits(b, we, i)?;
    if !(1..=31).contains(&v) {
        return None;
    }
    // Optional ordinal suffix — any of st/nd/rd/th (Recognizers accepts
    // "02nd"), and only when it ends the token ("20thx" is a word).
    if e + 2 <= we {
        for suf in [b"st", b"nd", b"rd", b"th"] {
            if b[e..e + 2].eq_ignore_ascii_case(suf) {
                let ne = e + 2;
                if ne >= we || !is_word_byte(b[ne]) {
                    e = ne;
                }
                break;
            }
        }
    }
    Some((e, v))
}

/// A month spelling immediately followed by day digits ("Dec31 , 1994").
fn month_prefix_digits_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    for (spelling, month) in MONTHS {
        let w = spelling.as_bytes();
        if i + w.len() > we || !b[i..i + w.len()].eq_ignore_ascii_case(w) {
            continue;
        }
        let e = i + w.len();
        if e < we && b[e].is_ascii_digit() {
            return Some((e, *month));
        }
    }
    None
}

/// A day of month as words: an ordinal word (`tenth`, `twenty-first`), a
/// compound tens + ordinal one (`twenty second` = 22), or a 1-31 cardinal
/// group (`twenty two`).
fn word_day_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    if let Some(hit) = word_ordinal_day_at(b, we, i) {
        return Some(hit);
    }
    if let Some(hit) = word_tens_ordinal_at(b, we, i) {
        return Some(hit);
    }
    let (e, g) = word_group_at(b, we, i)?;
    if (1..=31).contains(&g) {
        Some((e, g as u32))
    } else {
        None
    }
}

/// Two-digit year window (Recognizers rule): < 50 -> 2000+, else 1900+.
fn window_year(y2: u32) -> i64 {
    if y2 < 50 {
        2000 + i64::from(y2)
    } else {
        1900 + i64::from(y2)
    }
}

/// Exactly two ASCII digits (the 2-digit-year tail).
fn year2_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    if i + 2 > we || !b[i..i + 2].iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some((
        i + 2,
        i64::from(b[i] - b'0') * 10 + i64::from(b[i + 1] - b'0'),
    ))
}

/// Numeric `a SEP b [SEP year]`. Separators between components: `/ - .`;
/// before the year also `,` (`08/12,2015`). The year is 4 digits or a
/// TWO-digit window (< 50 -> 2000+, else 1900+). Ambiguity rule:
/// first component > 12 makes it D/M/Y, else M/D/Y (documented).
fn numeric_date_at(b: &[u8], we: usize, i: usize, reference: Option<Ref>) -> Option<DateHit> {
    if let Some(hit) = year_first_numeric_at(b, we, i) {
        return Some(hit);
    }
    let (ae, a) = one_two_digits(b, we, i)?;
    if ae >= we || !matches!(b[ae], b'/' | b'-' | b'.') {
        return None;
    }
    let (ce, c) = one_two_digits(b, we, ae + 1)?;
    let mut e = ce;
    let mut year: Option<i64> = None;
    if e < we && matches!(b[e], b'/' | b'-' | b'.' | b',') {
        if let Some((ye, y)) = year4_at(b, we, e + 1) {
            year = Some(y);
            e = ye;
        } else if let Some((ye, y)) = year2_at(b, we, e + 1) {
            year = Some(window_year(y as u32));
            e = ye;
        }
    }
    if e < we && (is_word_byte(b[e]) || b[e] == b'/') {
        return None;
    }
    let (m, d) = if a > 12 { (c, a) } else { (a, c) };
    if m == 0 || m > 12 || d == 0 {
        return None;
    }
    let y = match year {
        Some(y) => y,
        None => reference?.y,
    };
    if d > days_in_month(y, m) {
        return None;
    }
    if !(1..=9999).contains(&y) {
        return None;
    }
    Some(DateHit {
        end: e,
        y,
        m,
        d,
        absolute: year.is_some(),
    })
}

/// Numeric year-first `y4 SEP a SEP b` (`2015/08/12`, `2019-7-6`,
/// `2015.5.13`). The y/m/d reading is unambiguous.
fn year_first_numeric_at(b: &[u8], we: usize, i: usize) -> Option<DateHit> {
    let (ye, y) = year4_at(b, we, i)?;
    if ye >= we || !matches!(b[ye], b'/' | b'-' | b'.') {
        return None;
    }
    let (ae, a) = one_two_digits(b, we, ye + 1)?;
    if ae >= we || !matches!(b[ae], b'/' | b'-' | b'.') {
        return None;
    }
    let (ce, c) = one_two_digits(b, we, ae + 1)?;
    let e = ce;
    if e < we && (is_word_byte(b[e]) || matches!(b[e], b'/' | b'-' | b'.')) {
        return None;
    }
    if a == 0 || a > 12 || c == 0 {
        return None;
    }
    if !(1..=9999).contains(&y) || c > days_in_month(y, a) {
        return None;
    }
    Some(DateHit {
        end: e,
        y,
        m: a,
        d: c,
        absolute: true,
    })
}

/// Year-first with a month NAME: `2019-sep-1`, `2019/sep/01`.
fn year_month_name_day_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64, u32, u32)> {
    let (ye, y) = year4_at(b, we, i)?;
    if ye >= we || !matches!(b[ye], b'/' | b'-' | b'.') {
        return None;
    }
    let (me, m) = month_at(b, we, ye + 1)?;
    if me >= we || !matches!(b[me], b'/' | b'-' | b'.') {
        return None;
    }
    let (de, d) = one_two_digits(b, we, me + 1)?;
    let e = de;
    if e < we && is_word_byte(b[e]) {
        return None;
    }
    if !valid_ymd(y, m, d) {
        return None;
    }
    Some((e, y, m, d))
}

/// `d SEP Month[.] SEP? year4` — `05-Aug-2016`, `16. Nov. 2016` (the
/// European dotted form; the month may carry a trailing dot).
fn day_month_name_year_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64, u32, u32)> {
    let (de, d) = day_digits_at(b, we, i)?;
    let mut j = de;
    let mut had_sep = false;
    if j < we && matches!(b[j], b'-' | b'/' | b'.') {
        had_sep = true;
        j += 1;
    }
    let k = skip_ascii_ws(b, j, we);
    if !had_sep {
        return None; // the ws-separated day-month form is month_day_at's
    }
    let (me, m) = month_at(b, we, k)?;
    let mut e2 = me;
    if e2 < we && b[e2] == b'.' {
        e2 += 1;
    }
    // Year: ws-separated or `-`-joined.
    let mut j2 = e2;
    let mut had_sep2 = false;
    if j2 < we && b[j2] == b'-' {
        had_sep2 = true;
        j2 += 1;
    }
    let k2 = skip_ascii_ws(b, j2, we);
    if !had_sep2 && k2 == j2 {
        return None;
    }
    let (ye, y) = year4_at(b, we, k2)?;
    let e = ye;
    if e < we && is_word_byte(b[e]) {
        return None;
    }
    if !valid_ymd(y, m, d) {
        return None;
    }
    Some((e, y, m, d))
}

fn month_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    for (spelling, month) in MONTHS {
        if let Some(e) = ascii_word_at(b, we, i, spelling) {
            return Some((e, *month));
        }
    }
    None
}

fn weekday_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    for (spelling, dow) in WEEKDAYS {
        if let Some(e) = ascii_word_at(b, we, i, spelling) {
            return Some((e, *dow));
        }
    }
    None
}

/// The month-day matcher's output: the absolute hit (explicit year) and/or
/// the anchored month-day span (end, month, day).
type MonthDayHits = (Option<DateHit>, Option<(usize, u32, u32)>);

/// `October 20[, 2026]` / `20 October[, 2026]` (and abbreviated months).
/// The month may carry a dot (`Dec. 31`), the day may be attached
/// (`Dec31`) or a word (`jan twenty two`), and an optional `the` may sit
/// between month and day (`September the twenty-first`).
fn month_day_at(b: &[u8], we: usize, i: usize) -> Option<MonthDayHits> {
    let month_hit = month_at(b, we, i).or_else(|| month_prefix_digits_at(b, we, i));
    if let Some((me, month)) = month_hit {
        // Optional dot after the month abbreviation.
        let mut e2 = me;
        if e2 < we && b[e2] == b'.' {
            e2 += 1;
        }
        let mut j = skip_ascii_ws(b, e2, we);
        if j > e2 {
            // Optional "the" between month and day.
            if let Some(te) = ascii_word_at(b, we, j, "the") {
                let jt = skip_ascii_ws(b, te, we);
                if jt > te {
                    j = jt;
                }
            }
            if let Some((de, day)) = day_digits_at(b, we, j).or_else(|| word_day_at(b, we, j)) {
                return finish_month_day(b, we, de, month, day);
            }
            // Attached day digits ("Dec31") sit directly after the month.
            if let Some((de, day)) = day_digits_at(b, we, e2) {
                return finish_month_day(b, we, de, month, day);
            }
        } else if let Some((de, day)) = day_digits_at(b, we, e2) {
            // Attached day with no separator at all ("Dec31").
            return finish_month_day(b, we, de, month, day);
        }
        return None;
    }
    if let Some((de, day)) = day_digits_at(b, we, i) {
        let j = skip_ascii_ws(b, de, we);
        if j > de
            && let Some((me, month)) = month_at(b, we, j)
        {
            return finish_month_day(b, we, me, month, day);
        }
    }
    None
}

/// Shared tail of both month-day forms: an optional year (digits or
/// written — `two thousand`, `nineteen zero one`), then boundary checks
/// and calendar validation. `after` is the offset just past the
/// day/month token.
fn finish_month_day(
    b: &[u8],
    we: usize,
    after: usize,
    month: u32,
    day: u32,
) -> Option<MonthDayHits> {
    let mut e = after;
    let mut year: Option<i64> = None;
    let mut j = skip_ascii_ws(b, after, we);
    if j < we && b[j] == b',' {
        j = skip_ascii_ws(b, j + 1, we);
        if let Some((ye, y)) = year_or_word_year(b, we, j) {
            year = Some(y);
            e = ye;
        }
    } else if j > after
        && let Some((ye, y)) = year_or_word_year(b, we, j)
    {
        year = Some(y);
        e = ye;
    }
    if e < we && is_word_byte(b[e]) {
        return None;
    }
    let abs = year.and_then(|y| {
        if day <= days_in_month(y, month) {
            Some(DateHit {
                end: e,
                y,
                m: month,
                d: day,
                absolute: true,
            })
        } else {
            None
        }
    });
    Some((abs, Some((after, month, day))))
}

/// A 4-digit year or a written year (`two thousand and eighteen`).
fn year_or_word_year(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    if let Some(hit) = year4_at(b, we, i) {
        return Some(hit);
    }
    word_year_at(b, we, i)
}

/// `[the] <word-day> of <month>[, year]` — `the tenth of September, two
/// thousand`, `twenty second of June 2017`. The span starts at the day
/// word (a leading "the" stays outside it, matching the upstream span).
#[allow(clippy::type_complexity)]
fn word_day_of_month_at(
    b: &[u8],
    we: usize,
    i: usize,
) -> Option<(usize, usize, i64, u32, u32, bool)> {
    let mut j = i;
    if let Some(te) = ascii_word_at(b, we, j, "the") {
        let jt = skip_ascii_ws(b, te, we);
        if jt > te {
            j = jt;
        }
    }
    let (de, day) = word_day_at(b, we, j)?;
    let k = skip_ascii_ws(b, de, we);
    if k == de {
        return None;
    }
    let oe = ascii_word_at(b, we, k, "of")?;
    let k2 = skip_ascii_ws(b, oe, we);
    if k2 == oe {
        return None;
    }
    let (me, month) = month_at(b, we, k2)?;
    let (abs, anch) = finish_month_day(b, we, me, month, day)?;
    if let Some(a) = abs {
        return Some((de, a.end, a.y, month, day, true));
    }
    let (ae, _, _) = anch?;
    Some((de, ae, 0, month, day, false))
}

/// Relatives: `today|tomorrow|yesterday`, `next|last <weekday>`,
/// `in <n> days|weeks|months`, `<n> days|weeks|months ago`. All need the
/// reference anchor — without one there is NO candidate.
fn relative_at(
    b: &[u8],
    we: usize,
    i: usize,
    reference: Option<Ref>,
) -> Option<(usize, i64, u32, u32)> {
    let r = reference?;
    for (word, delta) in [("today", 0i64), ("tomorrow", 1), ("yesterday", -1)] {
        if let Some(e) = ascii_word_at(b, we, i, word) {
            return shift_days(r, delta).map(|(y, m, d)| (e, y, m, d));
        }
    }
    for (word, forward) in [("next", true), ("last", false)] {
        if let Some(e) = ascii_word_at(b, we, i, word) {
            let j = skip_ascii_ws(b, e, we);
            if j > e
                && let Some((dwe, dow)) = weekday_at(b, we, j)
            {
                return shift_weekday(r, dow, forward).map(|(y, m, d)| (dwe, y, m, d));
            }
        }
    }
    if let Some(e) = ascii_word_at(b, we, i, "in") {
        let j = skip_ascii_ws(b, e, we);
        if j > e
            && let Some((ne, n)) = short_number_at(b, we, j)
        {
            let k = skip_ascii_ws(b, ne, we);
            if k > ne
                && let Some((ue, unit)) = rel_unit_at(b, we, k)
            {
                let hit = match unit {
                    0 => shift_days(r, n),
                    1 => shift_days(r, n * 7),
                    _ => shift_months(r, n),
                };
                return hit.map(|(y, m, d)| (ue, y, m, d));
            }
        }
    }
    if let Some((ne, n)) = short_number_at(b, we, i) {
        let j = skip_ascii_ws(b, ne, we);
        if j > ne
            && let Some((ue, unit)) = rel_unit_at(b, we, j)
        {
            let k = skip_ascii_ws(b, ue, we);
            if k > ue
                && let Some(ae) = ascii_word_at(b, we, k, "ago")
            {
                let hit = match unit {
                    0 => shift_days(r, -n),
                    1 => shift_days(r, -n * 7),
                    _ => shift_months(r, -n),
                };
                return hit.map(|(y, m, d)| (ae, y, m, d));
            }
        }
    }
    None
}

fn rel_unit_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    for (spelling, unit) in REL_UNITS {
        if let Some(e) = ascii_word_at(b, we, i, spelling) {
            return Some((e, *unit));
        }
    }
    None
}

/// 1-4 digit unsigned integer (relative offsets), value <= 9999; word
/// numbers compose too (`in two months`, issue 953 lever 10).
fn short_number_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    if i < we && b[i].is_ascii_digit() {
        let mut v: i64 = 0;
        let mut e = i;
        while e < we && b[e].is_ascii_digit() && e - i < 4 {
            v = v * 10 + i64::from(b[e] - b'0');
            e += 1;
        }
        return Some((e, v));
    }
    let (e, v) = word_number_at(b, we, i)?;
    if (0.0..=9999.0).contains(&v) && v.fract() == 0.0 {
        Some((e, v as i64))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Time matchers
// ---------------------------------------------------------------------------

/// `noon` | `midnight` | `<h>[:mm[:ss]]( )?[am|pm]` | `H:MM[:SS]` (24h) |
/// the written clock (`eight forty`, `seven thirty pm`).
fn time_at(b: &[u8], we: usize, i: usize) -> Option<TimeHit> {
    if let Some(e) = ascii_word_at(b, we, i, "noon") {
        return Some(TimeHit {
            end: e,
            h: 12,
            mi: 0,
            s: None,
        });
    }
    if let Some(e) = ascii_word_at(b, we, i, "midnight") {
        return Some(TimeHit {
            end: e,
            h: 0,
            mi: 0,
            s: None,
        });
    }
    if let Some(t) = meridiem_time_at(b, we, i) {
        return Some(t);
    }
    if let Some(t) = clock24_at(b, we, i) {
        return Some(t);
    }
    if let Some(t) = daypart_time_at(b, we, i) {
        return Some(t);
    }
    written_time_at(b, we, i)
}

/// Day-part modifiers — the meridiem's prose forms (issue 953 lever K).
/// Longest first; `ascii_word_at`'s boundary keeps `afternoon` exact.
const DAYPARTS: &[(&str, bool)] = &[
    ("in the afternoon", true),
    ("in the evening", true),
    ("in the morning", false),
    ("in the night", true),
    ("afternoon", true),
    ("at night", true),
];

fn daypart_at(b: &[u8], we: usize, i: usize) -> Option<(usize, bool)> {
    for (w, pm) in DAYPARTS {
        if let Some(e) = ascii_word_at(b, we, i, w) {
            return Some((e, *pm));
        }
    }
    None
}

/// The meridiem transform shared by both spellings (`7 in the night` ->
/// 19, `12 in the morning` -> 0).
fn apply_daypart(h: u32, pm: bool) -> u32 {
    if pm {
        if h < 12 { h + 12 } else { h }
    } else if h == 12 {
        0
    } else {
        h
    }
}

/// Day-part times, both orders: `five thirty in the night` (suffix) and
/// `afternoon 7:00` / `in the morning at 7` (prefix — a bare hour is the
/// daypart's meridiem substitute).
fn daypart_time_at(b: &[u8], we: usize, i: usize) -> Option<TimeHit> {
    // Suffix: <any time shape> + daypart.
    if let Some((te, h, mi, s)) = hourish_at(b, we, i) {
        let j = skip_ascii_ws(b, te, we);
        if j > te
            && let Some((de, pm)) = daypart_at(b, we, j)
        {
            return Some(TimeHit {
                end: de,
                h: apply_daypart(h, pm),
                mi,
                s,
            });
        }
    }
    // Prefix: daypart [+ at] + time (a bare hour is legal here).
    let (de, pm) = daypart_at(b, we, i)?;
    let mut j = skip_ascii_ws(b, de, we);
    if j == de {
        return None;
    }
    if let Some(ae) = ascii_word_at(b, we, j, "at") {
        let k = skip_ascii_ws(b, ae, we);
        if k > ae {
            j = k;
        }
    }
    let (te, h, mi, s) = hourish_at(b, we, j)?;
    Some(TimeHit {
        end: te,
        h: apply_daypart(h, pm),
        mi,
        s,
    })
}

/// The hour-ish shapes a daypart can attach to: digit `h[:mm[:ss]]` or a
/// written hour [+ written minutes]. The meridiem is NOT consumed here.
/// A bare hour is legal — the caller guarantees a daypart follows.
fn hourish_at(b: &[u8], we: usize, i: usize) -> Option<(usize, u32, u32, Option<u32>)> {
    if i < we && b[i].is_ascii_digit() {
        let (mut e, h) = one_two_digits(b, we, i)?;
        if h <= 23 {
            let mut mi = 0u32;
            let mut s = None;
            if e + 1 < we
                && b[e] == b':'
                && let Some((me, v)) = two_digits(b, we, e + 1)
            {
                if v > 59 {
                    return None;
                }
                mi = v;
                e = me;
                if e + 1 < we
                    && b[e] == b':'
                    && let Some((se, v)) = two_digits(b, we, e + 1)
                {
                    if v > 59 {
                        return None;
                    }
                    s = Some(v);
                    e = se;
                }
            }
            return Some((e, h, mi, s));
        }
        return None;
    }
    let (he, h) = word_group_at(b, we, i)?;
    if !(0..=23).contains(&h) {
        return None;
    }
    let j = skip_ascii_ws(b, he, we);
    if j > he
        && let Some((me, m)) = word_group_at(b, we, j)
        && m <= 59
    {
        return Some((me, h as u32, m as u32, None));
    }
    Some((he, h as u32, 0, None))
}

/// Written clock: `<hour words> [<minute words>] [loose meridiem]` —
/// `eight forty` = 08:40, `ten forty five` = 10:45, `fifteen fifteen p m`
/// = 15:15, `seven thirty pm` = 19:30. A bare written hour with neither
/// minutes nor a meridiem is not a time (the digit rule's written twin).
fn written_time_at(b: &[u8], we: usize, i: usize) -> Option<TimeHit> {
    let (he, h) = word_group_at(b, we, i)?;
    if !(0..=23).contains(&h) {
        return None;
    }
    let mut h = h as u32;
    let mut e = he;
    let mut mi = 0u32;
    let mut has_meridiem = false;
    let j = skip_ascii_ws(b, he, we);
    if j > he {
        if let Some((me, is_pm)) = meridiem_loose_at(b, we, j) {
            has_meridiem = true;
            e = me;
            h = if is_pm {
                if h < 12 { h + 12 } else { h }
            } else if h == 12 {
                0
            } else {
                h
            };
        } else if let Some((me2, m)) = word_group_at(b, we, j)
            && m <= 59
        {
            mi = m as u32;
            e = me2;
            // A meridiem may still follow the minutes.
            let k = skip_ascii_ws(b, me2, we);
            if k > me2
                && let Some((me, is_pm)) = meridiem_loose_at(b, we, k)
            {
                has_meridiem = true;
                e = me;
                h = if is_pm {
                    if h < 12 { h + 12 } else { h }
                } else if h == 12 {
                    0
                } else {
                    h
                };
            }
        }
    }
    if mi == 0 && !has_meridiem {
        return None;
    }
    Some(TimeHit {
        end: e,
        h,
        mi,
        s: None,
    })
}

/// Loose meridiem: `am|pm` with any mix of dots and whitespace between the
/// letter and the `m` (`p. m.`, `p . m .`), optional trailing dot.
fn meridiem_loose_at(b: &[u8], we: usize, i: usize) -> Option<(usize, bool)> {
    if i >= we {
        return None;
    }
    let c = b[i] | 0x20;
    if c != b'a' && c != b'p' {
        return None;
    }
    let mut j = i + 1;
    while j < we && (is_ws_byte(b[j]) || b[j] == b'.') {
        j += 1;
    }
    if j >= we || (b[j] | 0x20) != b'm' {
        return None;
    }
    j += 1;
    if j < we && b[j] == b'.' {
        j += 1;
    }
    if j < we && is_word_byte(b[j]) {
        return None;
    }
    Some((j, c == b'p'))
}

fn two_digits(b: &[u8], we: usize, i: usize) -> Option<(usize, u32)> {
    if i + 2 > we || !b[i..i + 2].iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some((
        i + 2,
        u32::from(b[i] - b'0') * 10 + u32::from(b[i + 1] - b'0'),
    ))
}

/// 1..=12 hour with an optional `:mm[:ss]` tail, attached minutes before
/// a meridiem (`340pm` = 3:40pm, `1140 a.m.` = 11:40), optional space,
/// then the meridiem (loose: dots/spaces allowed). `12am` -> 00, `12pm` ->
/// 12. A daypart may substitute for the meridiem (`5 in the morning`).
fn meridiem_time_at(b: &[u8], we: usize, i: usize) -> Option<TimeHit> {
    let (mut e, mut h) = one_two_digits(b, we, i)?;
    let mut attached_mi = false;
    if !(1..=12).contains(&h) {
        // "340pm": the greedy 2-digit hour is 34 — retry as a 1-digit hour
        // plus attached minutes (which the shared block below consumes).
        h = u32::from(b[i] - b'0');
        if !(1..=12).contains(&h) || e != i + 2 {
            return None;
        }
        e = i + 1;
        // The two attached digits must be there and small enough.
        let (me, mi2) = two_digits(b, we, e)?;
        if mi2 > 59 {
            return None;
        }
        attached_mi = true;
        e = me;
    }
    let mut mi = 0u32;
    let mut s: Option<u32> = None;
    if attached_mi {
        mi = u32::from(b[e - 2] - b'0') * 10 + u32::from(b[e - 1] - b'0');
    } else if e + 1 < we
        && b[e] == b':'
        && let Some((me, v)) = two_digits(b, we, e + 1)
    {
        if v > 59 {
            return None;
        }
        mi = v;
        e = me;
        if e + 1 < we
            && b[e] == b':'
            && let Some((se, v)) = two_digits(b, we, e + 1)
        {
            if v > 59 {
                return None;
            }
            s = Some(v);
            e = se;
        }
    } else if e + 2 <= we && b[e].is_ascii_digit() && b[e + 1].is_ascii_digit() {
        // Attached minutes before a meridiem: "1140 a.m." (11:40). The
        // trailing meridiem requirement below gates false positives.
        let v = u32::from(b[e] - b'0') * 10 + u32::from(b[e + 1] - b'0');
        if v <= 59 {
            mi = v;
            e += 2;
        }
    }
    let j = skip_ascii_ws(b, e, we);
    if j >= we {
        return None;
    }
    let c = b[j] | 0x20;
    if c != b'a' && c != b'p' {
        // Day-part meridiem substitute ("5 in the morning").
        if let Some((de, pm)) = daypart_at(b, we, j) {
            return Some(TimeHit {
                end: de,
                h: apply_daypart(h, pm),
                mi,
                s,
            });
        }
        return None;
    }
    let (m_end, is_pm) = meridiem_loose_at(b, we, j)?;
    let h = apply_daypart(h, is_pm);
    Some(TimeHit {
        end: m_end,
        h,
        mi,
        s,
    })
}

/// 0..=23 hour, `:mm` required, optional `:ss`, hard boundary after.
fn clock24_at(b: &[u8], we: usize, i: usize) -> Option<TimeHit> {
    let (mut e, h) = one_two_digits(b, we, i)?;
    if h > 23 || e >= we || b[e] != b':' {
        return None;
    }
    let (me, mi) = two_digits(b, we, e + 1)?;
    if mi > 59 {
        return None;
    }
    e = me;
    let mut s: Option<u32> = None;
    if e + 1 < we
        && b[e] == b':'
        && let Some((se, v)) = two_digits(b, we, e + 1)
    {
        if v > 59 {
            return None;
        }
        s = Some(v);
        e = se;
    }
    if e < we && (is_word_byte(b[e]) || b[e] == b':') {
        return None;
    }
    Some(TimeHit { end: e, h, mi, s })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::span_text;
    use katgpt_core::extraction_wire::FieldValue;

    const REF: &str = "2026-10-20";

    fn dates(state: &str, kind: FieldKind, reference: Option<&str>) -> Vec<(String, u8, String)> {
        let mut out = Vec::new();
        parse_datetime_kind(state, kind, reference, &mut out);
        out.into_iter()
            .map(|pv| {
                (
                    span_text(state, &pv).to_string(),
                    pv.rank,
                    match pv.value {
                        FieldValue::Date(s) | FieldValue::Time(s) | FieldValue::DateTime(s) => s,
                        ref other => panic!("unexpected {other:?}"),
                    },
                )
            })
            .collect()
    }

    fn one(state: &str, kind: FieldKind, reference: Option<&str>) -> String {
        let v = dates(state, kind, reference);
        assert_eq!(v.len(), 1, "expected one candidate in {state:?}, got {v:?}");
        v[0].2.clone()
    }

    fn none(state: &str, kind: FieldKind, reference: Option<&str>) {
        let v = dates(state, kind, reference);
        assert!(
            v.is_empty(),
            "expected no candidate in {state:?}, got {v:?}"
        );
    }

    #[test]
    fn iso_dates_validate_real_calendar() {
        assert_eq!(one("2026-10-20", FieldKind::Date, None), "2026-10-20");
        assert_eq!(one("due 2024-02-29.", FieldKind::Date, None), "2024-02-29"); // leap
        // The date half of an ISO datetime is a Date candidate.
        assert_eq!(one("2026-10-21T17:00", FieldKind::Date, None), "2026-10-21");
    }

    #[test]
    fn impossible_dates_produce_no_candidate() {
        none("2026-02-30", FieldKind::Date, None);
        none("2023-02-29", FieldKind::Date, None); // not a leap year
        none("2026-13-01", FieldKind::Date, None);
        none("2026-00-10", FieldKind::Date, None);
    }

    #[test]
    fn iso_times() {
        assert_eq!(one("17:30", FieldKind::Time, None), "17:30");
        assert_eq!(one("17:30:45", FieldKind::Time, None), "17:30:45");
        assert_eq!(one("05:30", FieldKind::Time, None), "05:30");
        none("24:00", FieldKind::Time, None);
        none("17:60", FieldKind::Time, None);
        none("5", FieldKind::Time, None); // an hour alone is not a time
    }

    #[test]
    fn iso_datetimes_t_and_space() {
        assert_eq!(
            one("2026-10-21T17:00", FieldKind::DateTime, None),
            "2026-10-21T17:00"
        );
        assert_eq!(
            one("2026-10-21 17:00", FieldKind::DateTime, None),
            "2026-10-21T17:00"
        );
        assert_eq!(
            one("2026-10-21T09:30:15", FieldKind::DateTime, None),
            "2026-10-21T09:30:15"
        );
        none("2026-10-21T25:00", FieldKind::DateTime, None);
    }

    #[test]
    fn relative_days_anchored_to_reference() {
        assert_eq!(one("today", FieldKind::Date, Some(REF)), "2026-10-20");
        assert_eq!(one("tomorrow", FieldKind::Date, Some(REF)), "2026-10-21");
        assert_eq!(one("yesterday", FieldKind::Date, Some(REF)), "2026-10-19");
        assert_eq!(one("in 3 days", FieldKind::Date, Some(REF)), "2026-10-23");
        assert_eq!(one("3 days ago", FieldKind::Date, Some(REF)), "2026-10-17");
        assert_eq!(one("in 2 weeks", FieldKind::Date, Some(REF)), "2026-11-03");
        assert_eq!(one("2 weeks ago", FieldKind::Date, Some(REF)), "2026-10-06");
    }

    #[test]
    fn weekday_next_last() {
        // 2026-10-20 is a Tuesday.
        assert_eq!(one("next monday", FieldKind::Date, Some(REF)), "2026-10-26");
        assert_eq!(one("next friday", FieldKind::Date, Some(REF)), "2026-10-23");
        assert_eq!(one("last monday", FieldKind::Date, Some(REF)), "2026-10-19");
        // "next tuesday" is strictly after: one week out.
        assert_eq!(
            one("next tuesday", FieldKind::Date, Some(REF)),
            "2026-10-27"
        );
        assert_eq!(one("next mon", FieldKind::Date, Some(REF)), "2026-10-26");
    }

    #[test]
    fn month_clamp_on_month_arithmetic() {
        assert_eq!(one("in 1 month", FieldKind::Date, Some(REF)), "2026-11-20");
        assert_eq!(
            one("in 1 month", FieldKind::Date, Some("2026-01-31")),
            "2026-02-28"
        );
        assert_eq!(
            one("2 months ago", FieldKind::Date, Some("2026-03-31")),
            "2026-01-31"
        );
    }

    #[test]
    fn month_day_forms_anchored() {
        assert_eq!(one("October 20", FieldKind::Date, Some(REF)), "2026-10-20");
        assert_eq!(one("20 October", FieldKind::Date, Some(REF)), "2026-10-20");
        assert_eq!(one("Oct 20", FieldKind::Date, Some(REF)), "2026-10-20");
        assert_eq!(one("20 Oct", FieldKind::Date, Some(REF)), "2026-10-20");
        // A Feb-29 anchor year is not a leap year -> no candidate.
        none("February 29", FieldKind::Date, Some("2025-06-01"));
        assert_eq!(
            one("February 29", FieldKind::Date, Some("2024-06-01")),
            "2024-02-29"
        );
    }

    #[test]
    fn month_day_year_absolute() {
        assert_eq!(one("October 20, 2026", FieldKind::Date, None), "2026-10-20");
        assert_eq!(one("20 Oct 2026", FieldKind::Date, None), "2026-10-20");
        assert_eq!(one("October 20 2026", FieldKind::Date, None), "2026-10-20");
        assert_eq!(
            one("February 29, 2024", FieldKind::Date, None),
            "2024-02-29"
        );
        none("February 30, 2026", FieldKind::Date, None);
        // "October 2026" is a month with a number after it — not month-day.
        none("October 2026", FieldKind::Date, Some(REF));
    }

    #[test]
    fn numeric_month_day_year_us_rule() {
        assert_eq!(one("10/20/2026", FieldKind::Date, None), "2026-10-20");
        // Ambiguous (both <= 12): the US M/D reading wins — documented rule.
        assert_eq!(one("11/05/2026", FieldKind::Date, None), "2026-11-05");
        // First component > 12: the D/M/Y reading takes over.
        assert_eq!(one("20/10/2026", FieldKind::Date, None), "2026-10-20");
        assert_eq!(one("13/05/2026", FieldKind::Date, None), "2026-05-13");
        none("13/13/2026", FieldKind::Date, None);
        // Two components anchor the year.
        assert_eq!(one("10/20", FieldKind::Date, Some(REF)), "2026-10-20");
        assert_eq!(one("20/10", FieldKind::Date, Some(REF)), "2026-10-20");
    }

    #[test]
    fn relative_without_reference_abstains() {
        none("tomorrow", FieldKind::Date, None);
        none("yesterday", FieldKind::Date, None);
        none("in 3 days", FieldKind::Date, None);
        none("3 days ago", FieldKind::Date, None);
        none("next monday", FieldKind::Date, None);
        // A malformed anchor counts as absent.
        none("tomorrow", FieldKind::Date, Some("not-a-date"));
        none("tomorrow", FieldKind::Date, Some("2026-02-30"));
    }

    #[test]
    fn month_day_without_reference_abstains() {
        none("October 20", FieldKind::Date, None);
        none("20 Oct", FieldKind::Date, None);
        none("10/20", FieldKind::Date, None);
    }

    #[test]
    fn word_times_noon_midnight_meridiem() {
        assert_eq!(one("noon", FieldKind::Time, None), "12:00");
        assert_eq!(one("midnight", FieldKind::Time, None), "00:00");
        assert_eq!(one("5pm", FieldKind::Time, None), "17:00");
        assert_eq!(one("5:30pm", FieldKind::Time, None), "17:30");
        assert_eq!(one("5:30:15pm", FieldKind::Time, None), "17:30:15");
        assert_eq!(one("9 am", FieldKind::Time, None), "09:00");
        assert_eq!(one("9am", FieldKind::Time, None), "09:00");
        assert_eq!(one("12am", FieldKind::Time, None), "00:00");
        assert_eq!(one("12pm", FieldKind::Time, None), "12:00");
        assert_eq!(one("12:30 pm", FieldKind::Time, None), "12:30");
        none("0pm", FieldKind::Time, None);
        none("13pm", FieldKind::Time, None);
        none("noonish", FieldKind::Time, None);
    }

    #[test]
    fn composed_datetime_at_space_and_t() {
        assert_eq!(
            one("tomorrow at 5pm", FieldKind::DateTime, Some(REF)),
            "2026-10-21T17:00"
        );
        assert_eq!(
            one("October 20 at 14:00", FieldKind::DateTime, Some(REF)),
            "2026-10-20T14:00"
        );
        assert_eq!(
            one("tomorrow 5pm", FieldKind::DateTime, Some(REF)),
            "2026-10-21T17:00"
        );
        assert_eq!(
            one("next monday at 9 am", FieldKind::DateTime, Some(REF)),
            "2026-10-26T09:00"
        );
        assert_eq!(
            one("October 20, 2026 at 5pm", FieldKind::DateTime, None),
            "2026-10-20T17:00"
        );
        // A bare date is NOT a DateTime candidate (a time is required).
        none("2026-10-21", FieldKind::DateTime, None);
        none("tomorrow", FieldKind::DateTime, Some(REF));
        none("tomorrow at", FieldKind::DateTime, Some(REF));
    }

    #[test]
    fn date_kind_rank_absolute_over_anchored() {
        let v = dates("October 20, 2026", FieldKind::Date, Some(REF));
        assert!(
            v.len() >= 2,
            "expected the absolute + anchored overlap: {v:?}"
        );
        assert_eq!((v[0].1, v[0].0.as_str()), (0, "October 20, 2026"));
        assert_eq!((v[1].1, v[1].0.as_str()), (1, "October 20"));
        assert_eq!(v[0].2, "2026-10-20");
        assert_eq!(v[1].2, "2026-10-20");
    }

    #[test]
    fn out_of_range_results_abstain() {
        none("in 3 days", FieldKind::Date, Some("9999-12-31"));
        none("3 days ago", FieldKind::Date, Some("0001-01-01"));
        none("in 9000 months", FieldKind::Date, Some("9999-01-01"));
    }

    #[test]
    fn span_exactness_on_relatives_and_composed() {
        let state = "meet tomorrow at 5pm ok";
        let mut out = Vec::new();
        parse_datetime_kind(state, FieldKind::DateTime, Some(REF), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(span_text(state, &out[0]), "tomorrow at 5pm");
        assert_eq!((out[0].span.start, out[0].span.end), (5, 20));

        let state = "é Oct 20";
        let mut out = Vec::new();
        parse_datetime_kind(state, FieldKind::Date, Some(REF), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(span_text(state, &out[0]), "Oct 20");
    }

    // ---- G4: allocation gate (crate::alloc_gate; see its module doc) -------

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn scan_datetimes_is_zero_alloc() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        // SCAN phase only: materialization (the wire's `Date`/`Time`/
        // `DateTime` strings) is the documented cold edge of
        // `parsers::finish` — the scanner itself is push-only.
        let state = "issued 2026-10-20, due 2026-11-19, revised 10/25/2026 at 5pm";
        let reference = parse_reference(Some(REF));
        assert!(reference.is_some());
        // Dry run sizes the scratch; the measured run reuses it (clear keeps
        // capacity, so the same candidate count cannot grow the buffer).
        let mut scratch = Vec::new();
        scan_datetimes(
            state,
            0..state.len(),
            FieldKind::Date,
            reference,
            &mut scratch,
        );
        let expected = scratch.len();
        assert!(expected > 0, "dry run must produce candidates");
        scratch.clear();

        let _window = MeasuringWindow::enter();
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        scan_datetimes(
            state,
            0..state.len(),
            FieldKind::Date,
            reference,
            &mut scratch,
        );
        assert_eq!(
            ALLOC_COUNT.load(Ordering::Relaxed),
            0,
            "scan_datetimes allocated inside the measured window"
        );
        drop(_window);
        assert_eq!(scratch.len(), expected);
    }
}
