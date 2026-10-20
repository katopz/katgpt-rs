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
//!   US M/D/Y reading wins the ambiguous forms.
//! - Times: `5pm`, `5:30pm`, `5:30:15pm`, `9 am` (space optional),
//!   `17:30`, `17:30:45`, `05:30`, bare `H:MM` (24h), `noon`, `midnight`.
//!   `12am` = `00:00`, `12pm` = `12:00`. An hour alone (`5`) is not a time.
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

use super::{
    ParsedValue, RawCandidate, RawValue, ascii_word_at, finish, is_word_byte, next_boundary,
    prev_char_index, skip_ascii_ws,
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
    if let Some((e, y, m, d)) = iso_date_at(b, we, i) {
        push_date(sink, i, e, y, m, d, RANK_ABSOLUTE);
    }
    if let Some(h) = numeric_date_at(b, we, i, reference) {
        push_date(
            sink,
            i,
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
    if let Some((abs, anch)) = month_day_at(b, we, i) {
        if let Some(a) = abs {
            push_date(sink, i, a.end, a.y, a.m, a.d, RANK_ABSOLUTE);
        }
        if let Some((ae, m, d)) = anch
            && let Some(r) = reference
            && let Some(y) = anchored_year(r.y, m, d)
        {
            push_date(sink, i, ae, y, m, d, RANK_ANCHORED);
        }
    }
    if let Some((e, y, m, d)) = relative_at(b, we, i, reference) {
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
            raw: RawValue::Date { y, m, d },
        });
    }
}

fn push_time(sink: &mut Vec<RawCandidate>, start: usize, t: TimeHit) {
    sink.push(RawCandidate {
        start: start as u32,
        end: t.end as u32,
        rank: RANK_ABSOLUTE,
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
    let Some(hit) = compose_date_at(b, we, i, reference) else {
        return;
    };
    let time = if hit.end < we && b[hit.end] == b'T' {
        time_at(b, we, hit.end + 1)
    } else {
        let j = skip_ascii_ws(b, hit.end, we);
        if j == hit.end {
            None
        } else if let Some(ae) = ascii_word_at(b, we, j, "at") {
            let k = skip_ascii_ws(b, ae, we);
            if k > ae { time_at(b, we, k) } else { None }
        } else {
            time_at(b, we, j)
        }
    };
    if let Some(t) = time {
        sink.push(RawCandidate {
            start: i as u32,
            end: t.end as u32,
            rank: RANK_ABSOLUTE,
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
    let (e, v) = one_two_digits(b, we, i)?;
    if (1..=31).contains(&v) {
        Some((e, v))
    } else {
        None
    }
}

/// Numeric `a/b[year]`. Disposition: `a > 12` -> D/M, else M/D (the US
/// reading wins the ambiguous forms — the documented rule).
fn numeric_date_at(b: &[u8], we: usize, i: usize, reference: Option<Ref>) -> Option<DateHit> {
    let (ae, a) = one_two_digits(b, we, i)?;
    if ae >= we || b[ae] != b'/' {
        return None;
    }
    let (ce, c) = one_two_digits(b, we, ae + 1)?;
    let mut e = ce;
    let mut year: Option<i64> = None;
    if e < we
        && b[e] == b'/'
        && let Some((ye, y)) = year4_at(b, we, e + 1)
    {
        year = Some(y);
        e = ye;
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
fn month_day_at(b: &[u8], we: usize, i: usize) -> Option<MonthDayHits> {
    if let Some((me, month)) = month_at(b, we, i) {
        let j = skip_ascii_ws(b, me, we);
        if j > me
            && let Some((de, day)) = day_digits_at(b, we, j)
        {
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

/// Shared tail of both month-day forms: an optional year, then boundary
/// checks and calendar validation. `after` is the offset just past the
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
        if let Some((ye, y)) = year4_at(b, we, j) {
            year = Some(y);
            e = ye;
        }
    } else if j > after
        && let Some((ye, y)) = year4_at(b, we, j)
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

/// 1-4 digit unsigned integer (relative offsets), value <= 9999.
fn short_number_at(b: &[u8], we: usize, i: usize) -> Option<(usize, i64)> {
    if i >= we || !b[i].is_ascii_digit() {
        return None;
    }
    let mut v: i64 = 0;
    let mut e = i;
    while e < we && b[e].is_ascii_digit() && e - i < 4 {
        v = v * 10 + i64::from(b[e] - b'0');
        e += 1;
    }
    Some((e, v))
}

// ---------------------------------------------------------------------------
// Time matchers
// ---------------------------------------------------------------------------

/// `noon` | `midnight` | `<h>[:mm[:ss]]( )?[am|pm]` | `H:MM[:SS]` (24h).
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
    clock24_at(b, we, i)
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

/// 1..=12 hour with an optional `:mm[:ss]` tail, optional space, then
/// `am`/`pm`. `12am` -> 00, `12pm` -> 12.
fn meridiem_time_at(b: &[u8], we: usize, i: usize) -> Option<TimeHit> {
    let (mut e, mut h) = one_two_digits(b, we, i)?;
    if !(1..=12).contains(&h) {
        return None;
    }
    let mut mi = 0;
    let mut s: Option<u32> = None;
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
    let j = skip_ascii_ws(b, e, we);
    if j + 2 > we {
        return None;
    }
    let is_am = (b[j] | 0x20) == b'a' && (b[j + 1] | 0x20) == b'm';
    let is_pm = (b[j] | 0x20) == b'p' && (b[j + 1] | 0x20) == b'm';
    if !is_am && !is_pm {
        return None;
    }
    let m_end = j + 2;
    if m_end < we && is_word_byte(b[m_end]) {
        return None;
    }
    h = if is_am {
        if h == 12 { 0 } else { h }
    } else if h == 12 {
        12
    } else {
        h + 12
    };
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
}
