//! G5 external-fixture harness (Plan 628 T0.6).
//!
//! Scores this crate's typed parsers ([`katgpt_extract::parsers::parse_field`],
//! the `typed_parsers` feature) against an EXTERNAL corpus — Microsoft
//! Recognizers-Text's own English test specs — never fixtures authored for
//! this test. The corpus is BLAKE3-pinned: a drifted fixture refuses to
//! score (loud panic), because a silently-moved corpus would make every
//! number in the report incomparable to the one before it.
//!
//! Provenance: `tests/fixtures/external/PROVENANCE.md` (Recognizers-Text @
//! `da7edcff59f6`, MIT; 351 rows across number/date/time/datetime/duration;
//! 275 rows carry a mechanically-translated `expected_wire`, 76 are
//! span-only; rows are printable ASCII, so char indices == byte offsets).
//!
//! ## What is scored
//!
//! - **Value lane** (wire rows, rank-0 candidate only): number vs
//!   `{"value"}` (relative diff <= 1e-9), duration vs `{"unit","value"}`
//!   (exact unit string + relative diff <= 1e-9), date/time/datetime by
//!   exact string equality. Time/datetime ALSO count a secondary LENIENT
//!   bucket (seconds-less `HH:MM` padded `:00`; datetime space<->`T`
//!   separator) — reported beside the primary number, never folded into
//!   it. No rank-0 candidate at all = miss.
//! - **Span lane** (all rows, micro-averaged over every expected entity in
//!   `expected_raw`): exact span equality and containment (candidate
//!   span superset of expected), each over any-candidate and rank-0-only.
//!   Upstream spans arrive as `[Start, End]` with `End` INCLUSIVE (number
//!   family) or `[Start, Start+Length)` (DateTime family); both normalize
//!   to `[start, end)` here.
//!
//! ## What is asserted
//!
//! Liveness AND the accuracy ratchet (issue 953 ask 2 — measured-then-
//! pinned): per-dim floors on rows-with->=1-candidate ([`LIVENESS_FLOORS`])
//! and per-dim FLOORS on rank-0 value accuracy ([`VALUE_RATCHETS`]). Both
//! are ratchets — they fire DOWNWARD only; raising accuracy never reds.
//! The ratchet's floor is the exact measured record of the issue-953
//! coverage landing (2026-10-20); a parser regression past it reds.
//!
//! The table prints to stdout (captured on a green run — use
//! `--nocapture` to read it).

use katgpt_core::extraction_wire::{FieldConstraints, FieldKind, FieldSpec, FieldValue};
use katgpt_extract::parsers::parse_field;
use serde_json::Value;

/// The pinned corpus, relative to the crate root (manifest-anchored, so
/// the test runs from any working directory).
const FIXTURE_REL: &str = "tests/fixtures/external/recognizers_core.jsonl";

/// BLAKE3 pin over the fixture bytes (see PROVENANCE.md).
const BLAKE3_PIN: &str = "f3e1dfc3dae9d417b7e93e7005857a12ba3963be9085cac93adc35665a988163";

/// Row-count tripwire (the provenance census: 351 rows).
const ROWS_EXPECTED: usize = 351;

/// Per-dim liveness floors: rows with >= 1 candidate (order = [`DIMS`]).
/// Re-pinned just under the issue-953 coverage-landing record (2026-10-20:
/// number 103/105, date 39/80, time 50/67, datetime 7/43, duration 53/56) —
/// a tripwire for a broken harness or a wiped-out parser. The original
/// T0.6 pins ([0.65, 0.05, 0.25, 0.01, 0.15]) were calibration against the
/// pre-coverage baseline; the coverage levers raised every dim's ceiling.
const LIVENESS_FLOORS: [f64; 5] = [0.98, 0.48, 0.74, 0.16, 0.94];

/// The accuracy ratchet (issue 953 ask 2): per-dim rank-0 value-accuracy
/// FLOORS, pinned at the exact measured record of the coverage landing
/// (2026-10-20: number 74/85, date 39/65, time 8/50, datetime 3/23,
/// duration 52/52 — order = [`DIMS`]). A ratchet, never a ceiling: the
/// next coverage lever re-pins upward from its own measured record. The
/// known disclosed misses (compound units, day-part idioms, `to/past`
/// forms, anchored-only shapes) sit BELOW these floors by construction.
const VALUE_RATCHETS: [f64; 5] = [74.0 / 85.0, 39.0 / 65.0, 8.0 / 50.0, 3.0 / 23.0, 1.0];

/// Fixture dims, in emission order (the report's row order).
const DIMS: [&str; 5] = ["number", "date", "time", "datetime", "duration"];

/// Relative difference within 1e-9 (symmetric scale; exact equality
/// short-circuits the zero case).
fn rel_close(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    let scale = a.abs().max(b.abs());
    scale > 0.0 && (a - b).abs() / scale <= 1e-9
}

fn wire_str<'a>(wire: &'a Value, key: &str) -> Option<&'a str> {
    wire.get(key).and_then(Value::as_str)
}

fn wire_f64(wire: &Value, key: &str) -> Option<f64> {
    wire.get(key).and_then(Value::as_f64)
}

/// `HH:MM` padded to `HH:MM:00` (the lenient tier's seconds-less form;
/// both sides of the comparison normalize through this).
fn with_seconds(s: &str) -> std::borrow::Cow<'_, str> {
    if s.len() == 5 {
        std::borrow::Cow::Owned(format!("{s}:00"))
    } else {
        std::borrow::Cow::Borrowed(s)
    }
}

/// The lenient datetime tier: space separator accepted for `T`, and a
/// seconds-less clock padded `:00`.
fn lenient_datetime(cand: &str, expected: &str) -> bool {
    let mut norm = cand.replace(' ', "T");
    if norm.len() == 16 {
        norm.push_str(":00");
    }
    norm == expected
}

/// One row's wire-verdict at the rank-0 candidate. `Lenient` counts ONLY
/// in its own reported bucket — never in the primary accuracy number.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValueVerdict {
    Exact,
    Lenient,
    Miss,
}

fn value_match(dim: &str, cand: &FieldValue, wire: &Value) -> ValueVerdict {
    let exact = || ValueVerdict::Exact;
    let miss = ValueVerdict::Miss;
    match (dim, cand) {
        ("number", FieldValue::Number(v)) => {
            if wire_f64(wire, "value").is_some_and(|e| rel_close(*v, e)) {
                exact()
            } else {
                miss
            }
        }
        ("duration", FieldValue::Duration(d)) => {
            let unit_ok = wire_str(wire, "unit") == Some(d.unit.as_str());
            let val_ok = wire_f64(wire, "value").is_some_and(|e| rel_close(d.value, e));
            if unit_ok && val_ok { exact() } else { miss }
        }
        ("date", FieldValue::Date(s)) => {
            if wire_str(wire, "iso_date") == Some(s.as_str()) {
                exact()
            } else {
                miss
            }
        }
        ("time", FieldValue::Time(s)) => match wire_str(wire, "iso_time") {
            Some(e) if s == e => exact(),
            Some(e) if with_seconds(s) == e => ValueVerdict::Lenient,
            _ => miss,
        },
        ("datetime", FieldValue::DateTime(s)) => match wire_str(wire, "iso_datetime") {
            Some(e) if s == e => exact(),
            Some(e) if lenient_datetime(s, e) => ValueVerdict::Lenient,
            _ => miss,
        },
        _ => miss,
    }
}

/// One upstream entity's expected byte span, normalized to `[start, end)`.
/// Number-family rows carry `End` INCLUSIVE (`[Start, End+1)`); the
/// DateTime family carries `Length` (`[Start, Start+Length)`).
fn expected_span(entity: &Value, id: &str) -> (u32, u32) {
    let obj = entity
        .as_object()
        .unwrap_or_else(|| panic!("G5 row {id}: expected_raw entry is not an object: {entity}"));
    let start = obj
        .get("Start")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("G5 row {id}: entity without Start: {entity}"))
        as u32;
    if let Some(end) = obj.get("End").and_then(Value::as_u64) {
        (start, end as u32 + 1)
    } else if let Some(len) = obj.get("Length").and_then(Value::as_u64) {
        (start, start + len as u32)
    } else {
        panic!("G5 row {id}: entity carries neither End nor Length: {entity}")
    }
}

/// Per-dim accumulators (the report's cells).
#[derive(Default)]
struct DimStats {
    rows: u32,
    /// Rows with >= 1 candidate (the liveness numerator).
    alive: u32,
    wire_rows: u32,
    value_exact: u32,
    value_lenient: u32,
    /// Expected entities across all rows (the span denominator).
    entities: u32,
    exact_any: u32,
    exact_r0: u32,
    contain_any: u32,
    contain_r0: u32,
    candidates: u64,
}

fn pct(num: u32, den: u32) -> String {
    if den == 0 {
        "   n/a".to_string()
    } else {
        format!("{:5.1}%", 100.0 * f64::from(num) / f64::from(den))
    }
}

fn dim_index(dim: &str, id: &str) -> usize {
    DIMS.iter()
        .position(|d| *d == dim)
        .unwrap_or_else(|| panic!("G5 row {id}: unknown dim {dim:?}"))
}

#[test]
fn g5_external_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_REL);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("G5: cannot read fixture {}: {e}", path.display()));

    // 1. Pin gate — a drifted fixture must never silently score.
    let got = blake3::hash(&bytes).to_hex();
    assert_eq!(
        got.as_str(),
        BLAKE3_PIN,
        "G5 fixture drifted: BLAKE3 {} != pinned {} — re-pin via PROVENANCE.md before scoring",
        got.as_str(),
        BLAKE3_PIN
    );

    // 2. Row census.
    let text = std::str::from_utf8(&bytes).expect("G5 fixture is UTF-8");
    let rows: Vec<Value> = text
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("G5 line {}: not JSON: {e}", i + 1))
        })
        .collect();
    assert_eq!(
        rows.len(),
        ROWS_EXPECTED,
        "G5 fixture holds {} rows, expected {ROWS_EXPECTED}",
        rows.len()
    );

    // 3. Score every row.
    let mut stats: Vec<DimStats> = (0..DIMS.len()).map(|_| DimStats::default()).collect();
    for row in &rows {
        let id = row.get("id").and_then(Value::as_str).unwrap_or("???");
        let dim = row
            .get("dim")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("G5 row {id}: missing dim"));
        let state = row
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("G5 row {id}: missing text"));
        let entities = row
            .get("expected_raw")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("G5 row {id}: missing expected_raw"));
        assert!(!entities.is_empty(), "G5 row {id}: empty expected_raw");

        let kind = match dim {
            "number" => FieldKind::Number,
            "date" => FieldKind::Date,
            "time" => FieldKind::Time,
            "datetime" => FieldKind::DateTime,
            "duration" => FieldKind::Duration,
            other => panic!("G5 row {id}: unknown dim {other:?}"),
        };
        let field = FieldSpec {
            id: format!("g5-{dim}"),
            kind,
            description: None,
            constraints: FieldConstraints::default(),
            instructions: None,
        };
        let mut out = Vec::new();
        parse_field(state, &field, None, &mut out);

        let s = &mut stats[dim_index(dim, id)];
        s.rows += 1;
        s.candidates += out.len() as u64;
        if !out.is_empty() {
            s.alive += 1;
        }

        // Value lane: rank-0 candidate only, wire rows only.
        if let Some(wire) = row.get("expected_wire").filter(|w| !w.is_null()) {
            s.wire_rows += 1;
            if let Some(cand) = out.iter().find(|c| c.rank == 0) {
                match value_match(dim, &cand.value, wire) {
                    ValueVerdict::Exact => s.value_exact += 1,
                    ValueVerdict::Lenient => s.value_lenient += 1,
                    ValueVerdict::Miss => {}
                }
            }
        }

        // Span lane: every expected entity, any-candidate + rank-0-only.
        let rank0 = out.iter().find(|c| c.rank == 0);
        for entity in entities {
            let (es, ee) = expected_span(entity, id);
            s.entities += 1;
            if out.iter().any(|c| c.span.start == es && c.span.end == ee) {
                s.exact_any += 1;
            }
            if rank0.is_some_and(|c| c.span.start == es && c.span.end == ee) {
                s.exact_r0 += 1;
            }
            if out.iter().any(|c| c.span.start <= es && c.span.end >= ee) {
                s.contain_any += 1;
            }
            if rank0.is_some_and(|c| c.span.start <= es && c.span.end >= ee) {
                s.contain_r0 += 1;
            }
        }
    }

    // 4. Report.
    println!("G5 external fixture — typed parsers vs pinned Recognizers-Text corpus");
    println!("  fixture: {FIXTURE_REL} (BLAKE3-pinned, {ROWS_EXPECTED} rows)");
    println!();
    println!(
        "{:<9}{:>5}{:>6}{:>9}{:>10}  {:>11}{:>11}{:>11}{:>11}{:>10}{:>8}",
        "dim",
        "rows",
        "wire",
        "value",
        "lenient",
        "exact any",
        "exact r0",
        "cont any",
        "cont r0",
        "cand/row",
        "live"
    );
    for (dim, s) in DIMS.iter().zip(&stats) {
        let span_cell = |n: u32| format!("{n:>4} {}", pct(n, s.entities));
        let mean = if s.rows == 0 {
            0.0
        } else {
            s.candidates as f64 / f64::from(s.rows)
        };
        println!(
            "{:<9}{:>5}{:>6}{:>9}{:>10}  {:>11}{:>11}{:>11}{:>11}{:>10.2}{:>8}",
            dim,
            s.rows,
            s.wire_rows,
            pct(s.value_exact, s.wire_rows),
            pct(s.value_lenient, s.wire_rows),
            span_cell(s.exact_any),
            span_cell(s.exact_r0),
            span_cell(s.contain_any),
            span_cell(s.contain_r0),
            mean,
            pct(s.alive, s.rows),
        );
    }
    let total = stats.iter().fold(DimStats::default(), |mut a, s| {
        a.rows += s.rows;
        a.alive += s.alive;
        a.wire_rows += s.wire_rows;
        a.value_exact += s.value_exact;
        a.value_lenient += s.value_lenient;
        a.entities += s.entities;
        a.exact_any += s.exact_any;
        a.exact_r0 += s.exact_r0;
        a.contain_any += s.contain_any;
        a.contain_r0 += s.contain_r0;
        a.candidates += s.candidates;
        a
    });
    let span_cell = |n: u32| format!("{n:>4} {}", pct(n, total.entities));
    println!(
        "{:<9}{:>5}{:>6}{:>9}{:>10}  {:>11}{:>11}{:>11}{:>11}{:>10.2}{:>8}",
        "TOTAL",
        total.rows,
        total.wire_rows,
        pct(total.value_exact, total.wire_rows),
        pct(total.value_lenient, total.wire_rows),
        span_cell(total.exact_any),
        span_cell(total.exact_r0),
        span_cell(total.contain_any),
        span_cell(total.contain_r0),
        total.candidates as f64 / f64::from(total.rows),
        pct(total.alive, total.rows),
    );
    println!();
    println!(
        "  value   = rank-0 exact matches / wire rows (the primary accuracy number — reported, NOT asserted)"
    );
    println!(
        "  lenient = secondary bucket only: seconds-less HH:MM padded :00; datetime space<->T (never folded into value)"
    );
    println!(
        "  exact   = candidate span == expected entity span;  cont = candidate span is a superset of expected"
    );
    let floors = LIVENESS_FLOORS
        .map(|f| format!("{:.0}", 100.0 * f))
        .join("/");
    println!(
        "  live    = rows with >= 1 candidate — the ONLY asserted floors ({floors}% number/date/time/datetime/duration)"
    );

    // 5. Liveness floors + the accuracy ratchet (issue 953 ask 2). Both
    // fire DOWNWARD only — a coverage lever re-pins them from its own
    // measured record, a regression reds here.
    let mut failed: Vec<String> = DIMS
        .iter()
        .zip(&stats)
        .zip(LIVENESS_FLOORS)
        .filter(|((_, s), floor)| f64::from(s.alive) / f64::from(s.rows) < *floor)
        .map(|((dim, s), floor)| {
            format!(
                "  {dim}: {}/{} rows with >=1 candidate ({:.1}%) < {:.0}% floor",
                s.alive,
                s.rows,
                100.0 * f64::from(s.alive) / f64::from(s.rows),
                100.0 * floor
            )
        })
        .collect();
    for ((dim, s), ratchet) in DIMS.iter().zip(&stats).zip(VALUE_RATCHETS) {
        let acc = f64::from(s.value_exact) / f64::from(s.wire_rows);
        if s.wire_rows > 0 && acc < ratchet {
            failed.push(format!(
                "  {dim}: value accuracy {}/{} = {:.1}% < ratchet {:.1}% (issue 953 pin)",
                s.value_exact,
                s.wire_rows,
                100.0 * acc,
                100.0 * ratchet
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "G5 liveness/accuracy floor failed (a broken harness, broken parsers, or a regression past the issue-953 ratchet):\n{}",
        failed.join("\n")
    );
}
