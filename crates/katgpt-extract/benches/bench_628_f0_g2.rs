//! Bench 628 — katgpt-extract G2: the F0 extraction floor (Plan 628 T0.6).
//!
//! Not a `criterion` bench (the repo's zero-dep posture): a plain `main()`
//! over `std::time::Instant` + `std::hint::black_box`, in the
//! `bench_924_assignment_g2_baselines` style. Sections:
//!
//! 1. **Composite F0-floor request** — the number the plan records: one
//!    ~440-char invoice-flavored state, six fields (text via span_pointer
//!    with a choices alias table, number, date, duration, list-of-number,
//!    estimate). Per iteration: `parse_field` ×4, `SpanPointer::best_span`
//!    ×1, `EstimateHead::estimate_into` ×1, `TemperGate::update` +
//!    `gate_located` ×per-field. Cold builds (`SpanPointer::new`,
//!    `EstimateHead::new` + `calibrate`, `TemperGate::new`) happen OUTSIDE
//!    the timed loop; the loop measures the F0 hot path only. Warmup, then
//!    10_000 timed iterations; per-iter ns into a pre-allocated `Vec<u32>`;
//!    p50/p90/p99/max by nearest rank over the sorted samples.
//! 2. **Micro sections** (info-only, ns/call over 10_000 iterations each):
//!    `parse_number`, span `score_into` full sweep (out sized via
//!    `window_count()`), temper `calibration_spread`, `estimate_into`,
//!    `forecast_into` h=8. Micro timings include the two `Instant` calls
//!    (~20–40 ns) — the same overhead rides every section, so the SHAPE
//!    compares cleanly even at the ns scale.
//! 3. **One assertion**: composite p99 ≤ 10_000_000 ns (10 ms) — the
//!    Phase-0 pinned bar, ~46× headroom against Cloudflare Clairic's
//!    measured 0.46 s extraction. Deliberately NOT tighter: the measured
//!    numbers go into the plan record; a ratchet is a later, measured act.
//!
//! No box-state capture here (provenance is the coordinator's job at
//! record time). Run:
//! `cargo run --release -p katgpt-extract --bench bench_628_f0_g2 --features span_pointer,typed_parsers,temper,extraction_estimate`

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::extraction_wire::{
    EstimateDistribution, FieldConstraints, FieldKind, FieldSpec, ForecastDistribution, Grid,
    GridKind,
};
use katgpt_extract::estimate::{Comparable, EstimateHead, InMemorySource};
use katgpt_extract::forecast::ForecastHead;
use katgpt_extract::parsers::{parse_field, parse_number};
use katgpt_extract::span_pointer::{SpanConfig, SpanPointer, SpanQuery, SpanScore};
use katgpt_extract::temper::{GateVerdict, TemperConfig, TemperGate};

const WARMUP: u32 = 1_000;
const ITERS: u32 = 10_000;

/// The composite F0-floor request state: ~440 chars, invoice/ticket-
/// flavored English, carrying one instance of every located field class.
const STATE: &str = "INVOICE 2041 from Acme Corporation, billing department. \
Repair labor 1,250 dollars plus 42 replacement units at 18 percent tax. \
Issued 2026-10-20; payment due 2026-11-19; terms net 30 days. \
Requested part quantities: 3, 5 and 12 pieces. \
The Seattle warehouse ships line items within one week of approval. \
Late balances accrue 15 dollars per month after the due date. \
Contact the Acme Corporation help desk before 5pm.";

const REF_DATE: &str = "2026-10-20";
/// The estimate field's corpus key (the head calibrates against it).
const EST_ID: &str = "repair_estimate";

/// Nearest-rank percentile over ASCENDING-sorted samples
/// (`sorted[ceil(p/100 · n) − 1]`).
fn nearest_rank(sorted: &[u32], pct: f64) -> u32 {
    let n = sorted.len();
    assert!(!sorted.is_empty(), "no samples");
    let rank = ((pct / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

/// Run `body` WARMUP + ITERS times; the timed iterations record per-call
/// ns into a pre-allocated `Vec<u32>`. `body` returns a u64 folded into a
/// deterministic work checksum the caller prints (every output consumed —
/// nothing optimizes away). Returns (ascending-sortable samples, checksum).
fn bench_ns(mut body: impl FnMut() -> u64) -> (Vec<u32>, u64) {
    let mut samples = Vec::with_capacity(ITERS as usize);
    let mut sink = 0u64;
    for _ in 0..WARMUP {
        sink = sink.wrapping_add(black_box(body()));
    }
    for _ in 0..ITERS {
        let t = Instant::now();
        let v = body();
        let dt = t.elapsed().as_nanos() as u32; // µs-scale bodies cannot wrap
        sink = sink.wrapping_add(v);
        samples.push(dt);
    }
    (samples, sink)
}

fn print_stats(label: &str, sorted: &[u32]) {
    println!(
        "  {label:<46} p50 {:>7} ns | p90 {:>7} ns | p99 {:>7} ns | max {:>7} ns",
        nearest_rank(sorted, 50.0),
        nearest_rank(sorted, 90.0),
        nearest_rank(sorted, 99.0),
        sorted[sorted.len() - 1],
    );
}

fn field(id: &str, kind: FieldKind, cons: FieldConstraints) -> FieldSpec {
    FieldSpec {
        id: id.to_string(),
        kind,
        description: None,
        constraints: cons,
        instructions: None,
    }
}

fn main() {
    println!("bench 628 — katgpt-extract G2: the F0 extraction floor (Plan 628 T0.6)");
    println!(
        "state: {} chars, 6 fields; warmup {WARMUP} + {ITERS} timed iters per section",
        STATE.len()
    );
    println!();

    // ---- cold builds (outside every timed loop) ---------------------------

    let choices: Vec<String> = [
        "Acme Corporation",
        "Globex",
        "Initech",
        "Umbrella Corp",
        "Vandelay Industries",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let alias_refs: Vec<&str> = choices.iter().map(String::as_str).collect();

    let text_cons = FieldConstraints {
        choices: choices.clone(),
        ..FieldConstraints::default()
    };
    let _f_vendor = field("vendor", FieldKind::Text, text_cons);
    let f_total = field("total_due", FieldKind::Number, FieldConstraints::default());
    let f_due = field("due_date", FieldKind::Date, FieldConstraints::default());
    let f_terms = field(
        "payment_terms",
        FieldKind::Duration,
        FieldConstraints::default(),
    );
    let list_cons = FieldConstraints {
        element_kind: Some(FieldKind::Number),
        ..FieldConstraints::default()
    };
    let f_parts = field("part_quantities", FieldKind::List, list_cons);
    let grid = Grid {
        kind: GridKind::Linear,
        step: 25.0,
    };
    let est_cons = FieldConstraints {
        grid: Some(grid),
        ..FieldConstraints::default()
    };
    let _f_estimate = field(EST_ID, FieldKind::Estimate, est_cons);

    let sp = SpanPointer::new(STATE, SpanConfig::default());
    let text_query = SpanQuery {
        text: "vendor name billing contact",
        aliases: &alias_refs,
    };

    // 30 comparables, 3-feature schema (labor, units, tax) -> value. Fully
    // deterministic (arithmetic — no rng, no clock).
    let mut source = InMemorySource::new();
    for i in 0..30u32 {
        let labor = 0.5 + f64::from(i) * 0.25;
        let units = f64::from(i % 9);
        let tax = 5.0 + f64::from(i % 7) * 2.0;
        let value = 120.0 + 40.0 * labor + 7.5 * units + 2.0 * tax;
        source.push(
            EST_ID,
            Comparable {
                features: vec![labor as f32, units as f32, tax as f32],
                value: value as f32,
            },
        );
    }
    let mut est_head = EstimateHead::new(3, 0.05, Some(grid));
    est_head.domain = Some((0.0, 5000.0));
    est_head.calibrate(&source, EST_ID);
    let est_query = [2.0f32, 42.0, 18.0];
    let mut est_out = EstimateDistribution {
        point: 0.0,
        p10: 0.0,
        median: 0.0,
        p90: 0.0,
        p_below: 0.0,
        p_above: 0.0,
        level: 0.95,
    };

    let mut fc_head = ForecastHead::new(12, 8, 0.05);
    for t in 0..36u32 {
        let y = (core::f32::consts::TAU * t as f32 / 12.0).sin();
        fc_head.observe(y, &[]);
    }
    let mut fc_out = ForecastDistribution {
        steps: Vec::new(),
        level: 0.0,
    };
    fc_head.forecast_into(0.0, 8, 0.8, &mut fc_out); // pre-size the steps

    // Per-field gate inputs for the composite: (kind, best_score, margin).
    // Stand-ins for the typed fields — the engine (Phase 1) derives these;
    // the bench exercises the gate's cost and verdict path at realistic
    // values. The Text field substitutes the REAL span score at run time.
    let field_gates: [(FieldKind, f32, f32); 6] = [
        (FieldKind::Text, 0.9, 1.2),
        (FieldKind::Number, 0.85, 0.9),
        (FieldKind::Date, 0.8, 0.8),
        (FieldKind::Duration, 0.8, 0.8),
        (FieldKind::List, 0.8, 0.8),
        (FieldKind::Estimate, 0.85, 1.0),
    ];

    // Caller-owned output buffers, reserved once (clear keeps capacity —
    // the per-iteration cost is the parse, never a grow).
    let mut out_total = Vec::with_capacity(16);
    let mut out_due = Vec::with_capacity(16);
    let mut out_terms = Vec::with_capacity(16);
    let mut out_parts = Vec::with_capacity(16);

    // ---- section 1: the composite F0-floor request ------------------------

    let mut gate = TemperGate::new(TemperConfig::default());
    let mut drip = 0u32; // the residual pools' deterministic input drip

    let (mut composite, composite_sink) = bench_ns(|| {
        out_total.clear();
        parse_field(STATE, &f_total, None, &mut out_total);
        out_due.clear();
        parse_field(STATE, &f_due, Some(REF_DATE), &mut out_due);
        out_terms.clear();
        parse_field(STATE, &f_terms, None, &mut out_terms);
        out_parts.clear();
        parse_field(STATE, &f_parts, None, &mut out_parts);

        let best = sp.best_span(&text_query);

        est_head.estimate_into(&est_query, &source, EST_ID, &mut est_out);

        let mut answered = 0u32;
        for (kind, best_score, margin) in field_gates.iter() {
            let (score, margin) = if *kind == FieldKind::Text {
                match best {
                    Some(s) => (s.score, (s.score - 0.3).max(0.0)),
                    None => (*best_score, *margin),
                }
            } else {
                (*best_score, *margin)
            };
            // A deterministic realized/predicted drip keeps every residual
            // pool at steady state (ring wrapped) through the run.
            drip = drip.wrapping_add(1);
            let realized = 0.5 + (drip % 8) as f32 * 0.02;
            gate.update(*kind, realized, realized - 0.1);
            if matches!(gate.gate_located(*kind, score, margin), GateVerdict::Answer) {
                answered += 1;
            }
        }

        answered as u64
            + u64::try_from(out_total.len()).unwrap_or(0)
            + u64::try_from(out_due.len()).unwrap_or(0)
            + u64::try_from(out_terms.len()).unwrap_or(0)
            + u64::try_from(out_parts.len()).unwrap_or(0)
            + (est_out.point.abs() as u64)
            + best.map_or(0, |s| u64::from(s.start))
    });
    composite.sort_unstable();

    let (p50, p90, p99, max) = (
        nearest_rank(&composite, 50.0),
        nearest_rank(&composite, 90.0),
        nearest_rank(&composite, 99.0),
        composite[composite.len() - 1],
    );
    println!(
        "composite F0-floor request (6 fields, {}-char state):",
        STATE.len()
    );
    println!(
        "  p50 {p50} ns | p90 {p90} ns | p99 {p99} ns | max {max} ns | bar: p99 <= 10_000_000 ns"
    );
    if let Some(b) = sp.best_span(&text_query) {
        println!(
            "  span best: {:?} score {:.4} | estimate point {:.2} [{:.2}, {:.2}]",
            &STATE[b.start as usize..b.end as usize],
            b.score,
            est_out.point,
            est_out.p10,
            est_out.p90
        );
    }
    println!();

    // ---- section 2: micro sections (info-only) ----------------------------

    let num_cons = FieldConstraints::default();
    let mut num_out = Vec::with_capacity(16);
    let (mut s, sink1) = bench_ns(|| {
        num_out.clear();
        parse_number(STATE, &num_cons, &mut num_out);
        u64::try_from(num_out.len()).unwrap_or(0)
    });
    s.sort_unstable();
    print_stats("parse_number (single call)", &s);

    let sweep_len = sp.window_count();
    let mut sweep = vec![
        SpanScore {
            start: 0,
            end: 0,
            score: 0.0,
        };
        sweep_len
    ];
    let (mut s, sink2) = bench_ns(|| {
        let n = sp.score_into(&text_query, &mut sweep);
        u64::try_from(n).unwrap_or(0)
    });
    s.sort_unstable();
    print_stats(
        &format!("span score_into full sweep ({sweep_len} windows)"),
        &s,
    );

    let mut pool_gate = TemperGate::new(TemperConfig::default());
    for i in 0..512u32 {
        let r = (i % 16) as f32 * 0.05;
        pool_gate.update(FieldKind::Number, r + 0.1, r);
    }
    let (mut s, sink3) = bench_ns(|| {
        let spread = pool_gate.calibration_spread(FieldKind::Number);
        (spread * 1.0e3) as u64
    });
    s.sort_unstable();
    print_stats("temper calibration_spread (256-pool)", &s);

    let (mut s, sink4) = bench_ns(|| {
        est_head.estimate_into(&est_query, &source, EST_ID, &mut est_out);
        est_out.point.abs() as u64
    });
    s.sort_unstable();
    print_stats("estimate_into (30 comps, split-conformal)", &s);

    let (mut s, sink5) = bench_ns(|| {
        fc_head.forecast_into(0.0, 8, 0.8, &mut fc_out);
        u64::try_from(fc_out.steps.len()).unwrap_or(0)
    });
    s.sort_unstable();
    print_stats("forecast_into h=8 (36 obs, m=12)", &s);

    let work = composite_sink
        .wrapping_add(sink1)
        .wrapping_add(sink2)
        .wrapping_add(sink3)
        .wrapping_add(sink4)
        .wrapping_add(sink5);
    println!();
    println!("work checksum (deterministic sink): {}", black_box(work));

    // ---- section 3: the one assertion --------------------------------------

    // Phase-0 pinned bar: composite p99 <= 10 ms — ~46x headroom vs Cloud-
    // flare Clairic's measured 0.46 s extraction. NOT tightened here: the
    // measured numbers go into the plan; a ratchet is a later, measured act.
    if p99 > 10_000_000 {
        panic!("G2 bar FAILED: composite p99 {p99} ns > 10_000_000 ns");
    }
    println!(
        "G2 bar PASS: composite p99 {p99} ns <= 10_000_000 ns (Phase-0 pinned bar, ~46x Clairic headroom)"
    );
}
