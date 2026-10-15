//! Issue 942 report-only debug lane: latent pseudo-text trace demo.
//!
//! Builds a synthetic embedding table, walks three synthetic latent spans
//! (healthy / loop / collapse) plus one BLIND span, renders each as glyph-led
//! markdown, and exits 0 ALWAYS — the lane is report-only. The blindness
//! floor (`--min-steps N`, default 8) is what keeps an instrument that
//! inspects nothing from printing green.
//!
//! Run: `cargo run -p katgpt-transformer --features swir_switch_thinking
//!       --example latent_readout_trace`

use katgpt_transformer::swir::pseudo_text::{
    LatentLoopSensor, LatentReadoutBank, NearestToken, PseudoTextCollapseSensor, TraceRow,
    render_trace_markdown, span_readability,
};

const VOCAB: usize = 96;
const DIM: usize = 64;
const K: usize = 3;

/// Deterministic table: row j = sin/cos mixture in j (no zero rows, seeded).
fn synth_table() -> Vec<f32> {
    let mut t = Vec::with_capacity(VOCAB * DIM);
    for j in 0..VOCAB {
        for d in 0..DIM {
            let x = (j as f32 * 0.37 + d as f32 * 0.11).sin()
                + 0.5 * ((j as f32 * 0.13 - d as f32 * 0.07).cos());
            t.push(x + 0.25);
        }
    }
    t
}

struct SpanCfg {
    title: &'static str,
    steps: usize,
    /// How to build step t's latent: which rows mix, and the noise scale.
    kind: Kind,
}

enum Kind {
    /// Walk: different row pair per step, real noise — readable, no loops.
    Walk,
    /// Loop: same pair every step, noise 1e-4 — cos(v_t, v_{t-1}) ~ 1.
    Loop,
    /// Collapse: EXACTLY constant latent — nearest token constant, high cos.
    Collapse,
    /// Blind: zero steps — the blindness-floor demo.
    Blind,
}

fn make_latent(table: &[f32], step: usize, kind: &Kind) -> Vec<f32> {
    let mix = |ja: usize, jb: usize, noise: f32| -> Vec<f32> {
        (0..DIM)
            .map(|d| {
                let base = table[ja * DIM + d] * 0.7 + table[jb * DIM + d] * 0.3;
                let n = ((step + d) as f32 * 0.71).sin() * noise;
                base + n
            })
            .collect()
    };
    match *kind {
        Kind::Walk => mix((step * 7) % VOCAB, (step * 13 + 5) % VOCAB, 0.05),
        Kind::Loop => mix(41, 42, 1e-4),
        Kind::Collapse => mix(41, 42, 0.0),
        Kind::Blind => unreachable!("Blind spans generate no steps"),
    }
}

fn run_span(bank: &LatentReadoutBank, table: &[f32], cfg: &SpanCfg, min_steps: usize) -> String {
    let mut loop_sensor = LatentLoopSensor::new(0.99, 2, DIM);
    let mut collapse_sensor = PseudoTextCollapseSensor::new(4, 0.8);
    let mut scores = vec![0.0f32; bank.vocab()];
    let mut top = vec![(0usize, 0.0f32); K];
    let mut max_cos: Vec<f32> = Vec::new();
    let mut tops: Vec<Vec<(usize, f32)>> = Vec::new();
    let mut loop_states: Vec<bool> = Vec::new();
    let mut collapse_states: Vec<bool> = Vec::new();

    for t in 0..cfg.steps {
        let v = make_latent(table, t, &cfg.kind);
        // ONE matvec per inspected step: the top-k head IS the nearest token.
        let filled = bank.readout_topk_into(&v, &mut top, &mut scores);
        tops.push(top[..filled].to_vec());
        if filled > 0 {
            let near = NearestToken { token: top[0].0, cos: top[0].1 };
            max_cos.push(near.cos);
            collapse_states.push(collapse_sensor.observe(near));
        } else {
            // Zero/NaN-norm query: no collapse claim, no readability point.
            collapse_states.push(false);
        }
        loop_states.push(loop_sensor.observe(&v));
    }

    let rows: Vec<TraceRow<'_>> = tops
        .iter()
        .zip(loop_states.iter().zip(collapse_states.iter()))
        .enumerate()
        .map(|(step, (topk, (&lp, &cp)))| TraceRow {
            step,
            top: topk.as_slice(),
            loop_fired: lp,
            collapse_fired: cp,
        })
        .collect();
    render_trace_markdown(&rows, span_readability(&max_cos), min_steps, cfg.title)
}

fn main() {
    let min_steps = std::env::args()
        .skip_while(|a| a != "--min-steps")
        .nth(1)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);

    let table = synth_table();
    let bank = LatentReadoutBank::build(&table, VOCAB, DIM)
        .unwrap_or_else(|e| panic!("synthetic table must build: {e}"));

    let spans = [
        SpanCfg { title: "healthy walk", steps: 12, kind: Kind::Walk },
        SpanCfg { title: "loop span", steps: 8, kind: Kind::Loop },
        SpanCfg { title: "collapse span", steps: 8, kind: Kind::Collapse },
        SpanCfg { title: "blind span (zero steps inspected)", steps: 0, kind: Kind::Blind },
    ];

    let mut out = String::new();
    for cfg in &spans {
        out.push_str(&run_span(&bank, &table, cfg, min_steps));
        out.push_str("\n\n");
    }
    println!("{out}");
    println!("report-only lane: exit 0 regardless of sensor state (Issue 942)");
}
