//! Issue 747 P0.7 — real-model ASEntmax re-gate (G2 quality + G3 latency on
//! REAL prefill routing rows).
//!
//! Replays the committed fixture (`tests/data/asentmax_p07_bonsai8b.fixture`)
//! — captured from an actual Ternary-Bonsai-8B (qwen3, ternary Q2_0_g128)
//! prefill over real English prose by
//! `examples/asentmax_p07_gen_fixture.rs` — through the production router
//! `EntmaxRouter::forward_indexer` in both arms:
//!
//! - **None** — the shipped Plan 106 path (default; must stay bit-identical).
//! - **`with_asentmax_schedule()`** — the P0.7 wiring (rolling-σ̂ EMA α=0.8,
//!   the Bench 713 G2 configuration), fed the rows in stream order exactly
//!   as the estimator would observe them on the hot path.
//!
//! # G2 axes (quality on real rows)
//!
//! 1. **Routing fidelity vs ground truth** — the oracle is full softmax
//!    attention over the same real q/K at the captured step: mass coverage =
//!    Σ oracle mass over the router's selected blocks, plus top-m recall at
//!    equal budget against the oracle's top-m blocks.
//! 2. **Support vs n** — the over-sparsification axis: mean entmax support
//!    as the block count grows, raw vs scheduled (the paper's failure mode:
//!    fixed-α entmax collapses as the candidate set grows).
//! 3. **Needle retention** — the planted fact sentence (Bench 032 pattern):
//!    rows whose oracle top-1 block IS the needle block must keep it in the
//!    support.
//!
//! # G3 axis (no-regression)
//!
//! Steady-state scheduled-vs-None router latency on the same real rows.
//! The schedule adds one `powf` + an O(n) row scan (estimator) per call —
//! bounded by a generous ratio while the measured number is printed.
//!
//! The measured tables + verdict live in `.benchmarks/713` (P0.7 addendum).

#![cfg(feature = "asentmax_schedule")]

use std::time::Instant;

use katgpt_attn::dash_attn::asentmax::RollingSigmaEstimator;
use katgpt_attn::dash_attn::entmax_router::{EntmaxCache, EntmaxRouter};
use katgpt_attn::dash_attn::routing::score_blocks_entmax_into;
use katgpt_attn::dash_attn::vortex_flow::{VortexFlow, VortexScratch};
use katgpt_core::types::DashAttnConfig;

// ── fixture format (mirrors examples/asentmax_p07_gen_fixture.rs) ──────────

const FIXTURE: &[u8] = include_bytes!("data/asentmax_p07_bonsai8b.fixture");

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e: i32 = -1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            (sign << 31) | (((113 - e) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl Cur<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn f16(&mut self) -> f32 {
        let h = u16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]);
        self.p += 2;
        f16_to_f32(h)
    }
}

struct Stream {
    layer: usize,
    kv_head: usize,
    n_blocks: usize,
    /// [n_blocks][head_dim] mean-pooled post-rope K summaries.
    sums: Vec<f32>,
}

struct Row {
    layer: usize,
    q_head: usize,
    kv_head: usize,
    n_blocks: usize,
    query: Vec<f32>,
    /// [n_blocks] ground-truth softmax attention mass per block.
    masses: Vec<f32>,
}

struct Fixture {
    head_dim: usize,
    block: usize,
    n_tokens: usize,
    prompt_fnv: u64,
    /// The planted needle sentence's 64-token block (Bench 032 axis).
    needle_block: usize,
    streams: Vec<Stream>,
    rows: Vec<Row>,
}

fn parse_fixture(bytes: &[u8]) -> Fixture {
    let mut c = Cur { b: bytes, p: 0 };
    let magic = &bytes[0..8];
    assert_eq!(magic, b"ASEP07\0\0", "fixture magic");
    c.p = 8;
    let version = c.u32();
    assert_eq!(version, 1, "fixture version");
    let head_dim = c.u32() as usize;
    let block = c.u32() as usize;
    let n_streams = c.u32() as usize;
    let n_rows = c.u32() as usize;
    let n_tokens = c.u32() as usize;
    let prompt_fnv = c.u64();
    let needle_block = c.u64() as usize;
    let mut streams = Vec::with_capacity(n_streams);
    for _ in 0..n_streams {
        let layer = c.u32() as usize;
        let kv_head = c.u32() as usize;
        let n_blocks = c.u32() as usize;
        let mut sums = vec![0f32; n_blocks * head_dim];
        for s in &mut sums {
            *s = c.f16();
        }
        streams.push(Stream { layer, kv_head, n_blocks, sums });
    }
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let layer = c.u32() as usize;
        let q_head = c.u32() as usize;
        let kv_head = c.u32() as usize;
        let n_blocks = c.u32() as usize;
        let mut query = vec![0f32; head_dim];
        for q in &mut query {
            *q = c.f16();
        }
        let mut masses = vec![0f32; n_blocks];
        for m in &mut masses {
            *m = c.f32();
        }
        rows.push(Row { layer, q_head, kv_head, n_blocks, query, masses });
    }
    assert_eq!(c.p, bytes.len(), "fixture fully consumed");
    Fixture { head_dim, block, n_tokens, prompt_fnv, needle_block, streams, rows }
}

// ── replay harness ──────────────────────────────────────────────────────────

fn cache_for(fx: &Fixture, layer: usize, kv_head: usize, n_blocks: usize) -> EntmaxCache {
    let stream = fx
        .streams
        .iter()
        .find(|s| s.layer == layer && s.kv_head == kv_head)
        .unwrap_or_else(|| panic!("stream for layer {layer} kvh {kv_head}"));
    assert!(stream.n_blocks >= n_blocks);
    let mut cache = EntmaxCache::with_capacity(n_blocks, fx.head_dim);
    for b in 0..n_blocks {
        cache.summaries.push(stream.sums[b * fx.head_dim..(b + 1) * fx.head_dim].to_vec());
    }
    cache
}

fn stream_n_blocks(fx: &Fixture, layer: usize, kv_head: usize) -> usize {
    fx.streams
        .iter()
        .find(|s| s.layer == layer && s.kv_head == kv_head).map_or(0, |s| s.n_blocks)
}

struct Decision {
    blocks: Vec<usize>,
    weights: Vec<f32>,
}

/// Replay all rows through one arm. `scheduled` uses the P0.7 wiring exactly
/// (rolling-σ̂ α=0.8 fed every row in stream order, `to_schedule()` per call).
fn replay(fx: &Fixture, scheduled: bool) -> Vec<Decision> {
    let router = if scheduled {
        EntmaxRouter::default_router().with_asentmax_schedule()
    } else {
        EntmaxRouter::default_router()
    };
    let mut caches: std::collections::HashMap<(usize, usize), EntmaxCache> = Default::default();
    let mut scratch = VortexScratch::new(256);
    let mut out = Vec::with_capacity(fx.rows.len());
    for row in &fx.rows {
        let key = (row.layer, row.kv_head);
        let cache = caches
            .entry(key)
            .or_insert_with(|| cache_for(fx, row.layer, row.kv_head, stream_n_blocks(fx, row.layer, row.kv_head)));
        // ensure capacity for this row's n_blocks (caches only grow)
        if cache.summaries.len() < row.n_blocks {
            let full = cache_for(fx, row.layer, row.kv_head, row.n_blocks);
            *cache = full;
        }
        let dec = router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
        out.push(Decision { blocks: dec.blocks, weights: dec.weights });
    }
    out.shrink_to_fit();
    out
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

// ── tests ───────────────────────────────────────────────────────────────────

/// Provenance + non-vacuity: the fixture is present, well-formed, and covers
/// the expected surface. (If this fails, regenerate:
/// `cargo run --release -p katgpt-attn --features asentmax_schedule --example
/// asentmax_p07_gen_fixture -- --capture --model <Ternary-Bonsai-8B-Q2_0.gguf>`.)
#[test]
fn p07_fixture_shape_and_provenance() {
    let fx = parse_fixture(FIXTURE);
    assert_eq!(fx.head_dim, 128, "qwen3 head_dim");
    assert_eq!(fx.block, 64, "DashAttnConfig default chunk_size");
    assert!(fx.n_tokens >= 2000, "real-text prefill, got {} tokens", fx.n_tokens);
    assert!(fx.streams.len() >= 48, "8 layers × 8 kv-heads, got {}", fx.streams.len());
    assert!(fx.rows.len() >= 400, "routing rows, got {}", fx.rows.len());
    assert!(
        fx.rows.iter().map(|r| r.n_blocks).max().unwrap_or(0) >= 30,
        "n-axis coverage to ≥30 blocks"
    );
    // provenance: the fixture was captured from THIS committed prompt
    let prompt: &str = include_str!("data/asentmax_p07_prompt.txt");
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in prompt.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    assert_eq!(fx.prompt_fnv, h, "fixture prompt provenance drifted — regenerate");
    // surface coverage: 8 distinct layers × 8 distinct q-heads
    let layers: std::collections::HashSet<usize> = fx.rows.iter().map(|r| r.layer).collect();
    let heads: std::collections::HashSet<usize> = fx.rows.iter().map(|r| r.q_head).collect();
    assert!(layers.len() >= 8, "layer sample, got {}", layers.len());
    assert!(heads.len() >= 8, "q-head sample, got {}", heads.len());
    // every row's oracle masses sit on the simplex
    for row in fx.rows.iter().take(64) {
        let s: f32 = row.masses.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "masses sum {s}");
        assert!(row.masses.iter().all(|&m| m >= 0.0));
    }
}

/// G3 pre-pin: the None arm on real rows is byte-identical to the shipped
/// `score_blocks_entmax_into` path (the feature-gate-audit discipline —
/// enabling `asentmax_schedule` alone changes nothing).
#[test]
fn p07_none_arm_bit_identical_on_real_rows() {
    let fx = parse_fixture(FIXTURE);
    let router = EntmaxRouter::default_router();
    assert!(router.asentmax.is_none());
    let config = DashAttnConfig::default();
    let mut scratch = VortexScratch::new(256);
    let mut checked = 0;
    for row in fx.rows.iter().step_by(4) {
        let cache = cache_for(&fx, row.layer, row.kv_head, row.n_blocks);
        let via_router = router.forward_indexer(&row.query, &cache, row.n_blocks, row.n_blocks, &mut scratch);
        let direct =
            score_blocks_entmax_into(&row.query, &cache.summaries[..row.n_blocks], &config, &mut scratch.routing_scratch);
        assert_eq!(via_router.blocks.len(), direct.active_indices.len());
        for ((&b, &w), &db) in via_router.blocks.iter().zip(via_router.weights.iter()).zip(direct.active_indices.iter()) {
            assert_eq!(b, db);
            assert_eq!(w.to_bits(), direct.probs[db].to_bits());
        }
        checked += 1;
    }
    assert!(checked > 100, "checked {checked} rows");
}

/// G2: support-vs-n + mass coverage + top-m recall, raw vs scheduled, with
/// the measured table printed (the numbers land in Bench 713's P0.7 addendum).
#[test]
fn p07_g2_support_and_fidelity_on_real_rows() {
    let fx = parse_fixture(FIXTURE);
    let raw = replay(&fx, false);
    let sched = replay(&fx, true);
    assert_eq!(raw.len(), sched.len());

    // bucket rows by n (log-spaced report points)
    let ns: Vec<usize> = {
        let mut v: Vec<usize> = fx.rows.iter().map(|r| r.n_blocks).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    println!("n     |rawS |schdS |raw mass |schd mass |raw recall |schd recall");
    let mut worst_sched_mass = 1f32;
    let mut worst_raw_mass = 1f32;
    for &n in &ns {
        let idx: Vec<usize> = fx.rows.iter().enumerate().filter(|(_, r)| r.n_blocks == n).map(|(i, _)| i).collect();
        let rs: Vec<f32> = idx.iter().map(|&i| raw[i].blocks.len() as f32).collect();
        let ss: Vec<f32> = idx.iter().map(|&i| sched[i].blocks.len() as f32).collect();
        let mass = |arm: &Vec<Decision>| -> f32 {
            idx.iter()
                .map(|&i| {
                    let row = &fx.rows[i];
                    row.masses.iter().enumerate().filter(|(b, _)| arm[i].blocks.contains(b)).map(|(_, m)| m).sum::<f32>()
                })
                .sum::<f32>()
                / idx.len() as f32
        };
        let recall = |arm: &Vec<Decision>| -> f32 {
            let mut tot = 0f32;
            let mut hit = 0f32;
            for &i in &idx {
                let row = &fx.rows[i];
                let m = arm[i].blocks.len().min(row.n_blocks);
                let mut top: Vec<usize> = (0..row.n_blocks).collect();
                top.sort_by(|&a, &b| row.masses[b].total_cmp(&row.masses[a]));
                let truth: std::collections::HashSet<usize> = top[..m].iter().copied().collect();
                for b in &arm[i].blocks {
                    if truth.contains(b) {
                        hit += 1.0;
                    }
                }
                tot += m as f32;
            }
            if tot == 0.0 { 0.0 } else { hit / tot }
        };
        let rm = mass(&raw);
        let sm = mass(&sched);
        worst_raw_mass = worst_raw_mass.min(rm);
        worst_sched_mass = worst_sched_mass.min(sm);
        // weights sanity: a support selection sums within the simplex
        for arm in [&raw, &sched] {
            for &i in idx.iter().take(8) {
                let w: f32 = arm[i].weights.iter().sum();
                assert!(w <= 1.0 + 1e-4, "weights sum {w} > 1");
            }
        }
        println!(
            "{n:5} |{:5.1}|{:5.1} |{rm:8.4} |{sm:8.4}  |{:.4}    |{:.4}",
            mean(&rs),
            mean(&ss),
            recall(&raw),
            recall(&sched)
        );
    }

    // Gates (thresholds from the measured table — see Bench 713 P0.7 addendum):
    // 1. AVERAGED mass coverage: the scheduled arm must not lose more than
    //    0.05 mean oracle mass vs raw across n (measured: within ~0.01; the
    //    per-n table prints the honest per-point deltas, some negative).
    let raw_mean_mass: f32 = {
        let mut tot = 0f32;
        for (row, dec) in fx.rows.iter().zip(raw.iter()) {
            tot += row.masses.iter().enumerate().filter(|(b, _)| dec.blocks.contains(b)).map(|(_, m)| m).sum::<f32>();
        }
        tot / fx.rows.len() as f32
    };
    let sched_mean_mass: f32 = {
        let mut tot = 0f32;
        for (row, dec) in fx.rows.iter().zip(sched.iter()) {
            tot += row.masses.iter().enumerate().filter(|(b, _)| dec.blocks.contains(b)).map(|(_, m)| m).sum::<f32>();
        }
        tot / fx.rows.len() as f32
    };
    println!(
        "mean oracle mass: raw {raw_mean_mass:.4}, scheduled {sched_mean_mass:.4} (Δ {:+.4})",
        sched_mean_mass - raw_mean_mass
    );
    assert!(
        sched_mean_mass >= raw_mean_mass - 0.05,
        "scheduled arm loses mean oracle mass: {sched_mean_mass:.4} vs {raw_mean_mass:.4}"
    );
    // 2. Support sanity at the largest n: neither arm degenerates to a
    //    single block, and the scheduled arm holds ≥ the raw support
    //    (measured: no over-sparsification collapse on real rows at n≤32 —
    //    real σ̂ ≈ 0.14 — and the schedule holds slightly more support).
    let n_max = *ns.last().unwrap();
    let raw_s: Vec<f32> = fx.rows.iter().zip(raw.iter()).filter(|(r, _)| r.n_blocks == n_max).map(|(_, d)| d.blocks.len() as f32).collect();
    let sch_s: Vec<f32> = fx.rows.iter().zip(sched.iter()).filter(|(r, _)| r.n_blocks == n_max).map(|(_, d)| d.blocks.len() as f32).collect();
    let (rm, sm) = (mean(&raw_s), mean(&sch_s));
    println!("largest n={n_max}: raw support {rm:.2}, scheduled support {sm:.2}");
    assert!(rm >= 1.0, "raw support degenerate");
    assert!(sm + 1e-3 >= rm, "scheduled support {sm} below raw {rm}");
    let _ = (worst_raw_mass, worst_sched_mass);
}

/// G2c: needle retention (Bench 032 pattern) — the planted access-code
/// sentence's block. Measured honesty: summary-mean routing is a lossy
/// channel for a single 64-token needle (mean-pooling dilutes it), so BOTH
/// arms retain the oracle-concentrated needle at ~85-95% — the gate pins
/// that measured band rather than a synthetic-harness 100%.
#[test]
fn p07_g2_needle_retention_on_real_rows() {
    let fx = parse_fixture(FIXTURE);
    let raw = replay(&fx, false);
    let sched = replay(&fx, true);
    let needle = fx.needle_block;
    // rows whose oracle top-1 block IS the needle block (the model actually
    // attends there — position > needle block, by construction of capture)
    let mut checked = 0usize;
    let mut raw_kept = 0usize;
    let mut sched_kept = 0usize;
    let mut needle_mass_max = 0f32;
    for (i, row) in fx.rows.iter().enumerate() {
        if row.n_blocks <= needle {
            continue;
        }
        let (top, &_m) = row.masses.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        needle_mass_max = needle_mass_max.max(row.masses[needle]);
        if top != needle {
            continue;
        }
        checked += 1;
        if raw[i].blocks.contains(&top) {
            raw_kept += 1;
        }
        if sched[i].blocks.contains(&top) {
            sched_kept += 1;
        }
    }
    println!(
        "needle block {needle}: {checked} oracle-concentrated rows; kept raw {raw_kept} ({:.1}%), sched {sched_kept} ({:.1}%); max needle mass {needle_mass_max:.3}",
        raw_kept as f64 / checked as f64 * 100.0,
        sched_kept as f64 / checked as f64 * 100.0
    );
    assert!(checked >= 8, "expected needle-concentrated rows, got {checked} (needle block {needle})");
    assert!(
        raw_kept as f32 >= 0.80 * checked as f32,
        "raw arm drops the concentrated needle: {raw_kept}/{checked}"
    );
    assert!(
        sched_kept as f32 >= 0.80 * checked as f32,
        "scheduled arm drops the concentrated needle: {sched_kept}/{checked}"
    );
}

/// G3: no-regression — steady-state scheduled-vs-None router latency on the
/// same real rows. The schedule adds one powf + an O(n) estimator scan per
/// call; bound generously (≤ +50%) while printing the measured delta.
#[test]
fn p07_g3_latency_no_regression_on_real_rows() {
    let fx = parse_fixture(FIXTURE);
    let rows: Vec<&Row> = fx.rows.iter().filter(|r| r.n_blocks >= 16).collect();
    assert!(rows.len() >= 32, "latency rows");
    let caches: Vec<EntmaxCache> = rows.iter().map(|r| cache_for(&fx, r.layer, r.kv_head, r.n_blocks)).collect();

    let run = |scheduled: bool| -> f64 {
        let router = if scheduled {
            EntmaxRouter::default_router().with_asentmax_schedule()
        } else {
            EntmaxRouter::default_router()
        };
        let mut scratch = VortexScratch::new(256);
        // warm-up (estimator EMA + caches + branch predictors)
        for (row, cache) in rows.iter().zip(caches.iter()) {
            let _ = router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
        }
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            for (row, cache) in rows.iter().zip(caches.iter()) {
                let _ = router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        dt / (iters * rows.len()) as f64
    };
    let raw_us = run(false) * 1e6;
    let sched_us = run(true) * 1e6;
    println!("router latency/row: raw {raw_us:.2} µs, scheduled {sched_us:.2} µs (ratio {:.3})", sched_us / raw_us);
    assert!(
        sched_us <= raw_us * 1.5 + 2.0,
        "scheduled arm latency {sched_us:.2} µs vs raw {raw_us:.2} µs"
    );
}

/// Estimator honesty: σ̂ observed from the real rows lands in a plausible
/// band (not the warm-start 1.0, not clamped) — the schedule is acting on a
/// measured statistic of the real routing distribution.
#[test]
fn p07_sigma_hat_reflects_real_rows() {
    let fx = parse_fixture(FIXTURE);
    let router = EntmaxRouter::default_router().with_asentmax_schedule();
    let est = router.asentmax.as_ref().expect("estimator");
    let mut caches: std::collections::HashMap<(usize, usize), EntmaxCache> = Default::default();
    let mut scratch = VortexScratch::new(256);
    for row in fx.rows.iter().step_by(3) {
        let key = (row.layer, row.kv_head);
        let cache = caches.entry(key).or_insert_with(|| {
            let n = fx.streams.iter().find(|s| s.layer == row.layer && s.kv_head == row.kv_head).map_or(row.n_blocks, |s| s.n_blocks);
            cache_for(&fx, row.layer, row.kv_head, n)
        });
        if cache.summaries.len() < row.n_blocks {
            *cache = cache_for(&fx, row.layer, row.kv_head, row.n_blocks);
        }
        let _ = router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
    }
    let sigma = est.resolve_sigma();
    println!("rolling σ̂ after replay: {sigma:.4}");
    // Measured regime (Bench 713 P0.7): real Bonsai-8B routing logits have
    // σ̂ ≈ 0.14 — an order below the σ ≥ 1 over-sparsification regime the
    // synthetic harness tested. The band pins the measurement (a fixture
    // drift that changes the routing scale moves σ̂ out of it).
    assert!((0.05..=0.5).contains(&sigma), "σ̂ {sigma} outside the measured real-routing band [0.05, 0.5]");
    // The EMA observed real rows: it must differ from a virgin estimator.
    let virgin = RollingSigmaEstimator::new(0.8);
    assert_ne!(sigma.to_bits(), virgin.resolve_sigma().to_bits(), "estimator never observed anything");
}
