//! Issue 693 — support-regime GOAT gate (bench 685) + the pre-registered
//! T3 falsifiable PoC.
//!
//! # The toy world (LpWM `Piecewise` shape, fully deterministic)
//!
//! 2D room `[0,1]²`, `K = 4` quadrant zones. Each entity dwells + wiggles
//! within a zone (wiggle margin-bounded so dwells cannot cross), then
//! crosses to an adjacent zone (base step toward the target + the same
//! wiggle — the ONLY chatter source; a crossing may flip the zone id more
//! than once near the boundary, which is emergent, not engineered).
//! Latent proxy: `z = clamp0(W_zone · [x, y, 1])`, `D = 64`, per-zone
//! non-negative LCG-seeded `W`: dims `0..32` shared background (weak,
//! active in all zones), dims `32..64` four 8-dim per-zone primary blocks
//! (zone `k` owns block `k` with strong weights — the issue sketch's 16-dim
//! blocks do not close arithmetically at the pinned `D = 64`; 4×8 = 32
//! preserves the half-background structure) — a transition swaps the
//! primary mass, within-zone motion only varies magnitudes.
//!
//! # Pre-registered gate (stated BEFORE evaluation — one run at defaults)
//!
//! - **Detect**: ≥ 90% of zone-transition episodes detected (detector fire)
//!   within ≤ 2 ticks of the episode's FIRST zone-change tick.
//! - **False-fire**: ≤ 10% of fires outside every ±2-tick window around any
//!   zone-change tick.
//! - Population: ≥ 32 entities × 2000 ticks.
//! - Detector: pre-registered defaults (`THETA_FIRE = 0.6`,
//!   `THETA_CALM = 0.35`, window = 3).
//!
//! **Assertion policy (pre-registered):** the quality verdict is PRINTED
//! and recorded in `.benchmarks/685_support_regime_goat.md` — detect-rate
//! is an experimental result, NOT an asserted invariant (the issue's own
//! negative-result path). The safety axis (false-fire ≤ 10%) IS asserted —
//! it is structural (within-zone instability sits ~20× under `THETA_CALM`)
//! and a regression there is a real defect. G1/G2/G4 are asserted
//! invariants. The post-hoc sensitivity table (≤ 4 points, labeled) is
//! diagnostic only — NOT the gate verdict.
//!
//! Run:
//!
//! ```bash
//! cargo test -p katgpt-core --test bench_685_support_regime_goat \
//!   --features support_regime --release -- --nocapture
//! ```

#![cfg(feature = "functional_substitution_gate")]

use katgpt_core::functional_substitution::support_instability::{
    DetectorState, SupportInstabilityDetector, THETA_CALM, THETA_FIRE, support_instability,
};
use std::hint::black_box;
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// Toy-world constants (all pre-registered)
// ──────────────────────────────────────────────────────────────────────────

const TICKS: usize = 2000;
const POC_ENTITIES: usize = 32;
const G2_ENTITIES: usize = 64;
const D: usize = 64;
const N_ZONES: usize = 4;

/// Entity seed base (per-entity seed = `ENTITY_SEED_BASE + entity`).
const ENTITY_SEED_BASE: u64 = 0x0693_0001;
/// Zone-weight seed (fixed for the whole world).
const W_SEED: u64 = 0x0693_5131;

/// Dwell length range per episode (ticks) → ~8–16 transitions / 2000 ticks.
const DWELL_MIN: u32 = 120;
const DWELL_SPAN: u32 = 101; // 120..=220

/// Crossing base step length (per tick, toward the target).
const CROSS_STEP: f32 = 0.03;
/// Wiggle step length (per tick, random direction; dwell + crossing).
const WIGGLE: f32 = 0.02;
/// Home margin: dwell homes sit ≥ this from every zone boundary, and the
/// dwell wiggle is clamped to the zone's inner box → dwells never cross.
const HOME_MARGIN: f32 = 0.08;
/// Crossing arrival distance to the target home.
const ARRIVE: f32 = 0.05;

// ──────────────────────────────────────────────────────────────────────────
// Deterministic LCG (the whole world is LCG-only — G1 bit-identity by
// construction, no HashMap iteration order anywhere)
// ──────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Decorrelate nearby seeds (entity 0 vs 1) before the first draw.
        Self(
            seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0x1234_5678_9ABC_DEF0),
        )
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 65536.0 / 65536.0
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// World: per-zone weights + latent projection
// ──────────────────────────────────────────────────────────────────────────

type ZoneW = [[f32; 3]; D];

/// Per-zone non-negative weights: dims `0..32` background (`0.01..=0.15`,
/// active in all zones), zone `k` owns primary dims `32 + 8k .. 32 + 8k + 8`
/// (`0.5..=2.0`), everything else exactly zero. NOTE: the issue sketch said
/// 16-dim blocks, whose arithmetic (32 bg + 4×16 = 96) does not close at
/// the pinned `D = 64`; 8-dim blocks preserve the D=64 + half-background
/// structure (4 × 8 = 32 primary dims).
fn gen_zone_weights() -> [ZoneW; N_ZONES] {
    let mut rng = Lcg::new(W_SEED);
    let mut w = [[[0.0f32; 3]; D]; N_ZONES];
    for (z, wz) in w.iter_mut().enumerate() {
        for (i, wi) in wz.iter_mut().enumerate() {
            let (lo, hi) = match i {
                0..=31 => (0.01, 0.15),
                _ if i >= 32 && (i - 32) / 8 == z => (0.5, 2.0),
                _ => (0.0, 0.0),
            };
            for wk in wi.iter_mut() {
                *wk = rng.range(lo, hi);
            }
        }
    }
    w
}

/// `z = clamp0(W_zone · [x, y, 1])` — the affine position dependence is
/// what varies magnitudes within a zone.
fn project_into(z: &mut [f32; D], w: &ZoneW, x: f32, y: f32) {
    for (zi, wi) in z.iter_mut().zip(w.iter()) {
        *zi = (wi[0] * x + wi[1] * y + wi[2]).max(0.0);
    }
}

fn zone_of(x: f32, y: f32) -> u8 {
    (x >= 0.5) as u8 + 2 * (y >= 0.5) as u8
}

/// Adjacent (edge-sharing) zone in the quadrant grid — never diagonal.
fn zone_center(zone: u8) -> (f32, f32) {
    match zone {
        z @ (0..=3) => (0.25 + 0.5 * (z % 2) as f32, 0.25 + 0.5 * (z / 2) as f32),
        _ => (0.25, 0.25),
    }
}

fn home_in(zone: u8, rng: &mut Lcg) -> (f32, f32) {
    let (cx, cy) = zone_center(zone);
    let m = HOME_MARGIN;
    (rng.range(cx - m, cx + m), rng.range(cy - m, cy + m))
}

// ──────────────────────────────────────────────────────────────────────────
// Trajectory generation
// ──────────────────────────────────────────────────────────────────────────

struct EntityTrace {
    xs: Vec<f32>,
    ys: Vec<f32>,
    zones: Vec<u8>,
    /// Every tick `t ≥ 1` with `zones[t] != zones[t-1]`.
    change_ticks: Vec<usize>,
    /// First change tick of each crossing episode.
    episode_first: Vec<usize>,
}

fn gen_trace(seed: u64) -> EntityTrace {
    let mut rng = Lcg::new(seed);
    let mut xs = Vec::with_capacity(TICKS);
    let mut ys = Vec::with_capacity(TICKS);
    let mut zones = Vec::with_capacity(TICKS);
    let mut change_ticks = Vec::new();
    let mut episode_first = Vec::new();

    let mut zone = (rng.next_u32() & 3) as u8;
    let (mut x, mut y) = home_in(zone, &mut rng);
    push_tick(&mut xs, &mut ys, &mut zones, x, y);

    while xs.len() < TICKS {
        // ── dwell: margin-clamped wiggle (cannot leave the zone) ──
        let dwell = DWELL_MIN + rng.next_u32() % DWELL_SPAN;
        let (zx0, zx1, zy0, zy1) = zone_inner_box(zone);
        for _ in 0..dwell {
            if xs.len() >= TICKS {
                break;
            }
            let (dx, dy) = wiggle_step(&mut rng);
            x = (x + dx).clamp(zx0, zx1);
            y = (y + dy).clamp(zy0, zy1);
            push_tick(&mut xs, &mut ys, &mut zones, x, y);
        }
        if xs.len() >= TICKS {
            break;
        }

        // ── crossing: base step toward the target + free wiggle ──
        let target = adjacent_zone_of(zone, rng.next_u32());
        let (tx, ty) = home_in(target, &mut rng);
        let mut first_change: Option<usize> = None;
        while xs.len() < TICKS {
            let dx = tx - x;
            let dy = ty - y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < ARRIVE {
                break;
            }
            let (wx, wy) = wiggle_step(&mut rng);
            x = (x + CROSS_STEP / dist * dx + wx).clamp(0.0, 1.0);
            y = (y + CROSS_STEP / dist * dy + wy).clamp(0.0, 1.0);
            push_tick(&mut xs, &mut ys, &mut zones, x, y);
            let t = xs.len() - 1;
            if zones[t] != zones[t - 1] && first_change.is_none() {
                first_change = Some(t);
            }
        }
        if let Some(t) = first_change {
            episode_first.push(t);
        }
        zone = target;
    }

    // Full change-tick list (every flip — the false-fire window set).
    for t in 1..zones.len() {
        if zones[t] != zones[t - 1] {
            change_ticks.push(t);
        }
    }

    EntityTrace {
        xs,
        ys,
        zones,
        change_ticks,
        episode_first,
    }
}

fn push_tick(xs: &mut Vec<f32>, ys: &mut Vec<f32>, zones: &mut Vec<u8>, x: f32, y: f32) {
    xs.push(x);
    ys.push(y);
    zones.push(zone_of(x, y));
}

fn wiggle_step(rng: &mut Lcg) -> (f32, f32) {
    let theta = rng.unit() * std::f32::consts::TAU;
    (WIGGLE * theta.cos(), WIGGLE * theta.sin())
}

fn zone_inner_box(zone: u8) -> (f32, f32, f32, f32) {
    let (x0, x1) = match zone % 2 {
        0 => (HOME_MARGIN, 0.5 - HOME_MARGIN),
        _ => (0.5 + HOME_MARGIN, 1.0 - HOME_MARGIN),
    };
    let (y0, y1) = match zone / 2 {
        0 => (HOME_MARGIN, 0.5 - HOME_MARGIN),
        _ => (0.5 + HOME_MARGIN, 1.0 - HOME_MARGIN),
    };
    (x0, x1, y0, y1)
}

/// Correct adjacency: quadrants 0,1,2,3 — 0↔1 (vertical boundary),
/// 0↔2 (horizontal), 1↔3 (horizontal), 2↔3 (vertical).
fn adjacent_zone_of(zone: u8, pick: u32) -> u8 {
    match zone {
        0 => [1u8, 2][pick as usize & 1],
        1 => [0u8, 3][pick as usize & 1],
        2 => [0u8, 3][pick as usize & 1],
        _ => [1u8, 2][pick as usize & 1],
    }
}

// ──────────────────────────────────────────────────────────────────────────
// PoC runner + metrics
// ──────────────────────────────────────────────────────────────────────────

struct PocMetrics {
    episodes: usize,
    episodes_detected: usize,
    /// Diagnostic (not the gate): episodes whose FIRST-change-tick
    /// instability exceeds 0.5 — the raw un-debounced signal quality.
    raw_spiked: usize,
    fires: usize,
    false_fires: usize,
    /// Sum of (fire_tick − first_change_tick) over detected episodes.
    latency_sum: i64,
    fires_timeline: Vec<(usize, usize)>,
    /// Per-entity instability streams, bit-packed for G1.
    inst_bits: Vec<u32>,
}

fn near_any(tick: usize, ticks: &[usize], tol: i64) -> bool {
    ticks
        .iter()
        .any(|&gt| (tick as i64 - gt as i64).abs() <= tol)
}

/// Run the full PoC (generation + detection) for `n_entities` at the given
/// detector config. Fully deterministic per `(n_entities, config)`.
fn run_poc(n_entities: usize, cfg: Option<(f32, f32, usize)>) -> PocMetrics {
    let w = gen_zone_weights();
    let mut fires_timeline = Vec::new();
    let mut inst_bits = Vec::with_capacity(n_entities);
    let mut episodes = 0usize;
    let mut episodes_detected = 0usize;
    let mut raw_spiked = 0usize;
    let mut fires = 0usize;
    let mut false_fires = 0usize;
    let mut latency_sum = 0i64;

    let mut z_prev = [0.0f32; D];
    let mut z_cur = [0.0f32; D];

    for entity in 0..n_entities {
        let trace = gen_trace(ENTITY_SEED_BASE + entity as u64);
        let mut det = match cfg {
            Some((tf, tc, win)) => SupportInstabilityDetector::with_params(tf, tc, win),
            None => SupportInstabilityDetector::new(),
        };
        let mut entity_fires: Vec<usize> = Vec::new();
        let mut inst_stream: Vec<f32> = Vec::with_capacity(TICKS);

        project_into(
            &mut z_prev,
            &w[trace.zones[0] as usize],
            trace.xs[0],
            trace.ys[0],
        );
        inst_stream.push(0.0); // tick 0: no previous state.
        for t in 1..trace.xs.len() {
            project_into(
                &mut z_cur,
                &w[trace.zones[t] as usize],
                trace.xs[t],
                trace.ys[t],
            );
            let inst = support_instability(&z_prev, &z_cur);
            inst_stream.push(inst);
            let prev_state = det.state();
            let state = det.push(inst);
            if prev_state == DetectorState::Calm && state == DetectorState::Firing {
                entity_fires.push(t);
            }
            z_prev.copy_from_slice(&z_cur);
        }

        // Metrics: detect = fire within ±2 of an episode's FIRST change tick.
        for &gt in &trace.episode_first {
            episodes += 1;
            let detected = entity_fires
                .iter()
                .any(|&f| (f as i64 - gt as i64).abs() <= 2);
            if detected {
                episodes_detected += 1;
                // Earliest qualifying fire — the detection latency.
                if let Some(&f) = entity_fires
                    .iter()
                    .find(|&f| ((*f as i64) - (gt as i64)).abs() <= 2)
                {
                    latency_sum += f as i64 - gt as i64;
                }
            }
            // Raw signal diagnostic: the instability AT the first flip.
            if inst_stream[gt] > 0.5 {
                raw_spiked += 1;
            }
        }
        for &f in &entity_fires {
            fires += 1;
            if !near_any(f, &trace.change_ticks, 2) {
                false_fires += 1;
            }
        }
        for v in &inst_stream {
            inst_bits.push(v.to_bits());
        }
        fires_timeline.extend(entity_fires.iter().map(|&t| (entity, t)));
    }

    PocMetrics {
        episodes,
        episodes_detected,
        raw_spiked,
        fires,
        false_fires,
        latency_sum,
        fires_timeline,
        inst_bits,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T3 — the pre-registered run (one shot; verdict printed, recorded in the
// bench note; quality axes not asserted — see the file header)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn t3_poc_pre_registered_run() {
    let m = run_poc(POC_ENTITIES, None);
    let detect_rate = m.episodes_detected as f64 / m.episodes as f64;
    let raw_rate = m.raw_spiked as f64 / m.episodes as f64;
    let false_rate = m.false_fires as f64 / m.fires.max(1) as f64;
    let mean_latency = if m.episodes_detected > 0 {
        m.latency_sum as f64 / m.episodes_detected as f64
    } else {
        f64::NAN
    };

    println!("┌─ T3 pre-registered PoC (bench 685) ─────────────────────────");
    println!("│ entities           : {POC_ENTITIES} × {TICKS} ticks");
    println!("│ detector           : θ_fire={THETA_FIRE} θ_calm={THETA_CALM} window=3 (defaults)");
    println!("│ episodes           : {}", m.episodes);
    println!(
        "│ detected (≤2 ticks): {} / {} = {:.1}%",
        m.episodes_detected,
        m.episodes,
        100.0 * detect_rate
    );
    println!("│ mean latency       : {mean_latency:.2} ticks");
    println!(
        "│ raw signal >0.5 @ flip (diagnostic, not the gate): {:.1}%",
        100.0 * raw_rate
    );
    println!(
        "│ fires              : {} (false {} → {:.2}%)",
        m.fires,
        m.false_fires,
        100.0 * false_rate
    );
    println!(
        "│ GATE detect ≥ 90%  : {}",
        if detect_rate >= 0.90 { "PASS" } else { "FAIL" }
    );
    println!(
        "│ GATE false-fire ≤10%: {}",
        if false_rate <= 0.10 { "PASS" } else { "FAIL" }
    );
    println!("└─────────────────────────────────────────────────────────────");

    // Structural sanity (generator invariants — NOT quality verdicts).
    assert_eq!(POC_ENTITIES, 32, "pre-registered population");
    assert!(
        m.episodes >= POC_ENTITIES * 5,
        "expected ≥5 episodes/entity, got {}",
        m.episodes
    );
    assert!(
        m.episodes <= POC_ENTITIES * 20,
        "expected ≤20 episodes/entity, got {}",
        m.episodes
    );

    // Safety axis (structural: within-zone instability ~20× under θ_calm).
    assert!(
        false_rate <= 0.10,
        "false-fire rate {false_rate:.3} exceeds the 10% safety axis"
    );
}

/// Post-hoc sensitivity table (≤ 4 points, DIAGNOSTIC ONLY — the gate
/// verdict is the pre-registered defaults run above). Same streams, same
/// everything except the detector config.
#[test]
fn t3_post_hoc_sensitivity_table() {
    println!("┌─ post-hoc sensitivity (NOT the gate verdict) ───────────────");
    println!(
        "│ {:<22} {:>9} {:>9} {:>8}",
        "config", "detect%", "false%", "fires"
    );
    for (tf, tc, win) in [
        (THETA_FIRE, THETA_CALM, 3usize),
        (0.30, 0.15, 3),
        (0.50, 0.20, 1),
        (0.30, 0.15, 2),
    ] {
        let m = run_poc(POC_ENTITIES, Some((tf, tc, win)));
        let detect = 100.0 * m.episodes_detected as f64 / m.episodes as f64;
        let false_r = 100.0 * m.false_fires as f64 / m.fires.max(1) as f64;
        println!(
            "│ θf={:<4} θc={:<4} w={:<2}   {:>8.1} {:>8.1} {:>8}",
            tf, tc, win, detect, false_r, m.fires
        );
    }
    println!("└─────────────────────────────────────────────────────────────");
}

// ──────────────────────────────────────────────────────────────────────────
// G1 — determinism (bit-identical PoC timelines across independent runs)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn g1_poc_determinism_bit_identical() {
    let a = run_poc(POC_ENTITIES, None);
    let b = run_poc(POC_ENTITIES, None);
    assert_eq!(
        a.fires_timeline, b.fires_timeline,
        "fire timelines differ across independent runs"
    );
    assert_eq!(
        a.inst_bits, b.inst_bits,
        "instability streams differ at the bit level"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// G2 — perf (release-locked; debug runs are skipped)
// ──────────────────────────────────────────────────────────────────────────

fn best_of_3<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

/// G2: mean ns/entity/tick over the full detector path (iou at D=64 + ring
/// update + fire logic), ≥ 64 entities × 2000 ticks. Budget: < 100 ns.
#[cfg_attr(debug_assertions, ignore)]
#[test]
fn g2_support_regime_under_100ns_per_entity_tick() {
    let w = gen_zone_weights();

    // Pre-generate the latent streams ONCE (allocation outside the timed
    // region): entity-major [entity][tick][dim].
    let traces: Vec<EntityTrace> = (0..G2_ENTITIES)
        .map(|e| gen_trace(ENTITY_SEED_BASE + e as u64))
        .collect();
    let mut streams: Vec<[f32; D]> = Vec::with_capacity(G2_ENTITIES * TICKS);
    let mut z = [0.0f32; D];
    for tr in &traces {
        for t in 0..tr.xs.len() {
            project_into(&mut z, &w[tr.zones[t] as usize], tr.xs[t], tr.ys[t]);
            streams.push(z);
        }
    }

    let total_ticks: usize = traces.iter().map(|t| t.xs.len()).sum();
    let mut dets: Vec<SupportInstabilityDetector> = (0..G2_ENTITIES)
        .map(|_| SupportInstabilityDetector::new())
        .collect();
    let mut sink = 0u64;

    // Warmup (nothing lazy expected — but cheap insurance).
    {
        let inst = support_instability(&streams[0], &streams[1]);
        sink += dets[0].push(inst) as u8 as u64;
    }

    // Per pass: every entity, ticks 1..TICKS (tick 0 has no previous).
    let per_pass = total_ticks - G2_ENTITIES;
    let ns_per_entity_tick = best_of_3(1, || {
        for d in dets.iter_mut() {
            *d = SupportInstabilityDetector::new();
        }
        let mut local = 0u64;
        for (e, det) in dets.iter_mut().enumerate() {
            let base = e * TICKS;
            for t in 1..TICKS {
                let inst = support_instability(
                    black_box(&streams[base + t - 1]),
                    black_box(&streams[base + t]),
                );
                local += det.push(inst) as u8 as u64;
            }
        }
        sink = sink.wrapping_add(local);
    }) / per_pass as f64;

    black_box(&sink);
    println!(
        "G2: support-regime detector path = {ns_per_entity_tick:.2} ns/entity/tick \
         ({G2_ENTITIES} entities × {TICKS} ticks, D={D}; budget 100 ns)"
    );
    assert!(
        ns_per_entity_tick < 100.0,
        "{ns_per_entity_tick:.2} ns/entity/tick exceeds the 100 ns budget"
    );
}

// ──────────────────────────────────────────────────────────────────────
// Cousin cost table (release-locked; KARC measured on the same streams,
// stiff_anomaly + ICT cited-not-measured — see the bench note).
// G4 lives in the separate single-test binary
// bench_685_support_regime_alloc_check.rs (the CountingAllocator
// convention — parallel tests in THIS binary allocate and would pollute
// the global counter).
// ──────────────────────────────────────────────────────────────────────────

#[cfg(not(debug_assertions))]
mod cousin {
    use super::*;
    use katgpt_core::{FourierBasis, KarcForecaster};

    /// Fit one KARC forecaster on the first `n_train` ticks of an entity's
    /// latent stream (delay windows assembled exactly like the Plan 556
    /// bench). λ is raised to 1e-2 for numerical stability on the
    /// rank-deficient toy features — fit QUALITY is not the claim here,
    /// cost is.
    fn make_fitted_stream_forecaster<const D: usize, const M: usize, const K: usize>(
        stream: &[[f32; D]],
        n_train: usize,
    ) -> KarcForecaster<FourierBasis<M>, D, M, K> {
        let mut f = KarcForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
            FourierBasis::new(4.0),
            n_train,
        );
        let kd = K * D;
        let n = stream.len().min(n_train);
        for t in (K - 1)..(n - 1) {
            let mut delay = vec![0.0f32; kd];
            for lag in 0..K {
                delay[lag * D..(lag + 1) * D].copy_from_slice(&stream[t - lag]);
            }
            let target: [f32; D] = stream[t + 1];
            f.accumulate_pair(&delay, &target);
        }
        f.fit_ridge(1e-2).expect("fit_ridge on toy stream");
        f
    }

    /// The honest KARC per-tick loop: forecast û_t from the ring (which
    /// holds up to x_{t−1}), observe x_t, surprise = ‖x_t − û_t‖₂.
    /// Returns (ns/entity/tick, accumulated surprise).
    fn karc_arm<const D: usize, const M: usize, const K: usize>(
        streams: &[Vec<[f32; D]>],
        warm: usize,
        n_train: usize,
    ) -> (f64, f32) {
        let mut forecasters: Vec<_> = streams
            .iter()
            .map(|s| make_fitted_stream_forecaster::<D, M, K>(s, n_train))
            .collect();
        let total_ticks: usize = streams.iter().map(|s| s.len() - warm).sum();
        let mut out = vec![[0.0f32; D]; streams.len()];
        let mut sink = 0.0f32;

        let t0 = Instant::now();
        for (i, s) in streams.iter().enumerate() {
            for t in warm..s.len() {
                let ok = forecasters[i].forecast_now(&mut out[i]);
                let obs = s[t];
                forecasters[i].observe(&obs);
                if ok {
                    let mut acc = 0.0f32;
                    for (u, o) in out[i].iter().zip(obs.iter()) {
                        acc += (u - o) * (u - o);
                    }
                    sink += acc.sqrt();
                }
            }
        }
        let ns = t0.elapsed().as_nanos() as f64;
        (ns / total_ticks as f64, sink)
    }

    #[test]
    fn cousin_cost_table() {
        let w = gen_zone_weights();

        // Same latent streams for every arm (the T3 fixture, 8 entities —
        // KARC's fit is a one-off cost we do NOT hide, but the table row is
        // per-tick).
        const N: usize = 8;
        let mut streams64: Vec<Vec<[f32; 64]>> = Vec::with_capacity(N);
        let mut z = [0.0f32; D];
        for e in 0..N {
            let trace = gen_trace(ENTITY_SEED_BASE + e as u64);
            let mut s = Vec::with_capacity(trace.xs.len());
            for t in 0..trace.xs.len() {
                project_into(
                    &mut z,
                    &w[trace.zones[t] as usize],
                    trace.xs[t],
                    trace.ys[t],
                );
                s.push(z);
            }
            streams64.push(s);
        }
        // D=8 slice streams for the canonical HLA-shaped KARC config
        // (cost-only arm: consumes dims 0..8 — the background block).
        let streams8: Vec<Vec<[f32; 8]>> = streams64
            .iter()
            .map(|s| s.iter().map(|z| z[..8].try_into().unwrap()).collect())
            .collect();

        // Arm (a): support-instability over the same D=64 streams.
        {
            let per = TICKS - 1;
            let mut dets: Vec<_> = (0..N).map(|_| SupportInstabilityDetector::new()).collect();
            let t0 = Instant::now();
            let mut sink = 0u64;
            for (i, s) in streams64.iter().enumerate() {
                for t in 1..s.len() {
                    let inst = support_instability(black_box(&s[t - 1]), black_box(&s[t]));
                    sink += dets[i].push(inst) as u8 as u64;
                }
            }
            let ns = t0.elapsed().as_nanos() as f64;
            black_box(&sink);
            println!(
                "cousin │ support-instability D=64    : {:>10.1} ns/entity/tick",
                ns / (N * per) as f64
            );
        }

        // Arm (b): KARC same-fixture D=64/M=8/K=4 (d_h = 2048).
        {
            let (ns, surp) = karc_arm::<64, 8, 4>(&streams64, 400, 400);
            println!(
                "cousin │ KARC D=64 M=8 K=4 (d_h 2048): {ns:>10.1} ns/entity/tick (surprise Σ {surp:.1})"
            );
        }

        // Arm (c): KARC canonical HLA shape D=8/M=8/K=4 (d_h = 256) on the
        // 8-dim slice of the same streams.
        {
            let (ns, surp) = karc_arm::<8, 8, 4>(&streams8, 400, 400);
            println!(
                "cousin │ KARC D=8  M=8 K=4 (d_h 256) : {ns:>10.1} ns/entity/tick (8-dim slice; surprise Σ {surp:.1})"
            );
        }

        // Arm (d) + (e): cited-not-measured rows — katgpt-spectral is a
        // DOWNSTREAM crate (no dev-dep here); its own GOAT (bench 037) is
        // correctness-only with no per-tick latency. ICT lives in riir-ai.
        println!(
            "cousin │ stiff_anomaly (katgpt-spectral): cited-not-measured — eigendecomp + window vs frozen baseline (bench 037 has no latency axis)"
        );
        println!(
            "cousin │ ICT branching (riir-ai)        : cited-not-measured — JS-divergence over K sampled action dists, K samples/tick (R513 §Path-0)"
        );
    }
}
