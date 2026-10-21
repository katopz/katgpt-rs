//! Issue 952 T4 — the bet-search GOAT bench. Run with `--release`:
//!
//! ```text
//! cargo test -p katgpt-extract --features bet_search_puct --release \
//!     --test bench_952_bet_search_goat -- --nocapture --test-threads=1
//! ```
//!
//! Two axes: search budget b ∈ {16, 64, 256} × comps size n ∈ {3, 5, 10,
//! 30, 60}, over five synthetic estimate worlds, each built so a DIFFERENT
//! authored bet is the right answer (ratio, plane, smooth, flat, drift).
//! Every arm runs through the same `BetSearchHead` (same split, same
//! calibration-split conformal overlay), so the arms differ only in how the
//! composition is chosen:
//!
//! * (a) `floor` — the conformal-naive floor (`Fixed { mask: 0 }`).
//! * `incumb` — the incumbent `estimate::EstimateHead` (kNN k=3, the
//!   bench-628 posture; jackknife+ below 8 comps) at the same α.
//! * (b) `best1` — the best single authored bet (the 7 singletons).
//! * (c) `knn5` — plain kNN k=5, no search.
//! * (d) `exh@E` — exhaustive canonical search at the SAME number of
//!   distinct compositions PUCT scored on that task.
//! * `puct@b` — `katgpt_core::chance_puct` at budget b.
//! * `full` — the whole 63-composition space.
//! * `served` — `BetSearchConfig::default()`: full exhaustive search with
//!   the floor demotion armed (the shipped posture).
//!
//! Scores are on HELD-OUT queries (never comps): mean interval-CRPS,
//! coverage of the 80% interval, and the scale-free skill vs the floor,
//! `(CRPS_floor − CRPS_arm) / CRPS_floor`, averaged per task. Promotion per
//! Issue 952: the arm must beat the floor AND PUCT must beat exhaustive at
//! the equal budget — otherwise exhaustive ships.

#![cfg(feature = "bet_search_puct")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::chance_puct::ChancePuctConfig;
use katgpt_core::conformal::PredictiveInterval;
use katgpt_core::conformal::metrics::{crps_interval, winkler_score};
use katgpt_extract::bet_search::{BetPosture, BetSearchConfig, BetSearchHead, SearchMode};
use katgpt_extract::estimate::{Comparable, EstimateHead, InMemorySource};

const TASKS: u64 = 100;
const QUERIES: usize = 50;
const NS: [usize; 5] = [3, 5, 10, 30, 60];
const BUDGETS: [u32; 3] = [16, 64, 256];
const ALPHA: f32 = 0.2;

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    /// Approximately standard normal (Irwin–Hall, 4 uniforms).
    fn gauss(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732
    }
}

#[derive(Clone, Copy, Debug)]
enum World {
    Ratio,
    Plane,
    Smooth,
    Flat,
    Drift,
}

const WORLDS: [World; 5] = [
    World::Ratio,
    World::Plane,
    World::Smooth,
    World::Flat,
    World::Drift,
];

impl World {
    fn row(self, rng: &mut SplitMix, t: usize) -> Comparable {
        let x0 = rng.range(1.0, 20.0);
        let x1 = rng.range(1.0, 10.0);
        let x2 = rng.unit();
        let g = rng.gauss();
        let value = match self {
            Self::Ratio => 3.0 * x0 * (1.0 + 0.08 * g),
            Self::Plane => 2.0 * x0 + 5.0 * x1 + g,
            Self::Smooth => 10.0 + 4.0 * (x0 * 0.6).sin() + 0.3 * g,
            Self::Flat => 50.0 + 2.0 * g,
            Self::Drift => 20.0 + 0.5 * t as f32 + 0.5 * g,
        };
        Comparable {
            features: vec![x0, x1, x2],
            value,
        }
    }

    /// `n` comps (t = 0..n) + `QUERIES` held-out rows (t = n..), drawn from
    /// one seed — the queries continue the drift world's clock.
    fn task(self, n: usize, seed: u64) -> (Vec<Comparable>, Vec<Comparable>) {
        let mut rng = SplitMix(seed ^ (self as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407));
        let comps = (0..n).map(|t| self.row(&mut rng, t)).collect();
        let queries = (0..QUERIES).map(|q| self.row(&mut rng, n + q)).collect();
        (comps, queries)
    }
}

#[derive(Clone, Copy, Default)]
struct Score {
    crps: f64,
    winkler: f64,
    covered: f64,
    count: f64,
}

/// Run one head over a task's held-out queries.
fn score(head: &mut BetSearchHead, queries: &[Comparable]) -> Score {
    let mut s = Score::default();
    let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, ALPHA);
    for q in queries {
        head.estimate_into(&q.features, &mut iv);
        s.crps += f64::from(crps_interval(&iv, q.value));
        s.winkler += f64::from(winkler_score(&iv, q.value));
        s.covered += f64::from(u8::from(iv.contains(q.value)));
        s.count += 1.0;
    }
    s
}

fn head(mode: SearchMode, demote: bool) -> BetSearchHead {
    BetSearchHead::new(BetSearchConfig {
        mode,
        alpha: ALPHA,
        demote,
        ..BetSearchConfig::default()
    })
}

fn puct(budget: u32, seed: u64) -> SearchMode {
    SearchMode::Puct {
        cfg: ChancePuctConfig {
            budget,
            ..ChancePuctConfig::default()
        },
        seed,
    }
}

/// Per-cell accumulator for one arm.
#[derive(Clone, Copy, Default)]
struct Arm {
    crps: f64,
    winkler_skill: f64,
    covered: f64,
    queries: f64,
    skill: f64,
    tasks: f64,
    beats_floor: f64,
}

impl Arm {
    fn add(&mut self, s: Score, floor: Score) {
        let (c, f) = (s.crps / s.count, floor.crps / floor.count);
        self.crps += s.crps;
        self.covered += s.covered;
        self.queries += s.count;
        self.skill += if f > 0.0 { (f - c) / f } else { 0.0 };
        let (cw, fw) = (s.winkler / s.count, floor.winkler / floor.count);
        self.winkler_skill += if fw > 0.0 { (fw - cw) / fw } else { 0.0 };
        self.tasks += 1.0;
        self.beats_floor += f64::from(u8::from(c < f));
    }
    fn skill(&self) -> f64 {
        self.skill / self.tasks
    }
    fn coverage(&self) -> f64 {
        self.covered / self.queries
    }
}

const ARMS: [&str; 10] = [
    "floor", "incumb", "best1", "knn5", "full", "puct16", "exh@E16", "puct64", "exh@E64", "served",
];

/// The incumbent estimate head over the same comps and queries.
fn incumbent(comps: &[Comparable], queries: &[Comparable]) -> Score {
    let mut source = InMemorySource::new();
    source.insert_table("f", comps.to_vec());
    let mut h = EstimateHead::new(3, ALPHA, None);
    let mut s = Score::default();
    for q in queries {
        let (d, _) = h.estimate_with_posture(&q.features, &source, "f");
        let iv = PredictiveInterval::new(d.p10 as f32, d.point as f32, d.p90 as f32, ALPHA);
        s.crps += f64::from(crps_interval(&iv, q.value));
        s.winkler += f64::from(winkler_score(&iv, q.value));
        s.covered += f64::from(u8::from(iv.contains(q.value)));
        s.count += 1.0;
    }
    s
}

#[test]
#[ignore = "GOAT bench — run in --release with --nocapture (see the module doc)"]
fn bench_952_bet_search_goat() {
    println!(
        "Issue 952 T4 — bet-search GOAT bench ({TASKS} tasks × {QUERIES} held-out queries per cell, α={ALPHA})"
    );
    // Paired PUCT vs exhaustive@E, per budget, pooled over worlds × n ≥ 10.
    let mut duel = [(0u32, 0u32, 0u32, 0.0f64, 0.0f64, 0u64); BUDGETS.len()];
    // Pooled skill per (n, arm) across worlds.
    let mut pooled = vec![[Arm::default(); ARMS.len()]; NS.len()];
    let mut served_floor_rate = vec![0.0f64; NS.len()];

    for world in WORLDS {
        println!("\n## world {world:?}");
        println!(
            "{:>4} | {}",
            "n",
            ARMS.map(|a| format!("{a:>17}")).join(" | ")
        );
        for (ni, &n) in NS.iter().enumerate() {
            let mut arms = [Arm::default(); ARMS.len()];
            for t in 0..TASKS {
                let (comps, queries) = world.task(n, 0x952_0000 + t);
                let run = |mode: SearchMode, demote: bool| {
                    let mut h = head(mode, demote);
                    h.calibrate(&comps);
                    let rec = *h.record().unwrap();
                    (score(&mut h, &queries), rec)
                };
                let (floor, _) = run(SearchMode::Fixed { mask: 0 }, false);
                let (best1, _) = run(SearchMode::Exhaustive { evaluations: 7 }, false);
                let (knn5, _) = run(SearchMode::Fixed { mask: 1 << 2 }, false);
                let (full, _) = run(
                    SearchMode::Exhaustive {
                        evaluations: usize::MAX,
                    },
                    false,
                );
                let (served, srec) = run(BetSearchConfig::default().mode, true);
                served_floor_rate[ni] += f64::from(u8::from(srec.posture == BetPosture::Floor));
                let inc = incumbent(&comps, &queries);
                let mut row = [
                    floor, inc, best1, knn5, full, floor, floor, floor, floor, served,
                ];
                for (bi, &b) in BUDGETS.iter().enumerate() {
                    let (p, prec) = run(puct(b, t), false);
                    let e_budget = prec.search.evaluations as usize;
                    let (e, erec) = run(
                        SearchMode::Exhaustive {
                            evaluations: e_budget,
                        },
                        false,
                    );
                    if b == 16 {
                        row[5] = p;
                        row[6] = e;
                    } else if b == 64 {
                        row[7] = p;
                        row[8] = e;
                    }
                    if n >= 10 && prec.search.mask != 0 {
                        let d = &mut duel[bi];
                        // Selection-split score: what the search optimizes.
                        match prec.search.crps.total_cmp(&erec.search.crps) {
                            std::cmp::Ordering::Less => d.0 += 1,
                            std::cmp::Ordering::Equal => d.1 += 1,
                            std::cmp::Ordering::Greater => d.2 += 1,
                        }
                        // Held-out: relative test-CRPS difference.
                        let (pc, ec) = (p.crps / p.count, e.crps / e.count);
                        d.3 += (ec - pc) / ec.max(1e-12);
                        d.4 += e_budget as f64;
                        d.5 += 1;
                    }
                }
                for (a, s) in arms.iter_mut().zip(row) {
                    a.add(s, floor);
                }
            }
            for (p, a) in pooled[ni].iter_mut().zip(arms) {
                p.crps += a.crps;
                p.covered += a.covered;
                p.queries += a.queries;
                p.skill += a.skill;
                p.winkler_skill += a.winkler_skill;
                p.tasks += a.tasks;
                p.beats_floor += a.beats_floor;
            }
            println!(
                "{n:>4} | {}",
                arms.map(|a| format!(
                    "{:+.3} {:>4.0}% {:>3.0}w",
                    a.skill(),
                    100.0 * a.coverage(),
                    100.0 * a.beats_floor / a.tasks
                ))
                .join(" | ")
            );
        }
    }

    println!("\n## pooled over worlds — mean skill vs floor (coverage, % tasks beating floor)");
    println!(
        "{:>4} | {}",
        "n",
        ARMS.map(|a| format!("{a:>17}")).join(" | ")
    );
    for (ni, &n) in NS.iter().enumerate() {
        println!(
            "{n:>4} | {}",
            pooled[ni]
                .map(|a| format!(
                    "{:+.3} {:>4.0}% {:>3.0}w",
                    a.skill(),
                    100.0 * a.coverage(),
                    100.0 * a.beats_floor / a.tasks
                ))
                .join(" | ")
        );
    }
    println!(
        "\n## pooled — mean Winkler skill vs floor (penalizes misses by 2/α: undercoverage is not free)"
    );
    println!(
        "{:>4} | {}",
        "n",
        ARMS.map(|a| format!("{a:>8}")).join(" | ")
    );
    for (ni, &n) in NS.iter().enumerate() {
        println!(
            "{n:>4} | {}",
            pooled[ni]
                .map(|a| format!("{:+8.3}", a.winkler_skill / a.tasks))
                .join(" | ")
        );
    }
    println!("\nserved floor-posture rate per n:");
    for (ni, &n) in NS.iter().enumerate() {
        println!(
            "  n={n:>3}: {:.1}%",
            100.0 * served_floor_rate[ni] / (TASKS as f64 * WORLDS.len() as f64)
        );
    }

    println!("\n## PUCT vs exhaustive at EQUAL distinct-composition budget (n ≥ 10, all worlds)");
    for (bi, &b) in BUDGETS.iter().enumerate() {
        let d = duel[bi];
        println!(
            "  b={b:>3}: mean E={:.1} | selection CRPS: puct better {} / tie {} / worse {} | held-out rel. gain of puct over exh {:+.4}",
            d.4 / d.5 as f64,
            d.0,
            d.1,
            d.2,
            d.3 / d.5 as f64
        );
    }

    // Determinism: the same task and seed reproduce the same record.
    let (comps, _) = World::Plane.task(30, 7);
    let mut a = head(puct(64, 1), true);
    let mut b = head(puct(64, 1), true);
    a.calibrate(&comps);
    b.calibrate(&comps);
    assert_eq!(
        a.record(),
        b.record(),
        "PUCT bet search is not deterministic"
    );
}

/// G2 — calibrate (search) cost and the served-path latency, n = 60.
#[test]
#[ignore = "GOAT bench — run in --release with --nocapture (see the module doc)"]
fn bench_952_g2_latency() {
    let (comps, queries) = World::Plane.task(60, 0x952);
    let modes: [(&str, SearchMode); 5] = [
        ("knn5 fixed", SearchMode::Fixed { mask: 1 << 2 }),
        ("puct@16", puct(16, 1)),
        ("puct@64", puct(64, 1)),
        ("puct@256", puct(256, 1)),
        (
            "full exhaustive",
            SearchMode::Exhaustive {
                evaluations: usize::MAX,
            },
        ),
    ];
    println!(
        "Issue 952 G2 — calibrate cost (n=60, best of 20) + estimate_into latency (10k queries)"
    );
    for (name, mode) in modes {
        let mut best = f64::INFINITY;
        let mut h = head(mode, false);
        for _ in 0..20 {
            let t0 = Instant::now();
            h.calibrate(black_box(&comps));
            best = best.min(t0.elapsed().as_secs_f64() * 1e6);
        }
        let rec = *h.record().unwrap();
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, ALPHA);
        let mut lat: Vec<f64> = Vec::with_capacity(10_000);
        for i in 0..10_000 {
            let q = &queries[i % queries.len()].features;
            let t0 = Instant::now();
            h.estimate_into(black_box(q), &mut iv);
            black_box(&iv);
            lat.push(t0.elapsed().as_secs_f64() * 1e9);
        }
        assert!(iv.point.is_finite(), "{name}: served a non-finite point");
        // Loud zero (Issue 855): a timed region the optimiser deleted reads
        // 0 here rather than a flattering latency.
        assert!(best > 0.0, "{name}: calibrate timed 0 — region elided");
        assert!(
            lat.iter().sum::<f64>() > 0.0,
            "{name}: 10k estimates timed 0 — region elided"
        );
        lat.sort_unstable_by(f64::total_cmp);
        let p50 = lat[lat.len() / 2];
        // p99 index 9900 of 10 000: tail support 100 samples.
        let p99 = lat[lat.len() * 99 / 100];
        println!(
            "  {name:>16}: calibrate {best:>8.1} µs (E={:>2}, bets={}) | estimate p50 {p50:>6.0} ns p99 {p99:>6.0} ns",
            rec.search.evaluations, rec.search.bets_materialized
        );
    }
}
