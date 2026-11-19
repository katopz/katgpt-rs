//! Issue 1232 / Plan 639 Phase 3 T3.3 — the katgpt-hla cache γ novelty gate
//! GOAT harness. Gates and fixture are PRE-REGISTERED in Issue 1232 (pushed
//! `c4d03b05d` before any measurement); this file implements them verbatim.
//!
//! ```text
//! cargo test --release -p katgpt-hla --features novelty_gamma \
//!     --test novelty_gamma_goat -- --nocapture --test-threads=1
//! ```
//!
//! Cached latent: `m̂_t = mQ_t / Z_t` from the REAL `hla_state_update`
//! (hd = 8, one head); truth: the query-regime mean `m_t`.

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use katgpt_hla::novelty_gamma::NoveltyRetentionConfig;
use katgpt_hla::{HlaNoveltyGamma, HlaQHeadState, hla_state_update};

// ── G4 instrument: a counting allocator local to this binary ────────────────

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// ── pinned constants (Issue 1232) ───────────────────────────────────────────

const HD: usize = 8;
const TICKS: usize = 600;
const SEEDS: [u64; 3] = [1, 2, 3];
const STEP_TICKS: [usize; 3] = [150, 300, 450];
const SPIKE_EVERY: usize = 40;
const SPIKE_AMP: f32 = 3.0;
const SIGMA_STEADY: f32 = 0.3;
const SIGMA_STEPS: f32 = 0.1;
const WARMUP: usize = 50;
const ADAPT_CAP: usize = 100;
const ADAPT_CLOSE: f32 = 0.1;
const BASE_GAMMA: f32 = 0.9;
const PINNED_GAMMAS: [f32; 11] = [
    0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99,
];
const KAPPAS: [f32; 3] = [0.2, 0.4, 0.8];
const MUS: [f32; 3] = [0.0, 0.2, 0.4];
const BETAS: [f32; 2] = [4.0, 8.0];
const GMAXES: [f32; 2] = [0.99, 0.995];

fn dense_gammas() -> Vec<f32> {
    (50..=99).map(|k| k as f32 / 100.0).collect()
}

// ── fixture ─────────────────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn gauss(&mut self) -> f32 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Case {
    Steady,
    Steps,
    Spikes,
}

struct Trace {
    /// Per tick (q, k, v), each `[HD]`.
    q: Vec<[f32; HD]>,
    k: Vec<[f32; HD]>,
    v: Vec<[f32; HD]>,
    /// Per tick truth `m_t`.
    m: Vec<[f32; HD]>,
}

fn draw_mean(r: &mut Rng) -> [f32; HD] {
    std::array::from_fn(|_| r.uniform(-1.0, 1.0))
}

fn trace(case: Case, seed: u64) -> Trace {
    let salt = match case {
        Case::Steady => 11,
        Case::Steps => 23,
        Case::Spikes => 37,
    };
    let mut r = Rng::new(seed * 1000 + salt);
    let sigma = if case == Case::Steps {
        SIGMA_STEPS
    } else {
        SIGMA_STEADY
    };
    let mut mean = draw_mean(&mut r);
    let mut t = Trace {
        q: Vec::with_capacity(TICKS),
        k: Vec::with_capacity(TICKS),
        v: Vec::with_capacity(TICKS),
        m: Vec::with_capacity(TICKS),
    };
    for tick in 0..TICKS {
        if case == Case::Steps && STEP_TICKS.contains(&tick) {
            mean = draw_mean(&mut r);
        }
        let spike = case == Case::Spikes && tick > 0 && tick % SPIKE_EVERY == 0;
        let q: [f32; HD] = std::array::from_fn(|i| {
            if spike {
                let s = if r.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                mean[i] + SPIKE_AMP * s
            } else {
                mean[i] + sigma * r.gauss()
            }
        });
        let k: [f32; HD] = std::array::from_fn(|_| 0.3 * r.gauss());
        let v: [f32; HD] = std::array::from_fn(|_| r.gauss());
        t.q.push(q);
        t.k.push(k);
        t.v.push(v);
        t.m.push(mean);
    }
    t
}

// ── arms ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Arm {
    Fixed(f32),
    Gated(NoveltyRetentionConfig),
}

struct Cache {
    sk: Vec<f32>,
    head: HlaQHeadState,
    tmp: Vec<f32>,
}

impl Cache {
    fn new(hd: usize) -> Self {
        Self {
            sk: vec![0.0; hd * hd],
            head: HlaQHeadState::new(hd),
            tmp: vec![0.0; hd],
        }
    }
    #[inline]
    fn update(&mut self, q: &[f32], k: &[f32], v: &[f32], gamma: f32) {
        let hd = q.len();
        hla_state_update(
            &mut self.sk,
            &mut self.head,
            q,
            k,
            v,
            hd,
            gamma,
            &mut self.tmp,
        );
    }
}

fn arm_gate(cfg: NoveltyRetentionConfig) -> HlaNoveltyGamma {
    HlaNoveltyGamma::arm(cfg).unwrap_or_else(|| {
        panic!(
            "arm refused (NOVELTY_RETENTION=0 set, or {cfg:?} breaks a guard) — the bench is meaningless demoted"
        )
    })
}

/// Run one arm; returns the per-tick estimate `m̂_t = mQ_t / Z_t`.
fn run(arm: Arm, tr: &Trace) -> Vec<[f32; HD]> {
    let mut c = Cache::new(HD);
    let mut gate = match arm {
        Arm::Gated(cfg) => Some(arm_gate(cfg)),
        Arm::Fixed(_) => None,
    };
    let mut z = 0.0f32;
    let mut out = Vec::with_capacity(TICKS);
    for t in 0..tr.q.len() {
        let gamma = match (&mut gate, arm) {
            (Some(g), _) => g.step(&tr.q[t], &c.head.mq, BASE_GAMMA),
            (None, Arm::Fixed(a)) => a,
            (None, Arm::Gated(_)) => unreachable!(),
        };
        c.update(&tr.q[t], &tr.k[t], &tr.v[t], gamma);
        z = gamma * z + 1.0;
        if let Some(g) = &gate {
            assert_eq!(g.mass(), z, "gate mass drifted from Z");
        }
        out.push(std::array::from_fn(|i| c.head.mq[i] / z));
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct Point {
    adaptation: f64,
    noise_rmse: f64,
    capped: usize,
}

fn dist(a: &[f32; HD], b: &[f32; HD]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn measure(arm: Arm, seed: u64) -> Point {
    let tr = trace(Case::Steps, seed);
    let est = run(arm, &tr);
    let (mut sum, mut capped) = (0usize, 0usize);
    for &s in &STEP_TICKS {
        let (from, to) = (tr.m[s - 1], tr.m[s]);
        let need = ADAPT_CLOSE * dist(&to, &from);
        let k = (0..ADAPT_CAP)
            .find(|&j| dist(&est[s + j], &to) <= need)
            .unwrap_or(ADAPT_CAP);
        capped += usize::from(k == ADAPT_CAP);
        sum += k;
    }
    let (mut se, mut cnt) = (0.0f64, 0usize);
    for case in [Case::Steady, Case::Spikes] {
        let tr = trace(case, seed);
        let est = run(arm, &tr);
        for (e, m) in est[WARMUP..].iter().zip(&tr.m[WARMUP..]) {
            for (&ei, &mi) in e.iter().zip(m) {
                let d = (ei - mi) as f64;
                se += d * d;
                cnt += 1;
            }
        }
    }
    Point {
        adaptation: sum as f64 / STEP_TICKS.len() as f64,
        noise_rmse: (se / cnt as f64).sqrt(),
        capped,
    }
}

fn mean_point(ps: &[Point]) -> Point {
    let k = ps.len() as f64;
    Point {
        adaptation: ps.iter().map(|p| p.adaptation).sum::<f64>() / k,
        noise_rmse: ps.iter().map(|p| p.noise_rmse).sum::<f64>() / k,
        capped: ps.iter().map(|p| p.capped).sum(),
    }
}

/// Noise RMSE of the piecewise-linear dense curve at `adapt`; `None` outside.
fn curve_at(curve: &[(f32, Point)], adapt: f64) -> Option<f64> {
    let mut pts: Vec<(f64, f64)> = curve
        .iter()
        .map(|(_, p)| (p.adaptation, p.noise_rmse))
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if adapt >= x0 && adapt <= x1 {
            if x1 == x0 {
                return Some(y0.min(y1));
            }
            return Some(y0 + (y1 - y0) * (adapt - x0) / (x1 - x0));
        }
    }
    None
}

struct Verdict {
    i_gamma: Option<f32>,
    dominator: Option<f32>,
    iii: Option<f64>,
}

/// The Plan 639 T2.2 verdict, verbatim (Issue 1232 G2).
fn judge(
    cell_seeds: &[Point],
    pinned_seeds: &[(f32, Vec<Point>)],
    dense_mean: &[(f32, Point)],
) -> Verdict {
    let i_gamma = pinned_seeds.iter().find_map(|(a, ps)| {
        let ok = cell_seeds
            .iter()
            .zip(ps)
            .all(|(c, p)| c.adaptation <= 0.7 * p.adaptation && c.noise_rmse <= 1.1 * p.noise_rmse);
        ok.then_some(*a)
    });
    let cm = mean_point(cell_seeds);
    let dominator = dense_mean
        .iter()
        .find(|(_, p)| p.adaptation <= cm.adaptation && p.noise_rmse <= cm.noise_rmse)
        .map(|(a, _)| *a);
    let iii = curve_at(dense_mean, cm.adaptation).map(|y| y - cm.noise_rmse);
    Verdict {
        i_gamma,
        dominator,
        iii,
    }
}

// ── G1 ──────────────────────────────────────────────────────────────────────

fn bits(c: &Cache) -> Vec<u32> {
    c.sk.iter()
        .chain(&c.head.cqv)
        .chain(&c.head.mq)
        .chain(&c.head.g)
        .chain(&c.head.h)
        .map(|x| x.to_bits())
        .collect()
}

/// G1(c): armed κ = μ = 0 ⇒ γ_t == clamp(base) every tick and the cache is
/// bit-identical to the fixed-clamped-base incumbent run.
#[test]
fn g1_identity_arm_is_bit_exact() {
    for &seed in &SEEDS {
        for case in [Case::Steady, Case::Steps, Case::Spikes] {
            let tr = trace(case, seed);
            let cfg = NoveltyRetentionConfig {
                kappa: 0.0,
                mu: 0.0,
                ..NoveltyRetentionConfig::default()
            };
            let mut gate = arm_gate(cfg);
            let clamped = BASE_GAMMA.max(cfg.g_min).min(cfg.g_max);
            let (mut inc, mut gat) = (Cache::new(HD), Cache::new(HD));
            for t in 0..TICKS {
                let g = gate.step(&tr.q[t], &gat.head.mq, BASE_GAMMA);
                assert_eq!(g.to_bits(), clamped.to_bits(), "tick {t}");
                inc.update(&tr.q[t], &tr.k[t], &tr.v[t], clamped);
                gat.update(&tr.q[t], &tr.k[t], &tr.v[t], g);
            }
            assert_eq!(bits(&inc), bits(&gat));
        }
    }
    println!(
        "G1(c): identity arm bit-exact vs the fixed-clamped-base incumbent, 3 seeds × 3 cases × {TICKS} ticks"
    );
}

/// G1 companion: the armed path is deterministic — two runs, same bits.
#[test]
fn g1_armed_path_is_replay_deterministic() {
    let tr = trace(Case::Spikes, 2);
    let once = |_: ()| {
        let mut gate = arm_gate(NoveltyRetentionConfig::default());
        let mut c = Cache::new(HD);
        let mut gs = Vec::with_capacity(TICKS);
        for t in 0..TICKS {
            let g = gate.step(&tr.q[t], &c.head.mq, BASE_GAMMA);
            gs.push(g.to_bits());
            c.update(&tr.q[t], &tr.k[t], &tr.v[t], g);
        }
        (gs, bits(&c))
    };
    assert_eq!(once(()), once(()));
}

// ── G2 ──────────────────────────────────────────────────────────────────────

#[test]
fn g2_goat_sweep() {
    let pinned: Vec<(f32, Vec<Point>)> = PINNED_GAMMAS
        .iter()
        .map(|&a| {
            (
                a,
                SEEDS.iter().map(|&s| measure(Arm::Fixed(a), s)).collect(),
            )
        })
        .collect();
    let dense: Vec<(f32, Point)> = dense_gammas()
        .into_iter()
        .map(|a| {
            let ps: Vec<Point> = SEEDS.iter().map(|&s| measure(Arm::Fixed(a), s)).collect();
            (a, mean_point(&ps))
        })
        .collect();

    println!("\n## G2 fixed-γ curve (pinned γ, per seed: adaptation / noise_rmse)");
    println!("| γ | adapt s1 | adapt s2 | adapt s3 | noise s1 | noise s2 | noise s3 | capped |");
    println!("|---|---|---|---|---|---|---|---|");
    for (a, ps) in &pinned {
        let m = mean_point(ps);
        println!(
            "| {a:.2} | {:.2} | {:.2} | {:.2} | {:.4} | {:.4} | {:.4} | {} |",
            ps[0].adaptation,
            ps[1].adaptation,
            ps[2].adaptation,
            ps[0].noise_rmse,
            ps[1].noise_rmse,
            ps[2].noise_rmse,
            m.capped
        );
    }
    println!("\n## G2 dense fixed-γ curve (seed mean)");
    println!("| γ | adaptation | noise_rmse | capped |");
    println!("|---|---|---|---|");
    for (a, p) in &dense {
        println!(
            "| {a:.2} | {:.2} | {:.4} | {} |",
            p.adaptation, p.noise_rmse, p.capped
        );
    }

    println!("\n## G2 gated grid (base γ {BASE_GAMMA}; seed mean; Plan 639 T2.2 verdict)");
    println!(
        "| κ | μ | β | g_max | adaptation | noise_rmse | capped | (i) γ | (ii) | dominator | (iii) dist | PASS |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut passing: Vec<(NoveltyRetentionConfig, Point, f64)> = Vec::new();
    let mut cells = 0usize;
    for &kappa in &KAPPAS {
        for &mu in &MUS {
            for &beta in &BETAS {
                for &g_max in &GMAXES {
                    let ncfg = NoveltyRetentionConfig {
                        kappa,
                        mu,
                        beta,
                        g_max,
                        ..NoveltyRetentionConfig::default()
                    };
                    let ps: Vec<Point> = SEEDS
                        .iter()
                        .map(|&s| measure(Arm::Gated(ncfg), s))
                        .collect();
                    let m = mean_point(&ps);
                    let v = judge(&ps, &pinned, &dense);
                    let ii = v.dominator.is_none();
                    let pass = v.i_gamma.is_some() && ii && v.iii.is_some_and(|d| d > 0.0);
                    cells += 1;
                    println!(
                        "| {kappa} | {mu} | {beta} | {g_max} | {:.2} | {:.4} | {} | {} | {} | {} | {} | {} |",
                        m.adaptation,
                        m.noise_rmse,
                        m.capped,
                        v.i_gamma.map_or("—".into(), |a| format!("{a:.2}")),
                        if ii { "✓" } else { "✗" },
                        v.dominator.map_or("—".into(), |a| format!("{a:.2}")),
                        v.iii.map_or("OUT-OF-RANGE".into(), |d| format!("{d:+.4}")),
                        if pass { "**PASS**" } else { "fail" }
                    );
                    if pass {
                        passing.push((ncfg, m, v.iii.unwrap()));
                    }
                }
            }
        }
    }
    assert_eq!(cells, KAPPAS.len() * MUS.len() * BETAS.len() * GMAXES.len());
    passing.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "\nVERDICT: {}/{} gated cells pass (i)+(ii)+(iii)",
        passing.len(),
        cells
    );
    match passing.first() {
        Some((c, p, d)) => println!(
            "BEST: κ={} μ={} β={} g_max={} adaptation={:.2} noise_rmse={:.4} dist_below_curve={:+.4}",
            c.kappa, c.mu, c.beta, c.g_max, p.adaptation, p.noise_rmse, d
        ),
        None => println!("BEST: none — DONE-NEGATIVE"),
    }
}

// ── G4 ──────────────────────────────────────────────────────────────────────

#[test]
fn g4_armed_path_is_alloc_free() {
    let tr = trace(Case::Spikes, 1);
    let mut gate = arm_gate(NoveltyRetentionConfig::default());
    let mut c = Cache::new(HD);
    let g = gate.step(&tr.q[0], &c.head.mq, BASE_GAMMA);
    c.update(&tr.q[0], &tr.k[0], &tr.v[0], g);
    let before = ALLOCS.load(Ordering::Relaxed);
    for t in 1..TICKS {
        let g = gate.step(black_box(&tr.q[t]), &c.head.mq, BASE_GAMMA);
        c.update(&tr.q[t], &tr.k[t], &tr.v[t], g);
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    black_box(&c.head.mq);
    // Process-wide counter; a parallel test can only INFLATE it (false red).
    assert_eq!(
        after - before,
        0,
        "armed path allocated {} times",
        after - before
    );
    println!(
        "G4: armed step + kernel over {} ticks: 0 allocations",
        TICKS - 1
    );
}

// ── G3 ──────────────────────────────────────────────────────────────────────

/// Interleaved A/B per entity-tick (k = 1): a = kernel at fixed γ 0.9,
/// b = gate step + kernel at the gated γ. Statistic: `b − a` ns.
fn g3_delta(hd: usize) -> (f64, f64, f64) {
    const RING: usize = 64;
    let mut r = Rng::new(0xC0FFEE + hd as u64);
    let mut mean: Vec<f32> = (0..hd).map(|_| r.uniform(-1.0, 1.0)).collect();
    let mut toks: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(RING);
    for i in 0..RING {
        if i == RING / 2 {
            mean = (0..hd).map(|_| r.uniform(-1.0, 1.0)).collect();
        }
        let q = mean.iter().map(|&m| m + SIGMA_STEADY * r.gauss()).collect();
        let k = (0..hd).map(|_| 0.3 * r.gauss()).collect();
        let v = (0..hd).map(|_| r.gauss()).collect();
        toks.push((q, k, v));
    }
    let mut inc = Cache::new(hd);
    let mut gat = Cache::new(hd);
    let mut gate = arm_gate(NoveltyRetentionConfig::default());
    let (mut sink_a, mut sink_b) = (0.0f32, 0.0f32);
    let res = ab_timing::ab_median_ratio(
        51,
        20_000,
        4_000,
        |i| {
            let (q, k, v) = &toks[i % RING];
            inc.update(black_box(q), k, v, black_box(BASE_GAMMA));
            sink_a += black_box(inc.head.mq[i % hd]);
        },
        |i| {
            let (q, k, v) = &toks[i % RING];
            let g = gate.step(black_box(q), &gat.head.mq, black_box(BASE_GAMMA));
            gat.update(q, k, v, g);
            sink_b += black_box(gat.head.mq[i % hd]);
        },
    );
    black_box((sink_a, sink_b));
    res.report(&format!("G3 hd={hd} gated/incumbent"));
    let (a, b) = (res.a_ns_per_iter(), res.b_ns_per_iter());
    (a, b, b - a)
}

#[test]
fn g3_added_cost_per_entity_tick() {
    let (a8, b8, d8) = g3_delta(HD);
    println!(
        "G3 hd=8 (GATED): incumbent {a8:.2} ns/tick, gated {b8:.2} ns/tick, Δ {d8:.2} ns (bar k·Δ ≤ 10.0, k = 1; release={})",
        !cfg!(debug_assertions)
    );
    let (a64, b64, d64) = g3_delta(64);
    println!(
        "G3 hd=64 (REPORTED, not gated): incumbent {a64:.2} ns/tick, gated {b64:.2} ns/tick, Δ {d64:.2} ns"
    );
}
