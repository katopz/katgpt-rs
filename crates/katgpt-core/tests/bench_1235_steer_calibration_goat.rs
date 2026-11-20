//! Issue 1235 row (a) — closed-form steering-strength calibration GOAT gate
//! (ACE App. D controller, arXiv:2602.22600, transferred to a DECISION margin).
//!
//! Two arms on a planted-margin fixture, both applied through the SAME
//! `SubspaceSteeringField<8, 1>` and the SAME gate (mass + already-at-target),
//! so the comparison is purely about STRENGTH:
//! * hand — one α for the whole population, tuned FAIRLY: the closed-form
//!   strength of the MEDIAN state (the best single constant a careful hand
//!   would pick);
//! * calib — per-state `s* = (m_target − m0)/g` from one probe.
//!
//! Margins: LINEAR `m = w·s + b` and CURVED `m = tanh(w·s + b)` (a bounded
//! score difference; the one-probe secant error is MEASURED here).
//!
//! Pre-registered bars (Issue 1235, recorded before the first run):
//!   G0 determinism — two runs, BLAKE3 over (decisions, after-margins) equal.
//!   G1 linear      — every uncapped calibrated state lands |m − m_target| ≤ 1e-4,
//!                    and calib overshoot ≤ hand overshoot (the issue's bar).
//!   G2 curved      — calib mean |err| ≤ hand mean |err| AND calib mean
//!                    overshoot ≤ hand mean overshoot AND calib hit-rate
//!                    (|err| ≤ 0.05) ≥ hand hit-rate.
//!   G3 gates       — skip states get s = 0; AlreadyAtTarget count equals the
//!                    planted count with m0 ≥ m_target among mass-passing states.
//!   G4 refusal     — a direction ⟂ w and a direction −w refuse NoGain.
//!   G5 K=1 parity  — the calibrator's `steered` is bit-identical to
//!                    `SubspaceSteeringField::<8,1>::apply` at α = s*.
//!   G6 zero-alloc  — calibrate_strength over the population allocates 0.
#![cfg(feature = "relevance_core")]

use katgpt_core::steer_calibration::{
    SkipReason, SteerCalibConfig, SteerCalibError, SteerDecision, calibrate_strength, steered,
};
use katgpt_core::subspace_steering::SubspaceSteeringField;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    // Per-THREAD tracking: the harness runs tests in parallel, and a global
    // flag would count sibling tests' allocations.
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.with(Cell::get) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

const D: usize = 8;
const N: usize = 256;
const M_TARGET: f32 = 0.2;
const HIT_TOL: f32 = 0.05;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}

fn dot(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut acc = 0.0;
    for j in 0..D {
        acc += a[j] * b[j];
    }
    acc
}

fn unit(v: [f32; D]) -> [f32; D] {
    let n = dot(&v, &v).sqrt();
    v.map(|x| x / n)
}

#[derive(Clone, Copy)]
enum Shape {
    Linear,
    Curved,
}

struct Fixture {
    w: [f32; D],
    b: f32,
    u: [f32; D],
    shape: Shape,
    states: Vec<[f32; D]>,
    mass: Vec<f32>,
}

impl Fixture {
    fn margin(&self, s: &[f32; D]) -> f32 {
        let z = dot(&self.w, s) + self.b;
        match self.shape {
            Shape::Linear => z,
            Shape::Curved => z.tanh(),
        }
    }

    /// Plant N states whose pre-steer margin is uniform in `[lo, hi]`.
    fn plant(shape: Shape, lo: f32, hi: f32, seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let w = {
            let mut v = [0f32; D];
            for x in &mut v {
                *x = rng.gauss();
            }
            unit(v).map(|x| x * 1.3)
        };
        // Steering direction: correlated with w but not parallel (a
        // discovered basis is never exactly the margin normal).
        let u = {
            let mut v = [0f32; D];
            for (j, x) in v.iter_mut().enumerate() {
                *x = w[j] + 0.3 * rng.gauss();
            }
            unit(v)
        };
        // Fixture refusal: the planted direction must raise the margin.
        assert!(dot(&w, &u) > 0.5, "fixture: w·u = {}", dot(&w, &u));
        let b = -0.1;
        let ww = dot(&w, &w);
        let mut states = Vec::with_capacity(N);
        let mut mass = Vec::with_capacity(N);
        for i in 0..N {
            let want = lo + (hi - lo) * (i as f32 + 0.5) / N as f32;
            let z_want = match shape {
                Shape::Linear => want,
                Shape::Curved => want.atanh(),
            };
            let mut s = [0f32; D];
            for x in &mut s {
                *x = 0.3 * rng.gauss();
            }
            let fix = (z_want - (dot(&w, &s) + b)) / ww;
            for j in 0..D {
                s[j] += fix * w[j];
            }
            states.push(s);
            mass.push(rng.next_f32());
        }
        Self {
            w,
            b,
            u,
            shape,
            states,
            mass,
        }
    }
}

fn cfg() -> SteerCalibConfig {
    SteerCalibConfig {
        probe: 0.05,
        m_target: M_TARGET,
        s_max: 1.0,
        min_gain: 1e-4,
        mass_threshold: 0.1,
    }
}

#[derive(Default, Debug)]
struct ArmStats {
    steered: usize,
    hits: usize,
    sum_abs: f64,
    sum_over: f64,
    max_abs_uncapped: f32,
    capped: usize,
}

impl ArmStats {
    fn push(&mut self, err: f32, capped: bool) {
        self.steered += 1;
        if err.abs() <= HIT_TOL {
            self.hits += 1;
        }
        self.sum_abs += err.abs() as f64;
        self.sum_over += err.max(0.0) as f64;
        if capped {
            self.capped += 1;
        } else {
            self.max_abs_uncapped = self.max_abs_uncapped.max(err.abs());
        }
    }
    fn mean_abs(&self) -> f64 {
        self.sum_abs / self.steered as f64
    }
    fn mean_over(&self) -> f64 {
        self.sum_over / self.steered as f64
    }
    fn hit_rate(&self) -> f64 {
        self.hits as f64 / self.steered as f64
    }
}

struct Run {
    calib: ArmStats,
    hand: ArmStats,
    hand_alpha: f32,
    at_target: usize,
    below_mass: usize,
    digest: [u8; 32],
}

fn apply_field(state: &[f32; D], u: &[f32; D], a: f32) -> [f32; D] {
    let field = SubspaceSteeringField::<D, 1>::new_unchecked([*u], [a]);
    let mut out = *state;
    field.apply(&mut out);
    out
}

fn run(fx: &Fixture) -> Run {
    let cfg = cfg();
    let decisions: Vec<SteerDecision> = fx
        .states
        .iter()
        .zip(&fx.mass)
        .map(|(s, &m)| calibrate_strength(s, &fx.u, m, |x| fx.margin(x), &cfg).unwrap())
        .collect();

    // Fair hand α: the closed-form strength of the MEDIAN steered state.
    let mut steer_s: Vec<f32> = decisions
        .iter()
        .filter_map(|d| match *d {
            SteerDecision::Steer { s_star, .. } => Some(s_star),
            SteerDecision::Skip { .. } => None,
        })
        .collect();
    steer_s.sort_by(f32::total_cmp);
    let hand_alpha = steer_s[steer_s.len() / 2];

    let mut calib = ArmStats::default();
    let mut hand = ArmStats::default();
    let (mut at_target, mut below_mass) = (0, 0);
    let mut h = blake3::Hasher::new();
    for (s, d) in fx.states.iter().zip(&decisions) {
        match *d {
            SteerDecision::Skip { reason, .. } => {
                match reason {
                    SkipReason::AlreadyAtTarget => at_target += 1,
                    SkipReason::BelowMass => below_mass += 1,
                }
                // G3: a skip applies s = 0 — the state is untouched.
                assert_eq!(apply_field(s, &fx.u, d.strength()), *s);
                h.update(&[0u8]);
            }
            SteerDecision::Steer { s_star, capped, .. } => {
                // G5: K=1 field parity at α = s*.
                let via_field = apply_field(s, &fx.u, s_star);
                assert_eq!(via_field, steered(s, &fx.u, s_star), "G5 parity");
                let m_c = fx.margin(&via_field);
                calib.push(m_c - M_TARGET, capped);
                let m_h = fx.margin(&apply_field(s, &fx.u, hand_alpha));
                hand.push(m_h - M_TARGET, false);
                h.update(&s_star.to_le_bytes());
                h.update(&m_c.to_le_bytes());
            }
        }
    }
    Run {
        calib,
        hand,
        hand_alpha,
        at_target,
        below_mass,
        digest: *h.finalize().as_bytes(),
    }
}

fn report(name: &str, r: &Run) {
    println!(
        "[{name}] hand_alpha={:.4} skip(at_target={}, below_mass={})",
        r.hand_alpha, r.at_target, r.below_mass
    );
    for (arm, s) in [("calib", &r.calib), ("hand ", &r.hand)] {
        println!(
            "  {arm}: steered={} hit@{HIT_TOL}={:.3} mean|err|={:.5} mean_over={:.5} max|err|(uncapped)={:.2e} capped={}",
            s.steered,
            s.hit_rate(),
            s.mean_abs(),
            s.mean_over(),
            s.max_abs_uncapped,
            s.capped
        );
    }
}

fn expected_at_target(fx: &Fixture) -> usize {
    let cfg = cfg();
    fx.states
        .iter()
        .zip(&fx.mass)
        .filter(|(s, m)| **m >= cfg.mass_threshold && fx.margin(s) >= M_TARGET)
        .count()
}

#[test]
fn g0_g1_g3_g5_linear() {
    let fx = Fixture::plant(Shape::Linear, -1.0, 0.6, 12_350);
    let r = run(&fx);
    report("linear", &r);
    assert_eq!(r.digest, run(&fx).digest, "G0 determinism");
    assert!(r.calib.steered > N / 2);
    assert!(
        r.calib.max_abs_uncapped <= 1e-4,
        "G1: {}",
        r.calib.max_abs_uncapped
    );
    assert!(r.calib.mean_over() <= r.hand.mean_over(), "G1 overshoot");
    assert_eq!(r.at_target, expected_at_target(&fx), "G3");
}

#[test]
fn g0_g2_g3_g5_curved() {
    let fx = Fixture::plant(Shape::Curved, -0.6, 0.5, 12_351);
    let r = run(&fx);
    report("curved", &r);
    assert_eq!(r.digest, run(&fx).digest, "G0 determinism");
    assert!(r.calib.mean_abs() <= r.hand.mean_abs(), "G2 mean |err|");
    assert!(r.calib.mean_over() <= r.hand.mean_over(), "G2 overshoot");
    assert!(r.calib.hit_rate() >= r.hand.hit_rate(), "G2 hit-rate");
    assert_eq!(r.at_target, expected_at_target(&fx), "G3");
}

#[test]
fn g4_refuses_no_gain() {
    let fx = Fixture::plant(Shape::Linear, -1.0, 0.0, 12_352);
    // A direction orthogonal to w (Gram–Schmidt e0 against w).
    let mut e = [0f32; D];
    e[0] = 1.0;
    let wn = unit(fx.w);
    let p = dot(&e, &wn);
    let ortho = unit(std::array::from_fn(|j| e[j] - p * wn[j]));
    let neg = wn.map(|x| -x);
    let s = &fx.states[0];
    for dir in [ortho, neg] {
        let r = calibrate_strength(s, &dir, 1.0, |x| fx.margin(x), &cfg());
        assert!(matches!(r, Err(SteerCalibError::NoGain { .. })), "{r:?}");
    }
}

#[test]
fn g6_zero_alloc() {
    let fx = Fixture::plant(Shape::Curved, -0.6, 0.5, 12_353);
    let cfg = cfg();
    let mut acc = 0.0f32;
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING.with(|t| t.set(true));
    for (s, &m) in fx.states.iter().zip(&fx.mass) {
        acc += calibrate_strength(s, &fx.u, m, |x| fx.margin(x), &cfg)
            .unwrap()
            .strength();
    }
    TRACKING.with(|t| t.set(false));
    std::hint::black_box(acc);
    assert_eq!(ALLOC_COUNT.load(Ordering::SeqCst), 0, "G6");
}
