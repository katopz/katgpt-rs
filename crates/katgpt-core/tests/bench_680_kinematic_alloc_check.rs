//! Plan 578 G4 — zero-alloc steady state for the kinematics operators.
//!
//! Separate single-purpose binary (the CountingAllocator global pattern —
//! `bench_655/656/676` convention): parallel tests share the global counter,
//! so all checks live in ONE test function, serial by construction.

use katgpt_core::kinematics::fixture::{self, RangeLabel};
use katgpt_core::kinematics::perception::{
    Eps, RegimeClassifier, RegimeConfig, RegimeSnapshot, ResidualConfig, ResidualMonitor,
    closest_approach, extrapolation_horizon, intercept_time, predictive_half_width,
    time_to_contact, time_to_contact_log,
};
use katgpt_core::kinematics::{
    KinState, Sched, extrapolation_weight_ss, kinematic_extrapolate_capped_into,
    kinematic_extrapolate_into, reference_chain_extrapolate_into, terminal_velocity,
};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// G4: zero steady-state allocations across every hot-path operator
/// (observe, closed-form + reference-chain + capped extrapolate, TTC both
/// variants, closest approach, intercept, regime classify, residual update,
/// horizon gate, weight-ss, half-width, terminal velocity) — one function,
/// serial by construction.
#[test]
fn kinematics_g4_zero_alloc_steady_state() {
    // Warmup: settle the lattice OnceLock (a one-time static init — no heap,
    // but let any lazy machinery settle before measuring).
    const CALLS: usize = 10_000;

    {
        let mut st = KinState::<4>::new(1.0).unwrap();
        let mut x = [0.0f32; 4];
        for t in 0..8u32 {
            for (ch, v) in x.iter_mut().enumerate() {
                *v = 0.5 * t as f32 + 0.25 * ch as f32;
            }
            st.observe_into(&x, t).unwrap();
        }
        let mut out = [0.0f32; 4];
        for sched in [
            Sched::ZeroJerk,
            Sched::ConstJerk { j: 0.25 },
            Sched::Measured,
            Sched::ClampedCorrection {
                j_max: 1.0,
                lambda: 0.1,
            },
            Sched::GeometricDrag { rho: 0.5 },
        ] {
            kinematic_extrapolate_into(&st, 100, &sched, &mut out).unwrap();
            reference_chain_extrapolate_into(&st, 8, &sched, &mut out).unwrap();
            kinematic_extrapolate_capped_into(&st, 8, &sched, 2, &mut out).unwrap();
        }
        black_box(out[0]);
        black_box(terminal_velocity(1.0, -0.5, 1.0, 0.5));
        black_box(extrapolation_weight_ss(100, 3));
        black_box(predictive_half_width(0.05, 100, 3, 0.05));
    }

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    let mut st = KinState::<4>::new(1.0).unwrap();
    let mut x = [0.0f32; 4];
    let mut out = [0.0f32; 4];
    let eps = Eps::from_obs_noise(0.05, 1.0);
    let mut clf = RegimeClassifier::new(RegimeConfig::default());
    let mut mon = ResidualMonitor::new(ResidualConfig::default());
    let mut snap;
    let p1 = [0.0f32, 0.0];
    let v1 = [1.0f32, 0.25];
    let p2 = [10.0f32, 5.0];
    let v2 = [-2.0f32, 0.0];
    for i in 0..CALLS {
        let t = i as u32;
        for (ch, v) in x.iter_mut().enumerate() {
            *v = 0.5 * t as f32 + 0.125 * ch as f32;
        }
        let vel_before = st.vel;
        st.observe_into(&x, t).unwrap();
        for sched in [
            Sched::ZeroJerk,
            Sched::ConstJerk { j: 0.25 },
            Sched::Measured,
            Sched::ClampedCorrection {
                j_max: 1.0,
                lambda: 0.1,
            },
            Sched::GeometricDrag { rho: 0.5 },
        ] {
            kinematic_extrapolate_into(&st, 100, &sched, &mut out).unwrap();
            sink += out[0];
            kinematic_extrapolate_capped_into(&st, 8, &sched, 2, &mut out).unwrap();
            sink += out[1];
        }
        kinematic_extrapolate_into(&st, 1, &Sched::Measured, &mut out).unwrap();
        snap = RegimeSnapshot::from_state(&st, &vel_before, 0.25, 0.75, -0.01);
        let v = clf.classify(&snap);
        sink += regime_code(&v) as f32;
        if let Some(ev) = mon.update(out[0] - x[0], &vel_before, &st.vel, 1.0) {
            sink += event_code(&ev) as f32;
        }
        sink += time_to_contact(0.75, 0.8125, 1.0);
        sink += time_to_contact_log(0.75, 0.8125, 1.0);
        let r = closest_approach(&p1, &v1, &p2, &v2);
        sink += r.t_star + r.miss_dist;
        sink += intercept_time(&p1, &p2, &v2, 2.0).unwrap_or(0.0);
        let hv = extrapolation_horizon(1.0, &eps, 10.0);
        sink += hv.k_star as f32 + hv.conf;
        sink += extrapolation_weight_ss(100, 3);
        sink += predictive_half_width(0.05, 100, 3, 0.05);
        sink += terminal_velocity(1.0, -0.5, 1.0, 0.5).unwrap_or(0.0);
    }

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    std::hint::black_box(&sink);
    assert_eq!(
        alloc_delta, 0,
        "steady-state allocs leaked ({alloc_delta} allocs / {dealloc_delta} deallocs)"
    );
    assert_eq!(dealloc_delta, 0);

    // Fixture pipeline check (serial, same thread — see its doc).
    fixture_pipeline_per_tick_alloc_free();
}

/// Allocation-free regime discriminant (a `format!` here would itself
/// allocate — the exact failure the first version of this test caught).
fn regime_code(r: &katgpt_core::kinematics::perception::Regime) -> u32 {
    use katgpt_core::kinematics::perception::Regime;
    match r {
        Regime::Uniform => 1,
        Regime::Parabolic { .. } => 2,
        Regime::Impulse => 3,
        Regime::Looming => 4,
        Regime::Drag => 5,
    }
}

fn event_code(e: &katgpt_core::kinematics::perception::EventKind) -> u32 {
    use katgpt_core::kinematics::perception::EventKind;
    match e {
        EventKind::Impulse { .. } => 10,
        EventKind::Drift { cusum } => 20 + cusum.to_bits(),
        EventKind::Spike { z, gate } => 30 + z.to_bits() + gate.to_bits(),
    }
}

/// G4 support: the full-fixture pipeline (generator + state + classifier +
/// monitor) allocates only for the Vec frames themselves — nothing per tick.
/// NOT a separate #[test]: parallel tests share the global counter (the
/// 7-phantom-alloc failure the split version caught — the fixture's Vec
/// allocations from the sibling thread landed in this window).
fn fixture_pipeline_per_tick_alloc_free() {
    let fix = fixture::generate(1, RangeLabel::Id);
    let mut st = KinState::<2>::new(1.0).unwrap();
    let mut clf = RegimeClassifier::new(RegimeConfig {
        g: fix.params.g,
        ..RegimeConfig::default()
    });
    let mut mon = ResidualMonitor::new(ResidualConfig::default());
    let mut prev_vel = [0.0f32; 2];
    let mut running_acc = 0.0f32;
    let mut prev_extent = fix.frames[0].extent;
    let mut predicted = [0.0f32; 2];

    // Warmup through the first 32 frames (lattice + classifier state).
    for f in fix.frames.iter().take(32) {
        let sigma_rate = f.extent - prev_extent;
        prev_extent = f.extent;
        kinematic_extrapolate_into(&st, 8, &Sched::Measured, &mut predicted).ok();
        let vel_before = st.vel;
        st.observe_into(&f.pos, f.tick).unwrap();
        let snap = RegimeSnapshot::from_state(&st, &prev_vel, running_acc, f.extent, sigma_rate);
        let _ = clf.classify(&snap);
        running_acc += 0.1 * (snap.acc_mag - running_acc);
        prev_vel = st.vel;
        let _ = mon.update(predicted[0] - f.pos[0], &vel_before, &st.vel, 1.0);
    }

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    for f in fix.frames.iter().skip(32) {
        let sigma_rate = f.extent - prev_extent;
        prev_extent = f.extent;
        kinematic_extrapolate_into(&st, 8, &Sched::Measured, &mut predicted).ok();
        let vel_before = st.vel;
        st.observe_into(&f.pos, f.tick).unwrap();
        let snap = RegimeSnapshot::from_state(&st, &prev_vel, running_acc, f.extent, sigma_rate);
        let v = clf.classify(&snap);
        running_acc += 0.1 * (snap.acc_mag - running_acc);
        prev_vel = st.vel;
        sink += regime_code(&v) as f32;
        if let Some(ev) = mon.update(predicted[0] - f.pos[0], &vel_before, &st.vel, 1.0) {
            sink += event_code(&ev) as f32;
        }
    }
    std::hint::black_box(&sink);
    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    assert_eq!(alloc_delta, 0, "per-tick allocs leaked ({alloc_delta})");
}

use std::hint::black_box;
