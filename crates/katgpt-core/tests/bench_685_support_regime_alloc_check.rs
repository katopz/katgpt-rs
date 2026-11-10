//! Issue 693 / Bench 685 — G4 zero-alloc steady state for the
//! support-regime PoC loop (trajectory projection → `iou` → detector push).
//!
//! Separate single-purpose binary (the CountingAllocator global pattern —
//! `bench_680_kinematic_alloc_check` / `bench_576` convention): parallel
//! tests share the global counter, so the check lives in ONE test function,
//! serial by construction. (The lib-test TrackingAllocator check for the
//! module's own push loop lives in the module's unit tests.)

#![cfg(feature = "functional_substitution_gate")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::functional_substitution::support_instability::{
    SupportInstabilityDetector, support_instability,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

const D: usize = 64;
const N_ZONES: usize = 4;
const TICKS: usize = 2000;

// ── the exact toy-world construction from bench_685_support_regime_goat ──
// (duplicated verbatim so this binary is standalone; the fixture is
// deterministic and the parent binary's G1 test pins the generator
// bit-identity — any drift here shows up as a detect-rate change there).

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
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
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 65536.0 / 65536.0
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

type ZoneW = [[f32; 3]; D];

fn gen_zone_weights() -> [ZoneW; N_ZONES] {
    let mut rng = Lcg::new(0x0693_5131);
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

fn project_into(z: &mut [f32; D], w: &ZoneW, x: f32, y: f32) {
    for (zi, wi) in z.iter_mut().zip(w.iter()) {
        *zi = (wi[0] * x + wi[1] * y + wi[2]).max(0.0);
    }
}

fn zone_of(x: f32, y: f32) -> u8 {
    (x >= 0.5) as u8 + 2 * (y >= 0.5) as u8
}

const HOME_MARGIN: f32 = 0.08;
const CROSS_STEP: f32 = 0.03;
const WIGGLE: f32 = 0.02;
const ARRIVE: f32 = 0.05;

fn adjacent_zone_of(zone: u8, pick: u32) -> u8 {
    match zone {
        0 => [1u8, 2][pick as usize & 1],
        1 => [0u8, 3][pick as usize & 1],
        2 => [0u8, 3][pick as usize & 1],
        _ => [1u8, 2][pick as usize & 1],
    }
}

fn zone_center(zone: u8) -> (f32, f32) {
    match zone {
        z @ (0..=3) => (0.25 + 0.5 * (z % 2) as f32, 0.25 + 0.5 * (z / 2) as f32),
        _ => (0.25, 0.25),
    }
}

fn home_in(zone: u8, rng: &mut Lcg) -> (f32, f32) {
    let (cx, cy) = zone_center(zone);
    (
        rng.range(cx - HOME_MARGIN, cx + HOME_MARGIN),
        rng.range(cy - HOME_MARGIN, cy + HOME_MARGIN),
    )
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

fn wiggle_step(rng: &mut Lcg) -> (f32, f32) {
    let theta = rng.unit() * std::f32::consts::TAU;
    (WIGGLE * theta.cos(), WIGGLE * theta.sin())
}

/// Positions + zones per tick (the latent is projected on the fly — the
/// same loop shape the parent binary's T3 runs).
fn gen_trace(seed: u64) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let mut rng = Lcg::new(seed);
    let mut xs: Vec<f32> = Vec::with_capacity(TICKS);
    let mut ys: Vec<f32> = Vec::with_capacity(TICKS);
    let mut zones: Vec<u8> = Vec::with_capacity(TICKS);
    let push = |xs: &mut Vec<f32>, ys: &mut Vec<f32>, zones: &mut Vec<u8>, x: f32, y: f32| {
        xs.push(x);
        ys.push(y);
        zones.push(zone_of(x, y));
    };

    let mut zone = (rng.next_u32() & 3) as u8;
    let (mut x, mut y) = home_in(zone, &mut rng);
    push(&mut xs, &mut ys, &mut zones, x, y);

    while xs.len() < TICKS {
        let dwell = 120 + rng.next_u32() % 101;
        let (zx0, zx1, zy0, zy1) = zone_inner_box(zone);
        for _ in 0..dwell {
            if xs.len() >= TICKS {
                break;
            }
            let (dx, dy) = wiggle_step(&mut rng);
            x = (x + dx).clamp(zx0, zx1);
            y = (y + dy).clamp(zy0, zy1);
            push(&mut xs, &mut ys, &mut zones, x, y);
        }
        if xs.len() >= TICKS {
            break;
        }
        let target = adjacent_zone_of(zone, rng.next_u32());
        let (tx, ty) = home_in(target, &mut rng);
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
            push(&mut xs, &mut ys, &mut zones, x, y);
        }
        zone = target;
    }
    (xs, ys, zones)
}

/// G4: the full PoC loop (projection → iou → push) allocates nothing per
/// tick — only the trace Vecs themselves (allocated in the generator,
/// outside the measured window) may touch the allocator.
#[test]
fn g4_poc_loop_alloc_free() {
    let w = gen_zone_weights();
    let (xs, ys, zones) = gen_trace(0x0693_0001);

    // Warmup: one full pass settles any lazy machinery + proves the loop
    // compiles/executes before the measured window opens.
    {
        let mut z_prev = [0.0f32; D];
        let mut z_cur = [0.0f32; D];
        let mut det = SupportInstabilityDetector::new();
        project_into(&mut z_prev, &w[zones[0] as usize], xs[0], ys[0]);
        for t in 1..xs.len() {
            project_into(&mut z_cur, &w[zones[t] as usize], xs[t], ys[t]);
            let _ = det.push(support_instability(&z_prev, &z_cur));
            z_prev.copy_from_slice(&z_cur);
        }
    }

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    let mut z_prev = [0.0f32; D];
    let mut z_cur = [0.0f32; D];
    let mut det = SupportInstabilityDetector::new();
    let mut sink = 0u64;
    project_into(&mut z_prev, &w[zones[0] as usize], xs[0], ys[0]);
    for t in 1..xs.len() {
        project_into(&mut z_cur, &w[zones[t] as usize], xs[t], ys[t]);
        let inst = support_instability(&z_prev, &z_cur);
        sink += det.push(inst) as u8 as u64;
        z_prev.copy_from_slice(&z_cur);
    }
    black_box(&sink);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    assert_eq!(
        alloc_delta, 0,
        "PoC loop leaked {alloc_delta} allocs ({dealloc_delta} deallocs)"
    );
    assert_eq!(dealloc_delta, 0);
}
