#![cfg(feature = "sst_project")]
//! Issue 958 — the G4 zero-alloc canary over the armed hot paths.
//!
//! Separate test binary (the `#[global_allocator]` convention — crate-
//! binary-unique). The counting allocator wraps System; after one warm
//! call per path (scratch grow-on-demand), the steady-state anatomy /
//! factored-anatomy / factor / certify / apply / decode / compose paths
//! must allocate ZERO over 1000 cycles.
//!
//! NOT counted (calibration/shipping-time by contract, documented in the
//! module docs): `fit_anchor` (returns owned bases; f64 eigensolver work)
//! and `quantized` (clones the factor by definition).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) == 1 {
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

/// Deterministic LCG (the unseeded-global-RNG gate).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

use katgpt_core::sst_certified::{
    SstAnchor, SstScratch, TailFactor, compose_tail_into, fit_anchor_into,
};

const M: usize = 32;
const N: usize = 64;
const FREE: usize = 8;
const CYCLES: usize = 1000;

#[test]
fn g4_zero_alloc_armed_paths_1000_cycles() {
    // ── setup (uncounted): fixture + warm every path once ──
    let mut rng = Lcg(0x958_6004);
    let w: Vec<f32> = (0..M * N).map(|_| rng.next_gauss() as f32).collect();
    let delta: Vec<f32> = (0..M * N).map(|_| rng.next_gauss() as f32).collect();
    let delta2: Vec<f32> = (0..M * N).map(|_| rng.next_gauss() as f32).collect();

    let mut anchor_scratch = SstScratch::default();
    let anchor: SstAnchor = fit_anchor_into(&w, M, N, FREE, &mut anchor_scratch).expect("anchor");

    let mut scratch = SstScratch::default();
    let mut f1 = TailFactor::default();
    let mut f2 = TailFactor::default();
    let mut report = katgpt_core::sst_certified::SstFactorReport::default();
    anchor
        .factor_tail_into(&w, &delta, FREE, FREE, &mut f1, &mut report, &mut scratch)
        .expect("factor1");
    anchor
        .factor_tail_into(&w, &delta2, FREE, FREE, &mut f2, &mut report, &mut scratch)
        .expect("factor2");
    let mut composed = TailFactor::default();
    compose_tail_into(&[&f1, &f2], &mut composed).expect("compose");
    let mut decoded = vec![0.0f32; M * N];
    let mut applied = vec![0.0f32; M];
    let x = vec![1.0f32; N];
    let eye_r: Vec<f32> = {
        let mut e = vec![0.0f32; N * N];
        for i in 0..N {
            e[i * N + i] = 1.0;
        }
        e
    };

    // Warm pass (scratch capacities settled) — uncounted.
    {
        let mut s = scratch;
        let _ = anchor.anatomy_into(&delta, &mut s).expect("warm anatomy");
        let _ = anchor
            .anatomy_factored_into(&delta, &eye_r, N, &mut s)
            .expect("warm factored");
        let mut f = TailFactor::default();
        let mut rep = report;
        anchor
            .factor_tail_into(&w, &delta, FREE, FREE, &mut f, &mut rep, &mut s)
            .expect("warm factor");
        let _ = anchor
            .certify_decoded_tail_into(&delta, 1e-3, &mut s)
            .is_err(); // may refuse (dense delta) — the PATH is what warms
        f1.apply_into(&x, 1, &mut applied, &mut s);
        f1.decode_into(&mut decoded, &mut s);
        compose_tail_into(&[&f1, &f2], &mut composed).expect("warm compose");
        scratch = s;
    }

    // ── steady state: 1000 cycles over every armed path, counted ──
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING.store(1, Ordering::Relaxed);
    {
        let mut s = scratch;
        for _ in 0..CYCLES {
            let a = anchor.anatomy_into(&delta, &mut s).expect("anatomy");
            assert!(a.total_energy.is_finite());
            let af = anchor
                .anatomy_factored_into(&delta, &eye_r, N, &mut s)
                .expect("factored");
            assert!(af.total_energy.is_finite());
            anchor
                .factor_tail_into(&w, &delta, FREE, FREE, &mut f1, &mut report, &mut s)
                .expect("factor");
            // certify may refuse (the delta is dense) — refusal is a VALUE
            // here, never an allocation.
            let _ = anchor.certify_decoded_tail_into(&delta, 1e-3, &mut s);
            f1.apply_into(&x, 1, &mut applied, &mut s);
            f1.decode_into(&mut decoded, &mut s);
            compose_tail_into(&[&f1, &f2], &mut composed).expect("compose");
        }
        // (scratch's grown state stays local to this block — the counted loop
        // was the last consumer; moving it back would be a dead store.)
        drop(s);
    }
    TRACKING.store(0, Ordering::Relaxed);
    let count = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations over {CYCLES} armed-path cycles"
    );
}
