#![cfg(feature = "basis_split")]
//! Plan 625 T2.4 — the G4 zero-alloc canary over the whole split path.
//!
//! Separate test binary (the `#[global_allocator]` convention — crate-binary-
//! unique, would collide with the lib test binary's TrackingAllocator). The
//! counting allocator wraps System; the steady-state split path (accumulate →
//! finish → split → project → held_out_leakage) must allocate ZERO.
//!
//! ⚠ `linalg::symmetric_eig`'s `SymmetricEigScratch::ensure_capacity` grows
//! its buffers on first use — the harness WARMS UP once per (d, posture)
//! before the counted steady-state run (the same warm-then-count discipline
//! as the lib's TrackingAllocator gates).

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

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The two pipeline tests share ONE global allocator counter; libtest runs
/// them on parallel threads, so one test's SETUP allocations can land inside
/// the other's TRACKING window and be miscounted (measured post-931-round-3:
/// d16 reporting d64's setup allocs — 250/1 depending on scheduling). Hold
/// this lock for the WHOLE test body: the pipelines serialize, the counter
/// is never observed cross-test.
static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());
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

use katgpt_core::basis_split::{
    PairDifferenceMoment, Whitening, held_out_leakage, project_invariant_into, split,
};
use katgpt_core::linalg::symmetric_eig::SymmetricEigScratch;

/// Builds the planted fixture + all scratch, warms up once, then counts a
/// steady-state pipeline pass. Returns (allocs, verdict_ok).
fn counted_pipeline(d: usize, k: usize, n: usize) -> (usize, bool) {
    let mut rng = Lcg(0x6255_6201 + d as u64);
    let mut diffs: Vec<Vec<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut delta = vec![0.0; d];
        for (i, dv) in delta.iter_mut().enumerate() {
            if i >= k {
                *dv = rng.next_gauss();
            }
        }
        diffs.push(delta);
    }
    let held_refs: Vec<&[f64]> = diffs.iter().map(|dv| dv.as_slice()).collect();

    let mut m = vec![0.0; d * d];
    let mut eigvals = vec![0.0; d];
    let mut eigvecs = vec![0.0; d * d];
    let mut basis = vec![0.0; (d - k) * d];
    let mut sym = SymmetricEigScratch::new();
    let mut jacobi = vec![0.0; d * d];
    let mut m_scr = vec![0.0; d * d];
    let mut whitener = vec![0.0; d * d];
    let mut bx = vec![0.0; d - k];
    let mut proj = vec![0.0; d];
    let mut report_scratch = vec![0.0; d];
    let mut ident = vec![0.0; d * d];
    for i in 0..d {
        ident[i * d + i] = 1.0;
    }

    let mut run = || -> bool {
        let mut acc = PairDifferenceMoment::new(&mut m, d).unwrap();
        for delta in &diffs {
            acc.accumulate(delta).unwrap();
        }
        acc.finish();
        let (verdict, mut report) = split(
            &m,
            d,
            Some(Whitening::TotalCovariance(&ident)),
            Some(d - k),
            10.0,
            &mut eigvals,
            &mut eigvecs,
            &mut basis,
            &mut sym,
            &mut jacobi,
            &mut m_scr,
            &mut whitener,
        )
        .unwrap();
        if !matches!(
            verdict,
            katgpt_core::basis_split::Verdict::SufficientVariation { .. }
        ) {
            return false;
        }
        project_invariant_into(&basis, d - k, d, &diffs[0], &mut bx, &mut proj).unwrap();
        held_out_leakage(
            &basis,
            d - k,
            d,
            &held_refs,
            &mut report_scratch,
            &mut bx,
            &mut report,
        )
        .unwrap();
        true
    };

    // Warm-up (scratch capacities, allocator arenas) — uncounted.
    assert!(run(), "warm-up pass must split");
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING.store(1, Ordering::Relaxed);
    let ok = run();
    TRACKING.store(0, Ordering::Relaxed);
    (ALLOC_COUNT.load(Ordering::Relaxed), ok)
}

#[test]
fn g4_zero_alloc_steady_state_d64() {
    let _serialize = SERIALIZE.lock().unwrap();
    let (count, ok) = counted_pipeline(64, 16, 2048);
    assert!(ok);
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations on the steady-state split path (d=64)"
    );
}

#[test]
fn g4_zero_alloc_steady_state_d16_jacobi() {
    let _serialize = SERIALIZE.lock().unwrap();
    let (count, ok) = counted_pipeline(16, 4, 1024);
    assert!(ok);
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations on the steady-state Jacobi path (d=16)"
    );
}
