#![cfg(feature = "path_kernel_sim")]
//! Issue 949 T8 — the G4 zero-alloc canary over the fingerprint hot path.
//!
//! Separate test binary (the `#[global_allocator]` convention —
//! crate-binary-unique). The counting allocator wraps System; the steady
//! state of the HOT PATH (fingerprint_into + similarity, the per-decision
//! surfaces) must allocate ZERO with warm scratch.
//!
//! Scope note: `gram_spectrum_into` is a set-level DIAGNOSTIC (n² work,
//! once per anchor set), not a per-decision hot path — it allocates a small
//! descending-order Vec by design and is out of this gate's scope.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicUsize = AtomicUsize::new(0);
/// libtest runs tests on parallel threads; one test's SETUP allocations can
/// land inside the other's TRACKING window (the basis_split_g4 measured
/// precedent). Hold for the whole test body.
static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

use katgpt_core::path_kernel_sim::{FingerprintKernel, KnobScorer};
use katgpt_core::simd::{dot_f32_ordered, fast_sigmoid};

/// The canonical sigmoid-gated scorer (the T5a shape) at p = 64 knobs.
struct SigmoidGate64 {
    phi: [f32; 64],
}

impl SigmoidGate64 {
    fn new(x: &[f32; 64]) -> Self {
        let mut phi = [0.0f32; 64];
        for (p, v) in phi.iter_mut().zip(x.iter()) {
            *p = *v;
        }
        Self { phi }
    }
}

impl KnobScorer for SigmoidGate64 {
    fn knobs(&self) -> usize {
        64
    }
    fn eval(&self, _x: &[f32], w: &[f32]) -> f32 {
        fast_sigmoid(dot_f32_ordered(w, &self.phi))
    }
    fn grad_knobs_closed_form(&self, _x: &[f32], w: &[f32], out: &mut [f32]) -> bool {
        // σ'(z)·φ(x) with σ' from the same fast_sigmoid the scorer evaluates
        // with (the T2 closed-form protocol).
        let z = dot_f32_ordered(w, &self.phi);
        let s = fast_sigmoid(z);
        let s_prime = s * (1.0 - s);
        for (o, p) in out.iter_mut().zip(self.phi.iter()) {
            *o = s_prime * p;
        }
        true
    }
}

#[test]
fn g4_hot_path_zero_alloc() {
    let _serialize = SERIALIZE.lock().unwrap();
    let kernel = FingerprintKernel::default();

    // Anchor inputs + operating point.
    let mut x = [0.0f32; 64];
    let mut w0 = [0.0f32; 64];
    for i in 0..64 {
        x[i] = ((i % 7) as f32 - 3.0) * 0.25;
        w0[i] = ((i % 5) as f32 - 2.0) * 0.2;
    }
    let scorer = SigmoidGate64::new(&x);

    let mut fp_a = [0.0f32; 64];
    let mut fp_b = [0.0f32; 64];
    let mut w_scratch = w0;

    // Warm-up pass — uncounted (allocator arenas, any lazy state).
    let n1 = katgpt_core::path_kernel_sim::fingerprint_into(
        &kernel,
        &scorer,
        &x,
        &mut w_scratch,
        &mut fp_a,
    );
    let n2 = katgpt_core::path_kernel_sim::fingerprint_into(
        &kernel,
        &scorer,
        &x,
        &mut w_scratch,
        &mut fp_b,
    );
    assert!(n1 > 0.0 && n2 > 0.0);
    let _ = katgpt_core::path_kernel_sim::similarity(&kernel, &fp_a, &fp_b).unwrap();

    // Counted steady state: 100 fingerprint+similarity pairs.
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING.store(1, Ordering::Relaxed);
    let mut acc = 0.0f32;
    for _ in 0..100 {
        let na = katgpt_core::path_kernel_sim::fingerprint_into(
            &kernel,
            &scorer,
            &x,
            &mut w_scratch,
            &mut fp_a,
        );
        let nb = katgpt_core::path_kernel_sim::fingerprint_into(
            &kernel,
            &scorer,
            &x,
            &mut w_scratch,
            &mut fp_b,
        );
        acc += na + nb;
        acc += katgpt_core::path_kernel_sim::similarity(&kernel, &fp_a, &fp_b).unwrap();
    }
    TRACKING.store(0, Ordering::Relaxed);
    assert!(acc.is_finite());
    let count = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations on the steady-state hot path (p=64)"
    );
}
