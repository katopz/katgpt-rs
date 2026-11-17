#![cfg(feature = "ternary_circuit")]
//! Issue 1225 Arm B — the counting-allocator canary over the builder's
//! apply paths (its OWN test target: the counter is process-global, so a
//! shared binary would let sibling fixtures' allocations pollute the
//! window — the Arm-A gate split, same convention).
//!
//! Pins: `ThresholdProgram::apply_into`, `CausalMatrix::scan_into` and
//! `PaleyProjection::apply_into` perform ZERO heap allocations in steady
//! state (one warm pass, then 50 counted passes each); compilation MAY
//! allocate and is outside the window. Positive control: a `Vec` push.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use katgpt_core::ternary_circuit_build::{CausalSpec, PaleySpec, RowSignMode, ThresholdSpec};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

#[test]
fn builder_apply_paths_are_zero_alloc_in_steady_state() {
    let mut pattern = [0_u8; 16];
    pattern[0] = b'a';
    pattern[1] = b'b';
    let prog = ThresholdSpec {
        pattern,
        len: 2,
        requires: 0b01,
        vetoes: 0b10,
        out_scale: 1.0,
    }
    .compile()
    .expect("compile");
    let causal = CausalSpec::ParityKernel.compile(64).expect("compile");
    let paley = PaleySpec {
        prime: 19,
        block: true,
        out_cols: 16,
        row_signs: RowSignMode::AllPlus,
    }
    .compile()
    .expect("compile");

    let line = *b"xab ";
    let conditions = [0_u8, 0b11, 0, 0];
    let mut out_f = [0.0_f32; 4];
    let x = [1_i32; 64];
    let mut y = [0_i32; 64];
    let px = [1_i32; 16];
    let mut py = [0_i32; 40];

    // Warm-up (any lazy init), then the steady window.
    prog.apply_into(&line, &conditions, &mut out_f);
    causal.scan_into(&x, &mut y);
    paley.apply_into(&px, &mut py);
    let before = allocs();
    for _ in 0..50 {
        prog.apply_into(&line, &conditions, &mut out_f);
        causal.scan_into(&x, &mut y);
        paley.apply_into(&px, &mut py);
    }
    assert_eq!(allocs(), before, "steady apply allocated");

    // Positive control: the counter fires.
    let before = allocs();
    let _v: Vec<u8> = Vec::with_capacity(16);
    assert!(allocs() > before, "the counting allocator is live");
}
