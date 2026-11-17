#![cfg(feature = "ternary_circuit")]
//! Issue 1225 Arm A G4 — the counting-allocator canary over the landed
//! full-forward port's steady state.
//!
//! Pins (pre-registered in the issue):
//! - **G4-A** — [`forward_into`] (the zero-alloc apply path) performs ZERO
//!   heap allocations in steady state: after a 3-call warm-up at a fixed
//!   grid (which absorbs the scratch capacity growth and the two attention
//!   layers' v/hidden width changes, 96 → 128), 20 further calls allocate
//!   nothing — counted at the `#[global_allocator]` (alloc, realloc, and
//!   alloc_zeroed; dealloc is not an allocation).
//! - **Positive control** — the [`forward`] convenience wrapper DOES
//!   allocate (a fresh `ForwardOutput` per call). A counter that never
//!   fires pins nothing; this arm proves the counter is live.
//! - **Value stability** — the steady calls are bit-identical to the
//!   post-warm-up reference (every output surface compared by exact
//!   equality), so the alloc-free path is also value-stable.
//!
//! One `#[test]` fn, phases sequential: the counter is process-global, so
//! parallel test windows could pollute each other's counts.
//!
//! **Harness-noise guard:** libtest prints a one-shot "running for over 60
//! seconds" notice per test, and its `format!` allocates through THIS
//! binary's counting allocator — a noise event the window must never
//! overlap (measured: exactly 2 allocations, landing in-window on a fast
//! run and pre-window on a slow one — a pure wall-clock flake). The notice
//! is one-shot at t=60s, so the gate holds the window closed until the
//! test has been alive > 65s: at any profile and any box load the notice
//! has already fired before the count resets. Debug is the default arm
//! (~2.5 min), release the cheap re-check (~1 min).
//!
//! The loader is duplicated from `ternary_circuit_forward_gates.rs` to
//! keep the two gates independent (that file's own convention); fixture
//! integrity (the BLAKE3 pins) is the exactness gate's jurisdiction.

use std::alloc::{GlobalAlloc, Layout};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use katgpt_core::kpt_archive::KptArchive;
use katgpt_core::ternary_circuit::parse_meta;
use katgpt_core::ternary_circuit_forward::{
    CodeHygieneModel, ForwardOutput, ForwardScratch, MAX_COLS, encode_byte_grid,
};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: layout is valid (caller's contract); System.alloc is sound.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr was allocated by System with this layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: ptr was allocated by System with this layout; new_size is
        // the caller's contract.
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: layout is valid (caller's contract).
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The fixture model, loaded through the same verify ladder as the
/// exactness gate (meta grammar → archive verify → param resolution).
fn load_model() -> CodeHygieneModel {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ternary_circuit");
    let meta_text =
        String::from_utf8(std::fs::read(dir.join("intermediate/meta.txt")).expect("meta file"))
            .expect("meta utf8");
    let meta = parse_meta(meta_text.trim()).expect("meta grammar");
    let archive_bytes = std::fs::read(dir.join("compiled/weights.cact")).expect("archive file");
    let leaked: &'static [u8] = Box::leak(archive_bytes.into_boxed_slice());
    let archive = KptArchive::from_bytes(leaked).expect("archive verify ladder");
    CodeHygieneModel::load(&archive, &meta).expect("param resolution")
}

/// A deterministic ~40-line Rust-ish snippet (varied indentation, a doc
/// comment, a match) — the grid shape exercises every block path; the
/// fixed line count only fixes the buffer shapes.
fn synthetic_grid() -> Vec<[u8; MAX_COLS]> {
    let mut code = String::new();
    for i in 0..4 {
        code.push_str(&format!(
            "/// item {i} — a stable hash for the audit trail\n\
             pub fn stable_hash_{i}(input: &[u8]) -> u64 {{\n\
             \x20   let mut state: u64 = 0xcbf2_9ce4_8422_2325;\n\
             \x20   for (n, byte) in input.iter().enumerate() {{\n\
             \x20       state ^= u64::from(*byte);\n\
             \x20       match n % 3 {{\n\
             \x20           0 => {{ state = state.rotate_left(5); }}\n\
             \x20           _ => state = state.wrapping_mul(0x100_0000_01b3),\n\
             \x20       }}\n\
             \x20   }}\n\
             \x20   state\n\
             }}\n"
        ));
    }
    encode_byte_grid(&code)
}

fn outputs_equal(a: &ForwardOutput, b: &ForwardOutput) -> bool {
    a.line_probs == b.line_probs
        && a.noul == b.noul
        && a.subscores == b.subscores
        && a.style_probs == b.style_probs
}

#[test]
fn forward_alloc_gates() {
    let born = std::time::Instant::now();
    let model = load_model();
    let grid = synthetic_grid();
    let lines = grid.len();
    assert!(
        lines >= 30,
        "the synthetic grid should be a realistic multi-line shape, got {lines}"
    );

    let mut scratch = ForwardScratch::default();
    // Seed the output through the convenience form (outside the counting
    // window): its `line_probs` arrives with the steady-state capacity, and
    // the const-valued head arrays need no re-declaration here.
    let mut out: ForwardOutput = model.forward(&grid, &mut scratch).expect("seed forward");

    // Warm-up: the first calls grow the scratch capacities (including the
    // two attention layers' differing v/hidden widths inside one call) and
    // settle the output row's capacity. Three calls prove the capacities
    // have settled before the counting window opens.
    for _ in 0..3 {
        model
            .forward_into(&grid, &mut scratch, &mut out)
            .expect("warm-up forward_into");
    }
    let reference = out.clone();

    // Discharge the harness's one-shot 60s notice BEFORE the window: hold
    // until the test has been alive > 65s (the notice fires on the harness
    // main thread at t=60s and its format! allocates here — see the file
    // doc). On a slow box the lifetime is already past 65s and the pad is
    // a no-op.
    let alive = born.elapsed();
    let floor = std::time::Duration::from_secs(65);
    if alive < floor {
        std::thread::sleep(floor - alive);
    }

    // Phase 1 — G4-A: the steady state. No formatting, no panic paths, no
    // allocation of any kind inside the window.
    ALLOCS.store(0, Ordering::Relaxed);
    for _ in 0..10 {
        model
            .forward_into(&grid, &mut scratch, &mut out)
            .expect("steady forward_into");
        assert!(
            outputs_equal(&out, &reference),
            "steady-state drift: a value-stable alloc-free path is the pin"
        );
    }
    let steady = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(
        steady, 0,
        "G4-A: forward_into allocated {steady} time(s) in steady state \
         (10 calls at {lines} lines)"
    );

    // Phase 2 — positive control: the convenience wrapper allocates (a
    // fresh ForwardOutput per call). Proves the counter fires; a counter
    // that never fires would have pinned nothing in phase 1.
    ALLOCS.store(0, Ordering::Relaxed);
    let _ = model.forward(&grid, &mut scratch).expect("control forward");
    let control = ALLOCS.load(Ordering::Relaxed);
    assert!(
        control > 0,
        "positive control failed: forward() allocated nothing — \
         the counter is blind and this gate pins nothing"
    );
}
