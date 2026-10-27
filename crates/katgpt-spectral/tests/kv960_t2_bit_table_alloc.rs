//! Issue 960 T2 G4 — the decode-path allocation LOOKUP is zero-alloc in the
//! steady state. `BitTable::bits` is the surface a serving consumer reads
//! per token (one `u8` per channel); this binary pins 0 allocations over a
//! hot lookup loop. SINGLE-TEST BINARY by construction: the counting
//! allocator is per-process, so any sibling test in here would allocate
//! inside the counting window and false-red the pin. Calibration-class
//! work (`allocate_bits` itself) is exempt from the steady-state standard
//! by the issue's own G4 wording — it runs BEFORE the counted window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use katgpt_spectral::kv_sensitivity_alloc::{allocate_bits, behavioral_waterfill_weights};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn bit_table_lookup_is_zero_alloc() {
    // Calibration (exempt, before the counted window).
    let weights = behavioral_waterfill_weights(&[0.9, 0.2, 0.5, 0.1, 0.0, 0.3], 2);
    let table = allocate_bits(&weights, 2.0, 1, 8);
    assert_eq!(table.len(), 6);

    // Warm any lazy per-test machinery, then count.
    let mut acc = 0u32;
    acc += table.bits(0) as u32;
    ALLOCS.store(0, Ordering::SeqCst);

    let rounds = 10_000u32;
    for i in 0..rounds {
        // The decode-path shape: per token, per channel, an O(1) read.
        acc = acc.wrapping_add(table.bits((i as usize) % table.len()) as u32);
        acc = acc.wrapping_add(table.as_slice()[(i as usize) % table.len()] as u32);
    }

    let n_alloc = ALLOCS.load(Ordering::SeqCst);
    assert!(
        n_alloc == 0,
        "G4: {n_alloc} allocs over {rounds} lookup rounds"
    );
    // The checksum is data-dependent on the reads (bits() panics OOB), so
    // the loop cannot be deleted; printing it keeps it live.
    println!("G4 PASS — 0 allocs over {rounds} BitTable lookup rounds (checksum {acc})");
}
