//! The graft apply hot path's G4 alloc census (issue 988 T4's mechanical
//! leg, run at T1 landing — the apply path is the zero-alloc claim).
//!
//! Separate test binary (the `#[global_allocator]` convention —
//! crate-binary-scoped; see `tests/basis_split_g4_alloc.rs` for the
//! shared-buffer discipline this file copies).

#![cfg(feature = "grafts")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// One test body — the SERIALIZE lock still guards the convention (a
/// second test joining this binary later shares the counter).
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

use katgpt_core::graft::{
    ActivationReader, CellRule, Graft, GraftRead, GraftSwitch, GraftTable, GraftWrite, LogitWriter,
    ReadSegment, WriteEntry,
};

struct FixtureReader;

impl ActivationReader for FixtureReader {
    fn read(&self, layer: u32) -> &[f32] {
        match layer {
            0 => &[2.0, 4.0, 0.0, 0.0],
            _ => &[9.0],
        }
    }
}

struct NullWriter(usize);

impl LogitWriter for NullWriter {
    fn add_bias(&mut self, _token: u32, _bias: f32) {
        self.0 += 1;
    }
}

fn built_graft() -> Graft {
    let reads = vec![
        GraftRead::new(
            vec![ReadSegment {
                layer: 0,
                weights: vec![1.0, 0.0, 0.0, 0.0],
            }],
            0.0,
            5,
        )
        .expect("read0"),
        GraftRead::new(
            vec![ReadSegment {
                layer: 0,
                weights: vec![0.0, 1.0, 0.0, 0.0],
            }],
            0.0,
            5,
        )
        .expect("read1"),
    ];
    let mut entries = Vec::with_capacity(25);
    for a in 0..5u16 {
        for b in 0..5u16 {
            // Key packing mirrors graft::pack_key (v0 | v1<<16).
            entries.push((u64::from(a) | (u64::from(b) << 16), (a + b) % 5));
        }
    }
    let rule = CellRule::new(0, &[0, 1], entries).expect("rule");
    let table = GraftTable::from_rules(vec![rule]).expect("table");
    let writes = vec![
        GraftWrite::new(
            0,
            (0..5u16)
                .map(|v| WriteEntry {
                    value: v,
                    biases: vec![(100 + u32::from(v), 1.0)],
                })
                .collect(),
        )
        .expect("write"),
    ];
    let switch = GraftSwitch::new(
        vec![ReadSegment {
            layer: 1,
            weights: vec![1.0],
        }],
        0.0,
        0.5,
    )
    .expect("switch");
    Graft::new(reads, table, writes, vec![switch]).expect("graft")
}

#[test]
fn graft_apply_is_zero_alloc_in_steady_state() {
    let _guard = SERIALIZE.lock().expect("serialize");
    let graft = built_graft();
    let mut input_vals = [0u16; 2];
    let mut out = [0u16; 1];
    let mut writer = NullWriter(0);
    // Warm pass (any lazy one-time setup lands OUTSIDE the tracking window).
    let warm = graft.apply(&FixtureReader, &mut writer, &mut input_vals, &mut out);
    assert!(warm.fired, "warm apply fires");
    assert_eq!(writer.0, 1, "one decided-value bias per fired apply");

    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING.store(1, Ordering::Relaxed);
    let applied = graft.apply(&FixtureReader, &mut writer, &mut input_vals, &mut out);
    TRACKING.store(0, Ordering::Relaxed);
    assert!(applied.fired);
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(allocs, 0, "the apply hot path allocated {allocs} times");
}
