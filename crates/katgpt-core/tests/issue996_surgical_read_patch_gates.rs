#![cfg(feature = "formats_weights")]
//! Issue 996 surgical read-patch GATES (out-of-crate integration
//! battery).
//!
//! The `[[test]]` row in `Cargo.toml` carries
//! `required-features = ["formats_weights"]`, so without the umbrella
//! cargo SKIPS the target entirely (never a green zero); the
//! file-level `#![cfg]` is the second green-zero defense (the crate
//! convention: `required-features` protects the reader, the `#![cfg]`
//! the count).
//!
//! These gates need a binary-scoped counting allocator —
//! `#[global_allocator]` is binary-unique, so the G4 arm cannot live
//! in the lib test binary beside the crate's own tracking allocator.
//! The counter is THREAD-LOCAL (`Cell<u64>` per thread) so parallel
//! sibling tests in this binary cannot pollute the counted window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use katgpt_core::surgical_read_patch::{
    LayerPatch, Projection, SurgicalPatchArtifact, SurgicalPatchError, SurgicalReadPatch,
    construct_from_census_into,
};

thread_local! {
    static THREAD_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

/// Counts `alloc` + `realloc` per thread, then forwards to [`System`].
/// `try_with` (not `with`): a sibling thread's TLS may already be torn
/// down when late teardown allocations arrive — those must be
/// dropped silently, never panic the allocator.
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get().wrapping_add(1)));
        // SAFETY: a verbatim forward to `System` with the caller's
        // layout — sound by delegation.
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get().wrapping_add(1)));
        // SAFETY: `ptr` was allocated by the forwarded `System`
        // allocator with `layout`, and `new_size >= layout.size()` per
        // the `GlobalAlloc` contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came through this allocator chain (`System`)
        // under the same layout it was allocated with.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static G4_ALLOCATOR: CountingAllocator = CountingAllocator;

/// G4 (Issue 996): the merge path allocates NOTHING. 800 merges over
/// 4 layers × 2 projections inside the counted window after a warm-up
/// outside it — the counter must read exactly zero.
#[test]
fn g4_merge_path_is_alloc_free() {
    let d = 64usize;
    let rows = 128usize;
    let k = 2usize;
    let mut layers = Vec::new();
    for l in 1u32..=4 {
        layers.push(LayerPatch {
            layer_id: l,
            neurons: vec![l * 7 + 1, l * 7 + 3, l * 7 + 5],
            a: (0..k * d).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect(),
            gate: Some((0..3 * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect()),
            up: Some((0..3 * k).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect()),
        });
    }
    let patch = SurgicalReadPatch::from_parts(d, 1.0, layers).expect("fixture validates");
    let mut w: Vec<f32> = (0..rows * d)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
        .collect();

    // WARM outside the counted window (first-touch allocator state).
    patch
        .merge_projection_into(1, Projection::Gate, &mut w, rows)
        .expect("warm gate merge");
    patch
        .merge_projection_into(1, Projection::Up, &mut w, rows)
        .expect("warm up merge");

    THREAD_ALLOCS.with(|c| c.set(0));
    for _ in 0..100 {
        for l in 1u32..=4 {
            patch
                .merge_projection_into(l, Projection::Gate, &mut w, rows)
                .expect("gate merge");
            patch
                .merge_projection_into(l, Projection::Up, &mut w, rows)
                .expect("up merge");
        }
    }
    let allocs = THREAD_ALLOCS.with(Cell::get);
    assert_eq!(allocs, 0, "the merge path allocated {allocs} times");
}

/// The artifact lane end-to-end: merge directly, then merge through a
/// full mint → to_bytes → from_bytes → decode round trip — every
/// weight bit-identical — and a tampered wire byte refuses
/// [`SurgicalPatchError::DigestMismatch`].
#[test]
fn artifact_lane_end_to_end_bit_identical() {
    let d = 16usize;
    let rows = 12usize;
    let layers = vec![
        LayerPatch {
            layer_id: 1,
            neurons: vec![2, 5, 9],
            a: (0..2 * d).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect(),
            gate: Some((0..6).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect()),
            up: None,
        },
        LayerPatch {
            layer_id: 4,
            neurons: vec![0, 7],
            a: (0..2 * d).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect(),
            gate: Some(vec![0.5, 1.0, -0.25, 1.75]),
            up: Some(vec![-1.0, 0.5, 0.25, -0.75]),
        },
    ];
    let patch = SurgicalReadPatch::from_parts(d, 1.25, layers).expect("fixture validates");
    let mut direct: Vec<f32> = (0..rows * d)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect();
    let mut lane = direct.clone();

    patch
        .merge_projection_into(1, Projection::Gate, &mut direct, rows)
        .expect("direct merge");

    let artifact = SurgicalPatchArtifact::mint(&patch, 9);
    let bytes = artifact.to_bytes();
    let restored = SurgicalPatchArtifact::from_bytes(&bytes).expect("artifact round trip");
    assert_eq!(restored, artifact);
    let decoded = restored.decode().expect("decode");
    assert_eq!(decoded, patch);
    decoded
        .merge_projection_into(1, Projection::Gate, &mut lane, rows)
        .expect("lane merge");
    assert_eq!(direct.len(), lane.len());
    for (a, b) in direct.iter().zip(lane.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "artifact lane drifted from the direct merge"
        );
    }

    let mut tampered = bytes.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    assert!(matches!(
        SurgicalPatchArtifact::from_bytes(&tampered),
        Err(SurgicalPatchError::DigestMismatch { .. })
    ));
}

/// Lane (b) end-to-end: closed-form construction → assemble a
/// [`LayerPatch`] → validate → merge — the write-frozen law holds on
/// the constructed artifact (selected rows move, every unselected row
/// bit-identical).
#[test]
fn construct_to_merge_end_to_end() {
    let d = 8usize;
    let rows = 8usize;
    let persona_mean: Vec<f32> = (0..d).map(|i| i as f32 + 0.5).collect();
    let base_mean: Vec<f32> = (0..d).map(|i| i as f32 * 0.25).collect();
    let w: Vec<f32> = (0..rows * d)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();
    let neurons = [1u32, 4, 6];

    let mut a = vec![0.0f32; d];
    let mut b = vec![0.0f32; neurons.len()];
    construct_from_census_into(
        &persona_mean,
        &base_mean,
        &w,
        rows,
        &neurons,
        d,
        &mut a,
        &mut b,
    )
    .expect("construct");

    let patch = SurgicalReadPatch::from_parts(
        d,
        1.0,
        vec![LayerPatch {
            layer_id: 4,
            neurons: neurons.to_vec(),
            a,
            gate: Some(b),
            up: None,
        }],
    )
    .expect("assembled patch validates");

    let mut merged = w.clone();
    patch
        .merge_projection_into(4, Projection::Gate, &mut merged, rows)
        .expect("merge");

    for r in 0..rows {
        let selected = neurons.contains(&(r as u32));
        let base_row = &w[r * d..(r + 1) * d];
        let merged_row = &merged[r * d..(r + 1) * d];
        if selected {
            assert!(
                base_row
                    .iter()
                    .zip(merged_row)
                    .any(|(x, y)| x.to_bits() != y.to_bits()),
                "selected row {r} did not change"
            );
        } else {
            for (x, y) in base_row.iter().zip(merged_row) {
                assert_eq!(x.to_bits(), y.to_bits(), "unselected row {r} moved");
            }
        }
    }
}
