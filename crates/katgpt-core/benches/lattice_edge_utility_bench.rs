//! Plan 335 Phase 6 — GOAT gate, leaf-level lattice op bench (T6.2/T6.4).
//!
//! This is the **leaf** half of the GOAT gate. `katgpt-core` is the leaf of the
//! stack (riir-neuron-db → riir-ai depend downward on it), so it CANNOT reference
//! `ZoneGeometryPod`, `ValidatedZoneView`, `ZoneEggshellRuntime`, or
//! `pathfinder::find_path`. Those integrated-runtime benches live in
//! `riir-ai/crates/riir-engine/benches/zone_eggshell_goat.rs`.
//!
//! # Gates covered here
//!
//! - **G2 (lattice-side latency).** `lattice_edge_utility_into` on a 16×16 grid
//!   (480 edges) with raw f32 lanes. The plan's headline gate (eggshell
//!   dominates A* on latency) is evaluated in the integrated bench; here we
//!   report the per-tick lattice-op latency in isolation so the integrated
//!   bench can subtract it from the full eggshell pipeline.
//! - **G4 (zero-alloc hot path).** Verified by **code review** of
//!   `lattice_edge_utility_into` (see G4_NOTE below): the function body has no
//!   `Vec::new`, `Box`, `.collect()`, `format!`, or any other heap allocation.
//!   It reads input slices and writes to the output slice only. Runtime
//!   allocator instrumentation is out of scope for this GOAT pass (would require
//!   a global allocator wrapper).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features lattice_utility \
//!   --bench zone_lattice_goat -- --warm-up-time 1 --measurement-time 2 --sample-size 100
//! ```

#![cfg(feature = "lattice_utility")]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::dec::lattice_utility::{HlaToCohainWeights, lattice_edge_utility_into};

// ── G4 NOTE (zero-alloc hot path, code-review verdict) ───────────────────
//
// The body of `lattice_edge_utility_into` (crates/katgpt-core/src/dec/
// lattice_utility.rs:193-287) performs:
//   1. `debug_assert!` bounds checks (compile-time, no runtime alloc).
//   2. A `for e in 0..n_edges` loop reading from input slices via
//      `get_unchecked` and writing to `out_edge_utility.get_unchecked_mut(e)`.
//   3. A final `simd_sigmoid_inplace(&mut out_edge_utility[..n_edges])` call,
//      which operates in-place on the borrowed slice.
//
// No `Vec::new`, `Box::new`, `.collect()`, `format!`, `String`, `to_vec()`, or
// any other heap-allocating construct appears in the function body. The
// function takes `&mut [f32]` and writes into it — pure in-place.
//
// G4 VERDICT: PASS by construction. Runtime allocator instrumentation deferred.

/// Grid dimensions for the lattice op bench. 16×16 vertices → 480 edges
/// (15·16 horizontal + 16·15 vertical), matching the plan's G2 grid size.
const GRID_W: usize = 16;
const GRID_H: usize = 16;

/// Build the raw lanes + index arrays for a `GRID_W × GRID_H` vertex grid,
/// matching `grid_edge_topology` ordering (horizontal-first, then vertical).
fn build_16x16_lattice() -> (
    Vec<f32>, // interest_lane (n_vertices)
    Vec<f32>, // safety_lane
    Vec<f32>, // occupancy_lane (n_faces)
    Vec<f32>, // threat_lane (n_edges)
    Vec<f32>, // destruction_lane
    Vec<u32>, // edge_src_vertex_idx
    Vec<u32>, // edge_face_idx
    Vec<f32>, // out_edge_utility (scratch)
) {
    let n_vertices = GRID_W * GRID_H;
    let n_faces = (GRID_W - 1) * (GRID_H - 1);
    let n_h_edges = (GRID_W - 1) * GRID_H;
    let n_v_edges = GRID_W * (GRID_H - 1);
    let n_edges = n_h_edges + n_v_edges;

    // Deterministic, non-trivial lane values (no all-zero degenerate case).
    let mut interest_lane = vec![0.0f32; n_vertices];
    let mut safety_lane = vec![0.0f32; n_vertices];
    let mut destruction_lane = vec![0.0f32; n_vertices];
    for i in 0..n_vertices {
        let f = i as f32;
        interest_lane[i] = 0.3 + 0.1 * (f * 0.7).sin();
        safety_lane[i] = 0.5 + 0.2 * (f * 0.3).cos();
        destruction_lane[i] = 0.1 + 0.05 * (f * 1.1).sin();
    }

    let mut occupancy_lane = vec![0.0f32; n_faces];
    for f in 0..n_faces {
        occupancy_lane[f] = 0.4 + 0.3 * ((f as f32) * 0.5).sin().abs();
    }

    let mut threat_lane = vec![0.0f32; n_edges];
    for e in 0..n_edges {
        threat_lane[e] = 0.2 + 0.15 * ((e as f32) * 0.9).cos().abs();
    }

    // Reconstruct edge index arrays matching `grid_edge_topology` (same
    // ordering as riir-ai's `grid_edge_topology`). Horizontal edges first.
    let mut edge_src_vertex_idx = Vec::with_capacity(n_edges);
    let mut edge_face_idx = Vec::with_capacity(n_edges);

    // Horizontal edges: y in [0, GRID_H), x in [0, GRID_W-1).
    for y in 0..GRID_H {
        for x in 0..(GRID_W - 1) {
            let src = (y * GRID_W + x) as u32;
            edge_src_vertex_idx.push(src);
            let face = if y < GRID_H - 1 {
                y * (GRID_W - 1) + x
            } else {
                (y - 1) * (GRID_W - 1) + x
            };
            edge_face_idx.push(face as u32);
        }
    }
    // Vertical edges: y in [0, GRID_H-1), x in [0, GRID_W).
    for y in 0..(GRID_H - 1) {
        for x in 0..GRID_W {
            let src = (y * GRID_W + x) as u32;
            edge_src_vertex_idx.push(src);
            let face = if x < GRID_W - 1 {
                y * (GRID_W - 1) + x
            } else {
                y * (GRID_W - 1) + (x - 1)
            };
            edge_face_idx.push(face as u32);
        }
    }

    assert_eq!(edge_src_vertex_idx.len(), n_edges);
    assert_eq!(edge_face_idx.len(), n_edges);

    let out = vec![0.0f32; n_edges];
    (
        interest_lane,
        safety_lane,
        occupancy_lane,
        threat_lane,
        destruction_lane,
        edge_src_vertex_idx,
        edge_face_idx,
        out,
    )
}

/// G2 (lattice-side) — `lattice_edge_utility_into` on 480 edges.
///
/// Reports per-call latency for the full lattice op (gather + FMA chain +
/// vectorized sigmoid). This is the leaf-level cost the integrated eggshell
/// pipeline pays per tick; the integrated bench subtracts overhead (cache
/// lookup, triple emission) on top.
fn bench_lattice_edge_utility_16x16(c: &mut Criterion) {
    let (
        interest_lane,
        safety_lane,
        occupancy_lane,
        threat_lane,
        destruction_lane,
        edge_src_vertex_idx,
        edge_face_idx,
        mut out,
    ) = build_16x16_lattice();
    let weights = HlaToCohainWeights::default();

    let n_edges = edge_src_vertex_idx.len();
    let mut group = c.benchmark_group("lattice_edge_utility");
    group.throughput(Throughput::Elements(n_edges as u64));
    group.sample_size(300);

    group.bench_function("16x16_grid_480_edges", |b| {
        b.iter(|| {
            lattice_edge_utility_into(
                black_box(&interest_lane),
                black_box(&safety_lane),
                black_box(&occupancy_lane),
                black_box(&threat_lane),
                black_box(&destruction_lane),
                black_box(&edge_src_vertex_idx),
                black_box(&edge_face_idx),
                black_box(&weights),
                black_box(&mut out),
            );
        });
    });

    group.finish();
}

criterion_group!(benches, bench_lattice_edge_utility_16x16,);
criterion_main!(benches);
