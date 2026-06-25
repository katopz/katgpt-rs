//! Stokes Calculus Wrappers — GOAT gate benchmarks (Plan 314 Phase 3).
//!
//! Three benchmark groups covering the latency and A/B comparison targets
//! specified in the plan:
//!
//! - **`belief_mass_divergence/per_edge`** — per-edge cost of the L1 divergence
//!   validator. No formal target (G-A runs in riir-ai); this is a perf baseline.
//! - **`boundary_flux_mass/G-B_256x256`** — A/B: `boundary_flux_mass_only` vs
//!   naive full-volume summation (`exterior_derivative` + region sum) on a
//!   256×256 game map with a 64×64 zone. Target: ≥3× faster with
//!   `error_bound/mass < 5%`.
//! - **`line_integral/G-C_path_cost`** — A/B: `line_integral` latency per path
//!   step, plus a reversal-count comparison between a zigzag path and a smooth
//!   path (validates `line_integral` as a discriminating cost function).
//!   Target: ≥20% fewer reversals when `line_integral`-weighted.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features dec_operators \
//!   --bench stokes_calculus_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//!
//! # Feature gate
//!
//! Requires `dec_operators` (the underlying DEC operators from Plan 251).

#![cfg(feature = "dec_operators")]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::dec::operators::{codifferential_into, exterior_derivative_into};
use katgpt_core::dec::{
    CellComplex, CochainField, belief_mass_divergence, boundary_flux_mass_indexed,
    boundary_flux_mass_only, circulation_integral, line_integral,
};

// ─── G-A baseline: belief_mass_divergence per-edge ──────────────────────────

fn bench_belief_mass_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/belief_mass_divergence");

    // 32×32 grid: 1024 vertices, 1984 edges. Realistic NPC belief manifold.
    let cx = CellComplex::grid_2d(32, 32);
    let n_edges = cx.n_edges();
    group.throughput(Throughput::Elements(n_edges as u64));

    // Constant flow field (all edges = 1.0) — divergence is zero at interior
    // vertices, non-zero at boundary. Exercises the full codifferential path.
    let field = CochainField::from_vec(1, 1, vec![1.0f32; n_edges]);

    group.bench_function("32x32_constant_flow", |b| {
        b.iter(|| {
            let div = belief_mass_divergence(black_box(&cx), black_box(&field));
            black_box(div);
        });
    });

    group.finish();
}

// ─── G-B: boundary_flux_mass_only vs naive volume sum ───────────────────────
//
// On a 256×256 game map (Bomber arena scale), compute the boundary-flux mass
// of a 64×64 zone (4096 faces) via two methods:
//
// A (boundary): `boundary_flux_mass_only` — single pass over B₂ entries with
//    region-membership filter. Interior edges cancel; only boundary survives.
// B (naive): `exterior_derivative_into` (compute d₁(field) over all faces)
//    then sum the region-face subset.
//
// Target: A ≥ 3× faster than B with error_bound/mass < 5%.

fn build_256x256_grid_and_field() -> (CellComplex, CochainField, Vec<u32>) {
    let w = 256;
    let h = 256;
    let cx = CellComplex::grid_2d(w, h);

    // Swirl (tangential) edge field: counterclockwise flow around the grid
    // center. This field has NON-ZERO curl (= 2 everywhere in the continuum),
    // so `boundary_flux_mass` produces a non-zero mass. Verified:
    //   mass ≈ 8192 (= 2 × 4096 region faces), error_bound ≈ 310, ratio ≈ 3.8%.
    //
    // Horizontal edge (y, x) → value = -(y - cy)  [tangential -dy component]
    // Vertical edge   (y, x) → value =  (x - cx)  [tangential +dx component]
    //
    // This is the correct test field for G-B: it has non-trivial curl (so the
    // boundary flux is non-zero and meaningful) while keeping the harmonic
    // component small (error_bound/mass < 5%).
    let cx_coord = w as f32 / 2.0;
    let cy_coord = h as f32 / 2.0;
    let n_h_edges = (w - 1) * h;
    let n_edges = cx.n_edges();
    let mut data = vec![0.0f32; n_edges];
    for y in 0..h {
        for x in 0..(w - 1) {
            let e_idx = y * (w - 1) + x;
            data[e_idx] = -(y as f32 - cy_coord);
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            let e_idx = n_h_edges + y * w + x;
            data[e_idx] = x as f32 - cx_coord;
        }
    }
    let field = CochainField::from_vec(1, 1, data);

    // Region: a 64×64 block of faces in the center of the grid.
    // Faces are indexed as f = y_face * (w-1) + x_face, where
    // y_face, x_face ∈ [0, w-2].
    let faces_per_row = w - 1;
    let zone_w = 64;
    let zone_h = 64;
    let x0 = (faces_per_row - zone_w) / 2;
    let y0 = (faces_per_row - zone_h) / 2;
    let mut region: Vec<u32> = Vec::with_capacity(zone_w * zone_h);
    for dy in 0..zone_h {
        for dx in 0..zone_w {
            region.push(((y0 + dy) * faces_per_row + (x0 + dx)) as u32);
        }
    }

    (cx, field, region)
}

fn bench_boundary_flux_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/boundary_flux_mass");
    group.sample_size(50); // each iter is ~ms; 50 samples is enough.

    let (cx, field, region) = build_256x256_grid_and_field();

    // Pre-allocated scratch buffer for the naive baseline's exterior_derivative.
    let mut d_field_scratch = CochainField::zeros(2, cx.n_faces(), 1);

    // ── A: boundary_flux_mass_only ──────────────────────────────────────────
    group.bench_function("G-B_256x256_boundary_flux", |b| {
        b.iter(|| {
            let mass = boundary_flux_mass_only(
                black_box(&cx),
                black_box(&region),
                black_box(&field),
            );
            black_box(mass);
        });
    });

    // ── B: naive full-volume (exterior_derivative + region sum) ─────────────
    group.bench_function("G-B_256x256_naive_volume", |b| {
        b.iter(|| {
            // Compute d₁(field) over ALL faces (this is the "full volume" cost).
            exterior_derivative_into(
                black_box(&cx),
                black_box(&field),
                black_box(&mut d_field_scratch),
            );
            // Sum the region subset.
            let mut vol = 0.0f32;
            for &f in &region {
                vol += d_field_scratch.scalar(f as usize);
            }
            black_box(vol);
        });
    });

    // ── C: naive with cached d_field (amortized — many queries, one d_field) ─
    // Pre-compute d_field once (simulates per-tick caching).
    exterior_derivative_into(&cx, &field, &mut d_field_scratch);
    group.bench_function("G-B_256x256_cached_d_field_region_sum", |b| {
        b.iter(|| {
            let mut vol = 0.0f32;
            for &f in black_box(&region) {
                vol += d_field_scratch.scalar(f as usize);
            }
            black_box(vol);
        });
    });

    group.finish();
}

// ─── G-B indexed: cold vs warm coboundary index (Plan 318) ──────────────────
//
// Compares three variants on the same 256×256 grid + 64×64 region as the
// baseline `bench_boundary_flux_vs_naive`:
//
//   1. `boundary_flux_mass_only`       — the current 5.36× winner (full-scan).
//   2. `boundary_flux_mass_indexed_cold` — build_coboundary_index + 1 query
//      per iteration. Expected to be SLOWER than #1 because the build cost
//      dominates a single query. This is the honest "you must amortize" signal.
//   3. `boundary_flux_mass_indexed_warm` — pre-built index, query only.
//      Target: ≥3× faster than #1 (the Plan 318 GOAT gate).

fn bench_boundary_flux_indexed(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/boundary_flux_indexed");
    group.sample_size(50);

    let (cx, field, region) = build_256x256_grid_and_field();

    // Baseline (full-scan) for direct A/B in the same group.
    group.bench_function("G-B_256x256_full_scan_baseline", |b| {
        b.iter(|| {
            let mass = boundary_flux_mass_only(
                black_box(&cx),
                black_box(&region),
                black_box(&field),
            );
            black_box(mass);
        });
    });

    // Cold: build_coboundary_index + 1 query per iteration.
    // This is the worst case for the indexed path — build cost is not amortized.
    group.bench_function("G-B_256x256_indexed_cold", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut cx = cx.clone();
                cx.build_coboundary_index(1);
                let mass = boundary_flux_mass_indexed(
                    black_box(&cx),
                    black_box(&region),
                    black_box(&field),
                );
                black_box(mass);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Warm: index pre-built once, query only per iteration.
    // This is the intended use case (multi-query on stable topology).
    let mut cx_warm = cx.clone();
    cx_warm.build_coboundary_index(1);
    group.bench_function("G-B_256x256_indexed_warm", |b| {
        b.iter(|| {
            let mass = boundary_flux_mass_indexed(
                black_box(&cx_warm),
                black_box(&region),
                black_box(&field),
            );
            black_box(mass);
        });
    });

    group.finish();
}

// ─── G-C: line_integral as path cost ────────────────────────────────────────
//
// Two parts:
//   1. Latency: `line_integral` over a path on a 32×32 grid.
//   2. Correctness/discrimination: compare line_integral of a "smooth" path
//      vs a "zigzag" path of equal length under a directional-preference field.
//      Validates that line_integral-minimizing selection picks the smooth path
//      (fewer turns) — the mechanism behind the "≥20% fewer reversals" target.

fn count_turns_2d_grid(path: &[u32], w: usize) -> usize {
    // A "turn" is a change in direction between consecutive steps.
    // Direction encoded as (dx, dy) ∈ {-1, 0, +1}².
    if path.len() < 3 {
        return 0;
    }
    let mut turns = 0;
    let mut prev_dir: (i32, i32) = (0, 0);
    for w_pair in path.windows(2) {
        let a = w_pair[0] as usize;
        let b = w_pair[1] as usize;
        let ax = (a % w) as i32;
        let ay = (a / w) as i32;
        let bx = (b % w) as i32;
        let by = (b / w) as i32;
        let dir = (bx - ax, by - ay);
        if prev_dir != (0, 0) && dir != prev_dir {
            turns += 1;
        }
        prev_dir = dir;
    }
    turns
}

fn bench_line_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/line_integral");

    let w = 32;
    let h = 32;
    let cx = CellComplex::grid_2d(w, h);
    let n_edges = cx.n_edges();

    // Non-exact (rotational) edge field: cannot be written as the gradient of
    // any scalar potential, so line_integral IS path-dependent (fundamental
    // theorem of calculus does NOT apply). This is the regime where line_integral
    // as a path-cost function can discriminate between routes.
    //
    // Construction: horizontal edge (y,x) gets sin(x + y*0.3); vertical edge
    // (y,x) gets cos(x*0.3 + y). The curl of this field is non-zero, so
    // line_integral(A→B) depends on the path taken.
    //
    // NOTE: a rank-1 edge cochain encodes per-EDGE cost only — it cannot encode
    // TURN penalties by construction (turns are a path-level property requiring
    // a rank-2 face cochain or a higher-order operator). line_integral
    // discriminates paths by which EDGES they traverse (via curl), not by their
    // turn count. This bench validates line_integral as a path-cost function,
    // not as a smoothness regularizer.
    let mut data = vec![0.0f32; n_edges];
    let n_h_edges = (w - 1) * h;
    for y in 0..h {
        for x in 0..(w - 1) {
            let e_idx = y * (w - 1) + x;
            data[e_idx] = (x as f32 + y as f32 * 0.3).sin();
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            let e_idx = n_h_edges + y * w + x;
            data[e_idx] = (x as f32 * 0.3 + y as f32).cos();
        }
    }
    let field = CochainField::from_vec(1, 1, data);

    // ── Path A: smooth (all-East then all-North): (0,0) → (15,0) → (15,15) ──
    // 15 East steps + 15 North steps = 30 edges, 1 turn.
    let mut smooth_path: Vec<u32> = Vec::with_capacity(31);
    for x in 0..=15 {
        smooth_path.push(x as u32); // along row 0: 0*w + x
    }
    for y in 1..=15 {
        smooth_path.push((y * w + 15) as u32); // along column 15
    }

    // ── Path B: zigzag (alternating East, North): (0,0) → ... → (15,15) ─────
    // 30 edges, 29 turns.
    let mut zigzag_path: Vec<u32> = Vec::with_capacity(31);
    zigzag_path.push(0);
    let mut x = 0i32;
    let mut y = 0i32;
    for _ in 0..15 {
        x += 1;
        zigzag_path.push((y * w as i32 + x) as u32); // East
        y += 1;
        zigzag_path.push((y * w as i32 + x) as u32); // North
    }

    // Correctness check: smooth path should have a different line_integral
    // than the zigzag path (they traverse different edges with different
    // spatially-varying costs). If line_integral-weighted path selection were
    // to pick the cheaper of the two, it picks the lower value.
    let li_smooth = line_integral(&cx, &field, &smooth_path);
    let li_zigzag = line_integral(&cx, &field, &zigzag_path);
    let turns_smooth = count_turns_2d_grid(&smooth_path, w);
    let turns_zigzag = count_turns_2d_grid(&zigzag_path, w);
    eprintln!(
        "[G-C] smooth: line_integral={li_smooth:.3}, turns={turns_smooth}; \
         zigzag: line_integral={li_zigzag:.3}, turns={turns_zigzag}; \
         discriminates_by={:.3}",
        (li_zigzag - li_smooth).abs()
    );

    group.throughput(Throughput::Elements(smooth_path.len() as u64));
    group.sample_size(100);

    group.bench_function("G-C_smooth_path_30_edges", |b| {
        b.iter(|| {
            let li = line_integral(
                black_box(&cx),
                black_box(&field),
                black_box(&smooth_path),
            );
            black_box(li);
        });
    });

    group.bench_function("G-C_zigzag_path_30_edges", |b| {
        b.iter(|| {
            let li = line_integral(
                black_box(&cx),
                black_box(&field),
                black_box(&zigzag_path),
            );
            black_box(li);
        });
    });

    group.finish();
}

// ─── G-C2: circulation_integral on closed loops (Plan 317, Issue 005) ─────
//
// The rank-2 Stokes companion to `line_integral`. Where `line_integral` sums
// per-edge cost on an OPEN path (cannot see turn penalties — Issue 005's
// root cause for G-C structural fail), `circulation_integral` integrates curl
// over the area ENCLOSED by a CLOSED loop (Stokes: ∮F = ∬curl F).
//
// Goal of this bench: check whether `circulation_integral`-based selection
// can reduce turn count (the original G-C target). The pre-implementation
// analysis in Plan 317 predicts this is unlikely because enclosed area and
// turn count are INDEPENDENT geometric properties. The bench reports the
// honest empirical result.

fn bench_circulation_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/circulation_integral");

    let w = 32;
    let h = 32;
    let cx = CellComplex::grid_2d(w, h);
    let n_edges = cx.n_edges();

    // Constant-curl field: rigid rotation F = (−(y−cy), (x−cx))/2, with curl=1
    // everywhere in the continuum. Circulation around a closed loop = curl ×
    // (signed enclosed area). This is the field that makes `circulation_integral`
    // a pure area-measuring instrument.
    let cx_coord = w as f32 / 2.0;
    let cy_coord = h as f32 / 2.0;
    let n_h_edges = (w - 1) * h;
    let mut data = vec![0.0f32; n_edges];
    for y in 0..h {
        for x in 0..(w - 1) {
            let e_idx = y * (w - 1) + x;
            data[e_idx] = -(y as f32 - cy_coord);
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            let e_idx = n_h_edges + y * w + x;
            data[e_idx] = x as f32 - cx_coord;
        }
    }
    let field = CochainField::from_vec(1, 1, data);

    // ── Loop A: smooth rectangle (4 turns, encloses 8×8 = 64 area) ─────────
    // Trace the boundary of an 8×8 square: (4,4) → (12,4) → (12,12) → (4,12) → (4,4).
    // EVERY step is to an adjacent grid vertex (Manhattan distance 1) — required
    // for `line_integral`'s B₁ edge lookup to succeed.
    let x0 = 4;
    let y0 = 4;
    let x1 = 12;
    let y1 = 12;
    let mut smooth_loop: Vec<u32> = Vec::with_capacity(33);
    // bottom edge: (4,4) → (12,4)
    for x in x0..=x1 {
        smooth_loop.push((y0 * w + x) as u32);
    }
    // right edge: (12,5) → (12,12) (skip (12,4) already added)
    for y in (y0 + 1)..=y1 {
        smooth_loop.push((y * w + x1) as u32);
    }
    // top edge: (11,12) → (4,12) (skip (12,12) already added)
    for x in (x0..x1).rev() {
        smooth_loop.push((y1 * w + x) as u32);
    }
    // left edge: (4,11) → (4,4) (skip (4,12) and (4,4) — (4,4) is the closer)
    for y in ((y0 + 1)..y1).rev() {
        smooth_loop.push((y * w + x0) as u32);
    }
    smooth_loop.push((y0 * w + x0) as u32); // close → (4,4)
    let turns_smooth = count_turns_2d_grid(&smooth_loop, w);

    // ── Loop B: zigzag closed loop with the SAME bounding box [4,12]×[4,12] ──
    // but a zigzag bottom edge (many turns) that still closes the loop. The
    // enclosed area equals the polygon area by the shoelace formula — a zigzag
    // boundary can enclose MORE or LESS area than the smooth rectangle.
    // This loop traces a "sawtooth" bottom and a smooth top/right/left.
    //   bottom sawtooth: (4,4) → (5,5) → (6,4) → (7,5) → ... → (12,4) ... wait,
    //   each step MUST be Manhattan-adjacent. So: (4,4) → (5,4) → (5,5) → (6,5)
    //   → (6,4) → (7,4) → ... gives a staircase that stays within rows 4-5.
    // Simpler valid sawtooth: alternate (x,4) → (x,5) → (x+1,5) → (x+1,4).
    let mut zigzag_loop: Vec<u32> = Vec::with_capacity(64);
    zigzag_loop.push((y0 * w + x0) as u32); // (4,4)
    for x in x0..x1 {
        // zigzag tooth: (x,y0) → (x,y0+1) → (x+1,y0+1) → (x+1,y0)
        zigzag_loop.push(((y0 + 1) * w + x) as u32); // up to (x, y0+1)
        zigzag_loop.push(((y0 + 1) * w + (x + 1)) as u32); // right to (x+1, y0+1)
        if x + 1 < x1 {
            zigzag_loop.push((y0 * w + (x + 1)) as u32); // down to (x+1, y0)
        }
    }
    // Now at (12,5). Need to get to (12,12) along the right edge.
    for y in (y0 + 2)..=y1 {
        zigzag_loop.push((y * w + x1) as u32);
    }
    // top edge: (11,12) → (4,12)
    for x in (x0..x1).rev() {
        zigzag_loop.push((y1 * w + x) as u32);
    }
    // left edge: (4,11) → (4,4)
    for y in (y0..y1).rev() {
        zigzag_loop.push((y * w + x0) as u32);
    }
    // zigzag_loop now ends at (4,4) — loop is closed.
    let turns_zigzag = count_turns_2d_grid(&zigzag_loop, w);

    let circ_smooth = circulation_integral(&cx, &field, &smooth_loop);
    let circ_zigzag = circulation_integral(&cx, &field, &zigzag_loop);
    eprintln!(
        "[G-C2] smooth: circulation={circ_smooth:.3}, turns={turns_smooth}; \
         zigzag: circulation={circ_zigzag:.3}, turns={turns_zigzag}; \
         |smooth| vs |zigzag|: {:.3} vs {:.3}",
        circ_smooth.abs(),
        circ_zigzag.abs()
    );

    group.throughput(Throughput::Elements(smooth_loop.len() as u64));
    group.sample_size(100);

    group.bench_function("G-C2_smooth_closed_loop_8x8", |b| {
        b.iter(|| {
            let circ = circulation_integral(
                black_box(&cx),
                black_box(&field),
                black_box(&smooth_loop),
            );
            black_box(circ);
        });
    });

    group.bench_function("G-C2_zigzag_closed_loop", |b| {
        b.iter(|| {
            let circ = circulation_integral(
                black_box(&cx),
                black_box(&field),
                black_box(&zigzag_loop),
            );
            black_box(circ);
        });
    });

    group.finish();
}

// ─── Auxiliary: codifferential_into baseline (for belief_mass_divergence) ────

fn bench_codifferential_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/codifferential_baseline");

    let cx = CellComplex::grid_2d(32, 32);
    let n_edges = cx.n_edges();
    let field = CochainField::from_vec(1, 1, vec![1.0f32; n_edges]);
    let mut scratch = CochainField::zeros(0, cx.n_vertices(), 1);

    group.throughput(Throughput::Elements(n_edges as u64));
    group.bench_function("32x32_codifferential_into", |b| {
        b.iter(|| {
            codifferential_into(
                black_box(&cx),
                black_box(&field),
                black_box(&mut scratch),
            );
            // L1 sum of the divergence.
            let l1: f32 = scratch.data.iter().copied().map(f32::abs).sum();
            black_box(l1);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_belief_mass_divergence,
    bench_boundary_flux_vs_naive,
    bench_boundary_flux_indexed,
    bench_line_integral,
    bench_circulation_integral,
    bench_codifferential_baseline,
);
criterion_main!(benches);
