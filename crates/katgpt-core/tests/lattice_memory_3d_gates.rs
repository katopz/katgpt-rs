//! Issue 938 — the 3-axis lattice gate surface (`lattice_memory_3d`).
//!
//! T2.1/T2.2 mechanics + the G6 memory pricing (priced BEFORE the build —
//! the build is unblocked at the d_k = d_v = 32 posture and this gate pins
//! the arithmetic), the 3^A product support, the sized3 capacity contrast
//! (2D refuses the 10⁶/0.05σ operating point 3D accepts — the second
//! remedy, measured as a constructor verdict), and freeze/thaw on the 3D
//! shape. The recall/latency GOAT (G1/G2/G5) lives in
//! `bench_938_lattice_3d_goat`; G4's alloc arm sits behind the Issue-741
//! profile rule exactly as in `lattice_memory_gates`.
//!
//! Run:
//!
//! ```sh
//! cargo test -p katgpt-core --features lattice_memory_3d --test lattice_memory_3d_gates
//! cargo test -p katgpt-core --features lattice_memory_3d,alloc_tracking \
//!     --release --test lattice_memory_3d_gates
//! ```
#![cfg(feature = "lattice_memory_3d")]

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[global_allocator]
static GATE_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;

use katgpt_core::lattice_memory::{
    LatticeConfig, LatticeConfig3D, LatticeConfigError, LatticeMemory, LatticeMemory3D,
    grid_side_for,
};

/// The G6 posture: d = d_k = 32, R = 0.05σ — the exact cell Bench 919's
/// floor could not hold in 2D.
const D: usize = 32;

fn config3(grid: [u32; 3]) -> LatticeConfig3D {
    LatticeConfig3D {
        d: D,
        d_k: D,
        d_v: D,
        grid,
        w: [1.0; 3],
        seed: 0x938,
        byte_budget: None,
        warp: None,
        bump_kernel: katgpt_core::lattice_memory::BumpKernel::Tent,
    }
}

/// Always runs: names the posture that measured nothing (the bench_818
/// convention).
#[test]
fn alloc_gate_posture_disclosure() {
    if cfg!(any(debug_assertions, feature = "alloc_tracking")) {
        println!("alloc gate: ARMED (TrackingAllocator installed)");
    } else {
        println!(
            "G4 alloc-free ⛔ NOT MEASURED — this profile compiles no allocator. \
             Re-run with --features alloc_tracking (release) or in dev."
        );
    }
}

#[test]
fn g6_memory_is_priced_before_the_build() {
    // T2.2's arithmetic, pinned: L = ⌈3.76/0.05⌉ = 76 per axis, cells =
    // 76³ = 438,976, slab = cells·d_k·d_v·4 B = 1,798,156,288 B — under the
    // 2 GiB bar with ~16% headroom. The build is UNBLOCKED; the d_v = 64
    // posture is the recorded boundary row (over the bar, needs a budget
    // conversation or a thinner value).
    let side = grid_side_for(0.05);
    assert_eq!(side, 76, "⌈3.76/0.05⌉");
    let cfg = config3([side, side, side]);
    assert_eq!(cfg.cell_count(), 438_976, "76³");
    assert_eq!(
        cfg.slab_bytes(),
        1_798_045_696,
        "cells · 32·32 · 4 B = cells · 4096"
    );
    const TWO_GIB: usize = 2 * 1024 * 1024 * 1024;
    assert!(
        cfg.slab_bytes() <= TWO_GIB,
        "G6: the 3D posture at d_v=32 costs {} B, over the {} B bar",
        cfg.slab_bytes(),
        TWO_GIB
    );
    // The capacity envelope, both arities (the issue's headline arithmetic):
    // N_max ≈ 0.35·d_k·(3.76/R)^AXES — 2D ≈ 63k (the recorded ceiling),
    // 3D ≈ 4.76M (the remedy). Recomputed here, never quoted from
    // μLattice §4.7 (whose own scale-96 cell count carries an erratum).
    let n_max = |axes: u32| 0.35 * D as f32 * (3.76 / 0.05_f32).powi(axes as i32);
    assert!(
        (n_max(2) - 63_336.0).abs() < 5.0,
        "2D envelope {}",
        n_max(2)
    );
    assert!(
        (n_max(3) - 4_762_900.0).abs() < 100.0,
        "3D envelope {}",
        n_max(3)
    );
    // d_v = 64 doubles the cell — over the bar: the recorded boundary.
    let mut wide = config3([side, side, side]);
    wide.d_v = 64;
    assert!(
        wide.slab_bytes() > TWO_GIB,
        "the d_v=64 boundary row moved: {} B",
        wide.slab_bytes()
    );
}

#[test]
fn sized3_accepts_the_point_2d_refuses() {
    // The remedy as a constructor verdict: 10⁶ items at R = 0.05σ —
    // 2D: cells = 76² = 5,776, λ = 173.1 ≫ 2.8... wait, d_k = 32 →
    // capacity 11.2 — refused (OverCapacity, the Bench 919 negative).
    // 3D: cells = 438,976, λ = 2.278 ≤ 11.2 — accepted.
    let mut rng = fastrand::Rng::with_seed(0x938_0002);
    let samples: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..D).map(|_| rng.f32() * 2.0 - 1.0).collect())
        .collect();
    let refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
    match LatticeMemory::sized(D, D, D, 0x938, &refs, 1_000_000, 0.05, None) {
        Err(LatticeConfigError::OverCapacity {
            cells,
            capacity_per_cell,
            ..
        }) => {
            // λ = 10⁶/5,776 = 173.1 against capacity 11.2 — the Bench 919
            // negative, as a constructor verdict.
            assert_eq!(cells, 76 * 76);
            assert!((capacity_per_cell - 11.2).abs() < 1e-6);
        }
        Err(other) => panic!("expected OverCapacity from the 2D sizer, got {other}"),
        Ok(_) => panic!("the 2D sizer accepted the 10⁶/0.05σ point — the negative moved"),
    }
    // The 3D accept — at d_k = d_v = 8 so the gate's slab stays a test-sized
    // 112 MB (the FULL 1.8 GB posture is the bench's cell, not a unit
    // test's; the pricing above pins its arithmetic).
    let small: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..D).map(|_| rng.f32() * 2.0 - 1.0).collect())
        .collect();
    let small_refs: Vec<&[f32]> = small.iter().map(|s| s.as_slice()).collect();
    let lat = LatticeMemory3D::sized(D, 8, 8, 0x938, &small_refs, 1_000_000, 0.05, None)
        .expect("the 3D sizer must accept the point 2D refuses");
    assert_eq!(lat.config().grid, [76, 76, 76]);
    assert_eq!(lat.config().cell_count(), 438_976);
    assert_eq!(lat.config().slab_bytes(), 438_976 * 8 * 8 * 4);
}

#[test]
fn support_is_the_product_kernel_and_stays_graded() {
    // The 3^A support mechanics: a ±1-cell lattice neighbor recovers through
    // kernel mass; two cells away on the walk reads nothing (the tent's hard
    // support); exact hits read the stronger 3D dilution (0.6³ = 0.216 vs
    // 2D's 0.36) — and read_direct stays the exact lane.
    let mut lat = LatticeMemory3D::new(config3([16, 16, 16])).unwrap();
    let key = [0.5_f32; D];
    let v = [1.0_f32; D];
    lat.write_value(&key, &v);
    // Exact hit: the blended read is DIRECTION-PRESERVING and diluted —
    // per-axis center weight ≤ 0.6, so the 27-tap product caps at 0.6³ ≈
    // 0.216 (an arbitrary query's δ ≠ 0 sits below the cap; the per-axis
    // (0.2, 0.6, 0.2) shape is pinned in bump.rs). read_direct is v
    // bit-for-bit.
    let mut out = [0.0_f32; D];
    lat.read_blend(&key, &mut out);
    let g = out[0];
    assert!(
        g > 0.0 && g <= 0.22,
        "3D blend factor {g} outside (0, 0.216]"
    );
    for (o, &s) in out.iter().zip(v.iter()) {
        assert!(
            (o - g * s).abs() < 1e-6,
            "3D blend broke direction: {o} vs {g}·{s}"
        );
    }
    let mut direct = [9.0_f32; D];
    assert!(lat.read_direct(&key, &mut direct));
    assert_eq!(direct, v, "the unblended lane is exact at any arity");

    // Walk the query along input dim 0 (which feeds ALL THREE LSH axes) and
    // classify each stop by its MAX-axis cell distance from the base: every
    // axis within ±1 → inside the 3×3×3 support, must recover mass; any
    // axis at |Δ| ≥ 2 → outside, must read exactly zero. (L1 distance is
    // the WRONG classifier here — a cell at (1,1,0) is L1-2 but inside the
    // product support.)
    let base = lat.primary_cell(&key);
    let mut saw_neighbor = false;
    let mut t = 0.0_f32;
    while t < 8.0 {
        t += 0.01;
        let mut q = key;
        q[0] += t;
        let cell = lat.primary_cell(&q);
        let dist_max = cell
            .iter()
            .zip(base.iter())
            .map(|(a, &b)| a.abs_diff(b))
            .max()
            .expect("3-axis cell");
        lat.read_blend(&q, &mut out);
        let recall = out[0]; // v was all-ones → the blended written mass
        if dist_max == 1 {
            assert!(recall > 0.0, "a ±1-cell neighbor read a cliff");
            saw_neighbor = true;
        } else if dist_max >= 2 {
            assert_eq!(
                recall, 0.0,
                "the ±1-per-axis support leaked at max-axis distance {dist_max}"
            );
            break;
        }
    }
    assert!(saw_neighbor, "the sweep never crossed a cell boundary");
}

#[test]
fn three_d_near_miss_recovers_and_decay_is_graded() {
    // The class claim at A = 3: perturb a stored key by sub-cell steps on
    // every axis at once; recall decays with distance, never cliffs. The
    // step keeps the walk inside the ±1 support until the last stops (an
    // input move of δ moves each LSH axis by ~δ·√D cells on average).
    let mut lat = LatticeMemory3D::new(config3([32, 32, 32])).unwrap();
    let key = [0.3_f32; D];
    lat.write_value(&key, &[1.0_f32; D]);
    let mut out = [0.0_f32; D];
    let mut recalls = Vec::new();
    for step in 1..=8_u32 {
        let mut q = key;
        for slot in q.iter_mut() {
            *slot += step as f32 * 0.02;
        }
        lat.read_blend(&q, &mut out);
        recalls.push(out[0]);
    }
    assert!(
        recalls.iter().all(|&r| r > 0.0),
        "3D near-miss read a cliff: {recalls:?}"
    );
    assert!(
        recalls[0] > recalls[recalls.len() - 1],
        "3D recall must decay with distance: {recalls:?}"
    );
}

#[test]
fn snapshot_roundtrip_restores_exact_state_at_3d() {
    // Freeze/thaw is arity-blind (the Plan 199 surface): the thawed lattice's
    // reads — blended AND direct — are bitwise the source's.
    let mut lat = LatticeMemory3D::new(config3([16, 16, 16])).unwrap();
    let xa = [0.35_f32; D];
    let xb = {
        let mut b = [0.1_f32; D];
        b[7] = 0.9;
        b
    };
    let k = [0.5_f32; D];
    let v_unit = {
        let mut v = [0.0_f32; D];
        v[1] = 1.0;
        v
    };
    lat.write_value(&xa, &[2.0_f32; D]);
    lat.write_delta(&xb, &k, &v_unit, 1.0);
    let mut blend_before = [0.0_f32; D];
    lat.read_blend(&xa, &mut blend_before);
    let snap = lat.snapshot();
    lat.write_value(&xa, &[9.0_f32; D]); // mutate past the snapshot
    lat.restore(&snap).expect("restore");
    let mut blend_after = [0.0_f32; D];
    lat.read_blend(&xa, &mut blend_after);
    assert_eq!(blend_before, blend_after, "thaw must be bitwise");
    let mut direct_after = [0.0_f32; D];
    assert!(lat.read_direct(&xa, &mut direct_after));
    assert_eq!(
        direct_after, [2.0_f32; D],
        "the unblended lane thaws exactly"
    );
}

#[test]
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
fn hot_path_is_alloc_free_at_3d() {
    // G4 at the wider arity: after construction, write/read/score allocate
    // nothing (27-tap reads walk the same slab — no per-tap state).
    let mut lat = LatticeMemory3D::new(config3([8, 8, 8])).unwrap();
    let x = [0.35_f32; D];
    let k = [0.5_f32; D];
    let v = [1.0_f32; D];
    let mut out = [0.0_f32; D];
    lat.write_delta(&x, &k, &v, 1.0);
    lat.read_cells(&x, &k, &mut out);
    lat.read_blend(&x, &mut out);
    let _ = lat.read_scored(&x, &k, &mut out, 4.0);
    lat.read_direct(&x, &mut out);
    katgpt_core::alloc::reset_alloc_stats();
    for i in 0..1000_u32 {
        let mut w = v;
        w[0] = (i & 0xFF) as f32;
        lat.write_delta(&x, &k, &w, 0.5);
        lat.write_value(&x, &w);
        lat.read_cells(&x, &k, &mut out);
        lat.read_blend(&x, &mut out);
        lat.read_direct(&x, &mut out);
        let score = lat.read_scored(&x, &k, &mut out, 4.0);
        std::hint::black_box(score);
        std::hint::black_box(&out);
    }
    let (count, bytes) = katgpt_core::alloc::get_alloc_stats();
    assert_eq!(
        count, 0,
        "3D hot path allocated {count} times ({bytes} B) — the G4 law is 0 after construction"
    );
}

/// The 2D alias is untouched by the 3D work — one sanity read across the
/// sibling arity (the G3 surface stays load-bearing).
#[test]
fn two_d_alias_still_reads() {
    let cfg = LatticeConfig {
        d: 16,
        d_k: 4,
        d_v: 4,
        grid: [16, 16],
        w: [2.0, 2.0],
        seed: 42,
        byte_budget: None,
        warp: None,
        bump_kernel: katgpt_core::lattice_memory::BumpKernel::Tent,
    };
    let mut lat = LatticeMemory::new(cfg).unwrap();
    lat.write_value(&[0.3_f32; 16], &[1.0, 2.0, 3.0, 4.0]);
    let mut direct = [0.0_f32; 4];
    assert!(lat.read_direct(&[0.3_f32; 16], &mut direct));
    assert_eq!(direct, [1.0, 2.0, 3.0, 4.0]);
}
