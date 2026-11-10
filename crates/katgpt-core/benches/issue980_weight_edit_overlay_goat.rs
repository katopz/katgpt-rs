//! Issue 980 T4 — the WeightEditOverlay G2/G4 GOAT bench.
//!
//! G2: apply-at-load compose cost vs full-tensor bake (interleaved A/B
//! pairs, Issue-723 treatment, medians across pairs). Two shapes:
//! the fixture's 64×64 with |S|=1, and an LLM-layer-scale 4096×1024 with
//! |S|=64 (the shape where compose-vs-bake actually pays).
//!
//! G4: the counting-allocator canary over the apply path (compose +
//! forward-delta + bypass-forward + revert — zero steady-state allocs).
//!
//! harness = false + std::time::Instant (the DEC GOAT convention).
//! Box state is printed beside the numbers and MUST be quoted with them
//! (Issue-021 law: a latency number without its box state is not a
//! measurement). Run with: cargo bench -p katgpt-core --features
//! weight_edit_overlay --bench issue980_weight_edit_overlay_goat

#![cfg(feature = "gain_cost_halt")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::weight_edit_overlay::{EditForm, WeightEditOverlay, WeightEditSpec};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

const LAYER: u8 = 1;

fn make_overlay(columns: &[u32], out_dim: usize, strength: f32) -> WeightEditOverlay {
    let mut dir = vec![0.0f32; out_dim];
    dir[0] = 1.0;
    WeightEditOverlay::new(vec![WeightEditSpec {
        layer: LAYER,
        form: EditForm::DirectionSubtraction {
            columns: columns.to_vec(),
            direction: dir,
            strength,
        },
    }])
    .expect("bench overlay valid")
}

/// Interleaved A/B measurement (Issue-723): chunk pairs (a, b), median
/// across pairs. Returns (median_a_ns, median_b_ns) per single op.
fn interleaved_ab(
    pairs: usize,
    mut chunk_a: impl FnMut(),
    mut chunk_b: impl FnMut(),
    ops_a: u64,
    ops_b: u64,
) -> (f64, f64) {
    let mut a_ns = Vec::with_capacity(pairs);
    let mut b_ns = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let t0 = Instant::now();
        chunk_a();
        a_ns.push(t0.elapsed().as_nanos() as f64);
        let t1 = Instant::now();
        chunk_b();
        b_ns.push(t1.elapsed().as_nanos() as f64);
    }
    a_ns.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_ns.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med_a = a_ns[a_ns.len() / 2] / ops_a as f64;
    let med_b = b_ns[b_ns.len() / 2] / ops_b as f64;
    (med_a, med_b)
}

fn bench_shape(out_dim: usize, in_dim: usize, n_cols: usize, label: &str) {
    let weights: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (i as f64 * 0.17).sin() as f32)
        .collect();
    let columns: Vec<u32> = (0..n_cols).map(|i| ((i * 7 + 3) % in_dim) as u32).collect();
    let overlay = make_overlay(&columns, out_dim, 0.5);

    // Baseline (flag-off): a whole-tensor copy = the "full bake" proxy —
    // what a re-materialized (rebaked) tensor costs per load.
    let warm_weights = weights.clone();

    // Warmup + reach proof.
    let mut w = warm_weights.clone();
    let mut backup = vec![0.0f32; overlay.touched_count(LAYER, out_dim, in_dim)];
    overlay
        .compose_layer_into(LAYER, &mut w, out_dim, in_dim, &mut backup)
        .expect("backup sized");
    overlay.revert_layer(LAYER, &mut w, out_dim, in_dim, &backup);
    assert_eq!(w, warm_weights, "bench sanity: revert restores");
    let n_cols_touched = backup.len() / out_dim;
    assert_eq!(n_cols_touched, n_cols, "bench sanity: all columns touched");

    const PAIRS: usize = 31;

    // Both arms measure ONLY the transform on a PRE-OWNED buffer (the
    // load-time posture: the process already owns the weight tensor). The
    // compose arm = overlay compose + revert (the apply-at-load cycle on
    // the live buffer); the bake arm = full-tensor rewrite (what a
    // re-materialized variant costs per load). Each arm runs the loop 8×
    // per chunk to rise above timer noise at small shapes.
    let mut w_a = warm_weights.clone();
    let mut w_b = warm_weights.clone();
    let mut backup = vec![0.0f32; out_dim * n_cols];
    const INNER: u64 = 8;
    let (compose_ns, bake_ns) = interleaved_ab(
        PAIRS,
        || {
            for _ in 0..INNER {
                overlay
                    .compose_layer_into(LAYER, &mut w_a, out_dim, in_dim, &mut backup)
                    .unwrap();
                overlay.revert_layer(LAYER, &mut w_a, out_dim, in_dim, &backup);
            }
            black_box(&w_a);
        },
        || {
            for _ in 0..INNER {
                for (src, dst) in warm_weights.iter().zip(w_b.iter_mut()) {
                    *dst = *src;
                }
            }
            black_box(&w_b);
        },
        INNER,
        INNER,
    );

    println!(
        "| {label} | {out_dim}×{in_dim} | {n_cols} | compose {compose_ns:.1} ns | bake {bake_ns:.1} ns | {:.2}× |",
        bake_ns / compose_ns
    );

    // The forward-delta posture (the designed serving path for large/packed
    // bases): rank-1 masked contribution, O(|S| + out), ZERO matrix traffic
    // — where the compose form's strided-column penalty does not apply.
    let h_fd = vec![0.5f32; in_dim];
    let mut y_fd = vec![0.0f32; out_dim];
    overlay.forward_delta_into(LAYER, &h_fd, &mut y_fd); // warmup/reach
    const FD_INNER: u64 = 64;
    let mut fd_ns = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        let t = Instant::now();
        for _ in 0..FD_INNER {
            overlay.forward_delta_into(LAYER, &h_fd, &mut y_fd);
        }
        fd_ns.push(t.elapsed().as_nanos() as f64);
    }
    fd_ns.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let fd_med = fd_ns[fd_ns.len() / 2] / FD_INNER as f64;
    println!("| {label} | forward-delta (serving posture) | {n_cols} | {fd_med:.1} ns | — | — |");

    // G4: alloc-free compose + revert after warmup (the counters are
    // thread-local; this serial section is the whole measured window).
    // Setup (clone/backup/output Vecs) is caller-side and sits OUTSIDE the
    // measured window — the window covers ONLY the four apply calls.
    assert_counter_is_live();
    let mut w = warm_weights.clone();
    let mut backup = vec![0.0f32; out_dim * n_cols];
    let h = vec![0.5f32; in_dim];
    let mut y = vec![0.0f32; out_dim];
    overlay
        .compose_layer_into(LAYER, &mut w, out_dim, in_dim, &mut backup)
        .unwrap();
    overlay.forward_delta_into(LAYER, &h, &mut y);
    overlay.bypass_forward_into(LAYER, &h, &mut y);
    overlay.revert_layer(LAYER, &mut w, out_dim, in_dim, &backup);
    let (_, allocs) = alloc_delta(|| {
        overlay
            .compose_layer_into(LAYER, &mut w, out_dim, in_dim, &mut backup)
            .unwrap();
        overlay.forward_delta_into(LAYER, &h, &mut y);
        overlay.bypass_forward_into(LAYER, &h, &mut y);
        overlay.revert_layer(LAYER, &mut w, out_dim, in_dim, &backup);
        black_box(&w);
    });
    println!(
        "| {label} | G4 apply-path allocs (compose+delta+bypass+revert) | — | {allocs} | — | — |"
    );
    assert_eq!(allocs, 0, "G4: apply path must be alloc-free");
}

fn main() {
    println!("== Issue 980 WeightEditOverlay GOAT bench (G2 compose-vs-bake + G4 alloc) ==");
    println!("box: {}", std::env::consts::OS);
    println!(
        "box_state: cpus={} | note: quote `uptime` + power state beside these numbers (Issue-021 law)",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    println!();
    println!("| shape | matrix | \\|S\\| | compose (ns/op) | full bake (ns/op) | bake/compose |");
    println!("|---|---|---|---|---|---|");

    // The fixture posture: 64×64, one carrier column.
    bench_shape(64, 64, 1, "fixture");
    // LLM-layer scale: 4096 out × 1024 in, |S| = 64 (6.25% of columns).
    bench_shape(4096, 1024, 64, "llm-layer");
    // Wide-in extreme: 4096×16384, |S| = 64 (0.4% of columns — the compose
    // win is largest where the untouched surface dominates).
    bench_shape(4096, 16384, 64, "wide-in");

    println!();
    println!(
        "G1 = the T3 battery (tests/issue980_weight_edit_overlay_gates.rs); this bench carries G2/G4."
    );
}
