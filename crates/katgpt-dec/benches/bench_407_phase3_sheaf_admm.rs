//! Plan 407 Phase 3 — Sheaf-ADMM amplification perf gates.
//!
//! Three amplification gates for the Phase 3 tasks:
//!
//! - **T3.2 — Sparse selector latency (K=1000).** One `sheaf_admm_step_into`
//!   call on a 1000-vertex graph, comparing dense selector maps (general
//!   explicit-maps path, `O(d_e·d_v)` per edge) vs compact selector maps
//!   (gather-scatter fast path, `O(d_e)` per edge). Target: compact < 50% of
//!   dense latency.
//! - **T3.1 — CG vs GD z-update (K=1000, κ>100).** One step comparing GD
//!   z-update (20 diffusion steps) vs CG z-update (20 iterations). Target: CG
//!   reaches lower residual at equal matvec count.
//! - **T3.3 — Soft-constraint latency overhead.** One step comparing hard
//!   constraint vs soft constraint (gamma=0.5). Target: overhead < 20%.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan407_phase3 \
//! cargo bench -p katgpt-dec --features sheaf_admm --no-default-features \
//!   --bench bench_407_phase3_sheaf_admm -- --nocapture
//! ```

#![cfg(feature = "sheaf_admm")]

use katgpt_dec::{
    AdmmScratch, CellComplex, CochainField, LocalObjective, SheafMaps, sheaf_admm_step_cg_into,
    sheaf_admm_step_into, sheaf_admm_step_soft_into,
};
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SplitMix64 PRNG (deterministic, no external dep)
// ---------------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let bits = z >> 40;
        let u01 = (bits as f32) / ((1u64 << 24) as f32);
        u01 * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Workload builders
// ---------------------------------------------------------------------------

/// Build a path-graph workload with K vertices, d_v=8, d_e=5.
/// Returns (cx, primal, consensus, dual, objective, scratch).
fn build_path_graph_workload(
    k: usize,
) -> (
    CellComplex,
    CochainField,
    CochainField,
    CochainField,
    LocalObjective,
    AdmmScratch,
) {
    let edges: Vec<(usize, usize)> = (0..k - 1).map(|i| (i, i + 1)).collect();
    let cx = CellComplex::from_edges(k, &edges);
    let d_v = 8;
    let d_e = 5;
    let total = k * d_v;
    let mut rng = SplitMix64::new(0xBADC_407A_6026_0707);
    let mut primal = CochainField::zeros(0, k, d_v);
    let mut consensus = CochainField::zeros(0, k, d_v);
    let mut dual = CochainField::zeros(0, k, d_v);
    for i in 0..total {
        primal.data[i] = rng.next_f32();
        consensus.data[i] = rng.next_f32();
        dual.data[i] = rng.next_f32() * 0.1;
    }
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![-0.5; total],
    };
    let scratch = AdmmScratch::new(&cx, d_v, d_e);
    (cx, primal, consensus, dual, objective, scratch)
}

/// Build dense selector maps (non-identity dims [3,4,5,6,7]) and compact
/// selector maps with the same dims, for a given cell complex. Both produce
/// the SAME selector maps mathematically; the dense path goes through the
/// general explicit-maps matvec (`O(d_e·d_v)` per edge), the compact path
/// through the gather-scatter fast path (`O(d_e)` per edge).
fn build_selector_maps_pair(cx: &CellComplex, d_v: usize, d_e: usize) -> (SheafMaps, SheafMaps) {
    // Non-identity dims to force the general explicit-maps path on the dense
    // side (identity dims [0..d_e] would take the identity grid-stencil path).
    let dims_uniform: Vec<usize> = (d_v - d_e..d_v).collect(); // [3,4,5,6,7] for d_v=8,d_e=5
    let dense = SheafMaps::selector(cx, d_v, &dims_uniform);
    assert!(
        !dense.is_identity,
        "dense selector should NOT be identity (non-standard dims)"
    );
    let dims_per_edge: Vec<&[usize]> = vec![&dims_uniform; cx.n_edges()];
    let compact = SheafMaps::selector_per_edge(cx, d_v, &dims_per_edge);
    assert!(
        !compact.is_identity,
        "compact selector should NOT be identity"
    );
    (dense, compact)
}

/// Measure mean latency over N iterations of a closure.
fn measure_latency<F: FnMut()>(label: &str, n: usize, mut f: F) -> f64 {
    // Warmup.
    for _ in 0..(n.min(10)) {
        f();
    }
    let start = Instant::now();
    for _ in 0..n {
        f();
    }
    let elapsed = start.elapsed();
    let mean_ns = elapsed.as_nanos() as f64 / n as f64;
    println!("  {label}: mean = {mean_ns:.1} ns  ({n} iters)");
    mean_ns
}

// ===========================================================================
// T3.2 — Sparse selector latency gate (K=1000)
// ===========================================================================

fn t32_sparse_selector_latency() -> (f64, f64, bool) {
    println!("\n── T3.2: Sparse selector latency (K=1000, d_v=8, d_e=5) ──");
    let k = 1000;
    let d_v = 8;
    let d_e = 5;
    let n_iters = 500;

    // Dense selector workload.
    let (cx, mut px_d, mut cz_d, mut du_d, obj_d, mut sc_d) = build_path_graph_workload(k);
    let (dense_maps, compact_maps) = build_selector_maps_pair(&cx, d_v, d_e);

    // Compact selector workload (clone state so both start identically).
    let px_c = px_d.clone();
    let cz_c = cz_d.clone();
    let du_c = du_d.clone();
    let obj_c = obj_d.clone();
    let sc_c = sc_d.clone();

    let dense_ns = measure_latency("dense selector", n_iters, || {
        sheaf_admm_step_into(
            black_box(&cx),
            black_box(&dense_maps),
            black_box(&mut px_d),
            black_box(&mut cz_d),
            black_box(&mut du_d),
            black_box(&obj_d),
            1.0,
            0.2,
            5,
            black_box(&mut sc_d),
        );
    });

    let mut px_c = px_c;
    let mut cz_c = cz_c;
    let mut du_c = du_c;
    let mut sc_c = sc_c;
    let compact_ns = measure_latency("compact selector", n_iters, || {
        sheaf_admm_step_into(
            black_box(&cx),
            black_box(&compact_maps),
            black_box(&mut px_c),
            black_box(&mut cz_c),
            black_box(&mut du_c),
            black_box(&obj_c),
            1.0,
            0.2,
            5,
            black_box(&mut sc_c),
        );
    });

    let ratio = compact_ns / dense_ns;
    let pass = compact_ns < 0.5 * dense_ns;
    println!(
        "  ratio: compact/dense = {ratio:.3}  (gate < 0.50)  → {}",
        if pass { "PASS ✅" } else { "FAIL ❌" }
    );
    (dense_ns, compact_ns, pass)
}

// ===========================================================================
// T3.1 — CG vs GD z-update (K=1000, κ>100)
// ===========================================================================

fn t31_cg_vs_gd_residual() -> (f64, f64, bool) {
    println!("\n── T3.1: CG vs GD z-update residual (K=1000, κ>100) ──");
    let k = 1000;
    let d_v = 8;
    let d_e = 5;

    // GD path.
    let (cx, mut px_gd, mut cz_gd, mut du_gd, obj, mut sc_gd) = build_path_graph_workload(k);

    // CG path (clone state).
    let px_cg = px_gd.clone();
    let cz_cg = cz_gd.clone();
    let du_cg = du_gd.clone();
    let obj_cg = obj.clone();
    let sc_cg = sc_gd.clone();
    let maps = SheafMaps::identity(&cx, d_v, d_e);

    // One step: GD with T=20 diffusion, CG with 20 iters.
    sheaf_admm_step_into(
        &cx, &maps, &mut px_gd, &mut cz_gd, &mut du_gd, &obj, 1.0, 0.2, 20, &mut sc_gd,
    );
    let mut px_cg = px_cg;
    let mut cz_cg = cz_cg;
    let mut du_cg = du_cg;
    let mut sc_cg = sc_cg;
    sheaf_admm_step_cg_into(
        &cx, &maps, &mut px_cg, &mut cz_cg, &mut du_cg, &obj_cg, 1.0, 20, 1e-12, &mut sc_cg,
    );

    // Measure residual ‖L_F z‖ for both.
    let sc_r = AdmmScratch::new(&cx, d_v, d_e);
    // We approximate the residual by computing the max-edge disagreement of z.
    let gd_res = max_edge_disagreement_l1(&cx, &cz_gd);
    let cg_res = max_edge_disagreement_l1(&cx, &cz_cg);
    let _ = sc_r; // (sc_r unused — residual computed from disagreement)

    let pass = cg_res < gd_res;
    println!(
        "  gd_disagreement={gd_res:.6}, cg_disagreement={cg_res:.6}  (gate: CG < GD)  → {}",
        if pass { "PASS ✅" } else { "FAIL ❌" }
    );
    (gd_res as f64, cg_res as f64, pass)
}

/// Sum of |z_tail[d] - z_head[d]| over all edges and dims (L1 disagreement).
fn max_edge_disagreement_l1(cx: &CellComplex, x: &CochainField) -> f32 {
    let d_v = x.dim;
    let mut sum = 0.0f32;
    for pair in cx.boundary_entries(0).as_chunks::<2>().0 {
        let v_tail = pair[0].0;
        let v_head = pair[1].0;
        for d in 0..d_v {
            sum += (x.data[v_tail * d_v + d] - x.data[v_head * d_v + d]).abs();
        }
    }
    sum
}

// ===========================================================================
// T3.3 — Soft-constraint latency overhead
// ===========================================================================

fn t33_soft_constraint_overhead() -> (f64, f64, bool) {
    println!("\n── T3.3: Soft-constraint latency overhead (gamma=0.5) ──");
    let k = 1000;
    let d_v = 8;
    let d_e = 5;
    let n_iters = 500;

    let (cx, mut px_h, mut cz_h, mut du_h, obj, mut sc_h) = build_path_graph_workload(k);
    let maps = SheafMaps::identity(&cx, d_v, d_e);

    let px_s = px_h.clone();
    let cz_s = cz_h.clone();
    let du_s = du_h.clone();
    let obj_s = obj.clone();
    let sc_s = sc_h.clone();

    let hard_ns = measure_latency("hard constraint", n_iters, || {
        sheaf_admm_step_into(
            black_box(&cx),
            black_box(&maps),
            black_box(&mut px_h),
            black_box(&mut cz_h),
            black_box(&mut du_h),
            black_box(&obj),
            1.0,
            0.2,
            5,
            black_box(&mut sc_h),
        );
    });

    let mut px_s = px_s;
    let mut cz_s = cz_s;
    let mut du_s = du_s;
    let mut sc_s = sc_s;
    let soft_ns = measure_latency("soft constraint (γ=0.5)", n_iters, || {
        sheaf_admm_step_soft_into(
            black_box(&cx),
            black_box(&maps),
            black_box(&mut px_s),
            black_box(&mut cz_s),
            black_box(&mut du_s),
            black_box(&obj_s),
            1.0,
            0.2,
            0.5,
            5,
            black_box(&mut sc_s),
        );
    });

    let overhead = (soft_ns - hard_ns) / hard_ns;
    let pass = overhead < 0.20;
    println!(
        "  overhead: {overhead:+.1}x  (gate < 0.20x)  → {}",
        if pass { "PASS ✅" } else { "FAIL ❌" }
    );
    (hard_ns, soft_ns, pass)
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 407 Phase 3 — Sheaf-ADMM Amplification Perf Gates (T3.1-3.3) ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");

    let (dense_ns, compact_ns, t32_pass) = t32_sparse_selector_latency();
    let (gd_res, cg_res, t31_pass) = t31_cg_vs_gd_residual();
    let (hard_ns, soft_ns, t33_pass) = t33_soft_constraint_overhead();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Phase 3 Summary:");
    println!(
        "  T3.2 sparse selector: dense={dense_ns:.0}ns compact={compact_ns:.0}ns → {}",
        if t32_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  T3.1 CG vs GD:        gd_res={gd_res:.4} cg_res={cg_res:.4} → {}",
        if t31_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  T3.3 soft overhead:   hard={hard_ns:.0}ns soft={soft_ns:.0}ns → {}",
        if t33_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    let all_pass = t32_pass && t31_pass && t33_pass;
    println!(
        "\n══ Phase 3 {} ══",
        if all_pass {
            "ALL GATES PASS ✅"
        } else {
            "SOME GATES FAILED ❌"
        }
    );
}
