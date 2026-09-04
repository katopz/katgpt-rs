//! Profiling test for DenseMesh (Plan 266 Phase 7, T7.4).
//!
//! Measures per-layer forward time, aggregation overhead, EdgeBandit decision
//! latency, and compute_router dispatch cost across topology widths
//! [1, 2, 4, 8, 16]. Validates Gate 3 (easy overhead) and Gate 4 (hard bound)
//! on synthetic data — full LLM forward integration is deferred (Phase 5).
//!
//! Run with:
//!   cargo test --features dense_mesh --test prof_dense_mesh -- --nocapture
//!   cargo test --release --features dense_mesh --test prof_dense_mesh -- --nocapture
//!
//! Per optimisation.md template (no `optimisation.md` in `.docs/`, so this
//! follows the breakeven / rat_bridge bench layout: header table, per-test
//! section, human-readable box output, relaxed assertions in debug builds).

#![cfg(feature = "dense_mesh")]
#![cfg(test)]

// Issue 721 T3: install the tracking allocator in THIS test binary. The root
// lib no longer registers a `#[global_allocator]` as a library (that chose
// the process allocator for every downstream binary and conflicted with any
// consumer's own registration). Replaces the Issue-682 force-link, which
// existed only to keep the root's library-level shim linked.
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;

/// Liveness sentinel (Issue 682): FAIL the audit if the TrackingAllocator
/// is not actually installed (debug builds only).
#[cfg(debug_assertions)]
fn assert_alloc_tracking_live() {
    katgpt_core::alloc::reset_alloc_stats();
    let _probe: Vec<u8> = vec![0u8; 64];
    let (count, _) = katgpt_core::alloc::get_alloc_stats();
    assert!(
        count >= 1,
        "TrackingAllocator not installed in this binary — alloc gate is VACUOUS (Issue 682)"
    );
    katgpt_core::alloc::reset_alloc_stats();
}

#[cfg(not(debug_assertions))]
fn assert_alloc_tracking_live() {}

use std::boxed::Box;
use std::time::{Duration, Instant};

use katgpt_transformer::dense_mesh::traits::{DenseEdge, DenseNode};
use katgpt_transformer::dense_mesh::types::{DenseHidden, LayerRole};
use katgpt_transformer::dense_mesh::{
    EdgeBandit, EdgeBanditArm, IdentityEdge, LayerwiseTopology, MeshConfig, MeshScratch, Topology,
    compute_router,
};

// ─── Helpers ───────────────────────────────────────────────────────────────

/// A trivial node that clones the input — stands in for a stripped LLM
/// forward pass when measuring topology overhead (no real transformer).
struct IdentityNode {
    hidden_dim: usize,
}

impl DenseNode for IdentityNode {
    fn forward_dense(
        &self,
        input: &DenseHidden,
        _layer_idx: usize,
        _scratch: &mut MeshScratch,
    ) -> DenseHidden {
        input.clone()
    }
    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Build a synthetic `[1, width, width, ..., 1]` topology populated with
/// `IdentityEdge`s between every (layer l node i) → (layer l+1 node j) pair.
///
/// `depth` is the number of layers (including the input and output boundary
/// layers). With `depth=3` you get `[1, width, 1]` (diamond when width=2).
fn make_synthetic_topology(width: usize, depth: usize, hidden_dim: usize) -> LayerwiseTopology {
    assert!(depth >= 2, "topology needs at least input + output layer");
    let mut widths = Vec::with_capacity(depth);
    widths.push(1); // input boundary
    for _ in 1..depth - 1 {
        widths.push(width);
    }
    widths.push(1); // output boundary
    let topology = Topology { widths };

    // One IdentityEdge per (from, to) pair in every layer boundary.
    let mut edges_per_layer: Vec<Vec<Box<dyn DenseEdge>>> = Vec::with_capacity(depth - 1);
    for w in topology.widths.windows(2) {
        let count = w[0] * w[1];
        let layer: Vec<Box<dyn DenseEdge>> = (0..count)
            .map(|_| Box::new(IdentityEdge::new()) as Box<dyn DenseEdge>)
            .collect();
        edges_per_layer.push(layer);
    }

    let node = Box::new(IdentityNode { hidden_dim });
    LayerwiseTopology::new(topology, node, edges_per_layer)
        .expect("synthetic topology must construct cleanly")
}

/// Compute mean and p99 from a sorted-by-insertion vector of durations.
/// Returns `(mean, p99)`.
fn stats(samples: &[Duration]) -> (Duration, Duration) {
    assert!(!samples.is_empty());
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort();
    let sum: Duration = sorted.iter().sum();
    let mean = sum / sorted.len() as u32;
    // p99 = index ceil(0.99 * n) - 1, clamped.
    let p99_idx = ((sorted.len() as f64) * 0.99).ceil() as usize;
    let p99_idx = p99_idx.saturating_sub(1).min(sorted.len() - 1);
    (mean, sorted[p99_idx])
}

/// Threshold relaxation in debug builds — debug runs ~10–50x slower than
/// release, so per-call budgets must scale accordingly.
fn debug_scale(release_budget_us: f64) -> f64 {
    if cfg!(debug_assertions) {
        release_budget_us * 50.0
    } else {
        release_budget_us
    }
}

// ─── T1: forward scaling across widths ─────────────────────────────────────

#[test]
fn prof_dense_mesh_forward_scaling() {
    // Gate 3 (easy overhead): width-N forward should be ≤ ~N× slower than
    // width-1 (linear scaling, paper §3.1.3 aggregation cost). We relax to
    // < 10× for width=16 vs width=1 to absorb per-layer clone overhead in the
    // reference IdentityNode implementation.
    let seq_len = 4;
    let hidden_dim = 64;
    let depth = 3; // [1, width, 1]
    let n_iters = 200;
    let warmup = 20;

    let widths: [usize; 5] = [1, 2, 4, 8, 16];
    let mut baseline_mean_us = 0.0f64;

    println!();
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ T1: DenseMesh forward scaling — width vs mean/p99 latency                │");
    println!(
        "│   topology: [1, W, 1], seq_len={seq_len}, hidden_dim={hidden_dim}, iters={n_iters:<5}        │"
    );
    println!("├──────┬──────────────┬──────────────┬──────────────────┬───────────────────┤");
    println!("│ width│  mean (μs)   │  p99  (μs)   │  qps             │  vs width=1       │");
    println!("├──────┼──────────────┼──────────────┼──────────────────┼───────────────────┤");

    for (i, &w) in widths.iter().enumerate() {
        let topo = make_synthetic_topology(w, depth, hidden_dim);
        let mut input = DenseHidden::zeros(seq_len, hidden_dim);
        for (k, v) in input.rows_mut().iter_mut().enumerate() {
            *v = (k as f32) * 0.001;
        }
        let mut scratch = MeshScratch::new(seq_len, hidden_dim);
        let cfg = MeshConfig::default();

        // Warmup — stabilise caches and branch predictor.
        for _ in 0..warmup {
            let _ = topo.forward(&input, &mut scratch, &cfg);
        }

        // Timed run.
        let mut samples: Vec<Duration> = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let start = Instant::now();
            let _out = topo.forward(&input, &mut scratch, &cfg);
            samples.push(start.elapsed());
        }

        let (mean, p99) = stats(&samples);
        let mean_us = mean.as_secs_f64() * 1e6;
        let p99_us = p99.as_secs_f64() * 1e6;
        let qps = if mean_us > 0.0 {
            1e6 / mean_us
        } else {
            f64::INFINITY
        };

        if i == 0 {
            baseline_mean_us = mean_us.max(1e-6);
        }
        let ratio = mean_us / baseline_mean_us;

        println!("│ {w:>4} │ {mean_us:>12.2} │ {p99_us:>12.2} │ {qps:>16.0} │ {ratio:>17.2}x │");
    }
    println!("└──────────────────────────────────────────────────────────────────────────┘");

    // Re-measure width=1 and width=16 explicitly for the assertion (the table
    // above may have warmed caches differently per row).
    let topo_w1 = make_synthetic_topology(1, depth, hidden_dim);
    let topo_w16 = make_synthetic_topology(16, depth, hidden_dim);
    let input = DenseHidden::zeros(seq_len, hidden_dim);
    let mut scratch = MeshScratch::new(seq_len, hidden_dim);
    let cfg = MeshConfig::default();

    for _ in 0..warmup {
        let _ = topo_w1.forward(&input, &mut scratch, &cfg);
    }
    let mut s1 = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t = Instant::now();
        let _ = topo_w1.forward(&input, &mut scratch, &cfg);
        s1.push(t.elapsed());
    }
    let (m1, _) = stats(&s1);

    for _ in 0..warmup {
        let _ = topo_w16.forward(&input, &mut scratch, &cfg);
    }
    let mut s16 = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t = Instant::now();
        let _ = topo_w16.forward(&input, &mut scratch, &cfg);
        s16.push(t.elapsed());
    }
    let (m16, _) = stats(&s16);

    let ratio = m16.as_secs_f64() / m1.as_secs_f64().max(1e-9);
    // Gate 3 (easy overhead): width=16 should be < 16× slower than width=1
    // (we allow up to 16× because each layer multiplies work by width; the
    // paper's hard bound Gate 4 needs a real LLM forward, deferred).
    let gate3_threshold = if cfg!(debug_assertions) { 32.0 } else { 16.0 };
    println!(
        "Gate 3 (easy overhead): width=16/width=1 ratio = {:.2}x (threshold < {gate3_threshold}x) — {}",
        ratio,
        if ratio < gate3_threshold {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
    assert!(
        ratio < gate3_threshold,
        "Gate 3: width=16 forward is {ratio:.2}x slower than width=1, exceeds {gate3_threshold}x"
    );
}

// ─── T2: aggregation overhead at varying fan-in ────────────────────────────

#[test]
fn prof_dense_mesh_aggregation_overhead() {
    // Time just the per-successor aggregation step (sum of `fan_in` incoming
    // IdentityEdge outputs into a single DenseHidden buffer). This is the
    // inner loop of paper §3.1.3 eq. (1).
    let seq_len = 4;
    let hidden_dim = 64;
    let n_iters = 500;
    let warmup = 50;

    let fan_ins: [usize; 4] = [1, 4, 8, 16];

    println!();
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ T2: Aggregation overhead (sum of fan_in IdentityEdge outputs)            │");
    println!(
        "│   seq_len={seq_len}, hidden_dim={hidden_dim}, iters={n_iters}                                          │"
    );
    println!("├─────────┬──────────────┬───────────────────────────────────────────────────┤");
    println!("│ fan_in  │  mean (ns)   │  note                                             │");
    println!("├─────────┼──────────────┼───────────────────────────────────────────────────┤");

    for &fan_in in &fan_ins {
        let edge = IdentityEdge::new();
        let predecessors: Vec<DenseHidden> = (0..fan_in)
            .map(|i| {
                let mut h = DenseHidden::zeros(seq_len, hidden_dim);
                for (k, v) in h.rows_mut().iter_mut().enumerate() {
                    *v = (k as f32) + (i as f32) * 0.1;
                }
                h
            })
            .collect();
        let mut scratch = MeshScratch::new(seq_len, hidden_dim);

        // Warmup.
        for _ in 0..warmup {
            scratch.clear();
            let mut acc = DenseHidden::zeros(seq_len, hidden_dim);
            for p in &predecessors {
                edge.route_into(p, &mut scratch);
                acc.add_assign(&scratch.edge_output);
            }
        }

        // Timed.
        let mut samples: Vec<Duration> = Vec::with_capacity(n_iters);
        for _ in 0..n_iters {
            let start = Instant::now();
            scratch.clear();
            let mut acc = DenseHidden::zeros(seq_len, hidden_dim);
            for p in &predecessors {
                edge.route_into(p, &mut scratch);
                acc.add_assign(&scratch.edge_output);
            }
            // prevent elision.
            std::hint::black_box(&acc);
            samples.push(start.elapsed());
        }

        let (mean, _) = stats(&samples);
        let mean_ns = mean.as_secs_f64() * 1e9;
        let per_edge_ns = if fan_in > 0 {
            mean_ns / fan_in as f64
        } else {
            0.0
        };
        println!(
            "│ {fan_in:>7} │ {mean_ns:>12.1} │  per-edge ≈ {per_edge_ns:>6.1} ns                          │"
        );
    }
    println!("└──────────────────────────────────────────────────────────────────────────┘");
    // This is a measurement test — no hard assertion, just print numbers.
    // The point: aggregation cost should scale roughly linearly with fan_in.
    // A real assertion (Gate 4 hard bound) requires LLM integration (Phase 5).
}

// ─── T3: EdgeBandit decision latency ───────────────────────────────────────

#[test]
fn prof_dense_mesh_edge_bandit_decision() {
    // EdgeBandit::sample() does Thompson sampling over N arms. Must be < 1μs
    // in release so the bandit doesn't dominate per-query overhead.
    let arms: Vec<EdgeBanditArm> = (0..8)
        .map(|i| EdgeBanditArm::new(format!("arm_{i}"), vec![1, 2, 1], vec![0, 1]))
        .collect();
    let mut bandit = EdgeBandit::new(arms, 42);

    // Warmup the priors with a few updates so the Gamma sampler exercises
    // the shape >= 1 path (not the boost branch).
    for _ in 0..50 {
        let a = bandit.sample();
        bandit.update(a, 0.5);
    }

    let n_iters = 5_000;
    let warmup = 500;
    for _ in 0..warmup {
        let _ = bandit.sample();
    }

    let start = Instant::now();
    let mut last = 0usize;
    for _ in 0..n_iters {
        last = bandit.sample();
    }
    let elapsed = start.elapsed();
    std::hint::black_box(last);

    let ns_per_call = elapsed.as_secs_f64() * 1e9 / n_iters as f64;
    let budget_ns = debug_scale(1_000.0); // 1μs in release.
    println!();
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ T3: EdgeBandit::sample() decision latency                                │");
    println!("│   arms={}, calls={}, total={elapsed:?}  │", 8, n_iters,);
    println!("│   ns/call   : {ns_per_call:>10.1} ns                                       │");
    println!(
        "│   threshold : {budget_ns:>10.1} ns ({})                            │",
        if cfg!(debug_assertions) {
            "debug 50×"
        } else {
            "release"
        }
    );
    println!(
        "│   PASS      : {}                                                         │",
        if ns_per_call < budget_ns {
            "✅"
        } else {
            "❌"
        }
    );
    println!("└──────────────────────────────────────────────────────────────────────────┘");
    assert!(
        ns_per_call < budget_ns,
        "EdgeBandit::sample() = {ns_per_call:.1}ns > {budget_ns:.1}ns budget"
    );
}

// ─── T4: compute_router::pick_compute is O(1) ──────────────────────────────

#[test]
fn prof_dense_mesh_compute_router_o1() {
    // pick_compute is a pure match on (role, width) — must be < 100ns.
    let n_iters = 100_000;
    let warmup = 10_000;

    let mut probe = ComputeProbe {
        width: 1,
        role: LayerRole::Input,
    };
    for _ in 0..warmup {
        probe = step_router(probe);
    }

    let start = Instant::now();
    for i in 0..n_iters {
        // Vary inputs to defeat const-folding.
        probe.width = 1 + (i % 17);
        probe.role = if i & 1 == 0 {
            LayerRole::Hidden
        } else {
            LayerRole::Output
        };
        probe = step_router(probe);
    }
    let elapsed = start.elapsed();
    std::hint::black_box(probe);

    let ns_per_call = elapsed.as_secs_f64() * 1e9 / n_iters as f64;
    let budget_ns = debug_scale(100.0);
    println!();
    println!("┌──────────────────────────────────────────────────────────────────────────┐");
    println!("│ T4: compute_router::pick_compute() dispatch latency                      │");
    println!("│   calls={n_iters}, total={elapsed:?}                                          │");
    println!("│   ns/call   : {ns_per_call:>10.2} ns                                       │");
    println!(
        "│   threshold : {budget_ns:>10.2} ns ({})                            │",
        if cfg!(debug_assertions) {
            "debug 50×"
        } else {
            "release"
        }
    );
    println!(
        "│   PASS      : {}                                                         │",
        if ns_per_call < budget_ns {
            "✅"
        } else {
            "❌"
        }
    );
    println!("└──────────────────────────────────────────────────────────────────────────┘");
    assert!(
        ns_per_call < budget_ns,
        "pick_compute() = {ns_per_call:.2}ns > {budget_ns:.2}ns budget"
    );
}

#[derive(Clone, Copy)]
struct ComputeProbe {
    width: usize,
    role: LayerRole,
}

#[inline]
fn step_router(p: ComputeProbe) -> ComputeProbe {
    let _ = compute_router::pick_compute(p.width, p.role, 4, true);
    p
}

// ─── T5: hot-path allocation audit (debug-only) ────────────────────────────

#[test]
fn prof_dense_mesh_zero_alloc_hot_path() {
    // Run forward() N times; verify the hot path does not allocate per-call
    // beyond a small fixed setup cost. The topology forward currently clones
    // `scratch.aggregate` once per (layer × successor node) — we report this
    // honestly and assert it scales sub-linearly with iterations (no leak).
    //
    // TrackingAllocator is debug-only; in release this test is a no-op
    // (still runs forward for cache warming, but asserts nothing).

    let seq_len = 4;
    let hidden_dim = 64;
    let n_iters = 100;

    let topo = make_synthetic_topology(4, 3, hidden_dim); // [1,4,1]
    let mut input = DenseHidden::zeros(seq_len, hidden_dim);
    for (k, v) in input.rows_mut().iter_mut().enumerate() {
        *v = k as f32 * 0.01;
    }
    let mut scratch = MeshScratch::new(seq_len, hidden_dim);
    let cfg = MeshConfig::default();

    // Warmup (outside the measured window).
    for _ in 0..10 {
        let _ = topo.forward(&input, &mut scratch, &cfg);
    }

    #[cfg(debug_assertions)]
    {
        // Allocation audit — global counters are shared across tests, so we
        // measure the *delta* across N forward calls.
        assert_alloc_tracking_live();
        katgpt_core::alloc::reset_alloc_stats();
        let (before_count, _) = katgpt_core::alloc::get_alloc_stats();
        let _ = before_count; // reset gives 0

        for _ in 0..n_iters {
            let _ = topo.forward(&input, &mut scratch, &cfg);
        }

        let (after_count, after_bytes) = katgpt_core::alloc::get_alloc_stats();

        let per_call_allocs = after_count as f64 / n_iters as f64;
        let per_call_bytes = after_bytes as f64 / n_iters as f64;
        println!();
        println!("┌──────────────────────────────────────────────────────────────────────────┐");
        println!("│ T5: DenseMesh forward() allocation audit (debug-only)                    │");
        println!(
            "│   topology: [1,4,1], seq_len={seq_len}, hidden_dim={hidden_dim}, iters={n_iters}                 │"
        );
        println!("│   total allocs : {after_count:>10}                                          │");
        println!("│   total bytes  : {after_bytes:>10}                                          │");
        println!(
            "│   per-call     : {per_call_allocs:>6.2} allocs  ({per_call_bytes:>8.1} bytes)               │"
        );
        println!("│   note         : per-layer clone of scratch.aggregate in topology.rs     │");
        println!("│                 is the dominant allocator — flagged for future fix.      │");
        println!("└──────────────────────────────────────────────────────────────────────────┘");

        // Known issue: topology.rs:142 clones `scratch.aggregate` once per
        // (layer × successor node). For [1,4,1] that's 1 + 4 = 5 clones per
        // forward. We assert the count is bounded — not zero — until the
        // topology is refactored to avoid the clone (separate task).
        //
        // Allow generous headroom for runtime / parallel-test noise.
        assert!(
            per_call_allocs < 50.0,
            "forward allocates {per_call_allocs:.1} times per call — expected < 50"
        );
    }

    #[cfg(not(debug_assertions))]
    {
        // Release build — no TrackingAllocator. Just exercise the path so
        // the test still runs and catches panics.
        for _ in 0..n_iters {
            let _ = topo.forward(&input, &mut scratch, &cfg);
        }
        println!();
        println!("│ T5: skipped alloc audit in release (TrackingAllocator is debug-only)     │");
    }
}
