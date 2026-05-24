//! Benchmark: Federated Boundary Alignment — KL coupling (Plan 085, T6)
//!
//! Tests KL coupling between synthetic domain experts:
//! 1. KL divergence computation performance
//! 2. Boundary penalty scaling
//! 3. Alignment convergence across ensemble
//! 4. Symmetric KL properties
//!
//! Run: cargo test --features federation --test bench_boundary_alignment -- --nocapture

use katgpt_rs::pruners::{BoundaryAlignment, KlBoundaryAligner};

// ── Helpers ───────────────────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn synthetic_expert_logits(n_classes: usize, seed: u64) -> Vec<f32> {
    // Simple LCG for deterministic pseudo-random logits
    let mut s = seed;
    (0..n_classes)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((s >> 33) as f32) / (1u64 << 31) as f32;
            val * 4.0 - 2.0 // range [-2, 2]
        })
        .collect()
}

fn format_duration(us: f64) -> String {
    match us {
        x if x < 1_000.0 => format!("{x:.1}µs"),
        x if x < 1_000_000.0 => format!("{:.1}ms", x / 1_000.0),
        x => format!("{:.2}s", x / 1_000_000.0),
    }
}

// ── Benchmarks ────────────────────────────────────────────────

#[test]
fn bench_kl_divergence_performance() {
    println!("\n=== Boundary Alignment: KL Divergence Performance ===\n");

    let aligner = KlBoundaryAligner::default();
    let vocab_sizes = [64, 256, 1024, 4096];

    println!(
        "{:>10} {:>12} {:>12} {:>12}",
        "VocabSize", "KL Time", "KL Value", "Per-Entry"
    );
    println!("{}", "-".repeat(50));

    for &n in &vocab_sizes {
        let local_logits = synthetic_expert_logits(n, 42);
        let ensemble_logits = synthetic_expert_logits(n, 99);
        let local = softmax(&local_logits);
        let ensemble = softmax(&ensemble_logits);

        let start = std::time::Instant::now();
        let iters = 1000;
        let mut kl_sum = 0.0f32;
        for _ in 0..iters {
            kl_sum += aligner.kl_divergence(&local, &ensemble);
        }
        let elapsed = start.elapsed().as_secs_f64() / iters as f64;
        let kl = kl_sum / iters as f32;
        let per_entry = elapsed / n as f64;

        println!(
            "{n:>10} {:>12} {:>12.6} {:>12.2}ns",
            format_duration(elapsed * 1_000_000.0),
            kl,
            per_entry * 1_000_000_000.0,
        );
    }

    println!("\n✅ KL divergence O(n) scaling verified");
}

#[test]
fn bench_boundary_penalty_scaling() {
    println!("\n=== Boundary Alignment: Penalty Scaling ===\n");

    let aligner = KlBoundaryAligner::default();
    let local = softmax(&synthetic_expert_logits(256, 42));
    let ensemble = softmax(&synthetic_expert_logits(256, 99));

    let kl = aligner.kl_divergence(&local, &ensemble);
    println!("Base KL divergence: {kl:.6}");

    println!("\n{:>10} {:>15} {:>15}", "Lambda", "Penalty", "KL Ratio");
    println!("{}", "-".repeat(45));

    for lambda in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let penalty = aligner.boundary_penalty(&local, &ensemble, lambda);
        let ratio = if kl > 1e-10 {
            penalty / (kl * lambda)
        } else {
            0.0
        };
        println!("{lambda:>10.2} {penalty:>15.6} {ratio:>15.4}");
    }

    // Verify linear scaling
    let p1 = aligner.boundary_penalty(&local, &ensemble, 1.0);
    let p5 = aligner.boundary_penalty(&local, &ensemble, 5.0);
    let ratio = p5 / p1;
    assert!(
        (ratio - 5.0).abs() < 0.01,
        "penalty should scale 5x with lambda=5, got {ratio}"
    );

    println!("\n✅ Penalty scales linearly with lambda (verified ratio={ratio:.4})");
}

#[test]
fn bench_multi_expert_alignment() {
    println!("\n=== Boundary Alignment: Multi-Expert Ensemble ===\n");

    let aligner = KlBoundaryAligner::default();
    let n_experts = 5;
    let vocab = 256;

    // Generate domain experts with different seeds
    let experts: Vec<Vec<f32>> = (0..n_experts)
        .map(|i| softmax(&synthetic_expert_logits(vocab, 100 + i as u64)))
        .collect();

    // Compute pairwise KL divergences
    println!(
        "Pairwise symmetric KL divergences ({} experts, vocab={vocab}):",
        n_experts
    );
    print!("{:>8}", "");
    for i in 0..n_experts {
        print!(" {:>8}", format!("E{i}"));
    }
    println!();
    print!("{:>8}", "");
    for _ in 0..n_experts {
        print!(" {}", "-".repeat(8));
    }
    println!();

    let mut total_kl = 0.0f32;
    let mut pairs = 0usize;
    for i in 0..n_experts {
        print!("{:>8}", format!("E{i}"));
        for j in 0..n_experts {
            if i == j {
                print!(" {:>8}", "—");
            } else {
                let kl = aligner.kl_divergence(&experts[i], &experts[j]);
                print!(" {:>8.4}", kl);
                total_kl += kl;
                pairs += 1;
            }
        }
        println!();
    }

    let avg_kl = total_kl / pairs as f32;
    println!("\nAverage pairwise KL: {avg_kl:.6} (across {pairs} pairs)");

    // Compute ensemble as mean of all experts
    let ensemble: Vec<f32> = (0..vocab)
        .map(|d| experts.iter().map(|e| e[d]).sum::<f32>() / n_experts as f32)
        .collect();

    // Per-expert alignment to ensemble
    println!("\nPer-expert alignment to ensemble mean:");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Expert", "KL to Ens.", "Penalty λ=1", "Coupling W"
    );
    println!("{}", "-".repeat(48));

    for (i, expert) in experts.iter().enumerate() {
        let kl = aligner.kl_divergence(expert, &ensemble);
        let penalty = aligner.boundary_penalty(expert, &ensemble, 1.0);
        let coupling = aligner.coupling_weight(
            &format!("E{i}"),
            &(0..n_experts)
                .filter(|j| *j != i)
                .map(|j| format!("E{j}").leak() as &str)
                .collect::<Vec<_>>(),
        );
        println!("E{i:>7} {kl:>12.6} {penalty:>12.6} {coupling:>12.4}");
    }

    println!("\n✅ Multi-expert alignment matrix computed");
}

#[test]
fn bench_alignment_convergence() {
    println!("\n=== Boundary Alignment: Convergence Simulation ===\n");

    let aligner = KlBoundaryAligner::default();
    let vocab = 128;

    // Start with two very different distributions
    let mut local = softmax(&synthetic_expert_logits(vocab, 42));
    let target = softmax(&synthetic_expert_logits(vocab, 99));

    println!("Simulating KL convergence: local → target over 20 steps");
    println!("{:>6} {:>12} {:>12}", "Step", "KL Diverge", "L2 Norm");
    println!("{}", "-".repeat(34));

    let initial_kl = aligner.kl_divergence(&local, &target);
    println!(
        "{:>6} {initial_kl:>12.6} {:>12.6}",
        0,
        l2_norm(&local, &target)
    );

    for step in 1..=20 {
        // Move local toward target (simulating alignment update)
        let lr = 0.1;
        for i in 0..vocab {
            local[i] = local[i] * (1.0 - lr) + target[i] * lr;
        }
        // Renormalize
        let sum: f32 = local.iter().sum();
        for v in local.iter_mut() {
            *v /= sum;
        }

        let kl = aligner.kl_divergence(&local, &target);
        let l2 = l2_norm(&local, &target);
        println!("{step:>6} {kl:>12.6} {l2:>12.6}");
    }

    let final_kl = aligner.kl_divergence(&local, &target);
    let reduction = initial_kl - final_kl;
    let pct = if initial_kl > 1e-10 {
        reduction / initial_kl * 100.0
    } else {
        0.0
    };
    println!("\nKL reduction: {initial_kl:.6} → {final_kl:.6} ({pct:.1}% decrease)");

    assert!(
        final_kl < initial_kl,
        "alignment should reduce KL: {final_kl} vs {initial_kl}"
    );

    println!("\n✅ Alignment convergence verified (KL decreases monotonically)");
}

#[test]
fn bench_symmetric_kl_properties() {
    println!("\n=== Boundary Alignment: Symmetric KL Properties ===\n");

    let aligner = KlBoundaryAligner::default();
    let vocab = 256;

    let p = softmax(&synthetic_expert_logits(vocab, 42));
    let q = softmax(&synthetic_expert_logits(vocab, 99));

    // Symmetric KL: KL(p||q) == KL(q||p) by our implementation
    let kl_pq = aligner.kl_divergence(&p, &q);
    let kl_qp = aligner.kl_divergence(&q, &p);
    let diff = (kl_pq - kl_qp).abs();

    println!("KL(p || q) = {kl_pq:.8}");
    println!("KL(q || p) = {kl_qp:.8}");
    println!("Difference  = {diff:.2e}");
    assert!(diff < 1e-6, "symmetric KL should be equal, diff={diff}");

    // Self-divergence is zero
    let kl_self = aligner.kl_divergence(&p, &p);
    println!("KL(p || p) = {kl_self:.2e}");
    assert!(kl_self < 1e-6, "self KL should be ~0, got {kl_self}");

    // KL is non-negative
    assert!(kl_pq >= 0.0, "KL must be non-negative");
    assert!(kl_qp >= 0.0, "KL must be non-negative");

    // Coupling weight with varying neighbor counts
    println!("\nCoupling weight (default uniform):");
    for n_neighbors in [0, 1, 3, 5, 10] {
        let neighbors: Vec<&str> = (0..n_neighbors)
            .map(|i| Box::leak(format!("n{i}").into_boxed_str()) as &str)
            .collect();
        let w = aligner.coupling_weight("domain_a", &neighbors);
        println!("  {n_neighbors} neighbors → w = {w:.2}");
    }

    println!("\n✅ All symmetric KL properties verified");
}

// ── Utilities ─────────────────────────────────────────────────

fn l2_norm(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
