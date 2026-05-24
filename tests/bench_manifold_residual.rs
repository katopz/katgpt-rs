//! Benchmark: Manifold Residual vs Relevance Scoring (Plan 085 T3)
//!
//! Compares residual-based branch selection vs relevance-based selection
//! on synthetic DDTree candidates.
//!
//! Run: cargo test --features deep_manifold --test bench_manifold_residual -- --nocapture

#[cfg(feature = "deep_manifold")]
mod tests {
    use katgpt_rs::pruners::{
        KlResidualScorer, L2ResidualScorer, ManifoldResidual, ResidualRelevanceScorer,
    };
    use std::time::Instant;

    // ── Helpers ──────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Generate synthetic candidate logits around a base distribution.
    fn generate_candidates(base: &[f32], n: usize, spread: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = fastrand::Rng::with_seed(seed);
        (0..n)
            .map(|_| {
                base.iter()
                    .map(|&b| b + spread * (rng.f32() - 0.5) * 2.0)
                    .collect()
            })
            .collect()
    }

    /// Generate synthetic relevance scores for candidates.
    fn generate_relevances(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = fastrand::Rng::with_seed(seed);
        (0..n).map(|_| rng.f32()).collect()
    }

    /// Softmax to convert logits to probabilities (for KL scorer).
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    // ── Correctness Tests ───────────────────────────────────────

    #[test]
    fn l2_residual_identical_is_zero() {
        let scorer = L2ResidualScorer::default();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(scorer.residual(&v, &v), 0.0, 1e-6));
    }

    #[test]
    fn kl_residual_identical_is_zero() {
        let scorer = KlResidualScorer::default();
        let v = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(scorer.residual(&v, &v), 0.0, 1e-6));
    }

    #[test]
    fn l2_residual_ordering_near_vs_far() {
        let scorer = L2ResidualScorer::default();
        let base = vec![0.5; 8];
        let near = vec![0.51; 8];
        let far = vec![1.5; 8];
        assert!(
            scorer.residual(&near, &base) < scorer.residual(&far, &base),
            "near candidate should have smaller residual than far"
        );
    }

    #[test]
    fn kl_residual_ordering_near_vs_far() {
        let scorer = KlResidualScorer::default();
        let base = softmax(&[1.0, 2.0, 3.0, 4.0]);
        let near = softmax(&[1.01, 2.01, 3.01, 4.01]);
        let far = softmax(&[4.0, 1.0, 2.0, 3.0]);
        assert!(
            scorer.residual(&near, &base) < scorer.residual(&far, &base),
            "near distribution should have smaller KL than far"
        );
    }

    // ── Selection Comparison ────────────────────────────────────

    #[test]
    fn relevance_only_vs_residual_only_selection() {
        let base = vec![0.5; 16];
        let candidates = generate_candidates(&base, 20, 0.5, 42);
        let relevances = generate_relevances(20, 123);

        let l2 = L2ResidualScorer::default();

        // Find best by pure relevance
        let best_rel_idx = relevances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Find best by pure residual (lowest residual = closest to fixed point)
        let best_res_idx = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2.residual(c, &base)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // They may differ — that's expected. Just verify both are valid.
        assert!(best_rel_idx < 20, "best_rel_idx out of range");
        assert!(best_res_idx < 20, "best_res_idx out of range");

        println!(
            "Best by relevance: candidate {best_rel_idx} (relevance={:.4})",
            relevances[best_rel_idx]
        );
        println!(
            "Best by residual:  candidate {best_res_idx} (residual={:.4})",
            l2.residual(&candidates[best_res_idx], &base)
        );
    }

    #[test]
    fn blended_selection_trades_off_correctly() {
        let base = vec![0.5; 8];
        let candidates = generate_candidates(&base, 50, 0.3, 99);
        let relevances = generate_relevances(50, 77);

        let l2 = L2ResidualScorer::default();

        // Weight=0.0 → pure relevance
        let pure_rel = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 0.0);
        // Weight=1.0 → pure residual
        let pure_res = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 1.0);
        // Weight=0.5 → balanced
        let balanced = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 0.5);

        let best_pure_rel = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, pure_rel.score(c, &base, relevances[i])))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let best_pure_res = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, pure_res.score(c, &base, relevances[i])))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let best_balanced = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, balanced.score(c, &base, relevances[i])))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        println!("Pure relevance winner: candidate {best_pure_rel}");
        println!("Pure residual winner:  candidate {best_pure_res}");
        println!("Balanced winner:       candidate {best_balanced}");

        // Pure relevance best should have highest relevance
        let best_rel_relevance = relevances[best_pure_rel];
        for (i, &rel) in relevances.iter().enumerate() {
            assert!(
                rel <= best_rel_relevance + 1e-6,
                "candidate {i} has relevance {rel} > {best_rel_relevance}"
            );
        }

        // Pure residual best should have lowest residual
        let best_res_residual = l2.residual(&candidates[best_pure_res], &base);
        for (i, c) in candidates.iter().enumerate() {
            let res = l2.residual(c, &base);
            assert!(
                res >= best_res_residual - 1e-6,
                "candidate {i} has residual {res} < {best_res_residual}"
            );
        }
    }

    // ── Convergence Test ────────────────────────────────────────

    #[test]
    fn convergence_tracking_works() {
        let scorer = L2ResidualScorer { tolerance: 0.01 };
        let base = vec![0.5; 8];

        // Simulate a sequence of candidates approaching the base
        let alphas = [1.0, 0.5, 0.2, 0.05, 0.005, 0.001];
        let mut converged_at = None;

        for (step, &alpha) in alphas.iter().enumerate() {
            let candidate: Vec<f32> = base.iter().map(|&b| b + alpha).collect();
            let residual = scorer.residual(&candidate, &base);
            let converged = scorer.is_converged(residual, scorer.tolerance);

            println!("Step {step}: α={alpha:.3}, residual={residual:.6}, converged={converged}");

            if converged && converged_at.is_none() {
                converged_at = Some(step);
            }
        }

        assert!(
            converged_at.is_some(),
            "should converge as candidates approach base"
        );
        println!("Converged at step {}", converged_at.unwrap());
    }

    // ── Per-Position Analysis ───────────────────────────────────

    #[test]
    fn per_position_residual_identifies_hotspots() {
        let scorer = L2ResidualScorer::default();
        let base = vec![0.5; 8];
        // Candidate diverges at positions 3 and 7
        let mut candidate = base.clone();
        candidate[3] = 2.0;
        candidate[7] = 1.5;

        let pp = scorer.per_position_residual(&candidate, &base);

        assert!(
            pp[3] > pp[0],
            "position 3 should have higher residual than converged positions"
        );
        assert!(
            pp[7] > pp[1],
            "position 7 should have higher residual than converged positions"
        );
        assert!(
            approx_eq(pp[0], 0.0, 1e-6),
            "unchanged positions should have zero residual"
        );

        println!("Per-position residuals: {pp:?}");
        println!("Hotspots: pos 3={:.4}, pos 7={:.4}", pp[3], pp[7]);
    }

    // ── Performance Benchmarks ──────────────────────────────────

    #[test]
    fn bench_l2_residual_performance() {
        let scorer = L2ResidualScorer::default();
        let dims = [64, 256, 1024, 4096];
        let iterations = 10_000;

        println!("\n── L2 Residual Performance ──");
        println!("{:<10} {:<15} {:<15}", "Dim", "ns/call", "µs/call");
        println!("{}", "─".repeat(40));

        for &dim in &dims {
            let base: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let candidate: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).cos()).collect();

            // Warmup
            for _ in 0..100 {
                let _ = scorer.residual(&candidate, &base);
            }

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = scorer.residual(&candidate, &base);
            }
            let elapsed = start.elapsed();
            let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;

            println!(
                "{:<10} {:<15.1} {:<15.3}",
                dim,
                ns_per_call,
                ns_per_call / 1000.0
            );
        }
    }

    #[test]
    fn bench_kl_residual_performance() {
        let scorer = KlResidualScorer::default();
        let dims = [64, 256, 1024];
        let iterations = 10_000;

        println!("\n── KL Residual Performance ──");
        println!("{:<10} {:<15} {:<15}", "Dim", "ns/call", "µs/call");
        println!("{}", "─".repeat(40));

        for &dim in &dims {
            let logits_base: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let logits_cand: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();
            let base = softmax(&logits_base);
            let candidate = softmax(&logits_cand);

            // Warmup
            for _ in 0..100 {
                let _ = scorer.residual(&candidate, &base);
            }

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = scorer.residual(&candidate, &base);
            }
            let elapsed = start.elapsed();
            let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;

            println!(
                "{:<10} {:<15.1} {:<15.3}",
                dim,
                ns_per_call,
                ns_per_call / 1000.0
            );
        }
    }

    #[test]
    fn bench_blended_scoring_throughput() {
        let n_candidates = 100;
        let dim = 256;
        let iterations = 1_000;

        let base: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let candidates: Vec<Vec<f32>> = (0..n_candidates)
            .map(|j| (0..dim).map(|i| ((i + j) as f32 * 0.01).sin()).collect())
            .collect();
        let relevances: Vec<f32> = (0..n_candidates)
            .map(|j| (j as f32 * 0.1).sin().abs())
            .collect();

        let scorer = ResidualRelevanceScorer::new(L2ResidualScorer::default(), 0.5);

        // Warmup
        for _ in 0..100 {
            for (i, c) in candidates.iter().enumerate() {
                let _ = scorer.score(c, &base, relevances[i]);
            }
        }

        let start = Instant::now();
        for _ in 0..iterations {
            for (i, c) in candidates.iter().enumerate() {
                let _ = scorer.score(c, &base, relevances[i]);
            }
        }
        let elapsed = start.elapsed();
        let total_scores = n_candidates * iterations;
        let ns_per_score = elapsed.as_nanos() as f64 / total_scores as f64;

        println!("\n── Blended Scoring Throughput ──");
        println!("Candidates: {n_candidates}, dim: {dim}, iterations: {iterations}");
        println!("Total scores: {total_scores}");
        println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("Per score: {ns_per_score:.1}ns");
        println!(
            "Throughput: {:.0} scores/sec",
            1_000_000_000.0 / ns_per_score
        );
    }

    // ── Bulk Selection Benchmark ────────────────────────────────

    #[test]
    fn bench_selection_from_candidates() {
        let n_candidates = 50;
        let dim = 256;
        let rounds = 10_000;

        let base: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let candidates: Vec<Vec<f32>> = (0..n_candidates)
            .map(|j| {
                let mut rng = fastrand::Rng::with_seed(j as u64);
                (0..dim)
                    .map(|_| base[0] + (rng.f32() - 0.5) * 0.4)
                    .collect()
            })
            .collect();
        let relevances: Vec<f32> = (0..n_candidates)
            .map(|j| (j as f32 * 0.3).sin() * 0.5 + 0.5)
            .collect();

        let l2 = L2ResidualScorer::default();
        let scorer = ResidualRelevanceScorer::new(l2, 0.3);

        let start = Instant::now();
        let mut best_indices = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let best = candidates
                .iter()
                .enumerate()
                .map(|(i, c)| (i, scorer.score(c, &base, relevances[i])))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            best_indices.push(best);
        }
        let elapsed = start.elapsed();

        let us_per_selection = elapsed.as_micros() as f64 / rounds as f64;
        println!("\n── Selection Benchmark ──");
        println!("Candidates: {n_candidates}, dim: {dim}, rounds: {rounds}");
        println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("Per selection: {us_per_selection:.1}µs");
        println!(
            "Throughput: {:.0} selections/sec",
            1_000_000.0 / us_per_selection
        );

        // Verify selection is deterministic
        assert!(
            best_indices.windows(2).all(|w| w[0] == w[1]),
            "selection should be deterministic with fixed inputs"
        );
    }
}
