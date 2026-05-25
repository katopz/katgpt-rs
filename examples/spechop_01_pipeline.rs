//! SpecHop Pipeline Demo — 4-Hop Continuous Speculation (Plan 131, T40)
//!
//! Demonstrates the continuous multi-hop speculation pipeline with:
//! - CacheSpeculator pre-populated with known tool responses
//! - RuleBasedVerifier for observation equivalence checking
//! - 4-hop trajectory simulating a multi-step retrieval agent
//! - Commit/rollback behavior on hit/miss
//! - Early termination on final answer
//!
//! The pipeline speculates on tool observations ahead of actual execution.
//! When the target tool returns, the verifier checks equivalence:
//! - Match → commit (speculation was correct)
//! - Mismatch → rollback (discard speculation, use real result)
//!
//! Run: `cargo run --example spechop_01_pipeline --features spechop`

use katgpt_rs::spechop::{
    CacheSpeculator, HopCandidate, HopMarginal, HopTreeConfig, PipelineResult, RuleBasedVerifier,
    SpecHopConfig, SpecHopPipeline, TrajectoryHop, build_hop_dd_tree, extract_best_hop_path,
    verify_hop_tree,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SpecHop Pipeline — 4-Hop Continuous Speculation Demo      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Configuration ────────────────────────────────────────────
    println!("═══ Configuration ═══");
    let config = SpecHopConfig {
        alpha: 0.2, // Speculator takes 20% of target tool time
        beta: 0.15, // Decode takes 15% of target tool time
        p: 0.7,     // 70% speculator accuracy
        k: Some(4), // Up to 4 speculative threads
        volatility: 0.4,
    };

    println!("  α = {} (speculator latency ratio)", config.alpha);
    println!("  β = {} (decode-to-tool ratio)", config.beta);
    println!("  p = {} (speculator hit rate)", config.p);
    println!("  k = {} (max speculative threads)", config.effective_k());
    println!();

    // ── Scenario: Multi-step retrieval agent ─────────────────────
    println!("═══ Scenario: Multi-Step Retrieval Agent ═══");
    println!();
    println!("  Agent trajectory:");
    println!("    1. search:rust      → Find Rust language info");
    println!("    2. search:python    → Find Python comparison");
    println!("    3. search:go        → Find Go comparison");
    println!("    4. compute:summarize → Summarize findings");
    println!();

    // ── Cache Speculator Setup ───────────────────────────────────
    // Pre-populate cache with known tool responses.
    // Some entries match the actual results (hit), some differ (miss),
    // and some actions are not in cache at all (direct commit).

    let speculator = CacheSpeculator::with_entries(vec![
        // Hop 1: Exact match — speculator predicts correctly
        (
            "search:rust",
            "Rust is a systems programming language focused on safety and performance",
        ),
        // Hop 3: Partial mismatch — speculator has a different version
        ("search:go", "Go is a compiled language designed at Google"),
        // Hop 2 and 4 are NOT in cache → direct commit path
    ]);

    let verifier = RuleBasedVerifier::default();

    // ── Build Pipeline ───────────────────────────────────────────
    let pipeline = SpecHopPipeline::new(config.clone(), speculator, verifier);
    let mut pipeline = pipeline.with_early_stop("SUMMARY:");

    println!("═══ Pipeline Execution ═══");
    println!();

    // Actual tool responses (what the real tools would return)
    let trajectory = vec![
        TrajectoryHop::new(
            "search:rust",
            "Rust is a systems programming language focused on safety and performance",
        ),
        TrajectoryHop::new(
            "search:python",
            "Python is an interpreted high-level programming language",
        ),
        TrajectoryHop::new(
            "search:go",
            "Go is a statically typed compiled language designed at Google for simplicity",
        ),
        TrajectoryHop::final_hop(
            "compute:summarize",
            "SUMMARY: Rust offers safety, Python offers simplicity, Go offers concurrency",
        ),
    ];

    // Execute the speculative pipeline
    let result = pipeline.execute(&trajectory);

    // ── Results ──────────────────────────────────────────────────
    print_results(&result);

    // ── Hop DDTree Integration ───────────────────────────────────
    println!();
    println!("═══ Hop-Level DDTree ═══");
    println!();
    println!("  Building hop DDTree from speculator confidence distributions...");
    println!();

    let marginals = vec![
        HopMarginal {
            action: "search:rust".to_string(),
            candidates: vec![
                HopCandidate::new(
                    "Rust is a systems programming language focused on safety and performance",
                    0.95,
                ),
                HopCandidate::new("Rust is a safe language", 0.5),
            ],
        },
        HopMarginal {
            action: "search:python".to_string(),
            candidates: vec![HopCandidate::new("Python is an interpreted language", 0.6)],
        },
        HopMarginal {
            action: "search:go".to_string(),
            candidates: vec![
                HopCandidate::new("Go is a compiled language designed at Google", 0.85),
                HopCandidate::new("Go is a concurrent language by Google", 0.4),
            ],
        },
        HopMarginal {
            action: "compute:summarize".to_string(),
            candidates: vec![HopCandidate::new(
                "SUMMARY: Rust offers safety, Python offers simplicity, Go offers concurrency",
                0.9,
            )],
        },
    ];

    let tree_config = HopTreeConfig {
        tree_budget: 32,
        confidence_floor: 0.01,
        chain_seed: true,
    };

    let tree = build_hop_dd_tree(&marginals, &tree_config);
    println!("  Tree nodes: {}", tree.len());

    let best_path = extract_best_hop_path(&tree);
    println!("  Best path length: {} hops", best_path.len());
    println!();

    for (i, (action, obs)) in best_path.iter().enumerate() {
        let truncated = if obs.len() > 50 {
            format!("{}...", &obs[..47])
        } else {
            obs.clone()
        };
        println!("  Hop {i}: {action}");
        println!("         {truncated}");
    }

    // Verify hop tree against actual observations
    let actual: Vec<(String, String)> = trajectory
        .iter()
        .map(|h| (h.action.clone(), h.o_target.clone()))
        .collect();

    let verifier = RuleBasedVerifier::default();
    let verified = verify_hop_tree(&tree, &actual, &verifier);

    println!();
    println!("  Verification results:");
    println!(
        "    Commits:       {} (speculation matched)",
        verified.commits
    );
    println!(
        "    Rollbacks:     {} (speculation mismatched)",
        verified.rollbacks
    );
    println!(
        "    Direct commits: {} (no speculation available)",
        verified.direct_commits
    );
    println!("    Accuracy:      {:.1}%", verified.accuracy() * 100.0);

    // ── Summary ──────────────────────────────────────────────────
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  SpecHop provides hop-level speculation for multi-step agents.");
    println!("  The pipeline speculates on tool observations while the LLM");
    println!("  continues decoding. When the target returns, verification");
    println!("  determines commit or rollback — lossless under verifier.");
    println!();
    println!("  Key metrics for this demo:");
    println!(
        "    Pipeline:  {} hits, {} misses, {} direct commits",
        result.speculation_hits, result.speculation_misses, result.direct_commits
    );
    println!(
        "    DDTree:    {} commits, {} rollbacks, {:.0}% accuracy",
        verified.commits,
        verified.rollbacks,
        verified.accuracy() * 100.0
    );
    println!("══════════════════════════════════════════════════════════════");
}

fn print_results(result: &PipelineResult) {
    println!("  Total hops:       {}", result.total_hops);
    println!(
        "  Speculation hits: {}  (predicted correctly)",
        result.speculation_hits
    );
    println!(
        "  Speculation miss: {}  (predicted incorrectly, rolled back)",
        result.speculation_misses
    );
    println!(
        "  Direct commits:   {}  (no prediction available)",
        result.direct_commits
    );
    println!(
        "  Total committed:  {}  (all hops eventually committed)",
        result.total_committed()
    );
    println!(
        "  Early terminated: {}  (final answer found)",
        result.early_terminated
    );
    println!();

    if result.speculation_hits + result.speculation_misses > 0 {
        println!("  Speculator accuracy: {:.1}%", result.accuracy() * 100.0);
        println!("  Speculation coverage: {:.1}%", result.coverage() * 100.0);
    }
    println!();

    // Show committed observations in order
    println!("  Committed observations:");
    for (i, obs) in result.committed.iter().enumerate() {
        let state_icon = match obs.state {
            katgpt_rs::spechop::HopState::Committed => "✓",
            katgpt_rs::spechop::HopState::RolledBack => "↺",
            _ => "?",
        };

        let target = obs.o_target.as_deref().unwrap_or("(pending)");
        let truncated = if target.len() > 55 {
            format!("{}...", &target[..52])
        } else {
            target.to_string()
        };

        println!("    {state_icon} Hop {i}: {action}", action = obs.action);
        println!("           {truncated}");
    }
}
