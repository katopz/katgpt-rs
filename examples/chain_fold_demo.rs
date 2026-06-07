//! ThoughtFold Chain Folding Demo — Before vs After (Plan 195 T7).
//!
//! Demonstrates inference-time chain folding that prunes redundant reasoning
//! steps during CoT generation, inspired by ThoughtFold (arXiv:2606.03503).
//!
//! Run with:
//!   cargo run --features chain_fold --example chain_fold_demo

#![cfg(feature = "chain_fold")]

use katgpt_rs::fold::{
    AttentionImportance, ChainFolder, FoldBandit, FoldCache, FoldContext, FoldDecision, FoldResult,
    FoldStats, StepBoundary, count_steps, detect_step_boundaries, fold_thinking_feedback,
    step_reduction_ratio, token_reduction_ratio,
};
use katgpt_rs::types::Rng;

fn main() {
    println!("=== ThoughtFold Chain Folding Demo ===\n");

    // ── Section 1: Step Boundary Detection ─────────────────────────
    println!("--- Step Boundary Detection ---");

    let cot_text = "First, I analyze the problem.\n\n\
                    Next, I identify constraints.\n\n\
                    <think_analysis>Deep reasoning about edge cases</think_analysis>\n\n\
                    Then I evaluate each option.\n\n\
                    Finally, I verify the solution.\n\n\
                    <think_analysis>Double-check the math</think_analysis>\n\n\
                    The answer is 42.";

    let boundaries = detect_step_boundaries(cot_text);
    println!(
        "  CoT text: {} chars, {} steps detected",
        cot_text.len(),
        boundaries.len()
    );
    println!();
    println!(
        "  {:>4}  {:>10}  {:>8}  Context",
        "Step", "Token Pos", "Anchor"
    );
    println!("  {:-<4}  {:-<10}  {:-<8}  {:-<20}", "", "", "", "");

    for b in &boundaries {
        let context_snippet = if b.token_pos < cot_text.len() {
            let end = (b.token_pos + 25).min(cot_text.len());
            let raw = &cot_text[b.token_pos..end];
            raw.replace('\n', "\\n")
        } else {
            "(end)".to_string()
        };
        println!(
            "  {:>4}  {:>10}  {:>8}  {}",
            b.step_index, b.token_pos, b.is_anchor, context_snippet
        );
    }

    println!();
    println!("  Total steps: {}", count_steps(cot_text));
    println!();

    // ── Section 2: Attention Importance Scoring ────────────────────
    println!("--- Attention Importance Scoring ---");

    // Simulate 6 steps with varying attention weights.
    // Steps 0, 3 are essential (high attention), steps 2, 5 are redundant (low).
    let raw_attention: Vec<f32> = vec![
        // Step 0: high attention (essential setup)
        0.9, 0.8, 0.85, // Step 1: moderate attention
        0.5, 0.6, 0.55, // Step 2: low attention (redundant detail)
        0.1, 0.15, 0.12, // Step 3: high attention (think-tag anchor)
        0.95, 0.88, 0.92, // Step 4: moderate attention
        0.45, 0.5, 0.48, // Step 5: low attention (redundant recap)
        0.08, 0.1, 0.09,
    ];

    let scorer = AttentionImportance::new();
    let importance = scorer.score_steps(&raw_attention, &boundaries);

    println!("  {:>4}  {:>12}  Category", "Step", "Importance");
    println!("  {:-<4}  {:-<12}  {:-<20}", "", "", "");
    for (i, &score) in importance.iter().enumerate() {
        let category = if score > 0.7 {
            "essential"
        } else if score > 0.4 {
            "moderate"
        } else {
            "redundant"
        };
        println!("  {:>4}  {:>12.4}  {}", i, score, category);
    }
    println!();

    // ── Section 3: Chain Folding — Before vs After ─────────────────
    println!("--- Chain Folding: Before vs After ---");

    // Build a 10-step context with realistic attention scores.
    // Steps 0, 3, 6 are anchors (think-tag transitions).
    // Steps 2, 5, 8 have low importance (redundant).
    let step_tokens = 15; // tokens per step
    let attention_scores: Vec<f32> = vec![
        // Step 0: anchor, essential
        0.9, 0.85, 0.88, 0.91, 0.87, 0.90, 0.86, 0.89, 0.92, 0.88, 0.85, 0.90, 0.87, 0.91, 0.89,
        // Step 1: moderate
        0.5, 0.55, 0.48, 0.52, 0.50, 0.53, 0.49, 0.51, 0.54, 0.47, 0.50, 0.52, 0.49, 0.53, 0.51,
        // Step 2: low importance (redundant detail)
        0.1, 0.12, 0.09, 0.11, 0.08, 0.10, 0.13, 0.09, 0.11, 0.12, 0.08, 0.10, 0.11, 0.09, 0.12,
        // Step 3: anchor, essential
        0.92, 0.88, 0.95, 0.90, 0.87, 0.93, 0.89, 0.91, 0.94, 0.88, 0.90, 0.92, 0.87, 0.93, 0.91,
        // Step 4: moderate
        0.45, 0.50, 0.48, 0.47, 0.52, 0.46, 0.49, 0.51, 0.48, 0.50, 0.47, 0.52, 0.49, 0.46, 0.50,
        // Step 5: low importance (redundant)
        0.08, 0.10, 0.09, 0.07, 0.11, 0.08, 0.10, 0.09, 0.12, 0.08, 0.09, 0.10, 0.07, 0.11, 0.09,
        // Step 6: anchor, essential
        0.88, 0.91, 0.87, 0.93, 0.90, 0.89, 0.92, 0.86, 0.91, 0.88, 0.90, 0.87, 0.93, 0.89, 0.91,
        // Step 7: moderate
        0.50, 0.48, 0.52, 0.49, 0.51, 0.47, 0.53, 0.50, 0.48, 0.52, 0.49, 0.51, 0.50, 0.48, 0.52,
        // Step 8: low importance (redundant)
        0.09, 0.11, 0.08, 0.10, 0.07, 0.12, 0.09, 0.08, 0.11, 0.10, 0.09, 0.07, 0.10, 0.08, 0.11,
        // Step 9: moderate (closing)
        0.55, 0.58, 0.52, 0.54, 0.56, 0.53, 0.57, 0.55, 0.54, 0.56, 0.53, 0.58, 0.55, 0.52, 0.56,
    ];

    let fold_boundaries: Vec<StepBoundary> = (0..10)
        .map(|i| {
            let is_anchor = i == 0 || i == 3 || i == 6;
            StepBoundary::new(i * step_tokens, i, is_anchor)
        })
        .collect();

    let context = FoldContext {
        importance_scores: attention_scores.clone(),
        boundaries: fold_boundaries,
        fold_budget: 0.6,
    };

    println!("  BEFORE: {} reasoning steps", context.step_count());
    println!(
        "  Anchors: steps {}",
        context
            .boundaries
            .iter()
            .filter(|b| b.is_anchor)
            .map(|b| b.step_index)
            .collect::<Vec<_>>()
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut folder = ChainFolder::new(0.6);
    let result = folder.binary_search_fold(&context);

    println!();
    println!("  AFTER (fold_budget = 0.6):");
    println!("    Total steps:  {}", result.total_steps);
    println!("    Kept steps:   {}", result.kept_steps);
    println!("    Folded steps: {}", result.folded_steps);
    println!("    Tokens saved: {}", result.tokens_saved);
    println!("    Retention:    {:.1}%", result.retention_ratio * 100.0);
    println!(
        "    Verification: {}",
        if result.verification_passed {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!();

    println!("  Per-step decisions:");
    println!("  {:>4}  {:>10}  {:>8}", "Step", "Decision", "Anchor");
    println!("  {:-<4}  {:-<10}  {:-<8}", "", "", "");
    for (i, decision) in folder.decisions().iter().enumerate() {
        let label = match decision {
            FoldDecision::Keep => "Keep",
            FoldDecision::Fold => "FOLD",
            FoldDecision::Anchor => "Anchor",
        };
        let anchor = context.boundaries[i].is_anchor;
        println!("  {:>4}  {:>10}  {:>8}", i, label, anchor);
    }
    println!();

    // ── Section 4: FoldCache KV Rollback Planning ──────────────────
    println!("--- FoldCache KV Rollback Planning ---");

    let mut cache = FoldCache::new(result.total_steps);

    // Truncate to the first folded step's position.
    let first_folded = folder
        .decisions()
        .iter()
        .position(|d| *d == FoldDecision::Fold);
    if let Some(step) = first_folded {
        cache.truncate_to_step(step, &context.boundaries);
        println!(
            "  Truncate KV cache to step {} (token pos: {})",
            step,
            cache.truncate_pos().unwrap_or(0)
        );
    }

    // Replay only essential steps.
    cache.replay_essential(folder.decisions(), &context.boundaries);
    println!("  Essential steps to replay: {:?}", cache.essential_steps());
    println!(
        "  {} of {} steps retained ({} folded)",
        cache.essential_count(),
        cache.total_steps(),
        cache.total_steps() - cache.essential_count()
    );

    let essential_positions = cache.essential_token_positions(&context.boundaries);
    println!("  Essential token positions: {:?}", essential_positions);
    println!();

    // ── Section 5: Bandit Self-Tuning ──────────────────────────────
    println!("--- Bandit Self-Tuning (50 episodes) ---");

    let mut bandit = FoldBandit::new();
    let mut rng = Rng::new(42);

    let budget_arms = [0.3_f32, 0.5, 0.7, 0.9, 1.0];
    let mut arm_pulls = [0u32; 5];

    for _ in 0..50 {
        let budget = bandit.select_budget(&mut rng);
        let arm_idx = budget_arms
            .iter()
            .position(|&b| (b - budget).abs() < 0.01)
            .unwrap_or(2);

        // Simulate: budget 0.7 gives high reward, others give low.
        let (accepted, savings) = match arm_idx {
            2 => (true, 0.45),  // arm 0.7: best
            1 => (true, 0.20),  // arm 0.5: moderate
            0 => (false, 0.05), // arm 0.3: too aggressive
            3 => (true, 0.10),  // arm 0.9: safe but low savings
            _ => (true, 0.05),
        };

        bandit.record_reward(budget, accepted, savings);
        arm_pulls[arm_idx] += 1;
    }

    println!("  Arm pulls:");
    println!("  {:>8}  {:>6}  {:>8}", "Budget", "Pulls", "Bandit");
    println!("  {:-<8}  {:-<6}  {:-<8}", "", "", "");
    for (i, &budget) in budget_arms.iter().enumerate() {
        let best = if i == bandit.best_arm() {
            " <- best"
        } else {
            ""
        };
        println!(
            "  {:>8.1}  {:>6}  {:>8}{}",
            budget,
            arm_pulls[i],
            bandit.pulls(i),
            best
        );
    }
    println!();
    println!(
        "  Best arm:    {} (budget = {:.1})",
        bandit.best_arm(),
        bandit.best_budget()
    );
    println!("  Total pulls: {}", bandit.total_pulls());
    println!();

    // ── Section 6: Cumulative Stats ────────────────────────────────
    println!("--- Cumulative Stats (10 queries) ---");

    let mut stats = FoldStats::default();
    let query_profiles: [(usize, usize, usize, usize, f32, bool); 10] = [
        // (total_steps, kept_steps, folded_steps, tokens_saved, retention_ratio, verification_passed)
        (10, 7, 3, 45, 0.70, true),
        (12, 8, 4, 72, 0.67, true),
        (8, 6, 2, 30, 0.75, true),
        (15, 10, 5, 95, 0.67, true),
        (10, 9, 1, 15, 0.90, true),
        (6, 4, 2, 28, 0.67, false), // verification failed
        (14, 9, 5, 85, 0.64, true),
        (11, 7, 4, 66, 0.64, true),
        (9, 6, 3, 40, 0.67, true),
        (13, 8, 5, 78, 0.62, true),
    ];

    println!(
        "  {:>5}  {:>6}  {:>5}  {:>6}  {:>7}  {:>5}  {:>4}",
        "Query", "Total", "Kept", "Folded", "Saved", "Ret%", "Pass"
    );
    println!(
        "  {:-<5}  {:-<6}  {:-<5}  {:-<6}  {:-<7}  {:-<5}  {:-<4}",
        "", "", "", "", "", "", ""
    );

    for (i, &(total, kept, folded, saved, retention, passed)) in query_profiles.iter().enumerate() {
        let result = FoldResult {
            total_steps: total,
            kept_steps: kept,
            folded_steps: folded,
            tokens_saved: saved,
            retention_ratio: retention,
            verification_passed: passed,
        };
        stats.record(&result);

        let pass_str = if passed { "yes" } else { "FAIL" };
        println!(
            "  {:>5}  {:>6}  {:>5}  {:>6}  {:>7}  {:>5.0}  {:>4}",
            i + 1,
            total,
            kept,
            folded,
            saved,
            retention * 100.0,
            pass_str
        );
    }

    println!();
    println!("  Summary:");
    println!("    Total tokens saved:    {}", stats.total_tokens_saved);
    println!("    Total steps folded:    {}", stats.total_steps_folded);
    println!("    Queries folded:        {}", stats.queries_folded);
    println!(
        "    Verification pass rate: {:.1}%",
        stats.verification_pass_rate * 100.0
    );
    println!(
        "    Token reduction ratio:  {:.2} tokens/query",
        token_reduction_ratio(&stats)
    );
    println!(
        "    Step reduction ratio:   {:.2} steps/query",
        step_reduction_ratio(&stats)
    );

    // Bonus: ThinkingFold feedback integration
    println!();
    println!("  ThinkingController feedback:");
    let last_result = FoldResult {
        total_steps: 13,
        kept_steps: 8,
        folded_steps: 5,
        tokens_saved: 78,
        retention_ratio: 0.62,
        verification_passed: true,
    };
    let feedback = fold_thinking_feedback(&last_result, 0.6);
    println!("    tokens_saved:  {}", feedback.tokens_saved);
    println!("    steps_folded:  {}", feedback.steps_folded);
    println!("    fold_budget:   {:.1}", feedback.fold_budget);
    println!();

    // ── TL;DR ──────────────────────────────────────────────────────
    println!("=== TL;DR ===");
    println!(
        "Chain folding achieved {:.0}% retention with {} tokens saved across {} queries.",
        stats.verification_pass_rate * 100.0,
        stats.total_tokens_saved,
        stats.queries_folded
    );
    println!(
        "Bandit converged to budget {:.1} after 50 episodes.",
        bandit.best_budget()
    );
}
