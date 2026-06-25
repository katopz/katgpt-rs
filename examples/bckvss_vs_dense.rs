//! Plan 265 Phase 1 Example: BCKVSS vs. Dense KV Cache Selection.
//!
//! Demonstrates the before/after KV size and perplexity-proxy trade-off
//! when using the Band-Conditioned KV Segment Selector (Fusion A) versus
//! retaining all segments (dense baseline).

fn main() {
    #[cfg(feature = "bckvss")]
    {
        use katgpt_rs::bckvss::{
            matthews_corr, perplexity_proxy, BandConditionerSelector,
            BandConditionerSelectorConfig, QueryEmb, SegmentSelector, SyntheticScm,
        };

        println!("=== Plan 265 Phase 1: BCKVSS vs. Dense KV Cache ===\n");

        // 4-task interleaved SCM, 80 tokens, L=4 → 20 segments.
        let seg_len = 4;
        let scm = SyntheticScm::generate(80, 8, 4, 0.95, 0.02, 7);
        let segments = scm.chunk_into_segments(seg_len);
        let n = segments.len();
        println!("Synthetic KV: {n} segments of L={seg_len} tokens (d=8).");

        // Query = task 0's first segment representative.
        let query = QueryEmb {
            data: segments[0].representative_key().to_vec(),
            task_id: 1,
        };

        // Dense baseline: retain ALL segments.
        let all_idx: Vec<usize> = (0..n).collect();
        let ppl_dense = perplexity_proxy(&query, &segments, &all_idx);
        println!("\nDense baseline: {n} segments retained, ppl-proxy = {ppl_dense:.4}");

        // Sweep budgets: 25%, 50%, 75% retention.
        println!("\nBudget sweep (segment_len = {seg_len}):");
        println!(
            "  {:>10}  {:>10}  {:>12}  {:>12}",
            "budget%", "retained", "reduction%", "ppl-delta"
        );
        for budget_pct in [25, 50, 75] {
            let budget = (n * budget_pct) / 100;
            let selector = BandConditionerSelector::new(
                BandConditionerSelectorConfig::default().with_segment_len(seg_len),
            );
            let selected = selector.select(&segments, &query, budget);
            let retained = selected.len();
            let reduction = 100.0 * (1.0 - (retained as f32 / n as f32));
            let ppl_sel = perplexity_proxy(&query, &segments, &selected);
            let delta = (ppl_sel - ppl_dense).abs();
            println!(
                "  {:>9}%  {:>10}  {:>11.1}%  {:>12.4}",
                budget_pct, retained, reduction, delta
            );
        }

        // MCC against ground-truth task-0 segments at full budget.
        let selector = BandConditionerSelector::new(
            BandConditionerSelectorConfig::default().with_segment_len(seg_len),
        );
        let selected = selector.select(&segments, &query, n);
        let mut y_pred = vec![0.0_f32; n];
        for &i in &selected {
            y_pred[i] = 1.0;
        }
        let y_true = scm.ground_truth_relevance(seg_len, 0);
        let mcc = matthews_corr(&y_true, &y_pred);
        println!("\nSelection MCC vs. ground-truth task-0 segments: {mcc:.4}");

        println!("\nDone.");
    }

    #[cfg(not(feature = "bckvss"))]
    println!("Enable feature: cargo run --example bckvss_vs_dense --features bckvss");
}
