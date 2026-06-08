//! Plan 226 Example: SegmentCheckpoint Demo

fn main() {
    #[cfg(feature = "segment_checkpoint")]
    {
        use katgpt_rs::segment_checkpoint::gating::compute_gates;
        use katgpt_rs::segment_checkpoint::{SegmentCheckpoint, SegmentStore};

        println!("=== Plan 226: SegmentCheckpoint Demo ===\n");

        let mut store = SegmentStore::new(10, 128);

        // Simulate adding segment checkpoints
        for i in 0..5u32 {
            let summary = vec![(i as f32 * 0.2).sin(), (i as f32 * 0.3).cos()];
            let cp = SegmentCheckpoint::new(
                i,
                vec![],
                vec![],
                summary,
                i as usize * 128,
                (i as usize + 1) * 128 - 1,
            );
            store.insert(cp);
        }
        println!("Inserted {} segments", store.len());

        // Query with a test vector
        let query = vec![0.5, 0.3];
        let gates = compute_gates(&query, &store.summaries());
        println!("\nGate values: {:?}", gates);

        // Top-k selection via gating module
        let top_gates =
            katgpt_rs::segment_checkpoint::gating::top_k_gates(&query, &store.summaries(), 3);
        println!("Top-3 gates: {:?}", top_gates);

        #[cfg(feature = "ssc_spec_draft")]
        {
            use katgpt_rs::segment_checkpoint::ssc::{SscDrafter, compute_and_select_top_k};

            // SSC top-k via pure gate selection
            // Collect summaries separately to avoid lifetime issues with closure captures
            let ids = store.segment_ids();
            let summaries: Vec<(u32, Vec<f32>)> = ids
                .iter()
                .filter_map(|&id| store.get(id).map(|s| (id, s.summary.clone())))
                .collect();
            let summary_refs: Vec<(u32, &[f32])> = summaries
                .iter()
                .map(|(id, s)| (*id, s.as_slice()))
                .collect();
            let top = compute_and_select_top_k(&query, &summary_refs, 3);
            println!("SSC Top-3 segments: {:?}", top);

            // SSC-enhanced speculative drafting
            let mut drafter = SscDrafter::new(3);
            drafter.update_context(&query, &summary_refs);
            println!(
                "Drafter context loaded: {} summaries",
                drafter.context_len()
            );

            let mut logits = vec![0.3, 0.5, -0.1, 0.8];
            let before = logits.clone();
            drafter.enhance_draft(&mut logits);
            println!("Logits before: {:?}", before);
            println!("Logits after:  {:?}", logits);
        }

        println!("\nDone.");
    }

    #[cfg(not(feature = "segment_checkpoint"))]
    println!(
        "Enable feature: cargo run --example segment_checkpoint_demo --features segment_checkpoint"
    );
}
