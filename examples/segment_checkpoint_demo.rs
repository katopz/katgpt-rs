//! Plan 223b Example: SegmentCheckpoint Demo

fn main() {
    #[cfg(feature = "segment_checkpoint")]
    {
        use katgpt_rs::segment_checkpoint::gating::compute_gates;
        use katgpt_rs::segment_checkpoint::ssc::top_k_segments;
        use katgpt_rs::segment_checkpoint::{SegmentCheckpoint, SegmentStore};

        println!("=== Plan 223b: SegmentCheckpoint Demo ===\n");

        let mut store = SegmentStore::new(10, 128);

        // Simulate adding segment checkpoints
        for i in 0..5 {
            let summary = vec![(i as f32 * 0.2).sin(), (i as f32 * 0.3).cos()];
            let cp = SegmentCheckpoint::new(i, vec![], vec![], summary, i * 128, (i + 1) * 128 - 1);
            store.insert(cp);
        }
        println!("Inserted {} segments", store.len());

        // Query with a test vector
        let query = vec![0.5, 0.3];
        let gates = compute_gates(&query, &store.summaries());
        println!("\nGate values: {:?}", gates);

        // Top-k selection
        let top = top_k_segments(&mut store, &query, 3);
        println!("Top-3 segments: {:?}", top);

        println!("\nDone.");
    }

    #[cfg(not(feature = "segment_checkpoint"))]
    println!(
        "Enable feature: cargo run --example segment_checkpoint_demo --features segment_checkpoint"
    );
}
