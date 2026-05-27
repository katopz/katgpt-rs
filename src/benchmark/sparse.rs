#[cfg(feature = "sparse_mlp")]
pub fn bench_sparse_mlp() {
    use crate::types;

    println!("\n=== Sparse MLP Benchmark (Plan 022: TwELL-inspired) ===\n");

    let configs = [
        ("micro", 64, 16),
        ("bpe", 128, 32),
        ("small_target", 256, 64),
        ("large", 16384, 4096),
    ];

    let sparsity_levels = [0.0f32, 0.50, 0.90, 0.95, 0.99];

    let iterations = 10;

    for &(label, mlp_hidden, n_embd) in &configs {
        println!("--- Config: {label} (mlp_hidden={mlp_hidden}, n_embd={n_embd}) ---");

        let weight: Vec<f32> = (0..n_embd * mlp_hidden)
            .map(|i| (i % 100) as f32 * 0.01)
            .collect();
        let mut output_dense = vec![0.0f32; n_embd];
        let mut output_sparse = vec![0.0f32; n_embd];
        let mut active_indices = vec![0usize; mlp_hidden];
        let mut active_values = vec![0.0f32; mlp_hidden];

        for &sparsity in &sparsity_levels {
            // Build input with target sparsity
            let mut input = vec![0.0f32; mlp_hidden];
            let alive_count = ((1.0 - sparsity) * mlp_hidden as f32) as usize;
            for val in input.iter_mut().take(alive_count) {
                *val = 1.0;
            }

            // Dense benchmark
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                types::matmul(&mut output_dense, &weight, &input, n_embd, mlp_hidden);
            }
            let elapsed_dense = start.elapsed();

            // Sparse benchmark
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                types::sparse_matmul(
                    &mut output_sparse,
                    &weight,
                    &input,
                    n_embd,
                    mlp_hidden,
                    &mut active_indices,
                    &mut active_values,
                );
            }
            let elapsed_sparse = start.elapsed();

            // Verify correctness
            for i in 0..n_embd {
                let diff = (output_dense[i] - output_sparse[i]).abs();
                let d = output_dense[i];
                let s = output_sparse[i];
                assert!(diff < 1e-2, "Mismatch at {i}: dense={d}, sparse={s}");
            }

            let speedup = elapsed_dense.as_secs_f64() / elapsed_sparse.as_secs_f64();
            println!(
                "  Sparsity {:.0}%: Dense={:.2?} Sparse={:.2?} Speedup={:.1}x",
                sparsity * 100.0,
                elapsed_dense,
                elapsed_sparse,
                speedup,
            );
        }
        println!();
    }
}
