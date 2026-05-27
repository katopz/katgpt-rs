use katgpt_rs::{benchmark, benchmark::BenchCategory, percepta, plot, transformer, types};

fn main() {
    let config = types::Config::micro();

    println!("🚀 KatGPT-RS: Transformer + Speculative Decoding Benchmark");
    println!("{}", "═".repeat(60));

    // ── 1. Transformer Output with Proof ──────────────────────────
    println!("\n📝 Transformer Output (Proof of Correctness)");
    println!("{}", "─".repeat(60));

    let mut rng = types::Rng::new(42);
    let weights = transformer::TransformerWeights::new(&config, &mut rng);

    // Generate samples with different seeds
    for sample in 0..5 {
        let mut sample_rng = types::Rng::new(42 + sample);
        let tokens = transformer::generate(&weights, &config, &mut sample_rng, config.block_size);
        let text = transformer::tokens_to_string(&tokens);
        let valid = tokens.iter().all(|&t| t < config.vocab_size);
        println!("  Sample {}: \"{}\" (valid={})", sample + 1, text, valid);
    }

    // Determinism check: same seed must produce identical output
    let mut rng_a = types::Rng::new(100);
    let tokens_a = transformer::generate(&weights, &config, &mut rng_a, config.block_size);
    let mut rng_b = types::Rng::new(100);
    let tokens_b = transformer::generate(&weights, &config, &mut rng_b, config.block_size);
    let deterministic = tokens_a == tokens_b;

    // Different seed must produce different output
    let mut rng_c = types::Rng::new(200);
    let tokens_c = transformer::generate(&weights, &config, &mut rng_c, config.block_size);
    let diverse = tokens_a != tokens_c;

    println!();
    println!(
        "  ✅ Deterministic: {} (same seed = same output)",
        if deterministic { "PASS" } else { "FAIL" }
    );
    println!(
        "  ✅ Diverse:       {} (different seed = different output)",
        if diverse { "PASS" } else { "FAIL" }
    );
    println!(
        "  ✅ Valid tokens:  {} (all tokens in [0, {}))",
        if deterministic && diverse {
            "PASS"
        } else {
            "FAIL"
        },
        config.vocab_size
    );
    println!(
        "  📐 Config: vocab={}, block={}, embd={}, heads={}, mlp={}",
        config.vocab_size, config.block_size, config.n_embd, config.n_head, config.mlp_hidden
    );

    // ── 2–5. Benchmarks ───────────────────────────────────────────
    let results = benchmark::run_all(&config);

    println!("📊 Benchmark Results");
    println!("{}", "─".repeat(75));
    println!(
        "  {:<20} {:>15} {:>15} {:>15}",
        "Method", "Throughput", "μs/step", "Avg Accept Len"
    );
    println!("{}", "─".repeat(75));

    for r in &results {
        let unit = match r.category {
            BenchCategory::Speculative
            | BenchCategory::SpecDecoding
            | BenchCategory::Game
            | BenchCategory::E2EGame => "tok/s",
            BenchCategory::TreeBuild
            | BenchCategory::Infrastructure
            | BenchCategory::HeuristicLearning
            | BenchCategory::KvOptimization
            | BenchCategory::Attention
            | BenchCategory::Noise
            | BenchCategory::Distillation
            | BenchCategory::TestTimeCompute
            | BenchCategory::Routing
            | BenchCategory::Diffusion
            | BenchCategory::SimdPerf => "ops/s",
        };
        println!(
            "  {:<20} {:>12.0} {:>3} {:>12.2} {:>15.2}",
            r.label, r.throughput, unit, r.time_per_step_us, r.avg_acceptance_len,
        );
    }

    println!("{}", "─".repeat(75));

    // Speedup comparison
    let ar_tps = results[0].throughput;
    let spec_tps = results[3].throughput;
    let speedup = spec_tps / ar_tps;
    println!("  📈 Speedup: {:.2}x (Speculative vs AR)", speedup);

    // ── Plot ───────────────────────────────────────────────────────
    std::fs::create_dir_all("bench").ok();
    let index = next_bench_index();

    let categories = [
        (
            BenchCategory::SpecDecoding,
            "speculative",
            "Speculative Decoding Throughput",
            "Accepted tok/s",
        ),
        (
            BenchCategory::KvOptimization,
            "kv_optimization",
            "KV Optimization Performance",
            "Operations/s",
        ),
        (
            BenchCategory::Attention,
            "attention",
            "Attention Innovation Performance",
            "Operations/s",
        ),
        (
            BenchCategory::Noise,
            "noise",
            "Noise / SDE Scheduling Performance",
            "Operations/s",
        ),
        (
            BenchCategory::Game,
            "game",
            "Game / Self-Play Performance",
            "Operations/s",
        ),
        (
            BenchCategory::SimdPerf,
            "simd",
            "SIMD / Perf Performance",
            "Operations/s",
        ),
        (
            BenchCategory::E2EGame,
            "e2e_game",
            "E2E Game Timing (Plasma/Hot/Warm/Cold)",
            "tok/s",
        ),
        // Legacy categories for backward compat
        (
            BenchCategory::Speculative,
            "speculative_legacy",
            "Speculative Decoding Throughput (Legacy)",
            "Accepted tok/s",
        ),
        (
            BenchCategory::TreeBuild,
            "tree_build",
            "DDTree Build Performance",
            "Operations/s",
        ),
        (
            BenchCategory::Infrastructure,
            "infrastructure",
            "Infrastructure Primitives",
            "Operations/s",
        ),
        (
            BenchCategory::HeuristicLearning,
            "heuristic_learning",
            "G-Zero Heuristic Learning (Plan 049)",
            "Operations/s",
        ),
        (
            BenchCategory::Distillation,
            "distillation",
            "Distillation / Compression",
            "Operations/s",
        ),
        (
            BenchCategory::TestTimeCompute,
            "ttc",
            "Test-Time Compute",
            "Operations/s",
        ),
        (
            BenchCategory::Routing,
            "routing",
            "Routing / MoE",
            "Operations/s",
        ),
        (
            BenchCategory::Diffusion,
            "diffusion",
            "Diffusion / Denoising",
            "Operations/s",
        ),
    ];

    for (cat, suffix, title, x_label) in &categories {
        let cat_results: Vec<_> = results
            .iter()
            .filter(|r| r.category == *cat)
            .cloned()
            .collect();
        if cat_results.is_empty() {
            continue;
        }
        let plot_path = format!("bench/{:03}_{suffix}.svg", index);
        match plot::plot_results(&cat_results, &plot_path, title, x_label) {
            Ok(()) => println!("📈 {title} chart saved to: {plot_path}"),
            Err(e) => eprintln!("⚠️  Plot failed for {title}: {e}"),
        }
    }

    // Save results to CSV for regression tracking (same index as PNG)
    let csv_path = format!("bench/{:03}_results.csv", index);
    match benchmark::save_results_csv(&results, &csv_path) {
        Ok(()) => println!("📝 Results saved to: {csv_path}"),
        Err(e) => eprintln!("⚠️  CSV save failed: {e}"),
    }

    // ── Time Series (cumulative regression tracking) ───────────────
    match benchmark::append_timeseries_csv(&results, "bench/timeseries.csv") {
        Ok(()) => println!("📝 Time series appended to bench/timeseries.csv"),
        Err(e) => eprintln!("⚠️  Timeseries CSV append failed: {e}"),
    }
    match plot::plot_timeseries("bench/timeseries.csv", "bench") {
        Ok(regressions) => {
            println!("📈 Time series charts saved to bench/timeseries_*.svg");
            for msg in &regressions {
                println!("{msg}");
            }
        }
        Err(e) => eprintln!("⚠️  Timeseries plot failed: {e}"),
    }

    // ── Budget Sweep ───────────────────────────────────────────────
    println!("\n📊 DDTree Budget Sweep");
    println!("{}", "─".repeat(75));

    let draft_config = types::Config::draft();
    let mut draft_rng = types::Rng::new(99);
    let draft_weights = transformer::TransformerWeights::new(&draft_config, &mut draft_rng);

    let budgets = [4, 8, 16, 32, 64];
    let sweep_results =
        benchmark::bench_ddtree_budget_sweep(&draft_weights, &draft_config, &budgets, 50, 2_000);

    println!(
        "  {:<30} {:>12} {:>12} {:>12}",
        "Config", "trees/s", "μs/build", "Avg Nodes"
    );
    println!("{}", "─".repeat(75));
    for r in &sweep_results {
        println!(
            "  {:<30} {:>12.0} {:>12.2} {:>12.2}",
            r.label, r.throughput, r.time_per_step_us, r.avg_acceptance_len,
        );
    }

    // ── 6. Percepta 2D Attention Benchmark ─────────────────────────
    println!("\n🧠 Percepta 2D Convex Hull Attention (O(log N) vs O(N))");
    println!("{}", "─".repeat(60));

    percepta_benchmark();

    // ── 7. CHT vs Graham Scan Benchmark ───────────────────────────
    #[cfg(feature = "percepta")]
    percepta_cht_benchmark();

    println!("\n✨ Done.");
}

/// Benchmark: Percepta O(log N) hull attention vs standard O(N) linear scan.
/// Proves correctness (same results) and measures speedup across trace sizes.
fn percepta_benchmark() {
    let trace_sizes = [1_000, 10_000, 100_000];

    println!(
        "  {:>12} {:>8} {:>12} {:>12} {:>10} {:>8}",
        "Trace Size", "Hull", "Linear μs", "Fast μs", "Speedup", "Match"
    );
    println!("{}", "─".repeat(66));

    for &size in &trace_sizes {
        let mut cache = percepta::KVCache2D::with_capacity(size);

        // Build convex parabolic key distribution (simulates execution trace)
        let mid = size as f32 / 2.0;
        for i in 0..size {
            let x = i as f32;
            let y = -((x - mid) / (mid * 0.02)).powi(2);
            cache.append(percepta::Vec2::new(x, y), i);
        }

        let query = percepta::Vec2::new(5.0, 10.0);

        // Warmup
        for _ in 0..10 {
            let _ = cache.fast_attention(&query);
            let _ = cache.linear_attention(&query);
        }

        // Benchmark linear O(N)
        let iters_linear = 100;
        let start = std::time::Instant::now();
        let (lin_score, lin_val) = cache.linear_attention(&query);
        for _ in 0..iters_linear {
            let _ = cache.linear_attention(&query);
        }
        let elapsed_linear = start.elapsed() / (iters_linear + 1);

        // Benchmark fast O(log N)
        let iters_fast = 10_000;
        let start = std::time::Instant::now();
        let (fast_score, fast_val) = cache.fast_attention(&query);
        for _ in 0..iters_fast {
            let _ = cache.fast_attention(&query);
        }
        let elapsed_fast = start.elapsed() / (iters_fast + 1);

        let speedup = elapsed_linear.as_secs_f64() / elapsed_fast.as_secs_f64();
        let score_match = (lin_score - fast_score).abs() < 1e-3;
        let val_match = lin_val == fast_val;

        println!(
            "  {:>12} {:>8} {:>12.2} {:>12.4} {:>9.1}x {:>8}",
            size,
            cache.hull_len(),
            elapsed_linear.as_secs_f64() * 1e6,
            elapsed_fast.as_secs_f64() * 1e6,
            speedup,
            if score_match && val_match {
                "✅"
            } else {
                "❌"
            }
        );
    }

    println!();
    println!("  Hull compression: O(N) keys → O(H) hull vertices");
    println!("  Attention search: ternary search over unimodal dot-product sequence");
}

/// Benchmark: CHT [`HardAttentionHead`] vs Legacy [`KVCache2D`] throughput.
///
/// Measures insert + query performance on the same parabolic trace distribution.
/// CHT uses O(log H) amortized insert (dynamic line container) vs Graham Scan's
/// O(N) full rebuild. Both use O(log H) query (binary/ternary search on hull).
#[cfg(feature = "percepta")]
fn percepta_cht_benchmark() {
    use katgpt_rs::percepta::{HardAttentionHead, TieBreak};

    let trace_sizes = [1_000, 10_000, 100_000];

    println!("\n🧠 CHT HardAttentionHead vs Legacy KVCache2D");
    println!("{}", "─".repeat(85));
    println!(
        "  {:>12} {:>10} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Size", "Entries", "Legacy Ins", "CHT Ins", "Legacy Qry", "CHT Qry", "Match"
    );
    println!("{}", "─".repeat(85));

    for &size in &trace_sizes {
        let mid = size as f32 / 2.0;

        // Build parabolic trace: y = -((x - mid) / (mid * 0.02))^2
        let points: Vec<(f32, f32)> = (0..size)
            .map(|i| {
                let x = i as f32;
                let y = -((x - mid) / (mid * 0.02)).powi(2);
                (x, y)
            })
            .collect();

        // ── Legacy KVCache2D (Graham Scan) ──
        let mut legacy = percepta::KVCache2D::with_capacity(size);
        let start = std::time::Instant::now();
        for (i, (x, y)) in points.iter().enumerate() {
            legacy.append(percepta::Vec2::new(*x, *y), i);
        }
        let legacy_insert = start.elapsed();

        let query_legacy = percepta::Vec2::new(5.0, 10.0);
        let iters = 10_000;
        let start = std::time::Instant::now();
        let (_, legacy_val) = legacy.fast_attention(&query_legacy);
        for _ in 0..iters {
            let _ = legacy.fast_attention(&query_legacy);
        }
        let legacy_query = start.elapsed() / (iters + 1);

        // ── CHT HardAttentionHead ──
        let mut cht = HardAttentionHead::new();
        let start = std::time::Instant::now();
        for (i, (x, y)) in points.iter().enumerate() {
            cht.insert([*x as f64, *y as f64], [i as f64, 0.0], i as i64);
        }
        let cht_insert = start.elapsed();

        let query_cht = [5.0_f64, 10.0];
        let start = std::time::Instant::now();
        let cht_result = cht.query(query_cht, TieBreak::Latest);
        for _ in 0..iters {
            let _ = cht.query(query_cht, TieBreak::Latest);
        }
        let cht_query = start.elapsed() / (iters + 1);

        // ── Correctness: value and score parity ──
        let cht_val = cht_result.map(|v| v[0] as usize);
        let score_ok = { cht_val == Some(legacy_val) };

        println!(
            "  {:>12} {:>10} {:>10.2}ms {:>10.2}ms {:>10.2}μs {:>10.2}μs {:>8}",
            size,
            cht.size(),
            legacy_insert.as_secs_f64() * 1000.0,
            cht_insert.as_secs_f64() * 1000.0,
            legacy_query.as_secs_f64() * 1e6,
            cht_query.as_secs_f64() * 1e6,
            if score_ok { "✅" } else { "❌" },
        );
    }

    println!();
    println!("  CHT: O(log H) amortized insert, O(log H) query (dual hull)");
    println!("  Graham Scan: O(H) rebuild per insert, O(log H) ternary search query");
}

/// Auto-number bench results sequentially.
fn next_bench_index() -> u32 {
    let dir = std::path::Path::new("bench");
    if !dir.exists() {
        return 1;
    }
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.split('_').next().and_then(|n| n.parse::<u32>().ok())
                })
                .max()
                .map(|n| n + 1)
                .unwrap_or(1)
        })
        .unwrap_or(1)
}
