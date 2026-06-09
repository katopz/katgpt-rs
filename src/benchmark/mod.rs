//! Benchmark framework for KatGPT-RS.
//!
//! Organized into sub-modules mapped to the 10 feature dimensions from the
//! Paper Feature Comparison Matrix (`.docs/15_paper_feature_comparison.md`):
//!
//! | Dimension | Module | Description |
//! |-----------|--------|-------------|
//! | SD  | `speculative` | AR, DFlash, DDTree, speculative decoding, Leviathan variants |
//! | KV  | `infrastructure` | KV cache, prefill, Raven, TurboQuant, SpectralQuant, PFlash, MaxSim |
//! | Attn | `hla` | HLA/AHLA attention, SIMD micro-benchmarks |
//! | Noise | `noise` | ELF SDE noise scheduling benchmarks |
//! | Distill | `distillation` | BT ranking, BanditPruner, AbsorbCompress |
//! | TTC | `ttc` | Test-Time Compute — UCB1/Thompson exploration cycles, BanditPruner episodes |
//! | Route | `routing` | Raven slot routing, update, readout, delta routing |
//! | Diff | `diffusion` | D2F block decode, pipeline, confidence thresholding |
//! | Game | `heuristic`, `games` | G-Zero self-play, E2E game timing (plasma/hot/warm/cold) |
//! | SIMD | `simd` | Dense/sparse matmul, forward pass, PlasmaPath, Minkowski lattice |
//!
//! Legacy modules: `asymmetric`, `sparse`, `batch`

mod asymmetric;
mod batch;
#[cfg(feature = "dllm")]
mod diffusion;
mod distillation;
mod games;
mod heuristic;
#[cfg(feature = "hla_attention")]
mod hla;
mod infrastructure;
#[cfg(feature = "llmexec_guard")]
mod llmexec_guard;
mod noise;
mod routing;
mod simd;
mod sparse;
mod speculative;
mod ttc;

pub use asymmetric::cosine_similarity;
#[cfg(feature = "asymmetric_kv")]
pub use asymmetric::{AsymmetricBenchResult, bench_asymmetric_cross_method};
pub use batch::generate_batch;
pub use games::bench_e2e_game_timing;

pub use heuristic::bench_g_zero;
#[cfg(feature = "hla_attention")]
pub use hla::{bench_hla_memory, bench_hla_quality, bench_hla_vs_flat_cache, bench_simd};
#[cfg(feature = "spectral_quant")]
pub use infrastructure::bench_spectralquant_par_dequant;
#[cfg(feature = "turboquant")]
pub use infrastructure::bench_turboquant_store_dequant;
#[cfg(feature = "maxsim")]
pub use infrastructure::{bench_maxsim_score, bench_pflash_maxsim_block_scoring};
pub use infrastructure::{
    bench_pflash_block_select, bench_prefill_compression, bench_raven_recall,
    bench_raven_vs_flat_cache,
};
pub use noise::bench_elf_sde;
#[cfg(feature = "sparse_mlp")]
pub use sparse::bench_sparse_mlp;
pub use speculative::{
    bench_ar, bench_ddtree_budget_sweep, bench_ddtree_chain_seed, bench_ddtree_screened,
    bench_dflash, bench_dflash_parallel, bench_speculative, bench_speculative_ar,
};
#[cfg(feature = "domino_lora")]
pub use speculative::{bench_dflash_ar_domino_vs_baseline, bench_domino_lora_correction};

use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};

/// Benchmark category for grouping into separate graphs.
///
/// Maps to the 10 feature dimensions from the Paper Feature Comparison Matrix
/// (see `.docs/15_paper_feature_comparison.md`), plus legacy categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum BenchCategory {
    // ── Legacy categories (backward compat) ──
    /// Speculative decoding: accepted tok/s, μs/step
    Speculative,
    /// Tree/draft operations: builds/s or steps/s, μs/step
    TreeBuild,
    /// Infrastructure: KV cache, prefill, recall — steps/s, μs/step
    #[default]
    Infrastructure,
    /// G-Zero self-play components: Hint-δ, TemplateProposer, Δ-Absorb, Δ-Bandit
    HeuristicLearning,

    // ── Feature-dimension categories (paper matrix) ──
    /// SD: Speculative Decoding — draft/verify, tree search, multi-token prediction
    SpecDecoding,
    /// KV: KV Optimization — cache compression, pruning, quantization, paged attention
    KvOptimization,
    /// Attn: Attention Innovation — novel attention mechanisms, linear attention, hull queries
    Attention,
    /// Noise: Noise / Noise Scheduling — SDE injection, diffusion schedules, perturbation
    Noise,
    /// Distill: Distillation / Compression — LoRA, quantization, knowledge transfer, pruning
    Distillation,
    /// TTC: Test-Time Compute — adaptive budget, self-improvement, recursive refinement
    TestTimeCompute,
    /// Route: Routing / MoE — expert selection, domain routing, mixture-of-experts
    Routing,
    /// Diff: Diffusion / Denoising — discrete diffusion, block-parallel, flow matching
    Diffusion,
    /// Game: Game / Self-Play — puzzles, board games, RL arenas, heuristic learning
    Game,
    /// SIMD: SIMD / Perf — hardware acceleration, zero-alloc, GPU compute, kernels
    SimdPerf,

    // ── E2E game timing ──
    /// E2E game timing through plasma/hot/warm/cold cache states
    E2EGame,
}

/// All feature-dimension categories for grouped plotting.
pub const FEATURE_DIMS: [BenchCategory; 10] = [
    BenchCategory::SpecDecoding,
    BenchCategory::KvOptimization,
    BenchCategory::Attention,
    BenchCategory::Noise,
    BenchCategory::Distillation,
    BenchCategory::TestTimeCompute,
    BenchCategory::Routing,
    BenchCategory::Diffusion,
    BenchCategory::Game,
    BenchCategory::SimdPerf,
];

/// Short string for a benchmark category (used in filenames).
pub fn bench_category_str(cat: BenchCategory) -> &'static str {
    match cat {
        BenchCategory::Speculative => "speculative",
        BenchCategory::TreeBuild => "tree_build",
        BenchCategory::Infrastructure => "infrastructure",
        BenchCategory::HeuristicLearning => "heuristic_learning",
        BenchCategory::SpecDecoding => "SD",
        BenchCategory::KvOptimization => "KV",
        BenchCategory::Attention => "Attn",
        BenchCategory::Noise => "Noise",
        BenchCategory::Distillation => "Distill",
        BenchCategory::TestTimeCompute => "TTC",
        BenchCategory::Routing => "Route",
        BenchCategory::Diffusion => "Diff",
        BenchCategory::Game => "Game",
        BenchCategory::SimdPerf => "SIMD",
        BenchCategory::E2EGame => "e2e_game",
    }
}

/// Human-readable title for a benchmark category.
pub fn bench_category_title(cat: BenchCategory) -> &'static str {
    match cat {
        BenchCategory::Speculative => "Speculative Decoding Throughput",
        BenchCategory::TreeBuild => "DDTree Build Performance",
        BenchCategory::Infrastructure => "Infrastructure Primitives",
        BenchCategory::HeuristicLearning => "G-Zero Heuristic Learning",
        BenchCategory::SpecDecoding => "Speculative Decoding (SD)",
        BenchCategory::KvOptimization => "KV Optimization",
        BenchCategory::Attention => "Attention Innovation",
        BenchCategory::Noise => "Noise / SDE Scheduling",
        BenchCategory::Distillation => "Distillation / Compression",
        BenchCategory::TestTimeCompute => "Test-Time Compute",
        BenchCategory::Routing => "Routing / MoE",
        BenchCategory::Diffusion => "Diffusion / Denoising",
        BenchCategory::Game => "Game / Self-Play",
        BenchCategory::SimdPerf => "SIMD / Perf",
        BenchCategory::E2EGame => "E2E Game Timing (Plasma/Hot/Warm/Cold)",
    }
}

/// Single benchmark result.
#[derive(Clone, Default)]
pub struct BenchResult {
    pub label: String,
    /// Feature dimension tag (e.g. "SD", "KV", "Attn") for grouped plotting.
    /// Maps to the 10 feature dimensions from the Paper Feature Comparison Matrix.
    pub feature_dim: String,
    pub throughput: f64,
    pub time_per_step_us: f64,
    pub avg_acceptance_len: f64,
    pub category: BenchCategory,
    pub color: (u8, u8, u8),
}

// ---------------------------------------------------------------------------
// EP Accuracy@k — convergence speed metric (Plan 104)
// ---------------------------------------------------------------------------

/// Compute EP Accuracy@k: number of rounds to first reach target_accuracy.
/// Returns `None` if target was never reached within the data.
///
/// EP = Episode Precision: the episode index where accuracy first crosses the
/// target threshold. Lower is better (faster convergence).
///
/// Use: `ep_accuracy_k(&accuracies, 0.8)` → first episode where accuracy ≥ 80%.
pub fn ep_accuracy_k(accuracies: &[f32], target: f32) -> Option<usize> {
    accuracies.iter().position(|&a| a >= target)
}

/// Save benchmark results to numbered CSV (e.g. `bench/024_results.csv`).
///
/// CSV columns: `commit,date,method,throughput,us_per_step,avg_accept_len`
/// One file per run, numbered to match the SVG chart. Always writes header + data.
pub fn save_results_csv(results: &[BenchResult], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let date = chrono_like_now();
    let features = active_features();

    let mut file = std::fs::File::create(path)?;

    writeln!(
        file,
        "commit,date,features,method,throughput,us_per_step,avg_accept_len,feature_dim"
    )?;

    for r in results {
        let label = if r.label.contains(',') || r.label.contains('"') {
            format!("\"{}\"", r.label.replace('"', "\"\""))
        } else {
            r.label.clone()
        };
        writeln!(
            file,
            "{},{},{},{},{:.0},{:.2},{:.2},{}",
            commit,
            date,
            features,
            label,
            r.throughput,
            r.time_per_step_us,
            r.avg_acceptance_len,
            r.feature_dim,
        )?;
    }

    Ok(())
}

/// Append benchmark results to cumulative `bench/timeseries.csv` for regression tracking.
/// Creates file with header if missing; otherwise appends rows.
pub fn append_timeseries_csv(results: &[BenchResult], path: &str) -> std::io::Result<()> {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    let date = chrono_like_now();
    let features = active_features();

    let file_exists = std::path::Path::new(path).exists();

    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    if !file_exists {
        writeln!(
            file,
            "run_date,commit,features,category,method,throughput,us_per_step,avg_accept_len,feature_dim"
        )?;
    }

    for r in results {
        let cat = bench_category_str(r.category);
        let label = if r.label.contains(',') || r.label.contains('"') {
            format!("\"{}\"", r.label.replace('"', "\"\""))
        } else {
            r.label.clone()
        };
        writeln!(
            file,
            "{},{},{},{},{},{:.0},{:.2},{:.2},{}",
            date,
            commit,
            features,
            cat,
            label,
            r.throughput,
            r.time_per_step_us,
            r.avg_acceptance_len,
            r.feature_dim,
        )?;
    }

    Ok(())
}

/// Simple timestamp without chrono dependency: `YYYY-MM-DDTHH:MM:SS`
fn chrono_like_now() -> String {
    let output = std::process::Command::new("date")
        .arg("+%Y-%m-%dT%H:%M:%S")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    output.unwrap_or_else(|| "unknown".into())
}

/// Collect active feature flags that affect forward-path performance.
/// Appended to CSV for regression tracking across feature-gate changes.
fn active_features() -> String {
    let mut flags = Vec::new();
    if cfg!(feature = "sparse_mlp") {
        flags.push("sparse_mlp");
    }
    if cfg!(feature = "domain_latent") {
        flags.push("domain_latent");
    }
    if cfg!(feature = "ppot") {
        flags.push("ppot");
    }
    if cfg!(feature = "bandit") {
        flags.push("bandit");
    }
    if cfg!(feature = "g_zero") {
        flags.push("g_zero");
    }
    if cfg!(feature = "delta_mem") {
        flags.push("delta_mem");
    }
    if cfg!(feature = "fft") {
        flags.push("fft");
    }
    if cfg!(feature = "stepcode") {
        flags.push("stepcode");
    }
    if cfg!(feature = "bomber") {
        flags.push("bomber");
    }
    if cfg!(feature = "bt_rank") {
        flags.push("bt_rank");
    }
    if cfg!(feature = "spectral_quant") {
        flags.push("spectral_quant");
    }
    if cfg!(feature = "hybrid_oct_pq") {
        flags.push("hybrid_oct_pq");
    }
    if cfg!(feature = "elf_sde") {
        flags.push("elf_sde");
    }
    if cfg!(feature = "cna_steering") {
        flags.push("cna_steering");
    }
    if cfg!(feature = "deep_manifold") {
        flags.push("deep_manifold");
    }
    if cfg!(feature = "tes_loop") {
        flags.push("tes_loop");
    }
    if cfg!(feature = "lattice_deduction") {
        flags.push("lattice_deduction");
    }
    if cfg!(feature = "delta_routing") {
        flags.push("delta_routing");
    }
    if cfg!(feature = "gdn2_attention") {
        flags.push("gdn2_attention");
    }
    if cfg!(feature = "dash_attn") {
        flags.push("dash_attn");
    }
    if cfg!(feature = "dreamer") {
        flags.push("dreamer");
    }
    if cfg!(feature = "lt2_looped") {
        flags.push("lt2_looped");
    }
    if cfg!(feature = "plasma_path") {
        flags.push("plasma_path");
    }
    if cfg!(feature = "tf_loop") {
        flags.push("tf_loop");
    }
    if flags.is_empty() {
        flags.push("(none)");
    }
    flags.join("+")
}

/// Cooldown pause between benchmark groups to reduce thermal throttling noise.
fn cooldown(_secs: u64) {
    // Cooldowns disabled for fast benchmark runs
}

/// Run all benchmarks and return results.
///
/// Order: KV → SD → Attn → Game → Noise → Distill → TTC → Route → Diff → SIMD → E2E games.
/// Inter-group cooldowns (3s) reduce thermal throttling noise on sustained runs.
/// Every result is tagged with a `feature_dim` matching the Paper Feature Comparison Matrix.
pub fn run_all(config: &Config) -> Vec<BenchResult> {
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);

    let draft_config = Config::draft();
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

    let warmup = 50;
    let iters = 2_000;

    println!("\n📊 Running benchmarks ({iters} iterations, {warmup} warmup)...");
    println!(
        "   Target model: embd={}, heads={}, mlp={}",
        config.n_embd, config.n_head, config.mlp_hidden
    );
    println!(
        "   Draft  model: embd={}, heads={}, mlp={}",
        draft_config.n_embd, draft_config.n_head, draft_config.mlp_hidden
    );
    println!("   Features: {}", active_features());

    #[allow(unused_mut)]
    let mut results = Vec::new();

    // ── Phase 1: Infrastructure / KV Optimization (cool CPU) ──
    let (flat_br, paged_br) = infrastructure::bench_paged_vs_flat_cache(config);
    let (_, raven_br) = infrastructure::bench_raven_vs_flat_cache(config);
    let recall_br = infrastructure::bench_raven_recall(config);
    #[cfg(feature = "turboquant")]
    let (tq_alloc_br, tq_zero_br) = infrastructure::bench_turboquant_store_dequant(config);
    #[cfg(not(feature = "turboquant"))]
    let (tq_alloc_br, tq_zero_br) = (BenchResult::default(), BenchResult::default());
    let pflash_br = infrastructure::bench_pflash_block_select();
    let (nocompress_br, compress_br) =
        infrastructure::bench_prefill_compression(&draft_weights, &draft_config, warmup, iters);

    // Tag infrastructure results with feature dims
    for mut r in [
        flat_br,
        paged_br,
        raven_br,
        recall_br,
        tq_alloc_br,
        tq_zero_br,
        pflash_br,
    ] {
        r.feature_dim = "KV".into();
        r.category = BenchCategory::KvOptimization;
        results.push(r);
    }
    for mut r in [nocompress_br, compress_br] {
        r.feature_dim = "KV".into();
        r.category = BenchCategory::KvOptimization;
        results.push(r);
    }
    cooldown(3);

    // ── Phase 2: Speculative Decoding (SD) ──
    let mut ar = speculative::bench_ar(&weights, config, warmup, iters);
    ar.feature_dim = "SD".into();
    ar.category = BenchCategory::SpecDecoding;
    results.push(ar);

    let mut dflash = speculative::bench_dflash(&draft_weights, &draft_config, warmup, iters);
    dflash.feature_dim = "SD".into();
    dflash.category = BenchCategory::SpecDecoding;
    results.push(dflash);

    let mut ddtree = speculative::bench_ddtree(&draft_weights, &draft_config, warmup, iters);
    ddtree.feature_dim = "SD".into();
    ddtree.category = BenchCategory::SpecDecoding;
    results.push(ddtree);

    let mut spec = speculative::bench_speculative(&draft_weights, &draft_config, warmup, iters);
    spec.feature_dim = "SD".into();
    spec.category = BenchCategory::SpecDecoding;
    results.push(spec);

    let mut spec_ar =
        speculative::bench_speculative_ar(&draft_weights, &draft_config, warmup, iters);
    spec_ar.feature_dim = "SD".into();
    spec_ar.category = BenchCategory::SpecDecoding;
    results.push(spec_ar);
    cooldown(3);

    // ── Phase 3: Leviathan variants (SD) ──
    {
        let mut leviathan = speculative::bench_leviathan(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        leviathan.feature_dim = "SD".into();
        leviathan.category = BenchCategory::SpecDecoding;
        results.push(leviathan);

        let (no_rollback, with_rollback) = speculative::bench_snapshot_rollback(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        for mut r in [no_rollback, with_rollback] {
            r.feature_dim = "SD".into();
            r.category = BenchCategory::SpecDecoding;
            results.push(r);
        }

        let (uncond_br, cond_br) = speculative::bench_conditioned_vs_unconditioned(
            &draft_weights,
            &draft_config,
            &weights,
            config,
            warmup,
            iters,
        );
        for mut r in [uncond_br, cond_br] {
            r.feature_dim = "SD".into();
            r.category = BenchCategory::SpecDecoding;
            results.push(r);
        }

        let (mtp_off, mtp_on) = speculative::bench_mtp_leviathan(warmup, iters);
        for mut r in [mtp_off, mtp_on] {
            r.feature_dim = "SD".into();
            r.category = BenchCategory::SpecDecoding;
            results.push(r);
        }

        let (mtp_shared_off, mtp_shared_on) = speculative::bench_mtp_shared_kv(warmup, iters);
        for mut r in [mtp_shared_off, mtp_shared_on] {
            r.feature_dim = "SD".into();
            r.category = BenchCategory::SpecDecoding;
            results.push(r);
        }
    }
    cooldown(3);

    // ── Phase 4: Tree variants (SD) ──
    let (no_chain, chain) =
        speculative::bench_ddtree_chain_seed(&draft_weights, &draft_config, warmup, iters);
    for mut r in [no_chain, chain] {
        r.feature_dim = "SD".into();
        r.category = BenchCategory::SpecDecoding;
        results.push(r);
    }

    let (screened_noop, screened_adapter) =
        speculative::bench_ddtree_screened(&draft_weights, &draft_config, warmup, iters);
    for mut r in [screened_noop, screened_adapter] {
        r.feature_dim = "SD".into();
        r.category = BenchCategory::SpecDecoding;
        results.push(r);
    }
    cooldown(3);

    // ── Phase 4.5: DFlash parallel proof (SD) ──
    {
        let (_df_seq, mut df_par) =
            speculative::bench_dflash_parallel(&draft_weights, &draft_config, warmup, iters);
        df_par.feature_dim = "SD".into();
        df_par.category = BenchCategory::SpecDecoding;
        results.push(df_par);
    }

    // ── Phase 5: SpectralQuant parallel dequant (KV) ──
    #[cfg(feature = "spectral_quant")]
    {
        let (sq_seq, sq_par) = infrastructure::bench_spectralquant_par_dequant(config);
        for mut r in [sq_seq, sq_par] {
            r.feature_dim = "KV".into();
            r.category = BenchCategory::KvOptimization;
            results.push(r);
        }
    }

    // ── Phase 5.5: MaxSim benchmarks (Attn + SIMD) ──
    #[cfg(feature = "maxsim")]
    {
        let maxsim_results = infrastructure::bench_maxsim_score();
        for mut r in maxsim_results {
            r.feature_dim = "Attn".into();
            r.category = BenchCategory::Attention;
            results.push(r);
        }
        let mut pflash_ms = infrastructure::bench_pflash_maxsim_block_scoring();
        pflash_ms.feature_dim = "Attn".into();
        pflash_ms.category = BenchCategory::Attention;
        results.push(pflash_ms);
    }

    cooldown(3);

    // ── Phase 6: Heuristic learning / TTC + Game (feature-gated) ──
    #[cfg(feature = "g_zero")]
    {
        let gz_results = heuristic::bench_g_zero();
        for mut r in gz_results {
            r.feature_dim = "Game".into();
            r.category = BenchCategory::Game;
            results.push(r);
        }
    }

    // ── Phase 7: HLA attention (Attn) ──
    #[cfg(feature = "hla_attention")]
    {
        let mut hla_br = hla::bench_hla_vs_flat_cache(config);
        hla_br.feature_dim = "Attn".into();
        hla_br.category = BenchCategory::Attention;
        results.push(hla_br);

        let mut hla_mem_br = hla::bench_hla_memory(config);
        hla_mem_br.feature_dim = "Attn".into();
        hla_mem_br.category = BenchCategory::Attention;
        results.push(hla_mem_br);

        let mut hla_quality_br = hla::bench_hla_quality(config);
        hla_quality_br.feature_dim = "Attn".into();
        hla_quality_br.category = BenchCategory::Attention;
        results.push(hla_quality_br);
    }

    // ── Phase 8: Noise / SDE scheduling ──
    let noise_results = noise::bench_elf_sde(config);
    results.extend(noise_results);
    cooldown(3);

    // ── Phase 9: Distillation / Compression ──
    let distill_results = distillation::bench_distillation();
    results.extend(distill_results);
    cooldown(3);

    // ── Phase 10: Test-Time Compute (TTC) ──
    #[cfg(feature = "bandit")]
    {
        let ttc_results = ttc::bench_ttc();
        results.extend(ttc_results);
    }
    cooldown(3);

    // ── Phase 11: Routing / MoE ──
    let routing_results = routing::bench_routing(config);
    results.extend(routing_results);
    cooldown(3);

    // ── Phase 12: Diffusion / Denoising ──
    #[cfg(feature = "dllm")]
    {
        let diff_results = diffusion::bench_diffusion();
        results.extend(diff_results);
        cooldown(3);
    }

    // ── Phase 13: SIMD / Perf ──
    let simd_results = simd::bench_simd_perf();
    results.extend(simd_results);
    cooldown(3);

    // ── Phase 14: E2E Game timing (plasma/hot/warm/cold) ──
    let game_results = games::bench_e2e_game_timing(config);
    results.extend(game_results);

    results
}

/// Run all core benchmarks in parallel using rayon's `par_iter`.
/// Run all benchmarks in parallel using rayon.
///
/// Groups independent phases and runs them concurrently via rayon.
/// New modules (Distill, TTC, Route, Diff, SIMD) run in parallel.
pub fn run_all_parallel(config: &Config) -> Vec<BenchResult> {
    // Just delegate to run_all which now has reduced iterations + no cooldowns.
    // Full parallel execution of phase groups is a future enhancement.
    run_all(config)
}
