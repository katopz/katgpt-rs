//! Thicket Variance Probe (TVP) — before/after thinking vs non-thinking demo.
//!
//! Plan 267, Research 235. Demonstrates the creative fusion of RandOpt
//! (Neural Thickets, arXiv:2603.12228) into a modelless routing signal.
//!
//! # What this shows
//!
//! Two synthetic query populations:
//! - **Easy** (dense thicket — high solution density): all probes agree.
//!   TVP detects low disagreement → stays on CPU, no CoT expansion.
//! - **Hard** (needle regime — low solution density): probes diverge.
//!   TVP detects high disagreement → promotes to GPU, expands CoT budget.
//!
//! # Before vs After
//!
//! | Metric                | Non-thinking baseline | TVP-routed          |
//! |-----------------------|----------------------|--------------------|
//! | Easy tokens           | 100%                 | 10-30% (skip CoT)  |
//! | Easy substrate        | mixed                | CPU only           |
//! | Hard substrate        | CPU only             | GPU promoted       |
//! | Hard accuracy         | baseline             | +2-5pp (more CoT)  |
//! | Total probes launched | 0                    | K=4 per query      |
//!
//! Run: `cargo run --example thicket_variance_probe_01_basic --features thicket_variance_probe`

use katgpt_rs::pruners::thicket_variance_probe::{
    ProbeOutput, SyntheticProbeSource, TvpAggregator, TvpConfig, TvpProbeCountBandit,
    TvpThresholdAdapter,
};

/// Synthetic tier enum — stands in for the real `ComputeTier`.
///
/// The `Cpu` prefix is intentional — each variant names a substrate stack
/// ("CPU only", "CPU+GPU", "CPU+GPU+ANE"), mirroring the real `ComputeTier`
/// taxonomy this demo substitutes for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // Cpu prefix is the documented substrate taxonomy
enum Tier {
    CpuOnly,
    CpuGpu,
    CpuGpuAne,
}

impl Tier {
    #[allow(dead_code)]
    fn name(&self) -> &'static str {
        match self {
            Tier::CpuOnly => "CPU",
            Tier::CpuGpu => "CPU+GPU",
            Tier::CpuGpuAne => "CPU+GPU+ANE",
        }
    }
}

/// Synthetic CoT mode — stands in for the real `ThinkingMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CotMode {
    Direct,
    Thinking,
}

/// A single query's routing decision.
#[derive(Clone, Debug)]
struct RoutingDecision {
    tier: Tier,
    cot_mode: CotMode,
    tokens_used: u32,
    correct: bool,
}

/// Population of synthetic queries. Each query is a closure that produces
/// probe outputs for K arms — easy queries agree, hard queries diverge.
struct QueryPopulation {
    /// Easy queries: all K probes return the same token (dense thicket).
    easy: Vec<u32>,
    /// Hard queries: probes diverge across K distinct tokens (needle).
    hard: Vec<u32>,
}

impl QueryPopulation {
    fn synthetic(n_easy: usize, n_hard: usize) -> Self {
        let easy = (0..n_easy).map(|i| 100 + (i % 10) as u32).collect();
        let hard = (0..n_hard).map(|i| 200 + (i % 10) as u32).collect();
        Self { easy, hard }
    }

    /// Make a probe source for an easy query — all probes return the same token.
    fn easy_source(&self, answer: u32) -> SyntheticProbeSource<impl Fn(u8) -> ProbeOutput + '_> {
        SyntheticProbeSource::new(move |_arm| {
            // All arms agree → format hash constant.
            ProbeOutput::from_token(answer, answer as u64)
        })
    }

    /// Make a probe source for a hard query — each arm returns a different token.
    /// This simulates the "needle" regime: the model is genuinely uncertain.
    fn hard_source(&self, base: u32) -> SyntheticProbeSource<impl Fn(u8) -> ProbeOutput + '_> {
        SyntheticProbeSource::new(move |arm| {
            // Each arm returns a different token — high disagreement.
            // Same format hash (they're all "answers", just different ones).
            ProbeOutput::from_token(base + arm as u32, 999)
        })
    }
}

/// Simulate non-thinking baseline: every query on CPU, direct mode, fixed tokens.
fn run_baseline(pop: &QueryPopulation) -> Vec<RoutingDecision> {
    let mut out = Vec::with_capacity(pop.easy.len() + pop.hard.len());
    // Easy: correct, but wasted CoT tokens.
    for &answer in &pop.easy {
        out.push(RoutingDecision {
            tier: Tier::CpuOnly,
            cot_mode: CotMode::Thinking, // baseline always thinks
            tokens_used: 100,
            correct: true,
        });
        let _ = answer;
    }
    // Hard: wrong on CPU direct (model can't solve without more compute).
    for &_base in &pop.hard {
        out.push(RoutingDecision {
            tier: Tier::CpuOnly,
            cot_mode: CotMode::Direct, // baseline doesn't know to escalate
            tokens_used: 10,
            correct: false,
        });
    }
    out
}

/// Simulate TVP-routed: probe each query, decide tier + CoT from signal.
fn run_tvp_routed(
    pop: &QueryPopulation,
    config: &TvpConfig,
    adapter: &mut TvpThresholdAdapter,
    bandit: &mut TvpProbeCountBandit,
) -> Vec<RoutingDecision> {
    let aggregator = TvpAggregator::new(*config);
    let mut out = Vec::with_capacity(pop.easy.len() + pop.hard.len());
    let promote_at = adapter.promote_at();
    let demote_at = adapter.demote_at();

    // Easy queries: dense thicket → low disagreement → CPU, no CoT.
    for &answer in &pop.easy {
        let source = pop.easy_source(answer);
        let signal = aggregator.aggregate_k(&source);
        // Decision based on reasoning_disagreement.
        let (tier, cot_mode, tokens) = decide(&signal, promote_at, demote_at, /*gpu*/ true);
        // Easy queries are always correct (model knows them).
        let decision = RoutingDecision {
            tier,
            cot_mode,
            tokens_used: tokens,
            correct: true,
        };
        adapter.observe(signal, decision.correct);
        out.push(decision);
    }

    // Hard queries: needle regime → high disagreement → promote, expand CoT.
    for &base in &pop.hard {
        let source = pop.hard_source(base);
        let signal = aggregator.aggregate_k(&source);
        let (tier, cot_mode, tokens) = decide(&signal, promote_at, demote_at, /*gpu*/ true);
        // Hard queries: correct only if we promoted AND thought.
        let correct = tier != Tier::CpuOnly && cot_mode == CotMode::Thinking;
        let decision = RoutingDecision {
            tier,
            cot_mode,
            tokens_used: tokens,
            correct,
        };
        adapter.observe(signal, decision.correct);
        // Touch the bandit so it has observations (synthetic reward).
        let reward = if decision.correct { 1.0 } else { 0.0 };
        bandit.observe(reward);
        out.push(decision);
    }

    out
}

/// Decide tier + CoT mode from a TVP signal. Mirrors the router cascade logic.
fn decide(
    signal: &katgpt_rs::pruners::thicket_variance_probe::TvpSignal,
    promote_at: f32,
    demote_at: f32,
    gpu_available: bool,
) -> (Tier, CotMode, u32) {
    use katgpt_rs::pruners::thicket_variance_probe::TvpSignal;
    let _ = (demote_at, TvpSignal::zero()); // silence unused warnings if needed
    if signal.reasoning_disagreement > promote_at && gpu_available {
        // Needle regime → invest compute.
        (Tier::CpuGpu, CotMode::Thinking, 200)
    } else if signal.reasoning_disagreement < demote_at {
        // Dense thicket → cheap path, skip CoT.
        (Tier::CpuOnly, CotMode::Direct, 10)
    } else {
        // Ambiguous → think on CPU, medium tokens.
        (Tier::CpuOnly, CotMode::Thinking, 100)
    }
}

/// Print summary stats for a population of decisions.
fn summarize(name: &str, decisions: &[RoutingDecision]) {
    let n = decisions.len() as f32;
    let cpu = decisions
        .iter()
        .filter(|d| d.tier == Tier::CpuOnly)
        .count();
    let gpu = decisions
        .iter()
        .filter(|d| d.tier == Tier::CpuGpu)
        .count();
    let direct = decisions
        .iter()
        .filter(|d| d.cot_mode == CotMode::Direct)
        .count();
    let thinking = decisions
        .iter()
        .filter(|d| d.cot_mode == CotMode::Thinking)
        .count();
    let correct = decisions.iter().filter(|d| d.correct).count();
    let tokens: u32 = decisions.iter().map(|d| d.tokens_used).sum();

    println!("  {name}:");
    println!("    queries:       {}", decisions.len());
    println!(
        "    tier:          {} CPU / {} GPU / {} ANE",
        cpu,
        gpu,
        decisions
            .iter()
            .filter(|d| d.tier == Tier::CpuGpuAne)
            .count()
    );
    println!("    cot mode:      {} direct / {} thinking", direct, thinking);
    println!("    accuracy:      {:.1}% ({}/{})", correct as f32 / n * 100.0, correct, decisions.len());
    println!("    total tokens:  {}", tokens);
}

fn main() {
    println!("=== Thicket Variance Probe (TVP) — Before/After Demo (Plan 267) ===");
    println!();
    println!("Paper: Neural Thickets (Gan & Isola, MIT CSAIL, arXiv:2603.12228)");
    println!("Fusion: decoding-config-space probes → variance → router signal #8");
    println!();

    let pop = QueryPopulation::synthetic(100, 100);
    let config = TvpConfig::default();

    println!("Generating {} easy + {} hard synthetic queries...", pop.easy.len(), pop.hard.len());
    println!();

    // --- Baseline (non-thinking, no TVP) ---
    println!("── Baseline (no TVP, no probe) ────────────────────────────────");
    let baseline = run_baseline(&pop);
    summarize("baseline", &baseline);
    println!();

    // --- TVP-routed ---
    println!("── TVP-routed (K=4 probes, threshold-adaptive) ───────────────");
    let mut adapter = TvpThresholdAdapter::new(&config);
    let mut bandit = TvpProbeCountBandit::new();
    let tvp = run_tvp_routed(&pop, &config, &mut adapter, &mut bandit);
    summarize("tvp-routed", &tvp);
    println!();

    // --- Comparison ---
    println!("── Comparison ────────────────────────────────────────────────");
    let baseline_tokens: u32 = baseline.iter().map(|d| d.tokens_used).sum();
    let tvp_tokens: u32 = tvp.iter().map(|d| d.tokens_used).sum();
    let baseline_correct = baseline.iter().filter(|d| d.correct).count();
    let tvp_correct = tvp.iter().filter(|d| d.correct).count();
    let baseline_gpu = baseline
        .iter()
        .filter(|d| d.tier != Tier::CpuOnly)
        .count();
    let tvp_gpu = tvp.iter().filter(|d| d.tier != Tier::CpuOnly).count();

    println!(
        "  Token delta:         {:+.1}% ({baseline_tokens} → {tvp_tokens}) — hard queries get more compute",
        (tvp_tokens as f32 / baseline_tokens as f32 - 1.0) * 100.0
    );
    println!(
        "  Accuracy gain:       +{:.1}pp ({}/{}) → ({}/{})",
        (tvp_correct - baseline_correct) as f32 / baseline.len() as f32 * 100.0,
        baseline_correct,
        baseline.len(),
        tvp_correct,
        tvp.len()
    );
    println!(
        "  GPU promotions:      baseline used GPU on {} queries; TVP on {}",
        baseline_gpu, tvp_gpu
    );
    println!(
        "  Adapter thresholds:  promote_at={:.3}, demote_at={:.3} (after {} obs)",
        adapter.promote_at(),
        adapter.demote_at(),
        adapter.observations()
    );
    println!(
        "  Bandit selected K:   {} (after {} arm pulls)",
        bandit.current_k(),
        bandit.total_pulls()
    );
    println!();

    // --- Section 8 decomposition demo ---
    println!("── Format-vs-Reasoning Decomposition (paper Section 8) ───────");
    println!("Constructing three probe populations to show the decomposition:");
    println!();

    // All-same-token (no disagreement at all).
    let agree: Vec<ProbeOutput> = (0..4).map(|_| ProbeOutput::from_token(42, 1)).collect();
    // Format-only disagreement (same token, different surface forms).
    let format_only: Vec<ProbeOutput> = (0..4)
        .map(|i| ProbeOutput {
            token_id: 42,
            top_logits: [0.0; 32],
            logit_count: 0,
            format_hash: i as u64,
        })
        .collect();
    // Reasoning disagreement (different tokens, same format).
    let reasoning_only: Vec<ProbeOutput> = (0..4)
        .map(|i| ProbeOutput {
            token_id: i as u32 + 1,
            top_logits: [0.0; 32],
            logit_count: 0,
            format_hash: 999,
        })
        .collect();

    let agg = TvpAggregator::new(config);
    for (name, probes) in [
        ("all-agree      ", &agree[..]),
        ("format-only    ", &format_only[..]),
        ("reasoning-only ", &reasoning_only[..]),
    ] {
        let s = agg.aggregate(probes);
        println!(
            "  {name}: reasoning={:.3}  format={:.3}  promote={}  (tokens used: {})",
            s.reasoning_disagreement,
            s.format_disagreement,
            if s.should_promote(config.promote_at) { "YES" } else { "no" },
            probes.len()
        );
    }
    println!();
    println!("  → Only reasoning-only disagreement triggers promotion.");
    println!("  → Format-only disagreement stays on cheap substrate (canonicalize output).");
    println!();

    println!("=== Demo complete ===");
    println!();
    println!("TVP is a modelless, self-learning, CPU/GPU/ANE adaptive routing signal.");
    println!("GOAT gate (G1-G7) must pass before promotion to default-on.");
    println!("Critical gate G4: TVP+RV ablation — demote to research-only if redundant.");
}
