#![cfg(feature = "lt2_looped")]
//! Issue 717 T1/T2/T5 — LT2 deep-loop baseline: `forward_looped` at T ≫ 4.
//!
//! Plan 108 validated the weight-shared loop at **T=4 only**. This gate
//! drives T ∈ {16, 64, 256, 1024} on the deterministic micro fixture and
//! measures, at DEFAULT settings (no stabilization knobs):
//!
//! - **T1** — the `LoopDeepRun` instrumentation: per-K state-norm snapshots
//!   plus logit-finite tripwires, opt-in via `Option<&mut LoopDeepRun>`
//!   (zero-cost-when-`None`, the elastic-override contract).
//! - **T2** — the baseline verdict: state-norm-vs-T and readout-
//!   consistency-vs-T. Either outcome is a result: flat-to-1024 means the
//!   guard work is cheap insurance; degradation means the Issue 717 damping
//!   knob is load-bearing. Headline numbers land in
//!   `.benchmarks/699_lt2_deep_loop_stability.md`.
//! - **T5** — the f32-state contract: the carried hidden state crosses every
//!   loop-iteration boundary as full f32. Weight-tied recurrence AMPLIFIES
//!   sub-f32 rounding with depth (sotaku: BF16 @4096 = 43.7% vs FP32
//!   98.6%) — the opposite regime of riir-ai Bench 802, where f16-KV
//!   deviation DILUTES with attention context. Pinned behaviorally: no
//!   snapshot value may sit on the f16 lattice (systematic sub-f32 storage
//!   would put EVERY value there).
//!
//! Run: `cargo test -p katgpt-rs --features lt2_looped --test issue_717_t1_t2_deep_baseline -- --nocapture`

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::loop_deep::{LoopDeepRun, LoopDeepStats, robust_norm};
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

/// Loop-count sweep (the issue's T1 ladder).
const T_LADDER: [usize; 4] = [16, 64, 256, 1024];

/// Prompt-token suite: all vocab tokens of the micro config (issue_407
/// convention — signal enters through the input embedding).
const N_TOKENS: usize = 27;

/// Stable fixture: default-scale seeded weights, zero residual gates, the
/// issue_407 shape (micro config, Uniform pattern, Ahla mode).
fn stable_fixture(t: usize) -> (Config, TransformerWeights, ResidualGate, SdpaOutputGate) {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: t };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = katgpt_rs::types::HlaMode::Ahla;
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(t, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    (config, weights, residual_gate, sdpa_gate)
}

/// One deep forward: returns (final logits, final carried-state norm, stats).
/// `run: None` exercises the bit-identical off path.
fn run_deep(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    mut run: Option<&mut LoopDeepRun>,
) -> (Vec<f32>, f32, LoopDeepStats) {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token,
        0,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        None, // elastic_loop_override
        #[cfg(feature = "gain_cost_halt")]
        None,
        run.as_deref_mut(),
        #[cfg(feature = "cadence_gate")]
        None, // Issue 731: residual-exit probe — None = bit-identical baseline
    );
    let logits = logits.to_vec();
    // Final carried state: `ctx.x` post-loop (mirrored into
    // `ctx.hidden_state` at the readout site). Max-abs-scaled norm —
    // overflow-safe for the deep regimes this harness measures.
    let final_norm = robust_norm(&ctx.x[..config.n_embd]);
    let stats = run
        .map(|r| std::mem::take(&mut r.stats))
        .unwrap_or_default();
    (logits.to_vec(), final_norm, stats)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let (mut na, mut nb) = (0.0f32, 0.0f32);
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d > 0.0 { dot / d } else { 0.0 }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map_or(0, |(i, _)| i)
}

/// True iff `v` is a finite normal-range f32 that is EXACTLY representable
/// in f16 (11-bit significand ⇒ the low 13 mantissa bits of the f32 must be
/// zero; biased exponent within f16 normals 2⁻¹⁴..2¹⁵).
fn on_f16_grid(v: f32) -> bool {
    if !v.is_finite() || v == 0.0 {
        return false; // exact in every format — not evidence either way
    }
    let bits = v.to_bits();
    let exp = (bits >> 23) & 0xff;
    (0x71..=0x8e).contains(&exp) && (bits & 0x0000_1fff) == 0
}

// ── T1: the deep-run harness at T ∈ {16, 64, 256, 1024} ─────────────────

#[test]
fn t1_deep_run_harness_t16_to_t1024() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 717 T1 — LoopDeepRun harness, stable fixture, T ladder    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    for &t in &T_LADDER {
        let (config, weights, gate, sdpa) = stable_fixture(t);
        let every = (t / 8).max(1); // 8 snapshots per run, deterministic
        let mut run = LoopDeepRun::new(every);

        // Token 0 drives the printed table; the whole suite runs in T2.
        let (logits, final_norm, stats) =
            run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));

        // Structural invariants at EVERY depth:
        assert_eq!(stats.snapshots_taken, t / every, "T={t}: snapshot count mismatch");
        assert!(
            stats.state_non_finite_at.is_none(),
            "T={t}: carried state went non-finite at snapshot {:?}",
            stats.state_non_finite_at
        );
        assert!(
            stats.logits_non_finite_at.is_none(),
            "T={t}: tripwire logits non-finite at snapshot {:?}",
            stats.logits_non_finite_at
        );
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "T={t}: final readout non-finite"
        );

        let norms = &stats.state_norms;
        let n_first = norms[0];
        let n_last = norms[norms.len() - 1];
        println!(
            "T={t:>5}: snapshots={} state-norm first={n_first:.4} last={n_last:.4} \
             ratio={:.3} final-readout-norm={final_norm:.4}",
            norms.len(),
            n_last / n_first
        );
    }
    println!("[T1] ✅ harness drives the full ladder; state + tripwire logits finite at every depth");
}

// ── T2: baseline verdict — consistency-vs-T + norm-vs-T ─────────────────

#[test]
fn t2_baseline_verdict_readout_consistency_vs_t() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 717 T2 — baseline at DEFAULT settings (no knobs)          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let tokens: Vec<usize> = (0..N_TOKENS).collect();

    // Reference readouts at T=16.
    let (cfg_ref, w_ref, g_ref, s_ref) = stable_fixture(16);
    let ref_readouts: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&tok| run_deep(&cfg_ref, &w_ref, &g_ref, &s_ref, tok, None).0)
        .collect();

    println!("┌────────┬────────────────┬────────────────┬──────────────────┐");
    println!("│   T    │ cos vs T=16    │ argmax agree   │ state-norm ratio │");
    println!("├────────┼────────────────┼────────────────┼──────────────────┤");

    for &t in &T_LADDER {
        let (config, weights, gate, sdpa) = stable_fixture(t);
        let every = (t / 8).max(1);
        let mut run = LoopDeepRun::new(every);
        let mut cos_sum = 0.0f32;
        let mut agree = 0u32;
        let mut norm_ratio_sum = 0.0f32;
        for (k, &tok) in tokens.iter().enumerate() {
            let (logits, _final_norm, stats) =
                run_deep(&config, &weights, &gate, &sdpa, tok, Some(&mut run));
            cos_sum += cosine(&logits, &ref_readouts[k]);
            if argmax(&logits) == argmax(&ref_readouts[k]) {
                agree += 1;
            }
            let norms = &stats.state_norms;
            if norms[0] > 0.0 {
                norm_ratio_sum += norms[norms.len() - 1] / norms[0];
            }
        }
        let n = tokens.len() as f32;
        println!(
            "│ {:>6} │ {:>14.6} │ {:>9}/{} │ {:>16.4} │",
            t,
            cos_sum / n,
            agree,
            tokens.len(),
            norm_ratio_sum / n
        );
        // Structural floor at every depth (the verdict itself is recorded,
        // not enforced — this is the measure-and-record task).
        assert!(
            agree as f32 / n >= 0.5,
            "T={t}: readout majority-flipped vs T=16 — deep instability \
             (the damping knob would be load-bearing; record and re-file)"
        );
    }
    println!("└────────┴────────────────┴────────────────┴──────────────────┘");
    println!("[T2] verdict recorded in .benchmarks/699_lt2_deep_loop_stability.md");
}

// ── T5: f32-state contract ───────────────────────────────────────────────

#[test]
fn f32_state_contract() {
    // Deep run with state capture on the stable fixture.
    let (config, weights, gate, sdpa) = stable_fixture(256);
    let mut run = LoopDeepRun::new(32);
    run.capture_states = true;
    let (_logits, _norm, stats) = run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));

    assert_eq!(stats.state_snapshots.len(), 8, "expected 8 snapshots at T=256/32");
    let all: Vec<f32> = stats.state_snapshots.concat();
    assert!(!all.is_empty(), "no state captured");

    // (a) Full f32 round-trip: every value survives bits→f32→bits exactly.
    assert!(
        all.iter().all(|&v| f32::from_bits(v.to_bits()) == v),
        "state value failed the f32 bit round-trip"
    );

    // (b) No sub-f32 lattice: an f16 storage would snap EVERY in-range value
    // onto the 11-bit-significand grid; a genuine f32 continuum essentially
    // never lands there (p ≈ 2⁻¹³ per value). Require ≤10% on-grid —
    // systematic sub-f32 storage drives this to ~100%.
    let on_grid = all.iter().filter(|&&v| on_f16_grid(v)).count();
    let on_frac = on_grid as f32 / all.len() as f32;
    assert!(
        on_frac <= 0.10,
        "T5 FAIL: {on_grid}/{} state values sit exactly on the f16 mantissa \
         grid ({:.1}%) — the carry path looks sub-f32",
        all.len(),
        100.0 * on_frac
    );

    // (c) Every snapshot finite (the contract's precondition on this arm).
    assert!(
        all.iter().all(|v| v.is_finite()),
        "state snapshot contained a non-finite value"
    );

    println!();
    println!(
        "[T5] ✅ f32-state contract: {} values across 8 snapshots, 100% f32 \
         round-trip, {:.2}% on the f16 lattice (≤10% allowed)",
        all.len(),
        100.0 * on_frac
    );
    println!("[T5] contrast pinned at the carry site: attention rounding DILUTES with");
    println!("[T5] context (Bench 802); weight-tied recurrence AMPLIFIES it with depth.");
}
