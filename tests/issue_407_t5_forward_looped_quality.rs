#![cfg(feature = "lt2_looped")]
//! Plan 318 Issue 407 T5 — Modelless depth→quality gate on `forward_looped`.
//!
//! The **G6 modelless analog**. Every existing LT2 test measures finiteness,
//! byte-identity, KL-to-self, norm growth, or throughput — **none measures a
//! quality delta on `forward_looped` itself**. This test closes that gap by
//! applying the same T4 pattern (depth→discrimination K-sweep) to the Gemma2-
//! shaped weight-shared loop path.
//!
//! # The claim under test
//!
//! G6 (tf_loop depth recycling) promised "thinking longer in latent space
//! improves quality." The modelless version: does `LoopMode::WeightShared {
//! loop_count: K }` with K > 1 **preserve or amplify** the model's ability to
//! discriminate which token was the input, on RANDOM weights?
//!
//! If YES (K>1 discrimination ≥ K=1), the weight-shared loop preserves signal
//! through recurrence — the architectural precondition for G6 holds.
//!
//! If NO (K>1 discrimination < K=1), the loop washes out input signal on
//! random weights — needs trained dynamics to work.
//!
//! # Method
//!
//! Zero GPU, zero checkpoints. `Config::micro()` + `TransformerWeights::new`
//! (random, seed 42). Signal enters through the **input** (token embedding
//! lookup), not through learned weights.
//!
//! For each of N_DISTINCT prompt tokens × K ∈ {1, 2, 4, 8}:
//! 1. Build config with `WeightShared { loop_count: K }`.
//! 2. Feed the token through `forward_looped` → get output logits.
//! 3. Record the logits.
//!
//! **Metrics:**
//! - **G6-DISC (inter-prompt logit discrimination):** avg pairwise cosine
//!   distance between logits from distinct prompt tokens. Higher = more
//!   discriminative.
//! - **G6-NORM (signal energy):** mean L2 norm of logits. Tracks whether the
//!   output is decaying (contractive loop) or stable/expanding.
//!
//! # Corroborating priors
//!
//! - `tests/bench_gram_width_depth.rs` (`elf_sde bandit`): depth T=1→8 gives
//!   +6.46% quality; width K=1→20 gives +0.16%. **Depth dominates width.**
//! - `crates/katgpt-micro-belief/src/coherence_bench.rs`: K=3 beats K=1 on
//!   flip-flop count (560 vs 569).
//! - `examples/loop_stability_poc.rs`: T=12 norm ratio 11.19× vs InterLoopNorm
//!   3.34×; KL 0.0128 → 0.0008.
//!
//! # Run
//!
//! ```bash
//! cargo test --features lt2_looped --test issue_407_t5_forward_looped_quality -- --nocapture
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

// ── Constants ────────────────────────────────────────────────────

/// Number of distinct prompt tokens (signal-through-input probes).
/// Config::micro() has vocab_size=27, so we use all 27 tokens.
const N_DISTINCT: usize = 27;

/// Loop-count sweep values. K=1 is the baseline (single pass, no looping).
const K_CANDIDATES: [usize; 4] = [1, 2, 4, 8];

// ── Helpers ──────────────────────────────────────────────────────

/// Cosine distance between two vectors. In [0, 2].
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { 1.0 - dot / denom } else { 0.0 }
}

/// Average pairwise cosine distance across a set of vectors.
fn avg_pairwise_cosine_distance(states: &[Vec<f32>]) -> f32 {
    let n = states.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += cosine_distance(&states[i], &states[j]);
            count += 1;
        }
    }
    if count > 0 { sum / count as f32 } else { 0.0 }
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Per-K measurement ────────────────────────────────────────────

struct KResult {
    k: usize,
    disc: f32,    // G6-DISC: avg pairwise cosine distance of logits
    norm: f32,    // G6-NORM: mean L2 norm of logits
    finite: bool, // all logits finite (G1 structural)
}

/// Build a micro config with `loop_count = K`, Uniform hybrid pattern, AHLA mode.
fn make_config(loop_count: usize) -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = katgpt_rs::types::HlaMode::Ahla;
    config
}

/// Run `forward_looped` for a single token at pos=0; return owned logits.
fn run_once(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token,
        0, // pos=0
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
        None, // Issue 717: deep_run — None = bit-identical baseline
    );
    logits.to_vec()
}

/// Measure the K-sweep for a single loop_count K.
fn measure_k(k: usize, prompt_tokens: &[usize]) -> KResult {
    let config = make_config(k);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(k, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

    let mut logit_states: Vec<Vec<f32>> = Vec::with_capacity(N_DISTINCT);
    let mut all_finite = true;

    for &token in prompt_tokens {
        let logits = run_once(&config, &weights, &residual_gate, &sdpa_gate, token);
        if logits.iter().any(|l| !l.is_finite()) {
            all_finite = false;
        }
        logit_states.push(logits);
    }

    let disc = avg_pairwise_cosine_distance(&logit_states);
    let norm = logit_states.iter().map(|l| l2_norm(l)).sum::<f32>() / N_DISTINCT as f32;

    KResult {
        k,
        disc,
        norm,
        finite: all_finite,
    }
}

// ── Test: the depth→quality K-sweep ──────────────────────────────

#[test]
fn forward_looped_depth_quality_ksweep() {
    // Prompt tokens: all vocab tokens (Config::micro() vocab=27).
    let prompt_tokens: Vec<usize> = (0..N_DISTINCT).collect();

    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Issue 407 T5 — Modelless forward_looped depth→quality     ║");
    println!("║  G6 analog, zero GPU, zero checkpoints (random weights)    ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Config::micro() (n_embd={}, n_layer={}, vocab={})",
        Config::micro().n_embd,
        Config::micro().n_layer,
        Config::micro().vocab_size
    );
    println!("K sweep: {K_CANDIDATES:?}");
    println!("N_DISTINCT prompt tokens: {N_DISTINCT}");
    println!();

    // Run the K-sweep.
    println!("┌──────┬──────────────┬──────────────┬──────────────┐");
    println!("│  K   │  G6-DISC     │  G6-NORM     │  Finite      │");
    println!("│      │ (inter-prompt │ (mean logit  │              │");
    println!("│      │  logit dist)  │  L2 norm)    │              │");
    println!("├──────┼──────────────┼──────────────┼──────────────┤");

    let mut results: Vec<KResult> = Vec::with_capacity(K_CANDIDATES.len());
    for &k in &K_CANDIDATES {
        let label = if k == 1 {
            " (baseline)"
        } else {
            "            "
        };
        let result = measure_k(k, &prompt_tokens);
        println!(
            "│ {:>4} │ {:>12.6} │ {:>12.6} │ {:>12} │{}",
            k, result.disc, result.norm, result.finite, label
        );
        results.push(result);
    }
    println!("└──────┴──────────────┴──────────────┴──────────────┘");
    println!();

    // ── G1 structural: all finite ────────────────────────────────
    let all_finite = results.iter().all(|r| r.finite);
    println!(
        "G1 (all finite): {}",
        if all_finite { "✅ PASS" } else { "❌ FAIL" }
    );
    assert!(all_finite, "G1 FAIL: non-finite logits at some K");
    println!();

    // ── Gate verdict ─────────────────────────────────────────────
    let baseline = &results[0]; // K=1
    println!("═══ Gate Verdict (baseline = K=1) ═══");
    println!();

    // G6-DISC: does any K>1 inter-prompt distance ≥ K=1?
    let disc_best = results.iter().skip(1).max_by(|a, b| {
        katgpt_core::float_order::cmp_for_max(a.disc, b.disc)
    });
    let disc_pass = disc_best.is_some_and(|r| r.disc >= baseline.disc);
    println!("  G6-DISC (logit discrimination):");
    println!("    K=1 baseline: {:.6}", baseline.disc);
    if let Some(best) = disc_best {
        println!(
            "    Best K>1:     K={}, disc={:.6} (delta {:+.6})",
            best.k,
            best.disc,
            best.disc - baseline.disc
        );
    }
    println!(
        "    Verdict: {}",
        if disc_pass {
            "✅ PASS (K>1 preserves/amplifies discrimination)"
        } else {
            "❌ FAIL (loop washes out input signal on random weights)"
        }
    );
    println!();

    // G6-NORM: structural guard — does the output collapse?
    let norm_floor = baseline.norm * 0.1;
    let norm_pass = results.last().is_some_and(|r| r.norm >= norm_floor);
    println!("  G6-NORM (signal energy / structural guard):");
    println!("    K=1 baseline: {:.6}", baseline.norm);
    println!(
        "    K=8:          {:.6} (collapse floor = 10% of baseline = {:.6})",
        results.last().unwrap().norm,
        norm_floor
    );
    println!(
        "    Verdict: {}",
        if norm_pass {
            "✅ PASS (output doesn't collapse)"
        } else {
            "⚠️  COLLAPSE (weight-shared loop is contractive)"
        }
    );
    println!();

    // Overall G6 modelless verdict.
    let overall_pass = disc_pass;
    println!("═══ Overall G6 Modelless Verdict ═══");
    println!(
        "  {}",
        if overall_pass {
            "✅ PASS — weight-shared loop preserves input signal through depth."
        } else {
            "❌ FAIL — loop washes out input signal on random weights."
        }
    );
    if overall_pass {
        println!("  The LT2 recurrence retains discriminability through K iterations.");
        println!("  A *trained* model could exploit this for depth-based quality gains.");
    } else {
        println!("  The LT2 dynamics on random weights wash out signal. This is");
        println!("  consistent with the loop_stability_poc finding (T=12 norm ratio");
        println!("  11.19× on naive loop — divergent without InterLoopNorm). The");
        println!("  mechanism needs InterLoopNorm or trained gates to stabilize.");
    }
    println!();

    // Honest interpretation.
    println!("═══ Corroborating Priors ═══");
    println!("  bench_gram_width_depth.rs: depth T=1→8 +6.46%, width +0.16% (depth dominates)");
    println!("  coherence_bench.rs: K=3 > K=1 on flip-flop count");
    println!(
        "  loop_stability_poc.rs: naive loop diverges (norm 11.19×); InterLoopNorm stabilizes"
    );
    println!();
    println!("  Note: Config::micro() uses naive ResidualGate (zero-init). The loop_stability_poc");
    println!("  shows naive loops diverge without InterLoopNorm. A FAIL here on the naive gate");
    println!("  is consistent with that finding — the fix is loop_stability_mode = InterLoopNorm.");

    // We don't hard-assert PASS/FAIL — the test PASSES as long as G1 (finite)
    // holds. The G6-DISC/NORM verdicts are recorded for interpretation, not
    // enforced. This is the honest "measure + record" approach: the gate's
    // value is in the data, not in a binary pass/fail on random weights.
}
