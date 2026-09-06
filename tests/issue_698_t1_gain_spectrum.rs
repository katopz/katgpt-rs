#![cfg(all(feature = "lt2_looped", feature = "loop_stability_fix"))]
//! Issue 698 T1 — Loop gain-spectrum measurement bench (modelless, CPU-only).
//!
//! The GRT paper (arXiv:2608.15062, Research 519) claims for looped
//! transformers: **77% of loss reduction lands in the first 2 loop steps**,
//! per-step gain is concave, and extending R past the trained R degrades —
//! jointly licensing an `l_min = 2` halter floor (this issue's T4).
//!
//! # The falsifiable question here
//!
//! > Does OUR loop stack put most of its convergence gain in the first 2
//! > loops (paper's regime → `l_min = 2` transfers), or later (a finding —
//! > the paper's floor does not transfer)?
//!
//! # Modelless method
//!
//! The paper's numbers come from TRAINED anchor weights; katgpt-rs ships no
//! training (modelless-first mandate), and the training-side companion is
//! riir-train `.plans/364`. So the spectrum is measured on the established
//! Plan-428-stable modelless fixture: `Config::micro()` +
//! `TransformerWeights::new(seed 42)` + `LoopMode::WeightShared` +
//! `LoopStabilityMode::InterLoopNorm` (the only stability fix that shipped).
//!
//! Quality curve: `loss(r) = mean over all 27 prompt tokens of
//! KL(softmax(logits at r loops) || softmax(logits at R_REF loops))` — the
//! convergence distance to the loop fixed point. The modelless analog of the
//! paper's Δloss(r): gain(r→r+1) = loss(r) − loss(r+1).
//!
//! # What is asserted vs recorded
//!
//! - **G1 (asserted):** double in-process re-measurement is bit-identical;
//!   the committed `PINNED_SPECTRUM` table reproduces exactly (same-platform
//!   determinism; a cross-platform run that trips the pin should relax to a
//!   tolerance and record the delta — the Bench-773 metric lesson applies).
//! - **Convergence sanity (asserted):** loss(1) > loss(R_STEPS) ≥ 0, all
//!   finite; the reference is (recorded) near-fixed: KL(32, 40) / loss(1)
//!   ≈ 2.6e-9.
//! - **Monotonicity (asserted, measured shape):** loss(r) strictly decreases
//!   through r = 15, then sits in a noise-floor plateau (< 1e-5) — the f32
//!   cancellation jitter at the fixed point, not divergence.
//! - **Concavity (RECORDED, measured FALSE mid-run):** the paper claims
//!   per-step gain is concave; this fixture shows a two-phase structure
//!   (gain dips r=3→4 then jumps r=4→5). A T4 concavity rule must read the
//!   table, not assume smooth decay.
//! - **Verdict (recorded):** 54.0% of total convergence by loop 2 (paper:
//!   77%), 81.6% by loop 4, 99.2% by loop 8 — the same front-loaded shape,
//!   with a heavier tail. `l_min = 2` is BORDERLINE on this fixture per the
//!   pre-registered bands (≥60% would transfer cleanly); T4 should pair the
//!   floor with the measured table rather than adopt it blind.
//!
//! # Run
//!
//! ```bash
//! cargo test --features loop_stability_fix --test issue_698_t1_gain_spectrum -- --nocapture
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng,
    SdpaOutputGate,
};

// ── Constants ────────────────────────────────────────────────────

/// Loop counts measured: r = 1..=R_STEPS.
const R_STEPS: usize = 24;

/// Loop count of the fixed-point reference output.
const R_REF: usize = 32;

/// Reference-soundness probe: KL(R_REF, R_REF + 8) must be negligible
/// against loss(1) for the fixed-point reference to be sound.
const R_REF_PROBE: usize = R_REF + 8;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches the 407 quality-gate convention).
const SEED: u64 = 42;

/// The committed gain-spectrum table: loss(r) for r = 1..=R_STEPS, measured
/// 2026-08-30 on the M3 (aarch64, debug profile — release re-validated
/// bit-identical). Values are pinned as raw bits so re-measurement must
/// reproduce them exactly on this platform (see the cross-platform note in
/// the pin assert).
const PINNED_SPECTRUM: [u32; R_STEPS] = [
    0x4146_90c0, // r=1  1.241e1
    0x40b6_a83e, // r=2  5.708e0
    0x4028_ab07, // r=3  2.635e0
    0x4012_7fbe, // r=4  2.289e0
    0x3f65_fdd5, // r=5  8.984e-1
    0x3f0c_35f7, // r=6  5.477e-1
    0x3ed6_68d4, // r=7  4.188e-1
    0x3dd5_a64d, // r=8  1.043e-1
    0x3cc5_03b9, // r=9  2.405e-2
    0x3c50_494d, // r=10 1.271e-2
    0x3b72_2213, // r=11 3.695e-3
    0x3b47_2589, // r=12 3.039e-3
    0x3a13_1fee, // r=13 5.612e-4
    0x380d_1857, // r=14 3.364e-5
    0x3650_8725, // r=15 3.107e-6
    0x3684_f1a2, // r=16 3.962e-6 (plateau noise)
    0x3598_8818, // r=17 1.136e-6
    0x3412_bf61, // r=18 1.367e-7
    0x32d0_6b24, // r=19 2.426e-8
    0x32ad_f0bc, // r=20 2.025e-8
    0x32fa_72e5, // r=21 2.916e-8
    0x3302_1022, // r=22 3.028e-8
    0x3344_64b8, // r=23 4.573e-8
    0x3292_d2af, // r=24 1.709e-8
];

// ── Fixture ──────────────────────────────────────────────────────

/// The pinned fixture config: micro + weight-shared loop at R_REF + InterLoopNorm.
fn make_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config.loop_stability_mode = LoopStabilityMode::InterLoopNorm;
    config
}

/// Deterministic BLAKE3 over every active f32 weight slice (bit-exact bytes).
/// Pins the fixture identity: any change to weight init (seed handling,
/// scales, config dims) breaks the pin loudly instead of silently re-basing
/// the committed spectrum.
fn fixture_hash(config: &Config, weights: &TransformerWeights) -> String {
    let mut hasher = blake3::Hasher::new();
    let feed = |h: &mut blake3::Hasher, v: &[f32]| {
        for f in v {
            h.update(&f.to_le_bytes());
        }
    };
    feed(&mut hasher, &weights.wte);
    feed(&mut hasher, &weights.wpe);
    feed(&mut hasher, &weights.lm_head);
    for layer in &weights.layers {
        feed(&mut hasher, &layer.attn_wq);
        feed(&mut hasher, &layer.attn_wk);
        feed(&mut hasher, &layer.attn_wv);
        feed(&mut hasher, &layer.attn_wo);
        feed(&mut hasher, &layer.mlp_w1);
        feed(&mut hasher, &layer.mlp_w2);
    }
    // Config dims ride the hash: a fixture-config change re-keys the table.
    for d in [
        config.vocab_size,
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.head_dim,
        config.mlp_hidden,
        R_REF,
    ] {
        hasher.update(&d.to_le_bytes());
    }
    hasher.finalize().to_hex()[..16].to_string()
}

// ── Forward + metrics ────────────────────────────────────────────

/// Run `forward_looped` for one prompt token at `r` loops (elastic override);
/// returns owned logits. Fresh ctx/caches per call — no cross-call state.
fn run_once(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    loops: usize,
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
        0, // pos = 0 (single-position, matches the 407 / PoC convention)
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        Some(loops), // elastic_loop_override
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // Issue 717: deep_run — None = bit-identical baseline
        #[cfg(feature = "cadence_gate")]
        None, // Issue 731: residual-exit probe — None = bit-identical baseline
    );
    logits.to_vec()
}

/// log-softmax (max-shifted, sequential f32 — deterministic).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&l| l - max).collect();
    let sum_exp: f32 = shifted.iter().map(|&x| x.exp()).sum();
    let log_sum = sum_exp.ln();
    shifted.iter().map(|&x| x - log_sum).collect()
}

/// KL(P ‖ Q) between the categorical distributions of two logit vectors.
/// f64 accumulator + clamp: near the fixed point both distributions are
/// near-identical and the naive f32 sum of `p·(lp−lq)` terms can cancel to a
/// tiny negative — mathematically KL ≥ 0, so clamp (deterministically).
fn kl(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let lp = log_softmax(p_logits);
    let lq = log_softmax(q_logits);
    let mut acc = 0.0f64;
    for i in 0..lp.len() {
        let p = lp[i].exp() as f64;
        acc += p * ((lp[i] - lq[i]) as f64);
    }
    if acc <= 0.0 {
        0.0
    } else {
        (acc as f32).max(0.0)
    }
}

struct Spectrum {
    /// loss(r) for r = 1..=R_STEPS (mean KL to the R_REF reference).
    table: [f32; R_STEPS],
    /// Reference-soundness probe: mean KL(R_REF, R_REF_PROBE).
    ref_drift: f32,
    /// loss(1) against the R_REF+8 output instead of R_REF (sanity symmetry).
    loss1_alt_ref: f32,
}

fn measure_spectrum(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
) -> Spectrum {
    // Reference outputs at R_REF (and the R_REF_PROBE drift arm).
    let ref_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_once(config, weights, residual_gate, sdpa_gate, t, R_REF))
        .collect();
    let probe_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_once(config, weights, residual_gate, sdpa_gate, t, R_REF_PROBE))
        .collect();

    let ref_drift = (0..N_PROMPTS)
        .map(|t| kl(&ref_logits[t], &probe_logits[t]))
        .sum::<f32>()
        / N_PROMPTS as f32;

    let mut table = [0.0f32; R_STEPS];
    for (idx, r) in (1..=R_STEPS).enumerate() {
        let mut acc = 0.0f32;
        for (t, ref_l) in ref_logits.iter().enumerate() {
            let lr = run_once(config, weights, residual_gate, sdpa_gate, t, r);
            acc += kl(&lr, ref_l);
        }
        table[idx] = acc / N_PROMPTS as f32;
    }

    let loss1_alt_ref = (0..N_PROMPTS)
        .map(|t| {
            let l1 = run_once(config, weights, residual_gate, sdpa_gate, t, 1);
            kl(&l1, &probe_logits[t])
        })
        .sum::<f32>()
        / N_PROMPTS as f32;

    Spectrum {
        table,
        ref_drift,
        loss1_alt_ref,
    }
}

// ── The gate ─────────────────────────────────────────────────────

#[test]
fn t698_t1_gain_spectrum_modelless() {
    let config = make_config();
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let hash = fixture_hash(&config, &weights);
    println!("fixture hash (blake3[16]): {hash}");

    // ── G1: double-run bit-identity ─────────────────────────────
    let a = measure_spectrum(&config, &weights, &residual_gate, &sdpa_gate);
    let b = measure_spectrum(&config, &weights, &residual_gate, &sdpa_gate);
    for i in 0..R_STEPS {
        assert_eq!(
            a.table[i].to_bits(),
            b.table[i].to_bits(),
            "G1 determinism: loss({}) differs between runs",
            i + 1
        );
    }
    assert_eq!(a.ref_drift.to_bits(), b.ref_drift.to_bits());

    // ── Structural sanity ────────────────────────────────────────
    for (i, &l) in a.table.iter().enumerate() {
        assert!(
            l.is_finite() && l >= 0.0,
            "loss({}) = {l} must be finite and non-negative",
            i + 1
        );
    }
    assert!(
        a.table[0] > a.table[R_STEPS - 1],
        "the loop must converge toward the fixed point: loss(1)={:.3e} vs loss({R_STEPS})={:.3e}",
        a.table[0],
        a.table[R_STEPS - 1]
    );

    // ── Pinned table (same-platform exact bits) ──────────────
    // CROSS-PLATFORM RECORD (2026-08-31, x86_64-windows 4090 box): the
    // exact-bit pin TRIPS off-aarch64 — measured 0x40b6a83b vs the pinned
    // 0x40b6a83e at r=2 (3 ulp), and at the tail the SAME ~3e-8 ABSOLUTE
    // drift is 7.8e-4 RELATIVE against the 3.4e-5 value at r=14 (the
    // Bench-773 small-denominator lesson: a pure relative band cannot
    // certify a spectrum that decays 6 orders). Verified IDENTICAL at clean
    // HEAD in a detached worktree — pre-existing platform libm ulp drift,
    // not a code change (T7 measured the same class at ~2.5e-7 rel).
    // Per this test's own documented escape hatch the pin is now a
    // hybrid band: |Δ| ≤ 1e-5·|pinned| + 5e-8 (relative + absolute floor;
    // the plateau values below the drift floor stay governed by the
    // plateau assert), + the same-platform double-run bit-identity assert
    // above (which still pins determinism exactly).
    for (i, (measured, &pinned_bits)) in a.table.iter().zip(PINNED_SPECTRUM.iter()).enumerate() {
        let pinned = f32::from_bits(pinned_bits);
        let band = 1e-5 * pinned.abs() + 5e-8;
        assert!(
            (measured - pinned).abs() <= band,
            "pinned spectrum drifted at r={}: measured {:.6e} vs pinned {:.6e} (|Δ| {:.2e} > band {:.2e})",
            i + 1,
            measured,
            pinned,
            (measured - pinned).abs(),
            band
        );
    }

    // ── Monotonicity + plateau (asserted from the measured table) ─
    // The measured curve is strictly decreasing through r = 15, then a
    // noise-floor plateau (KL ≤ 4.6e-6, f32 cancellation jitter at the
    // fixed point — negative "gains" of ~1e-8 are numeric noise, not
    // divergence). The assertion encodes exactly that shape:
    //   (a) strictly decreasing r = 1..=15 (the convergence phase),
    //   (b) plateau: every r ≥ 15 below NOISE_FLOOR (1e-5).
    const NOISE_FLOOR: f32 = 1e-5;
    let mut monotone = true;
    for i in 0..14 {
        if a.table[i + 1] >= a.table[i] {
            monotone = false;
        }
    }
    assert!(
        monotone,
        "loss(r) must strictly decrease through r=15 (convergence phase)"
    );
    for i in 14..R_STEPS {
        assert!(
            a.table[i] < NOISE_FLOOR,
            "loss({}) = {:.3e} must sit in the noise-floor plateau (<{NOISE_FLOOR:.0e})",
            i + 1,
            a.table[i]
        );
    }

    // Concavity (RECORDED, not asserted): the paper claims per-step gain is
    // concave. Measured: FALSE on this fixture even mid-run — the gain dips
    // at r=3→4 (+0.35) then jumps at r=4→5 (+1.39): a two-phase convergence
    // structure. A T4 concavity-floor rule cannot assume smooth decay on
    // random weights; it must read this table.
    let mut concave = true;
    let mut prev_gain = f32::INFINITY;
    for i in 0..R_STEPS - 1 {
        let gain = a.table[i] - a.table[i + 1];
        if gain > prev_gain {
            concave = false;
        }
        prev_gain = gain;
    }

    // ── Verdict: where does the gain land? ───────────────────────
    // ── Verdict: where does the gain land? ───────────────
    let total = a.table[0]; // loss(1) — everything after is convergence gain
    let frac_by = |k: usize| 1.0 - a.table[k - 1] / total;
    let f2 = frac_by(2);
    let f4 = frac_by(4);
    let f8 = frac_by(8);

    println!("\n═══ Issue 698 T1 — gain spectrum (mean KL to loop fixed point) ═══");
    println!("  fixture blake3[16] = {hash}  ·  seed {SEED}  ·  R_REF {R_REF}  ·  prompts {N_PROMPTS}");
    println!("   r   loss(r)      bits      gain(r→r+1)");
    for i in 0..R_STEPS {
        let gain = if i + 1 < R_STEPS {
            format!("{:+.3e}", a.table[i] - a.table[i + 1])
        } else {
            "—".to_string()
        };
        println!(
            "  {:>3}  {:.3e}  0x{:08x}  {}",
            i + 1,
            a.table[i],
            a.table[i].to_bits(),
            gain
        );
    }
    println!("  ref drift KL({R_REF},{R_REF_PROBE}) = {:.3e}  (vs loss(1) {:.3e} → ratio {:.2e})",
        a.ref_drift, a.table[0], a.ref_drift / a.table[0]);
    println!("  loss(1) vs alt ref {R_REF_PROBE} = {:.3e}", a.loss1_alt_ref);
    println!();
    println!("  convergence fraction by loop 2: {:.1}%   (paper: 77%)", f2 * 100.0);
    println!("  convergence fraction by loop 4: {:.1}%", f4 * 100.0);
    println!("  convergence fraction by loop 8: {:.1}%", f8 * 100.0);
    println!("  concave per-step gain (recorded): {concave}");
    println!();
    if f2 >= 0.60 {
        println!("  VERDICT: gain concentrates in the first 2 loops (≥60%) — the");
        println!("  paper's l_min=2 halter floor TRANSFERS to the modelless fixture.");
    } else if f2 >= 0.40 {
        println!("  VERDICT: borderline — a meaningful share of gain lands after loop 2.");
        println!("  l_min=2 needs the T4 concavity rule rather than a hard floor.");
    } else {
        println!("  VERDICT: >40% of gain lands AFTER loop 2 — the paper's l_min=2");
        println!("  does NOT transfer to this fixture. That is a finding, not a failure.");
    }
    println!("  Caveats: random weights (no anchor training — the paper's numbers are");
    println!("  from anchor-trained models; the ORDERING is the structural claim here),");
    println!("  single-position prompts, InterLoopNorm arm (the shipped stability fix).");
}
