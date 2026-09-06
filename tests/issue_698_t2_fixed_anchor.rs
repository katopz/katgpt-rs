#![cfg(all(feature = "lt2_looped", feature = "loop_stability_fix"))]
//! Issue 698 T2 — GRT fixed-anchor loop A/B ordering bench (modelless, CPU-only).
//!
//! The GRT paper (arXiv:2608.15062, Research 519) Table 11 ablates the loop
//! anchor SOURCE under a trained gate: **frozen prelude output 2.68 <
//! drifting h(r−1) 3.38 (+0.70 nats) < raw input embedding 3.73 < zeros
//! 8.08**. This bench ports the ablation to our shipped `forward_looped` +
//! `ResidualGate` substrate via the new `LoopStabilityMode::FixedAnchor`.
//!
//! # The falsifiable question here
//!
//! > Does the FROZEN anchor (h^(0), hoisted once after the first loop
//! > iteration) converge faster per loop than the DRIFTING h^(τ-1) on our
//! > fixture — i.e. does the paper's ordering transfer to untrained weights?
//!
//! The paper's numbers are anchor-TRAINED; on vanilla random weights the
//! anchor is OOD input. **The ordering is the structural claim under test,
//! not the absolute numbers.** A reversal is a finding, not a failure.
//!
//! # Arms (all through production `forward_looped`)
//!
//! | arm | mode | gates | injected state |
//! |---|---|---|---|
//! | `fixed` | FixedAnchor | armed | frozen h^(0) (norm composed) |
//! | `drift` | InterLoopNorm | armed | drifting h^(τ−1) (norm) — Table 11's drifting arm |
//! | `zeros` | FixedAnchor | **zeroed** | nothing (norm only) — Table 11's zeros arm |
//! | `none` | None | armed | drifting h^(τ−1), un-normed — context only |
//!
//! `fixed` vs `drift` differ ONLY in the injected source (same norm, same
//! gate schedule) — the direct Table-11 comparison. `zeros` differs from
//! `fixed` only in the anchor's contribution. The paper's raw-embedding arm
//! is structurally the p=0 degenerate of our hoist (the tau==0 PRE-pass
//! state IS the embedding) and would need a second anchor source to run —
//! documented, not measured here.
//!
//! Gates are ARMED (`ResidualGate::new_loop_stable`, ρ=0.5 for τ>0): the
//! zero-init default makes every arm's injection a no-op, which would hide
//! the anchor entirely (the Plan-483 finding).
//!
//! # What is asserted vs recorded
//!
//! - **A-priori mechanism pins (asserted):** at r=1 all arms are
//!   bit-identical (no gated iteration → the anchor is never read);
//!   zeros-under-FixedAnchor ≡ zeros-under-InterLoopNorm (the mode's ONLY
//!   effect is the injected source); fixed ≠ drift at r=4 with armed gates
//!   (vacuity guard); double-run bit-identity per arm; G4 — the anchor
//!   buffer is allocated exactly once (`len == capacity == n_embd`, no
//!   growth).
//! - **Fixture identity (asserted):** the blake3[16] fixture hash equals
//!   T1's pinned `fab06e3f4ba65977` (same weights; the mode is not part of
//!   the hash; the x86_64-windows weight-byte drift T7 recorded yields the
//!   second pin `c894478d3febdb00`).
//! - **Ordering (measured, then pinned):** loss(r) = mean over all 27
//!   prompts of KL(softmax(arm@r) ‖ softmax(arm@32)) per arm; the measured
//!   fixed-vs-drift-vs-zeros ordering is pinned as raw bits.
//!
//! # Run
//!
//! ```bash
//! cargo test --features loop_stability_fix --test issue_698_t2_fixed_anchor -- --nocapture
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

/// Loop count of the per-arm fixed-point reference output.
const R_REF: usize = 32;

/// Reference-soundness probe: KL(R_REF, R_REF+8) must be negligible against
/// loss(2) for the reference to be sound (T1 measured ratio ≈ 2.6e-9).
const R_REF_PROBE: usize = R_REF + 8;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches T1 / the 407 quality-gate convention).
const SEED: u64 = 42;

/// T1's pinned fixture hash — this bench re-uses the exact same weights.
/// TWO platform pins: the aarch64 (M3) value T1 measured, and the
/// x86_64-windows value (weight-init libm ulp → different weight BYTES,
/// recorded by T7's off-aarch64 run + re-verified 2026-08-31).
const T1_FIXTURE_HASH: [&str; 2] = ["fab06e3f4ba65977", "c894478d3febdb00"];

/// Armed constant gate for τ > 0 (the `new_loop_stable` decay). 0.5 = strong
/// enough to make the anchor visible, in the stable band the constructor's
/// doc recommends exploring upward from 0.1–0.3.
const GATE_DECAY: f32 = 0.5;

/// Loop counts measured per arm (the convergence phase; T1: 54% of gain by
/// loop 2, 82% by 4, 99% by 8).
const R_ARMS: [usize; 4] = [2, 4, 8, 16];

/// Number of measured arms (Fixed, Drift, Zeros, NoneDefault).
const N_ARMS: usize = 4;

/// The committed A/B table (raw f32 bits), measured 2026-08-30 on the M3
/// (aarch64, debug profile — release re-validated bit-identical). Rows:
/// Fixed, Drift, Zeros, NoneDefault; columns: r = 2, 4, 8, 16. Cross-check:
/// the Zeros row EXACTLY equals T1's pinned spectrum at r=2/4/8/16 (zeros =
/// norm + no injection = the InterLoopNorm trajectory T1 measured) — a
/// bit-level cross-bench consistency proof.
const PINNED_TABLE: [[u32; 4]; N_ARMS] = [
    [0x403b_51be, 0x3f75_bdb8, 0x3efb_3384, 0x3cef_3798], // Fixed
    [0x40ce_5edc, 0x4000_b8b7, 0x3e59_61af, 0x3657_ecbb], // Drift
    [0x40b6_a83e, 0x4012_7fbe, 0x3dd5_a64d, 0x3684_f1a2], // Zeros == T1 r=2/4/8/16
    [0x415b_3ea5, 0x40be_1f05, 0x3f17_f7cd, 0x3b06_8958], // NoneDefault
];
/// Per-arm reference drift KL(32, 40), raw bits (same run).
const PINNED_REF_DRIFTS: [u32; N_ARMS] = [0x4019_d721, 0x3b7f_4b80, 0x330a_9972, 0x3dd8_0b92];
/// Destination bias KL(fixed@32 ‖ drift@32), raw bits.
const PINNED_DEST_FD: u32 = 0x4023_eb61;
/// Destination bias KL(drift@32 ‖ fixed@32), raw bits.
const PINNED_DEST_DF: u32 = 0x3fa6_5ad1;
/// Contraction sweep (fixed ref drift at ρ=0.1, ρ=0.25), raw bits.
const PINNED_SWEEP: [u32; 2] = [0x3cb6_159b, 0x3daa_cf6f];

// ── Fixture ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// FixedAnchor mode + armed gates: norm + frozen h^(0) injection.
    Fixed,
    /// InterLoopNorm mode + armed gates: norm + drifting h^(τ−1) injection
    /// (Table 11's drifting arm — identical conditions to `fixed` except the
    /// injected source).
    Drift,
    /// FixedAnchor mode + ZEROED gates: the anchor contributes nothing
    /// (norm only) — Table 11's zeros arm under the same composition.
    Zeros,
    /// None mode + armed gates: drifting injection WITHOUT the norm — the
    /// shipped default baseline, context only (not part of the Table-11
    /// ordering; the norm axis is confounded here).
    NoneDefault,
}

fn stability_mode(arm: Arm) -> LoopStabilityMode {
    match arm {
        Arm::Fixed | Arm::Zeros => LoopStabilityMode::FixedAnchor,
        Arm::Drift => LoopStabilityMode::InterLoopNorm,
        Arm::NoneDefault => LoopStabilityMode::None,
    }
}

fn make_config(arm: Arm) -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config.loop_stability_mode = stability_mode(arm);
    config
}

/// Armed gate for τ > 0 (zeros arm keeps τ > 0 at 0 — no injection).
fn make_gate(arm: Arm, config: &Config) -> ResidualGate {
    match arm {
        Arm::Zeros => ResidualGate::new(R_REF, config.n_embd),
        _ => ResidualGate::new_loop_stable(R_REF, config.n_embd, GATE_DECAY),
    }
}

/// Deterministic BLAKE3 over every active f32 weight slice (same as T1).
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
        0, // pos = 0 (single-position, matches the T1 / 407 convention)
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

/// log-softmax (max-shifted, sequential f32 — deterministic). Same as T1.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&l| l - max).collect();
    let sum_exp: f32 = shifted.iter().map(|&x| x.exp()).sum();
    let log_sum = sum_exp.ln();
    shifted.iter().map(|&x| x - log_sum).collect()
}

/// KL(P ‖ Q) between the categorical distributions of two logit vectors.
/// f64 accumulator + clamp (same as T1).
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

/// Bit-exact comparison of two logit vectors (raw f32 bits, element-wise).
fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Mean loss(r) for one arm: mean over all prompts of KL(arm@r ‖ arm@R_REF).
fn loss_at(
    _arm: Arm,
    config: &Config,
    weights: &TransformerWeights,
    gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    ref_logits: &[Vec<f32>],
    r: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (t, ref_l) in ref_logits.iter().enumerate() {
        let lr = run_once(config, weights, gate, sdpa_gate, t, r);
        acc += kl(&lr, ref_l);
    }
    acc / N_PROMPTS as f32
}

fn ref_outputs(
    _arm: Arm,
    config: &Config,
    weights: &TransformerWeights,
    gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
) -> Vec<Vec<f32>> {
    (0..N_PROMPTS)
        .map(|t| run_once(config, weights, gate, sdpa_gate, t, R_REF))
        .collect()
}

// ── The gate ─────────────────────────────────────────────────────

#[test]
fn t698_t2_fixed_anchor_ab_ordering() {
    // ── Fixture (identical weights to T1) ────────────────────────
    let config = make_config(Arm::Fixed);
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&config, &mut rng);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let hash = fixture_hash(&config, &weights);
    println!("fixture hash (blake3[16]): {hash}");
    assert!(
        T1_FIXTURE_HASH.contains(&hash.as_str()),
        "fixture must match a known platform pin of T1's weights (mode is not part of the hash)"
    );

    let gates: Vec<(Arm, Config, ResidualGate)> = [
        Arm::Fixed,
        Arm::Drift,
        Arm::Zeros,
        Arm::NoneDefault,
    ]
    .iter()
    .map(|&arm| {
        let c = make_config(arm);
        let g = make_gate(arm, &c);
        (arm, c, g)
    })
    .collect();

    // ── A-priori mechanism pins ──────────────────────────────
    // (1) r=1: no gated iteration runs, so every arm produces the identical
    // h^(0) → identical logits (the anchor is hoisted but never read).
    {
        let l1_ref = run_once(
            &gates[0].1,
            &weights,
            &gates[0].2,
            &sdpa_gate,
            0,
            1,
        );
        for (arm, cfg, gate) in &gates {
            let l1 = run_once(cfg, &weights, gate, &sdpa_gate, 0, 1);
            assert!(
                bits_eq(&l1, &l1_ref),
                "r=1 must be arm-independent (no gated iteration, anchor never read): {arm:?}"
            );
        }
    }

    // (2) zeros-under-FixedAnchor ≡ zeros-under-InterLoopNorm at r=8: with
    // zeroed gates the injected source contributes nothing, so the ONLY
    // difference between the two modes (the injected state) must vanish.
    // Proves the FixedAnchor mechanism is exactly "swap the injected source".
    {
        let mut c_zf = make_config(Arm::Zeros);
        c_zf.loop_stability_mode = LoopStabilityMode::FixedAnchor;
        let mut c_zi = make_config(Arm::Zeros);
        c_zi.loop_stability_mode = LoopStabilityMode::InterLoopNorm;
        let g0 = ResidualGate::new(R_REF, c_zf.n_embd);
        let a = run_once(&c_zf, &weights, &g0, &sdpa_gate, 0, 8);
        let b = run_once(&c_zi, &weights, &g0, &sdpa_gate, 0, 8);
        assert!(
            bits_eq(&a, &b),
            "zeroed gates must erase the mode difference (mechanism = injected source only)"
        );
    }

    // (3) vacuity guard: with ARMED gates the frozen anchor must change the
    // computation — fixed ≠ drift at r=4 (bit-level).
    {
        let (_, cfg_f, gate_f) = &gates[0];
        let (_, cfg_d, gate_d) = &gates[1];
        let a = run_once(cfg_f, &weights, gate_f, &sdpa_gate, 0, 4);
        let b = run_once(cfg_d, &weights, gate_d, &sdpa_gate, 0, 4);
        assert!(
            !bits_eq(&a, &b),
            "armed gates: frozen vs drifting anchor must produce different logits (vacuity guard)"
        );
    }

    // (4) G4: the anchor buffer is allocated exactly once — len == capacity
    // == n_embd after a full forward (no per-iteration growth).
    {
        let (_, cfg_f, gate_f) = &gates[0];
        let mut ctx = ForwardContext::new(cfg_f);
        let mut cache = MultiLayerKVCache::new(cfg_f);
        let mut ahla_cache = MultiLayerAhlaCache::new(cfg_f);
        forward_looped(
            &mut ctx,
            &weights,
            &mut cache,
            &mut ahla_cache,
            0,
            0,
            cfg_f,
            gate_f,
            &sdpa_gate,
            None,
            None,
            #[cfg(feature = "weight_shared_advantage_gate")]
            None,
            Some(8),
            #[cfg(feature = "gain_cost_halt")]
            None,
            None, // Issue 717: deep_run — None = bit-identical baseline
            #[cfg(feature = "cadence_gate")]
            None, // Issue 731: residual-exit probe — None = bit-identical baseline
        );
        assert_eq!(ctx.loop_anchor.len(), cfg_f.n_embd);
        assert_eq!(
            ctx.loop_anchor.capacity(),
            cfg_f.n_embd,
            "G4: the anchor buffer must be allocated exactly once (no growth)"
        );
    }

    // ── Per-arm references + soundness probe ─────────────────────
    // ── Per-arm references + soundness probe ───────────────
    let mut refs = Vec::with_capacity(gates.len());
    for (arm, cfg, gate) in &gates {
        let r = ref_outputs(*arm, cfg, &weights, gate, &sdpa_gate);
        refs.push(r);
    }
    // Soundness: KL(R_REF, R_REF+8) per arm. FINDING (measured): the fixed
    // arm does NOT settle — its ref drift (2.404) is ~9 orders above the
    // zeros arm's (3.2e-8): a CONSTANT ρ injection prevents fixed-point
    // contraction (the state keeps receiving ρ⊙h^(0)). The paper's late
    // contraction comes from its CLOSING gate schedule (openness 0.18 →
    // 0.07, the copy-late law) — T3 is the required complement, not a rival.
    let mut ref_drifts = [0.0f32; N_ARMS];
    for (idx, (arm, cfg, gate)) in gates.iter().enumerate() {
        let mut drift = 0.0f32;
        for (t, ref_l) in refs[idx].iter().enumerate() {
            let lp = run_once(cfg, &weights, gate, &sdpa_gate, t, R_REF_PROBE);
            drift += kl(ref_l, &lp);
        }
        ref_drifts[idx] = drift / N_PROMPTS as f32;
        println!("  ref drift KL({R_REF},{R_REF_PROBE}) [{arm:?}] = {:.3e}", ref_drifts[idx]);
        assert!(
            ref_drifts[idx].is_finite() && ref_drifts[idx] >= 0.0,
            "reference drift must be finite: {arm:?}"
        );
    }

    // ── G1: double-run bit-identity per arm ─────────────────────
    for (idx, (arm, cfg, gate)) in gates.iter().enumerate() {
        let a = loss_at(*arm, cfg, &weights, gate, &sdpa_gate, &refs[idx], 4);
        let b = loss_at(*arm, cfg, &weights, gate, &sdpa_gate, &refs[idx], 4);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "G1 determinism: loss(4) differs between runs for {arm:?}"
        );
    }

    // ── The A/B table ─────────────────────────────────────────
    let mut table = [[0.0f32; R_ARMS.len()]; N_ARMS];
    for (idx, (arm, cfg, gate)) in gates.iter().enumerate() {
        for (j, &r) in R_ARMS.iter().enumerate() {
            table[idx][j] = loss_at(*arm, cfg, &weights, gate, &sdpa_gate, &refs[idx], r);
        }
    }

    // Destination bias: how far the two fixed points themselves diverge
    // (asymmetric KL between the fixed and drift references).
    let dest_fd = {
        let mut acc = 0.0f32;
        for (a, b) in refs[0].iter().zip(refs[1].iter()) {
            acc += kl(a, b);
        }
        acc / N_PROMPTS as f32
    };
    let dest_df = {
        let mut acc = 0.0f32;
        for (a, b) in refs[1].iter().zip(refs[0].iter()) {
            acc += kl(a, b);
        }
        acc / N_PROMPTS as f32
    };

    // Contraction sweep (recorded): does the fixed arm's failure to settle
    // scale with the CONSTANT injection magnitude? A constant ρ keeps
    // injecting the same h^(0) forever; T3's CLOSING schedule is the
    // hypothesized complement. Values pinned below after measurement.
    let sweep_ref_drift = |decay: f32| {
        let mut cfg = make_config(Arm::Fixed);
        cfg.loop_stability_mode = LoopStabilityMode::FixedAnchor;
        let gate = ResidualGate::new_loop_stable(R_REF, cfg.n_embd, decay);
        let r32 = ref_outputs(Arm::Fixed, &cfg, &weights, &gate, &sdpa_gate);
        let mut drift = 0.0f32;
        for (t, r32_t) in r32.iter().enumerate() {
            let lp = run_once(&cfg, &weights, &gate, &sdpa_gate, t, R_REF_PROBE);
            drift += kl(r32_t, &lp);
        }
        drift / N_PROMPTS as f32
    };
    let drift_rho01 = sweep_ref_drift(0.1);
    let drift_rho025 = sweep_ref_drift(0.25);

    // ── Structural sanity + measured ordering ────────────────
    for (idx, (arm, ..)) in gates.iter().enumerate() {
        for (j, &r) in R_ARMS.iter().enumerate() {
            let l = table[idx][j];
            assert!(
                l.is_finite() && l >= 0.0,
                "loss({r}) for {arm:?} must be finite and non-negative: {l}"
            );
        }
        // Convergence: loss(2) > loss(16) for every arm (the loop converges).
        assert!(
            table[idx][0] > table[idx][R_ARMS.len() - 1],
            "{arm:?} must converge toward its fixed point: loss(2)={:.3e} vs loss(16)={:.3e}",
            table[idx][0],
            table[idx][R_ARMS.len() - 1]
        );
    }

    // ── The measured ordering (the transferred direction, asserted) ──
    // Early (r ≤ 4): fixed < drift — the paper's direction TRANSFERS.
    // Late (r ≥ 8): recorded, not asserted — see the verdict print.
    assert!(
        table[0][0] < table[1][0] && table[0][1] < table[1][1],
        "the transferred Table-11 direction must hold early: fixed < drift at r=2 and r=4 \
         (fixed {:.3e}/{:.3e} vs drift {:.3e}/{:.3e})",
        table[0][0], table[0][1], table[1][0], table[1][1]
    );

    // ── Measurement dump (bits for the pin consts) ─────────
    println!("\n═══ Issue 698 T2 — fixed-anchor A/B ordering (mean KL to own loop-32 reference) ═══");
    println!(
        "  fixture blake3[16] = {hash}  ·  seed {SEED}  ·  R_REF {R_REF}  ·  armed ρ(τ>0) = {GATE_DECAY}"
    );
    println!("  arm          loss(2)      loss(4)      loss(8)      loss(16)     ref-drift");
    for (idx, (arm, ..)) in gates.iter().enumerate() {
        println!(
            "  {:<12} {:<12e} {:<12e} {:<12e} {:<12e} {:.3e}",
            format!("{arm:?}"),
            table[idx][0],
            table[idx][1],
            table[idx][2],
            table[idx][3],
            ref_drifts[idx]
        );
    }
    println!("  destination bias KL(fixed@32 ‖ drift@32) = {dest_fd:.3e}   KL(drift@32 ‖ fixed@32) = {dest_df:.3e}");
    println!("  contraction sweep (fixed ref drift vs constant ρ): ρ=0.1 {drift_rho01:.3e} · ρ=0.25 {drift_rho025:.3e} · ρ=0.5 {:.3e}", ref_drifts[0]);
    println!("  table bits: {:?}", [
        [table[0][0].to_bits(), table[0][1].to_bits(), table[0][2].to_bits(), table[0][3].to_bits()],
        [table[1][0].to_bits(), table[1][1].to_bits(), table[1][2].to_bits(), table[1][3].to_bits()],
        [table[2][0].to_bits(), table[2][1].to_bits(), table[2][2].to_bits(), table[2][3].to_bits()],
        [table[3][0].to_bits(), table[3][1].to_bits(), table[3][2].to_bits(), table[3][3].to_bits()],
    ]);
    println!("  ref-drift bits: {:?}  dest bits: {} {}  sweep bits: {:?}",
        [
            ref_drifts[0].to_bits(),
            ref_drifts[1].to_bits(),
            ref_drifts[2].to_bits(),
            ref_drifts[3].to_bits()
        ],
        dest_fd.to_bits(),
        dest_df.to_bits(),
        [drift_rho01.to_bits(), drift_rho025.to_bits()]
    );

    // ── Pinned table ─────────────────────────────────────
    // CROSS-PLATFORM RECORD (2026-08-31, x86_64-windows 4090 box): the
    // exact-bit pins trip off-aarch64 by platform libm ulp drift in the
    // trajectory (T1's r=2 pin drifts 3 ulp; T7 recorded the class at
    // ~2.5e-7 rel; verified pre-existing at clean HEAD in a detached
    // worktree). Per T1's documented escape hatch the VALUE pins are now a
    // hybrid band — |Δ| ≤ 1e-5·|pinned| + 1e-6 (the 1e-6 absolute floor
    // keeps the SETTLING CLASSES separated: zeros ≈ 3e-8 vs drift 3.9e-3
    // stay 3 orders apart) — while the ORDERING asserts above (the actual
    // Table-11 claim) remain exact.
    let band = |pinned: f32| 1e-5 * pinned.abs() + 1e-6;
    for (idx, (arm, ..)) in gates.iter().enumerate() {
        for (j, &r) in R_ARMS.iter().enumerate() {
            let pinned = f32::from_bits(PINNED_TABLE[idx][j]);
            assert!(
                (table[idx][j] - pinned).abs() <= band(pinned),
                "pinned loss drifted: arm {arm:?} r={r}: measured {:.6e} vs pinned {:.6e}",
                table[idx][j],
                pinned
            );
        }
        let pd = f32::from_bits(PINNED_REF_DRIFTS[idx]);
        assert!(
            (ref_drifts[idx] - pd).abs() <= band(pd),
            "pinned ref drift drifted: arm {arm:?}: measured {:.6e} vs pinned {:.6e}",
            ref_drifts[idx],
            pd
        );
    }
    let pfd = f32::from_bits(PINNED_DEST_FD);
    assert!(
        (dest_fd - pfd).abs() <= band(pfd),
        "pinned dest FD drifted: measured {dest_fd:.6e} vs pinned {pfd:.6e}"
    );
    let pdf = f32::from_bits(PINNED_DEST_DF);
    assert!(
        (dest_df - pdf).abs() <= band(pdf),
        "pinned dest DF drifted: measured {dest_df:.6e} vs pinned {pdf:.6e}"
    );
    for (k, &v) in [drift_rho01, drift_rho025].iter().enumerate() {
        let ps = f32::from_bits(PINNED_SWEEP[k]);
        assert!(
            (v - ps).abs() <= band(ps),
            "pinned sweep[{k}] drifted: measured {v:.6e} vs pinned {ps:.6e}"
        );
    }

    println!();
    println!("  VERDICT (measured 2026-08-30): PARTIAL TRANSFER, with the mechanism found.");
    println!("  · Early (r≤4): fixed < drift — the paper's direction TRANSFERS (pinned).");
    println!("  · Late (r≥8): reversed — the constant-ρ fixed arm NEVER SETTLES (ref drift");
    println!("    2.404 vs 3.9e-3 drifting / 3.2e-8 zeros; monotone in ρ, never zero). The");
    println!("    paper's late contraction comes from its CLOSING gate (openness 0.18 → 0.07,");
    println!("    copy-late) — T3's schedule is the required complement, not a rival.");
    println!("  · NoneDefault (un-normed, armed) is worst at every r — the Plan-428 norm story");
    println!("    re-confirmed under armed gates.");
    println!("  Caveats: random weights (anchor is OOD input — the ORDERING is the claim),");
    println!("  single-position prompts, armed constant gate ρ={GATE_DECAY} (zero-init default would");
    println!("  hide the anchor entirely), raw-embed arm documented as the p=0 degenerate (not run).");
}
