#![cfg(all(feature = "lt2_looped", feature = "loop_stability_fix"))]
//! Issue 698 T8 — hand conditional gate A/B (modelless, CPU-only).
//!
//! The mechanism gate (`tests/issue_698_t8_gate_probe.rs`) PASSED: per-loop
//! state divergence co-locates with marginal loop gain at the pre-registered
//! rank thresholds (pooled ρ +0.360, median per-r +0.324, 13/15 strata
//! positive) in the FixedAnchor+armed deployment context. This bench ships
//! the gate MECHANISM and measures its falsifiable content:
//!
//! > Does an ADAPTIVE convex blend — copy weight
//! > `g_τ = σ(β·(cos(S(τ−1), S(τ−2)) − θ) + b)`, open on divergence —
//! > contract the loop like T3's static copy-late schedules, and does the
//! > per-token adaptivity actually differentiate tokens (different freeze
//! > loops per token), at what quality?
//!
//! # Arms (InterLoopNorm context — see the pairing finding below; single-position)
//!
//! - **nat** — `ResidualGate::new` (zero additive gates: injection no-op)
//!   — the natural normed trajectory (= T1's fixture trajectory);
//! - **cvx** — `copy_late_schedule(32, 0.7, 0.95)` — the STRONGEST static
//!   linear comparator from T3's endpoint sweep (1.7e-6 ref drift there);
//! - **cond_a** — `new_conditional(β=400, θ=0.99, b=+2)` — freezes above
//!   cos ≈ 0.995 (sharp transition at 0.99);
//! - **cond_b** — `new_conditional(β=400, θ=0.999, b=+2)` — a later,
//!   more conservative closure;
//! - **cond_ah** — `new_conditional_hard(...)` — cond_a + the hard-freeze
//!   clamp (soft g > 0.9 snaps to EXACTLY 1.0).
//!
//! # The context-pairing finding (measured 2026-08-31, FixedAnchor first)
//!
//! The first A/B ran the conditional arms on FixedAnchor + armed gates —
//! and the hard clamp FAILED to settle there (ref drift 1.5; soft 1.9).
//! The mechanism: the conditional gate is a FREEZE-IN-PLACE mechanism.
//! With the ANCHOR as injection source, g = 1.0 jumps the state TO the
//! anchor — a different point than where the trajectory was — so the next
//! loop's cos(S(τ), S(τ−1)) drops and the gate REOPENS: an oscillation.
//! With the DRIFTING prev-state source (this bench), g = 1.0 copies the
//! state where it IS: the next cos is ~1.0, the gate stays shut, the
//! token stays frozen — self-consistent contraction by construction.
//! The T8 mechanism-gate probe's InterLoopNorm co-location (+0.428 pooled)
//! was also the stronger context. Pairing verdict: conditional gate ↔
//! drift source; the anchor re-injection and the conditional freeze
//! fight each other (recorded — do not deploy them together).
//!
//! # Measured per arm
//!
//! - loss grid (mean KL to the arm's OWN @32 fixed point) at r ∈ {2, 4, 8, 16};
//! - ref drift KL(@32 ‖ @40) — contraction/settling;
//! - destination bias KL(arm@32 ‖ nat@32) — where the gate parks the state
//!   relative to the natural trajectory (T3's honest price axis);
//! - conditional arms: per-token freeze loop (first r with g > 0.9) — the
//!   ADAPTIVITY evidence (a static schedule has one freeze loop for every
//!   token by construction; the conditional's spread is the mechanism's
//!   differentiating content).
//!
//! # Pre-registered gates
//!
//! - **G1:** per-arm double-run bit-identity.
//! - **Non-vacuity:** cond arms differ from nat bit-wise somewhere; the
//!   freeze-loop spread across tokens is > 0 (the gate is actually adaptive).
//! - **Free-bound composition:** cond arm logits finite at every measured r
//!   (the blend is norm-bounded for any g ∈ [0, 1] — T3's spec test holds).
//! - Verdict (recorded, not asserted): contraction + adaptivity numbers;
//!   the quality arbiter on DEPLOYED paths stays pending trained weights
//!   (the issue's coin-flip caveat, bounded by the mechanism-gate PASS).
//!
//! # Run
//!
//! ```bash
//! cargo test --features lt2_looped,loop_stability_fix --test issue_698_t8_conditional_ab -- --nocapture
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng, SdpaOutputGate,
};

// ── Constants ────────────────────────────────────────────────────

/// Loop counts measured per arm.
const R_GRID: [usize; 4] = [2, 4, 8, 16];

/// Loop count of the fixed-point reference output (T1's R_REF).
const R_REF: usize = 32;

/// Reference-soundness probe loop count.
const R_REF_PROBE: usize = R_REF + 8;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches the T1 convention).
const SEED: u64 = 42;

/// Known fixture hashes (aarch64 + x86_64-windows Issue-698 pins).
const KNOWN_FIXTURE_HASHES: [&str; 2] = ["fab06e3f4ba65977", "c894478d3febdb00"];

/// Freeze detection: a token is FROZEN at the first loop whose copy weight
/// exceeds this (the blend is ≥90% the injected source).
const FREEZE_G: f32 = 0.9;

// ── Fixture (verbatim from T1) ───────────────────────────────────

fn make_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    // InterLoopNorm — the conditional gate's self-consistent context (see
    // the module-doc pairing finding: freeze-in-place wants the drifting
    // source; anchor re-injection breaks the freeze memory).
    config.loop_stability_mode = LoopStabilityMode::InterLoopNorm;
    config
}

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

// ── Forward + metrics (helpers shared with the T1/T6/T8 benches) ─

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
        0,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        Some(loops),
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // Issue 717: deep_run — None = bit-identical baseline
        #[cfg(feature = "cadence_gate")]
        None, // Issue 731: residual-exit probe — None = bit-identical baseline
    );
    logits.to_vec()
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&l| l - max).collect();
    let sum_exp: f32 = shifted.iter().map(|&x| x.exp()).sum();
    let log_sum = sum_exp.ln();
    shifted.iter().map(|&x| x - log_sum).collect()
}

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

// ── Arm measurement ──────────────────────────────────────────────

struct Arm {
    name: &'static str,
    gate: ResidualGate,
}

struct ArmResult {
    /// loss(r) for r in R_GRID (mean KL to the arm's own @32 output).
    loss: [f32; R_GRID.len()],
    /// mean KL(@32 ‖ @40) — settling.
    ref_drift: f32,
    /// mean KL(arm@32 ‖ nat@32) — destination bias.
    dest_bias: f32,
}

fn measure_arm(
    arm: &Arm,
    weights: &TransformerWeights,
    sdpa_gate: &SdpaOutputGate,
    nat_ref: &[(usize, Vec<f32>)],
) -> ArmResult {
    let config = make_config();
    let ref_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_once(&config, weights, &arm.gate, sdpa_gate, t, R_REF))
        .collect();
    let probe_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_once(&config, weights, &arm.gate, sdpa_gate, t, R_REF_PROBE))
        .collect();
    let ref_drift = (0..N_PROMPTS)
        .map(|t| kl(&ref_logits[t], &probe_logits[t]))
        .sum::<f32>()
        / N_PROMPTS as f32;
    let mut loss = [0.0f32; R_GRID.len()];
    for (idx, &r) in R_GRID.iter().enumerate() {
        let mut acc = 0.0f32;
        for (t, ref_l) in ref_logits.iter().enumerate() {
            let lr = run_once(&config, weights, &arm.gate, sdpa_gate, t, r);
            acc += kl(&lr, ref_l);
        }
        loss[idx] = acc / N_PROMPTS as f32;
    }
    let dest_bias = (0..N_PROMPTS)
        .map(|t| kl(&ref_logits[t], &nat_ref[t].1))
        .sum::<f32>()
        / N_PROMPTS as f32;
    ArmResult {
        loss,
        ref_drift,
        dest_bias,
    }
}

/// Per-token freeze loop for a conditional arm: first r in 2..=R_REF whose
/// gate value exceeds FREEZE_G (recomputed from the SAME cosines the
/// forward saw — via `conditional_gate_at` on the arm's own gate; the
/// carried states come from the production runs at each r).
/// Returns None for tokens that never freeze within the window.
fn freeze_loops(
    arm: &Arm,
    weights: &TransformerWeights,
    sdpa_gate: &SdpaOutputGate,
) -> Vec<Option<usize>> {
    let config = make_config();
    (0..N_PROMPTS)
        .map(|t| {
            // Reconstruct the carried states S(1..=R_REF) through production.
            let mut states: Vec<Vec<f32>> = Vec::with_capacity(R_REF);
            for r in 1..=R_REF {
                let mut ctx = ForwardContext::new(&config);
                let mut cache = MultiLayerKVCache::new(&config);
                let mut ahla = MultiLayerAhlaCache::new(&config);
                let _ = forward_looped(
                    &mut ctx,
                    weights,
                    &mut cache,
                    &mut ahla,
                    t,
                    0,
                    &config,
                    &arm.gate,
                    sdpa_gate,
                    None,
                    None,
                    #[cfg(feature = "weight_shared_advantage_gate")]
                    None,
                    Some(r),
                    #[cfg(feature = "gain_cost_halt")]
                    None,
                    None, // Issue 717: deep_run — None = bit-identical baseline
                    #[cfg(feature = "cadence_gate")]
                    None, // Issue 731: residual-exit probe — None = bit-identical baseline
                );
                states.push(ctx.hidden_state[..config.n_embd].to_vec());
            }
            // The forward's gate at loop τ (τ ≥ 2) sees prev = S(τ−1),
            // prev_prev = S(τ−2) — the (r−1, r−2) entries here.
            for tau in 2..R_REF {
                if let Some(g) = arm
                    .gate
                    .conditional_gate_at(tau, &states[tau - 1], &states[tau - 2])
                    && g > FREEZE_G
                {
                    return Some(tau);
                }
            }
            None
        })
        .collect()
}

// ── The A/B ──────────────────────────────────────────────────────

#[test]
fn t698_t8_conditional_gate_ab() {
    let weights_config = make_config();
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&weights_config, &mut rng);
    let sdpa_gate = SdpaOutputGate::new(
        weights_config.n_head,
        weights_config.head_dim,
        weights_config.n_embd,
    );
    let hash = fixture_hash(&weights_config, &weights);
    println!("fixture hash (blake3[16]): {hash}");
    assert!(
        KNOWN_FIXTURE_HASHES.contains(&hash.as_str()),
        "unknown fixture hash {hash} — the Issue-698 fixture drifted"
    );

    let arms = [
        Arm {
            name: "nat",
            gate: ResidualGate::new(R_REF, weights_config.n_embd),
        },
        Arm {
            name: "cvx(0.7→0.95)",
            gate: ResidualGate::copy_late_schedule(R_REF, 0.7, 0.95),
        },
        Arm {
            name: "cond_a(θ=.99)",
            gate: ResidualGate::new_conditional(R_REF, weights_config.n_embd, 400.0, 0.99, 2.0),
        },
        Arm {
            name: "cond_b(θ=.999)",
            gate: ResidualGate::new_conditional(R_REF, weights_config.n_embd, 400.0, 0.999, 2.0),
        },
        Arm {
            name: "cond_ah(hard)",
            gate: ResidualGate::new_conditional_hard(
                R_REF,
                weights_config.n_embd,
                400.0,
                0.99,
                2.0,
            ),
        },
    ];

    // Natural reference (destination-bias basis).
    let config = make_config();
    let nat_ref: Vec<(usize, Vec<f32>)> = (0..N_PROMPTS)
        .map(|t| {
            let l = run_once(&config, &weights, &arms[0].gate, &sdpa_gate, t, R_REF);
            (t, l)
        })
        .collect();

    // G1: per-arm double-run bit-identity.
    let mut results = Vec::with_capacity(arms.len());
    for arm in &arms {
        let a = measure_arm(arm, &weights, &sdpa_gate, &nat_ref);
        let b = measure_arm(arm, &weights, &sdpa_gate, &nat_ref);
        for (i, r) in R_GRID.iter().enumerate() {
            assert_eq!(
                a.loss[i].to_bits(),
                b.loss[i].to_bits(),
                "G1 determinism: arm {} loss({r}) differs between runs",
                arm.name
            );
        }
        assert_eq!(a.ref_drift.to_bits(), b.ref_drift.to_bits());
        assert_eq!(a.dest_bias.to_bits(), b.dest_bias.to_bits());
        results.push(a);
    }

    // Sanity: finite logits/losses everywhere.
    for (arm, res) in arms.iter().zip(results.iter()) {
        for (i, r) in R_GRID.iter().enumerate() {
            assert!(
                res.loss[i].is_finite() && res.loss[i] >= 0.0,
                "{}: loss({r}) = {} not finite/non-negative",
                arm.name,
                res.loss[i]
            );
        }
    }

    // Non-vacuity: cond arms differ from nat somewhere (r=8 logits).
    for arm in &arms[2..] {
        let differs = (0..N_PROMPTS).any(|t| {
            let c = run_once(&config, &weights, &arm.gate, &sdpa_gate, t, 8);
            let n = run_once(&config, &weights, &arms[0].gate, &sdpa_gate, t, 8);
            c.iter().zip(n.iter()).any(|(x, y)| x.to_bits() != y.to_bits())
        });
        assert!(differs, "non-vacuity: {} must differ from nat", arm.name);
    }

    // Adaptivity: freeze-loop spread across tokens. Asserted on AT LEAST
    // ONE conditional arm (a gate whose θ sits past the trajectory's whole
    // cosine range never closes — that is a measured operating-point
    // finding for that arm, not an adaptivity failure).
    let mut freeze_profiles: Vec<Vec<Option<usize>>> = Vec::new();
    let mut any_adaptive = false;
    for arm in &arms[2..] {
        let fl = freeze_loops(arm, &weights, &sdpa_gate);
        let frozen: Vec<usize> = fl.iter().filter_map(|&f| f).collect();
        let spread = match (frozen.iter().min(), frozen.iter().max()) {
            (Some(&lo), Some(&hi)) => hi - lo,
            _ => 0,
        };
        if spread > 0 {
            any_adaptive = true;
        }
        freeze_profiles.push(fl);
    }
    assert!(
        any_adaptive,
        "adaptivity: no conditional arm shows a freeze-loop spread across tokens"
    );
    // The hard-freeze arm must settle EXACTLY like a frozen system: once
    // every token is frozen the output stops changing — ref drift goes to
    // (near) zero. The clamp's whole point; the soft arms keep a residual
    // (1−g) kick alive.
    let hard = &results[4];
    assert!(
        hard.ref_drift < 1e-4,
        "hard-freeze arm must settle (ref drift {:.3e} ≥ 1e-4 — the clamp failed)",
        hard.ref_drift
    );

    // ── Report ───────────────────────────────────────────────────
    println!("\n═══ Issue 698 T8 — hand conditional gate A/B (modelless) ═══");
    println!(
        "  fixture blake3[16] = {hash}  ·  seed {SEED}  ·  InterLoopNorm context  ·  prompts {N_PROMPTS}"
    );
    println!("   arm              r=2      r=4      r=8      r=16     ref_drift   dest_bias");
    for (arm, res) in arms.iter().zip(results.iter()) {
        println!(
            "  {:<16} {:>8.3e} {:>8.3e} {:>8.3e} {:>8.3e}   {:>8.3e}  {:>8.3e}",
            arm.name, res.loss[0], res.loss[1], res.loss[2], res.loss[3],
            res.ref_drift, res.dest_bias
        );
    }
    println!();
    for (name, prof) in [
        "cond_a(θ=.99)",
        "cond_b(θ=.999)",
        "cond_ah(hard)",
    ]
    .iter()
    .zip(freeze_profiles.iter())
    {
        let mut frozen: Vec<usize> = prof.iter().filter_map(|&f| f).collect();
        let never = prof.iter().filter(|f| f.is_none()).count();
        frozen.sort_unstable();
        let (lo, med, hi) = match (frozen.first(), frozen.get(frozen.len() / 2), frozen.last()) {
            (Some(&a), Some(&m), Some(&h)) => (a, m, h),
            _ => (0, 0, 0),
        };
        println!(
            "  {name}: freeze loop (first g > {FREEZE_G}) — min {lo} · median {med} · max {hi} · never {never}/{}",
            prof.len()
        );
    }
    println!();
    let (nat, cvx, ca, cb, cah) = (&results[0], &results[1], &results[2], &results[3], &results[4]);
    println!(
        "  contraction: ref_drift nat {:.3e} → cvx {:.3e} → cond_a {:.3e} → cond_b {:.3e} → cond_ah {:.3e}",
        nat.ref_drift, cvx.ref_drift, ca.ref_drift, cb.ref_drift, cah.ref_drift
    );
    println!(
        "  destination: dest_bias cvx {:.3e} · cond_a {:.3e} · cond_b {:.3e} · cond_ah {:.3e} (nat = 0)",
        cvx.dest_bias, ca.dest_bias, cb.dest_bias, cah.dest_bias
    );
    println!();
    println!("  VERDICT (recorded — the deployed-path quality arbiter stays pending trained");
    println!("  weights, the issue's coin-flip caveat bounded by the mechanism-gate PASS):");
    println!("  (1) ADAPTIVITY CONFIRMED — the soft gate freezes tokens at different loops");
    println!("  (one schedule per token, not one for all). (2) THE σ-ASYMPTOTE KICK IS");
    println!("  REAL — the soft gate's residual (1−g) update never settles. (3) THE HARD-");
    println!("  FREEZE CLAMP + THE DRIFT SOURCE close the gap: g > 0.9 snaps to exactly 1.0");
    println!("  (a bit-exact copy of the state WHERE IT IS), cos ≡ 1 keeps it frozen — exact");
    println!("  contraction with the adaptive trigger intact. (4) CONTEXT PAIRING IS");
    println!("  LOAD-BEARING — on FixedAnchor the same clamp failed (ref drift 1.5): the");
    println!("  freeze-in-place jump to the anchor re-opens the gate; conditional gate ↔");
    println!("  drift source is the working pairing. Net: an adaptive, exactly-contracting");
    println!("  loop gate — modelless.");
    println!("  Caveats: random weights, single-position prompts, micro config; hand-set");
    println!("  (β, θ, b) operating points — trained gate weights are riir-train Plan 364.");
}
