#![cfg(all(
    feature = "lt2_looped",
    feature = "loop_stability_fix",
    feature = "gain_cost_halt"
))]
//! Issue 698 T4 — halter floors, e2e through the production loop path.
//!
//! T4's kernel content (the `grt_default` `l_min = 2` floor, the opt-in
//! concavity floor with `HaltReason::NonContraction`, the noise-floor guard,
//! and the oscillation fallback precedence) is pinned by unit tests inside
//! `crates/katgpt-core/src/gain_cost_halt.rs`. This bench proves the COMPOSED
//! behavior on the measured T1 fixture (micro + seed-42 + InterLoopNorm,
//! fixture `fab06e3f4ba65977`) — and it is where T4's GRT-transfer premise
//! was **REFUTED**:
//!
//! # The refutation (measured 2026-08-30, probe numbers committed)
//!
//! The paper's exit-quality law assumes the loop signal DECAYS as the state
//! converges (contracting updates → the concavity floor detects expansion).
//! On this fixture the production gain signal — step size ‖Δh‖ under
//! `InterLoopNorm` — **GROWS to its fixed point instead**: 13.010 (loop 2) →
//! 17.174 → 19.604 → 19.621 → 19.997 → 20.349 → 20.361 (loop 8), then
//! plateaus at 20.2415 ± 4e-4 with cos θ = 1.0000 for the rest of the run.
//! `InterLoopNorm` makes the convergence DIRECTIONAL (cos θ → 1.0 and T1's
//! KL → 1e-8 both prove convergence), while the step magnitude tracks the
//! fixed point's residual update — six consecutive above-floor inversions on
//! the ramp alone. A concavity floor armed at patience 2 therefore halts at
//! loop 4 — cutting the run exactly at the two-phase knee T1 measured.
//!
//! # What is asserted (all bit-level)
//!
//! 1. **G1** — the halter path is deterministic (double-run bit-identical).
//! 2. **Disarmed no-traction verdict** — `grt_default` (concavity disarmed)
//!    NEVER halts on this fixture and its output is bit-identical to the
//!    no-halter baseline: the scissors cannot fire (cost = 1% of step-2 =
//!    0.130 ≪ the 20.24 plateau), the oscillation detector cannot fire
//!    (cos θ ≡ 1.0), and the concavity floor is off. The halter is a no-op
//!    on the modelless fixture — recorded, not assumed.
//! 3. **Armed-floor refutation pinned** — the concavity floor armed at
//!    patience 2 fires on the step ramp and cuts the run mid-ramp (the halt
//!    loop is PER-TOKEN — each token's trajectory ramps differently; the
//!    measured distribution is asserted floor-compliant and printed). The
//!    halt ≡ elastic equivalence holds per token (a halted state is
//!    bit-identical to the same-token `elastic_loop_override = r` run). **The floor must stay DISARMED on the InterLoopNorm
//!    step-size axis** — re-arming requires a decay-capable signal.
//! 4. **Never-halt control** — `l_min = 255` is bit-identical to the
//!    baseline (the Open-Question-3 guarantee, re-proven on this fixture).
//!
//! # Run
//!
//! ```bash
//! cargo test --features gain_cost_halt --test issue_698_t4_halter_floors -- --nocapture
//! ```

use katgpt_core::gain_cost_halt::GainCostLoopHalter;
use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng, SdpaOutputGate,
};

// ── Constants (T1's fixture, verbatim) ───────────────────────

/// Loop count of the fixed-point reference output.
const R_REF: usize = 32;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches the 407 / T1 convention).
const SEED: u64 = 42;

/// The pinned fixture identity — must equal one of T1's committed platform
/// hashes (aarch64 M3 + x86_64-windows; the T7-recorded weight-byte drift),
/// proving this bench measures the exact spectrum T1 recorded.
const EXPECTED_FIXTURE_HASH: [&str; 2] = ["fab06e3f4ba65977", "c894478d3febdb00"];

/// The measured per-token armed-halt distribution on this fixture:
/// every token halts in [4, 6] — 21 at loop 4, 6 at loop 6 — the ramp's
/// second consecutive above-floor inversion. A drift here means the
/// fixture or the rule changed; the load-bearing structural pins are:
/// every halt ≥ l_min = 2, every token halts early, disarmed == baseline.
const ARMED_HALT_RANGE: (usize, usize) = (4, 6);

// ── Fixture ──────────────────────────────────────────────────────

fn make_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config.loop_stability_mode = LoopStabilityMode::InterLoopNorm;
    config
}

/// Deterministic BLAKE3 over every active f32 weight slice (T1's fixture
/// hash, verbatim).
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

// ── Forward paths ────────────────────────────────────────────────

/// The T1 measurement path: elastic override runs exactly `loops` iterations,
/// no halter.
fn run_elastic(
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

/// The halter path: NO elastic override (T2.2 — a static override would
/// disable the halter), config loop_count = R_REF, halter decides.
fn run_halted(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    halter: &mut GainCostLoopHalter,
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
        None,
        #[cfg(feature = "gain_cost_halt")]
        Some(&mut *halter),
        None, // Issue 717: deep_run — None = bit-identical baseline
        #[cfg(feature = "cadence_gate")]
        None, // Issue 731: residual-exit probe — None = bit-identical baseline
    );
    logits.to_vec()
}

/// Assert two logit sets are bit-identical across every token and coordinate.
fn assert_bits_identical(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: token count differs");
    for t in 0..a.len() {
        assert_eq!(a[t].len(), b[t].len(), "{what}: token {t} length differs");
        for (i, (x, y)) in a[t].iter().zip(b[t].iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: token {t} logits[{i}] differs"
            );
        }
    }
}

// ── The gate ─────────────────────────────────────────────────────

#[test]
fn t698_t4_halter_floors_e2e() {
    let config = make_config();
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let hash = fixture_hash(&config, &weights);
    println!("fixture hash (blake3[16]): {hash}");

    // Cross-bench identity: this IS T1's fixture.
    assert!(
        EXPECTED_FIXTURE_HASH.contains(&hash.as_str()),
        "fixture drifted from T1's committed spectrum fixture"
    );

    // Reference logits at R_REF (elastic Some(32), no halter).
    let ref_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_elastic(&config, &weights, &residual_gate, &sdpa_gate, t, R_REF))
        .collect();

    // ── Arm A: grt_default (l_min = 2, concavity DISARMED) ───────
    let run_arm_a = || -> Vec<Vec<f32>> {
        (0..N_PROMPTS)
            .map(|t| {
                let mut h = GainCostLoopHalter::grt_default();
                run_halted(&config, &weights, &residual_gate, &sdpa_gate, t, &mut h)
            })
            .collect()
    };
    let a1 = run_arm_a();
    let a2 = run_arm_a();

    // G1: double-run bit-identity.
    assert_bits_identical(&a1, &a2, "G1: arm A double-run");

    // The disarmed no-traction verdict: no halt signal can fire on this
    // fixture (scissors: cost 0.130 ≪ the 20.24 plateau step; oscillation:
    // cos θ ≡ 1.0; concavity: disarmed) — so the output is bit-identical to
    // the no-halter baseline. This is the measured "keep the floor DISARMED
    // here" premise, and it doubles as the never-halts-spuriously check.
    assert_bits_identical(&a1, &ref_logits, "arm A (disarmed) vs baseline");

    // ── Arm B: concavity floor ARMED at patience 2 — the REFUTATION ──
    let b: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| {
            let mut h = GainCostLoopHalter::grt_default().with_concavity_floor(2);
            run_halted(&config, &weights, &residual_gate, &sdpa_gate, t, &mut h)
        })
        .collect();

    // The armed floor must differ from the disarmed run — it fires on the
    // step ramp. (If this equality ever returns, the ramp disappeared and
    // the refutation no longer holds — re-investigate the fixture.)
    let mut b_fired = false;
    'outer: for t in 0..N_PROMPTS {
        for (x, y) in a1[t].iter().zip(b[t].iter()) {
            if x.to_bits() != y.to_bits() {
                b_fired = true;
                break 'outer;
            }
        }
    }
    assert!(
        b_fired,
        "the armed floor no longer fires — the step ramp is gone; \
        the refutation record below is stale"
    );

    // Halt ≡ elastic, PER TOKEN: locate each token's halt loop from its own
    // elastic ladder (the halt decision is per-token — each token's hidden
    // trajectory ramps differently), verifying bit-equality at the match.
    let mut halt_loops = Vec::with_capacity(N_PROMPTS);
    for (t, h) in b.iter().enumerate() {
        let mut r_t = None;
        for r in 1..=R_REF {
            let e = run_elastic(&config, &weights, &residual_gate, &sdpa_gate, t, r);
            if e.len() == h.len()
                && e.iter()
                    .zip(h.iter())
                    .all(|(a, c)| a.to_bits() == c.to_bits())
            {
                r_t = Some(r);
                break;
            }
        }
        let r_t = r_t.unwrap_or_else(|| {
            panic!("halt≡elastic broke: token {t} matches no elastic loop 1..={R_REF}")
        });
        halt_loops.push(r_t);
    }

    // Floor compliance at EVERY halt: r_t ≥ l_min = 2 (structurally the
    // wiring skips the eval at 1-based loop 1; the kernel RefusedFloor
    // contract covers the adversarial loop-1 case). At least one token must
    // have halted strictly early — the ramp is real.
    assert!(
        halt_loops.iter().all(|&r| r >= 2),
        "a halt violated the l_min = 2 floor: {halt_loops:?}"
    );
    assert!(
        halt_loops.iter().any(|&r| r < R_REF),
        "no token halted early — the refutation record is stale"
    );
    // Content pin: the measured halt distribution is tight — every token
    // cuts at the ramp's second consecutive inversion (loop 4) or one
    // blip later (loop 6). A drift here is a fixture/rule change.
    let r_min = *halt_loops.iter().min().unwrap();
    let r_max = *halt_loops.iter().max().unwrap();
    assert_eq!(
        (r_min, r_max),
        ARMED_HALT_RANGE,
        "armed halt-loop distribution drifted: {halt_loops:?} (was 21×4 + 6×6)"
    );

    // ── Arm C: never-halt control (l_min = 255) ──────────────────
    let c: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| {
            let mut h = GainCostLoopHalter::new(1.0, 1, 255);
            run_halted(&config, &weights, &residual_gate, &sdpa_gate, t, &mut h)
        })
        .collect();
    assert_bits_identical(&ref_logits, &c, "never-halt control vs baseline");

    // ── Verdict record ────────────────────────────────────────
    let n_early = halt_loops.iter().filter(|&&r| r < R_REF).count();
    println!("\n═══ Issue 698 T4 — halter floors e2e (fixture {hash}) ═══");
    println!("  REFUTATION: the step-size gain axis GROWS to the fixed point under");
    println!("  InterLoopNorm (13.01 → 20.36 over loops 2–8, plateau 20.2415 ± 4e-4,");
    println!("  cos θ = 1.0000) — convergence is DIRECTIONAL, not contractive.");
    println!("  disarmed grt_default: never halts on this fixture == baseline (no traction:");
    println!("    cost 0.130 ≪ plateau 20.24; cos θ ≡ 1.0) → keep the floor DISARMED here");
    println!("  armed concavity floor (patience=2): {n_early}/{N_PROMPTS} tokens halted early,");
    println!("    halt loops min={r_min} max={r_max} (per-token: {halt_loops:?}) — the two-phase");
    println!("    knee T1 measured, cut per-token by the ramp's consecutive inversions");
    println!("  never-halt control (l_min=255): bit-identical to baseline ✓");
    println!("  VERDICT: the GRT concavity floor does NOT transfer to the InterLoopNorm");
    println!("  step-size axis. Re-arm only with a decay-capable signal (KL / erank");
    println!("  quality gain) — a companion wiring question, not a floor change.");
}
