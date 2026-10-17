//! Plan 617 T3.2–T3.4 — the LoopCD GOAT gate: the adjudication of
//! recurrent-depth contrast guidance against the shipped incumbents on the
//! frozen non-saturated fixture.
//!
//! **Ordering provenance (T3.1's frozen-fixture law):** the fixture and its
//! non-degeneracy assertions were committed (`tests/loopcd_fixture_gate.rs`,
//! commit `6c4b5b5f7`) BEFORE any guided arm ran. This bench consumes the
//! frozen fixture unchanged.
//!
//! Metric (T3.3, pinned): **greedy argmax** — no sampled metric, so no
//! unguided temperature front is required.
//!
//! G1 quality ladder (pre-registered posture: `default_adaptive_config` —
//! adaptive Eq-4 gate, ω_max 0.5, logit mode, ref_loop 1):
//! - (a) guided-full ≥ unguided-full;
//! - (b) **guided-half ≥ unguided-full** (the headline);
//! - (c) on the close-decision stratum (|margin(R)| < 0.5, the fixture
//!   gate's measured 19 cases): guided > ALL THREE incumbents —
//!   `early-tap` (the 865 lane: weak side = a masked readout of the SAME
//!   final state), `product-policy` (positive blend), `exit-only`
//!   (AdvantageMarginGate);
//! - (d) the cheap-weak control: a masked-logit weak side under the SAME
//!   combine must NOT match the depth arm — if it does, the
//!   alignment-source story is refuted on this fixture.
//!
//! Incumbent fairness: every combine arm uses the SAME adaptive Eq-4 gate
//! (`ω = ω_max·(1 − margin(z_R))`) on the pre-guidance readout — the
//! incumbents lose (or win) on their weak side's alignment source, never on
//! a weaker strength schedule. `product_policy_log` has no ω analog; it
//! runs at w = 0.5, disclosed.
//!
//! G2 (T3.4): interleaved median-of-ratios A/B (the shared `ab_timing`
//! harness — never sequential arms) for logit-mode and hidden-mode overhead;
//! net-FLOP accounting including ALL per-loop settle readouts; the
//! loop-reduction fraction for the settle-exit fusion.
//!
//! G3: flag-off byte-identity re-pinned here; the incumbents' own tests
//! (`issue_731_t1_residual_exit`, `issue_698_t4_halter_floors`,
//! `probe_guidance_goat`) are run at this feature set and green.
//! G4: from `loop_guidance_alloc_gate` (settle adds zero, both policies).
//!
//! Run: `cargo test --release -p katgpt-rs --features
//! loop_guidance,lt2_looped,weight_shared_advantage_gate --test
//! loop_guidance_goat -- --nocapture`

#![cfg(all(
    feature = "loop_guidance",
    feature = "lt2_looped",
    feature = "weight_shared_advantage_gate"
))]

#[path = "common/loopcd_fixture.rs"]
mod loopcd_fixture;

use katgpt_rs::transformer::loop_guidance::{
    GuidanceMode, LoopGuidance, LoopGuidanceConfig, SettleExit, SettleExitConfig, adaptive_omega,
};
use katgpt_rs::transformer::standard_lm_head;
use loopcd_fixture::{LoopCdCase, LoopCdFixture, MAX_LOOPS, all_cases, answer_margin, argmax};

/// Half depth for the headline leg (b).
const HALF_LOOPS: usize = MAX_LOOPS / 2;
/// Close-decision stratum threshold — same as the fixture gate's.
const CLOSE_MARGIN_EPS: f32 = 0.5;
/// Incumbents' ω_max for the shared Eq-4 gate.
const INCUMBENT_OMEGA_MAX: f32 = 0.5;

// ── arm machinery ────────────────────────────────────────────────────────

/// The depth-contrast arm through the real `forward_looped` path.
fn guided_logits(
    fixture: &LoopCdFixture,
    case: &LoopCdCase,
    depth: usize,
    mode: GuidanceMode,
    omega_max: f32,
    ref_loop: usize,
) -> Vec<f32> {
    let cfg = LoopGuidanceConfig {
        mode,
        omega: omega_max.min(0.25),
        omega_max,
        adaptive: true,
        ref_loop,
    };
    let mut g = LoopGuidance::new(cfg, loopcd_fixture::VOCAB, fixture.config.n_embd);
    fixture.run_tokens_guided(&case.prefix, depth, Some(&mut g), None)
}

/// Deterministic mask for the perturbation arms (fixed pattern, no RNG —
/// the global-RNG law). Zeroes odd coordinates of the 4-dim state / logit
/// row: [1, 0, 1, 0].
fn apply_fixed_mask(v: &mut [f32]) {
    for (i, x) in v.iter_mut().enumerate() {
        if i % 2 == 1 {
            *x = 0.0;
        }
    }
}

/// The Eq-4 adaptive gate shared by every combine arm:
/// `ω = ω_max·(1 − margin(z_R))` on the pre-guidance readout.
fn shared_omega(z_r: &[f32]) -> f32 {
    adaptive_omega(
        INCUMBENT_OMEGA_MAX,
        katgpt_rs::transformer::loop_guidance::top_two_margin(z_r),
    )
}

/// Combine arms (incumbents + controls): `z′ = z_R + ω·(z_R − z_weak)` —
/// the same affine family the depth arm uses, via T1.2's ONE home (the
/// shared kernel; no re-derivation).
fn combine(z_r: &[f32], z_weak: &[f32], omega: f32) -> Vec<f32> {
    let mut out = z_r.to_vec();
    katgpt_core::contrast_combine::affine_combine(&mut out, z_weak, 1.0 + omega);
    out
}

/// Early-tap incumbent (the 865 lane operationalized): weak side = a masked
/// readout of the SAME final state — `z_weak = head(mask(h_R))`.
fn early_tap_logits(fixture: &LoopCdFixture, case: &LoopCdCase, depth: usize) -> Vec<f32> {
    let (z_r, h_r) = fixture.run_tokens_with_hidden(&case.prefix, depth);
    let n = fixture.config.n_embd;
    let mut h_masked = h_r;
    apply_fixed_mask(&mut h_masked);
    let mut z_weak = vec![0.0f32; loopcd_fixture::VOCAB];
    standard_lm_head(
        &mut z_weak,
        &h_masked,
        &fixture.weights.lm_head,
        loopcd_fixture::VOCAB,
        n,
    );
    let omega = shared_omega(&z_r);
    combine(&z_r, &z_weak, omega)
}

/// Cheap-weak control: weak side = the masked LOGIT row itself
/// (`z_weak = mask(z_R)`) — "any contrast" without a depth contrast.
fn cheap_weak_logits(fixture: &LoopCdFixture, case: &LoopCdCase, depth: usize) -> Vec<f32> {
    let z_r = fixture.run_case(case, depth);
    let mut z_weak = z_r.clone();
    apply_fixed_mask(&mut z_weak);
    let omega = shared_omega(&z_r);
    combine(&z_r, &z_weak, omega)
}

/// Product-policy incumbent: `0.5·log π̂(z_k) + 0.5·log π(z_R)` — the
/// positive blend, weak side = the depth-`ref_loop` readout.
fn product_policy_logits(
    fixture: &LoopCdFixture,
    case: &LoopCdCase,
    depth: usize,
    ref_loop: usize,
) -> Vec<f32> {
    let z_k = fixture.run_case(case, ref_loop);
    let z_r = fixture.run_case(case, depth);
    let mut out = vec![0.0f32; z_r.len()];
    katgpt_rs::pruners::self_advantage::product_policy_log(&z_k, &z_r, 0.5, &mut out);
    out
}

/// Settle-exit fusion arm: guided + settle (the aggressive-but-windowed
/// config from the settle gates) — returns (answer, loops_used, fired).
fn settle_fusion_arm(fixture: &LoopCdFixture, case: &LoopCdCase) -> (usize, usize, bool) {
    let mut g = LoopGuidance::new(
        LoopGuidanceConfig {
            mode: GuidanceMode::Logits,
            omega: 0.25,
            omega_max: 0.5,
            adaptive: true,
            ref_loop: 1,
        },
        loopcd_fixture::VOCAB,
        fixture.config.n_embd,
    );
    let mut settle = SettleExit::new(
        SettleExitConfig {
            enabled: true,
            d_min: 3,
            settle_patience: 2,
            readout: katgpt_rs::transformer::loop_guidance::SettleReadout::Full,
        },
        loopcd_fixture::VOCAB,
        fixture.config.n_embd,
    );
    let logits =
        fixture.run_tokens_guided(&case.prefix, MAX_LOOPS, Some(&mut g), Some(&mut settle));
    let budget = settle.last_budget().expect("budget");
    (
        argmax(&logits),
        budget.loops_used,
        matches!(
            budget.exit,
            katgpt_rs::transformer::loop_guidance::ExitKind::Settled { .. }
        ),
    )
}

// ── the gate ─────────────────────────────────────────────────────────────

#[test]
fn loopcd_goat_g1_ladder() {
    let fixture = LoopCdFixture::new();
    let cases = all_cases();

    // Close stratum (from the UNGUIDED full-depth margin — the fixture
    // gate's definition).
    let close: Vec<&LoopCdCase> = cases
        .iter()
        .filter(|c| answer_margin(&fixture.run_case(c, MAX_LOOPS)).abs() < CLOSE_MARGIN_EPS)
        .collect();
    assert!(!close.is_empty(), "close stratum empty — the fixture moved");

    let acc = |answers: &[usize], pop: &[&LoopCdCase]| -> f64 {
        pop.iter()
            .zip(answers)
            .filter(|(c, a)| **a == c.answer)
            .count() as f64
            / pop.len() as f64
    };

    // ── full-population arms at full depth ──
    let unguided_full: Vec<usize> = cases
        .iter()
        .map(|c| argmax(&fixture.run_case(c, MAX_LOOPS)))
        .collect();
    let guided_full: Vec<usize> = cases
        .iter()
        .map(|c| {
            argmax(&guided_logits(
                &fixture,
                c,
                MAX_LOOPS,
                GuidanceMode::Logits,
                0.5,
                1,
            ))
        })
        .collect();
    let a_unguided = acc(&unguided_full, &cases.iter().collect::<Vec<_>>());
    let a_guided = acc(&guided_full, &cases.iter().collect::<Vec<_>>());

    // ── half depth ──
    let unguided_half: Vec<usize> = cases
        .iter()
        .map(|c| argmax(&fixture.run_case(c, HALF_LOOPS)))
        .collect();
    let guided_half: Vec<usize> = cases
        .iter()
        .map(|c| {
            argmax(&guided_logits(
                &fixture,
                c,
                HALF_LOOPS,
                GuidanceMode::Logits,
                0.5,
                1,
            ))
        })
        .collect();
    let a_unguided_half = acc(&unguided_half, &cases.iter().collect::<Vec<_>>());
    let a_guided_half = acc(&guided_half, &cases.iter().collect::<Vec<_>>());

    // ── incumbents + control at full depth (close stratum + full pop) ──
    let early_tap: Vec<usize> = cases
        .iter()
        .map(|c| argmax(&early_tap_logits(&fixture, c, MAX_LOOPS)))
        .collect();
    let product: Vec<usize> = cases
        .iter()
        .map(|c| argmax(&product_policy_logits(&fixture, c, MAX_LOOPS, 1)))
        .collect();
    let cheap_weak: Vec<usize> = cases
        .iter()
        .map(|c| argmax(&cheap_weak_logits(&fixture, c, MAX_LOOPS)))
        .collect();

    let close_idx: Vec<usize> = cases
        .iter()
        .enumerate()
        .filter(|(_, c)| answer_margin(&fixture.run_case(c, MAX_LOOPS)).abs() < CLOSE_MARGIN_EPS)
        .map(|(i, _)| i)
        .collect();

    let acc_idx = |answers: &[usize]| -> f64 {
        close_idx
            .iter()
            .filter(|&&i| answers[i] == cases[i].answer)
            .count() as f64
            / close_idx.len() as f64
    };

    let close_unguided = acc_idx(&unguided_full);
    let close_guided = acc_idx(&guided_full);
    let close_early = acc_idx(&early_tap);
    let close_product = acc_idx(&product);
    let close_cheap = acc_idx(&cheap_weak);

    println!("== LoopCD GOAT G1 ladder (greedy argmax) ==");
    println!(
        "full pop  (n={}): unguided {:.4} | guided {:.4} | early-tap {:.4} | product {:.4} | cheap-weak {:.4}",
        cases.len(),
        a_unguided,
        a_guided,
        acc(&early_tap, &cases.iter().collect::<Vec<_>>()),
        acc(&product, &cases.iter().collect::<Vec<_>>()),
        acc(&cheap_weak, &cases.iter().collect::<Vec<_>>()),
    );
    println!(
        "depths    : unguided-half({HALF_LOOPS}) {:.4} | guided-half {:.4}",
        a_unguided_half, a_guided_half
    );
    println!(
        "close (n={}): unguided {:.4} | guided {:.4} | early-tap {:.4} | product {:.4} | cheap-weak {:.4}",
        close_idx.len(),
        close_unguided,
        close_guided,
        close_early,
        close_product,
        close_cheap,
    );

    // ── THE ADJUDICATION VERDICT, pinned as the measured negative (the
    // Bench-847/850 inverted-bar pattern). The plan's G1 ladder asked
    // whether guided ≥ unguided and guided > incumbents on a NON-SATURATED
    // fixture; the measured answer on THIS fixture at the pre-registered
    // posture is NO on every leg. The asserts pin the measured verdict with
    // its numbers; a flip in either direction reds this gate and forces a
    // re-adjudication (T4.1).
    //
    // Mechanism (measured, not speculative): the depth-1 reference z_k on
    // this fixture is dominated by the head-BIAS axis (the bias gap decays
    // geometrically — the fixture's own crossing dynamic), so the contrast
    // direction z_R − z_k is mostly BIAS REMOVAL, not decision signal. The
    // combine faithfully amplifies it and flips close B-correct cases — the
    // paper's disagreement-tracking story presumes the k→R difference is
    // decision-dominated, which a biased-head fixture (like any real
    // checkpoint with head bias) violates. The posture sweep test below
    // measures whether a burn-in reference recovers the mechanism.

    // (a) measured: guided-full LOSES to unguided-full.
    assert!(
        a_guided <= a_unguided,
        "VERDICT FLIP (a): guided-full {a_guided:.4} now BEATS unguided-full \
         {a_unguided:.4} at the pre-registered posture — the negative verdict \
         has moved; re-run the adjudication and re-pin T4.1"
    );
    // (b) measured: guided-half does not reach unguided-full.
    assert!(
        a_guided_half <= a_unguided,
        "VERDICT FLIP (b): guided-half {a_guided_half:.4} now REACHES unguided-full \
         {a_unguided:.4} — the headline claim now holds; re-adjudicate"
    );
    // (c) measured: guided loses to early-tap (neutral on this fixture) on
    // the close stratum. The product-policy incumbent is ALSO destructive
    // (w=0.5 blends the biased weak side in policy space) — guided beats it;
    // both directions pinned as measured.
    assert!(
        close_guided <= close_early,
        "VERDICT FLIP (c-early): guided {close_guided:.4} now BEATS early-tap \
         {close_early:.4} on the close stratum — re-adjudicate"
    );
    assert!(
        close_guided >= close_product,
        "VERDICT FLIP (c-product): guided {close_guided:.4} now LOSES to \
         product-policy {close_product:.4} — re-adjudicate"
    );
    // (d) measured: the cheap-weak control is NEUTRAL (1.0) and beats the
    // depth arm — 'any contrast' is not the story here; the DEPTH contrast
    // is the destructive one (the alignment-source story is REFUTED in the
    // harmful direction: the depth-sourced contrast damages where the
    // state-sourced one does not).
    assert!(
        close_guided <= close_cheap,
        "VERDICT FLIP (d): guided {close_guided:.4} now BEATS the cheap-weak \
         control {close_cheap:.4} — the depth contrast became the better \
         alignment source; re-adjudicate"
    );
}

/// The exit-only incumbent: AdvantageMarginGate armed on `forward_looped`.
/// Measures accuracy parity and loop savings (its plan-recorded shape:
/// skip-settled-compute at argmax parity).
#[test]
fn loopcd_goat_exit_only_incumbent() {
    use katgpt_rs::hla::MultiLayerAhlaCache;
    use katgpt_rs::pruners::self_advantage::AdvantageMarginGate;
    use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, forward_looped};
    use katgpt_rs::types::{ResidualGate, SdpaOutputGate};

    let fixture = LoopCdFixture::new();
    let cases = all_cases();
    let mut answers: Vec<usize> = Vec::with_capacity(cases.len());
    let config = &fixture.config;

    for case in &cases {
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let mut ahla_cache = MultiLayerAhlaCache::new(config);
        let residual_gate = ResidualGate::new(MAX_LOOPS, config.n_embd);
        let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
        let mut gate = AdvantageMarginGate::default();
        for (p, &tok) in case.prefix.iter().enumerate() {
            let l = forward_looped(
                &mut ctx,
                &fixture.weights,
                &mut cache,
                &mut ahla_cache,
                tok,
                p,
                config,
                &residual_gate,
                &sdpa_gate,
                None,
                None,
                #[cfg(feature = "weight_shared_advantage_gate")]
                Some(&mut gate),
                Some(1),
                #[cfg(feature = "gain_cost_halt")]
                None,
                None,
                #[cfg(feature = "cadence_gate")]
                None,
                #[cfg(feature = "loop_guidance")]
                None,
                #[cfg(feature = "loop_guidance")]
                None,
            );
            let _ = l;
        }
        let l = forward_looped(
            &mut ctx,
            &fixture.weights,
            &mut cache,
            &mut ahla_cache,
            loopcd_fixture::QUERY,
            case.prefix.len(),
            config,
            &residual_gate,
            &sdpa_gate,
            None,
            None,
            #[cfg(feature = "weight_shared_advantage_gate")]
            Some(&mut gate),
            Some(MAX_LOOPS),
            #[cfg(feature = "gain_cost_halt")]
            None,
            None,
            #[cfg(feature = "cadence_gate")]
            None,
            #[cfg(feature = "loop_guidance")]
            None,
            #[cfg(feature = "loop_guidance")]
            None,
        );
        let logits = l.to_vec();
        answers.push(argmax(&logits));
    }
    let acc = answers
        .iter()
        .zip(cases.iter())
        .filter(|(a, c)| **a == c.answer)
        .count() as f64
        / cases.len() as f64;
    let close_idx: Vec<usize> = cases
        .iter()
        .enumerate()
        .filter(|(_, c)| answer_margin(&fixture.run_case(c, MAX_LOOPS)).abs() < CLOSE_MARGIN_EPS)
        .map(|(i, _)| i)
        .collect();
    let close_acc = close_idx
        .iter()
        .filter(|&&i| answers[i] == cases[i].answer)
        .count() as f64
        / close_idx.len() as f64;
    let unguided = cases
        .iter()
        .filter(|c| argmax(&fixture.run_case(c, MAX_LOOPS)) == c.answer)
        .count() as f64
        / cases.len() as f64;
    println!(
        "== exit-only incumbent (AdvantageMarginGate @ default 0.01): full {acc:.4} \
         | close {close_acc:.4} vs unguided {unguided:.4} =="
    );
    // MEASURED, not asserted-at: the gate's 100%-parity claim was proven on
    // ITS fixtures (vocab ≤ 128, Plan 283); on THIS fixture the margin gate
    // fires inside the stable-wrong window (the candidate-improvement
    // signal is tiny pre-crossing) and exits before the crossing — the
    // demote-the-loser data T3.3(c) reads. Pinned as the measured verdict:
    // if the incumbent ever EXCEEDS unguided here, the fixture or the gate
    // moved — re-adjudicate.
    assert!(
        acc <= unguided + 1e-9,
        "exit-only incumbent measured ABOVE unguided on this fixture — \
         re-adjudicate the G1 ladder"
    );
}

/// G2 (T3.4): interleaved overhead A/B for both guidance modes + the net
/// settle-exit accounting + the loop-reduction fraction.
#[test]
fn loopcd_goat_g2_overhead_and_budget() {
    #[path = "common/ab_timing.rs"]
    mod ab_timing;

    let fixture = LoopCdFixture::new();
    let cases = all_cases();
    // A fixed slice for the timing arms — a dozen cases spread across the
    // strata; the timing measures the STEP cost, not the accuracy.
    let timing_cases: Vec<&LoopCdCase> = cases
        .iter()
        .step_by(cases.len() / 12 + 1)
        .take(12)
        .collect();

    let run_unguided = |i: usize| {
        let case = &timing_cases[i % timing_cases.len()];
        let d = 1 + i % MAX_LOOPS;
        let l = fixture.run_case(case, d);
        std::hint::black_box(l[loopcd_fixture::ANS_A]);
    };
    let run_guided_logit = |i: usize| {
        let case = &timing_cases[i % timing_cases.len()];
        let d = 1 + i % MAX_LOOPS;
        let l = guided_logits(&fixture, case, d, GuidanceMode::Logits, 0.5, 1);
        std::hint::black_box(l[loopcd_fixture::ANS_A]);
    };
    let run_guided_hidden = |i: usize| {
        let case = &timing_cases[i % timing_cases.len()];
        let d = 1 + i % MAX_LOOPS;
        let l = guided_logits(&fixture, case, d, GuidanceMode::Hidden, 0.5, 1);
        std::hint::black_box(l[loopcd_fixture::ANS_A]);
    };

    let logit_ab = ab_timing::ab_median_ratio(24, 8, 8, run_unguided, run_guided_logit);
    logit_ab.report("logit-mode guided / unguided");
    let hidden_ab = ab_timing::ab_median_ratio(24, 8, 8, run_unguided, run_guided_hidden);
    hidden_ab.report("hidden-mode guided / unguided");

    // Hidden mode ≤ +5% step time (the plan's G2 bar for the one-head-pass
    // law). Logit mode is ALLOWED up to 2 head passes — assert only that it
    // is bounded (≤ +60%: one extra head pass of five on a 5-row vocab plus
    // margin work; the head is ~1/3 of the loop block at this scale).
    assert!(
        hidden_ab.median <= 1.05,
        "G2 FAILED: hidden-mode overhead {:.1}% exceeds the +5% bar",
        hidden_ab.overhead_pct()
    );
    assert!(
        logit_ab.median <= 1.60,
        "G2 FAILED: logit-mode overhead {:.1}% exceeds the two-head-pass bound",
        logit_ab.overhead_pct()
    );

    // ── net-FLOP accounting (printed; the half-depth claim reads this) ──
    let n = fixture.config.n_embd;
    let v = loopcd_fixture::VOCAB;
    // Per loop iteration: one block pass. At the fixture dims the layer is
    // ~2·(3·n·n + n·n + 2·mlp·n + n·mlp) ≈ 2·(4n² + 4n·mlp_hidden) FLOPs...
    let mlp = fixture.config.mlp_hidden;
    let block_flops = 2.0 * (4.0 * (n * n) as f64 + 4.0 * (n * mlp) as f64);
    let head_flops = 2.0 * (v * n) as f64;
    let unguided_total = MAX_LOOPS as f64 * (block_flops + head_flops);
    // Guided logit mode: +1 scratch head pass at ref_loop.
    let guided_total = unguided_total + head_flops;
    println!(
        "net-FLOPs/query-step: unguided {:.0} | guided-logit {:.0} (+{:.1}%) | head pass = {:.1}% of a loop block",
        unguided_total,
        guided_total,
        100.0 * head_flops / unguided_total,
        100.0 * head_flops / block_flops
    );
    // At a REAL vocabulary the head pass dominates — the honest disclosure
    // the plan demands (the paper's 1.01–1.31× is one extra pass, not R):
    let real_v = 128_000.0;
    let real_head = 2.0 * real_v * n as f64;
    println!(
        "at V=128k the scratch head pass would be {:.0}% of the loop block — \
         the bounded candidate-set readout (top-m refresh) is the mitigation \
         measured by the settle-exit accounting below",
        100.0 * real_head / block_flops
    );

    // ── settle-exit fusion: loop-reduction fraction + guided accuracy
    // parity at the fused posture ──
    let mut loops_used = 0usize;
    let mut fired = 0usize;
    for case in &cases {
        let (_, used, did) = settle_fusion_arm(&fixture, case);
        loops_used += used;
        fired += usize::from(did);
    }
    let reduction = 1.0 - loops_used as f64 / (cases.len() * MAX_LOOPS) as f64;
    println!(
        "settle-exit fusion (d_min=3, p=2, Full readout): fired {fired}/{} \
         cases, loop reduction {:.1}% (readout FLOPs included in the loop \
         side: {} full head passes)",
        cases.len(),
        100.0 * reduction,
        loops_used - 2, // d_min=3 → the first readout is iteration 3
    );
}

/// The posture sweep (disclosure, T3.4's hyperparameter honesty): does ANY
/// (mode, ref_loop burn-in, ω_max) posture beat unguided? The paper's own
/// h6/h7 burn-in subtlety says ref_loop matters — on this fixture the
/// depth-1 reference is dominated by the head-bias axis (the measured
/// bias-removal flips), and a later reference should be post-bias. The
/// pre-registered G1 ladder above stays the adjudication regardless of
/// what this table shows: a sweep-selected win is a hyperparameter
/// selection, recorded as such, never promoted.
#[test]
fn loopcd_goat_posture_sweep_disclosure() {
    let fixture = LoopCdFixture::new();
    let cases = all_cases();
    let close_idx: Vec<usize> = cases
        .iter()
        .enumerate()
        .filter(|(_, c)| answer_margin(&fixture.run_case(c, MAX_LOOPS)).abs() < CLOSE_MARGIN_EPS)
        .map(|(i, _)| i)
        .collect();
    let unguided_full: Vec<usize> = cases
        .iter()
        .map(|c| argmax(&fixture.run_case(c, MAX_LOOPS)))
        .collect();
    let acc_all = |answers: &[usize]| -> f64 {
        answers
            .iter()
            .zip(cases.iter())
            .filter(|(a, c)| **a == c.answer)
            .count() as f64
            / cases.len() as f64
    };
    let acc_close = |answers: &[usize]| -> f64 {
        close_idx
            .iter()
            .filter(|&&i| answers[i] == cases[i].answer)
            .count() as f64
            / close_idx.len() as f64
    };
    let a0 = acc_all(&unguided_full);
    let a0c = acc_close(&unguided_full);

    println!("== LoopCD posture sweep (unguided-full = {a0:.4}, close = {a0c:.4}) ==");
    let mut best: Option<(f64, f64, &str)> = None;
    for (mode_name, mode) in [
        ("logit", GuidanceMode::Logits),
        ("hidden", GuidanceMode::Hidden),
    ] {
        for ref_loop in [1usize, 3, 6, 9] {
            for omega_max in [0.1f32, 0.25, 0.5] {
                let answers: Vec<usize> = cases
                    .iter()
                    .map(|c| {
                        argmax(&guided_logits(
                            &fixture, c, MAX_LOOPS, mode, omega_max, ref_loop,
                        ))
                    })
                    .collect();
                for a in &answers {
                    assert!(
                        *a < loopcd_fixture::VOCAB,
                        "non-vocabulary argmax — broken logits"
                    );
                }
                let a = acc_all(&answers);
                let ac = acc_close(&answers);
                println!(
                    "{mode_name:>6} ref_loop={ref_loop} ω_max={omega_max}: full {a:.4} | close {ac:.4}"
                );
                if best.is_none() || a > best.unwrap().0 {
                    best = Some((a, ac, ""));
                }
            }
        }
    }
    if let Some((a, ac, _)) = best {
        println!("best sweep posture: full {a:.4} | close {ac:.4} (vs unguided {a0:.4}/{a0c:.4})");
    }
}

/// G3 leg: the pre-guidance/post-guidance flag-off byte-identity re-pinned
/// at the GOAT's feature set (ω = 0 structural no-op).
#[test]
fn loopcd_goat_g3_flag_off_byte_identity() {
    let fixture = LoopCdFixture::new();
    for case in all_cases().iter().take(8) {
        let base = fixture.run_case(case, 6);
        let mut g = LoopGuidance::new(
            LoopGuidanceConfig {
                mode: GuidanceMode::Logits,
                omega: 0.0,
                omega_max: 0.0,
                adaptive: false,
                ref_loop: 1,
            },
            loopcd_fixture::VOCAB,
            fixture.config.n_embd,
        );
        let off = fixture.run_tokens_guided(&case.prefix, 6, Some(&mut g), None);
        assert_eq!(base, off, "ω=0 must be byte-identical (G3)");
    }
}
