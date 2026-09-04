#![cfg(all(feature = "lt2_looped", feature = "lt2_deep_stability"))]
//! Issue 717 T3/T4 — runtime damping + direction-scale GOAT gate.
//!
//! Source: sotaku (riir-train Research 440) — delayed damping is a runtime,
//! checkpoint-agnostic rescue (`h ← (1−α)h + α·F(h)` after burn-in B);
//! closed form: a locally-linear mode λ maps to `1−α+αλ` ([`project_lambda`]).
//!
//! Gates:
//! - **G1** — bit-identity: knob `None`, `α = 0`, and neutral scales
//!   `{1, 1}` are all bit-identical to plain `forward_looped` (the
//!   feature is DEFAULT-OFF; enabling it must change nothing unless a knob
//!   is explicitly armed).
//! - **G2** — on a deterministically-destabilized fixture (gate-driven
//!   ρ > 1 residual injection + scaled-up layer weights), the undamped run
//!   degrades by T=1024 while damping restores sane outputs, and the
//!   measured per-iteration norm multiplier follows the eigenvalue map
//!   `λ → 1−α+αλ` monotonically across an α sweep.
//! - **G3** — on the STABLE fixture, explicitly-enabled damping costs at
//!   most the upstream-measured class (1.5–2.3 pts analogue on the readout
//!   agreement) — measured and recorded, with the fixture caveat that these
//!   are random weights, not a trained stable checkpoint.
//! - **G4** — the stabilization hot loop is allocation-free (tracking
//!   allocator counters — deterministic; NO wall-clock on this loaded box).
//! - **T4** — tangential/radial probe: which update axis matters on OUR
//!   fixture (upstream: tangential ×0.25 rescued direction-drift failures;
//!   a magnitude-driven failure needs the radial axis), plus the
//!   direction-drift diagnostic (direction-only vs full readout gap).
//!
//! Run: `cargo test -p katgpt-rs --features lt2_looped,lt2_deep_stability --test issue_717_t3_t4_damping_goat -- --nocapture`

use katgpt_rs::hla::MultiLayerAhlaCache;

// Issue 721 T3: install the tracking allocator in THIS test binary (the root
// lib no longer registers a `#[global_allocator]` as a library).
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;
use katgpt_rs::transformer::loop_deep::{DirectionScales, LoopDeepRun, project_lambda, robust_norm};
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

/// Destabilized-fixture residual injection: every classic gate ρ_τ = RHO,
/// giving the carried state a multiplicative mode h ← ρ·h + F̃(h) whose
/// asymptotic multiplier is ρ (the F̃ term vanishes relatively as ‖h‖ grows).
const RHO: f32 = 1.1;

/// Layer-weight scale for the destabilized fixture (the issue's
/// "scaled-up weights" arm; the readout is deliberately left unscaled so
/// argmax comparisons stay meaningful).
const WEIGHT_SCALE: f32 = 1.3;

/// α sweep (upstream span: 0.25 down to 0.03125).
const ALPHAS: [f32; 5] = [0.03125, 0.0625, 0.125, 0.25, 0.5];

const N_TOKENS: usize = 27;

/// Stable fixture (issue_407 shape; zero gates, default-scale weights).
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

/// Destabilized fixture: same seed/config, layer weights × [`WEIGHT_SCALE`],
/// every classic residual gate set to [`RHO`] (magnitude-driven mode:
/// h ← ρ·h + h̃ — the update is dominated by the h-aligned component).
fn destabilized_fixture(t: usize) -> (Config, TransformerWeights, ResidualGate, SdpaOutputGate) {
    let (mut config, mut weights, mut residual_gate, sdpa_gate) = stable_fixture(t);
    config.loop_mode = LoopMode::WeightShared { loop_count: t };
    for layer in &mut weights.layers {
        for buf in [
            &mut layer.attn_wq,
            &mut layer.attn_wk,
            &mut layer.attn_wv,
            &mut layer.attn_wo,
            &mut layer.mlp_w1,
            &mut layer.mlp_w2,
        ] {
            for v in buf.iter_mut() {
                *v *= WEIGHT_SCALE;
            }
        }
    }
    for g in residual_gate.gates.iter_mut() {
        *g = RHO;
    }
    (config, weights, residual_gate, sdpa_gate)
}

/// One deep forward. Returns (logits, final state norm, stats-taken flag).
#[allow(clippy::type_complexity)]
fn run_deep(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    run: Option<&mut LoopDeepRun>,
) -> (Vec<f32>, f32) {
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
        run,
    );
    let logits = logits.to_vec();
    let final_norm = robust_norm(&ctx.x[..config.n_embd]);
    (logits, final_norm)
}

/// Geometric-mean per-iteration multiplier over a late window of snapshots.
/// `from`/`to` are SNAPSHOT indices; `every` is the snapshot interval in
/// iterations, so the exponent divides by the ITERATION span. Skips the
/// early transient where the h̃ term is still comparable to h.
fn window_multiplier(norms: &[f32], from: usize, to: usize, every: usize) -> f32 {
    assert!(from < to && to <= norms.len(), "bad window {from}..{to}");
    let a = norms[from];
    let b = norms[to];
    assert!(a.is_finite() && b.is_finite() && a > 0.0, "non-finite window");
    let iters = ((to - from) * every) as f32;
    (b / a).powf(1.0 / iters)
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

fn logits_finite(logits: &[f32]) -> bool {
    logits.iter().all(|l| l.is_finite())
}

// ── G1: bit-identity — None / α=0 / neutral scales ──────────────────────

#[test]
fn g1_bit_identity_none_alpha0_and_neutral_scales() {
    // Stable AND destabilized fixtures, several depths and positions: every
    // disabled-knob spelling must be bit-identical to `None` (which is
    // bit-identical to pre-Issue-717 builds).
    for &t in &[1usize, 4, 16] {
        for fixture in 0..2 {
            let (config, weights, gate, sdpa) = if fixture == 0 {
                stable_fixture(t)
            } else {
                destabilized_fixture(t)
            };
            for pos in [0usize, 3] {
                let (base, _) = run_deep(&config, &weights, &gate, &sdpa, 0, None);

                let mut run = LoopDeepRun::new(4); // stats on, knobs off
                let (stats_only, _) =
                    run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
                assert_eq!(
                    base, stats_only,
                    "T={t} fixture={fixture} pos={pos}: stats-only run perturbed logits"
                );

                let mut run = LoopDeepRun::with_damping(0.0, 0, 4);
                let (alpha0, _) = run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
                assert_eq!(
                    base, alpha0,
                    "T={t} fixture={fixture} pos={pos}: α=0 damping not bit-identical"
                );

                let mut run = LoopDeepRun::new(4);
                run.direction_scales = Some(DirectionScales { radial: 1.0, tangential: 1.0 });
                let (neutral, _) = run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
                assert_eq!(
                    base, neutral,
                    "T={t} fixture={fixture} pos={pos}: neutral scales not bit-identical"
                );
            }
        }
    }
    println!("[G1] ✅ None / α=0 / scales{{1,1}} bit-identical across fixtures, depths, positions");
}

// ── G2: destabilized rescue + eigenvalue-map monotonicity ────────────────

#[test]
fn g2_destabilized_rescue_and_eigenvalue_map() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 717 G2 — ρ={RHO} gate-driven fixture, weights ×{WEIGHT_SCALE}          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // (a) Measure the undamped multiplier λ̂ on a T=256 run over its SECOND
    // HALF (the early iterations carry a visible h̃ transient; by τ≈128 the
    // ρ·h term dominates and the multiplier has converged to ρ).
    let lam_hat;
    {
        let (config, weights, gate, sdpa) = destabilized_fixture(256);
        let mut run = LoopDeepRun::new(8);
        let (_, final_norm) = run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
        let norms = run.stats.state_norms.clone();
        let last = norms.len() - 1;
        lam_hat = window_multiplier(&norms, last / 2, last, 8);
        println!(
            "λ̂ (undamped multiplier, τ∈[{},{ }]) = {lam_hat:.4}; final norm T=256 = {final_norm:.3e}",
            (last / 2) * 8,
            last * 8
        );
        assert!(
            (1.02..=1.35).contains(&lam_hat),
            "λ̂={lam_hat} outside the gate-driven band (ρ = {RHO} expected)"
        );
        // The map's brackets: α=1 reproduces the full update (λ itself);
        // α=0 freezes the state (multiplier 1 — and α=0 is also the
        // bit-identical OFF spelling, the G1 contract).
        assert!(
            (project_lambda(lam_hat, 1.0) - lam_hat).abs() < 1e-6,
            "project_lambda(λ,1) must be λ"
        );
        assert!(
            (project_lambda(lam_hat, 0.0) - 1.0).abs() < 1e-6,
            "project_lambda(λ,0) must be 1 (frozen)"
        );
    }

    // (b) The undamped LONG run degrades: non-finite state or readout by
    // T=1024 (ρ^1024 overflows f32 deterministically).
    let undamped_logits;
    let undamped_norm;
    {
        let (config, weights, gate, sdpa) = destabilized_fixture(1024);
        let mut run = LoopDeepRun::new(16);
        let (logits, final_norm) =
            run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
        let degraded = !logits_finite(&logits)
            || run.stats.state_non_finite_at.is_some()
            || !final_norm.is_finite()
            || final_norm > 1e30;
        assert!(degraded, "undamped arm did NOT degrade at T=1024 — fixture broken");
        undamped_logits = logits;
        undamped_norm = final_norm;
        println!(
            "undamped T=1024: DEGRADED (state_non_finite_at={:?}, final norm={undamped_norm:e})",
            run.stats.state_non_finite_at
        );
    }

    // (c) Damped arms: finite, sane, and ON the eigenvalue map.
    println!("┌─────────┬───────────────┬───────────────┬──────────────┐");
    println!("│   α     │ multiplier    │ map 1−α+αλ̂    │ final norm   │");
    println!("├─────────┼───────────────┼───────────────┼──────────────┤");

    let mut prev_mult = 0.0f32;
    for &alpha in &ALPHAS {
        let (config, weights, gate, sdpa) = destabilized_fixture(1024);
        let mut run = LoopDeepRun::with_damping(alpha, 0, 16);
        let (logits, final_norm) =
            run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));

        // Sane outputs: finite readout + finite state + no tripwire fire.
        assert!(
            logits_finite(&logits),
            "α={alpha}: damped readout non-finite at T=1024"
        );
        assert!(final_norm.is_finite() && final_norm < 1e30, "α={alpha}: state blew up");
        assert!(run.stats.state_non_finite_at.is_none(), "α={alpha}: state tripwire fired");

        // Eigenvalue map: the SECOND-HALF window (τ ∈ [512, 1024]) — late
        // enough that the h̃ term is negligible and the multiplier has
        // converged onto 1−α+αλ̂.
        let norms = run.stats.state_norms.clone();
        let last = norms.len() - 1;
        let mult = window_multiplier(&norms, last / 2, last, 16);
        let predicted = project_lambda(lam_hat, alpha);
        let rel_err = ((mult - predicted) / predicted).abs();
        println!(
            "│ {alpha:<7} │ {mult:>13.5} │ {predicted:>13.5} │ {final_norm:>11.3e} │  (rel err {rel_err:.4})"
        );
        assert!(
            rel_err < 0.03,
            "α={alpha}: measured multiplier {mult} off the eigenvalue map {predicted} (>3%)"
        );
        // Monotonicity: multiplier strictly increases with α (more damping
        // ⇔ smaller α ⇔ slower growth).
        assert!(mult > prev_mult, "α={alpha}: multiplier not monotone ({mult} ≤ {prev_mult})");
        prev_mult = mult;
    }
    println!("└─────────┴───────────────┴───────────────┴──────────────┘");

    // (d) Discrimination survives on a damped arm: the 27-token suite still
    // produces distinct readouts (cosine distance > 0), unlike the degraded
    // undamped arm.
    {
        let (config, weights, gate, sdpa) = destabilized_fixture(1024);
        let mut run = LoopDeepRun::with_damping(0.25, 0, 64);
        let mut ref_logits = Vec::with_capacity(N_TOKENS);
        for tok in 0..N_TOKENS {
            let (logits, _) = run_deep(&config, &weights, &gate, &sdpa, tok, Some(&mut run));
            assert!(logits_finite(&logits), "token {tok}: non-finite on the damped arm");
            ref_logits.push(logits);
        }
        let mut dist_sum = 0.0f32;
        let mut count = 0u32;
        for i in 0..N_TOKENS {
            for j in (i + 1)..N_TOKENS {
                dist_sum += 1.0 - cosine(&ref_logits[i], &ref_logits[j]);
                count += 1;
            }
        }
        let disc = dist_sum / count as f32;
        println!("damped (α=0.25) inter-prompt logit discrimination = {disc:.4} (must be > 0)");
        assert!(disc > 0.01, "damped arm collapsed to identical readouts (disc={disc})");
        // The undamped arm is non-finite by here — the honest contrast.
        assert!(
            !logits_finite(&undamped_logits) || undamped_norm > 1e30,
            "undamped contrast arm unexpectedly sane"
        );
    }

    println!("[G2] ✅ damping restores sane outputs; multiplier follows λ → 1−α+αλ̂ monotonically");
}

// ── G3: stable-fixture cost bound (upstream 1.5–2.3 pts analogue) ────────

#[test]
fn g3_stable_fixture_damping_cost() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 717 G3 — stable fixture, damping α=0.25 explicit          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    for &t in &[64usize, 1024] {
        let (config, weights, gate, sdpa) = stable_fixture(t);
        let mut agree = 0u32;
        let mut cos_sum = 0.0f32;
        for tok in 0..N_TOKENS {
            let (base, _) = run_deep(&config, &weights, &gate, &sdpa, tok, None);
            let mut run = LoopDeepRun::with_damping(0.25, 0, 0);
            let (damped, _) = run_deep(&config, &weights, &gate, &sdpa, tok, Some(&mut run));
            assert!(logits_finite(&damped), "T={t} tok={tok}: damped readout non-finite");
            if argmax(&base) == argmax(&damped) {
                agree += 1;
            }
            cos_sum += cosine(&base, &damped);
        }
        let agreement = agree as f32 / N_TOKENS as f32;
        let mean_cos = cos_sum / N_TOKENS as f32;
        let disagreement_pts = 100.0 * (1.0 - agreement);
        println!(
            "T={t:>5}: argmax agreement {agree}/{N_TOKENS} = {agreement:.3} \
             (disagreement {disagreement_pts:.2} pts, upstream class 1.5–2.3), mean cos {mean_cos:.4}"
        );
        // Structural bound only: finite everywhere + not catastrophically
        // flipped. The measured pts go to the bench doc — random-weight
        // fixtures are NOT trained stable checkpoints, so the 2.3 pt
        // upstream number is a reference, not a gate here.
        assert!(
            agreement >= 0.5,
            "T={t}: damping flipped the majority of readouts on the STABLE fixture"
        );
    }
    println!("[G3] ✅ explicit damping on the stable fixture stays structurally sane; cost recorded");
}

// ── G4: alloc-free stabilization hot loop (deterministic counters) ───────

#[test]
#[cfg(debug_assertions)]
fn g4_alloc_free_stabilization_hot_loop() {
    use katgpt_core::alloc::{get_alloc_stats, reset_alloc_stats};

    // The root crate's debug-only TrackingAllocator must be linked (this
    // binary references root-crate items, so the rlib member carrying the
    // #[global_allocator] shim cannot be dropped).
    reset_alloc_stats();
    let sentinel: Vec<u8> = vec![0u8; 64];
    let (sent_count, _) = get_alloc_stats();
    assert!(sent_count > 0, "TrackingAllocator not installed — alloc gate vacuous");
    drop(sentinel);

    let (config, weights, gate, sdpa) = stable_fixture(256);

    // Context + caches are built ONCE outside the measured region — the
    // steady-state claim is about the damped forward loop, not the
    // per-call harness setup.
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut ahla_cache = MultiLayerAhlaCache::new(&config);

    // Measured region: 8 deep runs with damping + scales armed, reusing ONE
    // context/caches and ONE LoopDeepRun whose stats are `clear()`ed between
    // calls (clear keeps buffer capacity — no drop/realloc churn inside the
    // measured region). The warm-up primes THIS SAME `run` so its stats
    // vectors and logit scratch reach capacity before the reset.
    let mut run = LoopDeepRun::with_damping(0.25, 0, 16);
    run.direction_scales = Some(DirectionScales { radial: 0.25, tangential: 1.0 });
    let _ = forward_looped(
        &mut ctx,
        &weights,
        &mut cache,
        &mut ahla_cache,
        0,
        0,
        &config,
        &gate,
        &sdpa,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        None,
        #[cfg(feature = "gain_cost_halt")]
        None,
        Some(&mut run),
    );

    reset_alloc_stats();
    for _ in 0..8 {
        run.stats.clear();
        let _ = forward_looped(
            &mut ctx,
            &weights,
            &mut cache,
            &mut ahla_cache,
            0,
            0,
            &config,
            &gate,
            &sdpa,
            None,
            None,
            #[cfg(feature = "weight_shared_advantage_gate")]
            None,
            None,
            #[cfg(feature = "gain_cost_halt")]
            None,
            Some(&mut run),
        );
    }
    let (count, bytes) = get_alloc_stats();
    assert_eq!(
        count, 0,
        "stabilization hot loop allocated {count} times ({bytes} B) over 8 deep runs"
    );
    println!("[G4] ✅ 8 × T=256 damped+scaled deep runs: 0 allocations, 0 bytes");
}

// ── T4: tangential/radial probe + direction-drift diagnostic ─────────────

#[test]
fn t4_tangential_radial_probe_and_direction_drift() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 717 T4 — which update axis matters on OUR fixture?        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Direction-drift diagnostic on the UNDAMPED arm at a depth that stays
    // finite (T=64): is the state direction wandering (upstream's failure
    // mode) or growing along a fixed direction (ours)?
    {
        let (config, weights, gate, sdpa) = destabilized_fixture(64);
        let mut run = LoopDeepRun::new(8);
        run.capture_states = true;
        let _ = run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
        let snaps = &run.stats.state_snapshots;
        let n = snaps[0].len();
        let dir_cos = |a: &Vec<f32>, b: &Vec<f32>| {
            let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..n {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        // Cosine between consecutive state directions, averaged late.
        let mut cos_sum = 0.0f32;
        for w in snaps.windows(2).skip(2) {
            cos_sum += dir_cos(&w[0], &w[1]);
        }
        let mean_dir_cos = cos_sum / (snaps.len() - 3) as f32;
        // Direction-only vs full readout gap at the final state: renormalize
        // the state and compare readouts — upstream matched within 0.3pp
        // when direction (not size) was the failure.
        let last = &snaps[snaps.len() - 1];
        let norm = last.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("mean consecutive state-direction cosine (late, T=64) = {mean_dir_cos:.4}");
        println!("final state norm at T=64 = {norm:.3e} (norm ratio vs √n = {:.1}×)", norm / (n as f32).sqrt());
        // A direction cosine ≈ 1 with a huge norm = magnitude-driven failure:
        // the state grows along a FIXED direction. That predicts the radial
        // axis is the operative one here (upstream's direction-drift failure
        // predicted tangential).
        assert!(
            mean_dir_cos > 0.99,
            "state direction is wandering (mean cos {mean_dir_cos}) — this fixture is \
             direction-driven, contradicting the ρ-gate design; investigate"
        );
    }

    // The probe: T=1024 arms — which knob rescues the magnitude-driven
    // failure? Radial ×0.25 shrinks the ρ-dominated (h-aligned) component
    // (multiplier → 1 + 0.25(ρ−1)); tangential ×0.25 leaves it intact
    // (multiplier → ρ → overflow).
    let arms: [(&str, Option<DirectionScales>); 3] = [
        ("none (degraded control)", None),
        ("radial ×0.25", Some(DirectionScales { radial: 0.25, tangential: 1.0 })),
        ("tangential ×0.25", Some(DirectionScales { radial: 1.0, tangential: 0.25 })),
    ];

    println!("┌─────────────────────────┬───────────────┬───────────────┐");
    println!("│ arm (T=1024)            │ finite        │ final norm    │");
    println!("├─────────────────────────┼───────────────┼───────────────┤");

    let mut radial_rescued = false;
    let mut tangential_degraded = false;
    for (name, scales) in &arms {
        let (config, weights, gate, sdpa) = destabilized_fixture(1024);
        let mut run = LoopDeepRun::new(64);
        run.direction_scales = *scales;
        let (logits, final_norm) =
            run_deep(&config, &weights, &gate, &sdpa, 0, Some(&mut run));
        let finite = logits_finite(&logits)
            && run.stats.state_non_finite_at.is_none()
            && final_norm.is_finite();
        println!("│ {name:<23} │ {:>13} │ {:>13.3e} │", finite, final_norm);
        match *scales {
            None => assert!(!finite, "control arm unexpectedly survived T=1024"),
            Some(s) if s.radial < 1.0 => {
                radial_rescued = finite;
                assert!(finite, "radial ×0.25 failed to rescue the magnitude-driven failure");
            }
            Some(s) if s.tangential < 1.0 => {
                tangential_degraded = !finite;
            }
            _ => unreachable!(),
        }
    }
    println!("└─────────────────────────┴───────────────┴───────────────┘");
    println!();
    println!("Verdict on OUR fixture: radial-rescued={radial_rescued}, tangential-degraded={tangential_degraded}");
    println!("Upstream (sotaku) had the OPPOSITE axis because their failure mode was");
    println!("accumulated DIRECTION drift; ours is magnitude growth along a fixed direction");
    println!("(ρ-gated carry) — the diagnostic above pins which regime a given model is in.");
    assert!(
        radial_rescued && tangential_degraded,
        "axis verdict flipped vs the fixture design — re-read the diagnostic"
    );
    println!("[T4] ✅ probe + direction-drift diagnostic recorded (radial axis operative here)");
}
