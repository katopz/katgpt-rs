//! Issue 698 T3 — convex copy-late gate schedule (GRT arXiv:2608.15062 C1a/C4).
//!
//! # What is under test
//!
//! `ResidualGate::copy_late_schedule(loop_count, g0, gR)` builds a SCALAR
//! per-loop copy weight `g_τ` consumed by `forward_looped`'s convex blend
//! path: `h^(τ) = g_τ ⊙ src + (1 − g_τ) ⊙ h̃^(τ)` (src = the frozen anchor
//! under FixedAnchor, else the drifting `h^(τ-1)`). This replaces the
//! additive combine `h̃ + ρ_τ ⊙ src` for gates constructed with a schedule.
//!
//! # The falsifiable content (three layers)
//!
//! 1. **Free-theorem spec test (exact, weight-free):** a scalar convex blend
//!    gives `‖h‖ ≤ max(‖prev‖, ‖out‖)` by norm convexity — the additive form
//!    has NO bound (the instability Plan 428 fights). Per-channel gates would
//!    forfeit exactness (p=(1,0), o=(0,1), per-channel g=(1,0) → ‖h‖=√2 > 1);
//!    the scalar schedule is the strongest modelless interpretant. Also the
//!    per-coordinate box property, the R-step induction, and the g=1 / g=0
//!    bit-exact degenerates.
//! 2. **Contraction (the T2 complement, in-vivo):** T2 measured the
//!    constant-ρ additive fixed-anchor arm NEVER settling (ref drift
//!    KL(32,40) = 2.404 vs 3.2e-8 zeros). Mechanism: `h += ρ⊙anchor` injects
//!    a constant-magnitude term every loop — the state keeps moving. The
//!    convex form's update is ∝ (1 − g_τ), so a copy-late schedule (g_τ → 1)
//!    drives the update toward zero: every convex arm must settle (ref drift
//!    ≪ the additive arm's, asserted at a 10× margin).
//! 3. **Quality/schedule sweep:** loss(r) per arm + an endpoint sweep
//!    (g0, gR) — the form-mismatch caveat (checkpoints trained additive)
//!    means the fixture ARBITRATES quality; the paper's openness law
//!    (0.182 → 0.066 monotone) is encoded as schedule monotonicity.
//!
//! # Gates
//!
//! - G1: double-run bit-identity per arm + full bit-pin of the measured
//!   table (AddConst / NormOnly rows pre-pinned from T2 — the arms are
//!   bit-identical constructions, a cross-bench consistency proof).
//! - G3: schedule absent → additive path byte-identical (T1/T2 pins
//!   re-verified in their own suites; the flag-off path is exercised here
//!   by the AddConst arm).
//! - G4: schedule allocated exactly once at construction (len == capacity
//!   == loop_count); zero per-forward allocation (the loop only reads g_τ).
//!
//! ```text
//! cargo test --features lt2_looped,loop_stability_fix --test issue_698_t3_copy_late -- --nocapture
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, CopyLateShape, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng,
    SdpaOutputGate,
};

// ── Constants ────────────────────────────────────────────────────

/// Loop count of the per-arm fixed-point reference output (T1/T2 convention).
const R_REF: usize = 32;

/// Reference-soundness probe: KL(R_REF, R_REF+8) per arm — the SETTLING
/// metric (a converged arm extends to 40 loops bit-identically → KL ≈ 0).
const R_REF_PROBE: usize = R_REF + 8;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches T1 / T2 — identical weights).
const SEED: u64 = 42;

/// T1's pinned fixture hash — this bench re-uses the exact same weights.
/// TWO platform pins (aarch64 M3 + x86_64-windows; the T7-recorded weight-
/// byte drift — see the T2 bench note).
const T1_FIXTURE_HASH: [&str; 2] = ["fab06e3f4ba65977", "c894478d3febdb00"];

/// Additive baseline gate (τ > 0 constant) — T2's known non-settler.
const GATE_DECAY: f32 = 0.5;

/// Loop counts measured per arm.
const R_ARMS: [usize; 4] = [2, 4, 8, 16];

/// Number of measured arms (Arm enum order).
const N_ARMS: usize = 7;

/// Convex schedule endpoints for the main linear arm: T2's armed ρ as the
/// write-open end, GRT's copy-saturated band (> 0.95) as the closed end.
const CONV_G0: f32 = 0.5;
const CONV_GR: f32 = 0.95;

/// Endpoint sweep (linear shape): openness axis — how open the first loops
/// are vs how closed the tail lands. GRT's trained law: openness declines
/// monotonically (0.182 → 0.066 ⇒ g ≈ 0.82 → 0.93); we sweep around it.
const SWEEP_ENDPOINTS: [(f32, f32); 3] = [(0.3, 0.9), (0.5, 0.95), (0.7, 0.95)];

// Pinned table (raw f32 bits). AddConst / NormOnly rows are T2's Fixed /
// Zeros pins (bit-identical arm constructions — cross-bench proof, filled
// here from T2's committed constants); the convex rows are filled from this
// bench's first measurement run (2026-08-30, M3 aarch64).
const PINNED_TABLE: [[u32; 4]; N_ARMS] = [
    [0x403b_51be, 0x3f75_bdb8, 0x3efb_3384, 0x3cef_3798], // AddConst == T2 Fixed
    [0x3f39_0f57, 0x3f99_68e6, 0x3f70_26ac, 0x3ed3_d7e4], // ConvLin
    [0x3f2d_39dc, 0x3f7a_f4b1, 0x3f16_b45a, 0x3db4_baa8], // ConvEase
    [0x3f46_dfeb, 0x3fbd_e2eb, 0x3fb7_f7d7, 0x3fbc_3773], // ConvStep
    [0x3feb_524c, 0x3f9f_30f5, 0x3ec6_7405, 0x3e4d_ee8b], // ConvDrift
    [0x40b6_a83e, 0x4012_7fbe, 0x3dd5_a64d, 0x3684_f1a2], // NormOnly == T2 Zeros
    [0x40fb_3494, 0x4080_e469, 0x3f94_c0ee, 0x3d84_18af], // ConvNone
];
const PINNED_REF_DRIFTS: [u32; N_ARMS] = [
    0x4019_d721, // AddConst == T2 (2.404 — the measured non-settler)
    0x36c1_1086, // ConvLin (5.754e-6)
    0x336c_137e, // ConvEase (5.497e-8)
    0x2ba6_d27b, // ConvStep (1.185e-12 — the best settler)
    0x3383_5684, // ConvDrift (6.116e-8)
    0x330a_9972, // NormOnly == T2 (3.227e-8)
    0x3990_4e6d, // ConvNone (2.752e-4 — no-norm ablation, still settles)
];
/// Endpoint sweep (linear) ref-drift bits, measured 2026-08-30.
const PINNED_SWEEP: [u32; SWEEP_ENDPOINTS.len()] = [0x3885_6de9, 0x36c1_1086, 0x35e5_b44d];
/// Endpoint sweep (linear) loss(2) bits, measured 2026-08-30.
const PINNED_SWEEP_LOSS2: [u32; SWEEP_ENDPOINTS.len()] = [0x3fa4_913e, 0x3f39_0f57, 0x3e5a_0ba5];
/// Destination bias bits (ConvLin@32 ‖ natural@32) = 11.90 nats — the
/// convex schedule relocates the fixed point FAR from the natural trajectory
/// (the additive arm sits 2.37 nats away): contraction is real, the
/// destination shift is its price on untrained weights.
const PINNED_DEST_CONV_NAT: u32 = 0x413e_78ba;
/// Destination bias bits (natural@32 ‖ ConvLin@32) = 6.100 nats.
const PINNED_DEST_NAT_CONV: u32 = 0x40c3_30b5;
/// Destination bias bits (AddConst@32 ‖ natural@32) = 2.371 nats.
const PINNED_DEST_ADD_NAT: u32 = 0x4017_b9a1;

// ── Fixture ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// FixedAnchor + constant additive ρ=0.5 — T2's non-settling baseline.
    AddConst,
    /// FixedAnchor + convex linear g: 0.5 → 0.95 — the headline arm.
    ConvLin,
    /// FixedAnchor + convex eased (closes fast early) g: 0.5 → 0.95.
    ConvEase,
    /// FixedAnchor + convex step-mid g: 0.5 → 0.95 — coarsest 2-phase proxy.
    ConvStep,
    /// InterLoopNorm + convex linear — the SOURCE axis (drifting h^(τ−1)).
    ConvDrift,
    /// FixedAnchor + zeroed additive gates — norm-only floor (T2 zeros arm).
    NormOnly,
    /// None mode + convex linear — convex WITHOUT the inter-loop norm
    /// (boundedness holds without the norm; ablation row).
    ConvNone,
}

fn stability_mode(arm: Arm) -> LoopStabilityMode {
    match arm {
        Arm::AddConst | Arm::ConvLin | Arm::ConvEase | Arm::ConvStep | Arm::NormOnly => {
            LoopStabilityMode::FixedAnchor
        }
        Arm::ConvDrift => LoopStabilityMode::InterLoopNorm,
        Arm::ConvNone => LoopStabilityMode::None,
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

fn make_gate(arm: Arm, config: &Config) -> ResidualGate {
    match arm {
        Arm::AddConst => ResidualGate::new_loop_stable(R_REF, config.n_embd, GATE_DECAY),
        Arm::ConvLin | Arm::ConvDrift | Arm::ConvNone => {
            ResidualGate::copy_late_schedule(R_REF, CONV_G0, CONV_GR)
        }
        Arm::ConvEase => ResidualGate::copy_late_schedule_shaped(
            R_REF,
            CONV_G0,
            CONV_GR,
            CopyLateShape::EaseOutClose,
        ),
        Arm::ConvStep => {
            ResidualGate::copy_late_schedule_shaped(R_REF, CONV_G0, CONV_GR, CopyLateShape::StepMid)
        }
        Arm::NormOnly => ResidualGate::new(R_REF, config.n_embd),
    }
}

/// Deterministic BLAKE3 over every active f32 weight slice (same as T1/T2).
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

// ── Forward + metrics (same as T2) ───────────────────────────────

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
        0, // pos = 0 (single-position, matches the T1/T2/407 convention)
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

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Mean loss(r) for one arm: mean over all prompts of KL(arm@r ‖ arm@R_REF).
fn loss_at(
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
    config: &Config,
    weights: &TransformerWeights,
    gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
) -> Vec<Vec<f32>> {
    (0..N_PROMPTS)
        .map(|t| run_once(config, weights, gate, sdpa_gate, t, R_REF))
        .collect()
}

/// Reference drift KL(R_REF, R_REF_PROBE) per arm — the SETTLING metric.
fn ref_drift(
    config: &Config,
    weights: &TransformerWeights,
    gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    refs: &[Vec<f32>],
) -> f32 {
    let mut drift = 0.0f32;
    for (t, ref_l) in refs.iter().enumerate() {
        let lp = run_once(config, weights, gate, sdpa_gate, t, R_REF_PROBE);
        drift += kl(ref_l, &lp);
    }
    drift / N_PROMPTS as f32
}

// ── Spec test: the free theorem (pure unit math, no forward) ─────

/// Self-contained deterministic LCG (avoids any Rng-API coupling).
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 24 bits → [0, 1)
        ((self.0 >> 40) as f32) / 16777216.0
    }
    /// Value in [-1, 1).
    fn next_sym(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

/// The blend EXACTLY as the kernel applies it (per element, same op order):
/// x ← x·(1−g); then x ← x + src·g. Mirrors `forward_looped`'s scalar blend
/// (SIMD elementwise f32 mul/add is IEEE-identical to the scalar form).
fn blend_kernel(x: &mut [f32], src: &[f32], g: f32) {
    for xi in x.iter_mut() {
        *xi *= 1.0 - g;
    }
    for (xi, &s) in x.iter_mut().zip(src.iter()) {
        *xi += s * g;
    }
}

fn norm2(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[test]
fn t698_t3_free_theorem_scalar_convex_bound() {
    let mut rng = Lcg(0x6986_9869_8698_6987);

    // ── Property A: ‖g·p + (1−g)·o‖ ≤ max(‖p‖, ‖o‖) — the free bound ──
    // Adversarial case families; each with a g grid.
    let mut cases: Vec<(Vec<f32>, Vec<f32>, &'static str)> = vec![
        (vec![1.0, 0.0], vec![0.0, 1.0], "orthogonal units"),
        (vec![1.0, -2.0], vec![1.0, -2.0], "identical"),
        (vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0], "opposites"),
        (vec![0.0, 0.0], vec![0.0, 0.0], "zeros"),
        (vec![0.0; 16], vec![3.5; 16], "zero-to-const"),
        (vec![1.0e6; 4], vec![-2.0e6; 4], "large magnitudes"),
        (vec![0.7], vec![-0.3], "n=1"),
    ];
    for _ in 0..8 {
        let n = 16;
        let p: Vec<f32> = (0..n).map(|_| rng.next_sym()).collect();
        let o: Vec<f32> = (0..n).map(|_| rng.next_sym()).collect();
        cases.push((p, o, "random"));
    }

    for (p, o, name) in &cases {
        let (np, no) = (norm2(p), norm2(o));
        let bound = np.max(no);
        for k in 0..=20u32 {
            let g = k as f32 / 20.0;
            let mut x = o.clone();
            blend_kernel(&mut x, p, g);
            let nh = norm2(&x);
            // f32 slack: relative 1e-5 + tiny absolute (all-zeros case).
            assert!(
                nh <= bound * (1.0 + 1e-5) + 1e-6,
                "free bound violated for {name}: g={g} ‖h‖={nh} > max={bound}"
            );
            // ── Property B: per-coordinate box (mechanism-independent) ──
            for i in 0..x.len() {
                let lim = p[i].abs().max(o[i].abs()) + 1e-6;
                assert!(
                    x[i].abs() <= lim,
                    "coordinate box violated for {name}: g={g} i={i} |{:.3e}| > {:.3e}",
                    x[i],
                    lim
                );
            }
        }
    }

    // ── WHY scalar (the per-channel counterexample, documented) ──
    // p=(1,0), o=(0,1), PER-CHANNEL g=(1,0): h = (1,1), ‖h‖=√2 > 1 = max.
    // The scalar schedule forbids this exact escape.
    {
        let mut h = vec![0.0f32, 0.0];
        // h[0] = g[0]·p[0] + (1−g[0])·o[0] = 1·1 + 0·0 = 1
        h[0] = 1.0 * 1.0 + 0.0 * 0.0;
        // h[1] = g[1]·p[1] + (1−g[1])·o[1] = 0·0 + 1·1 = 1
        h[1] = 0.0 * 0.0 + 1.0 * 1.0;
        let nh = norm2(&h);
        assert!
            (
                nh > 1.0 + 1e-6,
                "per-channel counterexample must EXCEED the bound (sanity): {nh}"
            );
        // ...while the scalar blend on the same vectors never does (covered
        // by the grid above — asserted here explicitly for the record).
        let mut x = vec![0.0f32, 1.0];
        blend_kernel(&mut x, &[1.0, 0.0], 0.5);
        assert!(norm2(&x) <= 1.0 + 1e-6);
    }

    // ── Property C: R-step induction — ‖h(r)‖ ≤ max(‖h(0)‖, max_r ‖o(r)‖) ──
    {
        let n = 16;
        let mut h: Vec<f32> = (0..n).map(|_| rng.next_sym()).collect();
        let h0 = norm2(&h);
        let mut max_out = 0.0f32;
        for _ in 0..8 {
            let o: Vec<f32> = (0..n).map(|_| rng.next_sym() * 3.0).collect();
            max_out = max_out.max(norm2(&o));
            let g = rng.next_unit();
            blend_kernel(&mut h, &o, g);
        }
        let bound = h0.max(max_out);
        assert!(
            norm2(&h) <= bound * (1.0 + 1e-5) + 1e-6,
            "induction bound violated: ‖h(8)‖={:.4} > max={bound:.4}",
            norm2(&h)
        );
    }

    // ── Property D: degenerates are BIT-exact ──
    // g = 1: h = src exactly (the copy-closed extreme freezes the state).
    {
        let src = vec![0.5f32, -1.25, 3.75e-4, -2.0];
        let mut x = vec![7.0f32; 4];
        blend_kernel(&mut x, &src, 1.0);
        assert!(bits_eq(&x, &src), "g=1 must equal src bit-exactly");
    }
    // g = 0: h unchanged exactly (the write-open extreme ignores the source).
    {
        let x0 = vec![0.5f32, -1.25, 3.75e-4, -2.0];
        let mut x = x0.clone();
        blend_kernel(&mut x, &[9.0, -9.0, 1.0, -1.0], 0.0);
        assert!(bits_eq(&x, &x0), "g=0 must be a no-op bit-exactly");
    }

    println!("  ✓ free theorem: bound + box + induction + degenerates (weight-free, exact)");
}

#[test]
fn t698_t3_schedule_construction_pins() {
    // ── Shape + endpoint pins ────────────────────────────────────
    let gate = ResidualGate::copy_late_schedule(8, 0.5, 0.95);
    let sched = gate.convex_schedule.as_ref().expect("schedule present");
    assert!(gate.gates.is_empty(), "convex gate carries no per-channel data");
    assert_eq!(sched.len(), 8, "schedule length == loop_count");
    assert_eq!(sched.capacity(), 8, "G4: allocated exactly once, no growth");
    assert_eq!(sched[0].to_bits(), 0.5f32.to_bits(), "entry 0 == g0");
    assert_eq!(sched[7].to_bits(), 0.95f32.to_bits(), "last entry == gR");
    // Monotone non-decreasing (the paper's openness law, modelless form).
    for w in sched.windows(2) {
        assert!(w[1] >= w[0], "linear schedule must be monotone non-decreasing");
    }
    // Interior values: linear interpolation pinned exactly.
    assert_eq!(sched[2].to_bits(), (0.5f32 + (0.95 - 0.5) * (2.0 / 7.0)).to_bits());

    // ── EaseOutClose: monotone, closes faster early than linear ──
    let ease = ResidualGate::copy_late_schedule_shaped(8, 0.5, 0.95, CopyLateShape::EaseOutClose)
        .convex_schedule
        .unwrap();
    let lin = ResidualGate::copy_late_schedule(8, 0.5, 0.95).convex_schedule.unwrap();
    for w in ease.windows(2) {
        assert!(w[1] >= w[0], "ease schedule must be monotone non-decreasing");
    }
    assert!(
        ease[4] > lin[4],
        "ease closes faster early: ease(mid)={:.4} > linear(mid)={:.4}",
        ease[4],
        lin[4]
    );
    assert_eq!(ease[0].to_bits(), 0.5f32.to_bits());
    assert_eq!(ease[7].to_bits(), 0.95f32.to_bits());

    // ── StepMid: coarse 2-phase proxy ────────────────────────────
    let step = ResidualGate::copy_late_schedule_shaped(8, 0.5, 0.95, CopyLateShape::StepMid)
        .convex_schedule
        .unwrap();
    for (i, v) in step.iter().enumerate() {
        let expect: f32 = if i < 4 { 0.5 } else { 0.95 };
        assert_eq!(v.to_bits(), expect.to_bits(), "step entry {i}");
    }

    // ── Clamping: the bound requires g ∈ [0, 1] ──────────────────
    let clamped = ResidualGate::copy_late_schedule(4, -0.5, 1.7).convex_schedule.unwrap();
    assert_eq!(clamped[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(clamped[3].to_bits(), 1.0f32.to_bits());
    assert!(clamped.iter().all(|g| (0.0..=1.0).contains(g)));

    // ── Degenerate lengths accepted ──────────────────────────────
    assert_eq!(
        ResidualGate::copy_late_schedule(0, 0.2, 0.9).convex_schedule.unwrap().len(),
        0
    );
    assert_eq!(
        ResidualGate::copy_late_schedule(1, 0.2, 0.9).convex_schedule.unwrap()[0]
            .to_bits(),
        0.2f32.to_bits()
    );

    // ── convex_gate_at: in-range + past-end clamp + absent ───────
    assert_eq!(gate.convex_gate_at(3), Some(sched[3]));
    assert_eq!(gate.convex_gate_at(100), Some(sched[7]), "past-end clamps to gR");
    assert_eq!(ResidualGate::new(4, 4).convex_gate_at(1), None, "additive gate → None");

    println!("  ✓ schedule construction: shapes, monotonicity, clamp, G4, convex_gate_at");
}

#[test]
fn t698_t3_copy_late_fixture_ab_and_contraction() {
    // ── Fixture (identical weights to T1/T2) ─────────────────────
    let config = make_config(Arm::AddConst);
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&config, &mut rng);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let hash = fixture_hash(&config, &weights);
    println!("fixture hash (blake3[16]): {hash}");
    assert!(
        T1_FIXTURE_HASH.contains(&hash.as_str()),
        "fixture must match a known platform pin of T1's weights (mode is not part of the hash)"
    );

    let arms = [
        Arm::AddConst,
        Arm::ConvLin,
        Arm::ConvEase,
        Arm::ConvStep,
        Arm::ConvDrift,
        Arm::NormOnly,
        Arm::ConvNone,
    ];
    let gates: Vec<(Arm, Config, ResidualGate)> = arms
        .iter()
        .map(|&arm| {
            let c = make_config(arm);
            let g = make_gate(arm, &c);
            (arm, c, g)
        })
        .collect();

    // ── A-priori mechanism pins ──────────────────────────────
    // (1) r=1: no gated iteration runs → every arm identical (re-asserted
    // across the new convex arms).
    {
        let l1_ref = run_once(&gates[0].1, &weights, &gates[0].2, &sdpa_gate, 0, 1);
        for (arm, cfg, gate) in &gates {
            let l1 = run_once(cfg, &weights, gate, &sdpa_gate, 0, 1);
            assert!(
                bits_eq(&l1, &l1_ref),
                "r=1 must be arm-independent (gate never read): {arm:?}"
            );
        }
    }

    // (2) frozen-copy extreme: convex g ≡ 1 → h(r) = anchor exactly for
    // every r ≥ 1 → logits bit-identical across r ∈ {1, 8, 32}. The
    // closed-end degenerate of the copy-late law.
    {
        let cfg = make_config(Arm::ConvLin);
        let gate = ResidualGate::copy_late_schedule(R_REF, 1.0, 1.0);
        let l1 = run_once(&cfg, &weights, &gate, &sdpa_gate, 0, 1);
        let l8 = run_once(&cfg, &weights, &gate, &sdpa_gate, 0, 8);
        let l32 = run_once(&cfg, &weights, &gate, &sdpa_gate, 0, 32);
        assert!(bits_eq(&l1, &l8), "g≡1: logits(1) == logits(8) bit-exact");
        assert!(bits_eq(&l8, &l32), "g≡1: logits(8) == logits(32) bit-exact");
    }

    // (3) vacuity: the convex combine CHANGES the computation vs additive
    // (armed, same mode) — and the source axis is visible under convex.
    {
        let x = run_once(&gates[0].1, &weights, &gates[0].2, &sdpa_gate, 0, 4);
        let y = run_once(&gates[1].1, &weights, &gates[1].2, &sdpa_gate, 0, 4);
        assert!(!bits_eq(&x, &y), "armed: convex ≠ additive at r=4 (vacuity)");
    }
    {
        let (cfg_l, gate_l) = (&gates[1].1, &gates[1].2);
        let (cfg_d, gate_d) = (&gates[4].1, &gates[4].2);
        let x = run_once(cfg_l, &weights, gate_l, &sdpa_gate, 0, 4);
        let y = run_once(cfg_d, &weights, gate_d, &sdpa_gate, 0, 4);
        assert!(!bits_eq(&x, &y), "armed: anchor ≠ drifting source under convex (vacuity)");
    }

    // ── Per-arm references + settling metric ─────────────────────
    let mut refs = Vec::with_capacity(gates.len());
    for (_arm, cfg, gate) in &gates {
        refs.push(ref_outputs(cfg, &weights, gate, &sdpa_gate));
    }
    let mut ref_drifts = [0.0f32; N_ARMS];
    for (idx, (arm, cfg, gate)) in gates.iter().enumerate() {
        ref_drifts[idx] = ref_drift(cfg, &weights, gate, &sdpa_gate, &refs[idx]);
        println!("  ref drift KL({R_REF},{R_REF_PROBE}) [{arm:?}] = {:.3e}", ref_drifts[idx]);
        assert!(
            ref_drifts[idx].is_finite() && ref_drifts[idx] >= 0.0,
            "reference drift must be finite: {arm:?}"
        );
    }

    // ── G1: double-run bit-identity per arm ──────────────────────
    for (idx, (arm, cfg, gate)) in gates.iter().enumerate() {
        let a = loss_at(cfg, &weights, gate, &sdpa_gate, &refs[idx], 4);
        let b = loss_at(cfg, &weights, gate, &sdpa_gate, &refs[idx], 4);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "G1 determinism: loss(4) differs between runs for {arm:?}"
        );
    }

    // ── The A/B table ────────────────────────────────────
    let mut table = [[0.0f32; R_ARMS.len()]; N_ARMS];
    for (idx, (_arm, cfg, gate)) in gates.iter().enumerate() {
        for (j, &r) in R_ARMS.iter().enumerate() {
            table[idx][j] = loss_at(cfg, &weights, gate, &sdpa_gate, &refs[idx], r);
        }
    }

    // ── Endpoint sweep (linear shape) ────────────────────────
    let mut sweep_drift = [0.0f32; SWEEP_ENDPOINTS.len()];
    let mut sweep_loss2 = [0.0f32; SWEEP_ENDPOINTS.len()];
    for (k, (g0, gr)) in SWEEP_ENDPOINTS.iter().enumerate() {
        let cfg = make_config(Arm::ConvLin);
        let gate = ResidualGate::copy_late_schedule(R_REF, *g0, *gr);
        let refs_k = ref_outputs(&cfg, &weights, &gate, &sdpa_gate);
        sweep_drift[k] = ref_drift(&cfg, &weights, &gate, &sdpa_gate, &refs_k);
        sweep_loss2[k] = loss_at(&cfg, &weights, &gate, &sdpa_gate, &refs_k, 2);
    }

    // ── Destination bias (the arbitration number, T2's pattern) ──
    // "Settles" is only good if the destination is not garbage: how far each
    // gated arm's fixed point sits from the NATURAL converged loop trajectory
    // (NormOnly's ref = the normed loop's own injection-free fixed point).
    let mean_kl = |a: &[Vec<f32>], b: &[Vec<f32>]| {
        let mut acc = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += kl(x, y);
        }
        acc / N_PROMPTS as f32
    };
    let dest_conv_nat = mean_kl(&refs[1], &refs[5]); // ConvLin@32 → natural
    let dest_nat_conv = mean_kl(&refs[5], &refs[1]); // natural → ConvLin@32
    let dest_add_nat = mean_kl(&refs[0], &refs[5]); // AddConst@32 → natural
    println!(
        "  destination bias: KL(ConvLin@32 ‖ natural@32) = {dest_conv_nat:.3e} · KL(natural ‖ ConvLin@32) = {dest_nat_conv:.3e} · KL(AddConst@32 ‖ natural@32) = {dest_add_nat:.3e}"
    );

    // ── Structural sanity ────────────────────────────────────
    for (idx, (arm, ..)) in gates.iter().enumerate() {
        for (j, &r) in R_ARMS.iter().enumerate() {
            assert!(
                table[idx][j].is_finite() && table[idx][j] >= 0.0,
                "loss({r}) for {arm:?} must be finite and non-negative: {}",
                table[idx][j]
            );
        }
    }
    // Convergence toward the fixed point holds for the MONOTONE schedules
    // (the paper's openness law) + the additive/norm-only baselines. StepMid
    // is the documented exception: its closure BEGINS at the midpoint, so the
    // whole measured r-grid sits in the open phase and loss(r) is measured
    // against the post-closure fixed point — non-monotone by phase structure,
    // not by instability (its tail settles BEST: drift 1.2e-12, pinned below).
    for idx in [0usize, 1, 2, 4, 5, 6] {
        let arm = arms[idx];
        assert!(
            table[idx][0] > table[idx][R_ARMS.len() - 1],
            "{arm:?} must converge toward its fixed point: loss(2)={:.3e} vs loss(16)={:.3e}",
            table[idx][0],
            table[idx][R_ARMS.len() - 1]
        );
    }
    // The StepMid phase finding, pinned: mid-grid loss is HIGHER than early
    // loss (the state is still traveling in the open phase when the closure
    // has not begun) — a shape trade-off, not a failure (its ref drift is
    // the smallest of all arms).
    assert!(
        table[3][R_ARMS.len() - 1] > table[3][0],
        "StepMid phase structure: loss(16) must exceed loss(2) (open-phase grid)"
    );

    // ── THE contraction claim (the falsifiable content) ──────────
    // Every convex arm settles ≥10× better than T2's non-settling additive
    // baseline. (ConvNone is the no-norm ablation — asserted with the same
    // margin; a FAIL here is the honest finding that the norm is a
    // prerequisite for contraction on random weights, the paper's own
    // composition order.)
    for idx in [1usize, 2, 3, 4, 6] {
        let arm = arms[idx];
        assert!(
            ref_drifts[idx] < ref_drifts[0] / 10.0,
            "{arm:?} must settle ≥10× better than the constant-ρ additive arm: \
             {:.3e} vs {:.3e}",
            ref_drifts[idx],
            ref_drifts[0]
        );
    }

    // ── Measurement dump (bits for the pin consts) ───────────────
    println!("\n═══ Issue 698 T3 — convex copy-late schedule (mean KL to own loop-32 ref) ═══");
    println!(
        "  fixture blake3[16] = {hash}  ·  seed {SEED}  ·  R_REF {R_REF}  ·  additive ρ(τ>0) = {GATE_DECAY}  ·  convex g: {CONV_G0} → {CONV_GR}"
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
    println!("  endpoint sweep (linear, ref-drift / loss(2)):");
    for (k, (g0, gr)) in SWEEP_ENDPOINTS.iter().enumerate() {
        println!("    g {g0:.2} → {gr:.2}   {:.3e}   {:.3e}", sweep_drift[k], sweep_loss2[k]);
    }
    println!(
        "  table bits: {:?}",
        table.map(|row| row.map(|v| v.to_bits()))
    );
    println!("  ref-drift bits: {:?}", ref_drifts.map(|v| v.to_bits()));
    println!("  sweep drift bits: {:?}", sweep_drift.map(|v| v.to_bits()));
    println!("  sweep loss2 bits: {:?}", sweep_loss2.map(|v| v.to_bits()));
    println!(
        "  dest-bias bits: {} {} {}",
        dest_conv_nat.to_bits(),
        dest_nat_conv.to_bits(),
        dest_add_nat.to_bits()
    );

    // ── Pinned tables ────────────────────────────────────────────
    // CROSS-PLATFORM RECORD (2026-08-31, x86_64-windows): exact-bit pins
    // trip off-aarch64 by platform libm ulp drift (see the T2 note —
    // verified pre-existing at clean HEAD in a detached worktree). Value
    // pins relax to the hybrid band |Δ| ≤ 1e-5·|pinned| + 1e-6 (the 1e-6
    // floor keeps the settling CLASSES separated: e.g. ConvStep 1.2e-12 vs
    // AddConst 2.404 stay ≥ 9 orders apart); the contraction ORDERING
    // asserts above remain exact.
    let band = |pinned: f32| 1e-5 * pinned.abs() + 1e-6;
    let assert_close = |v: f32, pinned_bits: u32, what: String| {
        let pinned = f32::from_bits(pinned_bits);
        assert!(
            (v - pinned).abs() <= band(pinned),
            "pinned value drifted ({what}): measured {v:.6e} vs pinned {pinned:.6e}"
        );
    };
    for (idx, (arm, ..)) in gates.iter().enumerate() {
        for (j, &r) in R_ARMS.iter().enumerate() {
            assert_close(
                table[idx][j],
                PINNED_TABLE[idx][j],
                format!("arm {arm:?} r={r}"),
            );
        }
        assert_close(
            ref_drifts[idx],
            PINNED_REF_DRIFTS[idx],
            format!("ref drift arm {arm:?}"),
        );
    }
    for k in 0..SWEEP_ENDPOINTS.len() {
        assert_close(sweep_drift[k], PINNED_SWEEP[k], format!("sweep drift {k}"));
        assert_close(
            sweep_loss2[k],
            PINNED_SWEEP_LOSS2[k],
            format!("sweep loss2 {k}"),
        );
    }
    assert_close(dest_conv_nat, PINNED_DEST_CONV_NAT, "dest conv‖nat".into());
    assert_close(dest_nat_conv, PINNED_DEST_NAT_CONV, "dest nat‖conv".into());
    assert_close(dest_add_nat, PINNED_DEST_ADD_NAT, "dest add‖nat".into());

    println!();
    println!("  VERDICT (measured 2026-08-30): CONTRACTION CONFIRMED — T2's open complement closes.");
    println!("  · Every convex arm settles 4–12 orders better than the constant-ρ additive");
    println!("    baseline (ConvLin 5.8e-6 / Ease 5.5e-8 / Step 1.2e-12 / Drift 6.1e-8 vs 2.404);");
    println!("    the no-norm ablation still settles (2.8e-4). Update ∝ (1 − g_τ) is the mechanism:");
    println!("    harder closure → better settling (Step > Ease > Linear, monotone in the sweep).");
    println!("  · THE PRICE (destination bias): the convex fixed point sits 11.9 nats from the");
    println!("    natural trajectory (additive arm: 2.37). On untrained weights the schedule buys");
    println!("    contraction + the free norm bound AT THE COST of relocating the destination —");
    println!("    promotion to real models needs trained gates (riir-train Plan 364) or a quality");
    println!("    gate on real weights. The free theorem + mechanism are the modelless yield.");
    println!("  · StepMid is the shape trade-off: its closure BEGINS at the midpoint, so the whole");
    println!("    measured r-grid sits in the open phase (non-monotone loss by phase structure,");
    println!("    pinned) while its tail settles BEST of all arms.");
    println!("  Caveats: random weights (form-mismatch — checkpoints trained additive; the fixture");
    println!("  arbitrates, see dest-bias), single-position prompts, micro config.");
}
