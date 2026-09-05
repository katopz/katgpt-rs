#![cfg(all(feature = "lt2_looped", feature = "loop_stability_fix"))]
//! Issue 698 T8 GATE PROBE — hand conditional gate mechanism precondition.
//!
//! T8 proposes the hand conditional gate `g = σ(β·(cos(h, h_pre)−θ)+b)` —
//! open on DIVERGENCE (low cos between the carried state and its previous
//! value), distilled from the GRT paper's contrastive-projection finding.
//! The issue gates T8 on exactly one measurement: *"requires T1 mechanism
//! gate (gate-open events co-locate with high marginal gain, rank
//! correlation threshold) + T2 anchor + inter-loop norm prerequisite."*
//! Nobody had run that probe — this test is it.
//!
//! # The falsifiable question
//!
//! > On the T1 fixture, do gate-open events (large divergence between
//! > consecutive carried states) co-locate with high MARGINAL LOOP GAIN?
//! > If yes, a divergence-conditioned gate opens where looping still pays —
//! > the mechanism has legs and T8 design work unblocks. If no (the T7
//! > lesson class: the signal tracks fixed-point distance, not gain), the
//! > gate is a coin flip on random weights → REFUSED with the measurement
//! > on record.
//!
//! # Modelless method — NO src hook needed
//!
//! The per-loop carried state is reconstructible through production
//! `forward_looped` alone: running with elastic override `r` leaves the
//! post-loop state `S(r)` in `ctx.hidden_state` (snapshotted from `ctx.x`
//! after the last iteration). Fresh-context determinism (the T1/T2 G1
//! contract) makes `S(τ)` identical across runs, so `S(r)` from an r-run
//! equals the in-loop state at iteration r−1 of any longer run.
//!
//! Per token t (all 27 micro-vocab tokens) and loop r ∈ 2..=16:
//! - divergence signal: `cos_t(r) = cos(S(r), S(r−1))` — the gate's own
//!   conditioning pair (h, h_pre); openness = θ − cos, and the Spearman
//!   rank correlation is invariant to the constant θ (and to β, b), so the
//!   probe correlates `−cos` with gain directly — the rank gate is the
//!   co-location question, the constants are a deployment choice;
//! - marginal loop gain: `gain_t(r) = kl_t(r) − kl_t(r+1)` where
//!   `kl_t(r) = KL(softmax(logits at r) ‖ softmax(logits at 32))` — the
//!   improvement loop r+1 delivers, exactly what continuing from r buys
//!   (T1's gain(r→r+1) definition).
//!
//! Two contexts, both through production `forward_looped`:
//! - **InterLoopNorm** — the shipped stable trajectory (T1/T7's context);
//! - **FixedAnchor + armed gate (ρ=0.5)** — T8's prerequisite deployment
//!   context (T2 anchor + norm), the loop the conditional gate would ride.
//!
//! # Pre-registered gates (the issue's "rank correlation threshold")
//!
//! - **Gate A (pooled):** Spearman(−cos, gain) ≥ **+0.30** over all (t, r)
//!   pairs in the FixedAnchor context — gate-open co-locates with gain.
//! - **Gate B (within-loop):** the MEDIAN over r of the token-level
//!   Spearman ≥ **+0.20** — the per-token discriminative content a
//!   conditional gate would actually use. (Gate A alone is trivially
//!   satisfiable by the shared τ-decay of cos→1 and gain→0 — the T7
//!   lesson; B is the real bar.)
//! - T8 unblocks iff A AND B in the FixedAnchor context. InterLoopNorm is
//!   reported as context, not gated.
//!
//! # Harness parity + honesty
//!
//! - The mean of the per-token `kl_t(r)` MUST reproduce T1's pinned
//!   spectrum bits at r ∈ {1, 2, 4, 8} within 1e-5 relative (T7's
//!   cross-platform convention; x86_64-windows drifts ~2.5e-7).
//! - G1: double in-process measurement bit-identical (cos + kl fields).
//! - Honest scope: 27 tokens, micro config, random weights — the probe
//!   arbitrates the MECHANISM, not a deployed-path quality record.
//!
//! # Run
//!
//! ```bash
//! cargo test --features lt2_looped,loop_stability_fix --test issue_698_t8_gate_probe -- --nocapture
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng, SdpaOutputGate,
};

// ── Constants ────────────────────────────────────────────────────

/// Loops r with a full (state, gain) pair: gain needs kl(r+1), states need
/// S(r−1). r ∈ 2..=R_MAX; logits/states measured for r ∈ 1..=R_MAX+1.
const R_MAX: usize = 16;

/// Loop count of the fixed-point reference output (T1's R_REF).
const R_REF: usize = 32;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches the T1 convention).
const SEED: u64 = 42;

/// T1's pinned spectrum BITS at the parity grid (aarch64-measured; compared
/// as values at 1e-5 relative — the T7 cross-platform convention).
const T1_PINS: [(usize, u32); 4] = [
    (1, 0x4146_90c0), // 1.241034e1
    (2, 0x40b6_a83e), // 5.708259e0
    (4, 0x4012_7fbe), // 2.289042e0
    (8, 0x3dd5_a64d), // 1.043104e-1
];

/// Pre-registered rank-correlation thresholds (see module doc).
const GATE_A_POOLED: f32 = 0.30;
const GATE_B_MEDIAN: f32 = 0.20;

/// The FixedAnchor context's armed gate decay (T2's A/B arm value).
const ANCHOR_GATE_DECAY: f32 = 0.5;

// ── Fixture (verbatim from T1) ───────────────────────────────────

fn make_config(mode: LoopStabilityMode) -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config.loop_stability_mode = mode;
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

// ── Forward + metrics ────────────────────────────────────────────

/// Run `forward_looped` for one prompt token at `r` loops (elastic override);
/// returns owned logits AND the post-loop carried state S(r).
fn run_once_state(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    loops: usize,
) -> (Vec<f32>, Vec<f32>) {
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
    );
    (logits.to_vec(), ctx.hidden_state[..config.n_embd].to_vec())
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

/// cos(a, b) with an explicit zero-norm guard (returns 1.0 — "identical",
/// the converged degenerate — if either norm vanishes).
fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Spearman rank correlation (average ranks for ties, f64 accumulation).
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    assert_eq!(xs.len(), ys.len());
    rank_average(xs)
        .iter()
        .zip(rank_average(ys).iter())
        .map(|(&a, &b)| (a, b))
        .collect::<Vec<_>>()
        .pipe(|pairs| pearson_f64(&pairs)) as f32
}

fn rank_average(v: &[f32]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut ranks = vec![0.0f64; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + j + 2) as f64) / 2.0; // 1-based average of ranks i+1..=j+1
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn pearson_f64(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let (mx, my) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / n,
        pairs.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

// A tiny `pipe` combinator to keep `spearman` readable.
trait Pipe: Sized {
    fn pipe<F, Out>(self, f: F) -> Out
    where
        F: FnOnce(Self) -> Out,
    {
        f(self)
    }
}
impl<T> Pipe for T {}

// ── Context measurement ──────────────────────────────────────────

struct ContextResult {
    /// cos_t(r) for t ∈ 0..27, r ∈ 2..=R_MAX → [t][r−2].
    cos: Vec<Vec<f32>>,
    /// kl_t(r) for t, r ∈ 1..=R_MAX+1 → [t][r−1].
    kl: Vec<Vec<f32>>,
}

fn measure_context(
    mode: LoopStabilityMode,
    weights: &TransformerWeights,
    gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
) -> ContextResult {
    let config = make_config(mode);
    let mut cos = Vec::with_capacity(N_PROMPTS);
    let mut kl_table = Vec::with_capacity(N_PROMPTS);

    // Fixed-point reference logits (same context).
    let ref_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_once_state(&config, weights, gate, sdpa_gate, t, R_REF).0)
        .collect();

    // Per token: one run per r ∈ 1..=R_MAX+1 gives logits AND state.
    let mut logits_r: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
    let mut state_r: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
    for t in 0..N_PROMPTS {
        let mut ls = Vec::with_capacity(R_MAX + 1);
        let mut ss = Vec::with_capacity(R_MAX + 1);
        for r in 1..=(R_MAX + 1) {
            let (l, s) = run_once_state(&config, weights, gate, sdpa_gate, t, r);
            ls.push(l);
            ss.push(s);
        }
        logits_r.push(ls);
        state_r.push(ss);
    }

    for t in 0..N_PROMPTS {
        // kl_t(r), r ∈ 1..=R_MAX+1
        let mut kls = Vec::with_capacity(R_MAX + 1);
        for r in 1..=(R_MAX + 1) {
            kls.push(kl(&logits_r[t][r - 1], &ref_logits[t]));
        }
        kl_table.push(kls);
        // cos_t(r), r ∈ 2..=R_MAX
        let mut cs = Vec::with_capacity(R_MAX - 1);
        for r in 2..=R_MAX {
            cs.push(cos_sim(&state_r[t][r - 1], &state_r[t][r - 2]));
        }
        cos.push(cs);
    }

    ContextResult { cos, kl: kl_table }
}

// ── The probe ────────────────────────────────────────────────────

#[test]
fn t698_t8_conditional_gate_mechanism_probe() {
    let weights_config = make_config(LoopStabilityMode::InterLoopNorm);
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&weights_config, &mut rng);
    let sdpa_gate = SdpaOutputGate::new(
        weights_config.n_head,
        weights_config.head_dim,
        weights_config.n_embd,
    );
    let hash = fixture_hash(&weights_config, &weights);
    println!("fixture hash (blake3[16]): {hash} (T7's recorded x86_64-windows value)");
    // Both known platform pins (aarch64 + x86_64-windows) — any other value
    // means the fixture drifted and every Issue-698 measurement is stale.
    assert!(
        hash == "fab06e3f4ba65977" || hash == "c894478d3febdb00",
        "unknown fixture hash {hash}"
    );

    let plain_gate = ResidualGate::new(R_REF, weights_config.n_embd);
    let anchor_gate = ResidualGate::new_loop_stable(R_REF, weights_config.n_embd, ANCHOR_GATE_DECAY);

    // ── G1: double-run bit-identity + harness parity ─────────────
    let a = measure_context(LoopStabilityMode::InterLoopNorm, &weights, &plain_gate, &sdpa_gate);
    let b = measure_context(LoopStabilityMode::InterLoopNorm, &weights, &plain_gate, &sdpa_gate);
    for t in 0..N_PROMPTS {
        for (i, (ca, cb)) in a.cos[t].iter().zip(b.cos[t].iter()).enumerate() {
            assert_eq!(
                ca.to_bits(),
                cb.to_bits(),
                "G1 determinism: cos differs at token {t} r={}",
                i + 2
            );
        }
        for (i, (ka, kb)) in a.kl[t].iter().zip(b.kl[t].iter()).enumerate() {
            assert_eq!(
                ka.to_bits(),
                kb.to_bits(),
                "G1 determinism: kl differs at token {t} r={}",
                i + 1
            );
        }
    }

    // Harness parity: mean kl over tokens at the pinned grid = T1's spectrum.
    for (r, pin_bits) in T1_PINS {
        let pin = f32::from_bits(pin_bits);
        let mean = (0..N_PROMPTS)
            .map(|t| a.kl[t][r - 1])
            .sum::<f32>()
            / N_PROMPTS as f32;
        let rel = ((mean - pin) / pin).abs();
        assert!(
            rel < 1e-5,
            "harness parity broke at r={r}: mean kl {mean:.6e} vs T1 pin {pin:.6e} (rel {rel:.2e})"
        );
    }

    // ── Both contexts ────────────────────────────────────────────
    let fixed = measure_context(
        LoopStabilityMode::FixedAnchor,
        &weights,
        &anchor_gate,
        &sdpa_gate,
    );

    for (name, ctx) in [("InterLoopNorm (context)", &a), ("FixedAnchor+armed (gated)", &fixed)] {
        // Pairs (t, r): divergence = −cos, gain = kl(r) − kl(r+1), r ∈ 2..=R_MAX−1
        // (gain at r=R_MAX would need kl(R_MAX+1) — available; keep r ∈ 2..=R_MAX
        // with gain = kl(r) − kl(r+1) valid through r = R_MAX since kl goes to
        // R_MAX+1).
        let mut pool: Vec<(f32, f32)> = Vec::with_capacity(N_PROMPTS * (R_MAX - 1));
        let mut per_r: Vec<Vec<(f32, f32)>> = vec![Vec::new(); R_MAX - 1];
        for t in 0..N_PROMPTS {
            for r in 2..=R_MAX {
                let cosv = ctx.cos[t][r - 2];
                let gain = ctx.kl[t][r - 1] - ctx.kl[t][r];
                pool.push((-cosv, gain));
                per_r[r - 2].push((-cosv, gain));
            }
        }
        let pooled = spearman(&pool.iter().map(|p| p.0).collect::<Vec<_>>(), &pool.iter().map(|p| p.1).collect::<Vec<_>>());
        let mut per_r_rho: Vec<f32> = Vec::with_capacity(R_MAX - 1);
        for pairs in per_r.iter() {
            let rho = spearman(
                &pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
                &pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
            );
            per_r_rho.push(rho);
        }
        // Print ALL per-r correlations + the sign count (n=27 per stratum is
        // individually marginal — the sign pattern across strata is part of
        // the evidence).
        for (i, rho) in per_r_rho.iter().enumerate() {
            println!(
                "  {name}: r={:<2} token-level spearman(−cos, gain) = {:+.3}",
                i + 2,
                rho
            );
        }
        let pos_count = per_r_rho.iter().filter(|&&r| r > 0.0).count();
        println!(
            "  {name}: positive per-r correlations: {pos_count}/{}",
            per_r_rho.len()
        );
        let mut sorted = per_r_rho.clone();
        sorted.sort_by(|x, y| x.total_cmp(y));
        let median = sorted[sorted.len() / 2];
        let mean_cos_by_r: Vec<f32> = (0..R_MAX - 1)
            .map(|i| (0..N_PROMPTS).map(|t| ctx.cos[t][i]).sum::<f32>() / N_PROMPTS as f32)
            .collect();

        println!("\n  ── {name} ──");
        println!(
            "  pooled spearman(−cos, gain) over {} pairs = {pooled:+.3}   (Gate A ≥ +{GATE_A_POOLED})",
            pool.len()
        );
        println!(
            "  median token-level spearman = {median:+.3}   (Gate B ≥ +{GATE_B_MEDIAN})"
        );
        println!(
            "  mean cos(S(r), S(r−1)) by r: r=2 {:.4} · r=4 {:.4} · r=8 {:.6} · r=12 {:.6} · r=16 {:.6}",
            mean_cos_by_r[0],
            mean_cos_by_r[2],
            mean_cos_by_r[6],
            mean_cos_by_r[10],
            mean_cos_by_r[14]
        );
        if name.starts_with("FixedAnchor") {
            // ── The pre-registered verdict (measured 2026-08-31, pinned) ─
            // The probe was written with refusal as the expected outcome
            // (the T7 anti-correlation class); the measurement REFUTED that
            // expectation — divergence co-locates with marginal gain in the
            // deployment context. The pin below holds the MEASURED verdict
            // so any fixture/mechanism drift flips it loudly.
            let pass = pooled >= GATE_A_POOLED && median >= GATE_B_MEDIAN;
            if pass {
                println!("  VERDICT: PASS — divergence co-locates with marginal gain at the");
                println!("  pre-registered thresholds, in the T8 deployment context (anchored +");
                println!("  armed, the never-settles regime T2 measured — where late-loop divergence");
                println!("  and late-loop gain persist together). T8 implementation UNBLOCKS.");
                println!("  Interpretant: unlike T7's static step-1 difficulty metrics (which");
                println!("  ANTI-correlated at k≥4), the state's own divergence is a within-run,");
                println!("  per-loop, per-token signal — how much the state just moved predicts");
                println!("  how much the next loop moves it. The co-location GROWS with r (r=2 ≈ 0,");
                println!("  r≥7 substantial): the gate discriminates exactly where a copy-late");
                println!("  schedule must decide when to close.");
            } else {
                println!("  VERDICT: REFUSED — the divergence signal does not co-locate with");
                println!("  marginal loop gain at the pre-registered thresholds. The conditional");
                println!("  gate is a coin flip on this fixture: T8 STAYS DEFERRED with the");
                println!("  measurement on record (the T7 precedent).");
            }
            assert!(
                pass,
                "measured verdict flipped to REFUSED — the mechanism gate no longer holds \
                 (fixture drift or harness change); re-run and re-record before promoting T8"
            );
        }
    }

    // Context note (recorded): the G1-parity kl values are the T1 trajectory;
    // the FixedAnchor trajectory differs (T2's destination-shift finding) and
    // its own kl table is the gain basis for the gated verdict above.
    println!();
    println!("  Caveats: random weights, single-position prompts, micro config; Spearman is");
    println!("  invariant to the gate constants (θ, β, b) — they are deployment choices, the");
    println!("  rank gate is the co-location question; per-token branching economics remain");
    println!("  T7 territory (its routing gate was refused on the same fixture).");
}
