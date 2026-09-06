#![cfg(all(feature = "lt2_looped", feature = "loop_stability_fix"))]
//! Issue 698 T7 GATE PROBE — per-token difficulty routing precondition.
//!
//! The issue gates T7 (per-token difficulty routing) on exactly one
//! measurement: *"only if a T1-side probe shows ≥2× marginal-gain separation
//! by step-1 entropy/margin"*. Nobody had run that probe — this test is it.
//!
//! # The falsifiable question
//!
//! > On the T1 fixture, do HARD tokens (high step-1 entropy / low step-1
//! > top1−top2 margin) gain ≥2× more from additional loops than EASY tokens?
//! > If yes, early-exiting easy tokens and spending the loop budget on hard
//! > tokens has headroom (decode-only — per-token branching breaks batched
//! > prefill). If no, the gate is REFUSED and T7 stays deferred with the
//! > measurement on record.
//!
//! # Modelless method (reuses the T1 harness verbatim)
//!
//! Same fixture as T1: `Config::micro()` + seed-42 weights (blake3[16]
//! `fab06e3f4ba65977`, asserted) + `LoopMode::WeightShared` +
//! `InterLoopNorm`, single-position prompts (all 27 micro-vocab tokens),
//! reference = loop-32 output. Per token t:
//!
//! - difficulty at step 1: entropy of softmax(logits at r=1) (nats), and
//!   top1−top2 probability margin at r=1;
//! - marginal loop gain at budget k: `G_t(k) = kl_t(1) − kl_t(k)` where
//!   `kl_t(r) = KL(softmax(logits at r) ‖ softmax(logits at 32))`, for
//!   k ∈ {2, 4, 8}.
//!
//! Tokens are stratified into tertiles (9/9/9) by each difficulty metric
//! (hard = highest entropy / lowest margin); separation(k) = mean G over the
//! hard tertile ÷ mean G over the easy tertile. The GATE reads the max over
//! (metric, k) — a routing implementation would pick its own operating
//! point, so the gate asks whether ANY operating point clears 2×.
//!
//! # Harness parity + honesty
//!
//! - The mean of the per-token `kl_t(r)` MUST reproduce T1's pinned spectrum
//!   bits at r ∈ {1, 2, 4, 8} (summation in token order, identical call
//!   order). Exact bits on aarch64 (T1's measurement platform); on other
//!   targets a 1e-5 relative tolerance + printed delta (T1's documented
//!   cross-platform escape hatch).
//! - G1: double in-process measurement is bit-identical (all fields).
//! - Recorded context, not asserted: Spearman rank correlation between
//!   difficulty and total available gain (feeds T8's rank-correlation gate
//!   wording too).
//! - Honest scope: 27 tokens, micro config, random weights — correlational,
//!   tertile strata. A PASS opens the gate for T7 design work; it is not a
//!   deployed-path quality record (that needs the implementation's own A/B).
//!
//! # Run
//!
//! ```bash
//! cargo test --features lt2_looped,loop_stability_fix --test issue_698_t7_gate_probe -- --nocapture
//! ```

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng, SdpaOutputGate,
};

// ── Constants ────────────────────────────────────────────────────

/// Loop budgets probed per token: r = 1 is the difficulty baseline, the rest
/// are the gain budgets (G_t(k) = kl_t(1) − kl_t(k)).
const PROBE_RS: [usize; 4] = [1, 2, 4, 8];

/// Loop count of the fixed-point reference output (T1's R_REF).
const R_REF: usize = 32;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches T1 / the 407 quality-gate convention).
const SEED: u64 = 42;

/// T1's pinned spectrum bits at r ∈ {1, 2, 4, 8} — the harness-parity
/// cross-check. Any drift here means this probe is not measuring the same
/// fixture T1 measured.
const PINNED_AT_PROBES: [u32; 4] = [
    0x4146_90c0, // r=1  1.241e1
    0x40b6_a83e, // r=2  5.708e0
    0x4012_7fbe, // r=4  2.289e0
    0x3dd5_a64d, // r=8  1.043e-1
];

/// T1's recorded fixture hash — the fixture identity pin.
const T1_FIXTURE_HASH: &str = "fab06e3f4ba65977";

/// The gate threshold from the issue text.
const GATE_RATIO: f32 = 2.0;

// ── Fixture (verbatim from T1) ───────────────────────────────────

fn make_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
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
        0, // pos = 0 (single-position, matches the T1 convention)
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

// ── Metrics ──────────────────────────────────────────────────────

/// log-softmax (max-shifted, sequential f32 — deterministic, same as T1).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&l| l - max).collect();
    let sum_exp: f32 = shifted.iter().map(|&x| x.exp()).sum();
    let log_sum = sum_exp.ln();
    shifted.iter().map(|&x| x - log_sum).collect()
}

/// KL(P ‖ Q) between the categorical distributions of two logit vectors
/// (f64 accumulator + clamp — verbatim from T1).
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

/// Shannon entropy (nats) of softmax(logits) via the log-softmax values:
/// H = −Σ p·log p, computed in f64 for a stable sum.
fn entropy_nats(logits: &[f32]) -> f32 {
    let ls = log_softmax(logits);
    let mut acc = 0.0f64;
    for &l in &ls {
        let p = l.exp() as f64;
        if p > 0.0 {
            acc -= p * l as f64;
        }
    }
    acc as f32
}

/// Top1 − top2 probability margin of softmax(logits).
fn prob_margin(logits: &[f32]) -> f32 {
    let ls = log_softmax(logits);
    let mut probs: Vec<f32> = ls.iter().map(|&l| l.exp()).collect();
    probs.sort_by(|a, b| b.total_cmp(a));
    probs[0] - probs[1]
}

/// Average ranks (ties → average rank), ascending.
fn ranks(xs: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut r = vec![0.0f32; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f32 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation (Pearson over average ranks; NaN when a side
/// is constant).
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let rx = ranks(xs);
    let ry = ranks(ys);
    let n = xs.len() as f32;
    let mx = rx.iter().sum::<f32>() / n;
    let my = ry.iter().sum::<f32>() / n;
    let (mut num, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..xs.len() {
        let a = rx[i] - mx;
        let b = ry[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    if dx == 0.0 || dy == 0.0 {
        return f32::NAN;
    }
    num / (dx.sqrt() * dy.sqrt())
}

// ── Per-token rows ───────────────────────────────────────────────

struct TokenRow {
    /// Step-1 entropy (nats) — high = hard.
    entropy1: f32,
    /// Step-1 top1−top2 probability margin — low = hard.
    margin1: f32,
    /// kl_t(r) vs the R_REF reference, aligned with PROBE_RS.
    kls: [f32; 4],
}

fn measure_rows(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
) -> Vec<TokenRow> {
    let ref_logits: Vec<Vec<f32>> = (0..N_PROMPTS)
        .map(|t| run_once(config, weights, residual_gate, sdpa_gate, t, R_REF))
        .collect();
    (0..N_PROMPTS)
        .map(|t| {
            let l1 = run_once(config, weights, residual_gate, sdpa_gate, t, PROBE_RS[0]);
            let l2 = run_once(config, weights, residual_gate, sdpa_gate, t, PROBE_RS[1]);
            let l4 = run_once(config, weights, residual_gate, sdpa_gate, t, PROBE_RS[2]);
            let l8 = run_once(config, weights, residual_gate, sdpa_gate, t, PROBE_RS[3]);
            let kls = [
                kl(&l1, &ref_logits[t]),
                kl(&l2, &ref_logits[t]),
                kl(&l4, &ref_logits[t]),
                kl(&l8, &ref_logits[t]),
            ];
            TokenRow {
                entropy1: entropy_nats(&l1),
                margin1: prob_margin(&l1),
                kls,
            }
        })
        .collect()
}

/// T1-parity: exact bits on aarch64 (T1's measurement platform), else a
/// 1e-5 relative tolerance + printed delta (T1's documented cross-platform
/// escape hatch).
fn assert_spectrum_parity(measured: f32, pinned: u32, r: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(
            measured.to_bits(),
            pinned,
            "harness parity: mean kl at r={r} drifted from T1's pinned spectrum"
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let expected = f32::from_bits(pinned);
        let rel = (measured - expected).abs() / expected.abs().max(1e-30);
        println!(
            "  [cross-platform] r={r}: measured {measured:.6e} vs T1-pinned {expected:.6e} (rel {rel:.2e})"
        );
        assert!(
            rel < 1e-5,
            "harness parity at r={r} beyond cross-platform tolerance: rel {rel:.2e}"
        );
    }
}

/// Order token indices hard→easy by `key`. `hard_high`: hard tokens carry
/// the HIGHER value (entropy); otherwise the lower (margin).
fn hard_to_easy(rows: &[TokenRow], key: impl Fn(&TokenRow) -> f32, hard_high: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    if hard_high {
        idx.sort_by(|&a, &b| key(&rows[b]).total_cmp(&key(&rows[a])));
    } else {
        idx.sort_by(|&a, &b| key(&rows[a]).total_cmp(&key(&rows[b])));
    }
    idx
}

/// Mean marginal gain G(k) over a token-index set.
fn mean_gain(rows: &[TokenRow], set: &[usize], k_idx: usize) -> f32 {
    let mut acc = 0.0f32;
    for &t in set {
        acc += rows[t].kls[0] - rows[t].kls[k_idx];
    }
    acc / set.len() as f32
}

// ── The gate ─────────────────────────────────────────────────────

#[test]
fn t698_t7_gate_probe_difficulty_separation() {
    let config = make_config();
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&config, &mut rng);
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let hash = fixture_hash(&config, &weights);
    // Exact on aarch64 (T1's measurement platform). Cross-platform, the
    // weight-init libm can drift in the last ulp → different weight BYTES →
    // a different hash (measured on x86_64-windows: c894478d3febdb00).
    // Recorded, not asserted, off-platform — the spectrum-parity check below
    // still pins the measurement, and the gate verdict is a same-platform
    // ratio, so ulp noise cannot move it.
    #[cfg(target_arch = "aarch64")]
    assert_eq!(hash, T1_FIXTURE_HASH, "fixture drifted from T1's pin");
    #[cfg(not(target_arch = "aarch64"))]
    if hash != T1_FIXTURE_HASH {
        println!(
            "  [cross-platform] fixture hash {hash} != T1's {T1_FIXTURE_HASH} (weight-init libm ulp drift — recorded, not asserted off-platform)"
        );
    }

    // ── G1: double-run bit-identity ─────────────────────────────
    let rows = measure_rows(&config, &weights, &residual_gate, &sdpa_gate);
    let rows_b = measure_rows(&config, &weights, &residual_gate, &sdpa_gate);
    for (a, b) in rows.iter().zip(&rows_b) {
        assert_eq!(a.entropy1.to_bits(), b.entropy1.to_bits(), "G1 entropy1");
        assert_eq!(a.margin1.to_bits(), b.margin1.to_bits(), "G1 margin1");
        for (k, r) in PROBE_RS.iter().enumerate() {
            assert_eq!(a.kls[k].to_bits(), b.kls[k].to_bits(), "G1 kl r={r}",);
        }
    }
    for row in &rows {
        assert!(row.entropy1.is_finite() && row.margin1.is_finite());
        for k in 0..4 {
            assert!(row.kls[k].is_finite() && row.kls[k] >= 0.0);
        }
    }

    // ── Harness parity with T1's pinned spectrum ────────────────
    // Mean of the per-token kl over tokens, accumulated in token order —
    // exactly T1's `measure_spectrum` reduction.
    for (k, &r) in PROBE_RS.iter().enumerate() {
        let mut acc = 0.0f32;
        for row in &rows {
            acc += row.kls[k];
        }
        let mean = acc / N_PROMPTS as f32;
        assert_spectrum_parity(mean, PINNED_AT_PROBES[k], r);
    }

    // ── Strata + separation matrix ──────────────────────────────
    // Gate = max over (metric, k). k_idx 0 is the r=1 baseline (no gain).
    let by_entropy = hard_to_easy(&rows, |r| r.entropy1, true);
    let by_margin = hard_to_easy(&rows, |r| r.margin1, false);
    let n_third = N_PROMPTS / 3;
    let metrics: [(&str, &Vec<usize>); 2] = [("entropy", &by_entropy), ("margin", &by_margin)];

    println!("\n═══ Issue 698 T7 gate probe — per-token difficulty separation ═══");
    println!(
        "  fixture blake3[16] = {hash}  ·  seed {SEED}  ·  R_REF {R_REF}  ·  prompts {N_PROMPTS}"
    );
    println!("   tok  entropy1  margin1   kl(r=1)     G(2)      G(4)      G(8)");
    for (t, row) in rows.iter().enumerate() {
        println!(
            "  {:>4}  {:>8.4}  {:>7.4}  {:>9.4e}  {:>8.3e}  {:>8.3e}  {:>8.3e}",
            t,
            row.entropy1,
            row.margin1,
            row.kls[0],
            row.kls[0] - row.kls[1],
            row.kls[0] - row.kls[2],
            row.kls[0] - row.kls[3],
        );
    }

    let mut best = (0.0f32, "", 0usize);
    let mut degenerate = false;
    for (name, order) in metrics {
        let hard = &order[..n_third];
        let easy = &order[order.len() - n_third..];
        for (k_idx, &r) in PROBE_RS.iter().enumerate().skip(1) {
            let (h, e, ratio) = {
                let hm = mean_gain(&rows, hard, k_idx);
                let em = mean_gain(&rows, easy, k_idx);
                let ratio = if em < 1e-9 {
                    degenerate = true;
                    f32::INFINITY
                } else {
                    hm / em
                };
                (hm, em, ratio)
            };
            println!(
                "  separation by {name:<7} at k={:<2}: hard {:>9.3e} / easy {:>9.3e} = {:>7.3}×",
                r,
                h,
                e,
                if ratio.is_infinite() { f32::NAN } else { ratio }
            );
            if ratio.is_finite() && ratio > best.0 {
                best = (ratio, name, k_idx);
            } else if ratio.is_infinite() {
                // an infinite ratio cannot be a finite-gate pass; record it
                println!("    (easy stratum already converged — ratio degenerate)");
            }
        }
    }

    // Extreme strata (hardest 5 vs easiest 5) as context, at k=4.
    for (name, order) in metrics {
        let hard5 = &order[..5];
        let easy5 = &order[order.len() - 5..];
        let (h, e, ratio) = {
            let hm = mean_gain(&rows, hard5, 2);
            let em = mean_gain(&rows, easy5, 2);
            let r = if em < 1e-9 { f32::NAN } else { hm / em };
            (hm, em, r)
        };
        println!(
            "  [context] extreme-5 by {name} at k=4: hard {h:.3e} / easy {e:.3e} = {}",
            if ratio.is_nan() {
                "degenerate".to_string()
            } else {
                format!("{ratio:.3}×")
            }
        );
    }

    // Rank correlations (recorded context — also T8's gate vocabulary).
    let entropies: Vec<f32> = rows.iter().map(|r| r.entropy1).collect();
    let margins: Vec<f32> = rows.iter().map(|r| r.margin1).collect();
    let g4: Vec<f32> = rows.iter().map(|r| r.kls[0] - r.kls[2]).collect();
    let total: Vec<f32> = rows.iter().map(|r| r.kls[0]).collect();
    println!(
        "  spearman(entropy, G(4)) = {:.3}   spearman(margin, G(4)) = {:.3}",
        spearman(&entropies, &g4),
        spearman(&margins, &g4)
    );
    println!(
        "  spearman(entropy, kl(1)) = {:.3}   spearman(margin, kl(1)) = {:.3}",
        spearman(&entropies, &total),
        spearman(&margins, &total)
    );

    // ── Verdict ──────────────────────────────────────────────────
    println!();
    if best.0 >= GATE_RATIO {
        println!(
            "  VERDICT: GATE PASSES — max separation {:.3}× (by {} at k={}); T7",
            best.0, best.1, PROBE_RS[best.2]
        );
        println!("  (per-token difficulty routing, decode-only) is UNBLOCKED for design work.");
    } else {
        println!(
            "  VERDICT: GATE REFUSED — max finite separation {:.3}× < {GATE_RATIO}× (by {} at k={});",
            best.0, best.1, PROBE_RS[best.2]
        );
        println!("  T7 stays deferred with this measurement on record.");
    }
    if degenerate {
        println!("  NOTE: at least one (metric, k) cell had a near-zero easy-stratum mean —");
        println!(
            "  easy tokens are already converged there; the ratio is meaningless, not ∞-favorable.",
        );
    }
    println!("  Honest scope: 27 tokens, micro config, random weights, tertile strata —");
    println!("  correlational. Per-token branching breaks batched prefill: decode-only first.");
}
