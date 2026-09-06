#![cfg(all(feature = "lt2_looped", feature = "cadence_gate", feature = "loop_stability_fix"))]
//! Issue 731 T3 — the heterogeneous-depth corpora + the G2 GOAT gate.
//!
//! # Why (the T2 implication, verbatim input to this design)
//!
//! T2 measured the single-token micro fixture: the settle signal leads the
//! output knee by ~4 iters, every τ ≤ 3 fires 27/27 at median k = 6, and the
//! quality at k = 6 (0.051) misses the 0.01 knee bound — only a knee-scale
//! `d_min` (= 10, the recorded post-hoc lever) reaches knee parity, where the
//! probe reduces to a static knee-pinned floor with straggler adaptivity. A
//! static override is therefore EQUIVALENT on that fixture, and G2's
//! differentiated value cannot be demonstrated. G2 needs inputs that settle at
//! genuinely different depths: easy inputs (knee ~4–8) and hard inputs
//! (knee ≥ 12) in the SAME corpus, so a per-input-safe static override must
//! run deep for everyone while the probe exits each input at its own settle
//! point.
//!
//! # Common pre-registered protocol (both corpora; the Issue-073-T3 order)
//!
//! - Weight fixture: the T1/T2 convention — `Config::micro()` + seed-42 +
//!   `HybridPattern::Uniform` + `HlaMode::Ahla` +
//!   `LoopMode::WeightShared { loop_count: R_REF = 32 }`.
//! - **Per-input isolation:** each input is evaluated with its sequence
//!   prefix (positions < p) run at the NATURAL depth, and only the input's
//!   own position carries the probe or the fixed-k override. Cross-position
//!   exit compounding (an early exit at p−1 changing the cache that p attends
//!   into) is the DEPLOYMENT reality but is excluded here — per-input
//!   attribution is the thing G2 needs; composition is riir-ai Issue 881's
//!   territory (recorded non-goal).
//! - Per-input reference: logits at the full R_REF = 32 depth.
//! - Per-input quality: cosine distance to the input's OWN reference (T2's
//!   `cosine_distance`).
//! - Per-input knee: first k in K_GRID with dist_i(k) ≤ 0.01 (T2's knee
//!   bound, now per input); `undefined` if no grid k qualifies by 32.
//! - **Probe config (a-priori transfer — NOT re-tuned on either corpus):**
//!   `LoopResidualExit::new(tau = 1.0, d_min = 10)` — T2's recorded
//!   knee-parity lever. The transfer itself is part of the test: a config
//!   calibrated on the homogeneous fixture must hold on heterogeneous inputs,
//!   or the finding is recorded as a calibration-transfer failure (a GOAT
//!   verdict input, not a tuning loop).
//! - **Sanity bar (pre-registered, decides what the run MEANS):** the corpus
//!   demonstrates heterogeneous depth iff frac(knee ≥ 12) ≥ 20% AND
//!   frac(knee ≤ 8) ≥ 40% AND undefined ≤ 10%. On rejection, G2 is NOT
//!   evaluated — the recorded outcome is "corpus rejected" and the fixture
//!   search moves to the next pre-declared axis.
//! - **Invariants (hard asserts, hold regardless of the sanity outcome):**
//!   G1 — a fed-but-never-firing probe (`d_min = usize::MAX`) is bit-identical
//!   to `None` on every input; exit ≡ elastic bit-identity for every fired
//!   input; the InterLoopNorm negative control at the T2-AMENDED boundary —
//!   every τ ≤ 3 fires ZERO inputs on the same corpus content under
//!   `LoopStabilityMode::InterLoopNorm`. Any fire = the plateau regime moved
//!   — loud red, re-read before trusting any τ.
//! - **G2 verdict (measured, not asserted — the T2 precedent):** `cut =
//!   32 / median_all` where median_all is the median over ALL inputs of
//!   iterations-used (fired k, or 32 for ran-to-32). **PASS iff cut ≥ 2× AND
//!   corpus-mean dist at exit ≤ 0.01.** The adaptivity margin `K* / median_all`
//!   (K* = the corpus-safe static depth: the smallest grid k with
//!   max_i dist_i(k) ≤ 0.01; degraded form K* := 32 if the bound is
//!   unreachable by 32 with ≤ 10% undefined) is recorded as the evidence that
//!   a static override cannot match the probe on this corpus. p95/p99 fired
//!   depth reported with tail support (the percentile-index discipline).
//!
//! # Corpus v1 — the context/sequence axis (pre-registered `4332b056`)
//!
//! 27 singles + 4 seeded random sequences (seeds 4242..4245, token =
//! `Rng::next() % 27`) × 16 positions = 91 inputs.
//!
//! **MEASURED 2026-09-07 (pre-registration committed before the run):**
//! REJECTED — knees median 3, q1 2, q3 5, min 2, max 12, undefined 0/91;
//! frac(knee ≥ 12) = 1.1% (bar ≥ 20%), frac(knee ≤ 8) = 96.7%. G1 held on all
//! 91 inputs; the InterLoopNorm control held at τ ≤ 3 across all 91. The
//! context axis does NOT produce heterogeneous depth on the micro fixture —
//! every input (including deep-context positions) converges by k ≤ 12. Kept
//! as the recorded v1 outcome + a determinism witness (a rerun must print the
//! same verdict). Fixture search moved to the pre-declared v2 axis.
//!
//! # Corpus v2 — the embedding-scale difficulty axis (pre-registered HERE,
//! before its run)
//!
//! The v1 doc's pre-declared next candidate. One model; token `t`'s embedding
//! row is scaled by `SCALES[t % 3] = [1.0, 4.0, 16.0]` (9 tokens per tier) —
//! three input magnitudes in ONE weight fixture. Defensible synthetic proxy:
//! per-input residual scale spans orders of magnitude in real models (outlier
//! dims / attention sinks); the caveat is declared — this is an
//! input-magnitude axis, not a semantic one. Singles only (the sequence axis
//! was measured homogeneous in v1 — the recorded reason for dropping it).
//! Same sanity bar, same a-priori probe, same invariant asserts; the control
//! arm runs the SAME scaled fixture with the stability mode swapped.
//! Pre-declared rejection branch: if the scale-16 tier diverges (undefined
//! knees > 10%), the corpus is rejected and the issue records that the micro
//! fixture cannot hold the axis — T3 stays blocked pending a different
//! fixture family (e.g., a real checkpoint or a larger synthetic config).
//!
//! # Run
//!
//! ```bash
//! cargo test --features "lt2_looped,cadence_gate,loop_stability_fix" \
//!   --test bench_731_t3_heterogeneous_corpus -- --nocapture
//! ```

use katgpt_core::convergence_cadence::LoopResidualExit;
use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{
    Config, HlaMode, HybridPattern, LoopMode, LoopStabilityMode, ResidualGate, Rng, SdpaOutputGate,
};

const R_REF: usize = 32;
const SEED: u64 = 42;
const SEQ_SEEDS: [u64; 4] = [4242, 4243, 4244, 4245];
const SEQ_LEN: usize = 16; // Config::micro() block_size

/// Phase-A depth grid (T2's grid).
const K_GRID: [usize; 14] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32];

/// The knee bound (T2's), applied PER INPUT.
const KNEE_BOUND: f32 = 0.01;

/// The a-priori probe config (T2's recorded knee-parity lever — NOT re-tuned).
const PROBE_TAU: f32 = 1.0;
const PROBE_D_MIN: usize = 10;

/// The InterLoopNorm control's τ set — the T2-AMENDED boundary (τ ≤ 3).
const CONTROL_TAUS: [f32; 8] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0];

/// Corpus v2's input-magnitude tiers (see the module doc).
const SCALES: [f32; 3] = [1.0, 4.0, 16.0];

/// One corpus input: an optional sequence context (the prefix tokens run at
/// natural depth) plus the input's own token at position `pos`.
struct CorpusInput {
    label: String,
    /// `None` for the pos-0 singles; `Some(tokens)` for sequence positions.
    seq: Option<Vec<usize>>,
    pos: usize,
    token: usize,
}

fn make_config(stability: LoopStabilityMode) -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config.loop_stability_mode = stability;
    config
}

fn make_fixture(config: &Config) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(config, &mut rng);
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    (weights, residual_gate, sdpa_gate)
}

fn scale_token_rows(weights: &mut TransformerWeights, n_embd: usize, vocab: usize) {
    for t in 0..vocab {
        let s = SCALES[t % 3];
        if s == 1.0 {
            continue;
        }
        for v in &mut weights.wte[t * n_embd..(t + 1) * n_embd] {
            *v *= s;
        }
    }
}

/// Fixture builders parameterized by stability mode, so the harness's control
/// arm can swap the mode while keeping the corpus content and the fixture
/// construction (including v2's scaling) identical.
fn plain_fixture_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    make_fixture(&make_config(stability))
}

fn scaled_fixture_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let config = make_config(stability);
    let (mut weights, residual_gate, sdpa_gate) = make_fixture(&config);
    scale_token_rows(&mut weights, config.n_embd, config.vocab_size);
    (weights, residual_gate, sdpa_gate)
}

/// Corpus v1: 27 singles + 4 seeded sequences × 16 positions.
fn make_corpus_v1() -> Vec<CorpusInput> {
    let vocab = 27usize;
    let mut corpus: Vec<CorpusInput> = (0..vocab)
        .map(|t| CorpusInput {
            label: format!("S{t}"),
            seq: None,
            pos: 0,
            token: t,
        })
        .collect();
    for &seed in &SEQ_SEEDS {
        let mut rng = Rng::new(seed);
        let tokens: Vec<usize> = (0..SEQ_LEN).map(|_| (rng.next() as usize) % vocab).collect();
        for (p, &t) in tokens.iter().enumerate() {
            corpus.push(CorpusInput {
                label: format!("Q{seed}p{p}t{t}"),
                seq: Some(tokens.clone()),
                pos: p,
                token: t,
            });
        }
    }
    corpus
}

/// Corpus v2: the 27 tokens as singles, one fixture, three input magnitudes.
fn make_corpus_v2() -> Vec<CorpusInput> {
    (0..27usize)
        .map(|t| CorpusInput {
            label: format!("S{t}s{}", SCALES[t % 3]),
            seq: None,
            pos: 0,
            token: t,
        })
        .collect()
}

/// One `forward_looped` call for a single position: fresh ctx/caches, the
/// input's token at `pos`, optional elastic override, optional probe.
#[allow(clippy::too_many_arguments)]
fn run_one(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    pos: usize,
    elastic: Option<usize>,
    probe: Option<&mut LoopResidualExit>,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token,
        pos,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        elastic,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None,  // Issue 717: deep_run — None = bit-identical baseline
        probe, // Issue 731: the residual-exit probe (cadence_gate builds)
    )
    .to_vec()
}

/// Run the input's own position on top of its sequence prefix (positions
/// 0..pos at natural depth, fresh caches), or standalone when there is no
/// prefix — the per-input isolation semantics.
#[allow(clippy::too_many_arguments)]
fn run_on_prefix(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    prefix: Option<&[usize]>,
    pos: usize,
    token: usize,
    elastic: Option<usize>,
    probe: Option<&mut LoopResidualExit>,
) -> Vec<f32> {
    let Some(tokens) = prefix else {
        return run_one(config, weights, residual_gate, sdpa_gate, token, pos, elastic, probe);
    };
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    for (p, &tok) in tokens.iter().enumerate().take(pos) {
        forward_looped(
            &mut ctx,
            weights,
            &mut cache,
            &mut ahla_cache,
            tok,
            p,
            config,
            residual_gate,
            sdpa_gate,
            None,
            None,
            #[cfg(feature = "weight_shared_advantage_gate")]
            None,
            None,
            #[cfg(feature = "gain_cost_halt")]
            None,
            None,
            None,
        );
    }
    forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token,
        pos,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        elastic,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None,
        probe,
    )
    .to_vec()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { 1.0 - dot / denom } else { 0.0 }
}

fn median(v: &mut [usize]) -> usize {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Percentile with the tail support returned alongside (the percentile-index
/// discipline: below the 1/(1-p) boundary the index lands on the max).
fn percentile_report(sorted: &[usize], p: f64) -> (usize, usize) {
    let idx = ((sorted.len() as f64) * p) as usize;
    let idx = idx.min(sorted.len() - 1);
    (sorted[idx], sorted.len() - idx)
}

/// The shared G2 harness: phases A → E in the committed order, ending in the
/// G2 verdict. Invariants (G1, the InterLoopNorm control) hold regardless of
/// the corpus-sanity outcome; only the G2 machinery is gated on heterogeneity.
fn run_g2_harness(
    label: &str,
    corpus: Vec<CorpusInput>,
    fixture_of: fn(LoopStabilityMode) -> (TransformerWeights, ResidualGate, SdpaOutputGate),
) {
    let config = make_config(LoopStabilityMode::None);
    let (weights, residual_gate, sdpa_gate) = fixture_of(LoopStabilityMode::None);
    let n = corpus.len();
    println!("\n═══ {label}: {n} inputs ═══");

    // ── References (full depth per input, prefix at natural depth) ──────
    let refs: Vec<Vec<f32>> = corpus
        .iter()
        .map(|c| {
            run_on_prefix(
                &config, &weights, &residual_gate, &sdpa_gate,
                c.seq.as_deref(), c.pos, c.token, None, None,
            )
        })
        .collect();

    // ── Phase A — per-input knees ────────────────────────────────────────
    // knees[i] = first grid k with dist_i(k) ≤ KNEE_BOUND; None = undefined.
    let mut knees: Vec<Option<usize>> = vec![None; n];
    for &k in &K_GRID {
        for (i, c) in corpus.iter().enumerate() {
            if knees[i].is_some() {
                continue;
            }
            let out = run_on_prefix(
                &config, &weights, &residual_gate, &sdpa_gate,
                c.seq.as_deref(), c.pos, c.token, Some(k), None,
            );
            if cosine_distance(&out, &refs[i]) <= KNEE_BOUND {
                knees[i] = Some(k);
            }
        }
    }
    let mut defined: Vec<usize> = knees.iter().filter_map(|k| *k).collect();
    let undefined = n - defined.len();
    let frac_ge12 = defined.iter().filter(|&&k| k >= 12).count() as f32 / n as f32;
    let frac_le8 = defined.iter().filter(|&&k| k <= 8).count() as f32 / n as f32;
    if defined.is_empty() {
        println!("\n[Phase A] NO input reaches the knee bound by depth 32 (undefined {undefined}/{n}) — the fixture diverges or never converges. Invariants still run below; the corpus is rejected.");
    } else {
        defined.sort_unstable();
        let (k_med, _) = percentile_report(&defined, 0.50);
        let (k_q1, _) = percentile_report(&defined, 0.25);
        let (k_q3, _) = percentile_report(&defined, 0.75);
        println!("\n[Phase A] per-input knees (bound {KNEE_BOUND}): median {k_med}, q1 {k_q1}, q3 {k_q3}, min {}, max {}, undefined {undefined}/{n}", defined.first().unwrap_or(&0), defined.last().unwrap_or(&0));
    }
    println!("[Phase A] frac(knee ≥ 12) = {:.1}% (bar ≥ 20%), frac(knee ≤ 8) = {:.1}% (bar ≥ 40%)", frac_ge12 * 100.0, frac_le8 * 100.0);

    // ── Phase E1 — G1 on the corpus (hard assert; holds regardless of the
    // corpus's heterogeneity — it is a probe invariant, not a corpus one) ──
    for (i, c) in corpus.iter().enumerate() {
        let mut probe = LoopResidualExit::new(PROBE_TAU, usize::MAX);
        let with_probe = run_on_prefix(
            &config, &weights, &residual_gate, &sdpa_gate,
            c.seq.as_deref(), c.pos, c.token, None, Some(&mut probe),
        );
        assert_eq!(
            with_probe, refs[i],
            "{}: a fed-but-never-firing probe changed the logits",
            c.label
        );
        assert!(probe.fired_at_iteration().is_none());
    }
    println!("\n[Phase E1] G1: fed-but-never-firing ≡ None on all {n} inputs ✓");

    // ── Phase E3 — the InterLoopNorm negative control (hard assert; the
    // T2-amended boundary τ ≤ 3) on the SAME corpus content and the SAME
    // fixture construction (v2's scaling included), stability mode swapped ──
    {
        let control_config = make_config(LoopStabilityMode::InterLoopNorm);
        let (c_weights, c_residual_gate, c_sdpa_gate) =
            fixture_of(LoopStabilityMode::InterLoopNorm);
        for &tau in &CONTROL_TAUS {
            for c in corpus.iter() {
                let mut probe = LoopResidualExit::new(tau, PROBE_D_MIN);
                run_on_prefix(
                    &control_config, &c_weights, &c_residual_gate, &c_sdpa_gate,
                    c.seq.as_deref(), c.pos, c.token, None, Some(&mut probe),
                );
                assert_eq!(
                    probe.fired_at_iteration(),
                    None,
                    "InterLoopNorm control: tau {tau} fired on {} — the plateau regime moved; re-read the calibration before trusting any τ",
                    c.label
                );
            }
        }
    }
    println!("[Phase E3] InterLoopNorm control: τ ≤ 3 → 0/{n} fired on every τ ✓");

    // Pre-registered corpus sanity — decides what the G2 phases MEAN. The
    // invariants above hold (and were held) either way; only the G2
    // machinery is gated on the corpus being heterogeneous-depth.
    let corpus_heterogeneous = frac_ge12 >= 0.20 && frac_le8 >= 0.40 && undefined * 10 <= n;
    if !corpus_heterogeneous {
        println!("\n[VERDICT] corpus REJECTED: knees are not heterogeneous-depth (see the pre-registered sanity bar in the module doc). G2 not evaluated.");
        return;
    }

    // ── Phase B — the corpus-safe static override K* ─────────────────────
    let mut k_star: Option<usize> = None;
    for &k in &K_GRID {
        let mut worst = 0.0f32;
        for (i, c) in corpus.iter().enumerate() {
            let out = run_on_prefix(
                &config, &weights, &residual_gate, &sdpa_gate,
                c.seq.as_deref(), c.pos, c.token, Some(k), None,
            );
            worst = worst.max(cosine_distance(&out, &refs[i]));
        }
        println!("[Phase B] static k = {k}: max_i dist = {worst:.6}");
        if worst <= KNEE_BOUND {
            k_star = Some(k);
            break;
        }
    }
    let k_star = k_star.unwrap_or(R_REF); // degraded form: K* := 32
    println!("[Phase B] K* = {k_star} (the corpus-safe static depth)");

    // ── Phase C — the probe arm (a-priori config, no re-tuning) ──────────
    let mut used: Vec<usize> = Vec::with_capacity(n); // fired k or 32
    let mut fired: Vec<usize> = Vec::new();
    let mut dist_at_exit: Vec<f32> = Vec::with_capacity(n);
    let mut ran_to_32 = 0usize;
    for (i, c) in corpus.iter().enumerate() {
        let mut probe = LoopResidualExit::new(PROBE_TAU, PROBE_D_MIN);
        let exited = run_on_prefix(
            &config, &weights, &residual_gate, &sdpa_gate,
            c.seq.as_deref(), c.pos, c.token, None, Some(&mut probe),
        );
        dist_at_exit.push(cosine_distance(&exited, &refs[i]));
        match probe.fired_at_iteration() {
            Some(k) => {
                // Phase E2 (hard assert): exit ≡ elastic, bit-identical.
                let elastic = run_on_prefix(
                    &config, &weights, &residual_gate, &sdpa_gate,
                    c.seq.as_deref(), c.pos, c.token, Some(k), None,
                );
                assert_eq!(
                    exited, elastic,
                    "{}: exit-at-{k} diverged from elastic = {k}",
                    c.label
                );
                used.push(k);
                fired.push(k);
            }
            None => {
                used.push(R_REF);
                ran_to_32 += 1;
            }
        }
    }
    let mean_exit_dist = dist_at_exit.iter().sum::<f32>() / n as f32;
    let fired_count = fired.len();
    let max_exit_dist = dist_at_exit.iter().copied().fold(0.0, f32::max);
    let median_all = median(&mut used);
    let mut fired_sorted = fired.clone();
    let median_fired = if fired_sorted.is_empty() { 0 } else { median(&mut fired_sorted) };
    let (p95, sup95) = if fired_sorted.is_empty() { (0, 0) } else { percentile_report(&fired_sorted, 0.95) };
    let (p99, sup99) = if fired_sorted.is_empty() { (0, 0) } else { percentile_report(&fired_sorted, 0.99) };
    let max_fired = fired_sorted.last().copied().unwrap_or(0);
    let cut = R_REF as f32 / median_all as f32;
    let adaptivity = k_star as f32 / median_all as f32;
    println!("\n[Phase C] probe (τ = {PROBE_TAU}, d_min = {PROBE_D_MIN}): fired {fired_count}/{n} (median-fired {median_fired}, p95 {p95} [support {sup95}], p99 {p99} [support {sup99}], max {max_fired}), ran-to-32 {ran_to_32}");
    println!("[Phase C] iterations-used median (all inputs) = {median_all} → cut = {cut:.2}×; mean dist at exit = {mean_exit_dist:.6}; max dist at exit = {max_exit_dist:.6}");
    println!("[Phase C] adaptivity margin: K* = {k_star} / median_all = {adaptivity:.2}× (what a per-input-safe static override cannot save)");

    // ── Phase D — the G2 verdict (measured, not asserted) ────────────────
    let g2 = cut >= 2.0 && mean_exit_dist <= KNEE_BOUND;
    println!("\n[VERDICT] G2 (≥2× median iteration cut at mean dist ≤ {KNEE_BOUND}): {}", if g2 { "PASS" } else { "FAIL" });
}

/// Corpus v1 — the context/sequence axis. MEASURED 2026-09-07 (pre-registration
/// `4332b056`, committed before the run): REJECTED — knees median 3, q1 2,
/// q3 5, max 12, 0 undefined; frac(knee ≥ 12) = 1.1% (bar ≥ 20%). The context
/// axis does not produce heterogeneous depth on the micro fixture. Kept as the
/// recorded v1 outcome + a determinism witness (the rerun must print the same
/// verdict).
#[test]
fn bench_731_t3_corpus_v1_sequences() {
    run_g2_harness("corpus v1 — sequences", make_corpus_v1(), plain_fixture_of);
}

/// Corpus v2 — the embedding-scale difficulty axis (the v1 doc's pre-declared
/// next candidate, pre-registered in the module doc BEFORE this run). Same
/// sanity bar, same a-priori probe (τ = 1.0, d_min = 10 — the transfer test),
/// same invariant asserts; the control arm runs the SAME scaled fixture with
/// the stability mode swapped. Pre-declared rejection branch: if the scale-16
/// tier diverges (undefined knees > 10%), the corpus is rejected and the issue
/// records that the micro fixture cannot hold the axis — T3 stays blocked
/// pending a different fixture family.
#[test]
fn bench_731_t3_corpus_v2_embedding_scale() {
    run_g2_harness(
        "corpus v2 — embedding scale [1, 4, 16]",
        make_corpus_v2(),
        scaled_fixture_of,
    );
}
