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
//! # Common protocol (all corpora; the Issue-073-T3 order — every corpus
//! pre-registered in a commit BEFORE its run)
//!
//! - Weight fixture: the T1/T2 convention — `Config::micro()` + seed-42 +
//!   `HybridPattern::Uniform` + `HlaMode::Ahla` +
//!   `LoopMode::WeightShared { loop_count: R_REF = 32 }`.
//! - **Per-input isolation:** each input is evaluated with its sequence
//!   prefix (positions < p) run at the NATURAL depth, and only the input's
//!   own position carries the probe or the fixed-k override. Cross-position
//!   exit compounding is the DEPLOYMENT reality but is excluded here —
//!   per-input attribution is the thing G2 needs; composition is riir-ai
//!   Issue 881's territory (recorded non-goal).
//! - Per-input reference: logits at the full R_REF = 32 depth.
//! - Per-input quality: cosine distance to the input's OWN reference.
//! - Per-input knee: first k in K_GRID with dist_i(k) ≤ 0.01; `undefined` if
//!   no grid k qualifies by 32.
//! - **Probe config (a-priori transfer — NOT re-tuned, ever, across all
//!   corpora):** `LoopResidualExit::new(tau = 1.0, d_min = 10)` — T2's
//!   recorded knee-parity lever. The transfer itself is part of the test.
//! - **Invariants (hard asserts):** G1 — a fed-but-never-firing probe
//!   (`d_min = usize::MAX`) is bit-identical to `None` on every input;
//!   exit ≡ elastic bit-identity for every fired input; the InterLoopNorm
//!   negative control at the T2-AMENDED boundary — every τ ≤ 3 fires ZERO
//!   inputs on the same corpus content with the stability mode swapped.
//! - **The G2 verdict (measured, not asserted):** `cut = 32 / median_all`
//!   (median over ALL inputs of iterations-used: fired k, or 32 for
//!   ran-to-32). **PASS iff cut ≥ 2× AND corpus-mean dist at exit ≤ 0.01.**
//!   p95/p99 fired depth with tail support (percentile-index discipline).
//!
//! # The sanity-bar record (read before interpreting any verdict here)
//!
//! The bar's PURPOSE, stated in the v1 pre-registration: the corpus counts as
//! heterogeneous-depth when "a per-input-safe static override must run deep
//! for everyone while the probe exits each input at its own settle point" —
//! i.e. the ADAPTIVITY MARGIN `K*/median_all ≥ 2` (K* = the corpus-safe
//! static depth: the smallest grid k with max_i dist_i(k) ≤ 0.01; degraded
//! form K* := 32 when the bound is unreachable by 32 with ≤ 10% undefined).
//! The v1/v2 pre-registrations instead encoded a PROXY for that purpose —
//! frac(knee ≥ 12) ≥ 20% AND frac(knee ≤ 8) ≥ 40% — and the proxy proved
//! miscalibrated against its own purpose:
//!
//! - v1 (sequences axis): REJECTED under the frac bar (1.1% vs 20%); its
//!   margin is ~1.2× (K* = 12 vs the d_min = 10 floor-pinned probe median
//!   recorded by T2) — genuinely non-differentiating. Both forms agree.
//! - v2 (embedding scale [1, 4, 16]): REJECTED under the frac bar (11.1% vs
//!   20%) — yet its knees (median 5, max 16) would give margin ≈ 16/10 =
//!   1.6×: also < 2, so both forms AGREE here too, and the run exposes the
//!   REAL structural finding the frac bar could not name: **the a-priori
//!   `d_min = 10` floor caps the demonstrable margin at K*/10** — a corpus
//!   whose hard tail knees below 2·d_min = 20 cannot open the G2 gate under
//!   the transferred config, no matter how the probe behaves.
//!
//! **The correction (recorded, not silent):** from corpus v3 on, the sanity
//! gate is the margin form — margin ≥ 2× AND undefined ≤ 10% — which IS the
//! pre-stated purpose; the frac statistics remain in every printout as
//! context. The correction was committed BEFORE any margin-armed run; the
//! probe config is untouched; and the correction is falsifiable from this
//! record: it changes no v1/v2 verdict (both margins < 2), it names the
//! floor-cap mechanism, and v3 is designed against that mechanism.
//!
//! # Corpus v1 — the context/sequence axis (pre-registered `4332b056`)
//!
//! 27 singles + 4 seeded random sequences (seeds 4242..4245, token =
//! `Rng::next() % 27`) × 16 positions = 91 inputs.
//!
//! **MEASURED 2026-09-07:** REJECTED (frac bar) — knees median 3, q1 2,
//! q3 5, min 2, max 12, undefined 0/91; frac(knee ≥ 12) = 1.1%. G1 held on
//! all 91 inputs; the InterLoopNorm control held at τ ≤ 3 across all 91. The
//! context axis does NOT produce heterogeneous depth on the micro fixture.
//! Kept as a determinism witness (a rerun must print the same knees).
//!
//! # Corpus v2 — the embedding-scale axis, scale set [1, 4, 16]
//! (pre-registered `284942d0`)
//!
//! The v1 doc's pre-declared next candidate. One model; token `t`'s embedding
//! row is scaled by `V2_SCALES[t % 3]` (9 tokens per tier) — three input
//! magnitudes in ONE weight fixture. Declared caveat: an input-magnitude
//! axis, not a semantic one (per-input residual scale spans orders of
//! magnitude in real models — outlier dims / attention sinks).
//!
//! **MEASURED 2026-09-07:** REJECTED (frac bar) — knees median 5, q1 4,
//! q3 8, min 3, max 16, undefined 0/27; frac(knee ≥ 12) = 11.1%. The scale
//! axis moves knees (median 3 → 5, max 12 → 16) but not past the proxy bar.
//! G1 + control held on all 27.
//!
//! # Corpus v3 — the escalated scale axis [1, 8, 64] (pre-registered HERE,
//! before its run)
//!
//! Designed against the floor-cap mechanism above: the demonstrable margin
//! needs K* ≥ 20, and v2's scale→knee leak was sublinear (16× scale → tail
//! knee 16). v3 escalates one octave per non-base tier — `V3_SCALES =
//! [1.0, 8.0, 64.0]` — to push the hard tail toward/past K* = 20–32, with
//! the SAME single-fixture, three-tier, 27-single structure and the SAME
//! a-priori probe. Pre-declared branches:
//!
//! 1. **Non-finite branch:** if any full-depth reference is non-finite (the
//!    deep tier overflows), the corpus is REJECTED before the invariants —
//!    bit-identity asserts are meaningless on NaN (NaN ≠ NaN would false-red
//!    E1), and the issue records that the axis is stability-bounded at this
//!    scale.
//! 2. **Undefined branch:** undefined knees > 10% → REJECTED (the deep tier
//!    never converges within the budget; quality is not evaluable).
//! 3. **Margin branch:** margin < 2× → REJECTED with the floor-cap note —
//!    the recorded next lever is a `d_min` reduction, which requires its own
//!    pre-registration (it trades away the T2 knee-parity guarantee), or a
//!    larger fixture family (micro may simply be too easy to demonstrate
//!    EqR-style adaptivity).
//! 4. **Gate opens:** margin ≥ 2× AND undefined ≤ 10% → the G2 verdict is
//!    evaluated and recorded.
//!
//! **MEASURED 2026-09-07 (this run):** REJECTED (margin 1.20× < 2×) — and
//! the axis is REFUTED as a difficulty axis outright: the s64 tier measured
//! the EASIEST knees (3–6) while the max knee (12) sat in the s8 tier. The
//! micro loop's convergence depth is SCALE-INVARIANT — inter-iteration
//! normalization eats the embedding scale — and the knee variation that
//! exists (3..16) is token-identity lottery, not magnitude response. The
//! floor-cap mechanism was measured on all three corpora: median_all pins at
//! the d_min = 10 floor (everything fires 10–15) while K* ∈ {12, 16} →
//! margins {v1: 1.20×, v2: 1.60×, v3: 1.20×}, all < 2.
//!
//! # Micro-fixture conclusion (the campaign's final record)
//!
//! Three pre-registered corpora, three rejections, one mechanism: quality
//! parity needs `d_min ≥ ~10` (T2: dist at k = 6 = 0.051; knee medians 3–5)
//! while the margin needs `d_min ≤ K*/2 ≈ 6–8` — unsatisfiable on this
//! fixture family. G2 requires a FIXTURE-FAMILY change (a real checkpoint or
//! a larger synthetic config with genuinely heterogeneous convergence
//! profiles); further micro corpora are pre-committed to rejection by the
//! floor-cap. Post-hoc, NON-qualified observation (recorded as input to a
//! future pre-registration, not a pass): on all three corpora the probe's
//! (median = 10, max exit dist ≈ 1.6e-3) point sat strictly off the static
//! family's Pareto frontier (static-10: max dist 2.5e-2..3.9e-1; static-12:
//! median 12, max 1.6e-3..3.5e-3) — per-input adaptivity is visible in the
//! data but was never pre-registered as a gate. Invariants held on all 145
//! inputs across the three corpora: G1 bit-identity, exit ≡ elastic, the
//! InterLoopNorm control at τ ≤ 3.
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

/// The a-priori probe config (T2's recorded knee-parity lever — NOT re-tuned,
/// on any corpus).
const PROBE_TAU: f32 = 1.0;
const PROBE_D_MIN: usize = 10;

/// The InterLoopNorm control's τ set — the T2-AMENDED boundary (τ ≤ 3).
const CONTROL_TAUS: [f32; 8] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0];

/// Corpus v2's input-magnitude tiers (pre-registered `284942d0`).
const V2_SCALES: [f32; 3] = [1.0, 4.0, 16.0];

/// Corpus v3's escalated tiers (pre-registered in this file, before the run).
const V3_SCALES: [f32; 3] = [1.0, 8.0, 64.0];

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

fn scale_token_rows(
    weights: &mut TransformerWeights,
    n_embd: usize,
    vocab: usize,
    scales: &[f32; 3],
) {
    for t in 0..vocab {
        let s = scales[t % 3];
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
/// construction (scaling included) identical. Non-capturing, hence `fn` items.
fn plain_fixture_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    make_fixture(&make_config(stability))
}

fn scaled_fixture_v2_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let config = make_config(stability);
    let (mut weights, residual_gate, sdpa_gate) = make_fixture(&config);
    scale_token_rows(&mut weights, config.n_embd, config.vocab_size, &V2_SCALES);
    (weights, residual_gate, sdpa_gate)
}

fn scaled_fixture_v3_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let config = make_config(stability);
    let (mut weights, residual_gate, sdpa_gate) = make_fixture(&config);
    scale_token_rows(&mut weights, config.n_embd, config.vocab_size, &V3_SCALES);
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

/// Corpus v2/v3: the 27 tokens as singles, one fixture, three magnitudes.
fn make_corpus_scaled(scales: &[f32; 3]) -> Vec<CorpusInput> {
    (0..27usize)
        .map(|t| CorpusInput {
            label: format!("S{t}s{}", scales[t % 3]),
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

/// The shared G2 harness: phases in the committed order — refs, A0 (finite
/// reference gate), A (knees), E1 (G1), E3 (control), B (K*), C (probe) —
/// then the margin gate, then the G2 verdict. Invariants hold regardless of
/// the corpus outcome; only the G2 machinery is gated.
fn run_g2_harness(
    label: &str,
    corpus: Vec<CorpusInput>,
    fixture_of: fn(LoopStabilityMode) -> (TransformerWeights, ResidualGate, SdpaOutputGate),
    print_inputs: bool,
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

    // ── Phase A0 — the finite-reference gate (pre-declared; v3 branch 1).
    // Bit-identity asserts are meaningless on NaN (NaN ≠ NaN would false-red
    // E1), so a diverging fixture is rejected before any invariant runs. ──
    if let Some((label, i)) = refs.iter().enumerate().find_map(|(i, r)| {
        (!r.iter().all(|v| v.is_finite())).then(|| (corpus[i].label.clone(), i))
    }) {
        println!("\n[Phase A0] reference logits NON-FINITE for {label} (input {i}) — the fixture diverges at full depth. Corpus REJECTED (pre-declared branch 1); the axis is stability-bounded at this scale. Invariants skipped (nothing finite to verify).");
        return;
    }

    // ── Phase A — per-input knees ────────────────────────────────────────
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
        println!("\n[Phase A] NO input reaches the knee bound by depth 32 (undefined {undefined}/{n}).");
    } else {
        defined.sort_unstable();
        let (k_med, _) = percentile_report(&defined, 0.50);
        let (k_q1, _) = percentile_report(&defined, 0.25);
        let (k_q3, _) = percentile_report(&defined, 0.75);
        println!("\n[Phase A] per-input knees (bound {KNEE_BOUND}): median {k_med}, q1 {k_q1}, q3 {k_q3}, min {}, max {}, undefined {undefined}/{n}", defined.first().unwrap(), defined.last().unwrap());
    }
    println!("[Phase A] frac(knee ≥ 12) = {:.1}%, frac(knee ≤ 8) = {:.1}% (context — the gate is the margin, see the module doc)", frac_ge12 * 100.0, frac_le8 * 100.0);
    if print_inputs {
        for (i, c) in corpus.iter().enumerate() {
            println!("[Phase A]   {} knee = {:?}", c.label, knees[i]);
        }
    }

    // ── Phase E1 — G1 on the corpus (hard assert; a probe invariant, not a
    // corpus one — holds regardless of heterogeneity) ─────────────────────
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
    // fixture construction (scaling included), stability mode swapped ──────
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
    let margin = k_star as f32 / median_all as f32;
    println!("\n[Phase C] probe (τ = {PROBE_TAU}, d_min = {PROBE_D_MIN}): fired {fired_count}/{n} (median-fired {median_fired}, p95 {p95} [support {sup95}], p99 {p99} [support {sup99}], max {max_fired}), ran-to-32 {ran_to_32}");
    println!("[Phase C] iterations-used median (all inputs) = {median_all} → cut vs default = {cut:.2}×; mean dist at exit = {mean_exit_dist:.6}; max dist at exit = {max_exit_dist:.6}");
    println!("[Gate] adaptivity margin = K* / median_all = {k_star}/{median_all} = {margin:.2}× (bar ≥ 2×) — the floor-cap: the margin is bounded by K*/{PROBE_D_MIN} under the a-priori d_min");

    // ── The margin gate (the corrected sanity bar — module doc record) ───
    if !(margin >= 2.0 && undefined * 10 <= n) {
        println!("\n[VERDICT] corpus REJECTED: adaptivity margin {margin:.2}× < 2× (or undefined {undefined}/{n} > 10%) — the probe cannot demonstrate ≥2× adaptivity over the corpus-safe static on this corpus. G2 not evaluated. Recorded next lever: a d_min reduction (own pre-registration) or a larger fixture family.");
        return;
    }

    // ── Phase D — the G2 verdict (measured, not asserted) ────────────────
    let g2 = cut >= 2.0 && mean_exit_dist <= KNEE_BOUND;
    println!("\n[VERDICT] G2 (≥2× median iteration cut at mean dist ≤ {KNEE_BOUND}): {}", if g2 { "PASS" } else { "FAIL" });
}

/// Corpus v1 — the context/sequence axis. MEASURED 2026-09-07 (pre-registration
/// `4332b056`): REJECTED under the frac bar — knees median 3, q1 2, q3 5,
/// max 12, 0 undefined; frac(knee ≥ 12) = 1.1%. The context axis does not
/// produce heterogeneous depth on the micro fixture. Kept as the recorded v1
/// outcome + a determinism witness (the rerun must print the same knees).
#[test]
fn bench_731_t3_corpus_v1_sequences() {
    run_g2_harness("corpus v1 — sequences", make_corpus_v1(), plain_fixture_of, false);
}

/// Corpus v2 — the embedding-scale axis [1, 4, 16] (pre-registered
/// `284942d0`). MEASURED 2026-09-07: REJECTED under the frac bar — knees
/// median 5, q1 4, q3 8, max 16, 0 undefined; frac(knee ≥ 12) = 11.1%. The
/// scale axis moves knees (3 → 5 median) but not past the proxy bar; its
/// margin (≈ 16/10) exposes the floor-cap mechanism the correction records.
#[test]
fn bench_731_t3_corpus_v2_embedding_scale() {
    run_g2_harness(
        "corpus v2 — embedding scale [1, 4, 16]",
        make_corpus_scaled(&V2_SCALES),
        scaled_fixture_v2_of,
        true,
    );
}

/// Corpus v3 — the escalated scale axis [1, 8, 64] (pre-registered in the
/// module doc BEFORE this run), designed against the floor-cap mechanism:
/// the G2 gate needs K* ≥ 2·d_min = 20, and v2's 16× tail knee fell short
/// (16). Same 27-single structure, same a-priori probe (τ = 1.0,
/// d_min = 10), corrected margin gate; branches 1–4 pre-declared in the doc.
#[test]
fn bench_731_t3_corpus_v3_embedding_scale_escalated() {
    run_g2_harness(
        "corpus v3 — embedding scale [1, 8, 64]",
        make_corpus_scaled(&V3_SCALES),
        scaled_fixture_v3_of,
        true,
    );
}
