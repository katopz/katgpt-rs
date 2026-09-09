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
//! # Corpus v4 — the loop-weight-scale fixture family (the named lever,
//! pre-registered HERE, before its run)
//!
//! This is the FIXTURE-FAMILY change the conclusion above demands — not
//! another micro corpus (the config stays micro-SHAPED; "varied" sufficed,
//! "larger" was not needed). Why the axis exists at all: for a single token
//! with no prefix, attention over one key is trivially linear, so the loop
//! map h ← h + F(ĥ) is SHARED by all 27 tokens and only z0 = wte[t] +
//! wpe[0] differs — that is why knee variation was token-identity lottery.
//! The lever moves the MAP itself: scale the loop weights (the six attn/mlp
//! projections) by α. Larger α pushes the iteration's Jacobian toward the
//! oscillatory boundary; tokens whose z0 excites the slow modes become
//! genuinely hard (knee 20) while the rest stay easy — heterogeneous depth
//! in ONE fixture, the thing G2 needs. Embeddings and lm_head are untouched
//! (cosine quality is scale-free; the inter-iteration normalization eats
//! input-side scale, which is exactly why v3's axis was flat).
//!
//! **Selection disclosure (record, not concealment):** the family was found
//! by an EXPLORATORY scan — 144 fixture evaluations over {seed} × {α ∈
//! 1.0..10} — whose outcome variable WAS the G2 margin. That is an
//! existence-proof campaign by design: the conclusion names "a larger/varied
//! synthetic config whose per-input knees span ≥ 2× the quality-safe floor"
//! as the lever, and shopping for the family is how that lever is realized.
//! Scan structure (all rows deterministic):
//!
//! - Wave 1 (seeds 42/7/1234 × α 1..10): seed 42 gets FASTER with α (max
//!   knee 12 → 6); seed 7 is uniformly deep and the probe NEVER fires (the
//!   settle signal stays above threshold — churn, not convergence); seed
//!   1234 α 2 margin 1.09×. No candidate.
//! - Wave 2 (14 seeds × α {1, 1.5, 2, 2.5, 3}): the family is bimodal —
//!   contractive fixtures floor-pin at margin ≈ K*/10 ≤ 1.6×; churny
//!   fixtures never fire (margin 1.0×). Aggressive mixtures reach margin > 2
//!   but poison quality via floor-fires on hard inputs: seed 31337 α 1
//!   margin 3.2× / mean dist 0.180; 271828 α 1: 2.8× / 0.026; 21 α 1: 2.4× /
//!   0.019; 2 α 2.5: 2.13× / 0.014; 555 α 1: 2.18× / 0.010034 — fails by
//!   3.4e-5. Exactly one family passes everything.
//! - Wave 3 (α 2.6..4.0 on the five live seeds): the window is
//!   α ∈ [3.0, ~3.15] on seed 5; α 3.2 drops the tail knee back to 16
//!   (margin 1.6×).
//!
//! **Structural finding the scan surfaces: the weight-scale axis's margin
//! ceiling is EXACTLY 2.0×.** Every margin ≥ 2 fixture sits at exactly
//! K* = 2·d_min; a deeper tail (K* ≥ 24) always co-occurs with < 50%
//! floor-fires (the median lifts off the floor) or a poisoned mean. The G2
//! bar is therefore MET, never EXCEEDED, on this axis — the evidence-grade
//! caveat this record carries.
//!
//! > **REFUTED 2026-09-07 by the T6 held-out replication (§T6, P1) — read the
//! > paragraph above as the scan's claim, not as a fact.** Held-out seed 1002
//! > measures margin **2.40×** (K* 24 / median_all 10) at mean exit dist
//! > 5.44e-3 ≤ 0.01 — a K* ≥ 24 tail that does NOT co-occur with a lifted
//! > median or a poisoned mean, which is exactly the co-occurrence the
//! > paragraph asserts is universal. The "ceiling" was an artifact of the
//! > 144-row scan's COVERAGE, not a property of the weight-scale axis: the
//! > scan simply never drew a seed like 1002. (Post-T6-fix the same fixture
//! > measures 2.18× at a 2.4× better mean dist — still above 2.0×.) The
//! > evidence-grade caveat on v4 survives the refutation for a different
//! > reason: P2 landed inside its pre-declared band (1/12), so v4 remains an
//! > existence proof — it is no longer an existence proof *at a ceiling*.
//!
//! ## The v4 pre-registration
//!
//! - Fixture: seed **5**, loop-weight scale **α = 3.0** (the rounder window
//!   point; α 3.1 is the sensitivity neighbor — same margin, max exit dist
//!   0.0119 vs 0.0160).
//! - Corpus: the 27 micro tokens as singles (no input-side scaling).
//! - Probe: the a-priori transfer `τ = 1.0, d_min = 10` — NOT re-tuned.
//! - Gates: unchanged (margin ≥ 2× AND undefined ≤ 10%; G2 = cut ≥ 2× at
//!   mean dist ≤ 0.01).
//! - Control pre-checked as fixture design, BEFORE this commit: the
//!   InterLoopNorm control fires 0 across 8 τ values ≤ 3 on the α-scaled
//!   fixture (both window points).
//! - **Falsifiable prediction** (deterministic pipeline, so the run must
//!   reproduce the scan bit-exactly; any divergence indicts the scan or the
//!   wiring): knees median 8 / max 20, undefined 0/27; K* = 20; the probe
//!   fires 27/27 — 19 at the d_min = 10 floor + 8 late fires in 11..17 (the
//!   hard tail firing at ≈ its knee); median_all = 10; margin = 2.00×; mean
//!   dist at exit ≈ 8.7e-4, max ≈ 1.6e-2; **G2 PASS** (cut = 3.2×).
//!
//! **MEASURED 2026-09-07 (this run): the prediction reproduced bit-exactly —
//! G2 PASS.** Knees median 8 / q1 6 / q3 10 / min 4 / max 20, undefined
//! 0/27 (S14 = 20 the hard tail; S20 = 16; S5/S9 = 12); frac(knee ≥ 12) =
//! 14.8%. K* = 20. The probe fired 27/27: 19 at the d_min = 10 floor + 8
//! late fires [11 ×4, 13 ×2, 15, 17] — the hard tail firing at ≈ its knee
//! (S14: fire 17 vs knee 20, exit dist 0.0160 = the corpus max). median_all
//! = 10 → cut 3.20×, margin 2.00× (the bar met EXACTLY — the axis ceiling),
//! mean dist at exit 8.66e-4 ≤ 0.01, max 0.0160. Invariants: G1
//! fed-but-never-firing ≡ None 27/27; exit ≡ elastic bit-identity per fired
//! input; the InterLoopNorm control 0/27 at every τ ≤ 3. v1–v3 re-ran
//! identical (determinism witnesses). p99 fired depth 17 [support 1].
//!
//! **Verdict + evidence grade:** G2's pre-registered bar is MET — a ≥2×
//! median iteration cut at quality parity with per-input adaptivity visible
//! in the exit-depth distribution (floor-pinned easy majority, knee-tracking
//! hard tail). It is an EXISTENCE proof on a scan-selected synthetic
//! fixture, at the axis's measured margin ceiling (exactly 2.0×) — not
//! robustness evidence. T4 stays blocked: `cadence_gate` remains opt-in and
//! the probe slot is caller-owned (`None` = bit-identical), so default-on
//! would not change runtime behavior; promotion waits on real-workload
//! depth-spread evidence a synthetic micro campaign cannot produce. G1 held
//! in-harness; G4 is the T1 fixed-ring contract (`[f32; 4]`, no
//! steady-state allocation); p99 worst-case depth reported above.
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
/// discipline: nearest rank `ceil(p·n) − 1`, and below the 1/(1−p) boundary
/// the reported rank still sits at/near the max — the support column is the
/// honest statistic).
fn percentile_report(sorted: &[usize], p: f64) -> (usize, usize) {
    let idx = (((sorted.len() as f64) * p).ceil() as usize).saturating_sub(1);
    let idx = idx.min(sorted.len() - 1);
    (sorted[idx], sorted.len() - idx)
}

/// The harness's machine-readable outcome (Issue 731 T6). A println-only
/// harness cannot be AGGREGATED across fixtures, which is exactly what the
/// held-out replication needs — every field below is a quantity the v1-v4
/// records already print. v1-v4 ignore the return; their printed record and
/// their asserts are unchanged.
#[derive(Debug, Clone)]
struct G2Report {
    /// Inputs in the corpus.
    n: usize,
    /// `true` = Phase A0 rejected the fixture (non-finite full-depth
    /// reference). EVERY other field is meaningless when this is set, and no
    /// invariant ran — a diverged fixture is neither a pass nor a fail.
    diverged: bool,
    /// Inputs with no knee by depth `R_REF`.
    undefined: usize,
    /// Max per-input knee over the DEFINED ones (0 when none is defined).
    knee_max: usize,
    /// The corpus-safe static depth (Phase B); `R_REF` when none qualifies.
    k_star: usize,
    /// Median iterations used over ALL inputs (fired k, else `R_REF`).
    median_all: usize,
    /// `k_star / median_all` — the adaptivity margin (bar ≥ 2×).
    margin: f32,
    /// `R_REF / median_all` — the iteration cut vs the fixed default.
    cut: f32,
    /// Mean cosine distance to the full-depth reference at exit.
    mean_exit_dist: f32,
    /// Max cosine distance to the full-depth reference at exit.
    max_exit_dist: f32,
    /// Probe fire count.
    fired: usize,
    /// The margin gate's verdict (`false` = corpus REJECTED, G2 NOT evaluated).
    margin_gate: bool,
    /// The G2 verdict. `false` whenever `margin_gate` is false — read the two
    /// together: `!margin_gate` means NOT EVALUATED, never FAIL.
    g2: bool,
    /// Phase E3 — (τ, input, fired-at) for every InterLoopNorm control fire.
    /// MUST be empty: a fire is the Research-440 trap reading a churning loop
    /// as converged. Reported rather than asserted in-place (Issue 731 T6) so
    /// a multi-fixture sweep records its WHOLE table instead of stopping at
    /// the first violating fixture — every caller asserts it is empty.
    control_fires: Vec<(f32, String, usize)>,
}

impl G2Report {
    /// The Phase-A0 branch: the fixture diverges at full depth.
    fn diverged(n: usize) -> Self {
        Self {
            n,
            diverged: true,
            undefined: n,
            knee_max: 0,
            k_star: R_REF,
            median_all: R_REF,
            margin: 1.0,
            cut: 1.0,
            mean_exit_dist: f32::NAN,
            max_exit_dist: f32::NAN,
            fired: 0,
            margin_gate: false,
            g2: false,
            control_fires: Vec::new(),
        }
    }
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
) -> G2Report {
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
        return G2Report::diverged(n);
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
    let knee_max = defined.iter().copied().max().unwrap_or(0);
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
    let mut control_fires: Vec<(f32, String, usize)> = Vec::new();
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
                if let Some(k) = probe.fired_at_iteration() {
                    control_fires.push((tau, c.label.clone(), k));
                }
            }
        }
    }
    if control_fires.is_empty() {
        println!("[Phase E3] InterLoopNorm control: τ ≤ 3 → 0/{n} fired on every τ ✓");
    } else {
        println!(
            "[Phase E3] InterLoopNorm control VIOLATED: {} fire(s) over {} τ × {n} inputs — the Research-440 trap read a churning loop as converged. First: {:?}",
            control_fires.len(),
            CONTROL_TAUS.len(),
            control_fires.first().unwrap()
        );
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
    let margin = k_star as f32 / median_all as f32;
    println!("\n[Phase C] probe (τ = {PROBE_TAU}, d_min = {PROBE_D_MIN}): fired {fired_count}/{n} (median-fired {median_fired}, p95 {p95} [support {sup95}], p99 {p99} [support {sup99}], max {max_fired}), ran-to-32 {ran_to_32}");
    println!("[Phase C] iterations-used median (all inputs) = {median_all} → cut vs default = {cut:.2}×; mean dist at exit = {mean_exit_dist:.6}; max dist at exit = {max_exit_dist:.6}");
    println!("[Gate] adaptivity margin = K* / median_all = {k_star}/{median_all} = {margin:.2}× (bar ≥ 2×) — the floor-cap: the margin is bounded by K*/{PROBE_D_MIN} under the a-priori d_min");

    // ── The margin gate (the corrected sanity bar — module doc record) ───
    let margin_gate = margin >= 2.0 && undefined * 10 <= n;
    let mut report = G2Report {
        n,
        diverged: false,
        undefined,
        knee_max,
        k_star,
        median_all,
        margin,
        cut,
        mean_exit_dist,
        max_exit_dist,
        fired: fired_count,
        margin_gate,
        g2: false,
        control_fires,
    };
    if !margin_gate {
        println!("\n[VERDICT] corpus REJECTED: adaptivity margin {margin:.2}× < 2× (or undefined {undefined}/{n} > 10%) — the probe cannot demonstrate ≥2× adaptivity over the corpus-safe static on this corpus. G2 not evaluated. Recorded next lever: a d_min reduction (own pre-registration) or a larger fixture family.");
        return report;
    }

    // ── Phase D — the G2 verdict (measured, not asserted) ────────────────
    let g2 = cut >= 2.0 && mean_exit_dist <= KNEE_BOUND;
    println!("\n[VERDICT] G2 (≥2× median iteration cut at mean dist ≤ {KNEE_BOUND}): {}", if g2 { "PASS" } else { "FAIL" });
    report.g2 = g2;
    report
}

/// Corpus v1 — the context/sequence axis. MEASURED 2026-09-07 (pre-registration
/// `4332b056`): REJECTED under the frac bar — knees median 3, q1 2, q3 5,
/// max 12, 0 undefined; frac(knee ≥ 12) = 1.1%. The context axis does not
/// produce heterogeneous depth on the micro fixture. Kept as the recorded v1
/// outcome + a determinism witness (the rerun must print the same knees).
#[test]
fn bench_731_t3_corpus_v1_sequences() {
    let report = run_g2_harness("corpus v1 — sequences", make_corpus_v1(), plain_fixture_of, false);
    // Phase E3 is a GATE, not a measurement (Issue 731 T6 moved the assert
    // out of the harness so a sweep records its whole table).
    assert!(
        report.control_fires.is_empty(),
        "InterLoopNorm control fired {} time(s) — the plateau regime moved; re-read the calibration before trusting any τ: {:?}",
        report.control_fires.len(),
        report.control_fires
    );
}

/// Corpus v2 — the embedding-scale axis [1, 4, 16] (pre-registered
/// `284942d0`). MEASURED 2026-09-07: REJECTED under the frac bar — knees
/// median 5, q1 4, q3 8, max 16, 0 undefined; frac(knee ≥ 12) = 11.1%. The
/// scale axis moves knees (3 → 5 median) but not past the proxy bar; its
/// margin (≈ 16/10) exposes the floor-cap mechanism the correction records.
#[test]
fn bench_731_t3_corpus_v2_embedding_scale() {
    let report = run_g2_harness(
        "corpus v2 — embedding scale [1, 4, 16]",
        make_corpus_scaled(&V2_SCALES),
        scaled_fixture_v2_of,
        true,
    );
    // Phase E3 is a GATE, not a measurement (Issue 731 T6 moved the
    // assert out of the harness so a sweep records its whole table).
    assert!(
        report.control_fires.is_empty(),
        "InterLoopNorm control fired {} time(s) — the plateau regime moved; re-read the calibration before trusting any τ: {:?}",
        report.control_fires.len(),
        report.control_fires
    );
}

/// Corpus v3 — the escalated scale axis [1, 8, 64] (pre-registered in the
/// module doc BEFORE this run), designed against the floor-cap mechanism:
/// the G2 gate needs K* ≥ 2·d_min = 20, and v2's 16× tail knee fell short
/// (16). Same 27-single structure, same a-priori probe (τ = 1.0,
/// d_min = 10), corrected margin gate; branches 1–4 pre-declared in the doc.
#[test]
fn bench_731_t3_corpus_v3_embedding_scale_escalated() {
    let report = run_g2_harness(
        "corpus v3 — embedding scale [1, 8, 64]",
        make_corpus_scaled(&V3_SCALES),
        scaled_fixture_v3_of,
        true,
    );
    // Phase E3 is a GATE, not a measurement (Issue 731 T6 moved the
    // assert out of the harness so a sweep records its whole table).
    assert!(
        report.control_fires.is_empty(),
        "InterLoopNorm control fired {} time(s) — the plateau regime moved; re-read the calibration before trusting any τ: {:?}",
        report.control_fires.len(),
        report.control_fires
    );
}

//-──────────────────────────────────────────────────────────────────────────
// Corpus v4 — the loop-weight-scale fixture family (the issue's named
// "larger/varied synthetic config" lever).
//-──────────────────────────────────────────────────────────────────────────

/// Corpus v4's fixture family: weight-seed 5 + loop-weight scale `V4_ALPHA`.
const V4_SEED: u64 = 5;

/// Corpus v4's loop-weight scale (attn + mlp projections; embeddings and
/// lm_head untouched).
const V4_ALPHA: f32 = 3.0;

/// The v4 fixture builder: seed 5 + α-scaled LOOP weights. Both stability
/// modes build the same way (the E3 control arm swaps only the mode —
/// pre-verified 0-fires on this family BEFORE the pre-registration commit).
fn scaled_fixture_v4_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    build_alpha_scaled(V4_SEED, V4_ALPHA, stability)
}

/// The v4 fixture family's body, parameterized by seed (Issue 731 T6 needs
/// the SAME construction on held-out seeds — a second copy of it would make
/// the replication test a different fixture family, which is the one thing a
/// replication may not be).
fn build_alpha_scaled(
    seed: u64,
    alpha: f32,
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let config = make_config(stability);
    let mut rng = Rng::new(seed);
    let mut weights = TransformerWeights::new(&config, &mut rng);
    for layer in &mut weights.layers {
        for w in [
            &mut layer.attn_wq,
            &mut layer.attn_wk,
            &mut layer.attn_wv,
            &mut layer.attn_wo,
            &mut layer.mlp_w1,
            &mut layer.mlp_w2,
        ] {
            for v in w.iter_mut() {
                *v *= alpha;
            }
        }
    }
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    (weights, residual_gate, sdpa_gate)
}

/// Corpus v4 — the loop-weight-scale fixture family (pre-registered in the
/// module doc BEFORE this run, prediction included).
#[test]
fn bench_731_t3_corpus_v4_loop_weight_scale() {
    let corpus: Vec<CorpusInput> = (0..27usize)
        .map(|t| CorpusInput {
            label: format!("S{t}"),
            seq: None,
            pos: 0,
            token: t,
        })
        .collect();
    let report = run_g2_harness(
        "corpus v4 — loop-weight scale (seed 5, α 3.0)",
        corpus,
        scaled_fixture_v4_of,
        true,
    );
    // Phase E3 is a GATE, not a measurement (Issue 731 T6 moved the
    // assert out of the harness so a sweep records its whole table).
    assert!(
        report.control_fires.is_empty(),
        "InterLoopNorm control fired {} time(s) — the plateau regime moved; re-read the calibration before trusting any τ: {:?}",
        report.control_fires.len(),
        report.control_fires
    );
}

//-──────────────────────────────────────────────────────────────────────────
// T6 — the HELD-OUT replication of the v4 loop-weight-scale mechanism
// (pre-registered HERE, in the commit BEFORE its run)
//-──────────────────────────────────────────────────────────────────────────
//
// # Why this exists
//
// v4's G2 PASS is self-disclosed as **existence-proof grade**: the fixture
// (seed 5, α 3.0) was picked by a 144-evaluation scan whose outcome variable
// WAS the G2 margin, and the margin landed at EXACTLY the bar (2.00×). That
// is precisely the evidence-grade caveat T4 defers on. Two claims the v4
// record makes are testable OUT OF SAMPLE at zero extra design freedom —
// same builder (`build_alpha_scaled`), same α, same untuned probe
// (τ = 1.0, d_min = 10), same gates, only the weight seed changes:
//
// 1. **the ceiling claim** — "the weight-scale axis's margin ceiling is
//    EXACTLY 2.0×; every margin ≥ 2 fixture sits at K* = 2·d_min".
// 2. **the rarity claim** — the caveat's force comes from the pass being
//    rare (1 family in 144 scan rows).
//
// A replication can only strengthen or weaken those; it cannot re-tune
// anything, because nothing here is tunable.
//
// # The held-out seed set
//
// **A recorded reproducibility gap:** the scan's full seed list was never
// committed (it ran out-of-tree; the module doc names 9 of the ~17 seeds it
// touched — 42, 7, 1234, 5, 2, 21, 555, 31337, 271828). "Held out" is
// therefore a claim relative to the RECORD, not a proof. It is made
// structurally as strong as the record permits: the set is defined
// MECHANICALLY as the contiguous run `1001..=1012` — disjoint from every
// seed the scan record names, from this file's `SEED`/`SEQ_SEEDS`, and from
// the memorable-constant style the scan sampled in. `t6_holdout_seeds_are_
// disjoint_from_the_recorded_scan_seeds` asserts the disjointness so the
// claim cannot silently rot.
//
// # Pre-registered predictions
//
// - **P1 (ceiling, the primary):** NO held-out fixture attains
//   `margin > 2.0×` while `mean_exit_dist ≤ 0.01`. A refutation is GOOD news
//   for the probe — it would mean the 2.0× ceiling is an artifact of the
//   scan's coverage and G2 has headroom the v4 record denies it.
// - **P2 (rarity):** `≤ 2 of 12` held-out seeds produce a full G2 PASS
//   (margin ≥ 2.0× AND cut ≥ 2× AND mean dist ≤ 0.01). Pre-declared reading
//   of the outcome, committed before the data:
//     * `0/12` → v4 confirmed as a lottery draw. The synthetic axis is
//       CLOSED (as T3's conclusion already suspected) and T4's "needs
//       real-workload evidence" defer is CORROBORATED, not merely asserted.
//     * `1..=2 / 12` → consistent with the scan's own base rate; the
//       existence-proof grade stands, unchanged.
//     * `≥ 3 / 12` → the "scan-selected, one-in-144" caveat is OVERSTATED:
//       the mechanism generalizes across the seed lottery at fixed α, and
//       G2's grade upgrades from existence-proof to **replicated
//       out-of-sample** on the synthetic axis.
// - **P3 (the invariants — the part that must hold unconditionally):** on
//   EVERY held-out fixture that passes the Phase-A0 finite-reference gate,
//   G1 (fed-but-never-firing ≡ `None`), exit ≡ elastic bit-identity, and the
//   InterLoopNorm control (0 fires at all 8 τ ≤ 3) hold. These are hard
//   asserts inside `run_g2_harness`, so the test FAILS on violation — they
//   are the probe's correctness contract and are seed-independent by
//   construction. This is the half of T6 that is a gate rather than a
//   measurement, and the half whose value does not depend on the G2 outcome.
// - **Admissible non-outcome:** α = 3.0 on an unlucky seed may diverge at
//   full depth (non-finite reference). Phase A0 rejects those; they are
//   reported as DIVERGED and excluded from P1/P2's denominator, which is
//   reported explicitly. A diverged fixture is neither a pass nor a fail.
//
// # MEASURED 2026-09-07 — P1 REFUTED · P2 1/12 · P3 VIOLATED then FIXED
//
// All 12 held-out fixtures were live (0 diverged at α 3.0), 324 inputs.
//
// - **P1 — REFUTED, and that is the good outcome.** Held-out seed **1002**
//   measured margin **2.40×** (K* 24 / median_all 10) at mean exit dist
//   5.44e-3 ≤ 0.01 — strictly ABOVE the 2.0× the v4 record called the axis's
//   exact ceiling. The ceiling was an artifact of the scan's coverage, not a
//   property of the weight-scale axis, so G2 has headroom the v4 record
//   denied it. (Post-fix the same fixture measures margin 2.18× / cut 2.91×
//   at a 2.4× BETTER mean dist — still above the ceiling, still a G2 PASS.)
// - **P2 — 1/12 G2 PASS (seed 1002).** Pre-declared reading, taken as
//   written: consistent with the scan's base rate ⇒ **the existence-proof
//   grade STANDS, unchanged.** Non-qualified note (input to a future
//   pre-registration, NOT a pass): 1/12 at FIXED α = 3.0 is not comparable to
//   the scan's 1/144, which ranged over {seed} × {α} — the two denominators
//   count different things, so no rate claim is made here.
// - **P3 — VIOLATED on seed 1003: the InterLoopNorm negative control fired
//   40 times over 8 τ × 27 inputs, first at (τ = 0.001, S2, k = 14).** τ
//   cannot explain a fire at the SMALLEST τ in the set, and it did not: with
//   τ = 0.0 (magnitude arm mathematically disabled) the fire is identical, so
//   it is the SHAPE arm. Localized further — the fire is invariant from
//   `settle_floor` 0.5 down to 1e-5 and vanishes only at
//   `decay_ratio_max ≤ 0.3`, i.e. it is `classify`'s **rule-3 decay
//   fall-through** on a newer/older half-window ratio in [0.3, 0.5): a
//   transient 2-vs-2 DIP inside a plateau whose newer half stays above 0.5.
//   5/27 inputs false-positived via the shape arm alone.
//
//   **What P3 falsified:** (i) the T1 record's "guarded by construction"
//   claim — the arms are OR'd and rule 3 has no absolute floor, so arm 2 can
//   fire alone on a churning loop; (ii) T2's amendment of the control
//   boundary from τ ≤ 10 to τ ≤ 3, which treated a shape-arm false positive
//   as if τ bounded it — the τ ≤ 3 boundary was fixture luck; (iii) the
//   sufficiency of T5's calibration rule, which governs rules 1-2 only.
//
//   **Fix-forward (same session):** `LoopResidualExit::with_shape_persistence`
//   — the shape arm requires `DEFAULT_SHAPE_PERSISTENCE = 2` CONSECUTIVE
//   `Settled` windows (a `None` or `Churning` resets the run). Orthogonal to
//   calibration, one `u32` counter, zero-alloc. `with_shape_persistence(1)`
//   recovers the pre-T6 behavior exactly and is retained as the control arm
//   (no loser to demote). Pinned both directions by
//   `t6_seed_1003_control_violation_is_pinned_both_directions` (the defect
//   must still reproduce at persistence 1) and four katgpt-core unit tests.
//
// **The fix is a strict improvement on every corpus measured** — same
// verdicts, better quality, control clean:
//
// | corpus | verdict | margin | cut | mean dist | max dist | E3 control |
// |---|---|---|---|---|---|---|
// | v1 / v2 / v3 | REJECTED (unchanged) | 1.20× / 1.60× / 1.20× | — | — | — | clean |
// | v4 (seed 5) | **G2 PASS** (unchanged) | 2.00× | 3.20× | 8.66e-4 → **1.02e-4** | 1.60e-2 → **1.31e-3** | clean |
// | T6 seed 1002 | **G2 PASS** | 2.40× → 2.18× | 3.20× → 2.91× | 5.44e-3 → **2.23e-3** | 2.15e-2 → **1.26e-2** | clean |
// | T6 seed 1003 | REJECTED | 1.00× | — | — | — | **40 fires → 0** |
// | T6 1004/1005/1010/1011 | REJECTED | <2× | — | 9.2e-2/8.1e-2/8.9e-2/2.4e-2 → **0/0/4.2e-3/1.3e-2** | — | clean |
//
// v4's cut and margin are IDENTICAL post-fix while its worst-case exit
// distance improves 12×; the premature dip-triggered exits that were
// poisoning quality on four held-out fixtures are gone. So the fix removes a
// false-convergence class without costing any measured adaptivity.
//
// **What this does and does not do for T4.** It repairs a probe defect and
// refutes a structural claim, and P3's zero-fire control now covers 4 + 12
// fixtures / 496 inputs instead of 4 / 172. It does NOT change T4's status:
// P2 came out inside the pre-declared band, so the synthetic G2 evidence is
// still existence-proof grade, and T4's unblock remains real-workload
// depth-spread evidence at judgeable quality.

/// The held-out weight seeds — mechanically defined, disjoint from every
/// seed the v4 scan record names.
const T6_HOLDOUT_SEEDS: [u64; 12] = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
];

/// Every weight seed the v4 scan record names (module doc, waves 1-3) plus
/// this file's own fixture seeds — the set T6 must not draw from.
const T6_RECORDED_SCAN_SEEDS: [u64; 12] = [
    42, 7, 1234, 5, 2, 21, 555, 31337, 271828, 4242, 4243, 4244,
];

thread_local! {
    /// The seed the held-out fixture builder reads. `run_g2_harness` calls
    /// `fixture_of` twice (main arm + InterLoopNorm control) on the SAME
    /// thread, so a thread-local cell is sufficient and keeps the builder a
    /// non-capturing `fn` item (the harness takes a fn pointer by design).
    static T6_SEED: std::cell::Cell<u64> = const { std::cell::Cell::new(V4_SEED) };
}

/// The held-out fixture builder: the v4 family body at the seed currently in
/// `T6_SEED`, α unchanged.
fn scaled_fixture_holdout_of(
    stability: LoopStabilityMode,
) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    build_alpha_scaled(T6_SEED.with(|c| c.get()), V4_ALPHA, stability)
}

/// The disjointness assert behind the "held-out" claim (see the T6 doc).
#[test]
fn t6_holdout_seeds_are_disjoint_from_the_recorded_scan_seeds() {
    for s in T6_HOLDOUT_SEEDS {
        assert!(
            !T6_RECORDED_SCAN_SEEDS.contains(&s),
            "held-out seed {s} is named in the v4 scan record — the T6 replication would be in-sample"
        );
    }
    assert!(
        !T6_HOLDOUT_SEEDS.contains(&SEED) && !SEQ_SEEDS.iter().any(|s| T6_HOLDOUT_SEEDS.contains(s)),
        "held-out set overlaps this file's own fixture seeds"
    );
}

/// T6 — the held-out replication (predictions P1-P3 pre-registered above).
#[test]
fn bench_731_t6_holdout_replication_of_the_v4_mechanism() {
    let mut reports: Vec<(u64, G2Report)> = Vec::with_capacity(T6_HOLDOUT_SEEDS.len());
    for seed in T6_HOLDOUT_SEEDS {
        T6_SEED.with(|c| c.set(seed));
        let corpus: Vec<CorpusInput> = (0..27usize)
            .map(|t| CorpusInput {
                label: format!("S{t}"),
                seq: None,
                pos: 0,
                token: t,
            })
            .collect();
        let report = run_g2_harness(
            &format!("T6 held-out — loop-weight scale (seed {seed}, α {V4_ALPHA})"),
            corpus,
            scaled_fixture_holdout_of,
            false,
        );
        reports.push((seed, report));
    }
    T6_SEED.with(|c| c.set(V4_SEED));

    // ── The table (one row per held-out fixture) ─────────────────────────
    println!("\n═══ [T6] held-out replication table (α {V4_ALPHA}, τ {PROBE_TAU}, d_min {PROBE_D_MIN} — none re-tuned) ═══");
    println!("  seed | status   | knee_max | undef | K* | med_all | margin | cut   | mean_dist | max_dist | fired | E3 control");
    for (seed, r) in &reports {
        let status = match (r.diverged, r.margin_gate, r.g2) {
            (true, _, _) => "DIVERGED",
            (_, false, _) => "REJECTED",
            (_, true, true) => "G2 PASS ",
            (_, true, false) => "G2 FAIL ",
        };
        println!(
            "  {:>4} | {} | {:>8} | {:>2}/{:<2} | {:>2} | {:>7} | {:>5.2}× | {:>4.2}× | {:>9.6} | {:>8.6} | {:>2}/{} | {}",
            seed, status, r.knee_max, r.undefined, r.n, r.k_star, r.median_all, r.margin, r.cut,
            r.mean_exit_dist, r.max_exit_dist, r.fired, r.n,
            if r.control_fires.is_empty() { "clean".to_string() } else { format!("{} FIRE(S)", r.control_fires.len()) },
        );
    }

    // ── P1 — the ceiling claim ───────────────────────────────────────────
    let live: Vec<&G2Report> = reports.iter().map(|(_, r)| r).filter(|r| !r.diverged).collect();
    let diverged = reports.len() - live.len();
    let over_ceiling: Vec<(u64, f32)> = reports
        .iter()
        .filter(|(_, r)| !r.diverged && r.margin > 2.0 && r.mean_exit_dist <= KNEE_BOUND)
        .map(|(s, r)| (*s, r.margin))
        .collect();
    let max_margin = live.iter().map(|r| r.margin).fold(0.0f32, f32::max);
    println!(
        "\n[T6][P1] ceiling claim (no held-out fixture exceeds margin 2.0× at mean dist ≤ {KNEE_BOUND}): {} — max held-out margin {max_margin:.2}× over {} live fixture(s) ({diverged} diverged, excluded){}",
        if over_ceiling.is_empty() { "CORROBORATED" } else { "REFUTED" },
        live.len(),
        if over_ceiling.is_empty() { String::new() } else { format!("; over-ceiling: {over_ceiling:?}") }
    );

    // ── P2 — the rarity claim ────────────────────────────────────────────
    let passes: Vec<u64> = reports
        .iter()
        .filter(|(_, r)| r.g2 && r.margin >= 2.0 && r.cut >= 2.0 && r.mean_exit_dist <= KNEE_BOUND)
        .map(|(s, _)| *s)
        .collect();
    let n_pass = passes.len();
    let reading = match n_pass {
        0 => "v4 is a LOTTERY DRAW — the synthetic axis is CLOSED and T4's real-workload defer is CORROBORATED by measurement, not merely asserted",
        1 | 2 => "consistent with the scan's own base rate — the existence-proof grade STANDS, unchanged",
        _ => "the one-in-144 caveat is OVERSTATED — the mechanism generalizes across the seed lottery at fixed α; G2's grade upgrades to REPLICATED OUT-OF-SAMPLE on the synthetic axis",
    };
    println!(
        "[T6][P2] rarity claim (≤ 2 of {} G2-PASS): {n_pass}/{} pass{} — pre-declared reading: {reading}",
        T6_HOLDOUT_SEEDS.len(),
        T6_HOLDOUT_SEEDS.len(),
        if passes.is_empty() { String::new() } else { format!(" ({passes:?})") }
    );

    // ── P3 — the invariants (asserted inside the harness on every live
    // fixture; this line records the population they covered) ────────────
    let live_inputs: usize = live.iter().map(|r| r.n).sum();
    let violators: Vec<(u64, usize, (f32, String, usize))> = reports
        .iter()
        .filter(|(_, r)| !r.control_fires.is_empty())
        .map(|(s, r)| (*s, r.control_fires.len(), r.control_fires[0].clone()))
        .collect();
    println!(
        "[T6][P3] G1 ≡ None + exit ≡ elastic bit-identity: HELD on {} live held-out fixture(s) / {live_inputs} inputs (hard-asserted in run_g2_harness — reaching this line IS their pass).",
        live.len()
    );
    println!(
        "[T6][P3] InterLoopNorm control (0 fires at all {} τ ≤ 3): {} — {} of {} live fixture(s) violated{}",
        CONTROL_TAUS.len(),
        if violators.is_empty() { "HELD" } else { "VIOLATED" },
        violators.len(),
        live.len(),
        if violators.is_empty() { String::new() } else { format!("; (seed, fires, first) = {violators:?}") }
    );
    assert!(
        !live.is_empty(),
        "every held-out fixture diverged at α {V4_ALPHA} — P1/P2 are unevaluable and P3 covered nothing; the replication is INCONCLUSIVE, not a pass"
    );
    assert!(
        violators.is_empty(),
        "P3 VIOLATED on {}/{} live held-out fixture(s): the InterLoopNorm negative control fired, i.e. the Research-440 trap read a churning loop as converged. This is a probe finding, not a corpus one — see the T6 record. {violators:?}",
        violators.len(),
        live.len()
    );
}

/// The T6 P3 finding, pinned end-to-end on the fixture that produced it
/// (held-out seed 1003, `LoopStabilityMode::InterLoopNorm` — the Research-440
/// step-plateau regime). Both directions, so the mirror defect cannot pass:
/// the pre-T6 single-window shape arm FIRES on this churning loop (that is
/// the defect), and the shipped persistence default refuses it across all 27
/// inputs and all 8 control τ.
///
/// Mechanism, measured (`decay_ratio_max` sweep below): the fire is the shape
/// arm's rule-3 decay fall-through, not rule 1 — it is identical from
/// `settle_floor` 0.5 down to 1e-5 and disappears only at
/// `decay_ratio_max <= 0.3`, i.e. the newer/older half-window ratio at the
/// firing window sits in [0.3, 0.5).
#[test]
fn t6_seed_1003_control_violation_is_pinned_both_directions() {
    use katgpt_core::convergence_cadence::CadenceConfig;
    T6_SEED.with(|c| c.set(1003));
    let config = make_config(LoopStabilityMode::InterLoopNorm);
    let (w, rg, sg) = scaled_fixture_holdout_of(LoopStabilityMode::InterLoopNorm);

    // ── Direction 1: the DEFECT reproduces at persistence 1 ─────────────
    // tau = 0.0 disables the magnitude arm (`mean < 0` is false for any
    // non-negative norm), so a fire can only be the SHAPE arm.
    let mut p = LoopResidualExit::new(0.0, PROBE_D_MIN).with_shape_persistence(1);
    run_on_prefix(&config, &w, &rg, &sg, None, 0, 2, None, Some(&mut p));
    assert_eq!(
        p.fired_at_iteration(),
        Some(14),
        "the pre-T6 shape arm must still false-positive here — if this stops reproducing, the fixture moved and the T6 record no longer describes it"
    );

    // The fire is floor-INVARIANT: rule 3 carries no absolute threshold.
    for sf in [0.5f32, 0.05, 1e-3, 1e-5] {
        let cfg = CadenceConfig { plateau_floor: sf * 2.0, settle_floor: sf, decay_ratio_max: 0.5 };
        let mut p = LoopResidualExit::with_cadence_config(0.0, PROBE_D_MIN, cfg)
            .with_shape_persistence(1);
        run_on_prefix(&config, &w, &rg, &sg, None, 0, 2, None, Some(&mut p));
        assert_eq!(
            p.fired_at_iteration(),
            Some(14),
            "settle_floor {sf}: no absolute calibration can gate a ratio (the T5 seam alone is insufficient)"
        );
    }
    // ...and it IS the decay ratio: decay_ratio_max <= 0.3 suppresses it.
    // (0.0 is deliberately NOT swept: it is an ILLEGAL config — the Issue-720
    // constructor debug_asserts decay_ratio_max > 0.0 — and would panic this
    // test under debug_assertions. A legal drm below the measured [0.3, 0.5)
    // band pins the same mechanism: rule 3 fully suppressed.)
    for (drm, want_fire) in [(0.9f32, true), (0.5, true), (0.3, false), (0.1, false)] {
        let cfg = CadenceConfig { plateau_floor: 1e-9, settle_floor: 1e-12, decay_ratio_max: drm };
        let mut p = LoopResidualExit::with_cadence_config(0.0, PROBE_D_MIN, cfg)
            .with_shape_persistence(1);
        run_on_prefix(&config, &w, &rg, &sg, None, 0, 2, None, Some(&mut p));
        assert_eq!(
            p.fired_at_iteration().is_some(),
            want_fire,
            "decay_ratio_max {drm}: the firing window's newer/older ratio is measured to sit in [0.3, 0.5)"
        );
    }

    // ── Direction 2: the SHIPPED default refuses the whole control ──────
    for &tau in &CONTROL_TAUS {
        for t in 0..27usize {
            let mut p = LoopResidualExit::new(tau, PROBE_D_MIN);
            run_on_prefix(&config, &w, &rg, &sg, None, 0, t, None, Some(&mut p));
            assert_eq!(
                p.fired_at_iteration(),
                None,
                "shipped persistence must refuse the churning loop: tau {tau} fired on S{t}"
            );
        }
    }
    T6_SEED.with(|c| c.set(V4_SEED));
}
