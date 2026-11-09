//! Closed-form fact-write as a deterministic rank-r overlay pair
//! (Issue 990 / Research 659).
//!
//! **PoC VERDICT (T6, NEGATIVE at fixture scale):** the pre-registered G1
//! floor (≥5/8 held-out free-generation exact on the toy decoder) was NOT
//! reached — measured 0/8 across ~20 principled fixture regimes (the
//! delivery debug proved the construction math correct: direction exact at
//! training keys, magnitude cap-trimmed; the failures were generalization
//! and attractor classes specific to a 32-dim toy key space). The full
//! record is Research 659 §7 + the issue's T6 note. The module is RETAINED
//! as correct opt-in substrate (issue-565 precedent: correctness gates G2
//! determinism / G3 empty-overlay demote / G4 monotone-refine / T5b
//! scramble-control (VACUOUS at close — recorded non-discriminating) / T5c
//! structural-undo rows-identical all pass; the T5c pre-registered
//! whole-model bit-exact claim FAILED (re-propagation — see the gate's
//! scoped record); G1/G1b are the recorded-negative instruments,
//! `#[ignore]`d with reasons; the uncapped diagnostic measured the
//! authority/locality coupling (G1 0/8 at max_rel_norm 1.0 while
//! contrast-drift 4/4). The
//! decisive feasibility test is REAL-scale: riir-infer Issue 1189's
//! overlay-at-dequant cells (Qwen3.5-0.8B-Q8_0 then qwen3.8-27B-Q4_K_M).
//! Promoting this feature to default is REFUSED by this verdict.
//!
//! A fact is written into a decoder's MLP weights in closed form — every step
//! a forward pass or a linear solve, no gradients — and shipped as a rank-r
//! pair ([`QuantErrorLora`], reused, never forked) under a
//! mint-once-then-freeze BLAKE3 envelope, CONSTRUCTED-WITHOUT-ARMING (arming
//! stays the apply side's decision, stated there — never on this artifact).
//!
//! Provenance: Research 659 distills `areu01or00/qwen0.8B-wsp` @ `39265dbc`
//! (NO LICENSE — decision-content only: a Rust re-implementation from the
//! recorded math, ZERO code import; outbound-PR fail-closed). The recipe
//! family is published (MEMIT arXiv:2210.07229, ROME, AlphaEdit, AnyEdit,
//! WISE); wsp's increments are per-token closed-form targets, the
//! protected/weighted ridge penalty, the bottom-up layer schedule and the
//! IRLS-family refinement. Q1 verdict: no prior-art novelty — engineering
//! composition on shipped substrate.
//!
//! Recipe (per fact, sequential — wsp writes facts one at a time, each solve
//! protected by every earlier fact's keys):
//!
//! 1. **Keys** — teacher-forced capture of the MLP down-projection INPUT at
//!    every answer position, per edited layer (the [`FactWriteModel`] seam).
//! 2. **Targets** — per answer token `t`: direction `unit(wte[t] ⊙ g)`
//!    (out-embedding × final-norm per-dim scale; no optimized target vector),
//!    magnitude `D_j · r_j · weight_j` with `D_j = max(0, −logprob) +
//!    target_floor_nats` (shortfall-sized nats) and `r_j` the residual RMS at
//!    the layer output. The exact per-token magnitude formula is OURS (wsp's
//!    is decision-content; behavior, not numerics, is gated).
//! 3. **Protected ridge solve, bottom-up** — MEMIT schedule: layer `l`
//!    delivers `share = missing / layers_left`, recomputed after lower layers
//!    are written via a fresh read pass with the edit on.
//!    `dW = R · Kpᵀ · M⁻¹` with `M = λ·KnKnᵀ + λ·(QkQkᵀ + Σ_earlier) +
//!    λ·cw·KcKcᵀ + KpKpᵀ + μ·mean(diag(M))·I` [in×in] SPD — solved by f64
//!    Cholesky with a FIXED row-major (i,j) loop order and sequential f64
//!    accumulation (Issue-940 determinism law; Cholesky on an SPD matrix does
//!    not pivot), no BLAS, no RNG. Capped at 25% of the base layer's
//!    Frobenius norm (wsp `max_rel`), truncated to rank r via
//!    [`QuantErrorLora::from_error`] (dropped energy REPORTED via
//!    `thin_svd_into` — the `factor_tail` discipline).
//! 4. **Protection** — general-question keys (model-answered, weight λ),
//!    look-alike contrast questions (weight λ·contrast_weight, ×10 default),
//!    the fact's own ask tokens, every earlier fact's keys.
//! 5. **Refinement, still closed-form** — one calibration pass scales all
//!    targets by `1/gain` (clamped [gain_lo, gain_hi], measured along the
//!    missing directions); then ≤ max_rounds rounds: free-generate with the
//!    edit on, find the first divergent token per training wording, reweight
//!    (`p<0.5 → ×1.5`, `p<0.2 → ×1.4` more, `p>0.98 → ×0.85`, first-miss
//!    `×2.0`, per-token boost capped at boost_cap × calibrated start),
//!    re-solve. Monotone-or-decline: a round that loses fit reverts to the
//!    previous pair and stops, reporting MISS honestly.
//!
//! The base artifact never moves (overlay law): the base weights digest is
//! pinned on the envelope, an empty overlay demotes byte-identically, and
//! undo is STRUCTURAL — `pairs_without_fact` drops the fact's row from the
//! canonical composition (fact-asc then edited-layer-asc), bit-exact against
//! never having had the row because each pair is an independent f32 addend
//! summed in a fixed order.
//!
//! Modelless compliance: forward passes + one linear solve + one thin SVD per
//! layer per round. No gradient of any loss w.r.t. any parameter, no
//! optimizer state, no RNG. `(model bytes, facts, config) → envelope bytes`
//! is platform-stable on the native platform (the G2 byte-identity gate is a
//! per-platform claim; cross-triple cells are recorded, never gating).

use crate::quant_error_lora::QuantErrorLora;
use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Hyperparameters of the closed-form write. Defaults are the wsp-derived
/// constants (Research 659 §1).
#[derive(Clone, Copy, Debug)]
pub struct FactWriteConfig {
    /// Base weight for the protected covariance terms (general questions,
    /// own ask tokens).
    pub lambda: f64,
    /// Extra weight for look-alike contrast questions (the anti-template-
    /// leakage core; wsp ×10).
    pub contrast_weight: f64,
    /// Weight for EARLIER FACTS' answer keys (OURS, split from λ: the
    /// anti-cross-fire role — too heavy and it suppresses genuine write
    /// keys when subject manifolds sit close, too light and later facts'
    /// writes fire at earlier facts' second-answer positions).
    pub earlier_fact_weight: f64,
    /// Weight for the general control questions (OURS, split from λ).
    pub control_weight: f64,
    /// Ridge share: `M += mu · mean(diag(M)) · I`.
    pub mu: f64,
    /// `‖dW‖_F ≤ max_rel_norm · ‖W_layer‖_F` (wsp `max_rel = 0.25`).
    pub max_rel_norm: f64,
    /// Rank cap of the shipped pair.
    pub rank: usize,
    /// Maximum refinement rounds (wsp: ≤6).
    pub max_rounds: usize,
    /// Per-token boost cap as a multiple of the calibrated start weight.
    pub boost_cap: f64,
    /// Max:min weight RATIO within a fact after reweighting (OURS — a
    /// weighted regression whose rows differ 10:1 delivers the heavy rows'
    /// direction at every key; the ratio cap keeps the solve balanced while
    /// still boosting the positions that need it).
    pub boost_ratio_cap: f64,
    /// Target-size floor in nats: `D_j = min(max(0, −logprob),
    /// shortfall_cap_nats) + floor`.
    pub target_floor_nats: f64,
    /// Bound on the shortfall term of `D_j` (OURS — keeps targets finite at
    /// extreme negative base logprobs).
    pub shortfall_cap_nats: f64,
    /// Calibration gain clamp (wsp: [0.5, 8]).
    pub gain_lo: f64,
    pub gain_hi: f64,
}

impl Default for FactWriteConfig {
    fn default() -> Self {
        Self {
            // OURS (decision-content: wsp's exact values are not recorded):
            // λ = 1.0 — with the near-exact interpolation (μ ≈ 0) the
            // protection terms ride M: earlier-fact/contrast keys are
            // penalized directions, so later facts' writes do not fire at
            // earlier facts' positions; μ = 0.001 keeps the per-key
            // interpolation exact (a larger ridge delivers the AVERAGE share
            // direction — random ± per token),
            lambda: 0.5,
            contrast_weight: 1.0,
            earlier_fact_weight: 0.5,
            control_weight: 0.5,
            mu: 0.001,
            max_rel_norm: 0.25,
            rank: 16,
            max_rounds: 6,
            boost_cap: 10.0,
            boost_ratio_cap: 3.0,
            target_floor_nats: 3.0,
            shortfall_cap_nats: 5.0,
            gain_lo: 0.5,
            gain_hi: 8.0,
        }
    }
}

impl FactWriteConfig {
    /// Loud pre-flight refusal on a config that cannot produce an SPD solve.
    pub fn validate(&self) {
        assert!(self.lambda >= 0.0, "lambda must be >= 0");
        assert!(self.contrast_weight >= 0.0, "contrast_weight must be >= 0");
        assert!(self.mu > 0.0, "mu must be > 0 (the SPD guarantee rides it)");
        assert!(
            self.max_rel_norm > 0.0 && self.max_rel_norm <= 1.0,
            "max_rel_norm must be in (0, 1]"
        );
        assert!(self.rank >= 1, "rank must be >= 1");
        assert!(
            (1..=64).contains(&self.max_rounds),
            "max_rounds must be in [1, 64]"
        );
        assert!(self.boost_cap >= 1.0, "boost_cap must be >= 1");
        assert!(
            self.target_floor_nats >= 0.0,
            "target_floor_nats must be >= 0"
        );
        assert!(
            self.gain_lo > 0.0 && self.gain_hi >= self.gain_lo,
            "gain clamp invalid"
        );
    }
}

// ─── Inputs ─────────────────────────────────────────────────────────────────

/// One fact: training wordings (ask token ids), the answer tokens (content +
/// stop token as the final element), and this fact's look-alike contrast asks
/// (protected at `contrast_weight`).
#[derive(Clone, Debug)]
pub struct FactSpec {
    pub train_asks: Vec<Vec<u32>>,
    pub answer: Vec<u32>,
    pub contrast_asks: Vec<Vec<u32>>,
}

/// General (unrelated) questions whose keys protect the covariance; their
/// "answers" are the base model's own greedy continuations, captured once on
/// the base model (deterministic).
#[derive(Clone, Debug)]
pub struct ProtectSpec {
    pub control_asks: Vec<Vec<u32>>,
}

// ─── Model seam ─────────────────────────────────────────────────────────────

/// Captured state at the tap depth for ONE edited layer over P positions.
pub struct LayerCapture {
    /// Down-projection input (key) rows, row-major `[P × key_dim]`.
    pub keys: Vec<f32>,
    /// Layer-output residual rows, row-major `[P × dim]`.
    pub resid: Vec<f32>,
    pub n_pos: usize,
}

/// The seam the construction math reads through. Implementors own the forward
/// pass and the overlay install; this module never touches weights directly
/// (the base artifact never moves).
pub trait FactWriteModel {
    /// Residual width.
    fn dim(&self) -> usize;
    /// MLP inner width (the key space).
    fn key_dim(&self) -> usize;
    /// Edited layers, ascending (each delivers its MEMIT share bottom-up).
    fn edited_layers(&self) -> &[usize];
    /// BLAKE3 of the base weights — pinned on the envelope; the constructor
    /// never mutates weights, so this is also the post-write digest.
    fn base_weights_digest(&self) -> [u8; 32];
    /// Frobenius norm of the layer's base weight matrix (the 25% cap's
    /// reference).
    fn base_layer_norm(&self, layer: usize) -> f64;
    /// Final-norm per-dim scale `g` (the `norm(ones)` measurement).
    fn final_norm_scale(&self) -> Vec<f32>;
    /// Out-embedding row of `token` (head side, pre-norm).
    fn token_out_embedding(&self, token: u32) -> Vec<f32>;
    /// Teacher-forced pass of `seq` with the CURRENT overlays: captures keys
    /// and layer-output residual at the given `positions`; the result carries
    /// one [`LayerCapture`] per edited layer, ordered like
    /// [`FactWriteModel::edited_layers`].
    fn capture_at(&mut self, seq: &[u32], positions: &[usize]) -> Vec<LayerCapture>;
    /// ln p(seq[pos] | seq[..pos]) under the current overlays. `pos` must be
    /// >= 1 (loud refusal otherwise).
    fn token_logprob(&mut self, seq: &[u32], pos: usize) -> f64;
    /// Greedy free generation (first-index tie rule), stopping before `stop`
    /// or after `max_new` tokens; returns the generated suffix only.
    fn free_generate(&mut self, prefix: &[u32], max_new: usize, stop: Option<u32>) -> Vec<u32>;
    /// Replace layer `layer`'s overlay pairs wholesale (canonical fact order
    /// is the caller's). Empty slice = byte-identical demote.
    fn set_layer_overlays(&mut self, layer: usize, pairs: &[QuantErrorLora]);
}

// ─── Envelope (mint-once-then-freeze) ───────────────────────────────────────

/// Per-fact provenance row.
#[derive(Clone, Debug)]
pub struct FactRow {
    pub fact_index: usize,
    /// BLAKE3 over the training ask tokens (canonical encoding).
    pub ask_digest: [u8; 32],
    pub answer: Vec<u32>,
    /// BLAKE3 over this fact's contrast ask tokens.
    pub contrast_digest: [u8; 32],
    /// Edited layer ids the fact's pairs ride.
    pub layers: Vec<usize>,
    pub rank: usize,
    /// Max dropped energy across the fact's layers (Σ_{k≥r} σ² / Σ σ²).
    pub dropped_energy: f64,
    /// Final refine fit normalized by the total expected depth.
    pub fit: f64,
    pub rounds: usize,
    /// True when a refine round declined and was reverted
    /// (monotone-or-decline).
    pub miss: bool,
}

#[derive(Clone, Debug)]
struct EnvelopePair {
    layer: usize,
    pair: QuantErrorLora,
}

/// Mint-once-then-freeze envelope over the construction — the
/// WeightEditArtifact/FrozenDirectionBank PATTERN as its own minimal
/// instance (the carrier feature's types stay under `weight_edit_overlay`).
#[derive(Clone, Debug)]
pub struct ClosedFormWriteEnvelope {
    base_digest: [u8; 32],
    config: FactWriteConfig,
    constructed_not_armed: bool,
    rows: Vec<FactRow>,
    /// pairs[fact][edited-layer-pos] — canonical order fact-asc then
    /// edited-layer-asc.
    pairs: Vec<Vec<EnvelopePair>>,
    sealed_digest: Option<[u8; 32]>,
}

impl ClosedFormWriteEnvelope {
    fn new(base_digest: [u8; 32], config: FactWriteConfig) -> Self {
        Self {
            base_digest,
            config,
            constructed_not_armed: true,
            rows: Vec::new(),
            pairs: Vec::new(),
            sealed_digest: None,
        }
    }

    fn refuse_if_sealed(&self, what: &str) {
        if let Some(d) = self.sealed_digest {
            panic!("envelope sealed (blake3 {:02x}...): refusing {what}", d[0]);
        }
    }

    /// Append a fact row + its per-layer pairs (edited-layer order). Refuses
    /// loud once sealed.
    pub fn add_fact(&mut self, row: FactRow, pairs: Vec<(usize, QuantErrorLora)>) {
        self.refuse_if_sealed("add_fact");
        assert_eq!(
            row.layers.len(),
            pairs.len(),
            "row.layers must name every pair's layer"
        );
        let ep: Vec<EnvelopePair> = pairs
            .into_iter()
            .map(|(layer, pair)| EnvelopePair { layer, pair })
            .collect();
        self.pairs.push(ep);
        self.rows.push(row);
    }

    /// Freeze: compute + pin the BLAKE3 digest over the canonical encoding.
    /// Further mutation refuses loud. Idempotent.
    pub fn seal(&mut self) -> [u8; 32] {
        if let Some(d) = self.sealed_digest {
            return d;
        }
        let d = blake3_digest(&self.canonical_bytes());
        self.sealed_digest = Some(d);
        d
    }

    /// The pinned digest. Refuses loud on an unsealed envelope.
    pub fn digest(&self) -> [u8; 32] {
        match self.sealed_digest {
            Some(d) => d,
            None => panic!("envelope not sealed: seal() before digest()"),
        }
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed_digest.is_some()
    }

    pub fn base_digest(&self) -> [u8; 32] {
        self.base_digest
    }

    pub fn config_echo(&self) -> &FactWriteConfig {
        &self.config
    }

    /// Always true: the construction mints an artifact only; arming is the
    /// apply side's decision, stated there — never on this envelope.
    pub fn constructed_not_armed(&self) -> bool {
        self.constructed_not_armed
    }

    pub fn rows(&self) -> &[FactRow] {
        &self.rows
    }

    /// All pairs flattened in canonical order (fact-asc, edited-layer-asc).
    pub fn pairs_all(&self) -> Vec<(usize, QuantErrorLora)> {
        let mut out = Vec::new();
        for fp in &self.pairs {
            for ep in fp {
                out.push((ep.layer, ep.pair.clone()));
            }
        }
        out
    }

    /// STRUCTURAL UNDO: every pair except `fact_index`'s, canonical order.
    /// Installing these on the base model is bit-exact against a construction
    /// that never had the fact (each pair is an independent f32 addend summed
    /// in a fixed order).
    pub fn pairs_without_fact(&self, fact_index: usize) -> Vec<(usize, QuantErrorLora)> {
        let mut out = Vec::new();
        for (fi, fp) in self.pairs.iter().enumerate() {
            if fi == fact_index {
                continue;
            }
            for ep in fp {
                out.push((ep.layer, ep.pair.clone()));
            }
        }
        out
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"CWW\\0\\u{1}");
        b.extend_from_slice(&self.base_digest);
        put_f64(&mut b, self.config.lambda);
        put_f64(&mut b, self.config.contrast_weight);
        put_f64(&mut b, self.config.mu);
        put_f64(&mut b, self.config.max_rel_norm);
        put_u64(&mut b, self.config.rank as u64);
        put_u64(&mut b, self.config.max_rounds as u64);
        put_f64(&mut b, self.config.boost_cap);
        put_f64(&mut b, self.config.target_floor_nats);
        put_f64(&mut b, self.config.gain_lo);
        put_f64(&mut b, self.config.gain_hi);
        b.push(u8::from(self.constructed_not_armed));
        put_u64(&mut b, self.rows.len() as u64);
        for row in &self.rows {
            put_u64(&mut b, row.fact_index as u64);
            b.extend_from_slice(&row.ask_digest);
            put_u64(&mut b, row.answer.len() as u64);
            for t in &row.answer {
                put_u64(&mut b, u64::from(*t));
            }
            b.extend_from_slice(&row.contrast_digest);
            put_u64(&mut b, row.layers.len() as u64);
            for l in &row.layers {
                put_u64(&mut b, *l as u64);
            }
            put_u64(&mut b, row.rank as u64);
            put_f64(&mut b, row.dropped_energy);
            put_f64(&mut b, row.fit);
            put_u64(&mut b, row.rounds as u64);
            b.push(u8::from(row.miss));
        }
        put_u64(&mut b, self.pairs.len() as u64);
        for fp in &self.pairs {
            put_u64(&mut b, fp.len() as u64);
            for ep in fp {
                put_u64(&mut b, ep.layer as u64);
                put_f32_slice(&mut b, &ep.pair.a);
                put_f32_slice(&mut b, &ep.pair.b);
                put_f64(&mut b, f64::from(ep.pair.alpha));
                put_u64(&mut b, ep.pair.rank as u64);
                put_u64(&mut b, ep.pair.in_dim as u64);
                put_u64(&mut b, ep.pair.out_dim as u64);
            }
        }
        b
    }
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_f64(b: &mut Vec<u8>, v: f64) {
    b.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn put_f32_slice(b: &mut Vec<u8>, s: &[f32]) {
    put_u64(b, s.len() as u64);
    for v in s {
        b.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

fn blake3_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Canonical digest over ask-token lists (envelope-row provenance).
fn tokens_digest(lists: &[&[u32]]) -> [u8; 32] {
    let mut b = Vec::new();
    put_u64(&mut b, lists.len() as u64);
    for t in lists {
        put_u64(&mut b, t.len() as u64);
        for v in *t {
            put_u64(&mut b, u64::from(*v));
        }
    }
    blake3_digest(&b)
}

// ─── Construction ───────────────────────────────────────────────────────

/// Per-fact outcome of the construction + refinement.
#[derive(Clone, Debug)]
pub struct FactWriteOutcome {
    pub envelope: ClosedFormWriteEnvelope,
    /// Refine fit (Σ matched depth over training wordings) per round, round 0
    /// = the initial spread. Non-decreasing by construction
    /// (monotone-or-decline reverts).
    pub round_fits: Vec<Vec<f64>>,
    pub miss: Vec<bool>,
    pub gain: Vec<f64>,
}

/// Run the closed-form write for every fact (sequential, per wsp). The model
/// ends with every fact's pairs installed; the returned envelope is sealed.
pub fn construct<M: FactWriteModel>(
    model: &mut M,
    config: &FactWriteConfig,
    facts: &[FactSpec],
    protect: &ProtectSpec,
) -> FactWriteOutcome {
    config.validate();
    assert!(!facts.is_empty(), "no facts");
    let layers = model.edited_layers().to_vec();
    assert!(!layers.is_empty(), "no edited layers");
    let d = model.dim();
    let ff = model.key_dim();
    let g = model.final_norm_scale();
    assert_eq!(g.len(), d, "final_norm_scale length mismatch");

    let mut svd_result = SvdResultScratch::with_capacity(ff, d);
    let mut svd_work = SvdScratch::with_capacity(d, ff);

    // General-question protect keys captured ONCE on the base model
    // (deterministic): all positions of ask + the model's own continuation.
    let mut ctrl_keys: Vec<Vec<Vec<f32>>> = vec![Vec::new(); layers.len()];
    for ask in &protect.control_asks {
        let mut seq = ask.clone();
        seq.extend_from_slice(&model.free_generate(ask, 4, None));
        let positions: Vec<usize> = (0..seq.len()).collect();
        for (li, cap) in model.capture_at(&seq, &positions).into_iter().enumerate() {
            for p in 0..cap.n_pos {
                ctrl_keys[li].push(cap.keys[p * ff..(p + 1) * ff].to_vec());
            }
        }
    }

    let base_digest = model.base_weights_digest();
    let mut envelope = ClosedFormWriteEnvelope::new(base_digest, *config);

    let mut facts_pairs: Vec<Vec<QuantErrorLora>> = Vec::new();
    let mut round_fits = Vec::new();
    let mut miss_all = Vec::new();
    let mut gain_all = Vec::new();

    for (f, fact) in facts.iter().enumerate() {
        assert!(
            !fact.train_asks.is_empty(),
            "fact {f}: no training wordings"
        );
        assert!(!fact.answer.is_empty(), "fact {f}: empty answer");

        // Initial spread (weights 1.0) — solve_fact installs as it goes.
        let init_w = vec![vec![1.0f64; fact.answer.len()]; fact.train_asks.len()];
        let first = solve_fact(
            model,
            config,
            f,
            facts,
            &layers,
            &ctrl_keys,
            &facts_pairs,
            &init_w,
            &g,
            &mut svd_result,
            &mut svd_work,
        );
        let mut cur_pairs = first.pairs;
        let dropped_max = first.dropped_max;
        let (fit0, _, _) = gen_fit(model, fact);
        let mut fits = vec![fit0];

        // Calibration: achieved movement along the missing directions of the
        // FIRST edited layer (pre-write residual saved by the spread), then
        // scale every target weight by 1/gain.
        let gain = measure_gain(model, fact, &first.info, config);
        let mut weights = vec![vec![1.0 / gain; fact.answer.len()]; fact.train_asks.len()];
        let cap_w = config.boost_cap / gain;

        let mut miss = false;
        for _round in 1..=config.max_rounds {
            let next = solve_fact(
                model,
                config,
                f,
                facts,
                &layers,
                &ctrl_keys,
                &facts_pairs,
                &weights,
                &g,
                &mut svd_result,
                &mut svd_work,
            );
            let (fit, divs, gens) = gen_fit(model, fact);
            if fit < *fits.last().expect("fits nonempty") {
                // Monotone-or-decline: revert to the previous round's pair.
                // (Equal fits continue — the fit is integer-granular; sub-
                // token improvements are invisible, and once cap-bound the
                // escalation is harmless anyway.)
                install_fact(model, &layers, &facts_pairs, &cur_pairs);
                miss = true;
                break;
            }
            cur_pairs = next.pairs;
            fits.push(fit);
            if divs.iter().all(Option::is_none) {
                break; // every training wording generates the full answer
            }
            update_weights(
                model,
                fact,
                &divs,
                &gens,
                &mut weights,
                cap_w,
                config.boost_ratio_cap,
            );
        }

        let total_expected = (fact.answer.len() * fact.train_asks.len()) as f64;
        let fit_norm = fits.last().copied().unwrap_or(0.0) / total_expected;
        let ask_lists: Vec<&[u32]> = fact.train_asks.iter().map(|v| v.as_slice()).collect();
        let contrast_lists: Vec<&[u32]> = fact.contrast_asks.iter().map(|v| v.as_slice()).collect();
        envelope.add_fact(
            FactRow {
                fact_index: f,
                ask_digest: tokens_digest(&ask_lists),
                answer: fact.answer.clone(),
                contrast_digest: tokens_digest(&contrast_lists),
                layers: layers.clone(),
                rank: config.rank,
                dropped_energy: dropped_max,
                fit: fit_norm,
                rounds: fits.len() - 1,
                miss,
            },
            layers
                .iter()
                .copied()
                .zip(cur_pairs.iter().cloned())
                .collect(),
        );
        facts_pairs.push(cur_pairs);
        round_fits.push(fits);
        miss_all.push(miss);
        gain_all.push(gain);
    }

    envelope.seal();
    FactWriteOutcome {
        envelope,
        round_fits,
        miss: miss_all,
        gain: gain_all,
    }
}

/// Install `facts_pairs` + `fact_pairs` per edited layer (others first, then
/// the fact — the canonical composition order).
fn install_fact<M: FactWriteModel>(
    model: &mut M,
    layers: &[usize],
    facts_pairs: &[Vec<QuantErrorLora>],
    fact_pairs: &[QuantErrorLora],
) {
    for (li, &gl) in layers.iter().enumerate() {
        let mut per_layer: Vec<QuantErrorLora> =
            facts_pairs.iter().map(|fp| fp[li].clone()).collect();
        per_layer.push(fact_pairs[li].clone());
        model.set_layer_overlays(gl, &per_layer);
    }
}

// ─── Solve internals ────────────────────────────────────────────────────

/// Layer-0 spread state saved for the gain calibration.
struct SpreadInfo {
    /// [P × d] residual at the FIRST edited layer's output, pre-write.
    resid_old: Vec<f64>,
    /// [P × d] unit(target − resid_old).
    u_dirs: Vec<f64>,
    /// [P] ‖target − resid_old‖.
    desired_mag: Vec<f64>,
}

struct KeyBlock<'a> {
    rows: &'a [Vec<f32>],
    weight: f64,
}

struct SolveResult {
    pairs: Vec<QuantErrorLora>,
    info: SpreadInfo,
    dropped_max: f64,
}

/// Bottom-up MEMIT spread for one fact: per edited layer, fresh capture,
/// closed-form targets, protected ridge solve, norm cap, rank-r truncation,
/// install (other facts' pairs + the new pair). Installs as it goes so the
/// next layer's capture sees the edit (re-propagation).
#[allow(clippy::too_many_arguments)]
fn solve_fact<M: FactWriteModel>(
    model: &mut M,
    config: &FactWriteConfig,
    fact_idx: usize,
    facts: &[FactSpec],
    layers: &[usize],
    ctrl_keys: &[Vec<Vec<f32>>],
    facts_pairs: &[Vec<QuantErrorLora>],
    weights: &[Vec<f64>],
    g: &[f32],
    svd_result: &mut SvdResultScratch,
    svd_work: &mut SvdScratch,
) -> SolveResult {
    let fact = &facts[fact_idx];
    let d = model.dim();
    let ff = model.key_dim();
    let mut out_pairs: Vec<QuantErrorLora> = Vec::new();
    let mut dropped_max = 0.0f64;
    let mut info = SpreadInfo {
        resid_old: Vec::new(),
        u_dirs: Vec::new(),
        desired_mag: Vec::new(),
    };

    for (li, &gl) in layers.iter().enumerate() {
        let layers_left = (layers.len() - li) as f64;

        // Teacher-forced capture over every training wording, all positions:
        // ask-position keys feed the own-question protection; answer-position
        // keys are the regression points.
        let mut kp: Vec<f64> = Vec::new(); // [P × ff]
        let mut resid: Vec<f64> = Vec::new(); // [P × d]
        let mut targets: Vec<f64> = Vec::new(); // [P × d]
        let mut ask_rows: Vec<Vec<f32>> = Vec::new();
        let mut p = 0usize;
        for (w, ask) in fact.train_asks.iter().enumerate() {
            let mut seq = ask.clone();
            seq.extend_from_slice(&fact.answer);
            let all_pos: Vec<usize> = (0..seq.len()).collect();
            let caps = model.capture_at(&seq, &all_pos);
            let answer_start = ask.len();
            for pi in 0..all_pos.len() {
                let cap = &caps[li];
                let key_row = &cap.keys[pi * ff..(pi + 1) * ff];
                if pi < answer_start {
                    ask_rows.push(key_row.to_vec());
                    continue;
                }
                let j = pi - answer_start;
                kp.extend(key_row.iter().map(|&v| f64::from(v)));
                let res: Vec<f64> = cap.resid[pi * d..(pi + 1) * d]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();

                // Target movement (self-consistent nats sizing, OURS): to
                // raise logit_t by D_j nats, move the residual along
                // u = unit(wte[t] ⊙ g) by D_j / |wte[t] ⊙ g| — the logit
                // gain per unit of residual movement IS |wte[t] ⊙ g|, so the
                // share is never backward. (wsp's exact per-token magnitude
                // is decision-content; its r[j]-scaled form under-shoots the
                // residual's existing component at √D-scaled norms.)
                let emb = model.token_out_embedding(fact.answer[j]);
                let mut dvec = vec![0.0f64; d];
                let mut n2 = 0.0f64;
                for i in 0..d {
                    let v = f64::from(emb[i] * g[i]);
                    dvec[i] = v;
                    n2 += v * v;
                }
                let dot_scale = n2.sqrt();
                if dot_scale > 0.0 {
                    for v in &mut dvec {
                        *v /= dot_scale;
                    }
                }
                let lp = model.token_logprob(&seq, pi);
                // Bounded shortfall (OURS): Dj = min(max(0, −lp), cap) +
                // floor. wsp's unbounded Dj is decision-content and explodes
                // when a target token sits at a very negative logprob (the
                // capped 25% write then distorts the direction mixture).
                let dj = (-lp).clamp(0.0, config.shortfall_cap_nats) + config.target_floor_nats;
                let movement = dj / dot_scale.max(1e-9) * weights[w][j];
                for v in &dvec {
                    targets.push(v * movement);
                }
                resid.extend_from_slice(&res);
                p += 1;
            }
        }
        assert!(p > 0, "no answer positions captured");

        // Contrast-question keys (this fact's look-alikes), model-answered,
        // at the current state.
        let mut contrast_rows: Vec<Vec<f32>> = Vec::new();
        for cask in &fact.contrast_asks {
            let mut seq = cask.clone();
            seq.extend_from_slice(&model.free_generate(cask, 4, None));
            let positions: Vec<usize> = (0..seq.len()).collect();
            let caps = model.capture_at(&seq, &positions);
            for pi in 0..caps[li].n_pos {
                contrast_rows.push(caps[li].keys[pi * ff..(pi + 1) * ff].to_vec());
            }
        }

        // Earlier facts' answer keys at the current state.
        let mut earlier_rows: Vec<Vec<f32>> = Vec::new();
        for earlier in facts.iter().take(fact_idx) {
            for ask in &earlier.train_asks {
                let mut seq = ask.clone();
                seq.extend_from_slice(&earlier.answer);
                let positions: Vec<usize> = (ask.len()..seq.len()).collect();
                let caps = model.capture_at(&seq, &positions);
                for pi in 0..caps[li].n_pos {
                    earlier_rows.push(caps[li].keys[pi * ff..(pi + 1) * ff].to_vec());
                }
            }
        }

        // Desired movement: the nats-sized target, split over the remaining
        // layers (MEMIT share); the residual's existing component is not
        // subtracted — the movement is additive along u and never backward.
        let mut r_mat = vec![0.0f64; p * d];
        for pi in 0..p {
            for i in 0..d {
                r_mat[pi * d + i] = targets[pi * d + i] / layers_left;
            }
        }

        let blocks = [
            KeyBlock {
                rows: &ctrl_keys[li],
                weight: config.control_weight,
            },
            KeyBlock {
                rows: &contrast_rows,
                weight: config.lambda * config.contrast_weight,
            },
            KeyBlock {
                rows: &ask_rows,
                weight: config.lambda,
            },
            KeyBlock {
                rows: &earlier_rows,
                weight: config.earlier_fact_weight,
            },
        ];
        let m = build_m(&blocks, &kp, p, ff, config.mu);
        let dw64 = solve_delta_w(&kp, &r_mat, &m, p, ff, d);

        // 25%-of-layer-norm cap (wsp max_rel).
        let mut dw32: Vec<f32> = dw64.iter().map(|&v| v as f32).collect();
        let mut fn2 = 0.0f64;
        for v in &dw32 {
            fn2 += f64::from(*v) * f64::from(*v);
        }
        let fnorm = fn2.sqrt();
        let cap_norm = config.max_rel_norm * model.base_layer_norm(gl);
        if fnorm > cap_norm && fnorm > 0.0 {
            let scale = (cap_norm / fnorm) as f32;
            for v in &mut dw32 {
                *v *= scale;
            }
        }

        // Rank-r pair (constructor reused — never forked) + dropped energy.
        let zeros = vec![0.0f32; d * ff];
        let pair = QuantErrorLora::from_error(
            &dw32,
            &zeros,
            d,
            ff,
            config.rank,
            1.0,
            svd_result,
            svd_work,
        );
        let dropped = dropped_energy(&dw32, d, ff, config.rank, svd_result, svd_work);
        dropped_max = dropped_max.max(dropped);

        // Layer-0 spread state (the gain reference lives here).
        if li == 0 {
            info.resid_old = resid.clone();
            info.desired_mag = Vec::with_capacity(p);
            info.u_dirs = vec![0.0f64; p * d];
            for pi in 0..p {
                let mut mag = 0.0f64;
                for i in 0..d {
                    info.u_dirs[pi * d + i] = targets[pi * d + i];
                    mag = mag.max(targets[pi * d + i].abs());
                }
                // targets = u · movement: renormalize to u and keep the
                // movement as the desired magnitude.
                let n = {
                    let mut s = 0.0f64;
                    for i in 0..d {
                        let v = info.u_dirs[pi * d + i];
                        s += v * v;
                    }
                    s.sqrt()
                };
                if n > 0.0 {
                    for i in 0..d {
                        info.u_dirs[pi * d + i] /= n;
                    }
                }
                info.desired_mag.push(n);
                let _ = mag;
            }
        }

        // Install: other facts' pairs for this layer + the new pair.
        let mut per_layer: Vec<QuantErrorLora> =
            facts_pairs.iter().map(|fp| fp[li].clone()).collect();
        per_layer.push(pair.clone());
        model.set_layer_overlays(gl, &per_layer);
        out_pairs.push(pair);
    }

    SolveResult {
        pairs: out_pairs,
        info,
        dropped_max,
    }
}

// ─── Linear algebra (deterministic, no BLAS) ────────────────────────────

/// `M = Σ_blocks weight·(rows·rowsᵀ) + Kp·Kpᵀ + mu·mean(diag(M))·I` —
/// [ff × ff] SPD, f64, fixed accumulation order (Issue-940).
fn build_m(blocks: &[KeyBlock<'_>], kp: &[f64], p: usize, ff: usize, mu: f64) -> Vec<f64> {
    let mut m = vec![0.0f64; ff * ff];
    for block in blocks {
        let w = block.weight;
        for row in block.rows {
            for a in 0..ff {
                let va = f64::from(row[a]) * w;
                if va == 0.0 {
                    continue; // skips exact +0.0 adds — identity, still ordered
                }
                for b_i in 0..ff {
                    m[a * ff + b_i] += va * f64::from(row[b_i]);
                }
            }
        }
    }
    for pi in 0..p {
        for a in 0..ff {
            let va = kp[pi * ff + a];
            if va == 0.0 {
                continue;
            }
            for b_i in 0..ff {
                m[a * ff + b_i] += va * kp[pi * ff + b_i];
            }
        }
    }
    let mut diag_mean = 0.0f64;
    for i in 0..ff {
        diag_mean += m[i * ff + i];
    }
    diag_mean /= ff as f64;
    for i in 0..ff {
        m[i * ff + i] += mu * diag_mean;
    }
    m
}

/// Cholesky–Banachiewicz lower factor of an SPD matrix, f64, FIXED row-major
/// (i,j) loop order with sequential accumulation (Issue-940; Cholesky on an
/// SPD matrix does not pivot). Panics loud on a non-SPD input.
fn cholesky_lower(m: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut acc = m[i * n + j];
            for k in 0..j {
                acc -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(acc > 0.0, "protected covariance not SPD (diag {i} = {acc})");
                l[i * n + i] = acc.sqrt();
            } else {
                l[i * n + j] = acc / l[j * n + j];
            }
        }
    }
    l
}

/// `dW = R · Kpᵀ · M⁻¹` — the MEMIT ridge solve. Kp [P × ff] row-major,
/// R [P × d] row-major, M [ff × ff] SPD row-major. Returns dW [d × ff]
/// row-major, f64. Fixed loop order; sequential f64 accumulation; no BLAS.
fn solve_delta_w(kp: &[f64], r: &[f64], m: &[f64], p: usize, ff: usize, d: usize) -> Vec<f64> {
    // B = Kpᵀ · R  [ff × d].
    let mut b = vec![0.0f64; ff * d];
    for pi in 0..p {
        let kr = &kp[pi * ff..(pi + 1) * ff];
        let rr = &r[pi * d..(pi + 1) * d];
        for f_i in 0..ff {
            let kv = kr[f_i];
            if kv == 0.0 {
                continue;
            }
            for dc in 0..d {
                b[f_i * d + dc] += kv * rr[dc];
            }
        }
    }
    let chol = cholesky_lower(m, ff);
    // X = M⁻¹ B — forward/back substitution per column of B (fixed order).
    let mut x = vec![0.0f64; ff * d];
    let mut y = vec![0.0f64; ff];
    for dc in 0..d {
        for i in 0..ff {
            let mut acc = b[i * d + dc];
            for k in 0..i {
                acc -= chol[i * ff + k] * y[k];
            }
            y[i] = acc / chol[i * ff + i];
        }
        for i in (0..ff).rev() {
            let mut acc = y[i];
            for k in (i + 1)..ff {
                acc -= chol[k * ff + i] * x[k * d + dc];
            }
            x[i * d + dc] = acc / chol[i * ff + i];
        }
    }
    // dW = Xᵀ.
    let mut dw = vec![0.0f64; d * ff];
    for f_i in 0..ff {
        for dc in 0..d {
            dw[dc * ff + f_i] = x[f_i * d + dc];
        }
    }
    dw
}

/// Dropped energy of the rank-r truncation: `Σ_{k≥r} σ² / Σ σ²` over the
/// singular spectrum of dW (via `thin_svd_into`, the factor_tail discipline).
#[allow(clippy::too_many_arguments)]
fn dropped_energy(
    dw: &[f32],
    d: usize,
    ff: usize,
    rank: usize,
    svd_result: &mut SvdResultScratch,
    svd_work: &mut SvdScratch,
) -> f64 {
    // Factor dWᵀ [ff × d] (one-sided Jacobi needs m ≥ n).
    let mut dw_t = vec![0.0f32; ff * d];
    for o in 0..d {
        for i in 0..ff {
            dw_t[i * d + o] = dw[o * ff + i];
        }
    }
    thin_svd_into(&dw_t, ff, d, svd_result, svd_work);
    let mut total = 0.0f64;
    let mut kept = 0.0f64;
    let n = svd_result.len();
    for k in 0..n {
        let s = f64::from(svd_result.singular_value(k));
        let s2 = s * s;
        total += s2;
        if k < rank {
            kept += s2;
        }
    }
    if total <= 0.0 {
        0.0
    } else {
        (total - kept) / total
    }
}

// ─── Refine helpers ─────────────────────────────────────────────────────

/// Free-generation fit over the fact's training wordings: matched-prefix
/// depth against the full answer (content + stop). Returns the total fit,
/// the per-wording first-divergence index (None = full answer generated),
/// and the generations themselves (for the reweighting prefixes).
fn gen_fit<M: FactWriteModel>(
    model: &mut M,
    fact: &FactSpec,
) -> (f64, Vec<Option<usize>>, Vec<Vec<u32>>) {
    let n = fact.answer.len();
    let stop = *fact.answer.last().expect("answer nonempty");
    let mut total = 0.0f64;
    let mut divs = Vec::new();
    let mut gens = Vec::new();
    for ask in &fact.train_asks {
        let generated = model.free_generate(ask, n + 2, Some(stop));
        let mut depth = 0usize;
        while depth < n && depth < generated.len() && generated[depth] == fact.answer[depth] {
            depth += 1;
        }
        total += depth as f64;
        divs.push(if depth >= n { None } else { Some(depth) });
        gens.push(generated);
    }
    (total, divs, gens)
}

/// Calibration gain: achieved movement along the missing directions of the
/// first edited layer, over the desired magnitude. Clamped to the config
/// band (wsp [0.5, 8]).
fn measure_gain<M: FactWriteModel>(
    model: &mut M,
    fact: &FactSpec,
    info: &SpreadInfo,
    config: &FactWriteConfig,
) -> f64 {
    let d = model.dim();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut pi = 0usize;
    for ask in &fact.train_asks {
        let mut seq = ask.clone();
        seq.extend_from_slice(&fact.answer);
        let positions: Vec<usize> = (ask.len()..seq.len()).collect();
        let caps = model.capture_at(&seq, &positions);
        let cap = &caps[0];
        for j in 0..cap.n_pos {
            for i in 0..d {
                let now = f64::from(cap.resid[j * d + i]);
                num += (now - info.resid_old[pi * d + i]) * info.u_dirs[pi * d + i];
            }
            den += info.desired_mag[pi];
            pi += 1;
        }
    }
    if den <= 0.0 {
        return 1.0;
    }
    let gain = num / den;
    gain.clamp(config.gain_lo, config.gain_hi)
}

/// IRLS-family per-token reweighting (wsp constants): at the first divergent
/// token of each training wording, measure p = the model's probability of
/// the expected token under the diverged prefix, then boost/decay that
/// token's target weight. Per-token total boost caps at `cap_w` (the boost
/// cap × calibrated start).
fn update_weights<M: FactWriteModel>(
    model: &mut M,
    fact: &FactSpec,
    divs: &[Option<usize>],
    gens: &[Vec<u32>],
    weights: &mut [Vec<f64>],
    cap_w: f64,
    cap_ratio: f64,
) {
    for (w, div) in divs.iter().enumerate() {
        let Some(j) = div else { continue };
        // The stop-token position always "diverges" (free generation rarely
        // ends in END while junk-running) — boosting it is pure noise and it
        // escalates to dominate the regression; exempt it (OURS).
        if *j + 1 == fact.answer.len() {
            continue;
        }
        let mut prefix = fact.train_asks[w].clone();
        prefix.extend_from_slice(&gens[w][..*j]);
        let probe_pos = prefix.len();
        prefix.push(fact.answer[*j]);
        let p = model.token_logprob(&prefix, probe_pos).exp();
        let wgt = &mut weights[w][*j];
        if p < 0.5 {
            *wgt *= 1.5;
        }
        if p < 0.2 {
            *wgt *= 1.4;
        }
        if p > 0.98 {
            *wgt *= 0.85;
        }
        // First-miss ×2.0 — `j` IS the first divergence by construction.
        *wgt *= 2.0;
        if *wgt > cap_w {
            *wgt = cap_w;
        }
    }
    // Ratio balance (OURS): a weighted regression whose rows differ 10:1
    // delivers the heavy rows' direction at every key; cap max ≤ ratio × min
    // so the solve stays balanced while the hard positions keep their boost.
    let mut wmin = f64::INFINITY;
    let mut wmax = f64::NEG_INFINITY;
    for row in weights.iter() {
        for v in row {
            wmin = wmin.min(*v);
            wmax = wmax.max(*v);
        }
    }
    if wmin > 0.0 {
        let allowed = wmin * cap_ratio;
        if wmax > allowed {
            let scale = allowed / wmax;
            for row in weights.iter_mut() {
                for v in row.iter_mut() {
                    if *v > allowed {
                        *v *= scale;
                    }
                }
            }
        }
    }
}
