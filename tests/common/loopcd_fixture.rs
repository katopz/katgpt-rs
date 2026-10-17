//! Plan 617 T3.1 — the LoopCD toy looped fixture: a hand-constructed
//! modelless 1-layer weight-shared looped transformer with KNOWN iterative
//! semantics (the closed-form iterative-refinement task; NO training, NO
//! riir-train artifact — the plan's primary path).
//!
//! # The task
//!
//! Prefix vote counting. A case is a prefix of A/B symbols plus a QUERY
//! token; the answer is whichever class has the majority of votes. The
//! looped forward implements the paper's iterative-refinement shape:
//!
//! - **e0** accumulates A-vote mass, **e1** B-vote mass (W_v projects the
//!   normalized class votes; W_q = 0 makes attention uniform over the
//!   prefix, so the mean is exactly the vote tally diluted by prefix
//!   length — longer prefixes vote slower).
//! - **e2** is the decision accumulator: the ReLU MLP computes
//!   `out[e2] = δ·(x̂0 − x̂1)` per loop iteration, so x2 grows LINEARLY in
//!   depth with the majority's sign.
//! - The head reads `z_ansA = g·x2 + c_A`, `z_ansB = −g·x2 + c_B` with the
//!   B-side bias gap `Δ = c_B − c_A > 0`: at shallow depth every decode
//!   reads ansB (the bias side), and the argmax crosses to the true
//!   majority at a depth `τ* ∝ Δ/(g·δ·imbalance)` — the crossing depth
//!   encodes decision closeness.
//!
//! This construction gives the three non-degeneracy properties T3.1
//! demands, each with a DIRECT mechanism:
//!
//! (a) **depth→accuracy slope** — deeper loops cross more of the
//!     A-correct strata (accuracy is monotone by construction: the
//!     decision accumulator keeps its sign and grows in magnitude);
//! (b) **close-decision stratum** — small-imbalance prefixes cross late;
//!     at the fixture's max depth their `|z_ansA − z_ansB|` margin is
//!     small but the stratum is non-empty;
//! (c) **disagreement floor** — every A-correct case reads ansB at depth
//!     `ref_loop = 1` (the bias gap) and ansA at full depth once crossed,
//!     so `argmax(z_k) ≠ argmax(z_R)` has guaranteed mass.
//!
//! The paper-vocabulary note (the plan's own warning): the orthogonal
//! re-ranking component is real here — `z_R − z_k` is NOT parallel to
//! `z_R` (the depth-1 reference sits on the B side), so the contrast has
//! an argmax-moving component by construction, unlike a monotone-
//! contraction fixture whose greedy argmax is unchanged for structural
//! reasons.
//!
//! # Provenance + ordering (T3.1's frozen-fixture law)
//!
//! Hand-set constants; no RNG anywhere; `tests/loopcd_fixture_gate.rs`
//! commits the non-degeneracy assertions BEFORE any guided arm runs, and
//! the GOAT bench record states that ordering. The constants are tuned by
//! MEASUREMENT (the gate prints the strata), never fitted against a
//! guided outcome.

#![allow(dead_code)]

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, SdpaOutputGate};

// ── Vocabulary layout ────────────────────────────────────────────────

/// A-class prefix symbol (votes e0).
pub const SYM_A: usize = 0;
/// B-class prefix symbol (votes e1).
pub const SYM_B: usize = 1;
/// Answer token: "majority is A".
pub const ANS_A: usize = 2;
/// Answer token: "majority is B".
pub const ANS_B: usize = 3;
/// The query token that carries no vote; the decode happens at its position.
pub const QUERY: usize = 4;
/// vocab_size = 5.
pub const VOCAB: usize = 5;

/// Maximum loop depth the fixture's config declares (the natural loop_count;
/// callers sweep depths 1..=MAX_LOOPS via the elastic override).
pub const MAX_LOOPS: usize = 12;

/// Longest prefix (block_size must cover prefix + query).
pub const MAX_PREFIX: usize = 16;

// ── Hand-set constants (tuned by measurement, never against a guided arm) ─

/// MLP decision growth per iteration: `out[e2] = δ·(x̂0 − x̂1)`. DELIBERATELY
/// small: the head's bias axis decays geometrically under the in-layer RMSNorm
/// (x3 ← x3/rms per sublayer, never replenished), so the crossing threshold
/// `Δ·x3(τ)` falls with depth — δ sets the clock that decides which (len,
/// imbalance) strata cross within MAX_LOOPS. Measured: δ=1.0 collapses every
/// crossing into depths 1–6 (close stratum empty at 12); δ=0.2 spreads them.
const DELTA: f32 = 0.2;
/// Head slope on the decision axis: `z_ansA − z_ansB = 2g·x2 − Δ`.
const G_HEAD: f32 = 1.0;
/// A-side head bias (on the e3 axis, which is exactly 1.0 at the query).
const C_A: f32 = 0.5;
/// B-side head bias. The bias gap `Δ = c_B − c_A` sets the crossing depths
/// (the threshold decays with the bias axis; see DELTA).
const C_B: f32 = 3.1;
/// Input-row head suppression: `z_sym = −M·(x0 + x1) ≤ 0 < min(c_A, c_B)`,
/// so the answer tokens always dominate the symbol rows at the query.
const M_SUPPRESS: f32 = 1.0;
/// Query-row suppression — same shape as M_SUPPRESS but 10× steeper: the
/// contrast can push answer logits ~ω·|z−z_k| below their unguided values,
/// and the query row must stay strictly below even those (a measured
/// guidance-armed failure: with a zero query row the decode answered QUERY).
const Q_SUPPRESS: f32 = 10.0;

// ── The fixture ──────────────────────────────────────────────────────

pub struct LoopCdFixture {
    pub config: Config,
    pub weights: TransformerWeights,
}

/// One case: a prefix over {SYM_A, SYM_B} (counts never tie) + the ground
/// truth. The case space is fully enumerated — no RNG, byte-deterministic.
#[derive(Clone, Debug)]
pub struct LoopCdCase {
    pub prefix: Vec<usize>,
    /// Ground-truth answer token (`ANS_A` / `ANS_B`).
    pub answer: usize,
    /// Signed vote imbalance `(nA − nB)` — the closeness driver.
    pub imbalance: i32,
}

impl LoopCdFixture {
    /// Hand-set weights per the module doc. `micro()` supplies every field
    /// this fixture does not override (attention_mode=Causal, use_rope=false,
    /// no softcaps, gated_attn=false, LoopStabilityMode::None).
    pub fn new() -> Self {
        let n = 4usize;
        let kvd = 4usize; // n_kv_head(1) × head_dim(4)
        let mut config = Config::micro();
        config.vocab_size = VOCAB;
        config.block_size = 32;
        config.n_embd = n;
        config.n_head = 1;
        config.head_dim = 4;
        config.n_kv_head = 1;
        config.mlp_hidden = 4; // ≥ n_embd: the residual-gate path reuses ctx.hidden as an n-scratch
        config.n_layer = 1;
        config.loop_mode = LoopMode::WeightShared {
            loop_count: MAX_LOOPS,
        };
        config.hybrid_pattern = HybridPattern::Uniform;
        config.loop_min = 0;
        config.loop_max = 0;
        config.gated_attn = false;

        // wte: A → e0, B → e1; answers/query → e3 (inert identity axis).
        let mut wte = vec![0.0f32; VOCAB * n];
        wte[SYM_A * n] = 1.0;
        wte[SYM_B * n + 1] = 1.0;
        for t in [ANS_A, ANS_B, QUERY] {
            wte[t * n + 3] = 1.0;
        }
        // wpe: zero — the task is permutation-invariant (uniform attention).
        let wpe = vec![0.0f32; config.block_size * n];

        // lm_head rows [vocab, n]:
        //   symbols: −M·(x0 + x1) (never wins at the query);
        //   ansA:    +g·x2 + c_A;
        //   ansB:    −g·x2 + c_B;
        //   query:   −Q·(x0 + x1) — never wins, INCLUDING under guidance:
        //   the contrast can push BOTH answer logits below zero (the
        //   all-zero row would then win the decode — measured), so the
        //   query row must sit strictly below any post-guidance answer.
        let mut lm_head = vec![0.0f32; VOCAB * n];
        lm_head[SYM_A * n] = -M_SUPPRESS;
        lm_head[SYM_A * n + 1] = -M_SUPPRESS;
        lm_head[SYM_B * n] = -M_SUPPRESS;
        lm_head[SYM_B * n + 1] = -M_SUPPRESS;
        lm_head[QUERY * n] = -Q_SUPPRESS;
        lm_head[QUERY * n + 1] = -Q_SUPPRESS;
        lm_head[ANS_A * n + 2] = G_HEAD;
        lm_head[ANS_A * n + 3] = C_A;
        lm_head[ANS_B * n + 2] = -G_HEAD;
        lm_head[ANS_B * n + 3] = C_B;

        // One layer. W_q = W_k = 0 → uniform attention over the prefix.
        // W_v projects onto the vote plane; W_o routes it back unchanged.
        let attn_wq = vec![0.0f32; n * n];
        let attn_wk = vec![0.0f32; kvd * n];
        let mut attn_wv = vec![0.0f32; kvd * n];
        attn_wv[0] = 1.0; // out[e0] ← x0
        attn_wv[n + 1] = 1.0; // out[e1] ← x1 (row 1)
        let mut attn_wo = vec![0.0f32; n * n];
        for i in 0..n {
            attn_wo[i * n + i] = 1.0;
        }
        // ReLU MLP: h = relu([x̂0, x̂1, 0, 0]); out[e2] = δ·(h0 − h1).
        // Neurons 2–3 are inert (zero W1 rows → relu(0) = 0, zero W2 column).
        let mlp_w1 = vec![
            1.0, 0.0, 0.0, 0.0, // neuron 0 reads e0
            0.0, 1.0, 0.0, 0.0, // neuron 1 reads e1
            0.0, 0.0, 0.0, 0.0, // inert
            0.0, 0.0, 0.0, 0.0, // inert
        ];
        let mut mlp_w2 = vec![0.0f32; n * 4];
        mlp_w2[2 * 4] = DELTA;
        mlp_w2[2 * 4 + 1] = -DELTA;

        let weights = TransformerWeights::from_parts(
            wte,
            wpe,
            lm_head,
            vec![katgpt_rs::transformer::LayerWeights::from_parts(
                attn_wq, attn_wk, attn_wv, attn_wo, mlp_w1, mlp_w2,
            )],
            n,
            1,
        );
        Self { config, weights }
    }

    /// Full case run: prefix at depth 1 (each prefix position one loop
    /// iteration via the elastic override), then the QUERY at `depth`.
    /// Fresh context + caches per call — the fixture never reuses KV state.
    pub fn run_tokens(&self, prefix: &[usize], depth: usize) -> Vec<f32> {
        let config = &self.config;
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let mut ahla_cache = MultiLayerAhlaCache::new(config);
        let residual_gate = ResidualGate::new(MAX_LOOPS, config.n_embd);
        let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

        for (p, &tok) in prefix.iter().enumerate() {
            self.forward(
                &mut ctx,
                &mut cache,
                &mut ahla_cache,
                &residual_gate,
                &sdpa_gate,
                tok,
                p,
                Some(1),
            );
        }
        self.forward(
            &mut ctx,
            &mut cache,
            &mut ahla_cache,
            &residual_gate,
            &sdpa_gate,
            QUERY,
            prefix.len(),
            Some(depth),
        )
    }

    /// Case convenience: decode logits at the query for `depth` loops.
    pub fn run_case(&self, case: &LoopCdCase, depth: usize) -> Vec<f32> {
        self.run_tokens(&case.prefix, depth)
    }

    /// Full case run that ALSO returns the final query hidden state
    /// (`ctx.hidden_state`, the post-loop pre-head state) — the early-tap
    /// incumbent arm reads it to build its weak side.
    pub fn run_tokens_with_hidden(&self, prefix: &[usize], depth: usize) -> (Vec<f32>, Vec<f32>) {
        let config = &self.config;
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let mut ahla_cache = MultiLayerAhlaCache::new(config);
        let residual_gate = ResidualGate::new(MAX_LOOPS, config.n_embd);
        let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

        for (p, &tok) in prefix.iter().enumerate() {
            self.forward(
                &mut ctx,
                &mut cache,
                &mut ahla_cache,
                &residual_gate,
                &sdpa_gate,
                tok,
                p,
                Some(1),
            );
        }
        let logits = self.forward(
            &mut ctx,
            &mut cache,
            &mut ahla_cache,
            &residual_gate,
            &sdpa_gate,
            QUERY,
            prefix.len(),
            Some(depth),
        );
        let hidden = ctx.hidden_state[..config.n_embd].to_vec();
        (logits, hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        ctx: &mut ForwardContext,
        cache: &mut MultiLayerKVCache,
        ahla_cache: &mut MultiLayerAhlaCache,
        residual_gate: &ResidualGate,
        sdpa_gate: &SdpaOutputGate,
        token: usize,
        pos: usize,
        elastic: Option<usize>,
    ) -> Vec<f32> {
        self.forward_ext(
            ctx,
            cache,
            ahla_cache,
            residual_gate,
            sdpa_gate,
            token,
            pos,
            elastic,
            #[cfg(feature = "loop_guidance")]
            None,
            #[cfg(feature = "loop_guidance")]
            None,
        )
        .to_vec()
    }

    /// Guided variant (Plan 617 T2.3/Phase 3): optional LoopCD guidance and
    /// settle exit threaded through to `forward_looped`. Only compiled when
    /// the feature is on (the fixture's plain [`LoopCdFixture::run_tokens`]
    /// covers every other posture).
    #[cfg(feature = "loop_guidance")]
    #[allow(clippy::too_many_arguments)]
    pub fn run_tokens_guided(
        &self,
        prefix: &[usize],
        depth: usize,
        guidance: Option<&mut katgpt_rs::transformer::loop_guidance::LoopGuidance>,
        settle: Option<&mut katgpt_rs::transformer::loop_guidance::SettleExit>,
    ) -> Vec<f32> {
        let config = &self.config;
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let mut ahla_cache = MultiLayerAhlaCache::new(config);
        let residual_gate = ResidualGate::new(MAX_LOOPS, config.n_embd);
        let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

        for (p, &tok) in prefix.iter().enumerate() {
            self.forward_ext(
                &mut ctx,
                &mut cache,
                &mut ahla_cache,
                &residual_gate,
                &sdpa_gate,
                tok,
                p,
                Some(1),
                None,
                None,
            );
        }
        self.forward_ext(
            &mut ctx,
            &mut cache,
            &mut ahla_cache,
            &residual_gate,
            &sdpa_gate,
            QUERY,
            prefix.len(),
            Some(depth),
            guidance,
            settle,
        )
        .to_vec()
    }

    /// The one `forward_looped` invocation shape, covering every cfg'd slot
    /// so all fixture consumers compile at any feature combination.
    #[allow(clippy::too_many_arguments)]
    fn forward_ext<'a>(
        &self,
        ctx: &'a mut ForwardContext,
        cache: &'a mut MultiLayerKVCache,
        ahla_cache: &'a mut MultiLayerAhlaCache,
        residual_gate: &'a ResidualGate,
        sdpa_gate: &'a SdpaOutputGate,
        token: usize,
        pos: usize,
        elastic: Option<usize>,
        #[cfg(feature = "loop_guidance")] guidance: Option<
            &mut katgpt_rs::transformer::loop_guidance::LoopGuidance,
        >,
        #[cfg(feature = "loop_guidance")] settle: Option<
            &mut katgpt_rs::transformer::loop_guidance::SettleExit,
        >,
    ) -> &'a mut [f32] {
        let config = &self.config;
        forward_looped(
            ctx,
            &self.weights,
            cache,
            ahla_cache,
            token,
            pos,
            config,
            residual_gate,
            sdpa_gate,
            None, // gdn2_cache (sleep_consolidation)
            None, // sleep_config
            #[cfg(feature = "weight_shared_advantage_gate")]
            None,
            elastic,
            #[cfg(feature = "gain_cost_halt")]
            None,
            None, // deep_run
            #[cfg(feature = "cadence_gate")]
            None,
            #[cfg(feature = "loop_guidance")]
            guidance,
            #[cfg(feature = "loop_guidance")]
            settle,
        )
    }
}

impl Default for LoopCdFixture {
    fn default() -> Self {
        Self::new()
    }
}

/// The full enumerated case space: every prefix length 2..=MAX_PREFIX and
/// every non-tying majority split. Deterministic; no RNG.
pub fn all_cases() -> Vec<LoopCdCase> {
    let mut cases = Vec::new();
    for p in 2..=MAX_PREFIX {
        for na in 0..=p {
            let nb = p - na;
            if na == nb {
                continue; // ties have no ground truth
            }
            let mut prefix = Vec::with_capacity(p);
            prefix.resize(na, SYM_A);
            prefix.resize(p, SYM_B);
            let answer = if na > nb { ANS_A } else { ANS_B };
            cases.push(LoopCdCase {
                prefix,
                answer,
                imbalance: na as i32 - nb as i32,
            });
        }
    }
    cases
}

/// Greedy argmax with the house total order (the same comparator the
/// advantage-gate call site uses; fixture logits are finite, but the order
/// is total so the helper never depends on that).
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i)
}

/// The two-answer margin at the query: `z_ansA − z_ansB`. Positive = A.
pub fn answer_margin(logits: &[f32]) -> f32 {
    logits[ANS_A] - logits[ANS_B]
}
