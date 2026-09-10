//! Quantized Simplex Gossip (QSG) — the minimal statistical mechanics of
//! multi-agent belief interaction, with closed-form collapse scaling laws.
//!
//! > **Source:** Hidenori Tanaka, *When Is Collective Intelligence a Lottery?
//! > Multi-Agent Scaling Laws for Memetic Drift in LLMs*,
//! > [arXiv:2603.24676](https://arxiv.org/abs/2603.24676), Mar 2026.
//! > Distilled in `.research/542`; implemented per `.plans/589`.
//!
//! # The primitive
//!
//! `N` agents each hold a **belief**: a probability distribution `x_i ∈ Δ_{K−1}`
//! over `K` competing options (rumor variants, faction candidates, verdicts).
//! Each interaction picks an ordered speaker–listener pair; the speaker emits a
//! **sampled** message and only the listener adapts:
//!
//! ```text
//! x_L ← (1−α)·x_L + α·y ,      α ∈ (0,1]   (adaptation rate / persona plasticity)
//!
//! Hard   (m=1):  y = e_k*,   k* ~ Cat(x_S)                 (one sampled label)
//! Top-m:         y = (1/m)·Σ_j e_{k_j},  k_j ~iid~ Cat(x_S) (m sampled labels)
//! Soft   (m=∞):  y = x_S                                      (full-belief baseline)
//! ```
//!
//! All three modes share the same conditional mean `E[y | x_S] = x_S`; only the
//! variance differs, scaling as `1/m`. That variance is the whole mechanism:
//! under quantized communication the population injects sampling noise into
//! *itself* (one agent's arbitrary draw is the next agent's evidence —
//! "mutual in-context learning"), and symmetry breaking follows. Under Soft
//! exchange the symmetric state is an exact fixed point and nothing happens.
//!
//! # The laws (each one pinned by a test in this module)
//!
//! With `x̄` the population mean, `U = ‖x̄‖²₂ ∈ [1/K, 1]` (polarization: `1/K`
//! = perfect symmetry, `1` = consensus), `V = Σ_i ‖x_i − x̄‖²` (disagreement
//! energy) and `S = U − V/(N(N−1))` (coordination rate):
//!
//! | Law | Form | Test |
//! |---|---|---|
//! | Variance injection (Thm 1) | `E[ΔU]_hard − E[ΔU]_soft = (α²/N²)·E[1−‖x_S‖²]` | `theorem1_hard_variance_injection_identity` |
//! | Bandwidth law (Thm 2) | extra drift scales `1/m`; `E‖y(m)−x_S‖² = (1/m)(1−‖x_S‖²)` | `theorem2_topm_bandwidth_identity` |
//! | Symmetric corollary | per-step drift at symmetry `= α²/(mN²)·(1−1/K)` | `symmetric_1_over_m_corollary` |
//! | Soft martingale | `E[x̄' | X] = x̄` exactly; Soft never breaks symmetry | `soft_exchange_preserves_the_mean` , `soft_from_symmetry_never_breaks_it` |
//! | Soft contracts V | `E[ΔV|X]_soft = −(2α/(N−1))·(1−α+α/N)·V ≤ 0` | `soft_exchange_contracts_disagreement` |
//! | Voter reduction (α=1) | listeners snap one-hot; winner prob `= x̄_k(0)` | `alpha_one_reduces_to_uniform_winner_voter_model` |
//!
//! Mean-field collapse time: `U(t) = 1 − (1−1/K)·exp(−α²t/(mN²))`, i.e.
//! `t_cons ≈ (mN²/α²)·log((1−1/K)/(1−U⋆))` — validated against the shipped
//! kernel in `tests/bench_703_qsg_scaling_laws.rs` together with the
//! drift-vs-selection crossover `Γh = mNh/α` (fixation logistic in `Γh`,
//! `Nc ~ α/(m|h|)`).
//!
//! # Relationship to `signed_coupling` — sibling dials, do not blur
//!
//! Both are crowd-opinion dynamics; they differ in state, update, and axes:
//!
//! | | [`signed_coupling`] | this module |
//! |---|---|---|
//! | state | binary stance `s_i ∈ {−1,+1}` (or graded scalar) | simplex belief `x_i ∈ Δ_{K−1}`, persistent |
//! | update | heat-bath **resample**: `P(s_i=+1) = σ(h_i)` — memoryless | EMA **blend** toward a sampled message — beliefs have inertia |
//! | dial | social temperature `T` | bandwidth `m` + adaptation `α` |
//! | state space | interaction graph (signed ties) | well-mixed population (pair source is a caller seam) |
//!
//! `K=2` QSG at `α=1` reduces to the classical two-state voter/Moran copying
//! process — the memoryless limit of a signed-coupling crowd. A consumer that
//! wants *who influences whom* wants `signed_coupling`; one that wants *how a
//! population settles on one of K options, and how fast* wants this module.
//! The [`SusceptibilityAccumulator`] (temperature axis) and the `U`/`V`
//! reducers here (bandwidth/adaptation axis) compose on the same crowd.
//!
//! # The `m` axis is an engineering budget you already have
//!
//! In this stack, "how much of the speaker's state fits in one message" ships
//! as a wire budget (`NpcCommsBus` `DensityBudget::k_for` top-K slices). QSG's
//! result is that this budget is also a *dynamical* dial: extra collapse
//! pressure scales `1/m`, so halving the per-message budget doubles the drift
//! that pushes a crowd toward monoculture. The deterministic top-K slice and
//! the random sample differ exactly in whether that drift term exists.
//!
//! # External fields (selection vs drift)
//!
//! The kernel is neutral by construction. A weak systematic bias `h` (an
//! externally tilted speaker channel, `p̃_k ∝ p_k·e^h`) turns the same
//! dynamics into a selection process: fixation becomes logistic in
//! `Γh = mNh/α` with crossover population `Nc ~ α/(m|h|)` — tiny populations
//! pick winners by lottery ("memetic drift"), large ones amplify arbitrarily
//! small biases deterministically. Compose it from the exported pieces: tilt a
//! copy of the speaker row, [`qsg_draw_categorical`] from the tilted copy,
//! [`qsg_blend_listener_into`] — see the scaling-law test for the exact
//! recipe. Tempered sampling `g_T(x)_k ∝ x^{1/T}` is the same seam with
//! `ΓT = (mN/α)·|1/T − 1|`.
//!
//! # Latent vs raw boundary (per `AGENTS.md` §"Latent vs Raw Space Rules")
//!
//! Beliefs are **semantic/think-brain** state: local, never synced. Only the
//! zone-level scalars (`U`, `V` — one f32 each, the flock-centroid precedent)
//! may cross a boundary. Sampling from a simplex is a distribution *draw*
//! (required by the model); the house sigmoid-not-softmax rule governs
//! projections onto direction vectors and is untouched here. Never route a
//! belief vector through a sync path; never reconstruct one from a synced
//! scalar.
//!
//! # Honesty note (UQ)
//!
//! These are dynamics of a **model**, not calibrated forecasts of any real
//! crowd. The laws are exact identities of the shipped kernel (pinned by the
//! tests) and mean-field approximations of it; the paper additionally
//! validates the mean-field shapes against GPT-4o / Claude Haiku populations.
//! Any future *prediction-quality* claim about real consumers is UQ-bearing
//! and owes the conformal-naive floor per `AGENTS.md` §"Report the Floor".
//!
//! # Assumptions (the paper's A1–A4, explicit)
//!
//! 1. simplex state — beliefs are L1-normalized non-negative vectors;
//! 2. well-mixed pairs — the pair source is a caller seam (spatial proximity,
//!    social graphs, or [`uniform_ordered_pair`] for the canonical uniform case);
//! 3. quantized channel — messages are drawn from the speaker's belief, not
//!    copied from it (Hard/Top-m; Soft is the analytic baseline);
//! 4. first-order adaptation — one EMA step of rate `α` per interaction.
//!
//! [`signed_coupling`]: crate::signed_coupling
//! [`SusceptibilityAccumulator`]: crate::signed_coupling::SusceptibilityAccumulator

/// Hard capacity for one belief vector in stack-buffer paths ([`QsgConfig::k`]
/// must not exceed this). The paper's experiments use K = 2…10; 16 leaves
/// headroom without ever touching the heap.
pub const MAX_K: usize = 16;

/// Clamp floor for [`qsg_normalize_belief_into`]: a belief whose total mass is
/// at or below this (after clamping negatives) is treated as unauthored and
/// reset to the uniform distribution.
pub const SIMPLEX_EPS: f32 = 1e-6;

/// How the speaker's belief becomes a message.
///
/// All three modes share `E[y | x_S] = x_S`; they differ only in variance,
/// which scales as `1/bandwidth`. `Soft` exists as the analytic baseline that
/// *removes* quantization noise — under it the symmetric state is an exact
/// fixed point (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageMode {
    /// One categorical draw: `y = e_k*`, `k* ~ Cat(x_S)`. The paper's `m = 1`.
    Hard,
    /// `m` i.i.d. categorical draws, transmitted as their empirical
    /// distribution. `TopM(0)` is rejected by [`QsgConfig::new`] (the sampler
    /// itself degrades it to a copy — the zero-variance limit).
    TopM(u32),
    /// The full belief distribution. No sampling noise; consensus pressure
    /// comes only from heterogeneity. Symmetry is preserved *exactly*.
    Soft,
}

impl MessageMode {
    /// Uniforms consumed per message (the RNG-free kernel's fuel gauge).
    #[must_use]
    pub fn draws(self) -> u32 {
        match self {
            MessageMode::Hard => 1,
            MessageMode::TopM(m) => m,
            MessageMode::Soft => 0,
        }
    }

    /// The theory's bandwidth `m` (`u32::MAX` standing in for the Soft `∞`).
    #[must_use]
    pub fn bandwidth(self) -> u32 {
        match self {
            MessageMode::Hard => 1,
            MessageMode::TopM(m) => m.max(1),
            MessageMode::Soft => u32::MAX,
        }
    }
}

/// Why a [`QsgConfig`] could not be built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QsgError {
    /// Belief dimension `K < 2`. One option is not a choice.
    KTooSmall(usize),
    /// `K > [`MAX_K`]`. Carries `(k, max_k)`.
    KExceedsCapacity { k: usize, max_k: usize },
    /// Population `N < 2` — no ordered speaker–listener pair exists.
    PopulationTooSmall(usize),
    /// `α ∉ (0, 1]` or non-finite. `α = 0` would freeze every belief; `α > 1`
    /// over-shoots the message and leaves the simplex.
    AlphaOutOfRange(f32),
    /// `TopM(0)` — a message made of zero samples carries no information and
    /// no variance law; pass [`MessageMode::Soft`] explicitly instead.
    ZeroBandwidth,
}

impl std::fmt::Display for QsgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QsgError::KTooSmall(k) => write!(f, "belief dimension K = {k} < 2"),
            QsgError::KExceedsCapacity { k, max_k } => {
                write!(f, "belief dimension K = {k} exceeds MAX_K = {max_k}")
            }
            QsgError::PopulationTooSmall(n) => write!(f, "population N = {n} < 2"),
            QsgError::AlphaOutOfRange(a) => write!(f, "adaptation rate alpha = {a} outside (0, 1]"),
            QsgError::ZeroBandwidth => {
                write!(f, "TopM(0) carries no samples; use MessageMode::Soft")
            }
        }
    }
}

impl std::error::Error for QsgError {}

/// Validated QSG population configuration.
///
/// `beliefs` is a row-major `n × k` slice ([`QsgConfig::beliefs_len`] gives
/// the required length) owned and reused by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QsgConfig {
    /// Population size `N ≥ 2`.
    pub n: usize,
    /// Belief dimension `K ∈ [2, MAX_K]` — the number of competing options.
    pub k: usize,
    /// Adaptation rate `α ∈ (0, 1]` — persona plasticity: how far one heard
    /// message moves the listener.
    pub alpha: f32,
    /// Message channel ([`MessageMode::Hard`] is the paper's `m = 1`).
    pub mode: MessageMode,
}

impl QsgConfig {
    /// Validate and build. The only fallible constructor — kernels assert the
    /// same invariants once per call as a backstop, but callers should treat a
    /// constructed config as proof.
    pub fn new(n: usize, k: usize, alpha: f32, mode: MessageMode) -> Result<Self, QsgError> {
        if n < 2 {
            return Err(QsgError::PopulationTooSmall(n));
        }
        if k < 2 {
            return Err(QsgError::KTooSmall(k));
        }
        if k > MAX_K {
            return Err(QsgError::KExceedsCapacity { k, max_k: MAX_K });
        }
        if !(alpha > 0.0 && alpha <= 1.0 && alpha.is_finite()) {
            return Err(QsgError::AlphaOutOfRange(alpha));
        }
        if matches!(mode, MessageMode::TopM(0)) {
            return Err(QsgError::ZeroBandwidth);
        }
        Ok(Self { n, k, alpha, mode })
    }

    /// Required length of the flattened row-major belief buffer.
    #[must_use]
    pub fn beliefs_len(&self) -> usize {
        self.n * self.k
    }
}

/// Caller-supplied uniform stream — the RNG-free kernel's fuel.
///
/// Seed addressability lives with the consumer (the `sample_states_into`
/// precedent): a replayable rollout is a replayable uniform stream. [`draw`]
/// panics on exhaustion with a message naming how many floats were asked for
/// in total, which turns an off-by-one in a consumer's budget into a labeled
/// failure instead of a silent stall.
///
/// [`draw`]: UniformStream::draw
#[derive(Debug, Clone, Copy)]
pub struct UniformStream<'a> {
    rest: &'a [f32],
}

impl<'a> UniformStream<'a> {
    /// Wrap a uniform slice (values in `[0, 1)`).
    #[must_use]
    pub fn new(uniforms: &'a [f32]) -> Self {
        Self { rest: uniforms }
    }

    /// Floats left.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.rest.len()
    }

    /// Next uniform, advancing the cursor.
    ///
    /// # Panics
    ///
    /// If the stream is exhausted — a consumer under-budgeted its uniforms.
    pub fn draw(&mut self) -> f32 {
        let (u, rest) = self
            .rest
            .split_first()
            .expect("qsg uniform stream exhausted: budget draws via MessageMode::draws()");
        self.rest = rest;
        *u
    }
}

/// Fill every agent's belief with the uniform distribution `1/K`.
///
/// The paper's symmetric initialization — the state Soft exchange can never
/// leave, and the reference point for drift measurements.
///
/// # Panics
///
/// If `beliefs.len() != n * k` or `k == 0`.
pub fn qsg_init_uniform_into(beliefs: &mut [f32], n: usize, k: usize) {
    assert!(k >= 1, "qsg_init_uniform_into: k must be at least 1");
    assert_eq!(beliefs.len(), n * k, "beliefs length must equal n * k");
    let u = 1.0 / k as f32;
    beliefs.fill(u);
}

/// Project one belief back onto the simplex in place.
///
/// Clamps negative coordinates to `0`, then L1-normalizes. A belief with no
/// usable mass (all-negative or sub-[`SIMPLEX_EPS`] total) resets to uniform.
/// The QSG update never leaves the simplex on its own (convex combinations of
/// simplex points); this is for callers that *author* beliefs from arbitrary
/// floats (HLA projections, logits) before handing them to the dynamics.
pub fn qsg_normalize_belief_into(belief: &mut [f32]) {
    let k = belief.len();
    if k == 0 {
        return;
    }
    let mut total = 0.0f32;
    for v in belief.iter_mut() {
        *v = v.max(0.0);
        total += *v;
    }
    if total <= SIMPLEX_EPS {
        let u = 1.0 / k as f32;
        belief.fill(u);
        return;
    }
    for v in belief.iter_mut() {
        *v /= total;
    }
}

/// Inverse-CDF categorical draw over an L1-normalized `belief`.
///
/// Strict `<` threshold walk, matching `sample_states_into`'s tie rule (a
/// uniform exactly on a bucket boundary falls through to the next option).
/// Falls back to the last index if float rounding leaves the walk short —
/// the belief is normalized, so this fires only at the `u → 1` edge.
///
/// O(k); no allocation.
#[must_use]
pub fn qsg_draw_categorical(belief: &[f32], u: f32) -> usize {
    let mut acc = 0.0f32;
    for (idx, &p) in belief.iter().enumerate() {
        acc += p;
        if u < acc {
            return idx;
        }
    }
    belief.len() - 1
}

/// Sample one message from the speaker's belief into `msg`.
///
/// Zero-allocation; `msg` is caller-owned scratch of length **≥**
/// `speaker.len()` — a MAX_K-sized stack buffer is the intended shape, and
/// only `msg[..k]` is ever meaningful (the fill zeroes the whole scratch, so
/// stale tails stay harmless). Consumes [`MessageMode::draws`] uniforms from
/// the stream. `TopM(0)` degrades to a copy (the zero-variance limit) —
/// [`QsgConfig`] rejects it up front.
///
/// # Panics
///
/// If `msg.len() < speaker.len()`, or the uniform stream runs dry.
pub fn qsg_sample_message_into(
    msg: &mut [f32],
    speaker: &[f32],
    mode: MessageMode,
    uniforms: &mut UniformStream<'_>,
) {
    assert!(
        msg.len() >= speaker.len(),
        "message scratch must be at least the belief length"
    );
    let k = speaker.len();
    msg.fill(0.0);
    match mode {
        MessageMode::Soft => msg[..k].copy_from_slice(speaker),
        MessageMode::Hard => {
            let k_star = qsg_draw_categorical(speaker, uniforms.draw());
            msg[k_star] = 1.0;
        }
        MessageMode::TopM(0) => msg[..k].copy_from_slice(speaker),
        MessageMode::TopM(m) => {
            for _ in 0..m {
                let k_star = qsg_draw_categorical(speaker, uniforms.draw());
                msg[k_star] += 1.0;
            }
            let inv = 1.0 / m as f32;
            for v in msg[..k].iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// The listener half alone: `x_L ← (1−α)·x_L + α·message[..k]`, in place.
///
/// Split from [`qsg_gossip_step_into`] so callers that author their own
/// messages (deterministic top-K slices, tilted/tempered channels for the
/// drift-vs-selection experiments, budget-capped wire payloads) reuse the
/// exact adaptation the theory analyzes instead of re-deriving the blend.
/// A `message` longer than `cfg.k` is truncated — only the first `k`
/// coordinates participate.
///
/// # Panics
///
/// On any shape disagreement (`beliefs` vs `cfg`, `message.len() < cfg.k`,
/// `listener ≥ cfg.n`).
pub fn qsg_blend_listener_into(
    beliefs: &mut [f32],
    cfg: &QsgConfig,
    listener: usize,
    message: &[f32],
) {
    assert!(cfg.k <= MAX_K, "cfg.k exceeds MAX_K");
    assert_eq!(
        beliefs.len(),
        cfg.beliefs_len(),
        "beliefs length must equal cfg.beliefs_len()"
    );
    assert!(
        message.len() >= cfg.k,
        "message length must be at least cfg.k"
    );
    assert!(listener < cfg.n, "listener index out of range");
    let a = cfg.alpha;
    let base = listener * cfg.k;
    let row = &mut beliefs[base..base + cfg.k];
    for (b, &m) in row.iter_mut().zip(message[..cfg.k].iter()) {
        *b = (1.0 - a) * *b + a * m;
    }
}

/// One QSG interaction: sample the speaker's message, blend the listener.
///
/// Order matters and is fixed here: the message is sampled from the speaker's
/// **pre-update** row (copied to a stack buffer first, so a speaker that is
/// also structurally the listener's neighbor is unaffected — pairs are
/// distinct by contract anyway). The listener row is then blended in place.
/// Zero allocation; `msg` scratch must be ≥ `cfg.k` (a MAX_K-sized buffer is
/// the intended shape). Consumes `mode.draws()` uniforms.
///
/// # Panics
///
/// On shape disagreements, `speaker == listener`, out-of-range indices, or
/// `cfg.k > MAX_K`.
pub fn qsg_gossip_step_into(
    beliefs: &mut [f32],
    cfg: &QsgConfig,
    speaker: usize,
    listener: usize,
    msg: &mut [f32],
    uniforms: &mut UniformStream<'_>,
) {
    assert!(cfg.k <= MAX_K, "cfg.k exceeds MAX_K");
    assert_eq!(
        beliefs.len(),
        cfg.beliefs_len(),
        "beliefs length must equal cfg.beliefs_len()"
    );
    assert!(msg.len() >= cfg.k, "message scratch must be at least cfg.k");
    assert!(speaker < cfg.n, "speaker index out of range");
    assert!(listener < cfg.n, "listener index out of range");
    assert!(
        speaker != listener,
        "speaker and listener must be distinct agents"
    );

    let k = cfg.k;
    let mut src = [0.0f32; MAX_K];
    src[..k].copy_from_slice(&beliefs[speaker * k..speaker * k + k]);
    qsg_sample_message_into(msg, &src[..k], cfg.mode, uniforms);
    qsg_blend_listener_into(beliefs, cfg, listener, msg);
}

/// Run `interactions` QSG steps with pair selection delegated to the caller.
///
/// `next_pair` returns `(speaker, listener)`, distinct and in range — for the
/// paper's well-mixed protocol wrap [`uniform_ordered_pair`] over a uniform
/// source; game consumers substitute spatial or social pairing without
/// touching the kernel. Zero allocation; the message scratch lives on the
/// stack. Consumes `interactions × mode.draws()` uniforms.
///
/// # Panics
///
/// Propagates [`qsg_gossip_step_into`]'s panics; a pair source that returns
/// `speaker == listener` is a contract violation and will trip the step's
/// assert.
pub fn qsg_gossip_run_into(
    beliefs: &mut [f32],
    cfg: &QsgConfig,
    interactions: usize,
    next_pair: &mut impl FnMut() -> (usize, usize),
    uniforms: &mut UniformStream<'_>,
) {
    let mut msg = [0.0f32; MAX_K];
    for _ in 0..interactions {
        let (s, l) = next_pair();
        qsg_gossip_step_into(beliefs, cfg, s, l, &mut msg, uniforms);
    }
}

/// Map two uniforms to an ordered pair of **distinct** agents — the paper's
/// uniform speaker–listener protocol as a pure function.
///
/// `s = min(floor(u_speaker·N), N−1)`; the listener index covers the
/// remaining `N−1` slots by skipping `s`. Deterministic, allocation-free,
/// surjective onto all `N(N−1)` ordered pairs.
///
/// # Panics
///
/// If `n < 2` (no distinct pair exists).
#[must_use]
pub fn uniform_ordered_pair(n: usize, u_speaker: f32, u_listener: f32) -> (usize, usize) {
    assert!(n >= 2, "uniform_ordered_pair needs n >= 2");
    let s = ((u_speaker.clamp(0.0, 1.0)) * n as f32) as usize;
    let s = s.min(n - 1);
    let raw = ((u_listener.clamp(0.0, 1.0)) * (n - 1) as f32) as usize;
    let raw = raw.min(n - 2);
    let l = if raw >= s { raw + 1 } else { raw };
    (s, l)
}

/// Population mean belief `x̄` into caller-owned `out` (length `k`).
///
/// # Panics
///
/// If `out.len() != k` or `beliefs.len() != n * k`.
pub fn qsg_mean_into(out: &mut [f32], beliefs: &[f32], n: usize, k: usize) {
    assert_eq!(out.len(), k, "out length must equal k");
    assert_eq!(beliefs.len(), n * k, "beliefs length must equal n * k");
    if n == 0 {
        out.fill(0.0);
        return;
    }
    out.fill(0.0);
    for row in beliefs.chunks_exact(k) {
        for (o, &v) in out.iter_mut().zip(row.iter()) {
            *o += v;
        }
    }
    let inv = 1.0 / n as f32;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Polarization `U = ‖x̄‖²₂ ∈ [1/K, 1]` — the order parameter.
///
/// `1/K` at perfect symmetry, `1` at consensus; direction-blind, so a crowd
/// that collapses onto *any* option reads the same. `0.0` for an empty slice
/// (the module's degenerate-reducer convention).
#[must_use]
pub fn qsg_polarization_u(mean: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &m in mean {
        acc += m * m;
    }
    acc
}

/// Disagreement energy `V = Σ_i ‖x_i − x̄‖²₂` — how dispersed the agents are
/// around the population mean. `0.0` when `n == 0`.
///
/// # Panics
///
/// If `beliefs.len() != n * mean.len()`.
#[must_use]
pub fn qsg_disagreement_v(beliefs: &[f32], mean: &[f32], n: usize) -> f32 {
    let k = mean.len();
    assert_eq!(beliefs.len(), n * k, "beliefs length must equal n * k");
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for row in beliefs.chunks_exact(k) {
        for (&b, &m) in row.iter().zip(mean.iter()) {
            let d = b - m;
            acc += d * d;
        }
    }
    acc
}

/// Coordination rate `S = U − V/(N(N−1))` — the mean pairwise belief overlap.
///
/// Non-decreasing in expectation under every message mode (the paper's Eq.
/// 35). `0.0` for `n < 2` (no pairs), matching the degenerate-reducer
/// convention.
#[must_use]
pub fn qsg_coordination_s(u: f32, v: f32, n: usize) -> f32 {
    if n < 2 {
        return 0.0;
    }
    u - v / (n * (n - 1)) as f32
}

/// Speaker uncertainty `1 − ‖x‖²₂` — the fuel of memetic drift.
///
/// Zero at a one-hot vertex (a certain speaker injects no noise), maximal at
/// the simplex center. The Thm 1/Thm 2 drift terms are proportional to this.
/// `1.0` for an empty slice (maximal uncertainty — nothing is known).
#[must_use]
pub fn qsg_uncertainty(belief: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &p in belief {
        acc += p * p;
    }
    1.0 - acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal in-house RNG (splitmix64 — the house test pattern).
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in `[0, 1)` at 24-bit resolution.
        fn uniform(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32) / 16_777_216.0
        }
    }

    /// Random L1-normalized belief population with per-row peakedness variety
    /// (exponent 1 = flat-ish, 3 = peaked).
    fn random_beliefs(n: usize, k: usize, rng: &mut SplitMix64, out: &mut [f32]) {
        assert_eq!(out.len(), n * k);
        for i in 0..n {
            let exp = 1.0 + (i % 3) as f32;
            let row = &mut out[i * k..i * k + k];
            let mut total = 0.0f32;
            for v in row.iter_mut() {
                *v = rng.uniform().powf(exp);
                total += *v;
            }
            for v in row.iter_mut() {
                *v /= total;
            }
        }
    }

    fn sq_norm(row: &[f32]) -> f32 {
        row.iter().map(|p| p * p).sum()
    }

    // ── config + validation ──

    #[test]
    fn config_rejects_invalid_populations_dimensions_and_rates() {
        assert_eq!(
            QsgConfig::new(1, 3, 0.5, MessageMode::Hard).unwrap_err(),
            QsgError::PopulationTooSmall(1)
        );
        assert_eq!(
            QsgConfig::new(8, 1, 0.5, MessageMode::Hard).unwrap_err(),
            QsgError::KTooSmall(1)
        );
        assert_eq!(
            QsgConfig::new(8, MAX_K + 1, 0.5, MessageMode::Hard).unwrap_err(),
            QsgError::KExceedsCapacity {
                k: MAX_K + 1,
                max_k: MAX_K
            }
        );
        assert_eq!(
            QsgConfig::new(8, 3, 0.0, MessageMode::Hard).unwrap_err(),
            QsgError::AlphaOutOfRange(0.0)
        );
        assert_eq!(
            QsgConfig::new(8, 3, 1.5, MessageMode::Hard).unwrap_err(),
            QsgError::AlphaOutOfRange(1.5)
        );
        assert!(
            matches!(
                QsgConfig::new(8, 3, f32::NAN, MessageMode::Hard),
                Err(QsgError::AlphaOutOfRange(a)) if a.is_nan()
            ),
            "NaN alpha must be rejected as AlphaOutOfRange"
        );
        assert_eq!(
            QsgConfig::new(8, 3, 0.5, MessageMode::TopM(0)).unwrap_err(),
            QsgError::ZeroBandwidth
        );
        let ok = QsgConfig::new(8, 3, 0.5, MessageMode::TopM(4)).unwrap();
        assert_eq!(ok.beliefs_len(), 24);
        // α = 1 (the voter limit) is in range.
        assert!(QsgConfig::new(8, 3, 1.0, MessageMode::Hard).is_ok());
    }

    #[test]
    fn mode_dials_report_draws_and_bandwidth() {
        assert_eq!(MessageMode::Hard.draws(), 1);
        assert_eq!(MessageMode::TopM(7).draws(), 7);
        assert_eq!(MessageMode::Soft.draws(), 0);
        assert_eq!(MessageMode::Hard.bandwidth(), 1);
        assert_eq!(MessageMode::TopM(7).bandwidth(), 7);
        assert_eq!(MessageMode::Soft.bandwidth(), u32::MAX);
    }

    // ── init + normalize ──

    #[test]
    fn init_uniform_is_symmetric_and_normalized() {
        let mut b = vec![0.0f32; 6 * 4];
        qsg_init_uniform_into(&mut b, 6, 4);
        for row in b.as_chunks::<4>().0 {
            assert!(row.iter().all(|&p| (p - 0.25).abs() < 1e-7));
        }
        let mut mean = [0.0f32; 4];
        qsg_mean_into(&mut mean, &b, 6, 4);
        assert!((qsg_polarization_u(&mean) - 0.25).abs() < 1e-7);
    }

    #[test]
    fn normalize_clamps_renegades_and_resets_degenerate_rows() {
        let mut b = [-0.5f32, 1.0, 0.5];
        qsg_normalize_belief_into(&mut b);
        assert!((b[0] - 0.0).abs() < 1e-7);
        assert!((b[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((b[2] - 1.0 / 3.0).abs() < 1e-6);

        let mut dead = [-1.0f32, -2.0, 0.0];
        qsg_normalize_belief_into(&mut dead);
        assert!(dead.iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-7));

        let mut already = [0.25f32; 4];
        let snapshot = already;
        qsg_normalize_belief_into(&mut already);
        assert_eq!(already, snapshot);
    }

    // ── categorical draw + message sampling ──

    #[test]
    fn draw_categorical_walks_the_cdf_with_strict_less_ties() {
        let belief = [0.5f32, 0.5];
        assert_eq!(qsg_draw_categorical(&belief, 0.0), 0);
        assert_eq!(qsg_draw_categorical(&belief, 0.49), 0);
        // Strict <: a uniform exactly on the boundary falls to the next bucket.
        assert_eq!(qsg_draw_categorical(&belief, 0.5), 1);
        let skewed = [0.2f32, 0.3, 0.5];
        assert_eq!(qsg_draw_categorical(&skewed, 0.55), 2);
        // Fallback: u at the top edge still lands inside the support.
        assert_eq!(qsg_draw_categorical(&skewed, 1.0 - 1e-7), 2);
    }

    #[test]
    fn hard_message_is_one_hot_at_the_drawn_option() {
        let belief = [0.2f32, 0.3, 0.5];
        let uniforms = [0.55f32];
        let mut stream = UniformStream::new(&uniforms);
        let mut msg = [0.0f32; 3];
        qsg_sample_message_into(&mut msg, &belief, MessageMode::Hard, &mut stream);
        assert_eq!(msg, [0.0, 0.0, 1.0]);
        assert_eq!(stream.remaining(), 0);
    }

    #[test]
    fn topm_message_is_an_empirical_distribution() {
        let belief = [0.2f32, 0.3, 0.5];
        // 7 draws all at u = 0.55 → every draw lands on option 2.
        let uniforms = [0.55f32; 7];
        let mut stream = UniformStream::new(&uniforms);
        let mut msg = [0.0f32; 3];
        qsg_sample_message_into(&mut msg, &belief, MessageMode::TopM(7), &mut stream);
        assert!((msg[2] - 1.0).abs() < 1e-7);
        assert!((msg[0] + msg[1] + msg[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn topm_one_consumes_exactly_one_uniform_like_hard() {
        let belief = [0.2f32, 0.3, 0.5];
        let mut hard_msg = [0.0f32; 3];
        let mut topm_msg = [0.0f32; 3];
        let mut s1 = UniformStream::new(&[0.55]);
        let mut s2 = UniformStream::new(&[0.55]);
        qsg_sample_message_into(&mut hard_msg, &belief, MessageMode::Hard, &mut s1);
        qsg_sample_message_into(&mut topm_msg, &belief, MessageMode::TopM(1), &mut s2);
        assert_eq!(hard_msg, topm_msg);
    }

    #[test]
    fn soft_message_copies_and_topm_zero_degrades_to_copy() {
        let belief = [0.2f32, 0.3, 0.5];
        let mut msg = [0.0f32; 3];
        let mut stream = UniformStream::new(&[]);
        qsg_sample_message_into(&mut msg, &belief, MessageMode::Soft, &mut stream);
        assert_eq!(msg, belief);
        qsg_sample_message_into(&mut msg, &belief, MessageMode::TopM(0), &mut stream);
        assert_eq!(msg, belief);
    }

    // ── pairs ──

    #[test]
    fn uniform_pairs_are_distinct_in_range_and_deterministic() {
        let mut rng = SplitMix64::new(0xABCD);
        let mut hits = [[0usize; 5]; 5];
        for _ in 0..4000 {
            let (s, l) = uniform_ordered_pair(5, rng.uniform(), rng.uniform());
            assert!(s < 5 && l < 5 && s != l);
            hits[s][l] += 1;
        }
        // Every ordered pair is reachable and got non-starving coverage.
        for (s, row) in hits.iter().enumerate() {
            for (l, &count) in row.iter().enumerate() {
                if s != l {
                    assert!(count >= 40, "pair ({s},{l}) starved: {count}");
                }
            }
        }
        assert_eq!(uniform_ordered_pair(4, 0.0, 0.0), (0, 1));
        // u_listener = 1.0 clamps the raw slot to N−2 = 2; s = 3 < 2 is
        // false, so the listener stays 2 (never wraps onto the speaker).
        assert_eq!(uniform_ordered_pair(4, 1.0, 1.0), (3, 2));
    }

    // ── step + reducers on hand-computed numbers ──

    #[test]
    fn one_step_matches_the_hand_computed_blend_and_u() {
        // Rows: a = [1,0], b = [0,1], c = [½,½]. Speaker a, listener b, draw
        // u = 0 → message e₀. α = ½ → b becomes [½,½].
        let cfg = QsgConfig::new(3, 2, 0.5, MessageMode::Hard).unwrap();
        let mut beliefs = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let mut mean = [0.0f32; 2];
        qsg_mean_into(&mut mean, &beliefs, 3, 2);
        let u_before = qsg_polarization_u(&mean); // ‖(½,½)‖² = ½

        let mut msg = [0.0f32; 2];
        let mut stream = UniformStream::new(&[0.0]);
        qsg_gossip_step_into(&mut beliefs, &cfg, 0, 1, &mut msg, &mut stream);

        assert_eq!(beliefs[2], 0.5);
        assert_eq!(beliefs[3], 0.5);
        // Speaker untouched.
        assert_eq!((beliefs[0], beliefs[1]), (1.0, 0.0));

        qsg_mean_into(&mut mean, &beliefs, 3, 2);
        let u_after = qsg_polarization_u(&mean);
        assert!((u_before - 0.5).abs() < 1e-7);
        assert!((u_after - 5.0 / 9.0).abs() < 1e-6);

        // S = U − V/(N(N−1)) on the post state; V = Σ‖x_i − x̄‖².
        let v = qsg_disagreement_v(&beliefs, &mean, 3);
        let s = qsg_coordination_s(u_after, v, 3);
        assert!((s - (u_after - v / 6.0)).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "message length must be at least cfg.k")]
    fn blend_listener_rejects_shape_drift() {
        let cfg = QsgConfig::new(2, 3, 0.5, MessageMode::Hard).unwrap();
        let mut beliefs = vec![0.0f32; 6];
        let bad = [0.1f32; 2]; // wrong width
        qsg_blend_listener_into(&mut beliefs, &cfg, 0, &bad);
    }

    #[test]
    fn reducers_handle_degenerate_input_without_nan() {
        assert_eq!(qsg_polarization_u(&[]), 0.0);
        assert_eq!(qsg_disagreement_v(&[], &[], 0), 0.0);
        assert_eq!(qsg_coordination_s(0.5, 0.1, 1), 0.0);
        assert_eq!(qsg_coordination_s(0.5, 0.1, 0), 0.0);
        assert!((qsg_uncertainty(&[]) - 1.0).abs() < 1e-7);
        let mut mean = [9.0f32; 3];
        qsg_mean_into(&mut mean, &[], 0, 3);
        assert_eq!(mean, [0.0, 0.0, 0.0]);
    }

    // ── the laws (each pinned per the module-doc table) ──

    /// N=24, K=10, α=0.5 — the paper's simulation defaults.
    const MC_N: usize = 24;
    const MC_K: usize = 10;
    const MC_ALPHA: f32 = 0.5;

    #[test]
    fn theorem1_hard_variance_injection_identity() {
        // Paired estimator: per ordered pair, average the kernel-measured
        // ΔU over Q sampled Hard messages and subtract the deterministic Soft
        // ΔU; the pair average must match the analytic per-pair injection
        // α²/N²·(1 − ‖x_S‖²). Pair-selection variance cancels in the
        // difference of means, so a 1% tolerance is ~10σ.
        let n = MC_N;
        let k = MC_K;
        let alpha = MC_ALPHA;
        let mut rng = SplitMix64::new(0x51D_00B5);
        let mut snapshot = vec![0.0f32; n * k];
        random_beliefs(n, k, &mut rng, &mut snapshot);

        let soft_cfg = QsgConfig::new(n, k, alpha, MessageMode::Soft).unwrap();
        let hard_cfg = QsgConfig::new(n, k, alpha, MessageMode::Hard).unwrap();

        let mut mean = [0.0f32; MC_K];
        qsg_mean_into(&mut mean, &snapshot, n, k);
        let u0 = qsg_polarization_u(&mean);

        let pairs = 8000;
        let q_draws = 32;
        let mut uniforms = vec![0.0f32; pairs * q_draws];
        for u in uniforms.iter_mut() {
            *u = rng.uniform();
        }
        let mut stream = UniformStream::new(&uniforms);

        let mut working = vec![0.0f32; n * k];
        let mut msg = [0.0f32; MAX_K];
        let mut acc_diff = 0.0f64;
        let mut acc_analytic = 0.0f64;

        for _ in 0..pairs {
            let s = (rng.uniform() * n as f32) as usize % n;
            let l = (rng.uniform() * (n - 1) as f32) as usize % (n - 1);
            let l = if l >= s { l + 1 } else { l };

            // Deterministic Soft ΔU for this pair.
            working.copy_from_slice(&snapshot);
            qsg_gossip_step_into(&mut working, &soft_cfg, s, l, &mut msg, &mut stream);
            qsg_mean_into(&mut mean, &working, n, k);
            let du_soft = qsg_polarization_u(&mean) - u0;

            // Q sampled Hard ΔUs for the same pair.
            let mut du_hard_mean = 0.0f64;
            for _ in 0..q_draws {
                working.copy_from_slice(&snapshot);
                qsg_gossip_step_into(&mut working, &hard_cfg, s, l, &mut msg, &mut stream);
                qsg_mean_into(&mut mean, &working, n, k);
                du_hard_mean += (qsg_polarization_u(&mean) - u0) as f64;
            }
            du_hard_mean /= q_draws as f64;

            let analytic = (alpha * alpha / (n * n) as f32
                * (1.0 - sq_norm(&snapshot[s * k..s * k + k]))) as f64;
            acc_diff += du_hard_mean - du_soft as f64;
            acc_analytic += analytic;
        }

        let lhs = acc_diff / pairs as f64;
        let rhs = acc_analytic / pairs as f64;
        assert!(
            (lhs - rhs).abs() <= 0.01 * rhs.abs(),
            "Thm 1 identity drifted: measured {lhs:.3e} vs analytic {rhs:.3e}"
        );
        assert!(lhs > 0.0, "Hard sampling must add nonnegative drift");
    }

    #[test]
    fn theorem2_topm_bandwidth_identity() {
        // Same paired estimator, TopM channel: the excess drift over Soft must
        // equal α²/N²·(1/m)·(1 − ‖x_S‖²) for every m.
        let n = MC_N;
        let k = MC_K;
        let alpha = MC_ALPHA;
        let mut rng = SplitMix64::new(0x70F_B113);
        let mut snapshot = vec![0.0f32; n * k];
        random_beliefs(n, k, &mut rng, &mut snapshot);

        let soft_cfg = QsgConfig::new(n, k, alpha, MessageMode::Soft).unwrap();

        let mut mean = [0.0f32; MC_K];
        qsg_mean_into(&mut mean, &snapshot, n, k);
        let u0 = qsg_polarization_u(&mean);

        let pairs = 12000;
        let q_draws = 32;

        for m in [1u32, 2, 3, 5, 10] {
            let cfg = QsgConfig::new(n, k, alpha, MessageMode::TopM(m)).unwrap();
            let mut uniforms = vec![0.0f32; pairs * q_draws * m as usize];
            for u in uniforms.iter_mut() {
                *u = rng.uniform();
            }
            let mut stream = UniformStream::new(&uniforms);
            let mut working = vec![0.0f32; n * k];
            let mut msg = [0.0f32; MAX_K];
            let mut acc_diff = 0.0f64;
            let mut acc_analytic = 0.0f64;

            for _ in 0..pairs {
                let s = (rng.uniform() * n as f32) as usize % n;
                let l = (rng.uniform() * (n - 1) as f32) as usize % (n - 1);
                let l = if l >= s { l + 1 } else { l };

                working.copy_from_slice(&snapshot);
                qsg_gossip_step_into(&mut working, &soft_cfg, s, l, &mut msg, &mut stream);
                qsg_mean_into(&mut mean, &working, n, k);
                let du_soft = qsg_polarization_u(&mean) - u0;

                let mut du_m_mean = 0.0f64;
                for _ in 0..q_draws {
                    working.copy_from_slice(&snapshot);
                    qsg_gossip_step_into(&mut working, &cfg, s, l, &mut msg, &mut stream);
                    qsg_mean_into(&mut mean, &working, n, k);
                    du_m_mean += (qsg_polarization_u(&mean) - u0) as f64;
                }
                du_m_mean /= q_draws as f64;

                let fuel = 1.0 - sq_norm(&snapshot[s * k..s * k + k]);
                let analytic = (alpha * alpha / ((m as usize * n * n) as f32)) * fuel;
                acc_diff += du_m_mean - du_soft as f64;
                acc_analytic += analytic as f64;
            }

            let lhs = acc_diff / pairs as f64;
            let rhs = acc_analytic / pairs as f64;
            assert!(
                (lhs - rhs).abs() <= 0.015 * rhs.abs(),
                "Thm 2 identity drifted at m={m}: measured {lhs:.3e} vs analytic {rhs:.3e}"
            );
        }
    }

    #[test]
    fn symmetric_1_over_m_corollary() {
        // At the perfectly symmetric initialization the per-step drift is
        // α²/(mN²)·(1−1/K) — the paper's Fig. 6c line, (1−1/K)/m normalized.
        let n = MC_N;
        let k = MC_K;
        let alpha = MC_ALPHA;
        let mut rng = SplitMix64::new(0x10BEEF5);
        let draws = 4000;

        for m in [1u32, 2, 3, 5, 10] {
            let cfg = QsgConfig::new(n, k, alpha, MessageMode::TopM(m)).unwrap();
            let symmetric = vec![1.0 / k as f32; n * k];
            let mut uniforms = vec![0.0f32; draws * m as usize];
            for u in uniforms.iter_mut() {
                *u = rng.uniform();
            }
            let mut stream = UniformStream::new(&uniforms);
            let mut working = vec![0.0f32; n * k];
            let mut msg = [0.0f32; MAX_K];
            let mut acc = 0.0f64;
            for _ in 0..draws {
                working.copy_from_slice(&symmetric);
                let s = (rng.uniform() * n as f32) as usize % n;
                let l = (rng.uniform() * (n - 1) as f32) as usize % (n - 1);
                let l = if l >= s { l + 1 } else { l };
                qsg_gossip_step_into(&mut working, &cfg, s, l, &mut msg, &mut stream);
                let mut mean = [0.0f32; MC_K];
                qsg_mean_into(&mut mean, &working, n, k);
                acc += (qsg_polarization_u(&mean) - 1.0 / k as f32) as f64;
            }
            let measured = (acc / draws as f64) as f32;
            let theory = alpha * alpha / ((m as usize * n * n) as f32) * (1.0 - 1.0 / k as f32);
            assert!(
                (measured - theory).abs() <= 0.03 * theory,
                "1/m corollary drifted at m={m}: measured {measured:.3e} vs theory {theory:.3e}"
            );
            // The paper's normalized form: (N²/α²)·drift ≈ (1−1/K)/m.
            let normalized = (n * n) as f32 / (alpha * alpha) * measured;
            let line = (1.0 - 1.0 / k as f32) / m as f32;
            assert!(
                (normalized - line).abs() <= 0.03 * line,
                "normalized line off at m={m}"
            );
        }
    }

    #[test]
    fn soft_exchange_preserves_the_mean() {
        // E[x̄' | X] = x̄ coordinate-wise (a bounded martingale): Soft ΔU per
        // pair is deterministic, so averaging over pairs must return the same
        // mean to MC precision.
        let n = MC_N;
        let k = MC_K;
        let alpha = MC_ALPHA;
        let mut rng = SplitMix64::new(0x3A5E_BA11);
        let mut snapshot = vec![0.0f32; n * k];
        random_beliefs(n, k, &mut rng, &mut snapshot);
        let cfg = QsgConfig::new(n, k, alpha, MessageMode::Soft).unwrap();

        let mut mean = [0.0f32; MC_K];
        qsg_mean_into(&mut mean, &snapshot, n, k);

        let pairs = 20000;
        let mut drift = [0.0f64; MC_K];
        let mut working = vec![0.0f32; n * k];
        let mut msg = [0.0f32; MAX_K];
        let mut stream = UniformStream::new(&[]);
        for _ in 0..pairs {
            working.copy_from_slice(&snapshot);
            let s = (rng.uniform() * n as f32) as usize % n;
            let l = (rng.uniform() * (n - 1) as f32) as usize % (n - 1);
            let l = if l >= s { l + 1 } else { l };
            qsg_gossip_step_into(&mut working, &cfg, s, l, &mut msg, &mut stream);
            let mut new_mean = [0.0f32; MC_K];
            qsg_mean_into(&mut new_mean, &working, n, k);
            for j in 0..k {
                drift[j] += (new_mean[j] - mean[j]) as f64;
            }
        }
        for (j, d) in drift.iter().enumerate().take(k) {
            let avg = d / pairs as f64;
            assert!(
                avg.abs() < 4.0e-4,
                "coordinate {j} drifted by {avg:.3e} over {pairs} pairs"
            );
        }
    }

    #[test]
    fn soft_exchange_contracts_disagreement() {
        // E[ΔV | X]_soft = −(2α/(N−1))·(1−α+α/N)·V ≤ 0 — full-belief
        // exchange smooths heterogeneity and can NEVER break symmetry.
        // Provenance note: the scraped PDF text renders the last factor as
        // (1−α+αN); the α→1 limit disambiguates — that form would collapse V
        // in ONE step, but α=1 is the voter model whose consensus takes
        // ~N² steps, so the per-step contraction must be O(1/N²)-class. The
        // corrected form gives −2V/(N(N−1)) at α=1 ✓ and matches the kernel
        // (this test IS that check).
        // symmetry. Reset per pair so the conditional identity is measured
        // against the same X throughout.
        let n = MC_N;
        let k = MC_K;
        let alpha = MC_ALPHA;
        let mut rng = SplitMix64::new(0xD15ABEE);
        let mut snapshot = vec![0.0f32; n * k];
        random_beliefs(n, k, &mut rng, &mut snapshot);
        let cfg = QsgConfig::new(n, k, alpha, MessageMode::Soft).unwrap();

        let mut mean = [0.0f32; MC_K];
        qsg_mean_into(&mut mean, &snapshot, n, k);
        let v0 = qsg_disagreement_v(&snapshot, &mean, n);
        let theory = -2.0 * alpha / (n - 1) as f32 * (1.0 - alpha + alpha / n as f32) * v0;

        let pairs = 60000;
        let mut acc = 0.0f64;
        let mut working = vec![0.0f32; n * k];
        let mut msg = [0.0f32; MAX_K];
        let mut stream = UniformStream::new(&[]);
        for _ in 0..pairs {
            working.copy_from_slice(&snapshot);
            let s = (rng.uniform() * n as f32) as usize % n;
            let l = (rng.uniform() * (n - 1) as f32) as usize % (n - 1);
            let l = if l >= s { l + 1 } else { l };
            qsg_gossip_step_into(&mut working, &cfg, s, l, &mut msg, &mut stream);
            let mut new_mean = [0.0f32; MC_K];
            qsg_mean_into(&mut new_mean, &working, n, k);
            acc += (qsg_disagreement_v(&working, &new_mean, n) - v0) as f64;
        }
        let measured = (acc / pairs as f64) as f32;
        println!("soft contraction: measured {measured:.5e} vs theory {theory:.5e} (V0 = {v0:.4})");
        assert!(
            (measured - theory).abs() <= 0.015 * theory.abs(),
            "Soft contraction drifted: measured {measured:.4e} vs theory {theory:.4e}"
        );
        assert!(
            measured < 0.0,
            "Soft exchange must not increase disagreement"
        );
    }

    #[test]
    fn alpha_one_reduces_to_uniform_winner_voter_model() {
        // α = 1 overwrites listeners with one-hot messages → after the coupon-
        // collector warmup every row is one-hot and the tail is the K-state
        // voter process on the complete graph. Winner probability = x̄_k(0) =
        // 1/K at neutral initialization (the martingale/optional-stopping
        // result) — i.e. under pure drift the winner is a fair lottery.
        let n = 12;
        let k = 3;
        let cfg = QsgConfig::new(n, k, 1.0, MessageMode::Hard).unwrap();
        let runs = 1500;
        let steps = 1500;
        let mut rng = SplitMix64::new(0x0FEA5EB1E);
        let mut winner_freq = [0usize; 3];
        let mut consensus_runs = 0usize;

        for _ in 0..runs {
            let mut beliefs = vec![1.0 / k as f32; n * k];
            let mut uniforms = vec![0.0f32; steps];
            for u in uniforms.iter_mut() {
                *u = rng.uniform();
            }
            let mut stream = UniformStream::new(&uniforms);
            let mut pair_rng = SplitMix64::new(rng.next_u64());
            let mut next_pair = || uniform_ordered_pair(n, pair_rng.uniform(), pair_rng.uniform());
            qsg_gossip_run_into(&mut beliefs, &cfg, steps, &mut next_pair, &mut stream);

            let winner_row = &beliefs[..k];
            let w = winner_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap();
            let one_hot_consensus = beliefs.chunks_exact(k).all(|row| {
                row.iter().enumerate().all(|(j, &p)| {
                    if j == w {
                        (p - 1.0).abs() < 1e-5
                    } else {
                        p < 1e-5
                    }
                })
            });
            if one_hot_consensus {
                consensus_runs += 1;
                winner_freq[w] += 1;
            }
        }
        assert!(
            consensus_runs >= runs * 98 / 100,
            "voter consensus missed too often: {consensus_runs}/{runs}"
        );
        for (w, &count) in winner_freq.iter().enumerate() {
            let freq = count as f32 / consensus_runs as f32;
            assert!(
                (freq - 1.0 / 3.0).abs() < 0.05,
                "winner {w} frequency {freq:.3} off the uniform ⅓ lottery"
            );
        }
    }

    #[test]
    fn soft_from_symmetry_never_breaks_it() {
        // The exact neutral baseline: Soft exchange from x_i = 1/K keeps every
        // belief at 1/K forever (each update target equals the current row).
        let n = 8;
        let k = 4;
        let cfg = QsgConfig::new(n, k, 0.7, MessageMode::Soft).unwrap();
        let mut beliefs = vec![1.0 / k as f32; n * k];
        let mut pair_rng = SplitMix64::new(0x5ABE11E);
        let mut stream = UniformStream::new(&[]);
        let mut msg = [0.0f32; MAX_K];
        for _ in 0..500 {
            let (s, l) = uniform_ordered_pair(n, pair_rng.uniform(), pair_rng.uniform());
            qsg_gossip_step_into(&mut beliefs, &cfg, s, l, &mut msg, &mut stream);
        }
        for &p in &beliefs {
            assert!((p - 1.0 / k as f32).abs() < 1e-5, "symmetry broke: {p}");
        }
        let mut mean = [0.0f32; 4];
        qsg_mean_into(&mut mean, &beliefs, n, k);
        assert!((qsg_polarization_u(&mean) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn uncertainty_and_conviction_read_the_geometry() {
        // 1 − ‖x‖²: zero at the vertex, 1−1/K at the center (the paper's
        // Fig. 5c drift-strength geometry).
        let vertex = [1.0f32, 0.0, 0.0];
        let center = [1.0f32 / 3.0; 3];
        assert!(qsg_uncertainty(&vertex).abs() < 1e-7);
        assert!((qsg_uncertainty(&center) - (1.0 - 1.0 / 3.0)).abs() < 1e-7);
    }
}
