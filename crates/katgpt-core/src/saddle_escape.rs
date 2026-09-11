//! Saddle-Trap Escape Gate — three-way loop control for latent reasoners
//! (Plan 593, Research 546, arXiv:2609.04963 "Fractal basins trap latent
//! reasoning", Lai/Bao/Quinn/Gilpin 2026).
//!
//! The paper's diagnosis: looped latent reasoners (EqR/HRM, FPRM, Parcae,
//! TRM) are dynamical systems whose initial-latent basin boundaries are
//! **fractal**; reasoning slowdowns are transient chaos — trajectories
//! scattering off weakly-unstable **saddles that decode to nearly-correct
//! answers**. Two cheap O(1) observables carry the physics:
//!
//! 1. **Oscillation streak** — reversal of update direction (cos θ < 0),
//!    already tracked by [`crate::gain_cost_halt::GainCostLoopHalter`].
//! 2. **Decode-flip rate** — decode every loop and hash the answer; the EMA
//!    of "key changed vs previous loop" is the paper's solution-switch
//!    frequency, which correlates with the fast Lyapunov indicator λF.
//!
//! Every halter in this stack is **two-way** (Continue/Halt): halt-on-
//! oscillation *gives up while holding a nearly-correct answer*. This gate
//! adds the missing third way — when both observables say "trapped on a
//! saddle that decodes nearly-correct", it emits a **deterministic,
//! BLAKE3-seeded, budget-bounded escape kick** and resumes the loop instead
//! of committing the near-miss. Budget exhaustion halts honestly with
//! `HaltOutcome::Trapped` (a verdict the two-way family cannot express).
//!
//! # Composition over fork
//!
//! [`SaddleEscapeGate`] **owns and wraps** a [`GainCostLoopHalter`] — the
//! halter's decision logic, NaN contract, and gain baselines are untouched.
//! The gate intercepts ONLY `HaltReason::Oscillation` outcomes (the single
//! trap-shaped halt reason); `GainBelowCost` / `NonContraction` halts pass
//! through unchanged as [`HaltOutcome::Converged`]. Game-runtime wiring
//! guidance: riir-ai Research 374 (think-brain only; nothing here crosses
//! a sync boundary).
//!
//! # Determinism contract (G5)
//!
//! - The kick direction seed is `BLAKE3(state_bytes ‖ loop_idx ‖ kick_no)`.
//! - [`apply_kick`] expands the seed through BLAKE3 XOF and normalizes with
//!   a scalar sequential loop — every f32 op is IEEE-deterministic in order,
//!   so the perturbation is bit-reproducible on every platform (no SIMD
//!   reassociation, no libm `powi`).
//! - The eps schedule is an explicit multiplication chain, not `powi`.
//!
//! # NaN contract (mirrors the halter)
//!
//! NaN cos θ is non-oscillatory (inherited); a NaN probe drift never
//! confirms a trap; the flip-rate EMA is structurally NaN-free (EMA of 0/1
//! indicators with finite α); **NaN never fires a kick**. A non-finite eps
//! schedule result degrades to `Halt{Trapped}` rather than poisoning state.
//!
//! # Latent vs Raw
//!
//! Everything here is local latent (per-loop trajectory observables). The
//! kick perturbs the **think brain only**. If surfaced across a sync
//! boundary at all, Trapped/Converged exit as scalars (flip-rate EMA,
//! kicks-used) — bridge-compatible; never a latent vector on the wire.
//!
//! # Opt-in
//!
//! Feature `saddle_escape` (implies `gain_cost_halt`). Quality claims are
//! gated on the Phase-3 defend-wrong PoC (`tests/saddle_escape_poc.rs`:
//! gate vs halt-only vs always-on-noise on a saddle-trap toy + a decode-
//! keyed flip-ring toy); promotion to default is owner-gated.

use crate::gain_cost_halt::{GainCostLoopHalter, HaltDecision, HaltReason};

// ─────────────────────────────────────────────────────────────────────
// Configuration (T1.1)
// ─────────────────────────────────────────────────────────────────────

/// Configuration for [`SaddleEscapeGate`].
///
/// Defaults are calibration starting points validated on the Phase-3 toy
/// PoC (`tests/saddle_escape_poc.rs`); callers retune per domain scale:
/// `eps0` is an ABSOLUTE perturbation magnitude (the kick adds `eps * u`
/// with `u` a unit vector) — size it against your state norm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrapConfig {
    /// Decode-flip-rate EMA threshold: an oscillation halt is classified
    /// TRAPPED only when the EMA is at least this. Below it (or when the
    /// EMA is still warming up / unknown), the halt passes through as
    /// `Converged` — no trap evidence, no kick.
    pub flip_tau: f32,
    /// Maximum escape kicks per episode. `0` disarms kicking (every
    /// confirmed trap halts `Trapped` immediately).
    pub kick_budget: u8,
    /// First kick magnitude (absolute; see struct doc).
    pub eps0: f32,
    /// Geometric decay per kick: kick k applies `eps0 * eps_decay^k`.
    /// Default `0.5` (paper-consistent annealing: later kicks gentler).
    pub eps_decay: f32,
    /// Flip-EMA window: α = 2/(window+1), and `flip_rate()` stays `None`
    /// until `window` key-pair observations exist (hysteresis — a single
    /// flip never classifies a trap). Clamped to ≥ 1.
    pub window: u16,
    /// Probe-drift confirmation threshold (renoise-CE-shaped, caller-
    /// computed): when a probe drift is supplied it must be ≥ this to
    /// confirm a trap (high drift = unstable/saddle-near; Bench 406).
    pub probe_tau: f32,
}

impl TrapConfig {
    /// Calibration default: `flip_tau 0.5`, `kick_budget 2`, `eps0 0.1`,
    /// `eps_decay 0.5`, `window 4`, `probe_tau 0.1`. Retune `eps0` to the
    /// caller's state scale (see struct doc).
    pub const DEFAULT: Self = Self {
        flip_tau: 0.5,
        kick_budget: 2,
        eps0: 0.1,
        eps_decay: 0.5,
        window: 4,
        probe_tau: 0.1,
    };
}

impl Default for TrapConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ─────────────────────────────────────────────────────────────────────
// Observables (T1.1)
// ─────────────────────────────────────────────────────────────────────

/// Per-loop trajectory observables fed to [`SaddleEscapeGate::decide`].
///
/// `gain`/`cost`/`cos_theta` are exactly the wrapped halter's inputs (the
/// canonical wiring maps the loop's step norm onto `cost`; `step_norm` is
/// carried separately for caller telemetry — the gate itself does not read
/// it). `state_bytes` is the latent state's byte view, used ONLY to seed
/// kick directions (never stored).
pub struct TrapObservables<'a> {
    /// 1-based index of the loop just completed (halter convention).
    pub loop_idx: usize,
    /// Marginal refinement gain (halter input).
    pub gain: f32,
    /// Marginal drift cost (halter input; typically the step norm).
    pub cost: f32,
    /// Alignment of the last two update directions in `[-1, 1]` (halter
    /// input; negative = reversal).
    pub cos_theta: f32,
    /// Step norm `‖Δh‖` (telemetry; the gate passes `cost` to the halter).
    pub step_norm: f32,
    /// Hash of this loop's decoded answer (`None` = caller could not
    /// decode — a MISSING observation, never "no flip").
    pub decoded_key: Option<u64>,
    /// Optional one-step probe drift (renoise-CE-shaped). `None` = not
    /// supplied (the gate must be correct without it); `Some(NaN)` never
    /// confirms a trap.
    pub probe_drift: Option<f32>,
    /// Byte view of the latent state, consumed only when a kick is emitted.
    pub state_bytes: &'a [u8],
}

// ─────────────────────────────────────────────────────────────────────
// Flip-rate EMA ring (T1.2)
// ─────────────────────────────────────────────────────────────────────

/// Decode-keyed flip-rate EMA — the paper's solution-switch frequency.
///
/// O(1) per observation, fixed-size state, zero allocation. The rate is the
/// EMA of the indicator `key != prev_key` with α = 2/(window+1); it reads
/// `None` until `window` key-pair observations exist (hysteresis) and stays
/// `None` forever if keys are never supplied. Structurally NaN-free.
#[derive(Clone, Copy, Debug)]
pub struct FlipDetector {
    prev_key: Option<u64>,
    ema: f32,
    pairs: u16,
    alpha: f32,
    warmup: u16,
}

impl FlipDetector {
    /// New detector over a config window (clamped to ≥ 1).
    pub fn new(window: u16) -> Self {
        let warmup = window.max(1);
        Self {
            prev_key: None,
            ema: 0.0,
            pairs: 0,
            // 2/(w+1); w ≥ 1 → α ∈ (0, 1]. Plain arithmetic, no libm.
            alpha: 2.0 / (warmup as f32 + 1.0),
            warmup,
        }
    }

    /// Feed one loop's decoded key. `None` is a missing observation: no EMA
    /// update, no pair counted, `prev_key` retained so the next successful
    /// decode compares across the gap. Returns the post-observation rate.
    pub fn observe(&mut self, key: Option<u64>) -> Option<f32> {
        if let Some(k) = key {
            if let Some(pk) = self.prev_key {
                let indicator = if k != pk { 1.0 } else { 0.0 };
                self.ema += self.alpha * (indicator - self.ema);
                self.pairs = self.pairs.saturating_add(1);
            }
            self.prev_key = Some(k);
        }
        self.flip_rate()
    }

    /// Current flip-rate EMA, or `None` while warming up / no keys seen.
    /// "NaN flip rate = no trap" is expressed as `None` here — the gate
    /// treats unknown rate as NO trap evidence (pass-through halt).
    pub fn flip_rate(&self) -> Option<f32> {
        (self.pairs >= self.warmup).then_some(self.ema)
    }

    /// Pair observations seen so far (telemetry).
    pub fn observations(&self) -> u16 {
        self.pairs
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate decision types (T2.1)
// ─────────────────────────────────────────────────────────────────────

/// Three-way (+ kick) loop decision emitted by [`SaddleEscapeGate::decide`].
///
/// Caller mapping:
/// - [`Continue`](Self::Continue) → keep looping (the halter's
///   `RefusedFloor` folds in here — both mean "keep looping"; consult
///   [`SaddleEscapeGate::halter`] if the distinction matters).
/// - [`Halt`](Self::Halt) → exit the loop; the outcome says whether the
///   halt was a clean pass-through or an exhausted trap.
/// - [`Kick`](Self::Kick) → apply the perturbation with
///   [`apply_kick`], then RESUME looping and keep feeding observables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateDecision {
    /// Keep looping.
    Continue,
    /// Exit the loop. NOTE: `Converged` here means "the wrapped halter's
    /// normal halt path" — halt ≠ classification (the halter's own
    /// warning applies); consult the carried [`HaltReason`].
    Halt(HaltOutcome),
    /// Bounded deterministic escape: perturb the latent state by
    /// `eps * unit(BLAKE3(seed))` via [`apply_kick`], then resume looping.
    Kick {
        /// BLAKE3 direction seed (`state_bytes ‖ loop_idx ‖ kick_no`).
        dir_seed: [u8; 32],
        /// Kick magnitude `eps0 * eps_decay^kicks_used_so_far`.
        eps: f32,
    },
}

/// Why the gate halted. The two-way halter family cannot express the
/// second variant — that expressiveness is this module's point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HaltOutcome {
    /// The wrapped halter halted for non-trap reasons (oscillation WITHOUT
    /// flip-rate confirmation, gain exhaustion, or non-contraction). NOT a
    /// verdict that the answer is good — halt ≠ classification.
    Converged(HaltReason),
    /// Oscillation WITH flip-rate (and probe, if supplied) confirmation,
    /// kick budget exhausted — the loop was circling a saddle that decodes
    /// nearly-correct and could not escape. The caller should treat the
    /// decoded answer as a flagged near-miss (downshift / re-ask / damp),
    /// not as a clean answer.
    Trapped {
        /// The flip-rate EMA that confirmed the trap.
        flip_rate: f32,
        /// Kicks emitted this episode before giving up.
        kicks_used: u8,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Kick mechanics (T2.2)
// ─────────────────────────────────────────────────────────────────────

/// Deterministic kick-direction seed: `BLAKE3(state_bytes ‖ loop_idx ‖
/// kick_no)`. Distinct per loop AND per kick number.
fn kick_seed(state_bytes: &[u8], loop_idx: usize, kick_no: u8) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(state_bytes);
    h.update(&(loop_idx as u64).to_le_bytes());
    h.update(&[kick_no]);
    *h.finalize().as_bytes()
}

/// Invoke `f(index, u)` for each of the `d` direction components derived
/// from the seed via BLAKE3 XOF. Scalar sequential — the byte→f32 mapping
/// uses only exactly-representable constants, so the stream is
/// bit-identical across platforms.
fn each_direction(seed: &[u8; 32], d: usize, mut f: impl FnMut(usize, f32)) {
    let mut xof = blake3::Hasher::new();
    xof.update(seed);
    let mut reader = xof.finalize_xof();
    // 1024 f32 per fill; kicks are rare (≤ kick_budget/episode) so the
    // buffer size is a non-issue. 4 KiB stays well under stack guards.
    let mut chunk = [0u8; 4096];
    let mut idx = 0usize;
    while idx < d {
        reader.fill(&mut chunk);
        for b in chunk.as_chunks::<4>().0 {
            let x = u32::from_le_bytes(*b);
            // 24-bit uniform → [-1, 1): exact f32 arithmetic (integer ≤ 2^24
            // is exact; division by a power of two is exact).
            let u = ((x >> 8) as f32) * (1.0 / 8388608.0) - 1.0;
            f(idx, u);
            idx += 1;
            if idx == d {
                break;
            }
        }
    }
}

/// Apply a gate-emitted kick in place: `state[i] += eps * u[i]` with `u`
/// the unit-norm direction derived deterministically from the seed.
///
/// Bit-reproducible: two XOF passes over the same seed produce the same
/// component stream; normalization is a scalar sequential loop (IEEE ops
/// in order — no SIMD reassociation). Degenerate guards (both
/// deterministic): empty slice or `eps == 0` are exact no-ops; a zero /
/// non-finite norm (all-zero draw, caller NaN state) falls back to kicking
/// the first component by `eps`.
pub fn apply_kick(state: &mut [f32], seed: [u8; 32], eps: f32) {
    if state.is_empty() || eps == 0.0 {
        return;
    }
    // Pass 1: direction norm only.
    let mut norm_sq = 0.0f32;
    each_direction(&seed, state.len(), |_, u| norm_sq += u * u);
    let norm = norm_sq.sqrt();
    if !(norm > 0.0 && norm.is_finite()) {
        state[0] += eps;
        return;
    }
    // Pass 2: re-derive the identical stream and apply.
    each_direction(&seed, state.len(), |i, u| state[i] += eps * u / norm);
}

// ─────────────────────────────────────────────────────────────────────
// The gate (T1.3 / T2.3)
// ─────────────────────────────────────────────────────────────────────

/// Three-way loop controller wrapping a [`GainCostLoopHalter`] (Plan 593).
///
/// Per loop, feed [`TrapObservables`] to [`Self::decide`]:
///
/// 1. The flip detector observes the decoded key FIRST (the halting loop's
///    decode still counts as evidence).
/// 2. The wrapped halter runs unchanged (streaks, gain baselines, floors).
/// 3. `Halt{Oscillation}` is intercepted for trap classification:
///    flip-rate EMA ≥ `flip_tau` AND (if supplied) probe drift ≥
///    `probe_tau` ⇒ trap. Trap + budget remaining ⇒ [`GateDecision::Kick`]
///    (the halter's oscillation streak resets so the resumed loop starts
///    clean; gain baselines and the flip EMA persist — the EMA is a
///    property of the attractor region, not the exact orbit). Trap +
///    exhausted budget ⇒ [`HaltOutcome::Trapped`]. No trap evidence ⇒
///    pass-through [`HaltOutcome::Converged`].
/// 4. All other halter outcomes map 1:1 (`Continue`/`RefusedFloor` →
///    `Continue`; other halts → `Converged(reason)`).
///
/// Construct a NEW gate per episode (~90 bytes; the wrapped halter's
/// episode semantics assume fresh state). The eps schedule degrades to
/// `Trapped` (never a NaN-eps kick) if the config produces non-finite eps.
#[derive(Clone, Debug)]
pub struct SaddleEscapeGate {
    halter: GainCostLoopHalter,
    cfg: TrapConfig,
    flip: FlipDetector,
    kicks_used: u8,
}

impl SaddleEscapeGate {
    /// Wrap an externally-configured halter (composition entry point).
    pub fn wrap(halter: GainCostLoopHalter, config: TrapConfig) -> Self {
        Self {
            halter,
            flip: FlipDetector::new(config.window),
            cfg: config,
            kicks_used: 0,
        }
    }

    /// Default halter + explicit trap config.
    pub fn new(config: TrapConfig) -> Self {
        Self::wrap(GainCostLoopHalter::default(), config)
    }

    /// One loop's three-way decision. See the struct doc for the pipeline.
    pub fn decide(&mut self, obs: TrapObservables<'_>) -> GateDecision {
        // 1. Decode evidence first — the halting loop still counts.
        self.flip.observe(obs.decoded_key);

        // 2. Wrapped halter, untouched.
        let hd = self
            .halter
            .halt_decision(obs.loop_idx, obs.gain, obs.cost, obs.cos_theta);

        let HaltDecision::Halt { reason } = hd else {
            // Continue / RefusedFloor both mean "keep looping".
            return GateDecision::Continue;
        };

        // 3. Trap classification — only the oscillation halt is trap-shaped.
        if reason != HaltReason::Oscillation {
            return GateDecision::Halt(HaltOutcome::Converged(reason));
        }
        let Some(rate) = self.flip.flip_rate() else {
            // Unknown flip rate (warming up / never decoded) = no trap
            // evidence — honest pass-through, never a blind kick.
            return GateDecision::Halt(HaltOutcome::Converged(reason));
        };
        let rate_confirms = rate >= self.cfg.flip_tau; // rate is NaN-free
        let probe_confirms = match obs.probe_drift {
            None => true,                    // correct without the probe
            Some(d) => d >= self.cfg.probe_tau, // NaN drift never confirms
        };
        if !(rate_confirms && probe_confirms) {
            return GateDecision::Halt(HaltOutcome::Converged(reason));
        }

        // Confirmed trap: kick while budget remains, else Trapped.
        if self.kicks_used >= self.cfg.kick_budget {
            return GateDecision::Halt(HaltOutcome::Trapped {
                flip_rate: rate,
                kicks_used: self.kicks_used,
            });
        }
        // eps schedule: explicit multiplication chain (deterministic; no
        // libm powi). Kick #1 (kicks_used == 0) applies eps0.
        let mut eps = self.cfg.eps0;
        for _ in 0..self.kicks_used {
            eps *= self.cfg.eps_decay;
        }
        if !eps.is_finite() {
            // NaN/inf eps must never poison the state (NaN never fires a
            // kick) — degrade to the honest trapped halt.
            return GateDecision::Halt(HaltOutcome::Trapped {
                flip_rate: rate,
                kicks_used: self.kicks_used,
            });
        }
        self.kicks_used += 1;
        // Resume cleanly: the streak that fired the halt is spent; gain
        // baselines persist (the halter's own episode semantics).
        self.halter.oscillation_count = 0;
        let dir_seed = kick_seed(obs.state_bytes, obs.loop_idx, self.kicks_used);
        GateDecision::Kick { dir_seed, eps }
    }

    /// Read-only view of the wrapped halter (telemetry / prev_step wiring).
    pub fn halter(&self) -> &GainCostLoopHalter {
        &self.halter
    }

    /// Mutable view for the forward-path setters (`update_prev_step` /
    /// `update_prev_erank`) — the same wiring the bare halter exposes.
    pub fn halter_mut(&mut self) -> &mut GainCostLoopHalter {
        &mut self.halter
    }

    /// Kicks emitted so far this episode.
    pub fn kicks_used(&self) -> u8 {
        self.kicks_used
    }

    /// Current decode-flip-rate EMA (`None` while warming up).
    pub fn flip_rate(&self) -> Option<f32> {
        self.flip.flip_rate()
    }

    /// Active config (telemetry).
    pub fn config(&self) -> &TrapConfig {
        &self.cfg
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests (T1.4 / T2.5)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obs<'a>(
        loop_idx: usize,
        gain: f32,
        cost: f32,
        cos_theta: f32,
        key: Option<u64>,
        state: &'a [u8],
    ) -> TrapObservables<'a> {
        TrapObservables {
            loop_idx,
            gain,
            cost,
            cos_theta,
            step_norm: cost,
            decoded_key: key,
            probe_drift: None,
            state_bytes: state,
        }
    }

    fn trapped_cfg() -> TrapConfig {
        TrapConfig {
            flip_tau: 0.5,
            kick_budget: 2,
            eps0: 0.1,
            eps_decay: 0.5,
            window: 2,
            probe_tau: 0.1,
        }
    }

    // ── FlipDetector EMA math ─────────────────────────────────────────

    #[test]
    fn flip_ema_alternating_converges_high() {
        let mut f = FlipDetector::new(4); // α = 0.4
        let keys = [1u64, 2, 1, 2, 1, 2];
        let mut last = None;
        for k in keys {
            last = f.observe(Some(k));
        }
        let r = last.expect("rate ready after 5 pairs (warmup 4)");
        // After 5 flips: EMA = 1 − (1−α)^5 = 1 − 0.6^5 = 0.92224
        assert!((r - 0.922224).abs() < 1e-4, "ema={r}");
    }

    #[test]
    fn flip_ema_constant_decays_to_zero() {
        let mut f = FlipDetector::new(2); // α = 2/3
        // First pair: different (1 → 2), rest identical.
        f.observe(Some(1));
        f.observe(Some(2)); // pair, indicator 1
        for _ in 0..12 {
            f.observe(Some(2));
        }
        let r = f.flip_rate().unwrap();
        assert!(r < 1e-3, "ema should decay to 0, got {r}");
    }

    #[test]
    fn flip_ema_exact_first_values() {
        // Window 2 (α = 2/3): rate reads None below warmup, then exact EMA.
        let mut f = FlipDetector::new(2);
        assert_eq!(f.observe(Some(7)), None); // first key: no pair yet
        assert_eq!(f.observe(Some(8)), None); // pair 1 < warmup → None
        let r2 = f.observe(Some(9)).expect("pair 2 == warmup → Some");
        // ema = 2/3 + (2/3)·(1 − 2/3) = 8/9
        assert!((r2 - 8.0 / 9.0).abs() < 1e-6, "{r2}");
        let r3 = f.observe(Some(10)).unwrap();
        // ema = 8/9 + (2/3)·(1/9) = 26/27
        assert!((r3 - 26.0 / 27.0).abs() < 1e-6, "{r3}");
        // Warmup boundary is exact: warmup 5 stays None through pair 4,
        // turns Some at pair 5.
        let mut g = FlipDetector::new(5);
        g.observe(Some(1));
        for i in 2..=6u16 {
            let r = g.observe(Some(i as u64));
            if i <= 5 {
                assert_eq!(r, None, "pair {i} below warmup 5 must read None");
            } else {
                assert!(r.is_some(), "pair {i} at warmup must read Some");
            }
        }
    }

    #[test]
    fn flip_none_key_is_missing_not_noflip() {
        let mut f = FlipDetector::new(1); // α = 1: every pair sets ema directly
        f.observe(Some(1));
        f.observe(Some(1)); // pair, no flip
        // Gap: None key must NOT count and must NOT clear prev_key.
        assert_eq!(f.observe(None), Some(0.0));
        assert_eq!(f.observations(), 1);
        // Cross-gap comparison: 1 → 2 IS a flip against the retained 1.
        f.observe(Some(2));
        let r = f.flip_rate().unwrap();
        assert!((r - 1.0).abs() < 1e-6, "cross-gap flip must register, got {r}");
    }

    #[test]
    fn flip_never_supplied_stays_none() {
        let mut f = FlipDetector::new(1);
        assert_eq!(f.observe(None), None);
        assert_eq!(f.flip_rate(), None);
    }

    // ── Gate: kick lifecycle ──────────────────────────────────────────

    fn run_trap_sequence(gate: &mut SaddleEscapeGate, loops: usize, keys: [u64; 2]) -> Vec<GateDecision> {
        let state = [1u8, 2, 3, 4];
        let mut out = Vec::new();
        for i in 1..=loops {
            // Reversal every loop (cos θ = -1), keys alternate → trap shape.
            let key = if i % 2 == 1 { keys[0] } else { keys[1] };
            out.push(gate.decide(obs(i, 1.0, 0.01, -1.0, Some(key), &state)));
        }
        out
    }

    #[test]
    fn trap_fires_kick_then_trapped_after_budget() {
        let mut g = SaddleEscapeGate::wrap(
            GainCostLoopHalter::new(1.0, 2, 1),
            trapped_cfg(),
        );
        // Halter: patience 2, l_min 1. Flip: window 2. Keys alternate, cos = -1.
        // Loop 2: oscillation fires but pairs=1 < warmup → Converged(Osc).
        // Loop 3: rate 0.889 ≥ τ → KICK #1 (eps0); streak resets.
        // Loop 4: streak 1 < patience → Continue.
        // Loop 5: streak 2 → KICK #2 (eps0·decay).
        // Loop 6: Continue. Loop 7: budget exhausted → Trapped.
        let ds = run_trap_sequence(&mut g, 8, [10, 20]);
        assert!(matches!(ds[1], GateDecision::Halt(HaltOutcome::Converged(
            HaltReason::Oscillation
        ))));
        match ds[2] {
            GateDecision::Kick { eps, .. } => assert!((eps - 0.1).abs() < 1e-7),
            other => panic!("expected kick at loop 3, got {other:?}"),
        }
        assert_eq!(ds[3], GateDecision::Continue, "post-kick streak rebuild");
        match ds[4] {
            GateDecision::Kick { eps, .. } => {
                assert!((eps - 0.05).abs() < 1e-7, "eps schedule: {eps}")
            }
            other => panic!("expected kick at loop 5, got {other:?}"),
        }
        assert_eq!(ds[5], GateDecision::Continue);
        match ds[6] {
            GateDecision::Halt(HaltOutcome::Trapped { flip_rate, kicks_used }) => {
                assert_eq!(kicks_used, 2);
                assert!(flip_rate >= 0.5);
            }
            other => panic!("expected trapped at loop 7, got {other:?}"),
        }
        assert_eq!(g.kicks_used(), 2);
    }

    #[test]
    fn kick_budget_zero_disarms_to_trapped() {
        let mut cfg = trapped_cfg();
        cfg.kick_budget = 0;
        let mut g = SaddleEscapeGate::wrap(GainCostLoopHalter::new(1.0, 2, 1), cfg);
        let ds = run_trap_sequence(&mut g, 4, [1, 2]);
        // Loop 3 (first rate-ready oscillation halt): budget 0 → Trapped.
        assert!(matches!(ds[2], GateDecision::Halt(HaltOutcome::Trapped { .. })));
    }

    #[test]
    fn oscillation_with_low_flip_passes_through_converged() {
        let mut g = SaddleEscapeGate::wrap(
            GainCostLoopHalter::new(1.0, 2, 1),
            trapped_cfg(),
        );
        // Constant key: flip rate present but ~0 → no trap evidence.
        let ds = run_trap_sequence(&mut g, 4, [7, 7]);
        assert!(matches!(ds[3], GateDecision::Halt(HaltOutcome::Converged(
            HaltReason::Oscillation
        ))));
    }

    #[test]
    fn gain_below_cost_halt_is_pass_through() {
        let mut g = SaddleEscapeGate::new(trapped_cfg());
        let state = [0u8; 8];
        // Alternating keys (would-be trap) but the halt reason is scissors.
        let d = g.decide(obs(4, 0.01, 1.0, 0.9, Some(1), &state));
        assert!(matches!(d, GateDecision::Halt(HaltOutcome::Converged(
            HaltReason::GainBelowCost
        ))));
    }

    #[test]
    fn refused_floor_maps_to_continue() {
        let mut g = SaddleEscapeGate::wrap(
            GainCostLoopHalter::new(1.0, 1, 5), // l_min 5
            trapped_cfg(),
        );
        let state = [0u8; 8];
        let d = g.decide(obs(1, 0.0, 1.0, -1.0, Some(1), &state));
        assert_eq!(d, GateDecision::Continue);
    }

    #[test]
    fn nan_signals_never_fire_a_kick() {
        let mut g = SaddleEscapeGate::wrap(
            GainCostLoopHalter::new(1.0, 1, 1),
            trapped_cfg(),
        );
        let state = [0u8; 8];
        // NaN cos θ: non-oscillatory (inherited) → Continue.
        let d = g.decide(obs(1, 1.0, 0.01, f32::NAN, Some(1), &state));
        assert_eq!(d, GateDecision::Continue);
        // NaN probe drift: trap shape but the probe never confirms.
        for i in 1..=4u64 {
            let key = Some(100 + (i % 2));
            let mut o = obs(i as usize, 1.0, 0.01, -1.0, key, &state);
            o.probe_drift = Some(f32::NAN);
            let d = g.decide(o);
            assert!(
                !matches!(d, GateDecision::Kick { .. }),
                "NaN probe must never kick (loop {i})"
            );
        }
        // NaN gain: never halts (inherited) → Continue.
        let d = g.decide(obs(5, f32::NAN, 1.0, 0.5, Some(104), &state));
        assert_eq!(d, GateDecision::Continue);
    }

    #[test]
    fn probe_confirm_gate_changes_verdict() {
        // Same trap shape; probe below τ → Converged; above → Kick, then
        // Trapped once the budget spends.
        let state = [9u8; 8];
        let mut g = SaddleEscapeGate::wrap(
            GainCostLoopHalter::new(1.0, 1, 1),
            trapped_cfg(),
        );
        // Patience 1: every loop halts on oscillation. Window 2: rate is
        // None through loop 2, ready (high) from loop 3.
        for i in 1..=3u64 {
            let mut o = obs(i as usize, 1.0, 0.01, -1.0, Some(i % 2 + 5), &state);
            o.probe_drift = Some(0.01); // < probe_tau 0.1 → never confirms
            let d = g.decide(o);
            assert!(
                matches!(d, GateDecision::Halt(HaltOutcome::Converged(
                    HaltReason::Oscillation
                ))),
                "loop {i}: low probe must pass through, got {d:?}"
            );
        }
        // Probe confirms from here. Note loop 4's rate is high AND the
        // streak never reset (all Converged halts so far).
        let d = {
            let mut o = obs(4, 1.0, 0.01, -1.0, Some(5), &state);
            o.probe_drift = Some(0.5);
            g.decide(o)
        };
        assert!(matches!(d, GateDecision::Kick { .. }), "loop 4: {d:?}");
        // Loop 5: kick #2 (streak rebuilt 1 loop = patience 1).
        let d = {
            let mut o = obs(5, 1.0, 0.01, -1.0, Some(6), &state);
            o.probe_drift = Some(0.5);
            g.decide(o)
        };
        assert!(matches!(d, GateDecision::Kick { .. }), "loop 5: {d:?}");
        // Loop 6: budget (2) exhausted → Trapped.
        let d = {
            let mut o = obs(6, 1.0, 0.01, -1.0, Some(5), &state);
            o.probe_drift = Some(0.5);
            g.decide(o)
        };
        assert!(matches!(d, GateDecision::Halt(HaltOutcome::Trapped { .. })));
    }

    // ── Determinism (G5) ──────────────────────────────────────────────

    #[test]
    fn kick_seed_is_deterministic_and_distinct() {
        let s = [3u8; 16];
        let a = kick_seed(&s, 7, 1);
        let b = kick_seed(&s, 7, 1);
        assert_eq!(a, b, "same inputs → identical seed");
        assert_ne!(kick_seed(&s, 7, 2), a, "kick_no separates");
        assert_ne!(kick_seed(&s, 8, 1), a, "loop_idx separates");
        assert_ne!(kick_seed(&[4u8; 16], 7, 1), a, "state separates");
    }

    #[test]
    fn full_episode_bit_reproducible() {
        let run = || {
            let mut g = SaddleEscapeGate::wrap(
                GainCostLoopHalter::new(1.0, 2, 1),
                trapped_cfg(),
            );
            let state = [5u8; 12];
            let mut out = Vec::new();
            for i in 1..=7u64 {
                let key = Some(3 + (i % 2));
                out.push(g.decide(obs(i as usize, 1.0, 0.01, -1.0, key, &state)));
            }
            out
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "identical episodes → identical decision streams");
        assert!(a.iter().any(|d| matches!(d, GateDecision::Kick { .. })));
    }

    // ── apply_kick ────────────────────────────────────────────────────

    #[test]
    fn apply_kick_displaces_by_exactly_eps() {
        let mut x = [0.0f32; 64];
        let seed = [7u8; 32];
        apply_kick(&mut x, seed, 0.25);
        // ‖x′ − 0‖ = |eps| exactly (unit direction × eps).
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 0.25).abs() < 1e-5, "norm={norm}");
        // And it IS a kick (not all zero).
        assert!(x.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn apply_kick_bit_reproducible() {
        let mut a = [1.0f32; 33];
        let mut b = [1.0f32; 33];
        let seed = [42u8; 32];
        apply_kick(&mut a, seed, 0.3);
        apply_kick(&mut b, seed, 0.3);
        assert_eq!(a, b);
        // Non-trivial d (not a multiple of the 1024-fill boundary).
        assert!(a.iter().any(|v| *v != 1.0));
    }

    #[test]
    fn apply_kick_zero_eps_and_empty_are_exact_noops() {
        let mut x = [-0.0f32, 1.5];
        apply_kick(&mut x, [1u8; 32], 0.0);
        assert_eq!(x[0].to_bits(), (-0.0f32).to_bits(), "eps=0 preserves -0.0");
        let mut y: [f32; 0] = [];
        apply_kick(&mut y, [1u8; 32], 1.0); // must not panic
    }

    #[test]
    fn apply_kick_streak_reset_lets_loop_continue() {
        let mut g = SaddleEscapeGate::wrap(
            GainCostLoopHalter::new(1.0, 2, 1),
            trapped_cfg(),
        );
        let state = [0u8; 4];
        // Build to the kick.
        let mut kicked = false;
        for i in 1..=3u64 {
            let key = Some(1 + (i % 2));
            if let GateDecision::Kick { .. } =
                g.decide(obs(i as usize, 1.0, 0.01, -1.0, key, &state))
            {
                kicked = true;
            }
        }
        assert!(kicked);
        // Post-kick aligned loop with a STABLE key: must Continue (the
        // spent streak was reset by the kick; the flip EMA is not a halt
        // condition by itself).
        let d = g.decide(obs(4, 1.0, 0.01, 0.9, Some(9), &state));
        assert_eq!(d, GateDecision::Continue);
    }
}
