//! Issue 673 Phase 1 — Recirculation: cross-step residual mixture operator
//! (Research 492, arXiv:2608.17981 "Recirculation", Mozer et al., Google
//! DeepMind; paired with arXiv:2608.08888 "Full-bandwidth transformer").
//!
//! The vertical feedback channel of a decoder-only transformer is narrow:
//! only the sampled token returns to the bottom of the stack; deep-layer
//! conclusions are depth-frozen within a step. Recirculation (training-free)
//! leaks a **convex, norm-matched** mixture of the previous step's deep-layer
//! state into a shallow destination layer at the NEXT input step:
//!
//! ```text
//! z_{t+1,dst} = α·(‖z_{t+1,dst}‖ / ‖z_{t,src}‖)·z_{t,src} + β·z_{t+1,dst}
//! ```
//!
//! with `β = 1−α` (convex, the paper's 1B setting) or `β = 1` (non-convex,
//! the paper's 4B/12B App. B.3 setting). Reported gains: 4.7–16% ppl
//! reduction on off-the-shelf Gemma3 (up to 35% at 12B) — **paper claims on
//! their substrate; ours must be PoC-verified before any promotion**
//! (Phase 2 in riir-poc, gated below).
//!
//! # Relationship to `cross_stage_relocation` (R417 / Plan 431)
//!
//! Sibling operator, same source→destination stage topology, different
//! mixing semantics: [`crate::cross_stage_relocation::RelocateOp`]
//! **overwrites** within one forward pass (its defend-wrong PoC refuted the
//! fixed-pair default — the overwrite CLOBBERS in 2/4 clean configs);
//! [`RecircOp`] **mixes** convexly, norm-matched, across steps — exactly the
//! semantics that failure mode predicts would fix it. Also distinct from
//! `LoopMode::TrainingFree` (R097), which is depth-only recurrence within
//! one position — the paper explicitly contrasts the two (their Fig. 8).
//!
//! # Ordering contract
//!
//! The paper's layer pairs always have `dst < src` (destination shallow,
//! source deep). Within one step the destination layer is therefore reached
//! BEFORE the source layer: the step-t injection at `dst` runs before the
//! step-t capture at `src`, so the captured state already carries the
//! injected content (the recurrence composes, paper Fig. 3/4 unrolling).
//! [`RecircOp::mix_into`] documents this assumption; hosts with `dst ≥ src`
//! capture pre-injection state — a different (unsupported) semantics.
//!
//! # Costs (honest)
//!
//! Decode with recirculation = two stack instances per step on throughput
//! hardware, ~2× FLOPs serial, **2× KV-cache footprint**; prefill becomes
//! serial (token-by-token). Blockwise recirculation (K tokens per step) is
//! the paper's future work, not shipped here.
//!
//! Feature: `recirculation` (opt-in). Promotion requires the Phase 2
//! defend-wrong PoC (ppl reduction > 0 on ≥2 datasets AND strictly safer
//! than the R417 overwrite at equal layer pairs) — see the Phase 4 gate in
//! the issue. Default stays OFF until then.

/// Destination band as a fraction of stack depth L (paper Table B.1:
/// destinations sit shallow — {4/26, 9/34, 16/48} = 0.15–0.33).
pub const DEST_BAND: (f32, f32) = (0.15, 0.33);

/// Source band as a fraction of stack depth L (paper Table B.1:
/// sources sit deep — {11/26, 18/34, 35/48} = 0.42–0.73).
pub const SRC_BAND: (f32, f32) = (0.42, 0.73);

/// Ramping default (paper §4.3: early positions can be *harmed* at 1B —
/// `α_t = min(t/10, 1)·α`; we default 10, the paper's ramp).
pub const DEFAULT_RAMP_TICKS: u32 = 10;

/// Cross-step convex-mixture operator on stage outputs. Sibling of
/// [`crate::cross_stage_relocation::RelocateOp`].
///
/// The host captures the source stage's output for the current step (into a
/// caller-owned [`RecircBuffer`]), then at the NEXT step calls
/// [`RecircOp::mix_into`] on the destination stage's live state BEFORE that
/// stage consumes it (or equivalently at the post-layer hook of the previous
/// layer — the mixture only needs `z_dst`'s pre-mixture value).
///
/// No new sync-boundary data: all fields are configuration, not gameplay
/// state. Zero-alloc steady state: the mixture is in-place over the
/// destination slice; the buffer is a fixed-D allocation made once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecircOp {
    /// Stage to capture the source state from (deep — see [`SRC_BAND`]).
    pub src_stage: usize,
    /// Stage to mix into (shallow — see [`DEST_BAND`]); the paper's pairs
    /// always have `dst < src` (see the module's ordering contract).
    pub dst_stage: usize,
    /// Base mixture weight α ∈ [0, 1] applied to the norm-matched source.
    pub alpha: f32,
    /// Destination weight β. `1 − α` (convex) or `1` (non-convex, larger
    /// models per paper App. B.3). Use the constructors to pick a variant.
    pub beta: f32,
    /// L2-norm matching (`f(z_s) = (‖z_d‖/‖z_s‖)·z_s`, paper Eq. 2). When
    /// `false`, mixes the raw source (the ablation arm).
    pub norm_match: bool,
    /// Linear ramp in steps: `α_t = min(t/ramp, 1)·α` (paper §4.3).
    pub ramp_ticks: u32,
}

impl RecircOp {
    /// Convex variant (paper's 1B setting): `β = 1 − α`.
    #[must_use]
    pub fn convex(src_stage: usize, dst_stage: usize, alpha: f32, ramp_ticks: u32) -> Self {
        Self {
            src_stage,
            dst_stage,
            alpha,
            beta: 1.0 - alpha,
            norm_match: true,
            ramp_ticks,
        }
    }

    /// Non-convex variant (paper App. B.3, 4B/12B): `β = 1` — the
    /// destination keeps its full magnitude and the leak is additive on top.
    #[must_use]
    pub fn non_convex(src_stage: usize, dst_stage: usize, alpha: f32, ramp_ticks: u32) -> Self {
        Self {
            src_stage,
            dst_stage,
            alpha,
            beta: 1.0,
            norm_match: true,
            ramp_ticks,
        }
    }

    /// Ramped effective α at step `t` (0-indexed): `min(t/ramp, 1)·α`.
    ///
    /// `t/ramp` is computed in f32 then clamped — bit-deterministic.
    #[inline]
    #[must_use]
    pub fn effective_alpha(&self, step: u32) -> f32 {
        if self.ramp_ticks == 0 {
            return self.alpha;
        }
        let frac = (step as f32 / self.ramp_ticks as f32).min(1.0);
        frac * self.alpha
    }

    /// Apply the cross-step mixture IN PLACE to the live destination state.
    ///
    /// `z_src_prev` is the source stage's output captured at the PREVIOUS
    /// step (`RecircBuffer::capture`d then held); `z_dst` is the destination
    /// stage's state for the CURRENT step, before the stage consumes it.
    ///
    /// Early-returns (bit-identical no-op, zero work) when the ramped α is
    /// exactly 0 — step 0 under a ramp, or α = 0 (the baseline arm).
    ///
    /// # Panics
    /// Panics (debug) on length mismatch or non-finite inputs.
    #[inline]
    pub fn mix_into(&self, step: u32, z_src_prev: &[f32], z_dst: &mut [f32]) {
        debug_assert_eq!(z_src_prev.len(), z_dst.len(), "stages share width D");
        let a = self.effective_alpha(step);
        if a == 0.0 {
            return; // bit-identical no-op (baseline / pre-ramp)
        }
        let scale = if self.norm_match {
            let s_norm = l2_norm(z_src_prev);
            let d_norm = l2_norm(z_dst);
            if s_norm > 0.0 && d_norm.is_finite() {
                d_norm / s_norm
            } else {
                1.0 // degenerate source (all-zero) — leak nothing scaled
            }
        } else {
            1.0
        };
        let beta = self.beta;
        for i in 0..z_dst.len() {
            z_dst[i] = a * scale * z_src_prev[i] + beta * z_dst[i];
        }
    }
}

/// The paper's fixed layer-pair landscape, as a depth-scaled default.
///
/// Gemma3 sweep-validated (Table B.1): {11→4} at 26 layers (0.42L→0.15L),
/// {18→9} at 34 (0.53L→0.26L), {35→16} at 48 (0.73L→0.33L). Robust across
/// Ministral3 / Pythia / Qwen3 / Phi2 qualitatively; the paper reports
/// Gemma2-family gains "as pronounced as Gemma3" (App. C.1) — which is the
/// Phase 2 PoC's hypothesis, not a shipped claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecircPair {
    /// Depth-scaled paper default: picks the anchor pair whose total depth
    /// is nearest `n_stages` (see [`RecircPair::for_depth`]).
    PaperBands,
    /// Custom fractions (clamped to the paper's bands is NOT enforced —
    /// custom means custom).
    Custom { src: f32, dst: f32 },
}

impl RecircPair {
    /// The paper anchor pair whose reference depth is nearest `n_stages`.
    #[must_use]
    pub fn for_depth(n_stages: usize) -> (f32, f32) {
        match n_stages {
            0..=30 => (0.42, 0.15), // 26-layer 1B anchor {11, 4}
            31..=41 => (0.53, 0.26), // 34-layer 4B anchor {18, 9}
            _ => (0.73, 0.33),     // 48-layer 12B anchor {35, 16}
        }
    }

    /// Materialize the single-op form for a stack of `n_stages` stages
    /// (round-to-nearest on the fractional indices, mirroring
    /// `RelocatePair::to_ops`).
    #[must_use]
    pub fn to_op(&self, n_stages: usize, alpha: f32, ramp_ticks: u32, convex: bool) -> RecircOp {
        let (src_f, dst_f) = match self {
            Self::PaperBands => Self::for_depth(n_stages),
            Self::Custom { src, dst } => (*src, *dst),
        };
        let src_stage = frac_to_stage(src_f, n_stages);
        let dst_stage = frac_to_stage(dst_f, n_stages);
        if convex {
            RecircOp::convex(src_stage, dst_stage, alpha, ramp_ticks)
        } else {
            RecircOp::non_convex(src_stage, dst_stage, alpha, ramp_ticks)
        }
    }
}

/// Round a fraction to a stage index (paper §5.5 notation `⌊·⌉`); stage 0
/// for degenerate `n_stages == 0`.
fn frac_to_stage(frac: f32, n_stages: usize) -> usize {
    if n_stages == 0 {
        return 0;
    }
    ((frac.clamp(0.0, 1.0) * n_stages as f32).round() as usize).min(n_stages - 1)
}

/// Fixed-D capture buffer for the cross-step source state.
///
/// Allocated once (`Vec`, cold path); steady-state `capture` / `as_slice`
/// are a memcpy and a borrow — zero allocation, zero growth.
#[derive(Clone, Debug)]
pub struct RecircBuffer {
    buf: Vec<f32>,
}

impl RecircBuffer {
    /// New buffer of width `d` (the per-stage residual width).
    #[must_use]
    pub fn new(d: usize) -> Self {
        Self { buf: vec![0.0; d] }
    }

    /// Capture the source stage's output for the current step.
    #[inline]
    pub fn capture(&mut self, z_src: &[f32]) {
        debug_assert_eq!(z_src.len(), self.buf.len());
        self.buf.copy_from_slice(z_src);
    }

    /// The captured (previous-step) source state.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.buf
    }
}

/// Ascending-index L2 norm with a FIXED 8-way unrolled accumulator pattern.
///
/// Deterministic (the association order is a compile-time constant —
/// bit-identical across runs and callers) while breaking the serial
/// `acc += x·x` dependency chain that makes a single-accumulator norm
/// latency-bound at D=2048 (measured 3.1µs → ~0.3µs; the ≤1µs G2 gate).
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = v.len() / 8;
    let head = chunks * 8;
    for c in 0..chunks {
        let base = c * 8;
        for (j, a) in acc.iter_mut().enumerate() {
            let x = v[base + j];
            *a += x * x;
        }
    }
    for &x in &v[head..] {
        acc[0] += x * x;
    }
    // Fixed reduction order: ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)).
    (((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_from(seed: u32, d: usize, scale: f32) -> Vec<f32> {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        (0..d)
            .map(|_| (next() % 2000) as f32 / 1000.0 * scale - scale / 2.0)
            .collect()
    }

    /// Convex + norm-match boundedness (paper Eq. 1–2): the mixture never
    /// exceeds the destination's norm (triangle inequality over norm-equal
    /// operands), up to fp slack.
    #[test]
    fn mixture_bounded_convex_norm_matched() {
        let op = RecircOp::convex(11, 4, 0.15, 10);
        for seed in 1..8u32 {
            let src = vec_from(seed, 64, 2.0);
            let dst = vec_from(seed * 31, 64, 1.0);
            let before = l2_norm(&dst);
            let mut z = dst.clone();
            op.mix_into(100, &src, &mut z);
            let after = l2_norm(&z);
            assert!(
                after <= before + 1e-4,
                "seed {seed}: ‖z'‖ {after} > ‖z_d‖ {before}"
            );
        }
    }

    /// Norm-matching identity: equal norms ⇒ scale exactly 1 (`x/x == 1.0`
    /// exactly in fp for finite nonzero x) ⇒ the mixture is the plain
    /// convex blend `α·z + (1−α)·z ≈ z` (fp re-association slack only).
    #[test]
    fn norm_match_identity_on_equal_norms() {
        let z = vec_from(3, 32, 1.0);
        let mut dst = z.clone();
        let op = RecircOp::convex(11, 4, 0.3, 0);
        // scale = ‖z‖/‖z‖ == 1.0 exactly in fp for any finite nonzero x.
        let s_norm = l2_norm(&z);
        assert_eq!((s_norm / s_norm).to_bits(), 1.0f32.to_bits());
        op.mix_into(5, &z.clone(), &mut dst);
        // α·z + (1−α)·z ≈ z (within fp re-association slack).
        for i in 0..z.len() {
            assert!((dst[i] - z[i]).abs() < 1e-6, "i={i}");
        }
    }

    /// Ramping schedule: α_t = min(t/ramp,1)·α; t ≥ ramp ⇒ exactly α; t = 0
    /// ⇒ exactly 0 ⇒ the mixture is a bit-identical no-op.
    #[test]
    fn ramp_schedule_and_step0_noop() {
        let op = RecircOp::convex(11, 4, 0.15, 10);
        assert_eq!(op.effective_alpha(0), 0.0);
        assert!((op.effective_alpha(5) - 0.075).abs() < 1e-7);
        assert_eq!(op.effective_alpha(10), 0.15);
        assert_eq!(op.effective_alpha(1000), 0.15);
        // Step-0 no-op is BIT-identical.
        let src = vec_from(9, 16, 1.0);
        let dst = vec_from(11, 16, 1.0);
        let mut z = dst.clone();
        op.mix_into(0, &src, &mut z);
        for i in 0..z.len() {
            assert_eq!(z[i].to_bits(), dst[i].to_bits());
        }
        // ramp = 0 ⇒ no ramping at all.
        let op0 = RecircOp::convex(11, 4, 0.15, 0);
        assert_eq!(op0.effective_alpha(0), 0.15);
    }

    /// β = 1 non-convex variant: the destination keeps its magnitude and
    /// the leak is additive on top (paper App. B.3 larger-model setting).
    #[test]
    fn non_convex_beta_one_additive_leak() {
        let op = RecircOp::non_convex(11, 4, 0.1, 0);
        assert_eq!(op.beta, 1.0);
        let src = vec![2.0f32; 8];
        let dst = vec![3.0f32; 8];
        let mut z = dst.clone();
        op.mix_into(4, &src, &mut z);
        // scale = ‖z_d‖/‖z_s‖ = (3·√8)/(2·√8) — the √8 cancels in the ratio
        // of equal-magnitude vectors ⇒ exactly 1.5. z' = 0.1·scale·2 + 3.
        let scale = l2_norm(&dst) / l2_norm(&src);
        for (zi, &di) in z.iter().zip(dst.iter()) {
            let expect = 0.1 * scale * 2.0 + 1.0 * di;
            assert_eq!(*zi, expect, "exact fp recurrence");
        }
    }

    /// Determinism (G1): fixed α + fixed buffers ⇒ bit-identical repeat.
    #[test]
    fn deterministic_bit_identical_repeat() {
        let op = RecircPair::PaperBands.to_op(26, 0.10, 10, true);
        assert_eq!(op.src_stage, 11);
        assert_eq!(op.dst_stage, 4);
        let src = vec_from(21, 128, 2.0);
        let dst = vec_from(22, 128, 1.0);
        let mut a = dst.clone();
        let mut b = dst.clone();
        for step in 0..50u32 {
            op.mix_into(step, &src, &mut a);
            op.mix_into(step, &src, &mut b);
        }
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits());
        }
    }

    /// Cross-step composition on a synthetic 2-stage host: capture at src
    /// (post-injection ordering, dst < src), mix at dst next step — the
    /// recurrence composes and stays bounded over long horizons.
    #[test]
    fn cross_step_composition_on_synthetic_host() {
        // 4-stage "stack": identity stages; dst=1, src=3 (dst < src).
        let op = RecircOp::convex(3, 1, 0.1, 10);
        let d = 16;
        let mut buf = RecircBuffer::new(d);
        let mut state: Vec<f32> = vec_from(5, d, 1.0);
        let mut max_norm = l2_norm(&state);
        for step in 0..200u32 {
            // dst hook (stage 1 fires before stage 3 in-layer-order):
            op.mix_into(step, buf.as_slice(), &mut state);
            // src hook: capture this step's (post-injection) source state.
            buf.capture(&state);
            max_norm = max_norm.max(l2_norm(&state));
        }
        // Bounded + finite over the horizon (no divergence at convex α).
        assert!(max_norm.is_finite());
        assert!(max_norm < 5.0, "norm grew unboundedly: {max_norm}");
    }

    /// The paper's anchor pairs round-trip at the reference depths.
    #[test]
    fn paper_anchor_pairs() {
        assert_eq!(RecircPair::for_depth(26), (0.42, 0.15));
        assert_eq!(RecircPair::for_depth(34), (0.53, 0.26));
        assert_eq!(RecircPair::for_depth(48), (0.73, 0.33));
        let op26 = RecircPair::PaperBands.to_op(26, 0.1, 10, true);
        assert_eq!((op26.src_stage, op26.dst_stage), (11, 4));
        let op34 = RecircPair::PaperBands.to_op(34, 0.1, 10, true);
        assert_eq!((op34.src_stage, op34.dst_stage), (18, 9));
        let op48 = RecircPair::PaperBands.to_op(48, 0.1, 10, true);
        assert_eq!((op48.src_stage, op48.dst_stage), (35, 16));
        // Custom pair + degenerate stacks.
        // Custom pair: round(0.5·10)=5, round(0.25·10)=3 (half-away-from-zero).
        let opc = RecircPair::Custom { src: 0.5, dst: 0.25 }.to_op(10, 0.07, 10, false);
        assert_eq!((opc.src_stage, opc.dst_stage), (5, 3));
        assert_eq!(opc.beta, 1.0);
        assert_eq!(frac_to_stage(0.5, 1), 0);
    }

    /// G2 (latency, release): per-step mixture ≤ 2 µs at D = 2048 — it is
    /// O(D): two norm reductions + one fused axpy.
    ///
    /// Gate history (Bench 668 calibrated 553.7 ns; isolated re-measures
    /// 557–571 ns — ~1.8× under the original ≤1µs bar). Inside a
    /// full-workspace release run the suite's own ~4.6k parallel test
    /// threads contend for the box and the same code measured 1023.5 ns —
    /// a 2.4% breach of a 1µs gate that is really scheduler noise, which
    /// made `cargo test --workspace --all-features --release`
    /// nondeterministically red (the 718(a) pricing run hit it). The bar
    /// moves to 2µs: still ≥3.5× over the measured steady state, so it
    /// keeps catching real O(D)-violation regressions while tolerating
    /// in-suite scheduling noise.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "timing gate — release-only")]
    fn g2_step_mixture_under_1us_at_d2048() {
        let d = 2048usize;
        let op = RecircOp::convex(35, 16, 0.15, 10);
        let src = vec_from(1, d, 1.0);
        let mut z = vec_from(2, d, 1.0);
        // Warm up.
        for step in 0..100u32 {
            op.mix_into(step, &src, &mut z);
        }
        let n = 2000u32;
        let t0 = std::time::Instant::now();
        for step in 0..n {
            op.mix_into(step, &src, &mut z);
        }
        let per = t0.elapsed().as_nanos() as f64 / n as f64;
        println!("g2_step_mixture_under_1us_at_d2048: {per:.1} ns/step at D=2048");
        assert!(per < 2000.0, "per-step mixture {per}ns exceeds 2µs gate");
    }
}
