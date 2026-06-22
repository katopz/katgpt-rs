//! The 3-way salience gate: `Speak` / `Silent` / `Delegate`.
//!
//! See [`crate::salience`] for the module-level doc and paper citation.

use core::marker::PhantomData;

use super::types::SalienceDecision;

/// 3-way salience gate. Maps activation `a` + scalars `z`, `c` to one of
/// {`Speak`, `Silent`, `Delegate`}. Uses **two stacked sigmoids** — never
/// softmax (per AGENTS.md).
///
/// Generic over:
/// - activation dimension `D` (const generic),
/// - delegate payload type `A: Clone`.
///
/// **Zero-allocation on the hot path**: all state is fixed-size; `decide`
/// and `decide_batch` perform no heap allocation.
///
/// Reference: Plan 303 (T1.5–T1.10), Research 281,
/// source paper [arxiv 2606.14777](https://arxiv.org/abs/2606.14777)
/// (JoyAI-VL-Interaction, Yao et al., JD.com, Jun 2026).
pub struct SalienceTriGate<A, const D: usize> {
    /// Direction vector for "what makes this agent want to speak".
    /// BLAKE3-committed at freeze/thaw by the caller — this crate is agnostic.
    d_speak: [f32; D],
    /// Direction vector for "what makes this agent want to delegate vs answer
    /// inline".
    d_delegate: [f32; D],
    /// Weight for the zone-attention scalar `z`.
    w_z: f32,
    /// Weight for the curiosity scalar `c`.
    w_c: f32,
    /// Sigmoid inverse temperature (sharpness) for the speak path.
    beta_speak: f32,
    /// Sigmoid inverse temperature (sharpness) for the delegate path.
    beta_delegate: f32,
    /// Decision threshold on `salience` for the speak sigmoid.
    tau_speak: f32,
    /// Decision threshold on `delegate_dot` for the delegate sigmoid.
    tau_delegate: f32,
    /// Anti-babble floor — below this speak score, always `Silent`.
    floor_speak: f32,
    /// Delegate ceiling — above this delegate score, prefer `Delegate` over
    /// `Speak`.
    ceil_delegate: f32,
    /// Satisfy the unused type parameter `A` (we never store an `A` inside
    /// the gate; payloads are passed into `decide` by the caller).
    _marker: PhantomData<A>,
}

impl<A: Clone, const D: usize> SalienceTriGate<A, D> {
    /// Construct a gate from hand-tuned parameters.
    ///
    /// **Validation** (panics on invalid input — this is a one-time setup cost,
    /// not a hot-path check):
    /// - `D >= 1` (also enforced by the const generic, but we assert for
    ///   clarity at the type-system boundary).
    /// - All entries of `d_speak` and `d_delegate` must be finite
    ///   (`f32::is_finite`). NaN/inf direction vectors are programmer error.
    /// - Weights `w_z`, `w_c` must be non-negative (negative weights invert
    ///   the semantics of `z` / `c` and are almost always a wiring bug).
    ///
    /// `beta_*`, `tau_*`, `floor_speak`, `ceil_delegate` are passed through
    /// unchecked — caller is responsible for sane ranges (typically
    /// `beta > 0`, `floor_speak ∈ [0,1]`, `ceil_delegate ∈ [0,1]`).
    ///
    /// Reference: Plan 303 T1.6.
    #[must_use]
    pub fn new(
        d_speak: [f32; D],
        d_delegate: [f32; D],
        w_z: f32,
        w_c: f32,
        beta_speak: f32,
        beta_delegate: f32,
        tau_speak: f32,
        tau_delegate: f32,
        floor_speak: f32,
        ceil_delegate: f32,
    ) -> Self {
        // Const-generic sanity check. `D == 0` would compile (zero-sized
        // arrays are valid Rust) but produces a useless gate; reject it
        // loudly here.
        assert!(D >= 1, "SalienceTriGate::new: D must be >= 1, got {D}");

        // Direction vectors must be finite. NaN/inf propagate through the
        // dot product and corrupt both sigmoids.
        for (i, &v) in d_speak.iter().enumerate() {
            assert!(
                v.is_finite(),
                "SalienceTriGate::new: d_speak[{i}] is not finite (got {v})"
            );
        }
        for (i, &v) in d_delegate.iter().enumerate() {
            assert!(
                v.is_finite(),
                "SalienceTriGate::new: d_delegate[{i}] is not finite (got {v})"
            );
        }

        // Negative weights invert the meaning of z / c. Reject.
        assert!(
            w_z >= 0.0,
            "SalienceTriGate::new: w_z must be non-negative (got {w_z})"
        );
        assert!(
            w_c >= 0.0,
            "SalienceTriGate::new: w_c must be non-negative (got {w_c})"
        );

        Self {
            d_speak,
            d_delegate,
            w_z,
            w_c,
            beta_speak,
            beta_delegate,
            tau_speak,
            tau_delegate,
            floor_speak,
            ceil_delegate,
            _marker: PhantomData,
        }
    }

    /// Per-tick emit decision.
    ///
    /// # Decision rule (Plan 303 T1.7)
    /// ```text
    /// salience       = dot(a, d_speak)    + w_z * z + w_c * c
    /// score_speak    = sigmoid(beta_speak    * (salience    - tau_speak))
    /// delegate_dot   = dot(a, d_delegate)
    /// score_delegate = sigmoid(beta_delegate * (delegate_dot - tau_delegate))
    ///
    /// if score_speak    < floor_speak:    Silent
    /// elif score_delegate > ceil_delegate: Delegate(delegate_payload)
    /// else:                                Speak
    /// ```
    ///
    /// All three branches return a first-class [`SalienceDecision`] variant;
    /// `Silent` is not a default suppression.
    ///
    /// # Determinism
    /// Bit-identical across runs given the same `(a, z, c, gate)` — no RNG,
    /// no thread-local state, no allocation.
    ///
    /// # Zero allocation
    /// No `Vec`, `Box`, or heap traffic. All temporaries are stack scalars.
    ///
    /// # Parameters
    /// - `a`: activation vector (latent direction; the caller picks the
    ///   space — HLA, CGSP embedding, etc.).
    /// - `z`: zone-attention scalar (how much this NPC cares about the
    ///   current zone).
    /// - `c`: curiosity scalar.
    /// - `delegate_payload`: payload to wrap in `Delegate` if the delegate
    ///   branch fires. Owned by the caller; we move it into the variant.
    /// - `tick`: tick reserved for `SilenceToken` emission in Phase 3
    ///   (Plan 303 T3.x). Currently unused by the basic decision.
    #[must_use]
    pub fn decide(
        &self,
        a: &[f32; D],
        z: f32,
        c: f32,
        delegate_payload: A,
        _tick: u64,
    ) -> SalienceDecision<A> {
        // salience = dot(a, d_speak) + w_z * z + w_c * c
        //
        // Dot product first, then two FMAs for the scalar contributions.
        // Pattern matches `bridge/mod.rs` / `cumprodsum.rs` (`a.mul_add(b, sum)`).
        let d_speak = dot_fma(a, &self.d_speak);
        let salience = self.w_z.mul_add(z, self.w_c.mul_add(c, d_speak));

        // Two stacked sigmoids — never softmax (AGENTS.md).
        let score_speak = sigmoid(self.beta_speak * (salience - self.tau_speak));

        let delegate_dot = dot_fma(a, &self.d_delegate);
        let score_delegate = sigmoid(self.beta_delegate * (delegate_dot - self.tau_delegate));

        // Two-level decision. The chained boolean form reads cleanly as
        // "Silent has highest precedence, then Delegate, then Speak"; a
        // `match` would need a synthesized discriminant for no benefit.
        if score_speak < self.floor_speak {
            SalienceDecision::Silent
        } else if score_delegate > self.ceil_delegate {
            SalienceDecision::Delegate(delegate_payload)
        } else {
            SalienceDecision::Speak
        }
    }

    /// Batched form of [`Self::decide`].
    ///
    /// Caller owns the output buffer — no internal allocation. All slice
    /// lengths are checked with `debug_assert!` (cheap in dev, elided in
    /// release — caller's responsibility to keep lengths consistent in
    /// production hot paths).
    ///
    /// # Parameters
    /// - `activations`: `[N][D]` activation matrix (flat slice of arrays).
    /// - `z`, `c`: per-row scalars, length `N`.
    /// - `payloads`: per-row delegate payloads, length `N`.
    /// - `tick`: passed through to each per-row `decide` (currently unused
    ///   by the basic decision; reserved for Phase 3).
    /// - `out`: output buffer, length `N`. Overwritten in place.
    ///
    /// Reference: Plan 303 T1.10.
    pub fn decide_batch(
        &self,
        activations: &[[f32; D]],
        z: &[f32],
        c: &[f32],
        payloads: &[A],
        tick: u64,
        out: &mut [SalienceDecision<A>],
    ) {
        let n = activations.len();
        debug_assert_eq!(z.len(), n, "decide_batch: z length mismatch");
        debug_assert_eq!(c.len(), n, "decide_batch: c length mismatch");
        debug_assert_eq!(payloads.len(), n, "decide_batch: payloads length mismatch");
        debug_assert_eq!(out.len(), n, "decide_batch: out length mismatch");

        for i in 0..n {
            // Clone payload — the variant owns its copy. Caller passes `&[A]`
            // because the same payload table may be reused across ticks.
            out[i] = self.decide(&activations[i], z[i], c[i], payloads[i].clone(), tick);
        }
    }
}

// ── Private hot-path helpers ──────────────────────────────────────────────

/// Dot product of two fixed-size f32 slices, accumulated with `mul_add` for
/// single-rounding FMA contraction (matches the `bridge/mod.rs` /
/// `cumprodsum.rs` convention).
///
/// Marked `#[inline(always)]` so the unrolled loop fuses with the caller's
/// sigmoid computation; LLVM vectorizes the inner accumulation.
#[inline(always)]
fn dot_fma<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..D {
        // sum += a[i] * b[i], with FMA.
        sum = a[i].mul_add(b[i], sum);
    }
    sum
}

/// Numerically-stable logistic sigmoid. Never softmax (per AGENTS.md rule).
///
/// Standard libm-bounded branchless form:
/// - `x >= 0`: `1 / (1 + exp(-x))` — `exp` arg is non-positive, no overflow.
/// - `x <  0`: `exp(x) / (1 + exp(x))` — `exp` arg is negative, no overflow.
///
/// Both branches avoid the catastrophic cancellation that the naive form
/// hits for large negative `x`.
///
// TODO Plan 303 T1.8: hoist to `crate::simd::fast_sigmoid` when SIMD
// dispatcher lands. The current `crate::simd` is a pure re-export of
// `katgpt_core::simd::*`, which exposes no sigmoid symbol — hence the
// private copy here.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────
//
// Phase 1 acceptance (3 path tests) + Phase 2 G1/G2 property tests.
// All tests use D = 8 unless noted. Deterministic LCG — no `rand` dep.

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic LCG (numerical recipes constants). Used in
    /// place of `rand` so property tests stay bit-reproducible across runs
    /// and platforms.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        fn next_f32(&mut self) -> f32 {
            // Uniform in [0, 1). High bits only — well-mixed.
            (self.next() as f32) / (u32::MAX as f32)
        }
    }

    // Hand-tuned direction vectors for the path tests. We use unit-magnitude
    // vectors so `dot` maps directly to the projection of `a` onto the
    // direction.
    const D: usize = 8;
    const D_SPEAK: [f32; D] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const D_DELEGATE: [f32; D] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // ── Phase 1 acceptance: 3 path tests ───────────────────────────────

    #[test]
    fn test_silent_path() {
        // High floor_speak (0.9) + low-salience activation → Silent.
        // salience = a[0] * d_speak[0] = 0.1 (a[0] = 0.1) + 0 * z + 0 * c
        //         = 0.1, far below tau_speak=0.5. score_speak ≈ sigmoid(-2*0.4)
        //         ≈ sigmoid(-0.8) ≈ 0.31 < 0.9 → Silent.
        let gate: SalienceTriGate<&'static str, D> = SalienceTriGate::new(
            D_SPEAK, D_DELEGATE, 0.0, // w_z
            0.0, // w_c
            2.0, // beta_speak
            2.0, // beta_delegate
            0.5, // tau_speak
            0.5, // tau_delegate
            0.9, // floor_speak — high
            1.5, // ceil_delegate — unreachable
        );
        let a = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = gate.decide(&a, 0.0, 0.0, "delegate", 0);
        assert!(matches!(d, SalienceDecision::Silent), "got {d:?}");
    }

    #[test]
    fn test_speak_path() {
        // Low floor_speak (0.0) + moderate salience + low ceil_delegate (1.5,
        // unreachable since sigmoid ∈ (0,1)) → Speak.
        // salience = a[0] = 0.6 > tau_speak=0.5 → score_speak ≈ sigmoid(2*0.1)
        //         ≈ 0.55 > floor_speak=0.0, so not Silent.
        // delegate_dot = a[1] = 0.0 → score_delegate = sigmoid(2*(-0.5))
        //              ≈ sigmoid(-1.0) ≈ 0.27 ≯ ceil_delegate=1.5 → not Delegate.
        // → Speak.
        let gate: SalienceTriGate<&'static str, D> = SalienceTriGate::new(
            D_SPEAK, D_DELEGATE, 0.0, 0.0, 2.0, 2.0, 0.5, // tau_speak
            0.5, // tau_delegate
            0.0, // floor_speak — low
            1.5, // ceil_delegate — unreachable
        );
        let a = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = gate.decide(&a, 0.0, 0.0, "delegate", 0);
        assert!(matches!(d, SalienceDecision::Speak), "got {d:?}");
    }

    #[test]
    fn test_delegate_path() {
        // Low floor_speak + high delegate_dot + ceil_delegate (0.3) below
        // score_delegate → Delegate.
        // salience = a[0] = 0.6 → score_speak ≈ 0.55 > floor_speak=0.0.
        // delegate_dot = a[1] = 1.0 → score_delegate = sigmoid(2*(1.0 - 0.5))
        //              = sigmoid(1.0) ≈ 0.731 > ceil_delegate=0.3 → Delegate.
        let gate: SalienceTriGate<&'static str, D> = SalienceTriGate::new(
            D_SPEAK, D_DELEGATE, 0.0, 0.0, 2.0, 2.0, 0.5, // tau_speak
            0.5, // tau_delegate
            0.0, // floor_speak — low
            0.3, // ceil_delegate — low → fires Delegate
        );
        let a = [0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = gate.decide(&a, 0.0, 0.0, "payload-42", 0);
        match d {
            SalienceDecision::Delegate(s) => assert_eq!(s, "payload-42"),
            other => panic!("expected Delegate, got {other:?}"),
        }
    }

    // ── Phase 2 G1: determinism ────────────────────────────────────────

    #[test]
    fn test_g1_determinism() {
        // Same inputs twice → equal decisions. Verifies no hidden RNG /
        // thread-local / allocation nondeterminism leaks into `decide`.
        let gate: SalienceTriGate<u32, D> =
            SalienceTriGate::new(D_SPEAK, D_DELEGATE, 0.3, 0.2, 1.5, 1.5, 0.5, 0.5, 0.4, 0.6);
        let a = [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
        let d1 = gate.decide(&a, 0.5, 0.5, 7, 100);
        let d2 = gate.decide(&a, 0.5, 0.5, 7, 100);
        assert_eq!(d1, d2, "decide is not deterministic");
    }

    // ── Phase 2 G1: monotonicity in salience ──────────────────────────

    #[test]
    fn test_g1_monotonicity_in_salience() {
        // Sweep a[0] (the only nonzero entry of d_speak) from 0 → 1 with
        // delegate_dot pinned to 0 (a[1] = 0 ⇒ score_delegate ≈ 0.27 ≯ 1.5
        // ceil_delegate ⇒ never Delegate). Verify a single threshold crossing
        // from Silent → Speak.
        let gate: SalienceTriGate<i32, D> = SalienceTriGate::new(
            D_SPEAK, D_DELEGATE, 0.0, 0.0, 10.0, // sharp beta so the threshold is crisp
            10.0, 0.5, // tau_speak
            0.5, // tau_delegate
            0.5, // floor_speak
            1.5, // ceil_delegate — unreachable
        );

        let mut last_silent = true;
        let mut transitions = 0u32;
        let steps = 200u32;
        for s in 0..=steps {
            let a0 = s as f32 / steps as f32;
            let a = [a0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let d = gate.decide(&a, 0.0, 0.0, 0, 0);
            let is_silent = matches!(d, SalienceDecision::Silent);
            if s > 0 && is_silent != last_silent {
                transitions += 1;
            }
            // Never Delegate in this sweep — we pinned delegate_dot to 0
            // and ceil_delegate is unreachable.
            assert!(
                !matches!(d, SalienceDecision::Delegate(_)),
                "unexpected Delegate at a[0]={a0}"
            );
            last_silent = is_silent;
        }
        assert_eq!(
            transitions, 1,
            "expected exactly one Silent↔Speak threshold crossing, got {transitions}"
        );
        // Start (a[0]=0) is Silent, end (a[0]=1) is Speak — sanity.
        let a_lo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let a_hi = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(matches!(
            gate.decide(&a_lo, 0.0, 0.0, 0, 0),
            SalienceDecision::Silent
        ));
        assert!(matches!(
            gate.decide(&a_hi, 0.0, 0.0, 0, 0),
            SalienceDecision::Speak
        ));
    }

    // ── Phase 2 G1: monotonicity in delegate_dot ──────────────────────

    #[test]
    fn test_g1_monotonicity_in_delegate_dot() {
        // Sweep a[1] (the d_delegate direction) from 0 → 1. Pin a[0] high
        // enough that score_speak > floor_speak (so we never go Silent),
        // then verify Speak → Delegate is a single monotone transition.
        let gate: SalienceTriGate<i32, D> = SalienceTriGate::new(
            D_SPEAK, D_DELEGATE, 0.0, 0.0, 10.0, // sharp beta_speak
            10.0, // sharp beta_delegate
            0.5,  // tau_speak
            0.5,  // tau_delegate
            0.1,  // floor_speak — low so we don't accidentally go Silent
            0.5,  // ceil_delegate — Delegate fires once score_delegate > 0.5
        );

        let a0 = 0.8; // a[0] high → score_speak ≈ 1, never Silent
        let mut last_delegate = false;
        let mut transitions = 0u32;
        let steps = 200u32;
        for s in 0..=steps {
            let a1 = s as f32 / steps as f32;
            let a = [a0, a1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let d = gate.decide(&a, 0.0, 0.0, 0, 0);
            let is_delegate = matches!(d, SalienceDecision::Delegate(_));
            if s > 0 && is_delegate != last_delegate {
                transitions += 1;
            }
            // Never Silent in this sweep — a[0] is high enough.
            assert!(
                !matches!(d, SalienceDecision::Silent),
                "unexpected Silent at a[1]={a1}"
            );
            last_delegate = is_delegate;
        }
        assert_eq!(
            transitions, 1,
            "expected exactly one Speak→Delegate threshold crossing, got {transitions}"
        );
    }

    // ── Phase 2 G2: two-sigmoid ablation parity ───────────────────────

    #[test]
    fn test_g2_ablation_parity() {
        // Set ceil_delegate = +∞ so the delegate sigmoid can never fire.
        // The gate's output must then be bit-identical (variant-by-variant)
        // to a reference that uses ONLY `score_speak < floor_speak`.
        //
        // This proves the delegate sigmoid is *provably separable* from the
        // speak/silent decision: when ceil_delegate is unbounded, the
        // delegate path contributes nothing to the output stream.
        let gate: SalienceTriGate<u32, D> = SalienceTriGate::new(
            [0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.05, -0.05], // arbitrary d_speak
            [0.1, 0.2, 0.3, 0.0, -0.3, -0.2, 0.4, 0.1],    // arbitrary d_delegate
            0.5,                                           // w_z
            0.3,                                           // w_c
            2.5,                                           // beta_speak
            3.0,                                           // beta_delegate
            0.2,                                           // tau_speak
            0.1,                                           // tau_delegate
            0.4,                                           // floor_speak
            f32::INFINITY, // ceil_delegate — ablation: Delegate never fires
        );

        // Reference decision: speak/silent only.
        let reference = |a: &[f32; D], z: f32, c: f32| -> SalienceDecision<u32> {
            let salience = dot_fma(a, &gate.d_speak_reference()) + 0.5 * z + 0.3 * c;
            let score_speak = sigmoid(2.5 * (salience - 0.2));
            if score_speak < 0.4 {
                SalienceDecision::Silent
            } else {
                SalienceDecision::Speak
            }
        };

        let mut rng = Lcg::new(0xC0FFEE_BABE_1234);
        for _ in 0..100 {
            let mut a = [0f32; D];
            for v in a.iter_mut() {
                // Span [-1, 1] — covers both direction projections.
                *v = rng.next_f32() * 2.0 - 1.0;
            }
            let z = rng.next_f32();
            let c = rng.next_f32();

            let actual = gate.decide(&a, z, c, 0, 0);
            let expected = reference(&a, z, c);

            // Same variant (we don't care about the payload — both reference
            // and actual return `Speak`/`Silent` without payload).
            let actual_is_silent = matches!(actual, SalienceDecision::Silent);
            let expected_is_silent = matches!(expected, SalienceDecision::Silent);
            assert_eq!(
                actual_is_silent, expected_is_silent,
                "ablation parity failed at a={a:?} z={z} c={c}: \
                 actual={actual:?} expected={expected:?}"
            );
            // Also never Delegate under the ablation.
            assert!(
                !matches!(actual, SalienceDecision::Delegate(_)),
                "Delegate fired despite ceil_delegate=+∞"
            );
        }
    }

    // ── Auxiliary: batch path smoke test ──────────────────────────────

    #[test]
    fn test_decide_batch_smoke() {
        // Batch of 4 decisions; verify per-row variants and zero
        // allocation (compile-time guarantee: no `Vec` in fn signature).
        //
        // Tuned so each row lands in a distinct variant:
        //   row 0: a[0]=0   → salience=0   → score_speak≈sigmoid(-1)=0.27
        //                                                   < floor_speak=0.4 → Silent
        //   row 1: a[0]=0.8 → salience=0.8 → score_speak≈sigmoid(0.6)=0.65
        //                                                   ≥ floor_speak, delegate_dot=0
        //                                                   → score_delegate≈0.27 ≯ ceil=0.95 → Speak
        //   row 2: a[0]=0.8,a[1]=1 → delegate_dot=1 → score_delegate≈sigmoid(1)=0.73
        //                                                   > ceil=0.5 → Delegate(12)
        //   row 3: a[0]=0,a[1]=1   → salience=0 → Silent (precedence over Delegate)
        let gate: SalienceTriGate<u32, D> = SalienceTriGate::new(
            D_SPEAK, D_DELEGATE, 0.0, 0.0, 2.0, // beta_speak
            2.0, // beta_delegate
            0.5, // tau_speak
            0.5, // tau_delegate
            0.4, // floor_speak — Silent below this
            0.5, // ceil_delegate — Delegate above this
        );
        let activations: [[f32; D]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Silent (low salience)
            [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Speak (high salience, low delegate_dot)
            [0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Delegate (a[1] high → score_delegate>ceil)
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Silent (low salience — precedence over Delegate)
        ];
        let z = [0.0; 4];
        let c = [0.0; 4];
        let payloads = [10u32, 11, 12, 13];
        let mut out = [
            SalienceDecision::Silent,
            SalienceDecision::Silent,
            SalienceDecision::Silent,
            SalienceDecision::Silent,
        ];

        gate.decide_batch(&activations, &z, &c, &payloads, 7, &mut out);

        assert!(matches!(out[0], SalienceDecision::Silent));
        assert!(matches!(out[1], SalienceDecision::Speak));
        match &out[2] {
            SalienceDecision::Delegate(p) => assert_eq!(*p, 12),
            other => panic!("row 2: expected Delegate(12), got {other:?}"),
        }
        // Row 3 demonstrates Silent precedence: even though a[1]=1 would
        // trigger Delegate on its own, the low salience forces Silent first.
        assert!(matches!(out[3], SalienceDecision::Silent));
    }

    // Hook for the ablation test: expose the gate's d_speak to the reference
    // closure without making the field public. Test-only.
    impl<A: Clone, const D: usize> SalienceTriGate<A, D> {
        #[cfg(test)]
        fn d_speak_reference(&self) -> &[f32; D] {
            &self.d_speak
        }
    }
}
