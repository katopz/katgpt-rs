//! Latent Confounder Audit — three modelless forward-pass diagnostics
//! auditing a conditioning latent for action-irrelevant confounders.
//!
//! Distilled from CD-LAM §III-B + Appendix A
//! (Wei et al., Aether AI / UCSD, [arXiv:2607.09185](https://arxiv.org/abs/2607.09185)).
//! See `katgpt-rs/.research/460_*.md` for the research note and
//! `katgpt-rs/.issues/194_*.md` for the implementation task.
//!
//! # What this is
//!
//! The training recipe (`L_emb` + `L_ctr` + `L_cal`) is genuinely gradient
//! descent → riir-train. The **diagnostic framework** is modelless: three
//! forward-pass metrics computed on any encoder `E: (Obs, Obs) → Latent`,
//! with zero gradient steps and zero weight mutation.
//!
//! # The three diagnostics
//!
//! Let `D = RMS(‖E(x, x')‖)` over ordinary transitions (the normalization
//! denominator — a scale-invariant reference).
//!
//! 1. **Zero-transition response** `R₀ = RMS(‖E(x, x)‖) / D` — a no-op input
//!    pair (identical observations) should produce a near-zero latent. If it
//!    doesn't, the encoder is responding to static input — a confounder leak.
//! 2. **Shift-invariance response** `R_shift = RMS(‖E(x, T(x))‖) / D` — an
//!    input pair differing only by a nuisance transform `T` (that the
//!    encoder should be invariant to) should produce a near-zero latent.
//! 3. **Shortcut leakage** `L = mean_cos(diff-action, same-context) −
//!    mean_cos(same-action, diff-context)` — action similarity should
//!    dominate context similarity in cosine structure. **Clean = negative**
//!    (same-action pairs have higher cosine than diff-action pairs); leaky
//!    encoders score closer to zero or positive.
//!
//! # What this is NOT
//!
//! - NOT probabilities / confidence scores / predictive intervals. The
//!   fields are raw measurement ratios. The "Report the Floor" conformal-
//!   naive rule (Research 322 / Plan 340) does NOT apply.
//! - NOT a router / fix. The audit produces measurements; the caller decides
//!   what to do with them (re-mine, re-train, accept the risk).
//! - NOT a per-tick primitive. Run before deploying a mined direction (MAG
//!   Plan 418, TILR Plan 425, Latent Field Steering Plan 309) or as a CI
//!   sanity check on hand-constructed direction vectors (HLA, functor).
//!
//! # Sign convention (shortcut leakage)
//!
//! `shortcut_leakage < 0` = **clean** (action similarity dominates context
//! similarity in the cosine structure — same-action pairs project closer
//! than diff-action pairs). `shortcut_leakage > 0` = **confounded** (context
//! is leaking into the latent — diff-action/same-context pairs look more
//! similar than same-action/diff-context pairs).
//!
//! # Allocation
//!
//! **Zero steady-state allocation.** The [`AuditScratch`] buffers are sized
//! ONCE to `latent_dim` at construction; every audit call writes into them
//! in place. The G4 gate (CountingAllocator, 100 steady-state calls) enforces
//! this contract.
//!
//! # Encoder contract
//!
//! The encoder is any `Fn(&[f32], &[f32], &mut [f32])` — a closure taking
//! `(obs_a, obs_b, out_buf)`. The output buffer's length is the latent
//! dimension, fixed across calls. Real HLA / functor / MAG encoders wrap
//! their inner computation in a one-line closure that writes into `out_buf`.
//!
//! Latent Confounder Audit — three modelless forward-pass diagnostics
//! (R₀ zero-transition response + R_shift shift-invariance response + L
//! shortcut leakage) auditing a conditioning latent for action-irrelevant
//! confounders. Distilled from CD-LAM §III-B + Appendix A (Wei et al.,
//! arXiv:2607.09185; Research 460, Issue 194). The diagnostic half of CD-LAM
//! (the L_emb/L_ctr/L_cal training recipe → riir-train). Pure numeric over a
//! caller-supplied encoder closure; zero deps. Opt-in — diagnostic primitive,
//! not a runtime capability. Stays opt-in until a consumer (MAG/TILR/Steering)
//! benchmarks a quality gain from running the audit before deployment.

// (Module gating is handled by `#[cfg(feature = "latent_confounder_audit")]`
// on the `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

// ─── Result types ───────────────────────────────────────────────────────────

/// Three modelless forward-pass diagnostics auditing a conditioning latent
/// for action-irrelevant confounders.
///
/// Distilled from CD-LAM (arXiv:2607.09185) §III-B + Appendix A. All three
/// fields are raw measurement ratios — NOT probabilities, NOT confidence
/// scores. See the module docs for the sign conventions.
///
/// Construct via [`audit_confounders`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LatentConfounderAudit {
    /// Zero-transition response `R₀ = RMS(‖E(x, x)‖) / D`.
    ///
    /// **Lower = cleaner.** `≈ 0` = the encoder produces near-zero on a
    /// no-op (identical-observation) input pair, which is the correct
    /// behavior for an action/dynamics encoder.
    pub zero_transition_response: f32,

    /// Shift-invariance response `R_shift = RMS(‖E(x, T(x))‖) / D`.
    ///
    /// **Lower = cleaner.** `≈ 0` = the encoder is invariant to nuisance
    /// transforms (the caller's `T`). A non-zero value means the latent
    /// responds to transforms it should ignore.
    pub shift_invariance_response: f32,

    /// Shortcut leakage `mean_cos(diff-action, same-context) −
    /// mean_cos(same-action, diff-context)`.
    ///
    /// **More negative = cleaner.** `< 0` = action similarity dominates
    /// context similarity (same-action pairs project closer than diff-action
    /// pairs). `≥ 0` = context is leaking into the latent structure.
    pub shortcut_leakage: f32,

    /// The normalization denominator `D = RMS(‖E(x, x')‖) + ε` actually used.
    ///
    /// Exposed for caller-side inspection (e.g. detecting a degenerate
    /// encoder that produces near-zero on ordinary transitions — `D ≈ ε`).
    pub normalization_denominator: f32,
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffer for zero-alloc audit hot path.
///
/// Construct once via [`AuditScratch::new`] with the encoder's latent
/// dimension; reuse across [`audit_confounders`] calls. The two internal
/// `Vec`s are the output buffers passed to the encoder — pre-sized to
/// `latent_dim` so no reallocation ever happens inside the audit.
pub struct AuditScratch {
    /// Output buffer for the first latent of a pair. Length = `latent_dim`.
    pub latent_a: Vec<f32>,
    /// Output buffer for the second latent of a pair. Length = `latent_dim`.
    pub latent_b: Vec<f32>,
}

impl AuditScratch {
    /// Construct a scratch for an encoder producing `latent_dim` outputs.
    ///
    /// Allocates the two internal buffers ONCE (sized to `latent_dim`). All
    /// subsequent [`audit_confounders`] calls reuse these buffers in place —
    /// zero steady-state allocation.
    #[inline]
    pub fn new(latent_dim: usize) -> Self {
        Self {
            latent_a: vec![0.0_f32; latent_dim],
            latent_b: vec![0.0_f32; latent_dim],
        }
    }

    /// Resize the scratch to a new latent dim. Allocates only if the new dim
    /// exceeds the current capacity — `Vec::resize` is a no-op when shrinking
    /// in-place. Use when auditing multiple encoders of different dims with
    /// one scratch instance.
    #[inline]
    pub fn resize(&mut self, latent_dim: usize) {
        self.latent_a.resize(latent_dim, 0.0);
        self.latent_b.resize(latent_dim, 0.0);
    }

    /// Returns the latent dim this scratch is currently sized for.
    #[inline]
    pub fn latent_dim(&self) -> usize {
        self.latent_a.len()
    }
}

// ─── Primitive ──────────────────────────────────────────────────────────────

/// Tiny epsilon to prevent divide-by-zero when the encoder produces
/// all-zeros on ordinary transitions (degenerate encoder). Small enough to
/// not affect any real-workload measurement.
const EPS: f32 = 1.0e-12;

/// Audit an encoder for confounder purity.
///
/// `encoder` is any closure `Fn(&[f32], &[f32], &mut [f32])` that writes its
/// latent into the third argument (the output buffer). The buffer length is
/// `scratch.latent_dim()` — the encoder must write exactly that many f32s.
///
/// The caller supplies test pairs organized by category:
///
/// - `zero_transition_pairs`: `(x, x)` — identical-observation pairs.
/// - `shift_pairs`: `(x, T(x))` — observation vs nuisance-transformed
///   observation. The caller chooses `T` (constant offset, tick drift, view
///   transform — whatever the encoder should be invariant to).
/// - `ordinary_pairs`: `(x, x')` — typical input pairs for the RMS
///   normalization denominator.
/// - `same_action_diff_context`: pairs of pairs `((x_a, x_a'), (x_b, x_b'))`
///   that should produce the same action/dynamics but come from different
///   contexts. The two latents `E(x_a, x_a')` and `E(x_b, x_b')` are
///   compared by cosine.
/// - `diff_action_same_context`: pairs of pairs that should produce different
///   actions but share a context. Same cosine comparison.
///
/// Empty category slices skip that diagnostic (the corresponding field stays
/// at its `Default::default()` value, `0.0`). Empty `ordinary_pairs` falls
/// back to `D = EPS` (degenerate; `normalization_denominator` reports the
/// small value so the caller can detect this).
///
/// # Allocation
///
/// **Zero steady-state allocation.** All encoder outputs go through the
/// pre-sized `scratch.latent_a` / `scratch.latent_b` buffers.
///
/// # Example
///
/// ```
/// use katgpt_core::latent_confounder_audit::{AuditScratch, audit_confounders};
///
/// // A clean encoder: outputs the mean-subtracted displacement.
/// // Invariant to constant shifts (R_shift ≈ 0), zero on identical inputs
/// // (R_0 = 0), and action-dominated (shortcut_leakage < 0).
/// let clean_encoder = |a: &[f32], b: &[f32], out: &mut [f32]| {
///     let n = out.len();
///     let mut mean = 0.0_f32;
///     for i in 0..n {
///         out[i] = b[i] - a[i];
///         mean += out[i];
///     }
///     mean /= n as f32;
///     for v in out.iter_mut() {
///         *v -= mean;
///     }
/// };
///
/// let x  = [1.0_f32, 2.0, 3.0, 4.0];
/// // NON-CONSTANT displacement so the mean-subtracted version is non-zero.
/// let xp = [0.7_f32, 1.9, 2.8, 4.2]; // displacement [-0.3, -0.1, -0.2, 0.2]
/// // Same displacement, different context.
/// let xn = [10.0_f32, 20.0, 30.0, 40.0];
/// let xnp = [9.7_f32, 19.9, 29.8, 40.2];
/// // Different displacement, same context.
/// let xp_big = [1.5_f32, 1.5, 3.5, 4.5]; // displacement [0.5, -0.5, 0.5, 0.5]
/// // Nuisance transform: shift ONE frame by a constant. The mean-subtracting
/// // clean encoder is invariant to this — the constant displacement
/// // mean-subtracts to zero, producing a near-zero latent (R_shift ≈ 0).
/// let shifted = [x[0] + 7.0, x[1] + 7.0, x[2] + 7.0, x[3] + 7.0];
///
/// // Identical-obs pair (zero-transition).
/// let zero_pairs: &[(&[f32], &[f32])] = &[(&x, &x)];
/// // Nuisance-shift pair: one frame shifted by a constant.
/// let shift_pairs: &[(&[f32], &[f32])] = &[(&x, &shifted)];
/// // Ordinary pair for normalization.
/// let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x, &xp)];
/// // Same-action / diff-context: same displacement, different starting x.
/// let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[
///     ((&x, &xp), (&xn, &xnp)),
/// ];
/// // Diff-action / same-context: different displacement, same starting x.
/// let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[
///     ((&x, &xp), (&x, &xp_big)),
/// ];
///
/// let mut scratch = AuditScratch::new(4);
/// let audit = audit_confounders(
///     &clean_encoder,
///     zero_pairs,
///     shift_pairs,
///     ordinary_pairs,
///     same_action,
///     diff_action,
///     &mut scratch,
/// );
///
/// assert!(audit.zero_transition_response < 1.0e-5);
/// assert!(audit.shift_invariance_response < 1.0e-5);
/// assert!(audit.shortcut_leakage < 0.0, "clean encoder: action dominates");
/// ```
#[allow(clippy::type_complexity)] // The pair-of-pairs type is inherent to the API.
#[inline]
pub fn audit_confounders<E>(
    encoder: &E,
    zero_transition_pairs: &[(&[f32], &[f32])],
    shift_pairs: &[(&[f32], &[f32])],
    ordinary_pairs: &[(&[f32], &[f32])],
    same_action_diff_context: &[((&[f32], &[f32]), (&[f32], &[f32]))],
    diff_action_same_context: &[((&[f32], &[f32]), (&[f32], &[f32]))],
    scratch: &mut AuditScratch,
) -> LatentConfounderAudit
where
    E: Fn(&[f32], &[f32], &mut [f32]),
{
    // ── Compute the normalization denominator D = RMS(||E(x, x')||) + eps ──
    //
    // We accumulate sum-of-squares and divide once at the end. The encoder
    // writes into scratch.latent_a (one latent per ordinary pair; we don't
    // need the second buffer for the norm computation).
    let (ordinary_sum_sq, ordinary_count) =
        accumulate_norm_sq(encoder, ordinary_pairs, &mut scratch.latent_a);

    let d = if ordinary_count > 0 {
        (ordinary_sum_sq / ordinary_count as f32).sqrt() + EPS
    } else {
        // Degenerate: no ordinary pairs. Fall back to EPS so division by D
        // doesn't blow up; the caller can detect this via the exposed
        // normalization_denominator field.
        EPS
    };

    // ── R_0 = RMS(||E(x, x)||) / D ────────────────────────────────────────
    let (zero_sum_sq, zero_count) =
        accumulate_norm_sq(encoder, zero_transition_pairs, &mut scratch.latent_a);

    let r_zero = if zero_count > 0 {
        ((zero_sum_sq / zero_count as f32).sqrt()) / d
    } else {
        0.0
    };

    // ── R_shift = RMS(||E(x, T(x))||) / D ────────────────────────────────
    let (shift_sum_sq, shift_count) =
        accumulate_norm_sq(encoder, shift_pairs, &mut scratch.latent_a);

    let r_shift = if shift_count > 0 {
        ((shift_sum_sq / shift_count as f32).sqrt()) / d
    } else {
        0.0
    };

    // ── Shortcut leakage ─────────────────────────────────────────────────
    //
    // mean_cos(same_action_diff_context) — cos of the two latents in each
    // pair-of-pairs, averaged. Both latents share the encoder, so we need
    // both scratch buffers (latent_a for the first, latent_b for the second).
    let mean_cos_same = mean_cosine_pair(
        encoder,
        same_action_diff_context,
        &mut scratch.latent_a,
        &mut scratch.latent_b,
    );
    let mean_cos_diff = mean_cosine_pair(
        encoder,
        diff_action_same_context,
        &mut scratch.latent_a,
        &mut scratch.latent_b,
    );

    // Sign convention (issue spec): shortcut_leakage < 0 = clean (action
    // dominates context). So action-dominance must make the value negative.
    //
    // Clean encoder: same-action pairs have HIGHER cos than diff-action pairs
    // (mean_cos_same > mean_cos_diff). To make the value negative in that
    // case: shortcut_leakage = mean_cos_diff - mean_cos_same.
    //
    // Confounded encoder: same-action pairs have LOWER cos than diff-action
    // pairs (mean_cos_same < mean_cos_diff) → shortcut_leakage > 0.
    let shortcut_leakage = mean_cos_diff - mean_cos_same;

    LatentConfounderAudit {
        zero_transition_response: r_zero,
        shift_invariance_response: r_shift,
        shortcut_leakage,
        normalization_denominator: d,
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Sum of squared L2 norms of `encoder(obs_a, obs_b)` over all pairs.
/// Returns `(sum_of_squares, count)`. The encoder output goes into `buf`.
///
/// Count skips pairs whose observation lengths are zero or where `buf` has
/// zero length (defensive — diagnostic primitive must not panic).
#[inline]
fn accumulate_norm_sq<E>(encoder: &E, pairs: &[(&[f32], &[f32])], buf: &mut [f32]) -> (f32, u32)
where
    E: Fn(&[f32], &[f32], &mut [f32]),
{
    if buf.is_empty() {
        return (0.0, 0);
    }
    let mut sum_sq: f32 = 0.0;
    let mut count: u32 = 0;
    let dim = buf.len();
    for &(a, b) in pairs {
        if a.is_empty() || b.is_empty() {
            continue;
        }
        // Encoder writes into buf; we trust it to write `dim` values.
        encoder(a, b, buf);
        let mut norm_sq: f32 = 0.0;
        for i in 0..dim {
            // SAFETY: i < dim == buf.len(). Manual bounds-elision mirrors
            // latent_trajectory_geometry's hot-path pattern; the encoder
            // contract requires it to write exactly `dim` values, so the
            // checked-index proof obligation is on the caller/encoder.
            let v = unsafe { *buf.get_unchecked(i) };
            norm_sq += v * v;
        }
        sum_sq += norm_sq;
        count += 1;
    }
    (sum_sq, count)
}

/// Mean cosine similarity between the two latents of each pair-of-pairs.
///
/// For each `((a, b), (c, d))` entry: encode `(a, b)` into `buf_a` and
/// `(c, d)` into `buf_b`, then compute `cos(buf_a, buf_b)`. Average over
/// all entries. Entries with zero-length observations or zero-norm latents
/// contribute 0 to the average (defensive — diagnostic must not produce NaN).
#[allow(clippy::type_complexity)] // The pair-of-pairs type is inherent to the API.
#[inline]
fn mean_cosine_pair<E>(
    encoder: &E,
    pair_of_pairs: &[((&[f32], &[f32]), (&[f32], &[f32]))],
    buf_a: &mut [f32],
    buf_b: &mut [f32],
) -> f32
where
    E: Fn(&[f32], &[f32], &mut [f32]),
{
    if buf_a.is_empty() || buf_b.is_empty() || buf_a.len() != buf_b.len() {
        return 0.0;
    }
    let dim = buf_a.len();
    let mut sum_cos: f32 = 0.0;
    let mut count: u32 = 0;
    for &((a, b), (c, d)) in pair_of_pairs {
        if a.is_empty() || b.is_empty() || c.is_empty() || d.is_empty() {
            continue;
        }
        encoder(a, b, buf_a);
        encoder(c, d, buf_b);

        // Fused single pass: dot product + both norms.
        let mut dot: f32 = 0.0;
        let mut norm_a_sq: f32 = 0.0;
        let mut norm_b_sq: f32 = 0.0;
        for i in 0..dim {
            // SAFETY: i < dim == buf_a.len() == buf_b.len().
            let va = unsafe { *buf_a.get_unchecked(i) };
            let vb = unsafe { *buf_b.get_unchecked(i) };
            dot += va * vb;
            norm_a_sq += va * va;
            norm_b_sq += vb * vb;
        }
        let cos = if norm_a_sq < EPS || norm_b_sq < EPS {
            0.0 // zero-norm latent — undefined cos, treat as 0
        } else {
            // Defensive clamp: floating-point rounding can push dot/(|a|*|b|)
            // slightly outside [-1, 1] for nearly-parallel vectors.
            let c = dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt());
            c.clamp(-1.0, 1.0)
        };
        sum_cos += cos;
        count += 1;
    }
    if count > 0 {
        sum_cos / count as f32
    } else {
        0.0
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::type_complexity)] // Test bindings use the verbose pair-of-pairs type for readability.
mod tests {
    use super::*;

    // ─── Test encoders ────────────────────────────────────────────────────

    /// Build an encoder `E(x, x') = clean(x, x') + c * confounder(x)`.
    ///
    /// - `clean(x, x')[i] = (x'[i] - x[i]) - mean(x'[i] - x[i])`
    ///   (mean-subtracted displacement — invariant to constant shifts, zero
    ///   on identical inputs).
    /// - `confounder(x)[i] = x[i % (d/2)]` (reads the first half of x — a
    ///   context-correlated signal that the action latent should NOT carry).
    ///
    /// At `c = 0`: pure clean encoder (R_0 ≈ 0, R_shift ≈ 0, leakage < 0).
    /// At `c > 0`: confounder leaks (R_0 > 0, R_shift > 0, leakage rises).
    ///
    /// The encoder writes into `out` (length d).
    fn make_encoder(d: usize, c: f32) -> impl Fn(&[f32], &[f32], &mut [f32]) {
        move |a: &[f32], b: &[f32], out: &mut [f32]| {
            debug_assert_eq!(out.len(), d);
            let half = d / 2;
            // Pass 1: compute clean (mean-subtracted displacement).
            let mut mean: f32 = 0.0;
            for i in 0..d {
                out[i] = b[i] - a[i];
                mean += out[i];
            }
            mean /= d as f32;
            // Pass 2: subtract the mean, then add the confounder contribution.
            for i in 0..d {
                out[i] -= mean;
                let confounder_val = if half > 0 { a[i % half] } else { 0.0 };
                out[i] += c * confounder_val;
            }
        }
    }

    // ─── 2.1 — Identity / edge-case tests ─────────────────────────────────

    #[test]
    fn t2_1_empty_inputs_return_default() {
        let mut scratch = AuditScratch::new(4);
        let enc = |_: &[f32], _: &[f32], out: &mut [f32]| {
            for v in out.iter_mut() {
                *v = 1.0;
            }
        };
        let audit = audit_confounders(&enc, &[], &[], &[], &[], &[], &mut scratch);
        // Empty categories → all-zero defaults (D falls back to EPS).
        assert_eq!(audit.zero_transition_response, 0.0);
        assert_eq!(audit.shift_invariance_response, 0.0);
        assert_eq!(audit.shortcut_leakage, 0.0);
        assert!(audit.normalization_denominator <= EPS * 2.0);
    }

    #[test]
    fn t2_1b_zero_dim_scratch_returns_default() {
        let mut scratch = AuditScratch::new(0);
        let enc = |_: &[f32], _: &[f32], _out: &mut [f32]| {};
        let x = [1.0_f32];
        let ordinary: &[(&[f32], &[f32])] = &[(&x, &x)];
        let audit = audit_confounders(&enc, &[], &[], ordinary, &[], &[], &mut scratch);
        // With latent_dim = 0 the buffers are empty; the audit returns the
        // default + the EPS fallback for D (no encoders ran to produce real
        // norms). All three diagnostic fields stay at 0.0; only the
        // denominator differs from the struct default.
        assert_eq!(audit.zero_transition_response, 0.0);
        assert_eq!(audit.shift_invariance_response, 0.0);
        assert_eq!(audit.shortcut_leakage, 0.0);
        assert!(audit.normalization_denominator <= EPS * 2.0);
    }

    // ─── 2.2 — Clean encoder (c=0) properties ─────────────────────────────

    #[test]
    fn t2_2_clean_encoder_passes_all_three_diagnostics() {
        let d = 8;
        let c = 0.0;
        let enc = make_encoder(d, c);
        let mut scratch = AuditScratch::new(d);

        let (x_a, x_b, xp_a, xp_b, xp_a_diff, shifted_a, shifted_b) = fixture_pairs();

        // Zero-transition pairs.
        let zero_pairs: &[(&[f32], &[f32])] = &[(&x_a, &x_a), (&x_b, &x_b)];
        // Shift pairs (constant offset — a nuisance transform).
        let shift_pairs: &[(&[f32], &[f32])] = &[(&x_a, &shifted_a), (&x_b, &shifted_b)];
        // Ordinary pairs (typical transitions with non-trivial displacement).
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x_a, &xp_a), (&x_b, &xp_b)];
        // Same-action / diff-context: same displacement direction, different x.
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_b, &xp_b))];
        // Diff-action / same-context: opposite displacement direction, same x.
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_a, &xp_a_diff))];

        let audit = audit_confounders(
            &enc,
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action,
            diff_action,
            &mut scratch,
        );

        // c=0 ⇒ clean encoder.
        // R_0 ≈ 0 (identical inputs → mean-subtracted displacement is 0).
        assert!(
            audit.zero_transition_response < 1.0e-5,
            "R_0 = {} should be near-zero for clean encoder (c=0)",
            audit.zero_transition_response
        );
        // R_shift ≈ 0 (constant offset → mean-subtracted displacement is 0).
        assert!(
            audit.shift_invariance_response < 1.0e-5,
            "R_shift = {} should be near-zero for clean encoder (c=0)",
            audit.shift_invariance_response
        );
        // shortcut_leakage < 0 (same-action pairs have higher cos than diff-action).
        assert!(
            audit.shortcut_leakage < 0.0,
            "shortcut_leakage = {} should be < 0 for clean encoder (c=0); action must dominate",
            audit.shortcut_leakage
        );
        // Sanity: the denominator is non-trivial (encoder produces real latents
        // on ordinary transitions).
        assert!(
            audit.normalization_denominator > 1.0e-3,
            "D = {} should be a real value, not the EPS fallback",
            audit.normalization_denominator
        );
    }

    // ─── Shared test fixture (non-constant displacement) ────────────────
    //
    // The clean encoder mean-subtracts the displacement, so a CONSTANT
    // displacement produces the zero vector (cos undefined → treated as 0).
    // We use a ramp whose mean is zero so the encoder output is non-zero:
    //   disp_same  = [-0.28, -0.20, -0.12, -0.04, +0.04, +0.12, +0.20, +0.28]
    //   disp_diff  = opposite sign (different action direction)
    const DISP_SAME: [f32; 8] = [-0.28, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.28];
    const DISP_DIFF: [f32; 8] = [0.28, 0.20, 0.12, 0.04, -0.04, -0.12, -0.20, -0.28];

    /// Build the standard test-fixture observation pairs.
    fn fixture_pairs() -> (
        [f32; 8],
        [f32; 8],
        [f32; 8],
        [f32; 8],
        [f32; 8],
        [f32; 8],
        [f32; 8],
    ) {
        let x_a = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let x_b = [1.0_f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];
        let mut xp_a = [0.0_f32; 8]; // x_a + DISP_SAME
        let mut xp_b = [0.0_f32; 8]; // x_b + DISP_SAME
        let mut xp_a_diff = [0.0_f32; 8]; // x_a + DISP_DIFF
        let shifted_a = [
            x_a[0] + 10.0,
            x_a[1] + 10.0,
            x_a[2] + 10.0,
            x_a[3] + 10.0,
            x_a[4] + 10.0,
            x_a[5] + 10.0,
            x_a[6] + 10.0,
            x_a[7] + 10.0,
        ];
        let shifted_b = [
            x_b[0] + 5.0,
            x_b[1] + 5.0,
            x_b[2] + 5.0,
            x_b[3] + 5.0,
            x_b[4] + 5.0,
            x_b[5] + 5.0,
            x_b[6] + 5.0,
            x_b[7] + 5.0,
        ];
        for i in 0..8 {
            xp_a[i] = x_a[i] + DISP_SAME[i];
            xp_b[i] = x_b[i] + DISP_SAME[i];
            xp_a_diff[i] = x_a[i] + DISP_DIFF[i];
        }
        (x_a, x_b, xp_a, xp_b, xp_a_diff, shifted_a, shifted_b)
    }

    // ─── 2.3 — Confounded encoder (c>0) detected ──────────────────────────

    #[test]
    fn t2_3_confounded_encoder_detected_by_all_three_diagnostics() {
        let d = 8;
        let c = 2.0; // Large confounder coefficient.
        let enc = make_encoder(d, c);
        let mut scratch = AuditScratch::new(d);

        let (x_a, x_b, xp_a, xp_b, xp_a_diff, shifted_a, _) = fixture_pairs();

        let zero_pairs: &[(&[f32], &[f32])] = &[(&x_a, &x_a)];
        let shift_pairs: &[(&[f32], &[f32])] = &[(&x_a, &shifted_a)];
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x_a, &xp_a), (&x_b, &xp_b)];
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_b, &xp_b))];
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_a, &xp_a_diff))];

        let audit = audit_confounders(
            &enc,
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action,
            diff_action,
            &mut scratch,
        );

        // c > 0 ⇒ confounder leaks. At least R_0 must catch it (it directly
        // measures the encoder's response to identical inputs, where the
        // clean signal is zero but the confounder is c * a[i % half] ≠ 0).
        assert!(
            audit.zero_transition_response > 0.1,
            "R_0 = {} should be > 0 for confounded encoder (c=2.0)",
            audit.zero_transition_response
        );
        // R_shift also catches it (same mechanism — shift doesn't change the
        // confounder output since confounder only reads x, not T(x)).
        assert!(
            audit.shift_invariance_response > 0.1,
            "R_shift = {} should be > 0 for confounded encoder (c=2.0)",
            audit.shift_invariance_response
        );
        // shortcut_leakage is closer to 0 (or positive) for confounded.
        // The clean baseline was strongly negative; with the confounder the
        // context signal inflates diff-action/same-context cos, narrowing
        // the gap or reversing it. We assert leakage has grown (less negative)
        // — the comparison-vs-clean baseline assertion is in t2_4.
        assert!(
            audit.shortcut_leakage > -0.5,
            "shortcut_leakage = {} should be less negative than clean baseline",
            audit.shortcut_leakage
        );
    }

    // ─── 2.4 — Confounder coefficient monotonicity ───────────────────────

    #[test]
    fn t2_4_diagnostics_monotonic_in_confounder_coefficient() {
        // As c grows, R_0 and R_shift must grow (more confounder = larger
        // latent response on no-op / nuisance inputs). shortcut_leakage
        // must be non-decreasing (cleaner → leakier).
        let d = 8;
        let (x_a, x_b, xp_a, xp_b, xp_a_diff, shifted_a, _) = fixture_pairs();

        let zero_pairs: &[(&[f32], &[f32])] = &[(&x_a, &x_a)];
        let shift_pairs: &[(&[f32], &[f32])] = &[(&x_a, &shifted_a)];
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x_a, &xp_a), (&x_b, &xp_b)];
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_b, &xp_b))];
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_a, &xp_a_diff))];

        let mut scratch = AuditScratch::new(d);
        let coefficients = [0.0_f32, 0.5, 1.0, 2.0, 5.0];
        let mut prev_r0 = f32::MIN;
        let mut prev_r_shift = f32::MIN;
        let mut prev_leakage = f32::MIN;

        for &c in &coefficients {
            let enc = make_encoder(d, c);
            let audit = audit_confounders(
                &enc,
                zero_pairs,
                shift_pairs,
                ordinary_pairs,
                same_action,
                diff_action,
                &mut scratch,
            );

            // R_0 must be non-decreasing in c.
            assert!(
                audit.zero_transition_response >= prev_r0 - 1.0e-5,
                "R_0 not monotone at c={}: {} < prev {}",
                c,
                audit.zero_transition_response,
                prev_r0
            );
            // R_shift must be non-decreasing in c.
            assert!(
                audit.shift_invariance_response >= prev_r_shift - 1.0e-5,
                "R_shift not monotone at c={}: {} < prev {}",
                c,
                audit.shift_invariance_response,
                prev_r_shift
            );
            // shortcut_leakage non-decreasing in c (clean → leaky).
            assert!(
                audit.shortcut_leakage >= prev_leakage - 1.0e-5,
                "shortcut_leakage not monotone at c={}: {} < prev {}",
                c,
                audit.shortcut_leakage,
                prev_leakage
            );

            prev_r0 = audit.zero_transition_response;
            prev_r_shift = audit.shift_invariance_response;
            prev_leakage = audit.shortcut_leakage;
        }

        // Sanity: the sweep actually exercised the range — at c=0 leakage
        // should be substantially more negative than at c=5.0.
        let clean_leak = {
            let enc = make_encoder(d, 0.0);
            audit_confounders(
                &enc,
                zero_pairs,
                shift_pairs,
                ordinary_pairs,
                same_action,
                diff_action,
                &mut scratch,
            )
            .shortcut_leakage
        };
        let dirty_leak = {
            let enc = make_encoder(d, 5.0);
            audit_confounders(
                &enc,
                zero_pairs,
                shift_pairs,
                ordinary_pairs,
                same_action,
                diff_action,
                &mut scratch,
            )
            .shortcut_leakage
        };
        assert!(
            dirty_leak - clean_leak > 0.1,
            "shortcut_leakage should grow with c: clean={}, dirty={}, gap={}",
            clean_leak,
            dirty_leak,
            dirty_leak - clean_leak
        );
    }

    // ─── 2.5 — Zero-norm / NaN safety ────────────────────────────────────

    #[test]
    fn t2_5_zero_norm_encoder_does_not_produce_nan() {
        let mut scratch = AuditScratch::new(4);
        // Encoder that always produces zero (degenerate — D falls back to EPS).
        let zero_enc = |_: &[f32], _: &[f32], out: &mut [f32]| {
            for v in out.iter_mut() {
                *v = 0.0;
            }
        };
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let xp = [1.5_f32, 2.5, 3.5, 4.5];
        let zero_pairs: &[(&[f32], &[f32])] = &[(&x, &x)];
        let shift_pairs: &[(&[f32], &[f32])] = &[(&x, &xp)];
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x, &xp)];
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[((&x, &xp), (&x, &xp))];
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[((&x, &xp), (&x, &xp))];
        let audit = audit_confounders(
            &zero_enc,
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action,
            diff_action,
            &mut scratch,
        );
        // No NaNs anywhere — even though cos(zero, zero) is mathematically
        // undefined (we treat it as 0 per the helper).
        assert!(audit.zero_transition_response.is_finite());
        assert!(audit.shift_invariance_response.is_finite());
        assert!(audit.shortcut_leakage.is_finite());
        assert!(audit.normalization_denominator.is_finite());
        assert!(
            audit.shortcut_leakage.abs() < 1.0e-5,
            "zero-norm latents → cos treated as 0 → leakage should be 0, got {}",
            audit.shortcut_leakage
        );
    }

    // ─── 2.6 — Cosine clamp prevents out-of-range ─────────────────────────

    #[test]
    fn t2_6_parallel_vectors_cosine_is_one() {
        let d = 4;
        let mut scratch = AuditScratch::new(d);
        // Encoder produces identical output regardless of input — same_action
        // and diff_action pairs all yield identical latents → cos = 1 in both
        // → shortcut_leakage = 1 - 1 = 0.
        let const_enc = |_: &[f32], _: &[f32], out: &mut [f32]| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i as f32 + 1.0) * 0.1; // some non-zero pattern
            }
        };
        let x = [0.0_f32; 4];
        let xp = [1.0_f32; 4];
        let zero_pairs: &[(&[f32], &[f32])] = &[(&x, &x)];
        let shift_pairs: &[(&[f32], &[f32])] = &[(&x, &xp)];
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x, &xp)];
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[((&x, &xp), (&x, &xp))];
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[((&x, &xp), (&x, &xp))];
        let audit = audit_confounders(
            &const_enc,
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action,
            diff_action,
            &mut scratch,
        );
        // cos = 1 in both → shortcut_leakage = 1 - 1 = 0 exactly.
        assert!(
            audit.shortcut_leakage.abs() < 1.0e-5,
            "shortcut_leakage should be ~0 when both pair-types produce cos=1, got {}",
            audit.shortcut_leakage
        );
    }

    // ─── 2.7 — AuditScratch reuse ─────────────────────────────────────────

    #[test]
    fn t2_7_scratch_reuse_across_calls() {
        let d = 4;
        let mut scratch = AuditScratch::new(d);
        let enc = |a: &[f32], b: &[f32], out: &mut [f32]| {
            for i in 0..out.len() {
                out[i] = b[i] - a[i];
            }
        };
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let xp = [1.5_f32, 2.5, 3.5, 4.5];
        let pairs: &[(&[f32], &[f32])] = &[(&x, &xp)];
        let empty_pos: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[];

        let audit1 = audit_confounders(
            &enc,
            pairs,
            pairs,
            pairs,
            empty_pos,
            empty_pos,
            &mut scratch,
        );
        let audit2 = audit_confounders(
            &enc,
            pairs,
            pairs,
            pairs,
            empty_pos,
            empty_pos,
            &mut scratch,
        );
        // Deterministic + reused scratch must produce identical results.
        assert_eq!(audit1, audit2);
    }

    // ─── 2.8 — AuditScratch::resize ──────────────────────────────────────

    #[test]
    fn t2_8_scratch_resize_changes_latent_dim() {
        let mut scratch = AuditScratch::new(4);
        assert_eq!(scratch.latent_dim(), 4);
        scratch.resize(16);
        assert_eq!(scratch.latent_dim(), 16);
        assert_eq!(scratch.latent_a.len(), 16);
        assert_eq!(scratch.latent_b.len(), 16);
        // Shrink in place (Vec::resize doesn't reduce capacity but does
        // reduce len).
        scratch.resize(2);
        assert_eq!(scratch.latent_dim(), 2);
    }

    // ─── 2.9 — Mismatched-dim observations skip defensively ───────────────

    #[test]
    fn t2_9_mismatched_obs_dims_skip_silently() {
        let d = 4;
        let mut scratch = AuditScratch::new(d);
        let enc = |a: &[f32], b: &[f32], out: &mut [f32]| {
            let n = out.len().min(a.len()).min(b.len());
            for i in 0..n {
                out[i] = b[i] - a[i];
            }
        };
        let good_a = [1.0_f32, 2.0, 3.0, 4.0];
        let good_b = [1.5_f32, 2.5, 3.5, 4.5];
        let empty: &[f32] = &[];
        let zero_pairs: &[(&[f32], &[f32])] = &[(&good_a, &good_a), (good_a.as_slice(), empty)];
        let shift_pairs: &[(&[f32], &[f32])] = &[(&good_a, &good_b)];
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&good_a, &good_b)];
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] = &[
            ((&good_a, &good_b), (&good_a, &good_b)),
            ((&good_a, &good_b), (good_a.as_slice(), empty)), // skipped
        ];
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&good_a, &good_b), (&good_a, &good_b))];
        // Must not panic; must produce finite results.
        let audit = audit_confounders(
            &enc,
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action,
            diff_action,
            &mut scratch,
        );
        assert!(audit.zero_transition_response.is_finite());
        assert!(audit.shortcut_leakage.is_finite());
    }

    // ─── 2.10 — Field defaults ───────────────────────────────────────────

    #[test]
    fn t2_10_latent_confounder_audit_default_is_all_zero() {
        let d = LatentConfounderAudit::default();
        assert_eq!(d.zero_transition_response, 0.0);
        assert_eq!(d.shift_invariance_response, 0.0);
        assert_eq!(d.shortcut_leakage, 0.0);
        assert_eq!(d.normalization_denominator, 0.0);
    }

    // ─── G4 — Alloc-free steady state (CountingAllocator) ─────────────────

    // `crate::alloc`'s tracking machinery is `#[cfg(debug_assertions)]` by
    // design — in release the binary installs the plain `System` allocator and
    // there is nothing to measure. So this test cannot merely be `#[ignore]`d
    // in release: its imports would not resolve, which is what broke
    // `cargo test --release -p katgpt-core --lib` outright (Issue 716).
    //
    // Deliberately NOT solved with release no-op stubs returning 0: a
    // zero-alloc assertion would then PASS vacuously, which is the Issue
    // 705/714 failure — an instrument that cannot fail is not passing.
    #[cfg(debug_assertions)]
    #[test]
    fn g4_audit_confounders_zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        // The CountingAllocator is process-wide and only installed under
        // debug_assertions via the test binary's #[global_allocator]
        // (lib.rs `static TEST_GLOBAL_ALLOC`). If for some reason it's NOT
        // installed (e.g. consumer binary without the test-time static), the
        // sentinel below detects it and skips the assertion.
        let d = 8;
        let c = 1.0;
        let enc = make_encoder(d, c);
        let mut scratch = AuditScratch::new(d);

        let (x_a, x_b, xp_a, xp_b, xp_a_diff, shifted_a, _) = fixture_pairs();
        let zero_pairs: &[(&[f32], &[f32])] = &[(&x_a, &x_a)];
        let shift_pairs: &[(&[f32], &[f32])] = &[(&x_a, &shifted_a)];
        let ordinary_pairs: &[(&[f32], &[f32])] = &[(&x_a, &xp_a), (&x_b, &xp_b)];
        let same_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_b, &xp_b))];
        let diff_action: &[((&[f32], &[f32]), (&[f32], &[f32]))] =
            &[((&x_a, &xp_a), (&x_a, &xp_a_diff))];

        // Sentinel: confirm the allocator is installed by allocating a known
        // amount, then checking the counter increased. If it didn't, the
        // binary has no TrackingAllocator installed — measurement is
        // impossible, so skip with a clear message.
        reset_alloc_stats();
        let _sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        if sent_count == 0 {
            eprintln!(
                "g4_audit_confounders_zero_alloc_steady_state: \
                 TrackingAllocator not installed in this binary — SKIPPED"
            );
            return;
        }
        drop(_sentinel);

        // Warmup: one untimed call to allocate any lazy runtime state
        // (none expected, but defensive).
        let _ = audit_confounders(
            &enc,
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action,
            diff_action,
            &mut scratch,
        );

        // Reset + measure 100 calls. Steady state MUST be zero allocations
        // (the scratch is pre-sized; audit_confounders only reads its
        // buffers in place).
        reset_alloc_stats();
        for _ in 0..100 {
            let _ = audit_confounders(
                &enc,
                zero_pairs,
                shift_pairs,
                ordinary_pairs,
                same_action,
                diff_action,
                &mut scratch,
            );
        }
        let (count, bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "audit_confounders must be alloc-free in steady state; \
             observed {count} allocations ({bytes} bytes) across 100 calls"
        );
    }
}
