//! Theorem-backed ALiBi × α-entmax KV-eviction window
//! (Issue 747 P2, Research 549 — arXiv:2506.16640 App. E.2, "Prop 6").
//!
//! For an ALiBi-biased head (bias `−m_h·d`, `d = query_pos − key_pos ≥ 0`,
//! slope `m_h > 0`) whose RAW content logits are bounded
//! `z ∈ [z_min, z_max]`, entmax-1.5 attention mass beyond a hard distance
//! cutoff is **exactly** zero:
//!
//! ```text
//! d_max = ⌊(z_max − z_min + 1/(α−1)) / m_h + 1⌋      (α = 1.5 ⇒ 1/(α−1) = 2)
//! ```
//!
//! Any token at distance `d > d_max` is provably below the minimum possible
//! entmax threshold, so evicting its KV entry changes **nothing**: the
//! attention output is bit-identical (the G1 gate is bit-identity, not a
//! tolerance — the cleanest gate class in the Issue 747 batch).
//!
//! # Why bit-identity holds for OUR `entmax_1p5`
//!
//! The shipped `entmax_1p5_into` (Peters-style scan) satisfies, for every
//! input: threshold `τ_c ≥ s_max − 1` (single-support minimum — the top score
//! always survives and its probability is ≤ 1), and non-support entries are
//! exactly `{s ≤ τ_c}` (the passing prefix of the sorted scan is monotone:
//! once a step fails, weighted-averaging `t_{k+1} = (k·t_k + s_(k+1))/(k+1)`
//! forces every later step to fail too). Entries evicted by this window have
//! `s ≤ z_max − m_h·(d_max+1) < z_min − 1 ≤ s_max − 1 ≤ τ_c` — they are
//! non-support with probability EXACTLY 0.0, and removing exact-zero terms
//! from the index-ordered normalization sum (`x + 0.0 = x` bit-exactly)
//! leaves every kept probability bit-identical.
//!
//! The paper's margin `1/(α−1) = 2` is stated in the convention
//! `τ ≥ s_max − 1/(α−1)`; our normalized variant's floor is TIGHTER
//! (`τ_c ≥ s_max − 1`), so the theorem's window is conservative by ≥ 1
//! position for our implementation — the gate pins bit-identity at the
//! citable bound, not at our tighter one.
//!
//! # Scope assumptions (the theorem's setting)
//!
//! - **Causal self-attention**: the query attends to a token at distance 0
//!   (itself), so `s_max ≥ z_min`. Consumers applying this to a setting
//!   without a near token must tighten `z_min` by `−m_h·d_min`.
//! - **Raw logits bounded**: `[z_min, z_max]` must bound the CONTENT logits
//!   BEFORE the ALiBi bias. A Kamath-law estimate (`range ≈ 2σ̂√(2 ln n)`
//!   for near-Gaussian rows — the same law the `asentmax` estimator uses)
//!   is the modelless way to feed the bounds; use a generous σ̂ — the
//!   window grows only linearly in the assumed range.
//! - RoPE heads do NOT get this window (oscillatory re-entry — Prop E.3);
//!   eviction there must respect the frequency cutoff union (Issue 747 P4).
//!
//! Feature gate: `asentmax_schedule` (the Issue 747 family flag). Opt-in.

/// Hard attention cutoff `d_max` for an ALiBi-biased entmax-1.5 head —
/// paper Eq. 110 with `1/(α−1) = 2`.
///
/// Tokens at distance `d > d_max` receive EXACTLY zero attention; their KV
/// entries may be evicted with a bit-identical attention output (see module
/// docs for the proof sketch and the scope assumptions).
///
/// `z_min`/`z_max` bound the RAW content logits (pre-bias). `slope` is the
/// head's ALiBi slope `m_h > 0`.
///
/// Ill-formed input (non-finite, `slope ≤ 0`, `z_max < z_min`) returns
/// `usize::MAX` — keep everything, the safe direction.
#[inline]
pub fn alibi_entmax_window_1p5(z_min: f32, z_max: f32, slope: f32) -> usize {
    debug_assert!(slope.is_finite() && slope > 0.0, "ALiBi slope must be finite > 0");
    debug_assert!(
        z_min.is_finite() && z_max.is_finite() && z_max >= z_min,
        "raw-logit bounds must be finite and ordered"
    );
    if !(slope.is_finite() && slope > 0.0 && z_min.is_finite() && z_max.is_finite()
        && z_max >= z_min)
    {
        return usize::MAX;
    }
    // margin = 1/(α−1) = 2 for α = 1.5. Float→int `as` saturates at
    // usize::MAX on overflow (a degenerate z-range over a tiny slope keeps
    // everything rather than wrapping).
    ((z_max - z_min + 2.0) / slope + 1.0).floor() as usize
}

/// KV-retention predicate: keep the KV entry at `distance` iff it is inside
/// the eviction window `d_max` (from [`alibi_entmax_window_1p5`]).
///
/// This is the consumer wiring point — a KV cache trim loop becomes:
///
/// ```ignore
/// let d_max = alibi_entmax_window_1p5(z_min, z_max, head_slope);
/// retain: |pos| kv_within_window(query_pos - pos, d_max)
/// ```
#[inline]
pub fn kv_within_window(distance: usize, d_max: usize) -> bool {
    distance <= d_max
}

/// Fraction of an `n_tokens` KV row provably evictable at window `d_max`
/// (`1 − kept/n`, `kept = min(d_max + 1, n)`). 0.0 when nothing can go.
#[inline]
pub fn evicted_kv_fraction(n_tokens: usize, d_max: usize) -> f64 {
    if n_tokens == 0 {
        return 0.0;
    }
    let kept = (d_max as u64).saturating_add(1).min(n_tokens as u64) as f64;
    1.0 - kept / n_tokens as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_known_values() {
        // z-range 4, slope 2: ⌊(4+2)/2 + 1⌋ = 4
        assert_eq!(alibi_entmax_window_1p5(0.0, 4.0, 2.0), 4);
        // z-range 4, slope 0.5: ⌊6/0.5 + 1⌋ = 13
        assert_eq!(alibi_entmax_window_1p5(-2.0, 2.0, 0.5), 13);
        // z-range 13, slope 1: ⌊15 + 1⌋ = 16
        assert_eq!(alibi_entmax_window_1p5(-1.0, 12.0, 1.0), 16);
        // zero z-range still has the margin floor: ⌊2/m + 1⌋
        assert_eq!(alibi_entmax_window_1p5(3.0, 3.0, 0.5), 5);
    }

    #[test]
    fn window_monotone_in_range_and_slope() {
        let small = alibi_entmax_window_1p5(0.0, 2.0, 1.0);
        let large = alibi_entmax_window_1p5(0.0, 10.0, 1.0);
        assert!(large > small, "wider logit range ⇒ wider window");
        let steep = alibi_entmax_window_1p5(0.0, 4.0, 2.0);
        let shallow = alibi_entmax_window_1p5(0.0, 4.0, 0.02);
        assert!(shallow > steep, "smaller slope ⇒ wider window");
    }

    #[test]
    #[should_panic(expected = "ALiBi slope must be finite > 0")]
    fn ill_formed_slope_panics_in_debug() {
        let _ = alibi_entmax_window_1p5(0.0, 4.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "raw-logit bounds must be finite and ordered")]
    fn ill_formed_bounds_panics_in_debug() {
        let _ = alibi_entmax_window_1p5(5.0, 1.0, 1.0);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn ill_formed_input_keeps_everything_in_release() {
        assert_eq!(alibi_entmax_window_1p5(0.0, 4.0, 0.0), usize::MAX);
        assert_eq!(alibi_entmax_window_1p5(0.0, 4.0, -1.0), usize::MAX);
        assert_eq!(alibi_entmax_window_1p5(0.0, f32::NAN, 1.0), usize::MAX);
        assert_eq!(alibi_entmax_window_1p5(5.0, 1.0, 1.0), usize::MAX);
        assert_eq!(
            alibi_entmax_window_1p5(0.0, 4.0, f32::INFINITY),
            usize::MAX
        );
    }

    #[test]
    fn retention_predicate_and_fraction() {
        let d_max = 4;
        assert!(kv_within_window(0, d_max));
        assert!(kv_within_window(4, d_max));
        assert!(!kv_within_window(5, d_max));
        assert!(!kv_within_window(1_000_000, d_max));

        // n = 100, d_max = 4 → keep 5, evict 95%
        let frac = evicted_kv_fraction(100, 4);
        assert!((frac - 0.95).abs() < 1e-12);
        // window ≥ row ⇒ nothing evictable
        assert_eq!(evicted_kv_fraction(10, 10), 0.0);
        assert_eq!(evicted_kv_fraction(10, usize::MAX), 0.0);
        assert_eq!(evicted_kv_fraction(0, 4), 0.0);
    }

    #[test]
    fn degenerate_slope_saturates_not_wraps() {
        // z-range 40 over slope 1e-30: quotient overflows f32 → saturating
        // cast keeps everything instead of producing a garbage window.
        let d = alibi_entmax_window_1p5(-20.0, 20.0, 1e-30);
        assert_eq!(d, usize::MAX);
    }
}
