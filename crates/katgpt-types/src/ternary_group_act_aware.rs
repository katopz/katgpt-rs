//! Activation-aware group-scale fitting for the `Q2_0_g128` ternary
//! authoring path — Issue 886 P1, the **(a) weighted-fit** spelling
//! (llama.cpp imatrix class; Research 588 §2.1). Feature `act_aware_fit`.
//!
//! [`TernaryGroupWeights::quantize_from_f32`] fits each 128-weight group's
//! scale as the activation-BLIND `mean(|w|)`. The output error of a linear
//! layer is `E‖(W − Ŵ)x‖²`, which under the diagonal approximation of the
//! input second-moment matrix is `Σ_r Σ_j (w_rj − ŵ_rj)²·E[x_j²]` — errors on
//! channels whose activations are large cost more. This module weights the
//! scale fit by a caller-supplied **per-input-channel activation diagonal**
//! `h_j` (one value per column; typically `E[x_j²]` from katgpt-core's
//! `act_channel_moments`, or AWQ's `mean|x_j|`).
//!
//! # Boundary
//!
//! The diagonal is a plain `&[f32]` — katgpt-types does not depend on
//! katgpt-core (the collector lives there; the dependency points the other
//! way). The payload is **kernel-identical**: same bit-planes, same f16
//! group scale, same `threshold = 0.5·scale` and carry loop. Only the scale
//! VALUE changes, so every shipped matvec kernel consumes the result as-is.
//!
//! # The two fits
//!
//! - [`ActAwareScaleFit::WeightedMeanAbs`] — `s = Σ u_j|w_j| / Σ u_j` with
//!   `u_j = h_j / max_group(h)`. Closed form, one pass. **G3 anchor**: a
//!   uniform diagonal makes every `u_j` exactly `1.0` (`x / x == 1` in IEEE),
//!   so the sum, the divisor and therefore the payload are bit-identical to
//!   the mean-abs baseline — pinned by bytes in the tests below.
//! - [`ActAwareScaleFit::WeightedSearch`] — the imatrix-class fit: evaluate
//!   `E(s) = Σ u_j (w_j − s·q_j(s))²` through the ACTUAL carry loop at every
//!   candidate `s = s_wma × ACT_AWARE_SEARCH_GRID[k]` (the f16-rounded value
//!   the kernel applies), plus the weighted least-squares refit
//!   `Σ u w q / Σ u q²` on the best candidate's codes; keep the minimum.
//!   The `×1.0` candidate (the `WeightedMeanAbs` scale) is evaluated first
//!   and replaced only on a STRICT improvement, so on the weighted objective
//!   the search is never worse than `WeightedMeanAbs` group by group. The
//!   scale and the threshold move jointly (`threshold = 0.5·s`). NOT
//!   bit-identical at a uniform diagonal — at uniform `h` it is the
//!   activation-BLIND search, which is exactly the control that separates
//!   "search helps" from "the diagonal helps" (Bench 896 G1).
//!
//! Both are deterministic, allocation-free beyond the output container
//! (a `[f32; GROUP_SIZE]` stack buffer per group), and one-shot — the
//! closed-form counterpart of riir-train's `ZeroQatCalibrator`
//! (finite-difference GD on an injected loss, 100 steps × 2 forwards).
//!
//! # Honest prior (Issue 886 / Research 588 §2.6)
//!
//! At ternary, per-channel scale fitting is not the winning lever in the
//! literature (QuIP#, AQLM) — rotations are, and the Bonsai lane already
//! ships one. The carry loop is also tuned for the MEAN component of `x`
//! (it bounds `Σ Δw`), which the diagonal objective does not see. Expect a
//! small or negative delta here; the measurement, either sign, is the
//! deliverable (Bench 896).

use crate::GROUP_SIZE;
use crate::ternary_group::{TernaryGroupWeights, mean_abs_scale};
use half::f16;

/// How the per-group scale is fitted under an activation diagonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActAwareScaleFit {
    /// Diagonal-weighted mean-abs — closed form, G3 bit-identical at a
    /// uniform diagonal.
    WeightedMeanAbs,
    /// imatrix-class grid + least-squares refit on the diagonal-weighted
    /// reconstruction error through the carry loop.
    WeightedSearch,
}

/// Scale multipliers the [`ActAwareScaleFit::WeightedSearch`] fit tries
/// around the weighted mean-abs scale: `0.50, 0.55, …, 1.50` (21 points).
/// `1.0` sits at index 10 and is evaluated FIRST (the anchor).
pub const ACT_AWARE_SEARCH_GRID: [f32; 21] = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
    1.25, 1.30, 1.35, 1.40, 1.45, 1.50,
];

/// Index of the `1.0` anchor inside [`ACT_AWARE_SEARCH_GRID`].
const ANCHOR_IDX: usize = 10;

impl TernaryGroupWeights {
    /// Quantize with an **activation-aware** group-scale fit (Issue 886 P1).
    ///
    /// `diag[j]` is the activation diagonal of input channel `j` (length
    /// `cols`; finite and `≥ 0`, else panic — a calibration wiring bug, not
    /// data). A group whose diagonal slice is all zero carries no activation
    /// information and falls back to the activation-blind mean-abs scale.
    ///
    /// The payload format is unchanged — see the module docs.
    pub fn quantize_from_f32_act_aware(
        weights: &[f32],
        rows: usize,
        cols: usize,
        diag: &[f32],
        fit: ActAwareScaleFit,
    ) -> Self {
        assert_eq!(diag.len(), cols, "activation diagonal must have one entry per column");
        assert!(
            diag.iter().all(|h| h.is_finite() && *h >= 0.0),
            "activation diagonal must be finite and non-negative"
        );
        let mut u = [0.0f32; GROUP_SIZE];
        Self::quantize_with_group_scale(weights, rows, cols, |g_start, group| {
            let n = group.len();
            let h = &diag[g_start..g_start + n];
            if !normalized_weights(h, &mut u[..n]) {
                // No activation information in this group ⇒ the baseline.
                return mean_abs_scale(group);
            }
            let u = &u[..n];
            let s_wma = weighted_mean_abs(group, u);
            match fit {
                ActAwareScaleFit::WeightedMeanAbs => s_wma,
                ActAwareScaleFit::WeightedSearch => weighted_search(group, u, s_wma),
            }
        })
    }
}

/// `u_j = h_j / max(h)` into `out`. Returns `false` when the slice carries
/// no weight (max ≤ 0). Division, not multiply-by-reciprocal: `x / x` is
/// exactly `1.0`, `x · (1/x)` is not always — the G3 bit-identity rests on
/// this.
#[inline]
fn normalized_weights(h: &[f32], out: &mut [f32]) -> bool {
    let hmax = h.iter().copied().fold(0.0f32, f32::max);
    if hmax <= 0.0 {
        return false;
    }
    for (o, &v) in out.iter_mut().zip(h) {
        *o = v / hmax;
    }
    true
}

/// `Σ u|w| / Σ u` — written as the SAME iterator sum the baseline uses so a
/// unit `u` reproduces `mean_abs_scale` bit-for-bit. Falls back to the
/// unweighted mean-abs when every weighted channel holds a zero weight.
#[inline]
fn weighted_mean_abs(group: &[f32], u: &[f32]) -> f32 {
    let num: f32 = group.iter().zip(u).map(|(v, w)| v.abs() * w).sum();
    let den: f32 = u.iter().sum();
    if num > 0.0 { num / den } else { mean_abs_scale(group) }
}

/// Run the kernel's carry loop at the f16-exact scale `s` and return
/// `(Σ u (w − s q)², Σ u w q, Σ u q²)` — the weighted objective plus the
/// least-squares refit terms for these codes. Mirrors the write loop in
/// `quantize_with_group_scale` exactly (same `adjusted`, threshold, carry).
#[inline]
fn weighted_carry_error(group: &[f32], u: &[f32], s: f32) -> (f32, f32, f32) {
    let threshold = 0.5 * s;
    let mut carry = 0.0f32;
    let (mut err, mut num, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for (&val, &w) in group.iter().zip(u) {
        let adjusted = val + carry;
        let q: f32 = match adjusted {
            a if a > threshold => 1.0,
            a if a < -threshold => -1.0,
            _ => 0.0,
        };
        let e = val - q * s;
        err += w * e * e;
        num += w * val * q;
        den += w * q * q;
        carry = adjusted - q * s;
    }
    (err, num, den)
}

/// Round to the f16 value the kernel will apply (the carry loop and the
/// objective must see the stored scale, not the f32 ideal).
#[inline]
fn f16_exact(s: f32) -> f32 {
    f16::from_f32(s).to_f32()
}

/// Grid + least-squares-refit search on the weighted objective. Returns an
/// f16-exact scale (re-rounding it in the pipeline is the identity).
fn weighted_search(group: &[f32], u: &[f32], s_wma: f32) -> f32 {
    let mut best_s = f16_exact(s_wma * ACT_AWARE_SEARCH_GRID[ANCHOR_IDX]);
    let (mut best_err, mut best_num, mut best_den) = weighted_carry_error(group, u, best_s);
    for (k, &m) in ACT_AWARE_SEARCH_GRID.iter().enumerate() {
        if k == ANCHOR_IDX {
            continue;
        }
        let s = f16_exact(s_wma * m);
        if !(s > 0.0 && s.is_finite()) {
            continue;
        }
        let (err, num, den) = weighted_carry_error(group, u, s);
        if err < best_err {
            (best_s, best_err, best_num, best_den) = (s, err, num, den);
        }
    }
    // Least-squares refit on the winning codes; the codes may move under
    // the new threshold, so it is re-evaluated through the carry loop and
    // kept only on a strict improvement.
    if best_den > 0.0 {
        let s = f16_exact(best_num / best_den);
        if s > 0.0 && s.is_finite() {
            let (err, _, _) = weighted_carry_error(group, u, s);
            if err < best_err {
                best_s = s;
            }
        }
    }
    best_s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..rows * cols).map(|_| pseudo(&mut s) * 0.7).collect()
    }

    /// Verbatim transcription of the PRE-refactor quantizer loop (katgpt-rs
    /// `4cc5bd941`, `quantize_with_scale_rule` native arm) — the refactor
    /// and the G3 claim are both checked against this, not against the
    /// code they changed.
    fn legacy_quantize(w: &[f32], rows: usize, cols: usize) -> TernaryGroupWeights {
        let mut out = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let row_base = r * out.blocks64;
            let group_base = r * out.groups_per_row;
            for g in 0..out.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(cols);
                let group = &row[g_start..g_end];
                let abs_sum: f32 = group.iter().map(|v| v.abs()).sum();
                let scale = if abs_sum > 0.0 { abs_sum / group.len() as f32 } else { 1.0 };
                out.group_scale[group_base + g] = f16::from_f32(scale);
                let scale = out.group_scale[group_base + g].to_f32();
                let threshold = 0.5 * scale;
                let mut carry = 0.0f32;
                for (i, &val) in group.iter().enumerate() {
                    let adjusted = val + carry;
                    let q = match adjusted {
                        a if a > threshold => 1i8,
                        a if a < -threshold => -1i8,
                        _ => 0i8,
                    };
                    let col = g_start + i;
                    let idx = row_base + (col >> 6);
                    let mask = 1u64 << (col & 63);
                    out.pos_bits[idx] |= ((q == 1) as u64) * mask;
                    out.neg_bits[idx] |= ((q == -1) as u64) * mask;
                    carry = adjusted - (q as f32 * scale);
                }
            }
        }
        out
    }

    /// Canonical payload bytes: pos planes ‖ neg planes ‖ f16 scale bits, LE.
    fn payload_bytes(t: &TernaryGroupWeights) -> Vec<u8> {
        let mut b = Vec::new();
        for w in &t.pos_bits {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for w in &t.neg_bits {
            b.extend_from_slice(&w.to_le_bytes());
        }
        for s in &t.group_scale {
            b.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        b
    }

    /// FNV-1a 64 over the payload — a dependency-free pin for the fixture.
    fn fnv64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    // Includes a trailing partial group (cols % 128 != 0).
    const SHAPES: [(usize, usize); 3] = [(8, 256), (5, 300), (3, 128)];

    #[test]
    fn refactor_preserves_legacy_payload() {
        for (i, &(r, c)) in SHAPES.iter().enumerate() {
            let w = weights(r, c, 11 + i as u64);
            let new = TernaryGroupWeights::quantize_from_f32(&w, r, c);
            assert_eq!(payload_bytes(&new), payload_bytes(&legacy_quantize(&w, r, c)));
        }
    }

    /// G3: a uniform diagonal ⇒ `WeightedMeanAbs` payload bit-identical to
    /// the mean-abs baseline, for several uniform values (incl. ones whose
    /// reciprocal is inexact) and a trailing partial group.
    #[test]
    fn g3_uniform_diagonal_is_bit_identical() {
        for (i, &(r, c)) in SHAPES.iter().enumerate() {
            let w = weights(r, c, 101 + i as u64);
            let base = payload_bytes(&TernaryGroupWeights::quantize_from_f32(&w, r, c));
            for &hv in &[1.0f32, 3.0, 0.37, 1e-3, 7.0, 49.0] {
                let diag = vec![hv; c];
                let aa = TernaryGroupWeights::quantize_from_f32_act_aware(
                    &w,
                    r,
                    c,
                    &diag,
                    ActAwareScaleFit::WeightedMeanAbs,
                );
                assert_eq!(payload_bytes(&aa), base, "shape {r}x{c} h={hv}");
            }
            // An all-zero diagonal carries no information ⇒ the baseline too.
            let zero = vec![0.0f32; c];
            for fit in [ActAwareScaleFit::WeightedMeanAbs, ActAwareScaleFit::WeightedSearch] {
                let aa = TernaryGroupWeights::quantize_from_f32_act_aware(&w, r, c, &zero, fit);
                assert_eq!(payload_bytes(&aa), base, "zero diag {fit:?}");
            }
        }
    }

    /// The G3 fixture's payload digest, pinned by value (not only by the
    /// in-test reference) so a change to the shared pipeline that moves
    /// both arms together is still caught.
    #[test]
    fn g3_payload_digest_pinned() {
        let w = weights(8, 256, 101);
        let diag = vec![0.37f32; 256];
        let aa = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w,
            8,
            256,
            &diag,
            ActAwareScaleFit::WeightedMeanAbs,
        );
        assert_eq!(fnv64(&payload_bytes(&aa)), G3_DIGEST, "0x{:016x}", fnv64(&payload_bytes(&aa)));
        // …and the pre-refactor transcription lands on the same digest.
        assert_eq!(fnv64(&payload_bytes(&legacy_quantize(&w, 8, 256))), G3_DIGEST);
    }
    const G3_DIGEST: u64 = 0x2d77_e077_501d_6b6c;

    /// Known answer: a group whose diagonal is one-hot on channel k gets
    /// `s = |w_k|` under `WeightedMeanAbs`.
    #[test]
    fn one_hot_diagonal_selects_that_channel() {
        let c = 128;
        let mut w = vec![0.1f32; c];
        w[17] = -0.8;
        let mut diag = vec![0.0f32; c];
        diag[17] = 5.0;
        let t = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w,
            1,
            c,
            &diag,
            ActAwareScaleFit::WeightedMeanAbs,
        );
        assert_eq!(t.scale_at(0, 0), f16::from_f32(0.8).to_f32());
        assert_eq!(t.get(0, 17), -1);
    }

    /// The search never loses to its own anchor on the weighted objective
    /// (group by group), and it is deterministic.
    #[test]
    fn search_never_worse_than_anchor_on_weighted_objective() {
        let (r, c) = (6, 300);
        let w = weights(r, c, 7);
        let mut s = 99u64;
        let diag: Vec<f32> = (0..c)
            .map(|j| {
                let base = pseudo(&mut s).abs() + 0.05;
                if j % 37 == 0 { base * 400.0 } else { base }
            })
            .collect();
        let wma = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w,
            r,
            c,
            &diag,
            ActAwareScaleFit::WeightedMeanAbs,
        );
        let srch = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w,
            r,
            c,
            &diag,
            ActAwareScaleFit::WeightedSearch,
        );
        let again = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w,
            r,
            c,
            &diag,
            ActAwareScaleFit::WeightedSearch,
        );
        assert_eq!(payload_bytes(&srch), payload_bytes(&again), "deterministic");
        let obj = |t: &TernaryGroupWeights, row: usize, g: usize| -> f64 {
            let g0 = g * GROUP_SIZE;
            let g1 = (g0 + GROUP_SIZE).min(c);
            let hmax = diag[g0..g1].iter().copied().fold(0.0f32, f32::max);
            (g0..g1)
                .map(|j| {
                    let e = f64::from(w[row * c + j])
                        - f64::from(t.scale_at(row, g)) * f64::from(t.get(row, j));
                    f64::from(diag[j] / hmax) * e * e
                })
                .sum()
        };
        let mut strictly_better = 0;
        for row in 0..r {
            for g in 0..wma.groups_per_row {
                let (a, b) = (obj(&wma, row, g), obj(&srch, row, g));
                assert!(b <= a * (1.0 + 1e-5) + 1e-9, "row {row} g {g}: search {b} > anchor {a}");
                strictly_better += usize::from(b < a * (1.0 - 1e-4));
            }
        }
        assert!(strictly_better > 0, "search never moved off the anchor — inert");
    }

    #[test]
    #[should_panic(expected = "finite and non-negative")]
    fn negative_diagonal_panics() {
        let w = vec![0.5f32; 128];
        let mut d = vec![1.0f32; 128];
        d[3] = -1.0;
        let _ = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w,
            1,
            128,
            &d,
            ActAwareScaleFit::WeightedMeanAbs,
        );
    }
}
