//! Lepski agreement cascade — the `adaptive_k` family's third arm (Issue 1229,
//! Research 681, arXiv:2611.22198 Banerjee & Sarkar).
//!
//! A per-query **selection layer**, separable from the estimator: given the
//! estimates `f̂_k(x)` of ONE query over an ascending geometric grid of nested
//! neighbourhood sizes (prefixes of one top-K retrieval — never a retrieval
//! per candidate), keep the largest `k` whose estimate still agrees with every
//! smaller-grid estimate within a computable noise scale. Bias grows with `k`,
//! the stochastic scale decays with `k`; the first disagreement is the bias
//! surfacing, and the scan stops just before it.
//!
//! Family placement (substrate-first): the variance-sigmoid heuristic
//! (`katgpt-attn` `compute_adaptive_k`, Plan 256) consumes dispersion of one
//! score vector; the Lemma-2 closed form (`compute_derived_k`, Issue 747)
//! consumes the level gap; THIS arm consumes cross-k estimate agreement and
//! carries the paper's oracle inequality `|f̂_ad − f₀| ≤ (A+2)·τ_{k*}` on a
//! `1 − 2η` event. Generic and attention-free: consumers (dash_attn router,
//! reflex retrieval depth, refine fan-out) wire this, never fork it.
//!
//! Constants are read from the paper's TeX source (arXiv src, v1), never from
//! the research note's prose:
//! - grid (§4.2, eq. 4.2): `k_j = 2^j·k₀`, `k₀ = ⌈8·log(n/η)⌉`,
//!   `J = ⌊log₂(n / (k₀·log n))⌋`;
//! - rule, practical form (§5, eq. 5.1): `k̂ = max{k : |f̂_{k'} − f̂_ℓ| ≤ A·τ̂_ℓ
//!   ∀ ℓ < k' ≤ k}` — every pair inside the accepted prefix, so the upward scan
//!   that stops at the FIRST rejection is exact, not an approximation;
//! - `A`: the theory needs `A ≥ 4` (Lemma 4.7; "A = 4.5 satisfies this"); the
//!   paper's own experiments run `A = 1`, outside the proved range. Serving
//!   default here is [`A_DEFAULT`] = 1 (Issue 1229 G1 cell 2 PASS); the theory
//!   constant [`A_THEORY`] = 4.5 was falsified as a serving constant (cell 1);
//! - practical scale (§5): `τ̂_ℓ = σ̂·√(log n / ℓ)`;
//! - noise scale (§5): `σ̂ = median_i |Y_i − Y_nn(i)| / (0.6745·√2)`, with
//!   consecutive differences in one dimension;
//! - full scale (eq. 4.4): `τ_k = 6·√κ_σ·max{δ̄²_k, (2/k)·log(|K_n|/η)}^½`,
//!   `κ_σ = max{1, 16σ²}`;
//! - critical radius (Lemma 4.3): `δ̄²_k = 2048(s+1)/k · W(k(L+1)V²/(16(s+1)))`,
//!   `V = ∏_{ℓ=0}^{L+1}(p_ℓ+1)`, `W` the principal Lambert-W branch.
//!
//! **Determinism (Issue-940 law):** the pass predicate is `|f̂_{k'} − f̂_ℓ| ≤
//! A·τ_ℓ` (ties PASS — `≤`, never `<`; a NaN difference REJECTS), the scan
//! order is ascending grid index with the inner `ℓ` loop ascending, and `k̂` is
//! the last passing index before the first rejection. Prefix means are a
//! sequential ascending-rank fold. `k̂` is a LOCAL per-query value; any surface
//! that commits it across the sync boundary must re-pin this order + tie rule
//! at its own site.
//!
//! Hot path ([`lepski_cascade`], [`prefix_means_into`], [`tau_hat_into`],
//! [`geometric_grid_into`]) is zero-alloc: caller-owned slices only. The σ̂
//! helpers reorder a caller-owned scratch buffer in place (selection, no sort
//! allocation).

/// Serving default for the agreement constant `A`: the paper's own §5
/// experiment constant (eq. 5.1). Issue 1229 G1 cell 2 (fresh seeds, pinned
/// before its run) PASSED every structure/property/discrete threshold at this
/// value; cell 1 FALSIFIED the theory constant [`A_THEORY`] as a serving
/// constant (Doppler structure 0.536 < 0.80; a [0,1] vote share never
/// rejects at n ≤ 4000).
pub const A_DEFAULT: f64 = 1.0;

/// Theory-admissible agreement constant (paper §4.5: `A ≥ 4` suffices for the
/// oracle inequality; "A = 4.5 satisfies this constraint"). Use it to READ the
/// `(A+2)·τ_{k*}` bound, not as a serving constant — too wide to see bias at
/// practical `n` (Issue 1229 G1 cell 1).
pub const A_THEORY: f64 = 4.5;

/// Gaussian MAD consistency constant: `median|N(0,1)| ≈ 0.6745`.
pub const MAD_GAUSSIAN: f64 = 0.6745;

/// `-1/e`, the principal Lambert-W branch point.
pub const LAMBERT_W_BRANCH_POINT: f64 = -0.367_879_441_171_442_33;

/// Outcome of one cascade scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LepskiSelection {
    /// Grid index of the selected neighbourhood `k̂` (into the caller's grid).
    pub index: usize,
    /// First grid index whose agreement test rejected; `None` = the whole
    /// grid was accepted (the upper end is binding — the paper reports this
    /// for 52–88% of 1-D queries, so consumers should size the grid top).
    pub rejected_at: Option<usize>,
}

impl LepskiSelection {
    /// True when no rejection happened — `k̂` is the grid maximum.
    #[inline]
    pub fn upper_end_binding(&self) -> bool {
        self.rejected_at.is_none()
    }
}

/// The agreement cascade (paper eq. 5.1, practical form).
///
/// `estimates[j]` is `f̂_{k_j}(x)` and `tau[j]` is `τ_{k_j}` over an ascending
/// grid. Scans `j = 1, 2, …` upward; `j` is accepted iff `|f̂_{k_j} − f̂_{k_ℓ}| ≤
/// a·τ_{k_ℓ}` for every `ℓ < j`; stops at the first rejection. `k₀` is always
/// admissible, so a non-empty grid always yields a selection.
///
/// Returns `None` on an empty grid. Panics if the two slices differ in length
/// (a caller wiring error, never a data condition). `O(J²)` with `J ≈ log n`.
pub fn lepski_cascade(estimates: &[f64], tau: &[f64], a: f64) -> Option<LepskiSelection> {
    assert_eq!(
        estimates.len(),
        tau.len(),
        "lepski_cascade: estimates and tau must share the grid"
    );
    if estimates.is_empty() {
        return None;
    }
    for j in 1..estimates.len() {
        let fj = estimates[j];
        for l in 0..j {
            // Pass iff `d ≤ bound`; an incomparable pair (NaN) rejects.
            let d = (fj - estimates[l]).abs();
            let passes = matches!(
                d.partial_cmp(&(a * tau[l])),
                Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
            );
            if !passes {
                return Some(LepskiSelection {
                    index: j - 1,
                    rejected_at: Some(j),
                });
            }
        }
    }
    Some(LepskiSelection {
        index: estimates.len() - 1,
        rejected_at: None,
    })
}

/// `k₀ = ⌈8·log(n/η)⌉` (paper eq. 4.2). `n ≥ 1`, `η ∈ (0, 1)`.
pub fn implementable_k0(n: usize, eta: f64) -> usize {
    let v = (8.0 * (n as f64 / eta).ln()).ceil();
    if v.is_finite() && v >= 1.0 {
        v as usize
    } else {
        1
    }
}

/// `J = ⌊log₂(n / (k₀·log n))⌋` (paper eq. 4.2), saturating at 0 when the
/// argument is below 1 (small `n`: a one-point grid).
pub fn implementable_j(n: usize, k0: usize) -> usize {
    let n_f = n as f64;
    let arg = n_f / (k0.max(1) as f64 * n_f.ln());
    if arg.is_finite() && arg >= 1.0 {
        arg.log2().floor() as usize
    } else {
        0
    }
}

/// Fill `out` with the geometric grid `k_j = 2^j·k₀` for every `k_j ≤ k_max`,
/// up to `out.len()` points. Returns the number written (0 when `k₀ = 0` or
/// `k₀ > k_max`).
pub fn geometric_grid_into(k0: usize, k_max: usize, out: &mut [usize]) -> usize {
    if k0 == 0 {
        return 0;
    }
    let mut k = k0;
    let mut written = 0;
    for slot in out.iter_mut() {
        if k > k_max {
            break;
        }
        *slot = k;
        written += 1;
        match k.checked_mul(2) {
            Some(next) => k = next,
            None => break,
        }
    }
    written
}

/// kNN-mean estimates over nested prefixes of ONE ranked retrieval:
/// `out[j] = mean(responses_by_rank[..grid[j]])`, one sequential running sum
/// in ascending rank (the Issue-940 fold order).
///
/// `grid` must be ascending. Stops at the first grid size that exceeds the
/// retrieval length or `out.len()`; returns the number of estimates written.
pub fn prefix_means_into(responses_by_rank: &[f64], grid: &[usize], out: &mut [f64]) -> usize {
    let mut sum = 0.0_f64;
    let mut consumed = 0usize;
    let mut written = 0usize;
    for (&k, slot) in grid.iter().zip(out.iter_mut()) {
        if k == 0 || k > responses_by_rank.len() || k < consumed {
            break;
        }
        for &y in &responses_by_rank[consumed..k] {
            sum += y;
        }
        consumed = k;
        *slot = sum / k as f64;
        written += 1;
    }
    written
}

/// Practical stochastic scale `τ̂_k = σ̂·√(log n / k)` (paper §5).
#[inline]
pub fn tau_hat(sigma_hat: f64, n: usize, k: usize) -> f64 {
    sigma_hat * ((n as f64).ln() / k as f64).sqrt()
}

/// [`tau_hat`] over a grid, written in place. Returns the number written.
pub fn tau_hat_into(sigma_hat: f64, n: usize, grid: &[usize], out: &mut [f64]) -> usize {
    let mut written = 0;
    for (&k, slot) in grid.iter().zip(out.iter_mut()) {
        *slot = tau_hat(sigma_hat, n, k);
        written += 1;
    }
    written
}

/// Median of `values` by in-place selection (reorders the slice; no
/// allocation). Even length = mean of the two middle order statistics.
/// `None` on an empty slice. Total order (`f64::total_cmp`) — deterministic
/// for any bit pattern, NaN sorts last.
fn median_in_place(values: &mut [f64]) -> Option<f64> {
    median_in_place_by(values, f64::total_cmp, |v| v)
}

/// [`median_in_place`] over any `Copy` element with a total order and a
/// widening map to `f64` — one selection routine for the f64 and f32 forms.
fn median_in_place_by<T: Copy>(
    values: &mut [T],
    cmp: fn(&T, &T) -> core::cmp::Ordering,
    widen: fn(T) -> f64,
) -> Option<f64> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mid = n / 2;
    let (lower, upper, _) = values.select_nth_unstable_by(mid, cmp);
    let upper = widen(*upper);
    if n % 2 == 1 {
        return Some(upper);
    }
    let lower_max = lower
        .iter()
        .copied()
        .max_by(cmp)
        .map(widen)
        .unwrap_or(upper);
    Some(0.5 * (lower_max + upper))
}

/// Bulk-MAD noise scale `σ̂ = median_i |v_i − median(v)| / 0.6745` over an
/// f32 vector, computed IN PLACE (the slice is permuted and then overwritten
/// with the absolute deviations; no allocation).
///
/// **Which σ̂ this is (derivation, not a copy of the NN form).** The paper's
/// NN-MAD differences a response against its covariate neighbour, so a smooth
/// `f₀` cancels and `Y_i − Y_nn(i) ~ N(0, 2σ²)` (hence `√2`). A RANKED score
/// vector has no covariate: under the null "no structure" the regression
/// function is one constant level and every value is `level + ε`, so the
/// spread about the median is the noise itself — `|v − med| ~ |N(0, σ²)|`,
/// median `0.6745σ`, no `√2`. Differencing RANKED values instead measures
/// order-statistic spacing (`O(σ/n)`), not σ — that collapses `τ̂` and the
/// cascade rejects on any trend (the degeneracy this form avoids).
/// The median is robust to a minority of planted outliers (the "needles"),
/// which is what makes it the null-noise scale rather than the total spread.
///
/// `None` on an empty slice. NaN sorts last under `f32::total_cmp`.
pub fn sigma_hat_bulk_mad_f32(values: &mut [f32]) -> Option<f64> {
    let widen = |v: f32| v as f64;
    let med = median_in_place_by(values, f32::total_cmp, widen)?;
    for v in values.iter_mut() {
        *v = ((*v as f64) - med).abs() as f32;
    }
    median_in_place_by(values, f32::total_cmp, widen).map(|m| m / MAD_GAUSSIAN)
}

/// Nearest-neighbour MAD noise scale from caller-computed absolute
/// differences `|Y_i − Y_nn(i)|` (paper §5):
/// `σ̂ = median / (0.6745·√2)`. `Y_i − Y_nn(i) ~ N(0, 2σ²)` under locally
/// constant `f₀`, hence the `√2`. Reorders `abs_diffs` in place.
pub fn sigma_hat_from_abs_diffs(abs_diffs: &mut [f64]) -> Option<f64> {
    median_in_place(abs_diffs).map(|m| m / (MAD_GAUSSIAN * core::f64::consts::SQRT_2))
}

/// One-dimensional σ̂: responses sorted by covariate, consecutive differences
/// (the paper's 1-D nearest-neighbour form). `scratch.len() ≥ ys.len() − 1`;
/// `None` when fewer than two responses or the scratch is short.
pub fn sigma_hat_consecutive(ys_sorted_by_x: &[f64], scratch: &mut [f64]) -> Option<f64> {
    let pairs = ys_sorted_by_x.len().checked_sub(1)?;
    if pairs == 0 || scratch.len() < pairs {
        return None;
    }
    let buf = &mut scratch[..pairs];
    for (slot, w) in buf.iter_mut().zip(ys_sorted_by_x.windows(2)) {
        *slot = (w[1] - w[0]).abs();
    }
    sigma_hat_from_abs_diffs(buf)
}

/// Discrete-label analog of σ̂: the nearest-neighbour **disagreement rate**.
///
/// Derivation (not a copy of the MAD form). For a 0/1 label with
/// `P(Y = 1 | x) = p(x)`, the label-noise variance is `σ² = p(1 − p)`. Under
/// locally constant `p` (the same premise the MAD form uses), a neighbour pair
/// is two independent draws, so `P(Y_i ≠ Y_nn(i)) = 2p(1 − p) = 2σ²`, i.e.
/// `E[(Y_i − Y_nn(i))²] = 2σ²` — the second-moment twin of the Gaussian
/// `N(0, 2σ²)` difference law. Hence `σ̂ = √(D / 2)` with `D` the observed
/// disagreement fraction. The MAD form cannot transfer: `|Y_i − Y_nn(i)| ∈
/// {0, 1}`, so its median is 0 whenever `D < ½` — a degenerate σ̂ = 0 that
/// would make every cascade test reject. The scale is on the vote-share
/// (`f̂_k ∈ [0, 1]`) axis the majority-vote estimator thresholds at ½.
///
/// `None` when `pairs == 0` or `disagreeing > pairs`.
pub fn sigma_hat_disagreement(disagreeing: usize, pairs: usize) -> Option<f64> {
    if pairs == 0 || disagreeing > pairs {
        return None;
    }
    Some((disagreeing as f64 / pairs as f64 / 2.0).sqrt())
}

/// Principal-branch Lambert W, `W₀(z)`: the `w ≥ −1` solving `w·eʷ = z`.
///
/// Halley iteration on `f(w) = w − z·e^{−w}` (overflow-free for large `z`,
/// where plain fixed-point converges slowly and `w·eʷ` overflows). Starting
/// guesses: branch-point series `−1 + p − p²/3 + 11p³/72`, `p = √(2(ez + 1))`
/// near `−1/e`; `ln(1 + z)` on the middle range; `L₁ − L₂ + L₂/L₁` (`L₁ = ln z`,
/// `L₂ = ln L₁`) above `e`. Converges to `4ε·max(1, |w|)` in a handful of
/// steps; capped at 32.
///
/// `NaN` for `z < −1/e` or NaN input; `W(−1/e) = −1`; `W(+∞) = +∞`.
pub fn lambert_w0(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z <= LAMBERT_W_BRANCH_POINT {
        // Inputs within float noise of the branch point map onto it.
        return if LAMBERT_W_BRANCH_POINT - z <= 4.0 * f64::EPSILON {
            -1.0
        } else {
            f64::NAN
        };
    }
    if z == 0.0 {
        return z;
    }
    if z == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut w = if z < -0.25 {
        let p = (2.0 * (core::f64::consts::E * z + 1.0)).max(0.0).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if z <= core::f64::consts::E {
        z.ln_1p()
    } else {
        let l1 = z.ln();
        let l2 = l1.ln();
        l1 - l2 + l2 / l1
    };
    for _ in 0..32 {
        let ze = z * (-w).exp();
        let f = w - ze;
        let fp = 1.0 + ze;
        let fpp = -ze;
        let denom = fp - f * fpp / (2.0 * fp);
        if denom == 0.0 || !denom.is_finite() {
            break;
        }
        let dw = f / denom;
        w -= dw;
        if dw.abs() <= 4.0 * f64::EPSILON * w.abs().max(1.0) {
            break;
        }
    }
    w
}

/// `V = ∏_{ℓ=0}^{L+1}(p_ℓ + 1)` over the full width vector `p₀ = d, p₁ … p_L,
/// p_{L+1} = 1` (paper Lemma 4.3).
pub fn network_volume_v(widths: &[usize]) -> f64 {
    widths
        .iter()
        .fold(1.0_f64, |acc, &p| acc * (p as f64 + 1.0))
}

/// Entropy-derived critical radius (paper Lemma 4.3):
/// `δ̄²_k = 2048(s+1)/k · W(k(L+1)V² / (16(s+1)))` — computable from the
/// architecture alone (`s` nonzero parameters, `L` hidden layers, `V` from
/// [`network_volume_v`]). Turns a measured K-floor into a derived one.
pub fn critical_radius_sq(k: usize, s: usize, hidden_layers: usize, v: f64) -> f64 {
    let s1 = s as f64 + 1.0;
    let k_f = k as f64;
    2048.0 * s1 / k_f * lambert_w0(k_f * (hidden_layers as f64 + 1.0) * v * v / (16.0 * s1))
}

/// `κ_σ = max{1, 16σ²}` (paper Prop. 3.3).
#[inline]
pub fn kappa_sigma(sigma: f64) -> f64 {
    (16.0 * sigma * sigma).max(1.0)
}

/// Full stochastic scale (paper eq. 4.4):
/// `τ_k = 6·√κ_σ · max{δ̄²_k, (2/k)·log(|K_n|/η)}^½`. For when the architecture
/// constants are held; serving lanes use [`tau_hat`].
pub fn tau_full(sigma: f64, delta_bar_sq: f64, k: usize, grid_len: usize, eta: f64) -> f64 {
    let union = 2.0 / k as f64 * (grid_len as f64 / eta).ln();
    6.0 * kappa_sigma(sigma).sqrt() * delta_bar_sq.max(union).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_accepts_whole_grid_when_estimates_agree() {
        let est = [1.0, 1.01, 0.99, 1.0];
        let tau = [0.1, 0.07, 0.05, 0.035];
        let sel = lepski_cascade(&est, &tau, A_THEORY).unwrap();
        assert_eq!(sel.index, 3);
        assert!(sel.upper_end_binding());
    }

    #[test]
    fn cascade_stops_at_first_rejection_even_if_later_agrees() {
        // j=2 disagrees with ℓ=0; j=3 would agree again — never reached.
        let est = [0.0, 0.1, 5.0, 0.0];
        let tau = [0.1; 4];
        let sel = lepski_cascade(&est, &tau, 4.5).unwrap();
        assert_eq!(
            sel,
            LepskiSelection {
                index: 1,
                rejected_at: Some(2)
            }
        );
    }

    #[test]
    fn cascade_checks_every_smaller_ell_with_its_own_tau() {
        // j=1 passes ℓ=0 (0.4 ≤ 4.5·0.1); j=2 agrees with ℓ=1 (wide tau)
        // but not ℓ=0 (0.6 > 0.45) — the narrow ℓ=0 scale must still bind.
        let est = [0.0, 0.4, 0.6];
        let tau = [0.1, 1.0, 1.0];
        let sel = lepski_cascade(&est, &tau, 4.5).unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(sel.rejected_at, Some(2));
    }

    #[test]
    fn cascade_tie_passes_and_nan_rejects() {
        let tie = lepski_cascade(&[0.0, 0.45], &[0.1, 0.1], 4.5).unwrap();
        assert!(tie.upper_end_binding(), "|d| == A·τ must pass (≤, never <)");
        let nan = lepski_cascade(&[0.0, f64::NAN], &[0.1, 0.1], 4.5).unwrap();
        assert_eq!(nan.index, 0);
        assert_eq!(lepski_cascade(&[], &[], 4.5), None);
    }

    #[test]
    #[should_panic(expected = "share the grid")]
    fn cascade_refuses_mismatched_lengths() {
        let _ = lepski_cascade(&[0.0, 1.0], &[0.1], 4.5);
    }

    #[test]
    fn grid_follows_eq_4_2() {
        // n = 2000, η = 0.05 → k0 = ⌈8·ln(40000)⌉ = ⌈84.77⌉ = 85.
        let k0 = implementable_k0(2000, 0.05);
        assert_eq!(k0, 85);
        // J = ⌊log2(2000 / (85·ln 2000))⌋ = ⌊log2(3.096)⌋ = 1.
        assert_eq!(implementable_j(2000, k0), 1);
        assert_eq!(implementable_j(10, 50), 0);
        let mut g = [0usize; 16];
        let n = geometric_grid_into(4, 512, &mut g);
        assert_eq!(&g[..n], &[4, 8, 16, 32, 64, 128, 256, 512]);
        assert_eq!(geometric_grid_into(0, 512, &mut g), 0);
        assert_eq!(geometric_grid_into(600, 512, &mut g), 0);
        let mut short = [0usize; 3];
        assert_eq!(geometric_grid_into(4, 512, &mut short), 3);
    }

    #[test]
    fn prefix_means_are_nested_running_means() {
        let ys = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let grid = [1, 2, 4, 8, 16];
        let mut out = [0.0; 5];
        let n = prefix_means_into(&ys, &grid, &mut out);
        assert_eq!(n, 4, "16 exceeds the retrieval — stop, never extrapolate");
        assert_eq!(&out[..4], &[1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn tau_hat_decays_by_sqrt2_per_doubling() {
        let t1 = tau_hat(0.1, 1000, 10);
        let t2 = tau_hat(0.1, 1000, 20);
        assert!((t1 / t2 - core::f64::consts::SQRT_2).abs() < 1e-12);
        let mut out = [0.0; 3];
        assert_eq!(tau_hat_into(0.1, 1000, &[10, 20, 40], &mut out), 3);
        assert_eq!(out[1], t2);
    }

    #[test]
    fn sigma_hat_mad_matches_closed_form() {
        // median of {1,2,3,4} = 2.5.
        let mut d = [4.0, 1.0, 3.0, 2.0];
        let s = sigma_hat_from_abs_diffs(&mut d).unwrap();
        assert!((s - 2.5 / (0.6745 * core::f64::consts::SQRT_2)).abs() < 1e-12);
        let mut odd = [5.0, 1.0, 3.0];
        let s = sigma_hat_from_abs_diffs(&mut odd).unwrap();
        assert!((s - 3.0 / (0.6745 * core::f64::consts::SQRT_2)).abs() < 1e-12);
        assert_eq!(sigma_hat_from_abs_diffs(&mut []), None);
        let mut scratch = [0.0; 3];
        let s = sigma_hat_consecutive(&[0.0, 1.0, 1.0, 3.0], &mut scratch).unwrap();
        // diffs {1, 0, 2} → median 1.
        assert!((s - 1.0 / (0.6745 * core::f64::consts::SQRT_2)).abs() < 1e-12);
        assert_eq!(sigma_hat_consecutive(&[1.0], &mut scratch), None);
        assert_eq!(
            sigma_hat_consecutive(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut scratch),
            None
        );
    }

    #[test]
    fn sigma_hat_disagreement_inverts_two_p_one_minus_p() {
        // p = 0.2 → σ = 0.4, D = 2·0.16 = 0.32.
        let s = sigma_hat_disagreement(32, 100).unwrap();
        assert!((s - 0.4).abs() < 1e-12);
        assert_eq!(sigma_hat_disagreement(1, 0), None);
        assert_eq!(sigma_hat_disagreement(5, 4), None);
    }

    #[test]
    fn lambert_w_known_answers() {
        let e = core::f64::consts::E;
        assert_eq!(lambert_w0(0.0), 0.0);
        assert!((lambert_w0(e) - 1.0).abs() < 1e-15);
        assert!((lambert_w0(1.0) - 0.567_143_290_409_783_8).abs() < 1e-15);
        assert!((lambert_w0(10.0) - 1.745_528_002_740_699_4).abs() < 1e-14);
        // Large-argument vector — the Halley path the δ̄² argument lands on.
        assert!((lambert_w0(1e6) - 11.383_358_086_140_053).abs() < 1e-12);
        assert_eq!(lambert_w0(LAMBERT_W_BRANCH_POINT), -1.0);
        assert!(lambert_w0(-0.4).is_nan());
        assert!(lambert_w0(f64::NAN).is_nan());
        assert_eq!(lambert_w0(f64::INFINITY), f64::INFINITY);
        // Near the branch point: w ≥ −1 and the residual closes.
        let z = LAMBERT_W_BRANCH_POINT + 1e-10;
        let w = lambert_w0(z);
        assert!(w > -1.0 && (w * w.exp() - z).abs() < 1e-14);
    }

    #[test]
    fn lambert_w_monotone_with_closed_residual_across_decades() {
        let mut prev = -1.0_f64;
        let mut z = -0.3678;
        while z < 1e300 {
            let w = lambert_w0(z);
            assert!(
                w > prev,
                "W must be strictly increasing: W({z}) = {w} ≤ {prev}"
            );
            // Residual in log form (w + ln w = ln z) for z > 0 avoids overflow.
            if z > 1.0 {
                assert!(((w + w.ln()) - z.ln()).abs() <= 1e-13 * z.ln().abs().max(1.0));
            } else {
                assert!((w * w.exp() - z).abs() <= 1e-14 * z.abs().max(1e-300));
            }
            prev = w;
            z = if z < 0.0 {
                z + 0.05
            } else if z < 1.0 {
                z + 0.1
            } else {
                z * 7.3
            };
        }
    }

    #[test]
    fn critical_radius_matches_formula_and_shrinks_in_k() {
        // A small explicit architecture: p = (1, 3, 1), L = 1, s = 10.
        let v = network_volume_v(&[1, 3, 1]);
        assert_eq!(v, 2.0 * 4.0 * 2.0);
        let d1 = critical_radius_sq(10, 10, 1, v);
        let expect = 2048.0 * 11.0 / 10.0 * lambert_w0(10.0 * 2.0 * v * v / (16.0 * 11.0));
        assert_eq!(d1, expect);
        let d2 = critical_radius_sq(20, 10, 1, v);
        assert!(d2 < d1, "δ̄² must decrease in k");
        // Lemma 4.3's log upper bound holds.
        let bound = 2048.0 * 11.0 / 10.0 * (1.0 + 10.0 * 2.0 * v * v / 11.0).ln();
        assert!(d1 <= bound);
    }

    #[test]
    fn tau_full_obeys_lemma_regularity() {
        // τ_{2k} ≤ τ_k and τ_{2k} ≥ τ_k/√2 (paper Lemma 4.5).
        let v = network_volume_v(&[1, 3, 1]);
        let (s, l, eta, grid_len) = (10, 1, 0.05, 6);
        let mut k = 10;
        while k <= 320 {
            let t1 = tau_full(0.1, critical_radius_sq(k, s, l, v), k, grid_len, eta);
            let t2 = tau_full(
                0.1,
                critical_radius_sq(2 * k, s, l, v),
                2 * k,
                grid_len,
                eta,
            );
            assert!(t2 <= t1 && t2 >= t1 / core::f64::consts::SQRT_2 - 1e-12);
            k *= 2;
        }
        assert_eq!(kappa_sigma(0.1), 1.0);
        assert_eq!(kappa_sigma(0.5), 4.0);
    }

    #[test]
    fn bulk_mad_f32_reads_null_noise_and_ignores_needles() {
        // Level 2.0 + a symmetric ramp: median 2.0, MAD = 0.25 (the 51st of
        // 101 sorted |i − 50|·0.01).
        let mut v: Vec<f32> = (0..101).map(|i| 2.0 + (i as f32 - 50.0) * 0.01).collect();
        // Needles at the extremes keep their side of the median, so neither
        // the median nor the MAD moves (the robustness claim).
        v[0] = -40.0;
        v[1] = -30.0;
        v[100] = 50.0;
        let s = sigma_hat_bulk_mad_f32(&mut v).unwrap();
        assert!((s - 0.25 / MAD_GAUSSIAN).abs() < 1e-5, "σ̂ = {s}");
        // Constant vector reads exactly zero; empty reads None.
        assert_eq!(sigma_hat_bulk_mad_f32(&mut [3.0_f32; 8]), Some(0.0));
        assert_eq!(sigma_hat_bulk_mad_f32(&mut []), None);
    }
}
