//! Sigmoid-cut mean budget — the shared capped-mean allocation primitive
//! (Issue 956 T2b's bisection lifted to the leaf; consumed by the
//! speculative lattice reopen drive and the forward d2f κ remask arm).
//!
//! The problem shape: a pool of scored candidates must admit a TARGET
//! EXPECTED COUNT through per-candidate propensities `σ(γ_eff·(sᵢ − τ))`,
//! where the cut `τ` is solved so the propensities sum to the target. The
//! deterministic read is then the count of scores at or above `τ` — no RNG
//! (the workspace determinism law), the propensity read is a count.
//!
//! Two laws this primitive carries from measurement:
//!
//! 1. **Pool spacing** ([`pool_spaced_gamma`]): a fixed absolute slope is
//!    distribution-blind. On the T2b bench's ~60-candidate pools (spread
//!    0.11, gaps ~1e-3) a shipped γ = 20 absolute slope pushed the σ tail
//!    mass above every score — ZERO admits, byte-identical to the disabled
//!    posture (the loud-zero gate caught it). The repair: scale γ by
//!    `n/spread` so the σ transition width is `spread/(γ·n)` — one mean
//!    order-statistic gap at γ = 1, whatever the pool's score scale.
//!
//! 2. **Capped-mean bisection** ([`sigmoid_cut_for_target`]): `E(τ) =
//!    Σ σ(γ_eff·(sᵢ − τ))` is monotone decreasing in `τ` with
//!    `E(τ_lo) ≥ n ≥ target` and `E(τ_hi) ≈ 0 ≤ target` at the module's
//!    bounds, so the root exists for every target capped into `[0, n]`. 48
//!    bisection steps pin τ far past f32 resolution. Zero-alloc, no RNG.

use katgpt_types::simd::fast_sigmoid;

/// Core capped-mean bisection over a CALLER-SUPPLIED expected-count closure:
/// solve the cut `τ` where `expected_at(τ) = target`. `expected_at` must be
/// monotone decreasing in `τ`; the bracket `[s_min − 16/γ_eff, s_max +
/// 16/γ_eff]` reads `≥ n` and `≤ ~0` for the σ-cut shape. 48 bisection steps
/// pin τ far past f32 resolution. Zero-alloc, deterministic (no RNG).
pub fn bisect_sigmoid_cut(
    mut expected_at: impl FnMut(f32) -> f32,
    s_min: f32,
    s_max: f32,
    gamma_eff: f32,
    target: f32,
) -> f32 {
    let mut lo = s_min - 16.0 / gamma_eff;
    let mut hi = s_max + 16.0 / gamma_eff;
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if expected_at(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Slice convenience over [`bisect_sigmoid_cut`]: the cut where
/// `Σᵢ σ(γ_eff·(sᵢ − τ)) = target` for a score slice. `target` must arrive
/// capped into `[0, scores.len()]` (the caller's budget law). Pure
/// arithmetic — deterministic, zero-alloc.
pub fn sigmoid_cut_for_target(scores: &[f32], gamma_eff: f32, target: f32) -> f32 {
    let s_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s_min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    bisect_sigmoid_cut(
        |tau| {
            scores
                .iter()
                .map(|&s| fast_sigmoid(gamma_eff * (s - tau)))
                .sum()
        },
        s_min,
        s_max,
        gamma_eff,
        target,
    )
}

/// γ in units of pool spacing: the effective slope `γ·n/spread` gives a σ
/// transition width of `spread/(γ·n)` — one mean order-statistic gap at
/// γ = 1, independent of the pool's score scale. `spread <= 0` (an
/// all-equal pool) reads as one gap at the epsilon floor — the cut then
/// sits at the common score and the deterministic count read takes the
/// caller's cap, which is the honest degenerate posture.
pub fn pool_spaced_gamma(gamma: f32, n: usize, spread: f32) -> f32 {
    gamma * n as f32 / spread.max(f32::EPSILON)
}

/// The deterministic read: how many scores sit at or above the cut. With
/// γ_eff from [`pool_spaced_gamma`] this is the count the bisection targeted
/// (± the σ tail mass one gap wide), never an RNG draw.
pub fn count_at_or_above(scores: &[f32], cut: f32) -> usize {
    scores.iter().filter(|&&s| s >= cut).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_admits_the_target_count() {
        // 60 candidates on a realistic elimination-pool scale (spread 0.11),
        // the fixture class that strangled the absolute-slope version.
        let scores: Vec<f32> = (0..60).map(|i| 0.5 - 0.11 * (i as f32 / 59.0)).collect();
        let gamma_eff = pool_spaced_gamma(20.0, scores.len(), 0.11);
        for target in [1.0f32, 4.0, 16.0, 40.0, 59.0] {
            let cut = sigmoid_cut_for_target(&scores, gamma_eff, target);
            let n = count_at_or_above(&scores, cut);
            assert!(
                (n as f32 - target).abs() <= 1.5,
                "target {target}: cut admitted {n}"
            );
        }
    }

    #[test]
    fn pool_spacing_is_scale_free() {
        // The same pool rescaled ×100 must admit the same count — the law
        // the absolute slope violated.
        let base: Vec<f32> = (0..60).map(|i| 0.5 - 0.11 * (i as f32 / 59.0)).collect();
        let scaled: Vec<f32> = base.iter().map(|s| s * 100.0).collect();
        for (pool, spread) in [(&base, 0.11f32), (&scaled, 11.0f32)] {
            let gamma_eff = pool_spaced_gamma(20.0, pool.len(), spread);
            let cut = sigmoid_cut_for_target(pool, gamma_eff, 4.0);
            let n = count_at_or_above(pool, cut);
            assert_eq!(n, 4, "scale-free admit failed at spread {spread}");
        }
    }

    #[test]
    fn target_bounds_are_honored() {
        let scores = vec![0.4f32, 0.35, 0.3, 0.25];
        let gamma_eff = pool_spaced_gamma(8.0, 4, 0.15);
        // Target 0 → the cut sits above every score → zero admits.
        let cut_hi = sigmoid_cut_for_target(&scores, gamma_eff, 0.0);
        assert_eq!(count_at_or_above(&scores, cut_hi), 0);
        // Target n → the cut sits below every score → full pool.
        let cut_lo = sigmoid_cut_for_target(&scores, gamma_eff, 4.0);
        assert_eq!(count_at_or_above(&scores, cut_lo), 4);
    }

    #[test]
    fn degenerate_pool_takes_the_cap() {
        // All-equal scores: the cut converges to the common score; the
        // count read is the full pool (the caller's budget caps it).
        let scores = vec![0.5f32; 8];
        let gamma_eff = pool_spaced_gamma(20.0, 8, 0.0);
        let cut = sigmoid_cut_for_target(&scores, gamma_eff, 3.0);
        let n = count_at_or_above(&scores, cut);
        assert!(n >= 6, "degenerate pool must read at the plateau, got {n}");
    }

    #[test]
    fn deterministic_replay() {
        let scores: Vec<f32> = (0..32)
            .map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 1000.0)
            .collect();
        let gamma_eff = pool_spaced_gamma(12.0, scores.len(), 0.999);
        let a = sigmoid_cut_for_target(&scores, gamma_eff, 7.0);
        let b = sigmoid_cut_for_target(&scores, gamma_eff, 7.0);
        assert_eq!(a.to_bits(), b.to_bits(), "the cut must be deterministic");
    }
}
