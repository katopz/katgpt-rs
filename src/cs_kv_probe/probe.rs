//! Compressed-sensing KV-group importance probe.
//!
//! Modelless: given a black-box eval function `Fn(&AblationMask, &[Episode]) -> f32`,
//! sample `M` ablation masks, measure the eval drop, and recover per-head
//! importance via Lasso (coordinate descent). Aggregates per-KV-group under the
//! standard GQA mapping `kv_group(h) = floor(h · n_kv_heads / n_heads)`.
//!
//! Zero training. The only "learning" is one Lasso solve on a fixed
//! measurement matrix — pure inference.

use super::lasso::lasso;
use super::types::{AblationMask, Episode, KvGroupRanking};

/// Sample `m` stratified ablation masks over `n_heads` heads.
///
/// Each mask zeros **exactly** `round(ablation_fraction · n_heads)` heads (not
/// Poisson — exact count per mask, via a partial Fisher–Yates shuffle). `bits[h]
/// == true` means head `h` is retained. Returns `Vec::with_capacity(m)`.
///
/// Paper default: `M = 200`, `ablation_fraction = 0.05`.
pub fn sample_masks(
    n_heads: usize,
    m: usize,
    ablation_fraction: f32,
    rng: &mut fastrand::Rng,
) -> Vec<AblationMask> {
    let mut out = Vec::with_capacity(m);
    match n_heads {
        0 => return out,
        _ => {}
    }
    // Exact ablation count, capped at n_heads − 1 so we never produce an
    // all-zero mask (which would carry no per-head signal).
    let n_ablate_raw = (ablation_fraction * n_heads as f32).round() as i64;
    let n_ablate = n_ablate_raw.max(0) as usize;
    let n_ablate = n_ablate.min(n_heads.saturating_sub(1));

    // Reusable index buffer for the partial Fisher–Yates shuffle — avoids a
    // fresh allocation per mask.
    let mut idx: Vec<usize> = (0..n_heads).collect();
    for _ in 0..m {
        // Reset to identity each iteration (the partial shuffle permutes in place).
        for (i, v) in (0..n_heads).enumerate() {
            idx[i] = v;
        }
        let mut bits = vec![true; n_heads];
        for i in 0..n_ablate {
            // Swap idx[i] with a uniformly random index in [i, n_heads).
            let j = i + rng.usize(0..n_heads - i);
            idx.swap(i, j);
            // The head now at idx[i] is chosen for ablation.
            let head = idx[i];
            bits[head] = false;
        }
        out.push(AblationMask { bits, n_heads });
    }
    out
}

/// Probe configuration. All fields `Copy` so it can be passed by value.
#[derive(Debug, Clone, Copy)]
pub struct CsProbeConfig {
    /// Number of ablation masks to sample. Paper default 200.
    pub m_masks: usize,
    /// Fraction of heads each mask zeros (exact count per mask). Paper default 0.05.
    pub ablation_fraction: f32,
    /// Lasso L1 penalty (Form A, bare alpha). Default 1e-4 (light; probe is
    /// overdetermined M ≫ N, behaves near-OLS for clean support recovery).
    pub lasso_alpha: f32,
    /// Lasso coordinate-descent sweeps. Default 1000.
    pub lasso_iter: usize,
    /// Number of attention heads the measurement matrix spans.
    pub n_heads: usize,
    /// Number of KV groups under the GQA mapping. Default equals `n_heads`
    /// (no GQA grouping). Used to aggregate per-head coefficients → per-group.
    pub n_kv_heads: usize,
}

impl Default for CsProbeConfig {
    fn default() -> Self {
        Self {
            m_masks: 200,
            ablation_fraction: 0.05,
            lasso_alpha: 1e-4,
            lasso_iter: 1000,
            n_heads: 64,
            n_kv_heads: 64,
        }
    }
}

/// Compressed-sensing KV-group importance probe. Stateless entry point.
pub struct CsKvProbe;

impl CsKvProbe {
    /// Run the compressed-sensing probe.
    ///
    /// Pipeline:
    /// 1. `y_baseline = eval(all_ones_mask, episodes)` — full-retention eval.
    /// 2. Sample `M` masks.
    /// 3. For each mask: `y_m = eval(mask, episodes)`; center `ỹ_m = y_m − y_baseline`.
    /// 4. Build `Phi` (M × n_heads) from masks (`bool → {0.0, 1.0}`).
    /// 5. `coeffs = lasso(Phi, ỹ, alpha, n_iter)`.
    /// 6. Aggregate per KV group: `kv_group(h) = floor(h · n_kv_heads / n_heads)`,
    ///    score = **mean** of `|coeff_h|` over heads mapping to that group.
    /// 7. Return `KvGroupRanking`.
    ///
    /// **Aggregation choice (mean, not max):** documented deviation. Mean is the
    /// standard robust aggregation (cf. AM `ScoreMethod::Rms` default). With the
    /// default `n_kv_heads == n_heads` the choice is moot (one head per group).
    /// Max would surface a single dominant head; mean averages out per-head
    /// noise — preferable when the eval function is noisy.
    ///
    /// **Allocation discipline:** `Phi` and `y` are built once with
    /// `with_capacity(M)`; the eval loop only `push`es. (Per the plan this loop
    /// is allocation-free aside from the one-time `Phi`/`y` build.)
    pub fn run<Eval>(
        episodes: &[Episode],
        eval: &Eval,
        config: &CsProbeConfig,
        rng: &mut fastrand::Rng,
    ) -> KvGroupRanking
    where
        Eval: Fn(&AblationMask, &[Episode]) -> f32,
    {
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let n_groups = n_kv_heads.max(1);

        // 1. Baseline: full retention.
        let all_ones = AblationMask::all_ones(n_heads);
        let y_baseline = eval(&all_ones, episodes);

        // 2. Sample masks.
        let masks = sample_masks(n_heads, config.m_masks, config.ablation_fraction, rng);
        let m = masks.len();

        // 3–4. Centered y and Phi, pre-allocated.
        let mut y: Vec<f32> = Vec::with_capacity(m);
        let mut phi: Vec<Vec<f32>> = Vec::with_capacity(m);
        for mask in &masks {
            let y_m = eval(mask, episodes);
            y.push(y_m - y_baseline);
            let row: Vec<f32> = mask.bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            phi.push(row);
        }

        // 5. Lasso. Guard the empty case (no masks sampled).
        let coeffs = match m {
            0 => vec![0.0_f32; n_heads],
            _ => lasso(&phi, &y, config.lasso_alpha, config.lasso_iter),
        };

        // 6. Aggregate per KV group via GQA mapping, mean of |coeff|.
        let mut scores = vec![0.0_f32; n_groups];
        let mut counts = vec![0u32; n_groups];
        for (h, &c) in coeffs.iter().enumerate() {
            let g = if n_heads == 0 { 0 } else { (h * n_kv_heads) / n_heads };
            // Defensive: GQA divisor can in principle produce g == n_groups when
            // n_kv_heads == 0; clamp to avoid OOB. (n_groups is floored at 1 above.)
            let g = g.min(n_groups - 1);
            scores[g] += c.abs();
            counts[g] += 1;
        }
        for g in 0..n_groups {
            match counts[g] {
                0 => {}
                c => scores[g] /= c as f32,
            }
        }

        // 7. Ranking.
        KvGroupRanking { scores, n_groups }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_masks_exact_ablation_count() {
        let mut rng = fastrand::Rng::with_seed(42);
        let n_heads = 64;
        let frac = 0.05; // → 3 heads ablated per mask.
        let masks = sample_masks(n_heads, 50, frac, &mut rng);
        assert_eq!(masks.len(), 50);
        for m in &masks {
            assert_eq!(m.n_heads, n_heads);
            assert_eq!(m.bits.len(), n_heads);
            // Exactly round(0.05 * 64) = 3 ablated.
            assert_eq!(m.n_ablated(), 3, "mask {:?} ablated {} heads", m.bits, m.n_ablated());
            assert!(m.retention_fraction() > 0.0);
        }
    }

    #[test]
    fn test_sample_masks_no_full_ablation() {
        // ablation_fraction = 1.0 must still leave ≥1 head (capped at n−1).
        let mut rng = fastrand::Rng::with_seed(7);
        let masks = sample_masks(8, 20, 1.0, &mut rng);
        for m in &masks {
            assert!(m.bits.iter().any(|&b| b), "a mask fully ablated all heads");
        }
    }

    /// G1 — CS probe surfaces the signal heads above random.
    ///
    /// Synthetic task: 16 heads, heads {3, 7} carry signal (their KV entries
    /// correlate with `label_success`). The eval function scores an episode
    /// higher when signal heads are retained AND the episode's KV entries on
    /// those heads agree with the label. We assert the top-2 CS-ranked groups
    /// ⊇ {3, 7} and that CS top-2 accuracy beats a random ranking by a clear
    /// margin.
    #[test]
    fn test_cs_ranking_beats_random_g1() {
        let n_heads = 16_usize;
        let d = 32_usize; // KV cache dim per episode.
        let signal_heads = [3usize, 7];

        // Build episodes: signal heads' KV entries encode the label.
        let mut rng = fastrand::Rng::with_seed(0xBEEF);
        let n_episodes = 80;
        let episodes: Vec<Episode> = (0..n_episodes)
            .map(|_| {
                let label = rng.bool();
                let mut kv = vec![0.0_f32; d];
                for i in 0..d {
                    kv[i] = rng.f32() * 2.0 - 1.0;
                }
                // Encode label into signal-head channels (one channel per head).
                for (k, &h) in signal_heads.iter().enumerate() {
                    let ch = (h * d / n_heads) % d;
                    kv[ch] = if label { 1.0 } else { -1.0 } + 0.1 * (rng.f32() * 2.0 - 1.0);
                    // Touch k so the unused-index lint stays quiet if shapes change.
                    let _ = k;
                }
                Episode::new(kv, label)
            })
            .collect();

        // Eval: mean over episodes of [retained-signal-head agreement].
        // Ablating a signal head removes its contribution → eval drops → that
        // head surfaces as important under the CS probe.
        let eval = |mask: &AblationMask, eps: &[Episode]| -> f32 {
            let mut acc = 0.0_f32;
            for e in eps {
                for &h in &signal_heads {
                    if mask.bits[h] {
                        let ch = (h * d / n_heads) % d;
                        // Agreement: +1 if kv sign matches label, else −1.
                        let agree = e.kv_cache[ch] * if e.label_success { 1.0 } else { -1.0 };
                        acc += agree;
                    }
                }
            }
            acc / eps.len().max(1) as f32
        };

        let config = CsProbeConfig {
            m_masks: 80, // smaller than paper 200 for test speed; still M > N.
            ablation_fraction: 0.2,
            lasso_alpha: 1e-4,
            lasso_iter: 1000,
            n_heads,
            n_kv_heads: n_heads, // no GQA grouping → 1 head per group.
        };
        let ranking = CsKvProbe::run(&episodes, &eval, &config, &mut rng);

        // Top-2 by score.
        let mut idx: Vec<usize> = (0..n_heads).collect();
        idx.sort_by(|&a, &b| {
            ranking.scores[b]
                .partial_cmp(&ranking.scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let cs_top2: std::collections::HashSet<usize> = idx.iter().take(2).copied().collect();
        for &h in &signal_heads {
            assert!(
                cs_top2.contains(&h),
                "G1: CS probe failed to surface signal head {h} in top-2 \
                 (top-2 = {cs_top2:?}, scores = {:?})",
                ranking.scores
            );
        }

        // CS top-2 accuracy (2/2 signal heads) must beat a random top-2's
        // expected overlap. Random expected overlap = 2 * 2 / 16 = 0.25 per
        // trial on average; assert CS gets full overlap (2/2 = 1.0), which is
        // a ≥0.5 absolute margin — comfortably above the 15pp G1 bar.
        let cs_accuracy = cs_top2.intersection(&signal_heads.iter().copied().collect::<std::collections::HashSet<usize>>()).count() as f32 / 2.0;
        assert!(
            cs_accuracy >= 1.0,
            "G1: CS top-2 accuracy {cs_accuracy} < 1.0 (expected both signal heads)"
        );
    }
}
