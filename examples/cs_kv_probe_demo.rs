//! CS-KV-Importance Probe demo (Plan 280, Research 247).
//!
//! Synthetic 64-head task: heads {3, 17, 42} carry signal (their KV entries
//! correlate with `label_success`). Run the CS probe, print the recovered
//! ranking, and demonstrate the `K(ca)` interpolation for `ca ∈ {0.0, 0.25,
//! 0.5, 0.75, 1.0}` — the sparse/dense duality's bandwidth axis.
//!
//! Run: `cargo run --example cs_kv_probe_demo --features cs_kv_probe`

#[cfg(feature = "cs_kv_probe")]
fn main() {
    use katgpt_rs::cs_kv_probe::{
        CsKvProbe, CsProbeConfig, DensityBudget, Episode, GatedKvSlice,
    };

    let n_heads = 64_usize;
    let d = 128_usize; // KV cache dim per episode.
    let signal_heads = [3_usize, 17, 42];
    let mut rng = fastrand::Rng::with_seed(0xC5_280);

    // Build N=100 synthetic episodes: signal heads' KV channels encode the label.
    let n_episodes = 100;
    let episodes: Vec<Episode> = (0..n_episodes)
        .map(|_| {
            let label = rng.bool();
            let mut kv = vec![0.0_f32; d];
            for v in kv.iter_mut() {
                *v = rng.f32() * 2.0 - 1.0;
            }
            for &h in &signal_heads {
                let ch = (h * d / n_heads) % d;
                kv[ch] = if label { 1.0 } else { -1.0 } + 0.1 * (rng.f32() * 2.0 - 1.0);
            }
            Episode::new(kv, label)
        })
        .collect();

    // Eval: mean agreement between retained signal heads and label across episodes.
    let eval = |mask: &katgpt_rs::cs_kv_probe::AblationMask, eps: &[Episode]| -> f32 {
        let mut acc = 0.0_f32;
        for e in eps {
            for &h in &signal_heads {
                if mask.bits[h] {
                    let ch = (h * d / n_heads) % d;
                    acc += e.kv_cache[ch] * if e.label_success { 1.0 } else { -1.0 };
                }
            }
        }
        acc / eps.len().max(1) as f32
    };
    let cfg = CsProbeConfig {
        m_masks: 200,
        ablation_fraction: 0.05,
        lasso_alpha: 1e-4,
        lasso_iter: 1000,
        n_heads,
        n_kv_heads: n_heads,
    };
    let ranking = CsKvProbe::run(&episodes, &eval, &cfg, &mut rng);

    // Rank groups by score, print top-10.
    let mut idx: Vec<usize> = (0..n_heads).collect();
    idx.sort_by(|&a, &b| {
        ranking.scores[b]
            .partial_cmp(&ranking.scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("CS-KV Probe — top-10 groups (signal = {:?}):", signal_heads);
    for (rank, &g) in idx.iter().take(10).enumerate() {
        let marker = if signal_heads.contains(&g) { "★" } else { " " };
        println!("  {:>2}. group {:>2}  score {:+.4} {}", rank + 1, g, ranking.scores[g], marker);
    }

    // Density budget — paper floors 3.5% / 87% at D = 64.
    let budget = DensityBudget::for_dim(n_heads);
    println!(
        "\nDensityBudget(D={}): k_sparse={} (3.5%), k_dense={} (87%)",
        n_heads, budget.k_sparse, budget.k_dense
    );
    println!("\nK(ca) interpolation + GatedKvSlice finite-count:");
    let kv_dummy = vec![0.0_f32; n_heads];
    let mut bias = vec![0.0_f32; n_heads];
    let mut idx_scratch = vec![0_usize; n_heads];
    for ca in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let k = budget.k_for(ca);
        GatedKvSlice::apply(&ranking, &budget, ca, &kv_dummy, &mut idx_scratch, &mut bias);
        let finite = bias.iter().filter(|b| b.is_finite()).count();
        let top_in = idx.iter().take(k).filter(|&&g| signal_heads.contains(&g)).count();
        println!(
            "  ca={:>4} → K={:>2}  gate-finite={:>2}  signal-in-top-K={}/{}",
            ca, k, finite, top_in, signal_heads.len()
        );
    }
}

#[cfg(not(feature = "cs_kv_probe"))]
fn main() {
    println!("cs_kv_probe feature is disabled. Rebuild with --features cs_kv_probe.");
}
