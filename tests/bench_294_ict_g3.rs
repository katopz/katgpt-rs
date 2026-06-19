//! Plan 294 Phase 4 — GOAT Gate G3: ICT orthogonality to H_1 (MAKE-OR-BREAK).
//!
//! For each decision point in the same synthetic suite as G2, we collect
//! paired samples `(h1_s, u_max_s)` where:
//!   - `h1_s = shannon_h1(p_avg)` is the Shannon entropy of the population mean
//!   - `u_max_s = max_k u_{k,s}` is the maximum JS-divergence-to-mean
//!
//! We compute the **Spearman rank correlation ρ** between H_1 and u_max
//! across all paired samples. The Super-GOAT verdict hinges on this:
//!
//! - **ρ < 0.5**  → G3 PASS — JS captures structurally-different information
//!   from H_1. Super-GOAT proceeds.
//! - **0.5 ≤ ρ < 0.9** → G3 BORDERLINE — document honestly, do NOT promote
//!   to default, defer G8 (riir-ai Plan 324).
//! - **ρ ≥ 0.9**  → G3 FAIL — downgrade R270 from Super-GOAT to Gain, cancel
//!   Plan 324, file issue.
//!
//! ## Why Spearman, not Pearson
//!
//! Spearman is rank-based and robust to non-linear monotone relationships.
//! The paper's claim is that H_1 and JS-uniqueness carry *structurally
//! different* information — if they were monotone functions of each other
//! Spearman ρ would be ±1. A low |ρ| is the orthogonality signal.
//!
//! ## Implementation
//!
//! Spearman ρ = Pearson correlation of ranks. We compute:
//! 1. Rank-transform both vectors (average ranks for ties).
//! 2. Pearson correlation on the ranks.
//!
//! Bootstrap 95% CI: resample with replacement 1000 times, recompute ρ each
//! time, take the 2.5% and 97.5% quantiles.
//!
//! ## Run
//!
//! ```text
//! cargo test --features ict_branching --test bench_294_ict_g3 -- --nocapture
//! ```

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::math::{js_divergence_batch, shannon_h1};

const N_DECISION_POINTS: usize = 1000;
const K_TRAJECTORIES: usize = 8;
const ACTION_DIM: usize = 6;
const N_BOOTSTRAP: usize = 1000;

// ── Deterministic LCG (matches G2 for reproducibility). ───────────────────

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ── Synthetic suite (identical to G2 for direct comparison). ──────────────

fn sample_decision_point(rng: &mut Lcg) -> Vec<Vec<f32>> {
    let u = rng.next_f32();
    let regime = if u < 0.55 {
        "committed"
    } else if u < 0.85 {
        "undecided"
    } else {
        "noise"
    };
    let mut trajs = Vec::with_capacity(K_TRAJECTORIES);
    for _ in 0..K_TRAJECTORIES {
        let mut p = match regime {
            "committed" => {
                let dom = (rng.next_u64() % ACTION_DIM as u64) as usize;
                let dom_mass = 0.6 + 0.2 * rng.next_f32();
                let mut p = vec![0.0_f32; ACTION_DIM];
                p[dom] = dom_mass;
                let rest = (1.0 - dom_mass) / (ACTION_DIM - 1) as f32;
                for j in 0..ACTION_DIM {
                    if j != dom {
                        p[j] = rest * (0.5 + rng.next_f32());
                    }
                }
                p
            }
            "undecided" => {
                let top_count = 2 + (rng.next_u64() % 2) as usize;
                let mut p = vec![0.0_f32; ACTION_DIM];
                let base = 1.0 / top_count as f32;
                for k in 0..top_count {
                    p[k] = base * (0.8 + 0.4 * rng.next_f32());
                }
                for j in top_count..ACTION_DIM {
                    p[j] = 0.02 * rng.next_f32();
                }
                p
            }
            _ => {
                let mut p = vec![0.0_f32; ACTION_DIM];
                for j in 0..ACTION_DIM {
                    p[j] = 1.0 + rng.next_f32();
                }
                p
            }
        };
        normalize(&mut p);
        trajs.push(p);
    }
    trajs
}

fn normalize(p: &mut [f32]) {
    let s: f32 = p.iter().sum();
    if s > 0.0 {
        for v in p.iter_mut() {
            *v /= s;
        }
    }
}

// ── Spearman rank correlation (no external dep). ──────────────────────────

/// Compute average ranks of `xs` (rank 1 = smallest). Ties share the mean
/// of the ranks they would occupy. Writes ranks into `out_ranks`.
fn rank_average(xs: &[f32], out_ranks: &mut [f32]) {
    let n = xs.len();
    debug_assert_eq!(out_ranks.len(), n);
    // (value, original_index)
    let mut idx: Vec<(f32, usize)> = xs.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut i = 0;
    while i < n {
        // Find the run of equal values starting at i.
        let mut j = i + 1;
        while j < n && idx[j].0 == idx[i].0 {
            j += 1;
        }
        // Average rank for positions i..j (1-indexed): (i+1 + j) / 2.
        let avg_rank = ((i + 1) + j) as f32 / 2.0;
        for k in i..j {
            out_ranks[idx[k].1] = avg_rank;
        }
        i = j;
    }
}

/// Pearson correlation coefficient. Returns 0.0 if either input has zero
/// variance (denominator is zero).
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx: f32 = xs.iter().sum::<f32>() / n;
    let my: f32 = ys.iter().sum::<f32>() / n;
    let mut num = 0.0_f32;
    let mut dx2 = 0.0_f32;
    let mut dy2 = 0.0_f32;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let den = (dx2 * dy2).sqrt();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Spearman rank correlation: Pearson on ranks.
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len();
    let mut rx = vec![0.0_f32; n];
    let mut ry = vec![0.0_f32; n];
    rank_average(xs, &mut rx);
    rank_average(ys, &mut ry);
    pearson(&rx, &ry)
}

/// Bootstrap 95% CI for Spearman ρ. Resample `xs, ys` (paired) `n_boot`
/// times with replacement, recompute ρ each time, return `(lo, hi)` as the
/// 2.5% and 97.5% quantiles. Uses `rng` for reproducibility.
fn spearman_bootstrap_ci(xs: &[f32], ys: &[f32], n_boot: usize, rng: &mut Lcg) -> (f32, f32) {
    let n = xs.len();
    let mut rhos: Vec<f32> = Vec::with_capacity(n_boot);
    let mut bx = vec![0.0_f32; n];
    let mut by = vec![0.0_f32; n];
    for _ in 0..n_boot {
        for k in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            bx[k] = xs[idx];
            by[k] = ys[idx];
        }
        rhos.push(spearman(&bx, &by));
    }
    rhos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let lo_idx = (0.025 * n_boot as f32).floor() as usize;
    let hi_idx = (0.975 * n_boot as f32).ceil() as usize;
    let hi_idx = hi_idx.min(n_boot - 1);
    (rhos[lo_idx], rhos[hi_idx])
}

// ── ASCII scatter plot. ────────────────────────────────────────────────────

fn ascii_scatter(xs: &[f32], ys: &[f32]) {
    let xmin = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let xmax = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ymin = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let ymax = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let x_range = (xmax - xmin).max(1e-6);
    let y_range = (ymax - ymin).max(1e-6);
    let w = 60_usize;
    let h = 20_usize;
    let mut grid = vec![b' '; w * h];
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let col = (((x - xmin) / x_range) * (w as f32 - 1.0)) as usize;
        // Invert y so larger y is at the top.
        let row = h - 1 - (((y - ymin) / y_range) * (h as f32 - 1.0)) as usize;
        let row = row.min(h - 1);
        let col = col.min(w - 1);
        grid[row * w + col] = b'*';
    }
    println!("      H_1 (x) →");
    println!("      {:.3}                                                            {:.3}", xmin, xmax);
    for row in 0..h {
        let label = if row == 0 {
            format!("{:.3}", ymax)
        } else if row == h - 1 {
            format!("{:.3}", ymin)
        } else {
            String::new()
        };
        let line: String = grid[row * w..(row + 1) * w].iter().map(|&c| c as char).collect();
        println!("{:>6} |{}", label, line);
    }
    println!("       +-----------------------------------------------------------");
    println!("         u_max (y) ↑");
}

// ── The test. ─────────────────────────────────────────────────────────────

#[test]
fn g3_orthogonality_to_h1_make_or_break() {
    let mut rng = Lcg::new(0x294BEB0Bu64);
    let mut scratch_m = vec![0.0_f32; ACTION_DIM];

    // Collect paired samples (h1, u_max) across all decision points.
    let mut h1_samples: Vec<f32> = Vec::with_capacity(N_DECISION_POINTS);
    let mut umax_samples: Vec<f32> = Vec::with_capacity(N_DECISION_POINTS);

    for _ in 0..N_DECISION_POINTS {
        let trajs = sample_decision_point(&mut rng);
        let traj_refs: Vec<&[f32]> = trajs.iter().map(|t| t.as_slice()).collect();

        // ── Compute population mean P̄ into scratch_m. ──
        for v in scratch_m[..ACTION_DIM].iter_mut() {
            *v = 0.0;
        }
        for t in &trajs {
            for j in 0..ACTION_DIM {
                scratch_m[j] += t[j];
            }
        }
        for v in scratch_m[..ACTION_DIM].iter_mut() {
            *v /= K_TRAJECTORIES as f32;
        }

        // h1_s = H_1(P̄)
        let h1 = shannon_h1(&scratch_m[..ACTION_DIM]);

        // u_max = max_k JS(π_k, P̄) — recompute via js_divergence_batch
        // (which uses its own internal scratch).
        let u = js_divergence_batch(&traj_refs, &mut scratch_m);
        let u_max = u.iter().cloned().fold(0.0_f32, f32::max);

        h1_samples.push(h1);
        umax_samples.push(u_max);
    }

    // ── Spearman ρ + bootstrap 95% CI. ──
    let rho = spearman(&h1_samples, &umax_samples);
    let mut boot_rng = Lcg::new(0xC0FFEE_u64);
    let (ci_lo, ci_hi) = spearman_bootstrap_ci(&h1_samples, &umax_samples, N_BOOTSTRAP, &mut boot_rng);

    println!("\n=== G3 — Orthogonality to H_1 (MAKE-OR-BREAK) ===");
    println!("Samples: {N_DECISION_POINTS} decision points × (H_1(P̄), max_k u_k)");
    println!("\nSpearman ρ(H_1, u_max) = {:.4}", rho);
    println!("Bootstrap 95% CI       = [{:.4}, {:.4}]  (n_boot = {N_BOOTSTRAP})", ci_lo, ci_hi);
    println!("\nScatter (H_1 on x, u_max on y):");
    ascii_scatter(&h1_samples, &umax_samples);

    // ── Plan 294 T4.2 verdict. ──
    // ρ < 0.5  → PASS (Super-GOAT proceeds)
    // 0.5 ≤ ρ < 0.9 → BORDERLINE (no default promotion, defer G8/Plan 324)
    // ρ ≥ 0.9 → FAIL (downgrade R270 Super-GOAT → Gain, cancel Plan 324)
    println!("\n=== Verdict ===");
    if rho >= 0.9 {
        println!("G3 FAIL (ρ = {rho:.4} ≥ 0.9): H_1 already captures the signal.");
        println!("  → DOWNGRADE R270 Super-GOAT → Gain.");
        println!("  → Cancel riir-ai Plan 324.");
        println!("  → File issue: H_1 captures what we hoped β/JS would add.");
    } else if rho >= 0.5 {
        println!("G3 BORDERLINE (ρ = {rho:.4} ∈ [0.5, 0.9)): partial overlap with H_1.");
        println!("  → Do NOT promote ict_branching to default-on.");
        println!("  → Do NOT cancel Plan 324 — defer pending G8 (riir-ai Plan 324) validation.");
        println!("  → Mark Plan 294 Phase 8 T8.4 as 'deferred pending G8'.");
    } else {
        println!("G3 PASS (ρ = {rho:.4} < 0.5): JS captures structurally-different info from H_1.");
        println!("  → Super-GOAT PROCEEDS to G4-G6 + G10.");
    }

    // ── Honest assertion. ──
    // The hard Super-GOAT threshold is ρ < 0.5. We don't fail the test on
    // borderline (0.5 ≤ ρ < 0.9) because the plan says "document honestly,
    // do NOT promote to default" — that's a documentation action, not a
    // panic. We only fail the test on the hard ρ ≥ 0.9 downgrade path.
    assert!(
        rho < 0.9,
        "G3 FAIL: Spearman ρ(H_1, u_max) = {rho:.4} ≥ 0.9 — R270 MUST downgrade from \
         Super-GOAT to Gain per Plan §Phase 4 Downgrade path. CI = [{ci_lo:.4}, {ci_hi:.4}]."
    );

    // Mark the borderline case in the assertion message for visibility.
    if rho >= 0.5 {
        eprintln!(
            "\nNOTE: G3 borderline (ρ = {rho:.4} ∈ [0.5, 0.9)). ict_branching stays opt-in; \
             Plan 324 deferred pending G8."
        );
    }
}
