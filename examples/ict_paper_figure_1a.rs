//! Plan 294 T8.3 — Paper Figure 1a reproduction.
//!
//! Reproduces ICT Figure 1a: two distributions with identical Shannon entropy
//! H_1 but different collision purity β. Proves the paper's claim that β
//! captures structural information H_1 cannot. Run with:
//!
//! ```text
//! cargo run --example ict_paper_figure_1a --features ict_branching
//! ```

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::math::{collision_purity, shannon_h1};

/// Bisection solver: find `t` in `p_B(t) = (t, (1−t)/2, (1−t)/2, 0, 0, 0)`
/// with `H_1(p_B(t)) = ln 2`. Returns `t`. H_1(t) is strictly decreasing
/// in `t`, so the root is unique.
fn solve_equal_h1_three_outcome() -> f32 {
    let target = core::f32::consts::LN_2;
    let h1_of_t = |t: f32| -> f32 {
        let q = (1.0 - t) / 2.0;
        let mut h = 0.0_f32;
        if t > 0.0 {
            h -= t * t.ln();
        }
        if q > 0.0 {
            h -= 2.0 * q * q.ln();
        }
        h
    };
    let mut lo = 0.5_f32;
    let mut hi = 0.95_f32;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if h1_of_t(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn main() {
    println!("=== ICT Paper Figure 1a — H_1 vs β Bifurcation ===\n");

    // p_A: 50/50 split over 2 outcomes. H_1 = ln 2, β = 0.5.
    let p_a = [0.5_f32, 0.5, 0.0, 0.0, 0.0, 0.0];
    let beta_a = collision_purity(&p_a);
    let h1_a = shannon_h1(&p_a);

    // p_B: 3-outcome (t, (1−t)/2, (1−t)/2, 0, 0, 0) with same H_1.
    let t = solve_equal_h1_three_outcome();
    let q = (1.0 - t) / 2.0;
    let p_b = [t, q, q, 0.0, 0.0, 0.0];
    let beta_b = collision_purity(&p_b);
    let h1_b = shannon_h1(&p_b);

    println!("Distribution       H_1       β      Notes");
    println!("------------------|---------|-------|-----------------------------");
    println!(
        "p_A = (½,½,0,0,0,0)        {:.4}   {:.4}   two-outcome coin flip",
        h1_a, beta_a
    );
    println!(
        "p_B = ({:.3},{:.3},{:.3},0,0,0)     {:.4}   {:.4}   three-outcome, same H_1",
        p_b[0], p_b[1], p_b[2], h1_b, beta_b
    );

    println!("\n=== Verdict ===");
    println!("H_1 sees identical uncertainty:    ΔH_1 = {:.2e}", (h1_a - h1_b).abs());
    println!("β sees different concentration:    Δβ   = {:.4}", (beta_a - beta_b).abs());
    println!("\nThis is the ICT §1.5 / Figure 1a bifurcation in textbook form:");
    println!("H_1's blindness to the long tail (π < e⁻¹ ≈ 0.37) is why Bebop's");
    println!("acceptance forecast and Curiosity Pulse's underspecification signal");
    println!("are both wrong on long-tail tokens. β(π) = Σ π² is unconditionally");
    println!("monotone (ICT §A.2.5) — it sees what H_1 cannot.");

    // ASCII scatter: plot β on x, H_1 on y for both.
    println!("\n=== ASCII: β (x) vs H_1 (y) ===");
    println!("         0.0    0.1    0.2    0.3    0.4    0.5    0.6");
    let xs = [beta_a, beta_b];
    let ys = [h1_a, h1_b];
    let labels = ["p_A", "p_B"];
    for tick_idx in 0..=10 {
        let y = 0.85 - tick_idx as f32 * 0.07; // 0.85 down to 0.15
        let mut line = format!("{:.2} |", y);
        for col in 0..=60 {
            let x = col as f32 * 0.01;
            let mut ch = ' ';
            for (i, (&xi, &yi)) in xs.iter().zip(ys.iter()).enumerate() {
                if (xi - x).abs() < 0.005 && (yi - y).abs() < 0.035 {
                    ch = if i == 0 { 'A' } else { 'B' };
                }
            }
            line.push(ch);
        }
        println!("{line}");
    }
    println!("       +------------------------------------------------------------");
    let _ = labels; // labels shown inline above
}
