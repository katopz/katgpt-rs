//! Issue 747 P1 T1.2 — derived-k budget G1: the Lemma-2 length-independence
//! law (Research 549, arXiv:2506.16640 Lemma 2).
//!
//! The paper's Lemma 2 (α=1.5): k tied tokens at level-gap Δ above the bulk
//! hold attention 1/k each iff `Δ ≥ 2/√k` — equivalently the boundary block
//! size is `k̂ = 4/Δ²`. **The condition contains no n term.** This gate
//! plants two-level rows AT the boundary (k = k̂(Δ) needles, gap Δ) and
//! asserts the realized entmax support equals k̂ at every n from 1k to 1M —
//! the property the fitted sigmoid(`w·var+b`) budget of `compute_adaptive_k`
//! structurally lacks (its input — row variance — drifts with the
//! distribution; there is no level-gap law).
//!
//! Also pins the honest max−mean proxy calibration: on a two-level row the
//! proxy measures top-to-center (`Δ + bulk_mean_offset`), not the level gap
//! — `compute_derived_k_from_scores` resolves `4/(Δ+offset)²`, the
//! concentration-regime calibration documented on `compute_derived_k`.

#![cfg(feature = "asentmax_schedule")]

use katgpt_attn::dash_attn::adaptive_k::{compute_derived_k, AdaptiveKConfig};
use katgpt_attn::dash_attn::asentmax::AsentmaxSchedule;
use katgpt_attn::dash_attn::entmax::{entmax_1p5, entmax_support};

/// Deterministic splitmix64 uniform in [0, 1).
fn uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) as f32
}

/// Two-level row: bulk uniform [0, 1), `k` needles tied (tiny jitter) at
/// `1.0 + gap`. The level-gap Δ is exactly `gap` (bulk top → needle level).
fn two_level_row(n: usize, k: usize, gap: f32, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut row = vec![0.0_f32; n];
    for slot in row.iter_mut().take(n - k) {
        *slot = uniform(&mut state);
    }
    for slot in row.iter_mut().skip(n - k) {
        // Tiny jitter keeps the sort deterministic-in-expectation without
        // changing the boundary margin (1e-4 ≪ gap).
        *slot = 1.0 + gap + (uniform(&mut state) - 0.5) * 1e-4;
    }
    row
}

fn realized_support(row: &[f32]) -> usize {
    // No schedule — the raw two-level law is what Lemma 2 describes.
    let (probs, _) = entmax_1p5(row);
    entmax_support(&probs).len()
}

#[test]
fn g1_derived_k_boundary_law_is_length_independent() {
    // Δ grid with 2% boundary margin (needles at gap·0.99 clearance keeps
    // S(k̂+1) = k̂·Δ comfortably ≥ 1 on the Δ grid below).
    let config = AdaptiveKConfig::new(1, 1_048_576);
    let gaps: &[f32] = &[0.5, 0.7, 1.0, 1.4];
    let ns: &[usize] = &[1_000, 8_000, 65_536, 524_288, 1_000_000];

    for &gap in gaps {
        let k_hat = compute_derived_k(gap * 0.99, &config);
        assert!(k_hat >= 1, "Δ={gap}: k̂ must be ≥ 1, got {k_hat}");
        // The inverse map is self-consistent: k̂·Δ ≥ 1 (the support-hold
        // condition) with the boundary margin.
        assert!(
            k_hat as f32 * gap >= 1.0,
            "Δ={gap}: boundary condition k̂·Δ = {} must be ≥ 1",
            k_hat as f32 * gap
        );

        for &n in ns {
            let row = two_level_row(n, k_hat, gap, 7_000 + n as u64);
            let support = realized_support(&row);
            assert_eq!(
                support, k_hat,
                "Δ={gap}, n={n}: realized support {support} ≠ k̂ {k_hat} — the Lemma-2 \
                 boundary must be length-independent"
            );
        }
    }
}

/// The sigmoid arm on the SAME rows has no such law: its budget is a
/// function of row variance (which shifts with the bulk distribution and
/// needle mass fraction), not of the level gap. Reported, not asserted —
/// the point is the contrast in structure, not a particular drift direction.
#[test]
fn g1_sigmoid_arm_has_no_level_gap_law_reported() {
    let gaps: &[f32] = &[0.5, 0.7, 1.0, 1.4];
    let ns: &[usize] = &[1_000, 65_536, 1_000_000];
    let config = AdaptiveKConfig::new(1, 1_048_576);
    let mut report = String::new();
    for &gap in gaps {
        let k_hat = compute_derived_k(gap * 0.99, &config);
        for &n in ns {
            let row = two_level_row(n, k_hat, gap, 11_000 + n as u64);
            // Row variance drives the sigmoid arm.
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let var = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / row.len() as f32;
            let z = 5.0_f32 * var; // AdaptiveKConfig::default() w=5, b=0
            let sig = 1.0 / (1.0 + (-z).exp());
            let k_sigmoid = (1.0 + (1_048_576.0 - 1.0) * sig).round() as usize;
            report.push_str(&format!(
                "Δ={gap} n={n}: var={var:.5} → sigmoid_k={k_sigmoid} (k̂={k_hat})\n"
            ));
        }
    }
    // Structural assertion only: the sigmoid budget is a function of var
    // (monotone), and var varies with n at fixed structure (needle mass
    // fraction k̂/n shifts the row variance) — i.e. no length-independent
    // constant exists for it. Pin that variance actually moves across n.
    let gap = 0.7_f32;
    let k_hat = compute_derived_k(gap * 0.99, &config);
    let var_at = |n: usize| {
        let row = two_level_row(n, k_hat, gap, 13_000 + n as u64);
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / row.len() as f32
    };
    assert_ne!(
        var_at(1_000),
        var_at(1_000_000),
        "row variance must move with n at fixed structure (else the contrast is vacuous)"
    );
    eprintln!("{report}");
}

/// Pin the max−mean proxy calibration honestly: on the two-level row,
/// `compute_derived_k_from_scores` resolves 4/(Δ + bulk-mean-offset)² — the
/// documented concentration-regime calibration, not the level-gap law.
#[test]
fn g1_max_mean_proxy_calibration_pinned() {
    let config = AdaptiveKConfig::new(1, 1_048_576);
    let gap = 1.0_f32;
    let k_hat = compute_derived_k(gap * 0.99, &config); // 4
    let n = 8_000_usize;
    let row = two_level_row(n, k_hat, gap, 17_000);
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = row.iter().sum::<f32>() / n as f32;
    // Bulk uniform [0,1): mean ≈ 0.5; Δ̂ = (1 + gap) − 0.5 = gap + 0.5.
    let delta_hat = max - mean;
    assert!(
        (delta_hat - (gap + 0.5)).abs() < 0.05,
        "max−mean proxy should be ≈ gap + bulk mean offset: {delta_hat}"
    );
    // The proxy-budget is therefore 4/(gap+0.5)², NOT 4/gap² — the
    // documented divergence between the proxy and the level-gap law.
    let proxy_k = compute_derived_k(delta_hat, &config);
    let expected_proxy = (4.0 / ((gap + 0.5) * (gap + 0.5))).round() as usize;
    assert_eq!(proxy_k, expected_proxy.max(1));
}

/// The derived budget must also compose with the P0 schedule without
/// breaking the raw two-level law (schedule off = the Lemma-2 regime;
/// schedule on = the damped regime, gated by G1 of P0).
#[test]
fn g1_derived_k_composes_with_schedule_none() {
    let config = AdaptiveKConfig::new(1, 1_048_576);
    let gap = 0.7_f32;
    let k_hat = compute_derived_k(gap * 0.99, &config);
    let n = 65_536_usize;
    let row = two_level_row(n, k_hat, gap, 23_000);
    // AsentmaxSchedule::None must leave the row untouched (P0 no-op
    // contract) — the support law holds unchanged.
    let mut scores = row.clone();
    katgpt_attn::dash_attn::asentmax::apply_asentmax_inplace(
        &mut scores,
        &AsentmaxSchedule::None,
        (n as f32).ln(),
    );
    assert_eq!(scores, row, "None schedule must be a bit-identical no-op");
    assert_eq!(realized_support(&scores), k_hat);
}
