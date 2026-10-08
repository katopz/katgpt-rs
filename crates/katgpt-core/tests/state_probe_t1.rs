//! Plan 621 Phase 1 — `state_probe` probe-kernel fixtures (T1.2 + pins).
//!
//! Whole-file `#![cfg]` + a `required-features` Cargo row: without the
//! feature the target is skipped, never a green zero. The Miller–Madow
//! fixture is the T1.2 acceptance; the blake3 pin is a determinism guard
//! (formula drift detector), not a correctness oracle — correctness lives
//! in the math fixtures below.

#![cfg(feature = "state_probe")]

use katgpt_core::hint_regret::wilson_score_ci;
use katgpt_core::state_probe::{ProbeInput, miller_madow_entropy, probe};

const Z95: f64 = 1.959_963_984_540_054;

/// The blake3 pin over the canonical byte serialization of the probe read
/// for the fixed input below (`v_hat.to_bits()` big-endian, then `h_mm`,
/// `wilson_lo`, `wilson_hi` — bit-exact, replayable across runs). Any
/// formula change flips the pin LOUDLY here.
const PIN_INPUT_BLAKE3: &str = "dc33cba1f00e6c9e7da0e66bc4bb3c49ca493990ee57e67cf1d3faa672d1601f";

/// Canonical byte serialization of a [`katgpt_core::state_probe::ProbeEstimate`]
/// for the pin (field order fixed; f64 bit patterns, big-endian).
fn estimate_pin_bytes(e: &katgpt_core::state_probe::ProbeEstimate) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&e.v_hat.to_bits().to_be_bytes());
    out[8..16].copy_from_slice(&e.h_mm.to_bits().to_be_bytes());
    out[16..24].copy_from_slice(&e.wilson_lo.to_bits().to_be_bytes());
    out[24..32].copy_from_slice(&e.wilson_hi.to_bits().to_be_bytes());
    out
}

/// T1.2 — Miller–Madow is (first-order) UNBIASED on the uniform multinomial
/// K=4/N=32: the EXPECTATION of `h_mm` over seed-pinned multinomial draws
/// lands within tolerance of the exact true entropy ln(4). (A single
/// all-equal draw is the H_emp MAXIMUM — ln 4 + (K−1)/(2N) — not a typical
/// draw; the unbiasedness claim is about the mean, so the fixture measures
/// the mean. 20_000 draws; the estimator's own σ at N=32 makes the mean's
/// noise ≈ 1e-3, tolerance 5e-3 covers the O(1/N²) bias residual too.)
#[test]
fn t12_uniform_multinomial_k4_n32_matches_exact_expectation() {
    // Seed-pinned splitmix64 — deterministic draws, no dep, no global RNG.
    let mut state: u64 = 0x6b21_cea9_1e73_ad0d; // arbitrary seed, fixed pin
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    const DRAWS: usize = 20_000;
    const N: u64 = 32;
    let mut acc = 0.0f64;
    for _ in 0..DRAWS {
        let mut counts = [0u16; 4];
        for _ in 0..N {
            counts[(next() % 4) as usize] += 1;
        }
        acc += miller_madow_entropy(&counts);
    }
    let mean = acc / DRAWS as f64;
    let ln4 = std::f64::consts::LN_2 * 2.0;
    assert!(
        (mean - ln4).abs() < 5e-3,
        "E[h_mm]={mean} vs ln4={ln4} (|Δ|={})",
        (mean - ln4).abs()
    );
}

/// The deterministic formula identity on the extreme draw: a perfectly
/// uniform histogram has `H_emp = ln K` EXACTLY (every p = 1/K), so the
/// corrected value must be `ln K + (K−1)/(2N)` bit-for-bit-the-formula —
/// this pins the correction's SHAPE (sign, denominator, K′−1 numerator).
#[test]
fn t12_uniform_draw_formula_identity_exact() {
    let hist = [8u16, 8, 8, 8];
    let h_mm = miller_madow_entropy(&hist);
    let ln4 = std::f64::consts::LN_2 * 2.0;
    let expected = ln4 + 3.0 / 64.0;
    assert!(
        (h_mm - expected).abs() < 1e-12,
        "h_mm={h_mm} vs expected={expected}"
    );
}

/// T1.2 — monotone in added agreement: shifting counts from a minority bin
/// into the majority bin (the ensemble agreeing more) strictly DECREASES
/// the Miller–Madow entropy at every step.
#[test]
fn t12_entropy_monotone_in_added_agreement() {
    let mut hist = [8u16, 8, 8, 8];
    let mut prev = miller_madow_entropy(&hist);
    for _ in 0..7 {
        hist[0] += 1;
        hist[3] -= 1;
        let next = miller_madow_entropy(&hist);
        assert!(
            next < prev,
            "h_mm must strictly decrease as the histogram peaks: {prev} -> {next} at {hist:?}"
        );
        prev = next;
    }
}

/// T1.2 — the blake3 pin: same input, same bytes, every run.
#[test]
fn t12_output_bytes_blake3_stable() {
    let hist = [32u16, 16, 8, 4, 2, 1, 1]; // sums to 64
    let input = ProbeInput {
        pass_count: 21,
        n: 64,
        histogram: &hist,
    };
    let est = probe(&input, Z95);
    let hex = blake3::Hasher::new()
        .update(&estimate_pin_bytes(&est))
        .finalize()
        .to_hex()
        .to_string();
    assert_eq!(
        hex, PIN_INPUT_BLAKE3,
        "state_probe output-bytes pin moved — a formula or field-order change landed; \
         re-derive deliberately and re-pin"
    );
}

/// The Wilson half is the SUBSTRATE's, bit-identical — pins that no third
/// CI copy exists and none diverges (Research 609's DRY law).
#[test]
fn wilson_half_is_the_substrate_bit_identical() {
    let hist = [32u16, 16, 8, 4, 2, 1, 1];
    let input = ProbeInput {
        pass_count: 21,
        n: 64,
        histogram: &hist,
    };
    let est = probe(&input, Z95);
    let (lo, hi) = wilson_score_ci(est.v_hat, u64::from(input.n), Z95);
    assert_eq!(est.wilson_lo.to_bits(), lo.to_bits());
    assert_eq!(est.wilson_hi.to_bits(), hi.to_bits());
}

/// Edge posture: `n = 0` is the uninformative state — `v_hat = 0`, the
/// substrate's `(0, 1)` degenerate interval, `h_mm = 0`.
#[test]
fn n_zero_is_uninformative_not_a_panic() {
    let input = ProbeInput {
        pass_count: 0,
        n: 0,
        histogram: &[],
    };
    let est = probe(&input, Z95);
    assert_eq!(est.v_hat, 0.0);
    assert_eq!(est.wilson_lo, 0.0);
    assert_eq!(est.wilson_hi, 1.0);
    assert_eq!(est.h_mm, 0.0);
}

/// Degenerate histograms: a single populated bin carries zero entropy (the
/// correction vanishes with `K′ = 1`); an all-zero / empty histogram means
/// answers-not-tracked → `h_mm = 0`.
#[test]
fn degenerate_histograms_carry_zero_entropy() {
    assert_eq!(miller_madow_entropy(&[16]), 0.0);
    assert_eq!(miller_madow_entropy(&[0, 0, 0]), 0.0);
    assert_eq!(miller_madow_entropy(&[]), 0.0);
}

/// Pins the observed-support reading of the Miller–Madow correction: empty
/// bins never join `K′` (Grassberger 2003 §3.2 — Research 609's `(K−1)/(2N)`
/// coincides only on a fully populated histogram, which the T1.2 fixture is).
#[test]
fn correction_uses_nonzero_support_not_alphabet_length() {
    // N=8, support {a,b} fully populated, two trailing empty bins.
    let hist = [4u16, 4, 0, 0];
    let h_mm = miller_madow_entropy(&hist);
    let expected = std::f64::consts::LN_2 + (2.0 - 1.0) / (2.0 * 8.0);
    assert!(
        (h_mm - expected).abs() < 1e-12,
        "h_mm={h_mm} vs expected={expected}"
    );
}

/// One-member ensemble: `v_hat = 1`, Wilson upper clamps to 1.0 while the
/// lower bound stays strictly inside (0, 0.5] — the honest wide interval.
#[test]
fn single_pass_ensemble_clamps_upper_only() {
    let hist = [1u16];
    let input = ProbeInput {
        pass_count: 1,
        n: 1,
        histogram: &hist,
    };
    let est = probe(&input, Z95);
    assert_eq!(est.v_hat, 1.0);
    assert_eq!(est.wilson_hi, 1.0);
    assert!(
        est.wilson_lo > 0.0 && est.wilson_lo <= 0.5,
        "lo={}",
        est.wilson_lo
    );
}
