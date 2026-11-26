//! Issue 681 — cross-crate agreement: `katgpt_core::data_probe::gaussianity`
//! vs this crate's `ks_d_statistic`.
//!
//! katgpt-core's `ks_d_vs_fitted_gaussian` is a VERBATIM port of
//! `katgpt_spectral::spectral::ks_d_statistic` (same sort, same f64
//! accumulation order, same Abramowitz-Stegun `normal_cdf`) — made because
//! the katgpt-core leaf must not depend on katgpt-spectral (the `rrq_quant`
//! scalar-inversion rule). This test pins the two implementations together:
//! on identical 1D samples they must agree to the bit.
//!
//! Also reconstructs one full probe pass: project a population with the
//! scratch's exposed direction table and verify `report.per_direction[a]`
//! equals `ks_d_statistic` on the reconstructed projection.

use katgpt_core::data_probe::gaussianity::{
    GaussianityScratch, N_DIRECTIONS, ks_d_vs_fitted_gaussian, sketched_gaussianity,
};
use katgpt_core::types::Rng;
use katgpt_spectral::spectral::ks_d_statistic;

fn sample_gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.normal()).collect()
}

fn sample_bimodal(n: usize, mu: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
            rng.normal() + sign * mu
        })
        .collect()
}

#[test]
fn ks_d_bit_identical_on_identical_samples() {
    let samples: Vec<Vec<f32>> = vec![
        sample_gaussian(1024, 1),
        sample_bimodal(1024, 3.0, 2),
        // Heavy tail: 5% at ±10.
        {
            let mut rng = Rng::new(3);
            (0..1024)
                .map(|_| {
                    if rng.uniform() < 0.95 {
                        rng.normal()
                    } else {
                        10.0 * rng.normal()
                    }
                })
                .collect()
        },
        // Discrete lattice.
        {
            let mut rng = Rng::new(4);
            (0..256)
                .map(|_| if rng.uniform() < 0.5 { 0.0f32 } else { 1.0 })
                .collect()
        },
    ];

    for (i, sample) in samples.iter().enumerate() {
        let mut scratch_spectral = vec![0.0f32; sample.len()];
        let mut scratch_core = vec![0.0f32; sample.len()];
        let d_spectral = ks_d_statistic(sample, &mut scratch_spectral);
        let d_core = ks_d_vs_fitted_gaussian(sample, &mut scratch_core);
        assert_eq!(
            d_spectral.to_bits(),
            d_core.to_bits(),
            "sample {i}: spectral D={d_spectral:.9} != core D={d_core:.9} — the port drifted"
        );
    }
}

#[test]
fn probe_per_direction_matches_ks_d_statistic_on_reconstructed_projection() {
    let (n, d) = (512usize, 64usize);
    let mut rng = Rng::new(9);
    let mut states = vec![0.0f32; n * d];
    for (i, chunk) in states.chunks_mut(d).enumerate() {
        // Mild mixture structure so directions disagree meaningfully.
        let sign = if i % 3 == 0 { 3.0 } else { 0.0 };
        for v in chunk.iter_mut() {
            *v = rng.normal() + if rng.uniform() < 0.1 { sign } else { 0.0 };
        }
    }

    let mut scratch = GaussianityScratch::new(n, d, 1234);
    let report = sketched_gaussianity(&states, &mut scratch);

    for (a, &report_a) in report.per_direction[..N_DIRECTIONS].iter().enumerate() {
        let dir = scratch.direction(a);
        // Reconstruct with the SAME f64 accumulation the probe uses (the
        // core's projection is an f64 accumulator cast to f32 — matching it
        // is part of the pinned contract).
        let projection: Vec<f32> = states
            .chunks(d)
            .map(|row| {
                let mut acc = 0.0f64;
                for (x, dv) in row.iter().zip(dir) {
                    acc += (*x as f64) * (*dv as f64);
                }
                acc as f32
            })
            .collect();
        let mut sp = vec![0.0f32; n];
        let expected = ks_d_statistic(&projection, &mut sp);
        assert_eq!(
            expected.to_bits(),
            report_a.to_bits(),
            "direction {a}: reconstructed ks_d_statistic {expected:.9} != probe report {report_a:.9}"
        );
    }
}
