//! Conformal interval hot-path benchmark — GOAT gate G2 (Plan 340).
//!
//! Target: `interval_into` ≤ 1µs at H=1, ≤ 100µs at H=8×8 channels
//! (warm-tier target, not hot-path). Zero hot-path overhead — the overlay is
//! queried explicitly, never on the per-tick critical path.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::{
    ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, ResidualMode,
    SeasonalPoolForecaster,
};

/// Build a calibrator with `n_channels × n_buckets` rings, each filled with
/// `capacity` residuals.
fn make_fitted_calibrator(n_channels: usize, max_h: usize, m: usize, capacity: usize) ->
    ConformalIntervalCalibrator<SeasonalPoolForecaster>
{
    let forecaster = SeasonalPoolForecaster::new(capacity.max(m) * 2, m, 0.0, 0.0);
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster,
        n_channels,
        max_h,
        m,
        capacity,
        0.0,
        DecayUnit::Step,
        ResidualMode::HStep,
        false,
    );
    // Fill each (channel, bucket) with synthetic residuals.
    let mut seed: u64 = 0xA5A5_5A5A_5A5A_5A5A;
    for _ in 0..(capacity * 2) {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (((seed >> 33) as f32) / (1u64 << 31) as f32) * 2.0 - 1.0;
        for ch in 0..n_channels {
            for h in 1..=max_h {
                cal.update_residual(r + 0.1 * (ch as f32) + 0.01 * (h as f32), 0.0, ch, h);
            }
        }
    }
    cal
}

fn bench_interval_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("conformal_interval_into");
    group.throughput(Throughput::Elements(1));

    // Three configurations per Plan 340 T1.11:
    //   H=1 (1 channel, 1 horizon)  — target ≤ 1µs
    //   H=8 (1 channel, 8 horizons) — target ≤ 10µs
    //   H=8×8 (8 channels, 8 horizons) — target ≤ 100µs
    let configs: &[(&str, usize, usize)] = &[
        ("h1_1ch", 1, 1),
        ("h8_1ch", 1, 8),
        ("h8_8ch", 8, 8),
    ];

    for &(label, n_channels, max_h) in configs {
        let m = 12;
        let capacity = 256;
        let mut cal = make_fitted_calibrator(n_channels, max_h, m, capacity);
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
        let alpha = 0.05_f32;

        group.bench_with_input(BenchmarkId::new("interval", label), &(), |b, _| {
            b.iter(|| {
                // Sweep all channels × horizons to amortize the per-config setup.
                for ch in 0..n_channels {
                    for h in 1..=max_h {
                        cal.interval_into(black_box(ch), black_box(h), black_box(alpha), &mut iv);
                        black_box(iv.lower);
                    }
                }
            });
        });
    }

    group.finish();
}

fn bench_update_residual(c: &mut Criterion) {
    let mut group = c.benchmark_group("conformal_update_residual");
    group.throughput(Throughput::Elements(1));

    let configs: &[(&str, usize, usize)] = &[
        ("h1_1ch", 1, 1),
        ("h8_8ch", 8, 8),
    ];

    for &(label, n_channels, max_h) in configs {
        let m = 12;
        let capacity = 256;
        let mut cal = make_fitted_calibrator(n_channels, max_h, m, capacity);

        group.bench_with_input(BenchmarkId::new("update", label), &(), |b, _| {
            let mut i: usize = 0;
            b.iter(|| {
                let r = (i as f32) * 0.001;
                for ch in 0..n_channels {
                    for h in 1..=max_h {
                        cal.update_residual(black_box(r), black_box(0.0), black_box(ch), black_box(h));
                    }
                }
                i = i.wrapping_add(1);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_interval_into, bench_update_residual);
criterion_main!(benches);
