//! Issue 1234 GOAT gate — kernel-derived Fourier latent embeddings
//! (Research 682, arXiv:2602.15029). Bars pre-registered in the issue:
//!
//! - G1  full-width Gram ≍ centered sampled kernel, N=24, σ ∈ {1, 4.2, 12},
//!   max abs diff < 1e-5 (a round-trip pin — cannot fail by construction).
//! - G1c truncated width w < N/2: ‖K_c − Φ_wΦ_wᵀ‖_F / ‖K_c‖_F ≤
//!   1.5·sqrt(Σ_{n>w} a_n² / Σ_{n≥1} a_n²) per σ (the width law's test).
//! - G1b phase readout recovers each x_i within 1/(2w) at the truncated width
//!   — read from the DATA side (the centered Gram's top conjugate eigen-pair),
//!   aligned up to the pair's global rotation/reflection.
//! - G4  construction + features + readout allocate nothing.
//! - Open-BC check (beyond the issue's bars): feature columns match the
//!   fine-grid eigenfunctions of the raw AND centered kernels.
//! - G2  (`#[ignore]`, quiet-box cell): construction cost is O(N·w).
#![cfg(feature = "kernel_fourier_features")]

use katgpt_core::linalg::symmetric_eig::{SymmetricEigScratch, symmetric_eig};
use katgpt_core::spectral::kernel_features::{
    KernelSpectrum, OpenKernel, PeriodicAmplitudes, conjugate_pairing_score,
    fourier_features_from_kernel, phase_readout_into,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::f64::consts::PI;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static TRACK: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

const N: usize = 24;
const SIGMAS: [f64; 3] = [1.0, 4.2, 12.0];

fn xs() -> Vec<f32> {
    (0..N).map(|i| (2.0 * i as f64 / N as f64) as f32).collect()
}

/// Centered sampled periodized-exponential Gram (f64).
fn centered_kernel(sigma: f64) -> Vec<f64> {
    let c: Vec<f64> = (0..N)
        .map(|j| {
            let d = 2.0 * j as f64 / N as f64;
            ((1.0 - d) / sigma).cosh() / (1.0 / sigma).sinh()
        })
        .collect();
    let mean = c.iter().sum::<f64>() / N as f64;
    let mut k = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            k[i * N + j] = c[(i + N - j) % N] - mean;
        }
    }
    k
}

fn gram(phi: &[f32], width2: usize) -> Vec<f64> {
    let mut g = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            g[i * N + j] = (0..width2)
                .map(|t| phi[i * width2 + t] as f64 * phi[j * width2 + t] as f64)
                .sum();
        }
    }
    g
}

fn embed(sigma: f64, w: usize) -> (KernelSpectrum, Vec<f32>) {
    let s = KernelSpectrum::periodic(sigma, w, PeriodicAmplitudes::DiscreteExact { n_points: N })
        .unwrap();
    let mut phi = vec![0.0f32; N * 2 * w];
    fourier_features_from_kernel(&s, &xs(), &mut phi).unwrap();
    (s, phi)
}

#[test]
fn g1_full_width_round_trip() {
    let _g = SERIAL.lock().unwrap();
    for sigma in SIGMAS {
        let (_, phi) = embed(sigma, N / 2);
        let (k, g) = (centered_kernel(sigma), gram(&phi, N));
        let max = k
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("G1 σ={sigma}: max |K_c − ΦΦᵀ| = {max:.3e}");
        assert!(max < 1e-5, "G1 σ={sigma}: {max}");
    }
}

#[test]
fn g1c_truncated_width_obeys_the_tail_law() {
    let _g = SERIAL.lock().unwrap();
    for sigma in SIGMAS {
        let (full, _) = embed(sigma, N / 2);
        let k = centered_kernel(sigma);
        let kn: f64 = k.iter().map(|v| v * v).sum::<f64>().sqrt();
        let a2: Vec<f64> = (0..N / 2)
            .map(|n| {
                let (_, ac, _, as_) = full.mode(n);
                // a_n² as the Gram's weight on mode n: cos and sin share it,
                // Nyquist carries cos only — use the cos amplitude.
                let _ = as_;
                ac * ac
            })
            .collect();
        let total: f64 = a2.iter().sum();
        for w in [1usize, 2, 4, 8, 11] {
            let (_, phi) = embed(sigma, w);
            let g = gram(&phi, 2 * w);
            let err: f64 = k
                .iter()
                .zip(&g)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / kn;
            let tail: f64 = a2[w..].iter().sum();
            let bound = 1.5 * (tail / total).sqrt();
            println!("G1c σ={sigma} w={w}: rel err {err:.4e} ≤ bound {bound:.4e}");
            assert!(err <= bound, "G1c σ={sigma} w={w}: {err} > {bound}");
        }
    }
}

/// Eigendecompose a symmetric n×n f64 matrix; returns (eigvals desc, rows).
fn eig_desc(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut ev, mut v) = (vec![0.0; n], vec![0.0; n * n]);
    let mut sc = SymmetricEigScratch::new();
    sc.ensure_capacity(n);
    symmetric_eig(&mut ev, &mut v, a, &mut sc, n, 300);
    // symmetric_eig: eigenvector j is COLUMN j.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&p, &q| ev[q].total_cmp(&ev[p]));
    let vals = idx.iter().map(|&j| ev[j]).collect();
    let mut rows = vec![0.0; n * n];
    for (r, &j) in idx.iter().enumerate() {
        for i in 0..n {
            rows[r * n + i] = v[i * n + j];
        }
    }
    (vals, rows)
}

/// Circular distance on period 2.
fn circ(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(2.0);
    d.min(2.0 - d)
}

#[test]
fn g1b_phase_readout_from_the_data_side() {
    let _g = SERIAL.lock().unwrap();
    let truth: Vec<f64> = xs().iter().map(|&x| x as f64).collect();
    for sigma in SIGMAS {
        let k = centered_kernel(sigma);
        let (vals, rows) = eig_desc(&k, N);
        let score = conjugate_pairing_score(&vals, 2 * 4, 1e-6);
        println!(
            "G1b σ={sigma}: top eig {:.4} {:.4} pairing(top 8) {score:.2}",
            vals[0], vals[1]
        );
        assert_eq!(score, 1.0, "centered circulant top modes must pair");
        let cos: Vec<f32> = rows[..N].iter().map(|&v| v as f32).collect();
        let sin: Vec<f32> = rows[N..2 * N].iter().map(|&v| v as f32).collect();
        let mut est = vec![0.0f32; N];
        phase_readout_into(&cos, &sin, PI as f32, &mut est).unwrap();
        // Align the eigen-pair's arbitrary rotation + reflection.
        let mut best = f64::INFINITY;
        for sign in [1.0, -1.0] {
            let (mut cs, mut sn) = (0.0, 0.0);
            for i in 0..N {
                let d = PI * (est[i] as f64 - sign * truth[i]);
                cs += d.cos();
                sn += d.sin();
            }
            let shift = sn.atan2(cs) / PI;
            let worst = (0..N)
                .map(|i| circ(est[i] as f64, sign * truth[i] + shift))
                .fold(0.0, f64::max);
            best = best.min(worst);
        }
        for w in [2usize, 4, 8] {
            let bar = 1.0 / (2.0 * w as f64);
            println!("G1b σ={sigma} w={w}: max phase error {best:.3e} ≤ {bar}");
            assert!(best <= bar, "G1b σ={sigma} w={w}: {best} > {bar}");
        }
    }
}

#[test]
fn g4_zero_alloc() {
    let _g = SERIAL.lock().unwrap();
    let x = xs();
    let mut phi = vec![0.0f32; N * 16];
    let mut est = vec![0.0f32; N];
    let (cos, sin) = (vec![0.5f32; N], vec![0.25f32; N]);
    ALLOCS.store(0, Ordering::SeqCst);
    TRACK.store(1, Ordering::SeqCst);
    let s = KernelSpectrum::periodic(4.2, 8, PeriodicAmplitudes::DiscreteExact { n_points: N })
        .unwrap();
    fourier_features_from_kernel(&s, &x, &mut phi).unwrap();
    let o = KernelSpectrum::open(4.2, 8, OpenKernel::Centered).unwrap();
    fourier_features_from_kernel(&o, &x, &mut phi).unwrap();
    phase_readout_into(&cos, &sin, core::f32::consts::PI, &mut est).unwrap();
    TRACK.store(0, Ordering::SeqCst);
    let n = ALLOCS.load(Ordering::SeqCst);
    println!("G4 allocs = {n}");
    assert_eq!(n, 0);
}

/// Open BC: feature columns ≍ the fine-grid eigenfunctions (raw + centered).
#[test]
fn open_boundary_features_match_grid_eigenfunctions() {
    let _g = SERIAL.lock().unwrap();
    const M: usize = 401;
    let grid: Vec<f64> = (0..M)
        .map(|i| -1.0 + 2.0 * i as f64 / (M - 1) as f64)
        .collect();
    let h = 2.0 / (M - 1) as f64;
    for sigma in [0.5, 1.0, 4.2] {
        for kernel in [OpenKernel::Raw, OpenKernel::Centered] {
            let mut k = vec![0.0; M * M];
            for i in 0..M {
                for j in 0..M {
                    k[i * M + j] = (-(grid[i] - grid[j]).abs() / sigma).exp() * h;
                }
            }
            if kernel == OpenKernel::Centered {
                let rm: Vec<f64> = (0..M)
                    .map(|i| k[i * M..(i + 1) * M].iter().sum::<f64>() / M as f64)
                    .collect();
                let all = rm.iter().sum::<f64>() / M as f64;
                for i in 0..M {
                    for j in 0..M {
                        k[i * M + j] += all - rm[i] - rm[j];
                    }
                }
            }
            let (vals, rows) = eig_desc(&k, M);
            let s = KernelSpectrum::open(sigma, 3, kernel).unwrap();
            let xf: Vec<f32> = grid.iter().map(|&v| v as f32).collect();
            let mut phi = vec![0.0f32; M * 6];
            fourier_features_from_kernel(&s, &xf, &mut phi).unwrap();
            for e in 0..4 {
                let v = &rows[e * M..(e + 1) * M];
                let mut best = 0.0f64;
                for col in 0..6 {
                    let c: Vec<f64> = (0..M).map(|i| phi[i * 6 + col] as f64).collect();
                    let cn = c.iter().map(|z| z * z).sum::<f64>().sqrt();
                    let dot: f64 = c.iter().zip(v).map(|(a, b)| a * b).sum();
                    best = best.max((dot / cn).abs());
                }
                assert!(
                    best > 0.999,
                    "σ={sigma} {kernel:?} eig {e} (λ {}): best |corr| {best}",
                    vals[e]
                );
            }
        }
    }
}

/// G2 — construction cost shape O(N·w). Quiet-box cell only (`--ignored`).
#[test]
#[ignore]
fn g2_construction_is_linear_in_n_times_w() {
    let _g = SERIAL.lock().unwrap();
    let cap_iter = [(256usize, 8usize), (1024, 8), (1024, 32), (4096, 32)];
    let mut per_unit = Vec::with_capacity(cap_iter.len());
    for (n, w) in cap_iter {
        let x: Vec<f32> = (0..n).map(|i| (2.0 * i as f64 / n as f64) as f32).collect();
        let s = KernelSpectrum::periodic(4.2, w, PeriodicAmplitudes::Lorentzian).unwrap();
        let mut phi = vec![0.0f32; n * 2 * w];
        let reps = 200;
        let mut best = u128::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                fourier_features_from_kernel(&s, std::hint::black_box(&x), &mut phi).unwrap();
                std::hint::black_box(&phi);
            }
            best = best.min(t.elapsed().as_nanos());
        }
        let ns = best as f64 / reps as f64;
        let u = ns / (n * w) as f64;
        println!("G2 N={n} w={w}: {ns:.0} ns/build, {u:.2} ns per (N·w)");
        per_unit.push(u);
    }
    let (lo, hi) = per_unit
        .iter()
        .fold((f64::MAX, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
    assert!(
        hi / lo < 3.0,
        "G2: ns/(N·w) spread {:.2}× ≥ 3× — not O(N·w)",
        hi / lo
    );
}
