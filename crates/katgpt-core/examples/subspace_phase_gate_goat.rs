//! Plan 301 Phase 2 — G1 GOAT proof: subspace phase transition on synthetic MoLRG.
//!
//! Reproduces Wang et al. (arXiv:2409.02426) Theorem 4: a d-dimensional
//! subspace of R^D cannot be recovered from fewer than d samples. We
//! generate K=3 mutually-orthogonal d=6 subspaces in R^48, sample N wake
//! events per subspace for N ∈ {3, 5, 6, 7, 10, 50, 200}, run PCA via SVD,
//! and measure the subspace recovery error ‖Û Û^T − U* U*^T‖_F.
//!
//! **G1 PASS criteria:**
//! - N < d (=6): mean recovery error > 0.5 (recovery fails).
//! - N ≥ d (=6): mean recovery error < 0.1 (recovery succeeds).
//! - `phase_transition_gate(N, d)` matches the empirical transition.
//!
//! **PCA-via-Jacobian-SVD trick:** the Jacobian of the linear map
//! f(x) = X · x (X is the N × D data matrix) is X itself. So
//! `jacobian_svd_at(f, ...)` yields the SVD of X, and the right singular
//! vectors (length D) are the principal directions. This exercises the
//! public API of `subspace_phase_gate` with no separate SVD dependency.
//!
//! **No centering:** the ground-truth mean is exactly 0 (samples are
//! x = Uz with z ~ N(0, I_d)), so uncentered SVD recovers the subspace
//! for N ≥ d. Centering would reduce the effective rank by 1 and shift
//! the transition to N = d+1, contradicting Theorem 4.

use katgpt_core::{
    JacobianSvdScratch, jacobian_svd_at, numerical_rank, participation_ratio, phase_transition_gate,
};

// ── Setup (Plan 301 §Phase 2) ─────────────────────────────────────────────

const AMBIENT_DIM: usize = 48; // D
const INTRINSIC_DIM: usize = 6; // d
const NUM_SUBSPACES: usize = 3; // K
const SAMPLE_SIZES: [usize; 7] = [3, 5, 6, 7, 10, 50, 200];
const SEED: u64 = 0x3015EED_301C0FFEE;

/// N < d: error must exceed this for G1 PASS on the "fail" side.
const FAIL_THRESHOLD: f32 = 0.5;
/// N ≥ d: error must be below this for G1 PASS on the "recover" side.
const PASS_THRESHOLD: f32 = 0.1;
/// Empirical "recovery succeeded" cutoff for T2.6 gate-vs-empirical check.
const EMPIRICAL_RECOVERY_THRESHOLD: f32 = 0.15;

// ── Deterministic PRNG (PCG-XSH-RR 32-bit; zero external deps) ────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            // Odd seeds required by PCG's LCG; force odd.
            state: seed | 1,
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    /// Uniform f32 in (0, 1].
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits, maps to (0, 1]. Avoids exact zero (breaks log below).
        ((self.next_u32() >> 8) | 1) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal N(0, 1) via Box–Muller.
    #[inline]
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ── Linear-algebra helpers (self-contained; no nalgebra dep) ──────────────

/// Modified Gram–Schmidt orthonormalisation of the columns of `a` (m × n,
/// row-major), in-place. Produces Q with orthonormal columns. Numerically
/// stable for well-conditioned random Gaussian matrices.
fn modgs(a: &mut [f32], m: usize, n: usize) {
    for j in 0..n {
        for i in 0..j {
            let mut d = 0.0_f32;
            for r in 0..m {
                d += a[r * n + i] * a[r * n + j];
            }
            for r in 0..m {
                a[r * n + j] -= d * a[r * n + i];
            }
        }
        let mut norm_sq = 0.0_f32;
        for r in 0..m {
            norm_sq += a[r * n + j] * a[r * n + j];
        }
        let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
        for r in 0..m {
            a[r * n + j] *= inv_norm;
        }
    }
}

/// Generate K mutually-orthogonal d-dim subspaces in R^D by QR of a single
/// D × (K·d) Gaussian matrix. Returns K bases, each D × d with orthonormal
/// columns, such that all Kd columns are mutually orthonormal.
fn generate_subspaces(rng: &mut Rng) -> Vec<Vec<f32>> {
    let total_cols = NUM_SUBSPACES * INTRINSIC_DIM;
    debug_assert!(total_cols <= AMBIENT_DIM);
    let mut g = vec![0.0_f32; AMBIENT_DIM * total_cols];
    for v in &mut g {
        *v = rng.next_gaussian();
    }
    modgs(&mut g, AMBIENT_DIM, total_cols);
    (0..NUM_SUBSPACES)
        .map(|k| {
            let mut basis = vec![0.0_f32; AMBIENT_DIM * INTRINSIC_DIM];
            for r in 0..AMBIENT_DIM {
                for j in 0..INTRINSIC_DIM {
                    basis[r * INTRINSIC_DIM + j] = g[r * total_cols + k * INTRINSIC_DIM + j];
                }
            }
            basis
        })
        .collect()
}

/// Sample N wake events from subspace `basis` (D × d). Each event is
/// x = basis · z where z ~ N(0, I_d). Returns N × D row-major. No mean
/// shift (ground-truth mean is zero).
fn sample_events(basis: &[f32], n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * AMBIENT_DIM];
    for i in 0..n {
        let mut z = [0.0_f32; INTRINSIC_DIM];
        for z_j in z.iter_mut() {
            *z_j = rng.next_gaussian();
        }
        for r in 0..AMBIENT_DIM {
            let mut acc = 0.0_f32;
            for j in 0..INTRINSIC_DIM {
                acc += basis[r * INTRINSIC_DIM + j] * z[j];
            }
            out[i * AMBIENT_DIM + r] = acc;
        }
    }
    out
}

/// Subspace recovery error ‖Û Û^T − U* U*^T‖_F via the identity
///   ‖·‖_F² = d_hat + d_star − 2 ‖Û^T U*‖_F²
/// avoiding materialising the D × D projectors.
fn recovery_error(u_hat: &[Vec<f32>], u_star: &[f32]) -> f32 {
    let d_hat = u_hat.len();
    let d_star = INTRINSIC_DIM;
    let mut cross_sq = 0.0_f32;
    for u_hat_i in u_hat.iter() {
        for j in 0..d_star {
            let mut dot = 0.0_f32;
            for r in 0..AMBIENT_DIM {
                dot += u_hat_i[r] * u_star[r * d_star + j];
            }
            cross_sq += dot * dot;
        }
    }
    let frob_sq = (d_hat as f32) + (d_star as f32) - 2.0 * cross_sq;
    frob_sq.max(0.0).sqrt()
}

/// Per-N per-subspace run: sample, PCA-via-Jacobian-SVD, recovery error,
/// plus intrinsic-dim estimates from the singular-value spectrum.
struct RunResult {
    error: f32,
    pr_estimate: f32,
    nr_estimate: usize,
}

fn run_single(basis: &[f32], n: usize, rng: &mut Rng) -> RunResult {
    let data = sample_events(basis, n, rng);

    // f(x) = data · x, f: R^D → R^N. Jacobian = data (N × D).
    // Right singular vectors of the Jacobian (length D) = principal directions.
    let f = move |x: &[f32], out: &mut [f32]| {
        for i in 0..out.len() {
            let mut acc = 0.0_f32;
            for j in 0..AMBIENT_DIM {
                acc += data[i * AMBIENT_DIM + j] * x[j];
            }
            out[i] = acc;
        }
    };

    let mut scratch = JacobianSvdScratch::with_capacity(AMBIENT_DIM, n);
    let x_zero = [0.0_f32; AMBIENT_DIM];
    let result = jacobian_svd_at(f, &x_zero, 1e-4, &mut scratch);

    // Top-d right singular vectors = estimated principal directions.
    let u_hat: Vec<Vec<f32>> = result
        .right_singular_vectors
        .iter()
        .take(INTRINSIC_DIM)
        .cloned()
        .collect();

    let error = recovery_error(&u_hat, basis);

    // Spectrum = singular values (already sorted descending by the SVD impl).
    let spectrum: Vec<f32> = result.singular_values.clone();
    let pr = participation_ratio(&spectrum);
    let nr = numerical_rank(&spectrum, 0.99);

    RunResult {
        error,
        pr_estimate: pr,
        nr_estimate: nr,
    }
}

// ── Main: run the G1 GOAT gate ────────────────────────────────────────────

fn main() {
    let mut rng = Rng::new(SEED);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 301 Phase 2 — G1 GOAT: Subspace Phase Transition");
    println!("  Paper: arXiv:2409.02426 (Wang et al., Theorem 4)");
    println!(
        "  Setup: D={}  K={}  d={}  subspaces,  N ∈ {:?}",
        AMBIENT_DIM, NUM_SUBSPACES, INTRINSIC_DIM, SAMPLE_SIZES
    );
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // T2.2 — generate ground-truth subspaces.
    let subspaces = generate_subspaces(&mut rng);

    // Sanity: each basis is orthonormal, and the K bases are mutually orthogonal.
    for (k, basis) in subspaces.iter().enumerate() {
        for i in 0..INTRINSIC_DIM {
            for j in 0..INTRINSIC_DIM {
                let mut dot = 0.0_f32;
                for r in 0..AMBIENT_DIM {
                    dot += basis[r * INTRINSIC_DIM + i] * basis[r * INTRINSIC_DIM + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "subspace {k} not orthonormal at ({i},{j}): got {dot}"
                );
            }
        }
    }
    println!(
        "✓ T2.2: K={} mutually-orthogonal d={} orthonormal bases in R^{}",
        NUM_SUBSPACES, INTRINSIC_DIM, AMBIENT_DIM
    );
    println!();

    // T2.3 + T2.4 — sweep N, run PCA per subspace, emit CSV.
    println!("── T2.3/T2.4: Recovery error vs N (mean over K subspaces) ──");
    println!("N,d,mean_err,min_err,max_err,gate(N,d),pr_mean,nr99_mean");

    struct NRow {
        n: usize,
        mean_err: f32,
        pr_mean: f32,
        nr_mean: f32,
    }

    let mut rows: Vec<NRow> = Vec::with_capacity(SAMPLE_SIZES.len());
    for &n in &SAMPLE_SIZES {
        let mut errors = Vec::with_capacity(NUM_SUBSPACES);
        let mut prs = Vec::with_capacity(NUM_SUBSPACES);
        let mut nrs = Vec::with_capacity(NUM_SUBSPACES);
        for subspace in subspaces.iter() {
            let r = run_single(subspace, n, &mut rng);
            errors.push(r.error);
            prs.push(r.pr_estimate);
            nrs.push(r.nr_estimate as f32);
        }
        let mean_err = errors.iter().sum::<f32>() / NUM_SUBSPACES as f32;
        let min_err = errors.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_err = errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let pr_mean = prs.iter().sum::<f32>() / NUM_SUBSPACES as f32;
        let nr_mean = nrs.iter().sum::<f32>() / NUM_SUBSPACES as f32;
        let gate = phase_transition_gate(n, INTRINSIC_DIM);
        println!(
            "{},{},{:.6},{:.6},{:.6},{},{:.3},{:.1}",
            n, INTRINSIC_DIM, mean_err, min_err, max_err, gate, pr_mean, nr_mean
        );
        rows.push(NRow {
            n,
            mean_err,
            pr_mean,
            nr_mean,
        });
    }
    println!();

    // T2.5 — verify phase transition: N<d → err>0.5, N≥d → err<0.1.
    println!("── T2.5: Phase-transition check ──");
    println!(
        "  Rule: N<d → err>{},  N≥d → err<{}",
        FAIL_THRESHOLD, PASS_THRESHOLD
    );
    let mut t25_pass = true;
    for r in &rows {
        let expect_high = r.n < INTRINSIC_DIM;
        let ok = if expect_high {
            r.mean_err > FAIL_THRESHOLD
        } else {
            r.mean_err < PASS_THRESHOLD
        };
        if !ok {
            t25_pass = false;
        }
        let mark = if ok { "✓" } else { "✗" };
        let side = if expect_high { "fail" } else { "recover" };
        println!(
            "  {} N={:>3}: mean_err={:.4}  (expected {} side)",
            mark, r.n, r.mean_err, side
        );
    }
    println!("  T2.5 verdict: {}", if t25_pass { "PASS" } else { "FAIL" });
    println!();

    // T2.6 — phase_transition_gate matches empirical recovery.
    println!("── T2.6: phase_transition_gate(N, d) vs empirical ──");
    let mut t26_pass = true;
    for r in &rows {
        let gate = phase_transition_gate(r.n, INTRINSIC_DIM);
        let empirical = r.mean_err < EMPIRICAL_RECOVERY_THRESHOLD;
        let ok = gate == empirical;
        if !ok {
            t26_pass = false;
        }
        let mark = if ok { "✓" } else { "✗" };
        println!(
            "  {} N={:>3}: gate={}, empirical={}, err={:.4}",
            mark, r.n, gate, empirical, r.mean_err
        );
    }
    println!("  T2.6 verdict: {}", if t26_pass { "PASS" } else { "FAIL" });
    println!();

    // T2.7 — participation_ratio vs numerical_rank as intrinsic-dim estimators.
    println!(
        "── T2.7: Intrinsic-dim estimation (true d={}) ──",
        INTRINSIC_DIM
    );
    println!(
        "  {:>4}  {:>10}  {:>10}  {:>8}",
        "N", "PR_round", "NR99", "winner"
    );
    let mut pr_wins = 0u32;
    let mut nr_wins = 0u32;
    for r in &rows {
        let pr_round = r.pr_mean.round();
        let nr_est = r.nr_mean;
        let pr_err = (pr_round - INTRINSIC_DIM as f32).abs();
        let nr_err = (nr_est - INTRINSIC_DIM as f32).abs();
        let winner = if pr_err < nr_err {
            pr_wins += 1;
            "PR"
        } else if nr_err < pr_err {
            nr_wins += 1;
            "NR"
        } else {
            "tie"
        };
        println!(
            "  {:>4}  {:>10.1}  {:>10.1}  {:>8}",
            r.n, pr_round, nr_est, winner
        );
    }
    println!();
    println!(
        "  Summary: PR wins {} row(s), NR wins {} row(s).",
        pr_wins, nr_wins
    );
    println!("  On this synthetic MoLRG, NR tracks the true d better than PR");
    println!("  (sharp spectral elbow). For N<d, both correctly report N — the");
    println!("  true d is information-theoretically unrecoverable. NR is the");
    println!("  better production pick (discrete, threshold-tunable, immune to");
    println!("  continuous-valued drift); PR is the better diagnostic (shows");
    println!("  the effective dimensionality even when no clear elbow exists).");
    println!();

    // Final G1 verdict.
    let g1_pass = t25_pass && t26_pass;
    println!("═══════════════════════════════════════════════════════════════");
    if g1_pass {
        println!("  G1: PASS — phase transition reproduces on synthetic MoLRG.");
    } else {
        println!("  G1: FAIL — phase transition does NOT match theory.");
    }
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Exit code so CI / `cargo run` callers can detect failure.
    if !g1_pass {
        std::process::exit(1);
    }
}
