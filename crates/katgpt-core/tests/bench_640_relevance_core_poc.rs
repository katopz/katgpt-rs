//! Plan 640 Phase 1 — the relevance_core (ACE, arXiv:2602.22600) GOAT gate
//! on a planted bank with a NONLINEAR frozen head.
//!
//! Fixture (T1.1): `h = Q_pᵀ z + a_dec·u_dec·z_dec + σ·ε` in `R^D`, with a
//! rank-R planted subspace `Q_p` (unit-variance coordinates), a decoy
//! direction `u_dec` ORTHOGONAL to `Q_p` carrying `a_dec² ≫ 1` variance and
//! uncorrelated with the planted coordinates (the G6 constraint), and
//! isotropic noise. The frozen head reads EXACTLY the planted span:
//! `f_c(h) = Σ_j A[c,j]·tanh(β·q_j·h)` — per-sample Jacobians vary (tanh'),
//! and the Jacobian span IS `span(Q_p)`. Labels `y = argmax_c f_c(h)`.
//! Two LOUD fixture refusals: a near-constant per-sample Jacobian (the
//! linear-head degeneration — ACE ≡ diff-of-means there, R557 already owns
//! it) and measured Jacobian energy escaping the planted span.
//!
//! Jacobians (T1.2): central finite differences through `f` →
//! `accumulate_sens`. Arms (T1.3), all scored through the SAME frozen head
//! after `project_through_into`: ACE (zero labels) · ridge probe (R557, the
//! supervised incumbent; full / 32 / 8 labels) · PCA · seeded random ·
//! residual (complement of ACE). f64 `Q` narrows to f32 at ONE line
//! (`to_f32_basis`) — the evaluation boundary; G1/G6 score the f64 basis.
//!
//! Pre-registered bars (Research 683 §8, copied from the plan BEFORE the
//! first run): G0 BLAKE3 ×2 over the f64 (Q, energies) · G1 ACE basis
//! similarity vs planted ≥ 0.90, aligned ≥ 0.9·full, residual ≤
//! chance + 0.05 · G2 ACE contrast ≥ probe(full) contrast − 0.05 · G3 S = I
//! ⇒ ACE ≡ PCA (similarity ≥ 0.99) · G4 extract + eval zero-alloc (fit
//! allocs counted, printed) · G5 ACE contrast ≥ probe(8 labels) contrast +
//! 0.05 · G6 PCA top-1 on the decoy (≥ 0.9) while ACE vs planted ≥ 0.90.
//! contrast = acc(aligned@R) − acc(random@R).
#![cfg(feature = "relevance_core")]

use katgpt_core::relevance_core::{
    CoreScratch, RankRule, accumulate_sens, extract_core_from_matrices,
};
use katgpt_core::subspace_intervention::{
    InterventionScratch, project_through_into, random_basis_into, ridge_probe_fit_into,
};
use katgpt_core::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicUsize = AtomicUsize::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) == 1 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn counted<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    TRACKING.store(1, Ordering::SeqCst);
    let out = f();
    TRACKING.store(0, Ordering::SeqCst);
    (out, ALLOC_COUNT.load(Ordering::SeqCst))
}

const D: usize = 32;
const R: usize = 3; // planted rank
const C: usize = 4; // classes
const BETA: f64 = 1.5;
const A_DEC: f64 = 5.0; // decoy amplitude (variance 25)
const SIGMA: f64 = 0.3; // isotropic noise
const N_COV: usize = 4096;
const N_JAC: usize = 128;
const N_TEST: usize = 2048;
const N_TRAIN_FULL: usize = 1024;
const FD_EPS: f64 = 1e-5;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

struct World {
    q_p: Vec<f64>,   // R×D planted rows
    u_dec: Vec<f64>, // D decoy direction
    a: Vec<f64>,     // C×R head mixing
}

impl World {
    fn new(seed: u64) -> Self {
        let mut rng = Lcg(seed);
        // R+1 orthonormal rows: R planted + 1 decoy (orthogonal by GS).
        let mut f: Vec<f64> = (0..(R + 1) * D).map(|_| rng.gauss()).collect();
        for i in 0..=R {
            for p in 0..i {
                let dot: f64 = (0..D).map(|t| f[i * D + t] * f[p * D + t]).sum();
                for t in 0..D {
                    f[i * D + t] -= dot * f[p * D + t];
                }
            }
            let n: f64 = (0..D).map(|t| f[i * D + t].powi(2)).sum::<f64>().sqrt();
            for t in 0..D {
                f[i * D + t] /= n;
            }
        }
        let a = (0..C * R).map(|_| rng.gauss()).collect();
        Self {
            q_p: f[..R * D].to_vec(),
            u_dec: f[R * D..].to_vec(),
            a,
        }
    }

    fn sample(&self, rng: &mut Lcg, out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = SIGMA * rng.gauss();
        }
        for j in 0..R {
            let z = rng.gauss();
            for (o, &q) in out.iter_mut().zip(&self.q_p[j * D..(j + 1) * D]) {
                *o += z * q;
            }
        }
        let zd = A_DEC * rng.gauss();
        for (o, &u) in out.iter_mut().zip(&self.u_dec) {
            *o += zd * u;
        }
    }

    /// The frozen nonlinear head: f_c(h) = Σ_j A[c,j]·tanh(β·q_j·h).
    fn head(&self, h: &[f64], out: &mut [f64; C]) {
        let mut act = [0.0f64; R];
        for (j, a) in act.iter_mut().enumerate() {
            let dot: f64 = (0..D).map(|t| self.q_p[j * D + t] * h[t]).sum();
            *a = (BETA * dot).tanh();
        }
        for (c, o) in out.iter_mut().enumerate() {
            *o = (0..R).map(|j| self.a[c * R + j] * act[j]).sum();
        }
    }

    fn label(&self, h: &[f64]) -> usize {
        let mut o = [0.0; C];
        self.head(h, &mut o);
        argmax(&o)
    }
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Accuracy of the frozen head on (projected) f32 rows.
fn head_acc(w: &World, z: &[f32], y: &[usize], n: usize, row: &mut [f64]) -> f64 {
    let mut hits = 0usize;
    let mut o = [0.0; C];
    for i in 0..n {
        for t in 0..D {
            row[t] = z[i * D + t] as f64;
        }
        w.head(row, &mut o);
        if argmax(&o) == y[i] {
            hits += 1;
        }
    }
    hits as f64 / n as f64
}

/// (1/k)·‖A Bᵀ‖²_F over k×D row bases (f64 side of the precision boundary).
fn sim64(a: &[f64], b: &[f64], k: usize) -> f64 {
    let kb = b.len() / D;
    let mut fro = 0.0;
    for i in 0..k {
        for j in 0..kb {
            let dot: f64 = (0..D).map(|t| a[i * D + t] * b[j * D + t]).sum();
            fro += dot * dot;
        }
    }
    fro / k as f64
}

/// THE f64→f32 narrowing line (the evaluation boundary).
fn to_f32_basis(q: &[f64]) -> Vec<f32> {
    q.iter().map(|&v| v as f32).collect()
}

/// Top-k left singular vectors of a D×m f32 matrix, rows, descending σ.
fn top_left_singular(m_flat: &[f32], m_cols: usize, k: usize) -> Vec<f64> {
    let mut res = SvdResultScratch::with_capacity(D, m_cols);
    let mut work = SvdScratch::with_capacity(D, m_cols);
    // thin_svd_into wants rows ≥ cols; feed the D×m matrix, read U columns.
    thin_svd_into(m_flat, D, m_cols, &mut res, &mut work);
    let mut idx: Vec<usize> = (0..res.len()).collect();
    idx.sort_by(|&a, &b| res.singular_value(b).total_cmp(&res.singular_value(a)));
    let mut out = vec![0.0f64; k * D];
    for (row, &j) in idx.iter().take(k).enumerate() {
        let u = res.left_singular_vector(j);
        for t in 0..D {
            out[row * D + t] = u[t] as f64;
        }
    }
    out
}

/// Top-k right singular vectors of an n×D f32 matrix (PCA directions of
/// centered data), rows, descending σ.
fn top_right_singular(m_flat: &[f32], n: usize, k: usize) -> Vec<f64> {
    let mut res = SvdResultScratch::with_capacity(n, D);
    let mut work = SvdScratch::with_capacity(n, D);
    thin_svd_into(m_flat, n, D, &mut res, &mut work);
    let mut idx: Vec<usize> = (0..res.len()).collect();
    idx.sort_by(|&a, &b| res.singular_value(b).total_cmp(&res.singular_value(a)));
    let mut out = vec![0.0f64; k * D];
    for (row, &j) in idx.iter().take(k).enumerate() {
        let v = res.right_singular_vector(j);
        for t in 0..D {
            out[row * D + t] = v[t] as f64;
        }
    }
    out
}

struct Bank {
    /// The POPULATION covariance `Q_pᵀQ_p + a_dec²·u uᵀ + σ²I` — the
    /// pre-registered G6 fixture constraint (decoy uncorrelated with the
    /// planted coordinates IN Cov) holds exactly only here.
    cov: Vec<f64>,
    /// The sample covariance at N_COV (disclosed estimation-sensitivity arm).
    cov_sample: Vec<f64>,
    sens: Vec<f64>,
    x_cov_f32: Vec<f32>, // centered, for PCA
    x_test: Vec<f32>,
    y_test: Vec<usize>,
    x_train: Vec<f32>,
    y_train: Vec<usize>,
}

fn build_bank(w: &World, seed: u64) -> Bank {
    let mut rng = Lcg(seed);
    let mut h = vec![0.0f64; D];
    // Activity covariance (centered, f64).
    let mut xs = vec![0.0f64; N_COV * D];
    for i in 0..N_COV {
        w.sample(&mut rng, &mut h);
        xs[i * D..(i + 1) * D].copy_from_slice(&h);
    }
    let mut mean = [0.0f64; D];
    for i in 0..N_COV {
        for t in 0..D {
            mean[t] += xs[i * D + t];
        }
    }
    for m in mean.iter_mut() {
        *m /= N_COV as f64;
    }
    let mut cov = vec![0.0f64; D * D];
    for i in 0..N_COV {
        for a in 0..D {
            let xa = xs[i * D + a] - mean[a];
            for b in 0..D {
                cov[a * D + b] += xa * (xs[i * D + b] - mean[b]);
            }
        }
    }
    for v in cov.iter_mut() {
        *v /= (N_COV - 1) as f64;
    }
    let cov_sample = cov;
    let cov = population_cov(w);
    let x_cov_f32 = (0..N_COV * D)
        .map(|k| (xs[k] - mean[k % D]) as f32)
        .collect();

    // Relevance moment via central finite differences (T1.2), + the two
    // fixture refusals.
    let mut sens = vec![0.0f64; D * D];
    let mut n_seen = 0u64;
    let mut jac = vec![0.0f64; N_JAC * C * D];
    let (mut hp, mut hm) = (vec![0.0; D], vec![0.0; D]);
    let (mut op, mut om) = ([0.0; C], [0.0; C]);
    for s in 0..N_JAC {
        w.sample(&mut rng, &mut h);
        for t in 0..D {
            hp.copy_from_slice(&h);
            hm.copy_from_slice(&h);
            hp[t] += FD_EPS;
            hm[t] -= FD_EPS;
            w.head(&hp, &mut op);
            w.head(&hm, &mut om);
            for c in 0..C {
                jac[(s * C + c) * D + t] = (op[c] - om[c]) / (2.0 * FD_EPS);
            }
        }
        for c in 0..C {
            let row = &jac[(s * C + c) * D..(s * C + c + 1) * D];
            accumulate_sens(&mut sens, D, row, &mut n_seen).unwrap();
        }
    }
    // Refusal 1: per-sample Jacobian must VARY (nonlinear head).
    let mut jbar = vec![0.0f64; C * D];
    for s in 0..N_JAC {
        for k in 0..C * D {
            jbar[k] += jac[s * C * D + k] / N_JAC as f64;
        }
    }
    let (mut var, mut tot) = (0.0, 0.0);
    for s in 0..N_JAC {
        for k in 0..C * D {
            let v = jac[s * C * D + k];
            var += (v - jbar[k]).powi(2);
            tot += v * v;
        }
    }
    assert!(
        var / tot > 0.01,
        "FIXTURE REFUSAL: per-sample Jacobian ~constant (rel var {}) — linear-head degeneration",
        var / tot
    );
    // Refusal 2: Jacobian energy must stay inside the planted span.
    let tr_s: f64 = (0..D).map(|i| sens[i * D + i]).sum();
    let mut in_span = 0.0;
    for j in 0..R {
        let q = &w.q_p[j * D..(j + 1) * D];
        for a in 0..D {
            for b in 0..D {
                in_span += q[a] * sens[a * D + b] * q[b];
            }
        }
    }
    assert!(
        (tr_s - in_span) / tr_s < 1e-6,
        "FIXTURE REFUSAL: Jacobian span escapes the planted span ({} outside)",
        (tr_s - in_span) / tr_s
    );

    let mut draw = |n: usize, rng: &mut Lcg| {
        let mut x = vec![0.0f32; n * D];
        let mut y = vec![0usize; n];
        for i in 0..n {
            w.sample(rng, &mut h);
            y[i] = w.label(&h);
            for t in 0..D {
                x[i * D + t] = h[t] as f32;
            }
        }
        (x, y)
    };
    let (x_test, y_test) = draw(N_TEST, &mut rng);
    let (x_train, y_train) = draw(N_TRAIN_FULL, &mut rng);
    Bank {
        cov,
        cov_sample,
        sens,
        x_cov_f32,
        x_test,
        y_test,
        x_train,
        y_train,
    }
}

fn population_cov(w: &World) -> Vec<f64> {
    let mut cov = vec![0.0f64; D * D];
    for a in 0..D {
        cov[a * D + a] = SIGMA * SIGMA;
        for b in 0..D {
            let planted: f64 = (0..R).map(|j| w.q_p[j * D + a] * w.q_p[j * D + b]).sum();
            cov[a * D + b] += planted + A_DEC * A_DEC * w.u_dec[a] * w.u_dec[b];
        }
    }
    cov
}

/// Sample covariance of `n` fresh draws (the estimation-sensitivity rows).
fn sample_cov(w: &World, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut h = vec![0.0f64; D];
    let mut mean = [0.0f64; D];
    let mut m2 = vec![0.0f64; D * D];
    for _ in 0..n {
        w.sample(&mut rng, &mut h);
        for a in 0..D {
            mean[a] += h[a];
            for b in 0..D {
                m2[a * D + b] += h[a] * h[b];
            }
        }
    }
    let nf = n as f64;
    for a in 0..D {
        for b in 0..D {
            m2[a * D + b] = (m2[a * D + b] - mean[a] * mean[b] / nf) / (nf - 1.0);
        }
    }
    m2
}

fn ace(bank_cov: &[f64], sens: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut sc = CoreScratch::new(D);
    let (mut q, mut e) = (vec![0.0; D * D], vec![0.0; D]);
    let rep = extract_core_from_matrices(
        bank_cov,
        sens,
        D,
        RankRule::Fixed(k),
        &mut q,
        &mut e,
        &mut sc,
    )
    .unwrap();
    q.truncate(rep.r * D);
    (q, e)
}

fn digest(q: &[f64], e: &[f64]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for v in q.iter().chain(e) {
        h.update(&v.to_bits().to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// Probe arm: ridge fit on the first `n` training labels → top-R left
/// singular vectors of Wᵀ (D×C) — the R557 task-aligned basis.
fn probe_basis(bank: &Bank, n: usize) -> Vec<f64> {
    let mut sc = InterventionScratch::new(D, C, n, N_TEST);
    let mut wv = vec![0.0f32; C * D];
    ridge_probe_fit_into(
        &bank.x_train[..n * D],
        &bank.y_train[..n],
        n,
        D,
        C,
        1e-2,
        &mut sc,
        &mut wv,
    );
    let mut wt = vec![0.0f32; D * C];
    for c in 0..C {
        for t in 0..D {
            wt[t * C + c] = wv[c * D + t];
        }
    }
    top_left_singular(&wt, C, R)
}

struct ArmEval {
    full: f64,
    chance: f64,
}

fn eval_arm(w: &World, bank: &Bank, basis: &[f32], k: usize, complement: bool) -> f64 {
    let mut z = vec![0.0f32; N_TEST * D];
    let mut coeffs = vec![0.0f32; k];
    let mut row = vec![0.0f64; D];
    project_through_into(
        &bank.x_test,
        &mut z,
        basis,
        N_TEST,
        D,
        k,
        complement,
        &mut coeffs,
    );
    head_acc(w, &z, &bank.y_test, N_TEST, &mut row)
}

#[test]
fn relevance_core_goat_gate() {
    let w = World::new(0x640_0D1E);
    let bank = build_bank(&w, 0x640_BA4C);
    let mut row = vec![0.0f64; D];
    let base = ArmEval {
        full: head_acc(&w, &bank.x_test, &bank.y_test, N_TEST, &mut row),
        chance: {
            let mut cnt = [0usize; C];
            for &y in &bank.y_test {
                cnt[y] += 1;
            }
            *cnt.iter().max().unwrap() as f64 / N_TEST as f64
        },
    };

    // ── G0 determinism (two independent fixture builds) ─────────────────
    let (q_ace, e_ace) = ace(&bank.cov, &bank.sens, R);
    let bank2 = build_bank(&w, 0x640_BA4C);
    let (q2, e2) = ace(&bank2.cov, &bank2.sens, R);
    let (d1, d2) = (digest(&q_ace, &e_ace), digest(&q2, &e2));
    println!("G0 blake3 {:02x?}… == {:02x?}…", &d1[..6], &d2[..6]);
    assert_eq!(
        d1, d2,
        "G0: f64 (Q, energies) BLAKE3 must be bit-identical ×2"
    );

    // ── arms ────────────────────────────────────────────────────────────
    let ace_f32 = to_f32_basis(&q_ace);
    let mut rand_basis = vec![0.0f32; D * R];
    random_basis_into(&mut rand_basis, D, 0x640_5EED);
    let pca = top_right_singular(&bank.x_cov_f32, N_COV, R);
    let pca_f32 = to_f32_basis(&pca);
    let probe_full = probe_basis(&bank, N_TRAIN_FULL);
    let probe_32 = probe_basis(&bank, 32);
    let probe_8 = probe_basis(&bank, 8);

    let acc_rand = eval_arm(&w, &bank, &rand_basis, R, false);
    let acc_ace = eval_arm(&w, &bank, &ace_f32, R, false);
    let acc_resid = eval_arm(&w, &bank, &ace_f32, R, true);
    let acc_pca = eval_arm(&w, &bank, &pca_f32, R, false);
    let acc_pf = eval_arm(&w, &bank, &to_f32_basis(&probe_full), R, false);
    let acc_p32 = eval_arm(&w, &bank, &to_f32_basis(&probe_32), R, false);
    let acc_p8 = eval_arm(&w, &bank, &to_f32_basis(&probe_8), R, false);

    let sim_ace = sim64(&q_ace, &w.q_p, R);
    let sim_pca = sim64(&pca, &w.q_p, R);
    let sim_pf = sim64(&probe_full, &w.q_p, R);
    let sim_p8 = sim64(&probe_8, &w.q_p, R);
    let sim_p32 = sim64(&probe_32, &w.q_p, R);
    let pca_top1_dec = sim64(&pca[..D], &w.u_dec, 1);

    println!(
        "full={:.4} chance={:.4} | k=R={R}: random={acc_rand:.4}",
        base.full, base.chance
    );
    println!("arm        acc     contrast  sim(planted)");
    let rows = [
        ("ACE(0 lab)", acc_ace, sim_ace),
        ("probe 1024", acc_pf, sim_pf),
        ("probe 32", acc_p32, sim_p32),
        ("probe 8", acc_p8, sim_p8),
        ("PCA", acc_pca, sim_pca),
    ];
    for (name, acc, sim) in rows {
        println!("{name:<10} {acc:.4}  {:+.4}   {sim:.4}", acc - acc_rand);
    }
    // Disclosed estimation-sensitivity rows (NOT gated): the sample Cov's
    // O(1/√N) decoy↔planted cross-covariance is amplified by Cov^{1/2}
    // (×a_dec) into the mapped-back Q — the plan's G6 fixture warning,
    // measured.
    for (label, cov_s) in [
        ("ACE(sampleCov N=4096)", bank.cov_sample.clone()),
        ("ACE(sampleCov N=65536)", sample_cov(&w, 65_536, 0x640_C0F2)),
    ] {
        let (qs, _) = ace(&cov_s, &bank.sens, R);
        let acc = eval_arm(&w, &bank, &to_f32_basis(&qs), R, false);
        println!(
            "{label:<22} acc {acc:.4} contrast {:+.4} sim {:.4} decoy-leak {:.2e}",
            acc - acc_rand,
            sim64(&qs, &w.q_p, R),
            sim64(&qs, &w.u_dec, R) * R as f64
        );
    }
    println!("residual(ACE) {acc_resid:.4}   PCA top-1 · decoy = {pca_top1_dec:.4}");
    print!("ACE r-sweep:");
    for k in 1..=R + 2 {
        let (qk, _) = ace(&bank.cov, &bank.sens, k);
        print!(
            " k={k}:{:.4}",
            eval_arm(&w, &bank, &to_f32_basis(&qk), k, false)
        );
    }
    println!();

    // ── G1 ──────────────────────────────────────────────────────────────
    assert!(sim_ace >= 0.90, "G1: ACE sim vs planted {sim_ace} < 0.90");
    assert!(
        acc_ace >= 0.9 * base.full,
        "G1: aligned {acc_ace} < 0.9·full"
    );
    assert!(
        acc_resid <= base.chance + 0.05,
        "G1: residual {acc_resid} > chance+0.05 ({})",
        base.chance
    );
    // ── G2 label-free parity (the falsifier) ───────────────────────────
    let (c_ace, c_pf, c_p8) = (acc_ace - acc_rand, acc_pf - acc_rand, acc_p8 - acc_rand);
    assert!(
        c_ace >= c_pf - 0.05,
        "G2: ACE contrast {c_ace} < probe {c_pf} − 0.05"
    );
    // ── G5 label scarcity (strict) ──────────────────────────────────────
    assert!(
        c_ace >= c_p8 + 0.05,
        "G5: ACE contrast {c_ace} < probe@8 {c_p8} + 0.05"
    );
    // ── G6 decoy variance ───────────────────────────────────────────────
    assert!(
        pca_top1_dec >= 0.9,
        "G6: PCA top-1 · decoy {pca_top1_dec} < 0.9"
    );
    assert!(
        sim_ace >= 0.90,
        "G6: ACE sim vs planted {sim_ace} < 0.90 under the decoy"
    );

    // ── G3 isotropic-sensitivity positive control: ACE ≡ PCA ────────────
    let ident: Vec<f64> = (0..D * D)
        .map(|t| if t / D == t % D { 1.0 } else { 0.0 })
        .collect();
    // Same Cov on both sides (the sample Cov of the data PCA reads), and
    // only at boundaries with a real spectral gap: k = 1 (the decoy) and
    // k = R + 1 (decoy + planted vs noise). k = R cuts INSIDE the
    // degenerate planted eigenspace (three eigenvalues 1 + σ²), where any
    // basis is arbitrary — not a valid ordering test.
    let pca_wide = top_right_singular(&bank.x_cov_f32, N_COV, R + 1);
    for k in [1, R + 1] {
        let (q_iso, _) = ace(&bank.cov_sample, &ident, k);
        let sim_iso = sim64(&q_iso, &pca_wide[..k * D], k);
        println!("G3 S=I k={k}: sim(ACE, PCA) = {sim_iso:.6}");
        assert!(
            sim_iso >= 0.99,
            "G3: ACE(S=I) vs PCA at k={k}: {sim_iso} < 0.99"
        );
    }

    // ── G4 allocation ───────────────────────────────────────────────────
    let (_, fit_allocs) = counted(|| ace(&bank.cov, &bank.sens, R));
    let mut sc = CoreScratch::new(D);
    let (mut q, mut e) = (vec![0.0; D * D], vec![0.0; D]);
    let mut z = vec![0.0f32; N_TEST * D];
    let mut coeffs = vec![0.0f32; R];
    let (_, extract_allocs) = counted(|| {
        extract_core_from_matrices(
            &bank.cov,
            &bank.sens,
            D,
            RankRule::Fixed(R),
            &mut q,
            &mut e,
            &mut sc,
        )
        .unwrap()
    });
    let (_, eval_allocs) = counted(|| {
        project_through_into(
            &bank.x_test,
            &mut z,
            &ace_f32,
            N_TEST,
            D,
            R,
            false,
            &mut coeffs,
        );
        head_acc(&w, &z, &bank.y_test, N_TEST, &mut row)
    });
    println!(
        "G4 allocs: fit-with-scratch-build={fit_allocs} extract(warm scratch)={extract_allocs} eval={eval_allocs}"
    );
    assert_eq!(
        extract_allocs, 0,
        "G4: extract must be zero-alloc on warm scratch"
    );
    assert_eq!(eval_allocs, 0, "G4: eval path must be zero-alloc");
}
