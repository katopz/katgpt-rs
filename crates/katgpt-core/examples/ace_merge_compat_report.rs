//! Issue 1235 row (b) — core-space merge-compatibility diagnostic (REPORT).
//!
//! Research 683 §5: cores across seeds live in near-orthogonal subspaces, so
//! weight interpolation leaves the solution manifold. The diagnostic reads
//! `core_subspace_agreement` on the two models' ACE relevance cores BEFORE
//! any merge. This report proves it separates merges with KNOWN outcomes —
//! the outcome is measured by actually interpolating the weights.
//!
//! Model family (planted, modelless): `h = E x` (`x ~ N(0, I_8)`, `E`
//! diagonal — core axes scale 1.0, complement axes 2.0, i.e. high-variance
//! decoys), frozen head `f(h) = tanh(β·w1·h) + 0.5·tanh(β·w2·h)` with
//! `w1 = e0, w2 = e1`. Model B = `(R·E, R·w)` for an orthogonal `R`: a
//! FUNCTION-PRESERVING reparameterisation (a different "seed" layout), so
//! A and B compute the same function and only the merge can differ.
//! Merge = elementwise midpoint of `(E, w1, w2)`; merge loss =
//! `E[(f_mid − f_A)²] / Var f_A` over held-out samples.
//!
//! Arms: `R = I` · complement-only rotation (e4↔e6, e5↔e7 by 90°; the head
//! never reads those axes → merge is safe) · core rotation sweep θ ∈
//! {0..90°} mixing (e0, e4) and (e1, e5) (θ = 90° puts B's core orthogonal
//! to A's). Baseline: raw weight-space cosine over the flattened params —
//! the rotation-blind read the diagnostic replaces.
//!
//! Exit 1 iff the diagnostic fails to separate the known-good pair
//! (complement rotation) from the known-bad pair (core rotation 90°), or
//! fails to rank the core sweep in the merge-loss order.
//!
//! Run: `cargo run -p katgpt-core --example ace_merge_compat_report --features relevance_core --release`

use katgpt_core::relevance_core::{
    CoreScratch, RankRule, accumulate_sens, core_subspace_agreement, extract_core_from_matrices,
};

const D: usize = 8;
const N_FIT: usize = 4000;
const N_EVAL: usize = 4000;
const BETA: f64 = 1.5;

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

type Mat = [[f64; D]; D];

#[derive(Clone)]
struct Model {
    e: Mat,
    w: [[f64; D]; 2],
}

const HEAD_COEF: [f64; 2] = [1.0, 0.5];

impl Model {
    fn base() -> Self {
        let mut e = [[0.0; D]; D];
        for (i, row) in e.iter_mut().enumerate() {
            row[i] = if i < 4 { 1.0 } else { 2.0 };
        }
        let mut w = [[0.0; D]; 2];
        w[0][0] = 1.0;
        w[1][1] = 1.0;
        Self { e, w }
    }
    fn hidden(&self, x: &[f64; D]) -> [f64; D] {
        std::array::from_fn(|i| (0..D).map(|k| self.e[i][k] * x[k]).sum())
    }
    fn head(&self, h: &[f64; D]) -> f64 {
        (0..2)
            .map(|c| HEAD_COEF[c] * (BETA * dot(&self.w[c], h)).tanh())
            .sum()
    }
    /// ∂f/∂h at `h`.
    fn jac(&self, h: &[f64; D]) -> [f64; D] {
        let mut j = [0.0; D];
        for (wc, coef) in self.w.iter().zip(HEAD_COEF) {
            let t = (BETA * dot(wc, h)).tanh();
            let g = coef * BETA * (1.0 - t * t);
            for (jk, wk) in j.iter_mut().zip(wc) {
                *jk += g * wk;
            }
        }
        j
    }
    fn rotated(&self, r: &Mat) -> Self {
        Self {
            e: matmul(r, &self.e),
            w: [matvec(r, &self.w[0]), matvec(r, &self.w[1])],
        }
    }
    fn midpoint(&self, o: &Self) -> Self {
        Self {
            e: std::array::from_fn(|i| std::array::from_fn(|k| 0.5 * (self.e[i][k] + o.e[i][k]))),
            w: std::array::from_fn(|c| std::array::from_fn(|k| 0.5 * (self.w[c][k] + o.w[c][k]))),
        }
    }
    fn flat(&self) -> Vec<f64> {
        self.e
            .iter()
            .flatten()
            .chain(self.w.iter().flatten())
            .copied()
            .collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn matvec(r: &Mat, v: &[f64; D]) -> [f64; D] {
    std::array::from_fn(|i| dot(&r[i], v))
}
fn matmul(a: &Mat, b: &Mat) -> Mat {
    std::array::from_fn(|i| std::array::from_fn(|k| (0..D).map(|l| a[i][l] * b[l][k]).sum()))
}
fn identity() -> Mat {
    std::array::from_fn(|i| std::array::from_fn(|k| if i == k { 1.0 } else { 0.0 }))
}
/// Givens rotations by `theta` in each `(p, q)` plane.
fn givens(planes: &[(usize, usize)], theta: f64) -> Mat {
    let mut r = identity();
    let (c, s) = (theta.cos(), theta.sin());
    for &(p, q) in planes {
        r[p][p] = c;
        r[p][q] = -s;
        r[q][p] = s;
        r[q][q] = c;
    }
    r
}

/// ACE relevance core of `m` (row-major `r × D`) via the shipped extractor.
fn core_of(m: &Model, seed: u64) -> (Vec<f64>, usize) {
    let mut rng = Lcg(seed);
    let mut cov = vec![0.0; D * D];
    let mut sens = vec![0.0; D * D];
    let mut n_seen = 0u64;
    for _ in 0..N_FIT {
        let x: [f64; D] = std::array::from_fn(|_| rng.gauss());
        let h = m.hidden(&x);
        for i in 0..D {
            for k in 0..D {
                cov[i * D + k] += h[i] * h[k] / N_FIT as f64;
            }
        }
        accumulate_sens(&mut sens, D, &m.jac(&h), &mut n_seen).expect("sens");
    }
    let mut q = vec![0.0; D * D];
    let mut en = vec![0.0; D];
    let mut scratch = CoreScratch::new(D);
    let ex = extract_core_from_matrices(
        &cov,
        &sens,
        D,
        RankRule::Energy(0.999),
        &mut q,
        &mut en,
        &mut scratch,
    )
    .expect("extract");
    (q, ex.r)
}

/// Merge outcome: relative loss of the weight midpoint vs model A.
fn merge_loss(a: &Model, b: &Model, seed: u64) -> f64 {
    let mid = a.midpoint(b);
    let mut rng = Lcg(seed);
    let (mut se, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for _ in 0..N_EVAL {
        let x: [f64; D] = std::array::from_fn(|_| rng.gauss());
        let fa = a.head(&a.hidden(&x));
        let fm = mid.head(&mid.hidden(&x));
        se += (fm - fa) * (fm - fa);
        s1 += fa;
        s2 += fa * fa;
    }
    let n = N_EVAL as f64;
    (se / n) / (s2 / n - (s1 / n).powi(2))
}

fn weight_cosine(a: &Model, b: &Model) -> f64 {
    let (fa, fb) = (a.flat(), b.flat());
    dot(&fa, &fb) / (dot(&fa, &fa).sqrt() * dot(&fb, &fb).sqrt())
}

struct Row {
    name: String,
    agree: f64,
    wcos: f64,
    loss: f64,
}

fn measure(name: &str, a: &Model, core_a: &(Vec<f64>, usize), r: &Mat) -> Row {
    let b = a.rotated(r);
    let core_b = core_of(&b, 0xB);
    let agree =
        core_subspace_agreement(&core_a.0, core_a.1, &core_b.0, core_b.1, D).expect("agree");
    Row {
        name: name.to_string(),
        agree,
        wcos: weight_cosine(a, &b),
        loss: merge_loss(a, &b, 0xE),
    }
}

fn main() {
    let a = Model::base();
    let core_a = core_of(&a, 0xA);
    println!("model A core rank r = {} (planted 2)", core_a.1);

    let rows = [
        measure("identity", &a, &core_a, &identity()),
        measure(
            "complement-rot 90° (KNOWN GOOD)",
            &a,
            &core_a,
            &givens(&[(4, 6), (5, 7)], core::f64::consts::FRAC_PI_2),
        ),
    ];
    let sweep: Vec<Row> = [0u32, 15, 30, 45, 60, 75, 90]
        .iter()
        .map(|&deg| {
            let label = if deg == 90 {
                "core-rot 90° (KNOWN BAD)".to_string()
            } else {
                format!("core-rot {deg}°")
            };
            measure(
                &label,
                &a,
                &core_a,
                &givens(&[(0, 4), (1, 5)], (deg as f64).to_radians()),
            )
        })
        .collect();

    println!(
        "\n{:<34} {:>10} {:>12} {:>12}",
        "pair (B = R·A)", "core_agree", "weight_cos", "merge_loss"
    );
    for r in rows.iter().chain(&sweep) {
        println!(
            "{:<34} {:>10.4} {:>12.4} {:>12.4}",
            r.name, r.agree, r.wcos, r.loss
        );
    }

    let good = &rows[1];
    let bad = sweep.last().expect("sweep");
    let mut ok = true;

    // Separation of the known pair.
    let sep = good.agree - bad.agree;
    println!(
        "\nseparation: core_agree good−bad = {sep:.4} (merge_loss good {:.4} vs bad {:.4})",
        good.loss, bad.loss
    );
    if !(good.loss < bad.loss && sep >= 0.5) {
        println!("FAIL: diagnostic does not separate the known-good from the known-bad merge");
        ok = false;
    }
    // Baseline: does raw weight cosine misrank the known pair?
    let baseline_misranks = good.wcos <= bad.wcos;
    println!(
        "baseline weight_cos: good {:.4} vs bad {:.4} → {}",
        good.wcos,
        bad.wcos,
        if baseline_misranks {
            "MISRANKS (rotation-blind)"
        } else {
            "ranks correctly"
        }
    );
    // Sweep: agreement must be non-increasing while merge loss is
    // non-decreasing (rank agreement with the measured outcome).
    let monotone = sweep
        .windows(2)
        .all(|w| w[1].agree <= w[0].agree + 1e-9 && w[1].loss >= w[0].loss - 1e-9);
    println!("core sweep: agreement↓ ⇔ merge_loss↑ monotone = {monotone}");
    if !monotone {
        println!("FAIL: core agreement does not rank the sweep in merge-loss order");
        ok = false;
    }
    if !ok {
        std::process::exit(1);
    }
    println!("VERDICT: core-space agreement separates the known merge outcomes (report-only).");
}
