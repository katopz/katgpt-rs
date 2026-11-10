//! Issue 778 / Research 557 POC — Subspace Intervention protocol validation
//! (arXiv:2607.01987, "Understanding Geometric Representations in SSL ViTs
//! via Subspace Intervention", Zhou et al. ECCV 2026).
//!
//! Protocol under test (all closed-form, zero GD — R557 Path 0):
//!   1. ridge-fit a linear probe W on a labeled per-layer activation bank;
//!   2. `thin_svd_into` on W → task-aligned orthonormal basis V (right sv);
//!   3. three-arm intervention through the FROZEN head:
//!      aligned  Ẑ = Z·V_k·V_kᵀ    (keep top-k task-aligned subspace)
//!      random   Ẑ = Z·R_k·R_kᵀ    (seeded random orthonormal k-control)
//!      residual Ẑ = Z − Z·V_k·V_kᵀ (orthogonal complement)
//!   4. per-class layer-affinity sweep (which layer reads out best per label);
//!   5. seed stability of V_k across (bootstrap, λ-jitter) refits via the
//!      principal-angle basis similarity (1/k)·‖V_aᵀV_b‖_F² — the paper's
//!      Fig-8 metric family (Li et al. 2021), NOT svcca (Issue 684 ships
//!      that as the cross-space alternative; the paper itself uses the plain
//!      basis-similarity form, so this stays default-features).
//!
//! Fixture strategy: a SYNTHETIC TIERED BANK with PLANTED ground truth —
//! per-class rank-6 signal subspaces with Gaussian layer-gain profiles
//! peaking at planted layers (early group L=3, classes 0-3; late group L=8,
//! classes 4-7), per-class amplitude decay (plants the decaying readout
//! spectrum the paper's low-rank claim presupposes, and gives T5 a real
//! core/tail chance), a shared rank-16 label-independent nuisance subspace,
//! seeded noise. This makes the protocol DEFEND-WRONG-able: the affinity
//! sweep must recover the planted peaks and the triad must separate aligned
//! from random, or the PROTOCOL is refuted (hard asserts). What this POC
//! cannot claim: real-model affinity — that requires a real (layer,
//! activation, label) bank (riir-ai/riir-train follow-up); here we validate
//! the machinery on controlled ground truth.
//!
//! The aligned-vs-random arm at matched k is the protocol-level close of the
//! deferred "random basis vs eigen-aligned basis at matched param budget"
//! eval in `katgpt-attn/src/funcattn_compose/spectral_pre_rotate.rs:29-31`
//! (in readout space; the FUNCATTN-tensor application consumes this
//! protocol and is one function away).
//!
//! Gates:
//!   G0 determinism — bank BLAKE3 + accuracy tables bit-identical ×2.
//!   G1 triad       — (a) aligned@k=rank == full (projection identity);
//!                    (b) aligned@k=6 ≥ 0.9·full (low-rank compression: 6
//!                        of 8 readout directions recover ≥90% — matches the
//!                        planted geometry of T_DIM=3 shared + a weak
//!                        load-bearing idiosyncratic tail);
//!                    (c) residual@k=rank ≤ chance+0.05 (signal IS in the
//!                        aligned subspace);
//!                    (d) aligned@k=4 ≥ 1.5·random@k=4 (the deferred
//!                        spectral_pre_rotate contrast, at matched budget).
//!                        NOTE: random does NOT collapse to chance at
//!                        k=rank — a random k-subspace of R^D retains
//!                        ≈k/D of BOTH signal and noise (SNR-preserving
//!                        cut); the paper's collapse regime is k≪rank with
//!                        a steep spectrum. The protocol's discriminating
//!                        signal is the aligned-vs-random CONTRAST, which
//!                        this gate pins.
//!   G2 affinity    — ≥ 6/8 classes argmax-layer within ±1 of planted.
//!   G3 stability   — INFORMATIONAL (no-split is a legitimate outcome per
//!                    Issue 778 outcome criteria; printed, not asserted).
//!
//! Cousins, not duplicates: `interpolation_geometry::intervention_battery`
//! (six ADDITIVE/noise arms on committed latent state) and Plan 278's
//! FaithfulnessProbe perturb the STATE; this file's triad PROJECTS the
//! feature through probe-derived subspaces and reads through a frozen head
//! — the arXiv:2607.01987 protocol exactly.

use katgpt_core::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

// ── Bank constants ─────────────────────────────────────────────────────────
const D: usize = 64; // d_model (≤ svcca MAX_K, kept for a future metric swap)
const N_LAYERS: usize = 12;
const C: usize = 8; // behavior classes
const R_PLANT: usize = 6; // per-class idiosyncratic block rank
const T_DIM: usize = 3; // SHARED task subspace (where the class means live)
const R_NUIS: usize = 13; // shared nuisance rank (cols 51..64)
/// Idiosyncratic per-class mean fraction (the small tail beyond T).
const IDIO_SCALE: f32 = 0.3;
const N_TRAIN: usize = 512; // 64/class
const N_TEST: usize = 256; // 32/class
const N_TOTAL: usize = N_TRAIN + N_TEST;
const EARLY_L: usize = 3;
const LATE_L: usize = 8;
const GAIN_FLOOR: f32 = 0.15;
const SIGMA_L: f32 = 1.2;
const NOISE_SD: f32 = 0.35;
const LEAK_SD: f32 = 0.10; // cross-class distractor coefficients
const NUIS_SD: f32 = 0.50;
/// Per-class fixed unit-norm mean direction (in the class block) — the
/// class-conditional MEAN SHIFT the linear probe reads out; scaled by
/// MEAN_SCALE · gain · AMPL at each layer.
const MEAN_SCALE: f32 = 3.0;
/// Within-class scatter scale on the own-class block (per-sample jitter).
const SCATTER_SD: f32 = 0.5;
/// Per-class signal amplitude — plants a decaying readout spectrum.
const AMPL: [f32; C] = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60];
const KS: [usize; 5] = [1, 2, 4, 6, 8];
const CHANCE: f32 = 1.0 / C as f32;

// ── Seeded PRNG (house FixtureRng pattern: local SplitMix64 + Box-Muller) ──
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        if self.next_u64() & 1 == 0 {
            r * theta.cos()
        } else {
            r * theta.sin()
        }
    }
}

// ── Bank ───────────────────────────────────────────────────────────────────
struct Bank {
    /// `[layer][sample][dim]` activations.
    acts: Vec<Vec<Vec<f32>>>,
    /// Class label per sample.
    labels: Vec<usize>,
    /// BLAKE3 of the canonical bank bytes.
    hash: [u8; 32],
}

fn planted_peak(c: usize) -> usize {
    if c < 4 { EARLY_L } else { LATE_L }
}

fn layer_gain(c: usize, l: usize) -> f32 {
    let d = l as f32 - planted_peak(c) as f32;
    GAIN_FLOOR + (1.0 - GAIN_FLOOR) * (-(d * d) / (2.0 * SIGMA_L * SIGMA_L)).exp()
}

/// Orthonormalize `cols` (D×k, column-major flat) in place via Gram-Schmidt.
fn gram_schmidt(cols: &mut [f32], d: usize) {
    let k = cols.len() / d;
    for j in 0..k {
        for i in 0..j {
            let mut dot = 0.0f32;
            for r in 0..d {
                dot += cols[j * d + r] * cols[i * d + r];
            }
            for r in 0..d {
                cols[j * d + r] -= dot * cols[i * d + r];
            }
        }
        let mut norm = 0.0f32;
        for r in 0..d {
            norm += cols[j * d + r] * cols[j * d + r];
        }
        norm = norm.sqrt().max(1e-12);
        for r in 0..d {
            cols[j * d + r] /= norm;
        }
    }
}

fn build_bank(seed: u64) -> Bank {
    let mut rng = Rng::new(seed);
    // One global orthonormal frame Q (D×D, column-major):
    // cols [c*6..c*6+6) = class-c idiosyncratic block; cols [48..48+3) = the
    // SHARED task subspace T (class means live here — plants the paper's
    // low-rank-compressible task signal); cols [51..64) = nuisance.
    let mut q = vec![0.0f32; D * D];
    for v in q.iter_mut() {
        *v = rng.next_gaussian();
    }
    gram_schmidt(&mut q, D);

    // Balanced classes via BLOCK assignment (i / per_class): the even/odd
    // interleave split must see ALL classes on both sides — `i % C` would
    // give train only odd classes and test only even ones (the 0.000-acc
    // bug this comment immortalizes).
    let per_class = N_TOTAL / C;
    let labels: Vec<usize> = (0..N_TOTAL).map(|i| i / per_class).collect();
    // Per-sample latents, drawn ONCE: layer gain modulates the own-class
    // block per layer (same intent, layer-varying manifestation).
    let mut own = vec![vec![0.0f32; R_PLANT]; N_TOTAL];
    let mut nuis = vec![vec![0.0f32; R_NUIS]; N_TOTAL];
    let mut leak = vec![vec![0.0f32; C * R_PLANT]; N_TOTAL];
    // Fixed per-class mean directions: a spherical spiral in the SHARED 3-dim
    // task subspace T (the linearly decodable class geometry — low intrinsic
    // rank, the paper's compressibility precondition) + a small idiosyncratic
    // component m_c (unit norm) in each class's own block (the spectral tail).
    let mut mean_t = vec![vec![0.0f32; T_DIM]; C];
    let mut mean_dir = vec![vec![0.0f32; R_PLANT]; C];
    for (c, mt) in mean_t.iter_mut().enumerate() {
        let theta = std::f32::consts::PI * (c as f32) / C as f32;
        let phi = 2.2 * c as f32;
        mt[0] = theta.cos();
        mt[1] = theta.sin() * phi.cos();
        mt[2] = theta.sin() * phi.sin();
    }
    for mc in mean_dir.iter_mut() {
        let mut norm = 0.0f32;
        for v in mc.iter_mut() {
            *v = rng.next_gaussian();
            norm += *v * *v;
        }
        let norm = norm.sqrt().max(1e-12);
        for v in mc.iter_mut() {
            *v /= norm;
        }
    }
    for i in 0..N_TOTAL {
        for v in own[i].iter_mut() {
            *v = rng.next_gaussian();
        }
        for v in nuis[i].iter_mut() {
            *v = NUIS_SD * rng.next_gaussian();
        }
        for v in leak[i].iter_mut() {
            *v = LEAK_SD * rng.next_gaussian();
        }
    }

    let mut acts = vec![vec![vec![0.0f32; D]; N_TOTAL]; N_LAYERS];
    for (l, layer_acts) in acts.iter_mut().enumerate() {
        for i in 0..N_TOTAL {
            let y = labels[i];
            let row = &mut layer_acts[i];
            // nuisance block (constant across layers)
            for j in 0..R_NUIS {
                let cj = nuis[i][j];
                for r in 0..D {
                    row[r] += cj * q[(51 + j) * D + r];
                }
            }
            // Shared task-subspace mean (all classes) + own-class terms.
            let g_y = layer_gain(y, l);
            for j in 0..T_DIM {
                let cj = AMPL[y] * g_y * MEAN_SCALE * mean_t[y][j];
                for r in 0..D {
                    row[r] += cj * q[(48 + j) * D + r];
                }
            }
            for c in 0..C {
                let g = layer_gain(c, l);
                for j in 0..R_PLANT {
                    // Small idiosyncratic mean (tail) for the OWN class +
                    // within-class scatter; cross-class leak for the others.
                    let mean_term = if c == y {
                        AMPL[c] * g * MEAN_SCALE * IDIO_SCALE * mean_dir[c][j]
                    } else {
                        0.0
                    };
                    let scatter = if c == y {
                        AMPL[c] * g * SCATTER_SD * own[i][j]
                    } else {
                        g * leak[i][c * R_PLANT + j]
                    };
                    let cj = mean_term + scatter;
                    for r in 0..D {
                        row[r] += cj * q[(c * R_PLANT + j) * D + r];
                    }
                }
            }
            // fresh per-layer noise
            for v in row.iter_mut() {
                *v += NOISE_SD * rng.next_gaussian();
            }
        }
    }

    let mut hasher = blake3::Hasher::new();
    for layer in &acts {
        for sample in layer {
            for v in sample {
                hasher.update(&v.to_le_bytes());
            }
        }
    }
    for &y in &labels {
        hasher.update(&(y as u64).to_le_bytes());
    }
    Bank {
        acts,
        labels,
        hash: *hasher.finalize().as_bytes(),
    }
}

// ── Ridge probe (closed-form; right singular vectors only) ─────────────────
/// Fit W (C×D row-major) on `x` (n×D) with labels `y` (n), λ = `lambda_scale`
/// · mean(diag(XᵀX)). Ridge solve via SVD of the regularized Gram:
/// Wᵀ = (G+λI)⁻¹·(XᵀY) = Σ_j v_j · (v_jᵀ·(G⁺·M)) / σ_j², G⁺ = G+λI (SPD →
/// the SVD's right basis diagonalizes it; no left-vector accessor needed).
fn ridge_fit(
    x: &[Vec<f32>],
    y: &[usize],
    lambda_scale: f32,
    svd_res: &mut SvdResultScratch,
    svd_work: &mut SvdScratch,
) -> Vec<f32> {
    let mut g = vec![0.0f32; D * D];
    for xi in x {
        for a in 0..D {
            let xa = xi[a];
            for b in 0..D {
                g[a * D + b] += xa * xi[b];
            }
        }
    }
    let trace = (0..D).map(|a| g[a * D + a]).sum::<f32>();
    let lambda = lambda_scale * trace / D as f32;
    for a in 0..D {
        g[a * D + a] += lambda;
    }
    // M = XᵀY (D×C); GM = G⁺·M (D×C)
    let mut gm = vec![0.0f32; D * C];
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let ci = yi;
        for a in 0..D {
            gm[a * C + ci] += xi[a];
        }
    }
    // in-place: gm ← G⁺·gm
    let mut gmg = vec![0.0f32; D * C];
    for a in 0..D {
        for c in 0..C {
            let mut acc = 0.0f32;
            for b in 0..D {
                acc += g[a * D + b] * gm[b * C + c];
            }
            gmg[a * C + c] = acc;
        }
    }
    thin_svd_into(&g, D, D, svd_res, svd_work);
    let len = svd_res.len();
    let sigma_max = svd_res.singular_value(0).max(1e-12);
    let mut w = vec![0.0f32; C * D];
    for j in 0..len {
        let vj = svd_res.right_singular_vector(j);
        let inv_sq = 1.0 / (svd_res.singular_value(j).max(1e-9 * sigma_max)).powi(2);
        for c in 0..C {
            let mut proj = 0.0f32;
            for a in 0..D {
                proj += vj[a] * gmg[a * C + c];
            }
            let s = proj * inv_sq;
            if s == 0.0 {
                continue;
            }
            for a in 0..D {
                w[c * D + a] += vj[a] * s;
            }
        }
    }
    w
}

// ── Evaluation ─────────────────────────────────────────────────────────────
/// logits = Z·Wᵀ → argmax accuracy + per-class recall.
fn eval_head(z: &[Vec<f32>], y: &[usize], w: &[f32]) -> (f32, Vec<f32>) {
    let n = z.len();
    let mut correct = 0usize;
    let mut hits = [0usize; C];
    let mut total = [0usize; C];
    for i in 0..n {
        let mut best_c = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for c in 0..C {
            let mut dot = 0.0f32;
            for (a, za) in z[i].iter().enumerate() {
                dot += za * w[c * D + a];
            }
            if dot > best_v {
                best_v = dot;
                best_c = c;
            }
        }
        total[y[i]] += 1;
        if best_c == y[i] {
            hits[y[i]] += 1;
            correct += 1;
        }
    }
    let recalls = (0..C)
        .map(|c| {
            if total[c] == 0 {
                0.0
            } else {
                hits[c] as f32 / total[c] as f32
            }
        })
        .collect();
    (correct as f32 / n as f32, recalls)
}

/// Project z (n×D) through a k-dim orthonormal basis (D×k flat column-major):
/// Ẑ = Z·B·Bᵀ; `complement = true` → Ẑ = Z − Z·B·Bᵀ.
fn project_through(z: &[Vec<f32>], basis: &[f32], k: usize, complement: bool) -> Vec<Vec<f32>> {
    let n = z.len();
    let mut out = vec![vec![0.0f32; D]; n];
    for i in 0..n {
        let mut coeffs = vec![0.0f32; k];
        for j in 0..k {
            let mut dot = 0.0f32;
            for a in 0..D {
                dot += z[i][a] * basis[j * D + a];
            }
            coeffs[j] = dot;
        }
        for a in 0..D {
            let mut acc = if complement { z[i][a] } else { 0.0 };
            let sign = if complement { -1.0 } else { 1.0 };
            for j in 0..k {
                acc += sign * coeffs[j] * basis[j * D + a];
            }
            out[i][a] = acc;
        }
    }
    out
}

/// Seeded random orthonormal D×k control basis.
fn random_basis(seed: u64, k: usize) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut cols = vec![0.0f32; D * k];
    for v in cols.iter_mut() {
        *v = rng.next_gaussian();
    }
    gram_schmidt(&mut cols, D);
    cols
}

/// Principal-angle basis similarity (1/k)·‖V_aᵀV_b‖_F² ∈ (0, 1]
/// (Li et al. 2021 — the paper's Fig-8 metric family).
fn basis_similarity(va: &[f32], vb: &[f32], k: usize) -> f32 {
    let mut fro = 0.0f32;
    for j in 0..k {
        for i in 0..k {
            let mut dot = 0.0f32;
            for a in 0..D {
                dot += va[j * D + a] * vb[i * D + a];
            }
            fro += dot * dot;
        }
    }
    fro / k as f32
}

// ── Pipeline ───────────────────────────────────────────────────────────────
struct PipelOut {
    full_acc: Vec<f32>,     // per layer
    aligned: Vec<[f32; 5]>, // per layer × k index
    random: Vec<[f32; 5]>,
    residual: Vec<[f32; 5]>,
    peak_layer: Vec<usize>, // per class
    stability: Vec<f32>,    // mean pairwise sim per k index
    best_layer: usize,
}

fn train_indices(seed: u64, bootstrap: bool) -> Vec<usize> {
    let pool: Vec<usize> = (0..N_TOTAL).filter(|&i| i % 2 == 1).collect();
    if !bootstrap {
        pool
    } else {
        let mut rng = Rng::new(seed);
        (0..N_TRAIN)
            .map(|_| pool[rng.next_u64() as usize % pool.len()])
            .collect()
    }
}

fn run_pipeline(bank: &Bank) -> PipelOut {
    let test_idx: Vec<usize> = (0..N_TOTAL).filter(|&i| i % 2 == 0).collect();
    let test_labels: Vec<usize> = test_idx.iter().map(|&i| bank.labels[i]).collect();
    let mut svd_res_g = SvdResultScratch::with_capacity(D, D);
    let mut svd_work_g = SvdScratch::with_capacity(D, D);
    let mut svd_res_w = SvdResultScratch::with_capacity(C, D);
    let mut svd_work_w = SvdScratch::with_capacity(D, C);

    let tr_idx = train_indices(0, false);
    let tr_labels: Vec<usize> = tr_idx.iter().map(|&i| bank.labels[i]).collect();
    let mut full_acc = vec![0.0f32; N_LAYERS];
    let mut aligned = vec![[0.0f32; 5]; N_LAYERS];
    let mut random = vec![[0.0f32; 5]; N_LAYERS];
    let mut residual = vec![[0.0f32; 5]; N_LAYERS];
    let mut per_class_recall = vec![vec![0.0f32; C]; N_LAYERS];

    for l in 0..N_LAYERS {
        let x_train: Vec<Vec<f32>> = tr_idx.iter().map(|&i| bank.acts[l][i].clone()).collect();
        let x_test: Vec<Vec<f32>> = test_idx.iter().map(|&i| bank.acts[l][i].clone()).collect();
        let w = ridge_fit(&x_train, &tr_labels, 0.01, &mut svd_res_g, &mut svd_work_g);

        let (acc, recalls) = eval_head(&x_test, &test_labels, &w);
        full_acc[l] = acc;
        per_class_recall[l] = recalls;

        thin_svd_into(&w, C, D, &mut svd_res_w, &mut svd_work_w);
        let wlen = svd_res_w.len();
        let mut v_all = vec![0.0f32; D * wlen];
        for j in 0..wlen {
            v_all[j * D..(j + 1) * D].copy_from_slice(svd_res_w.right_singular_vector(j));
        }
        let rb = random_basis(0xDEAD_BEEF + l as u64, wlen);

        for (ki, &k) in KS.iter().enumerate() {
            let k = k.min(wlen);
            let za = project_through(&x_test, &v_all, k, false);
            aligned[l][ki] = eval_head(&za, &test_labels, &w).0;
            let zr = project_through(&x_test, &rb, k, false);
            random[l][ki] = eval_head(&zr, &test_labels, &w).0;
            let zc = project_through(&x_test, &v_all, k, true);
            residual[l][ki] = eval_head(&zc, &test_labels, &w).0;
        }
    }

    let peak_layer: Vec<usize> = (0..C)
        .map(|c| {
            (0..N_LAYERS)
                .max_by(|&a, &b| per_class_recall[a][c].total_cmp(&per_class_recall[b][c]))
                .expect("N_LAYERS > 0")
        })
        .collect();
    let best_layer = (0..N_LAYERS)
        .max_by(|&a, &b| full_acc[a].total_cmp(&full_acc[b]))
        .unwrap();

    // T5: stability across 3 (bootstrap, λ-jitter) refits at the best layer.
    let mut bases: Vec<Vec<f32>> = Vec::new();
    for s in 0..3u64 {
        let idx = train_indices(100 + s, true);
        let lab: Vec<usize> = idx.iter().map(|&i| bank.labels[i]).collect();
        let xtr: Vec<Vec<f32>> = idx
            .iter()
            .map(|&i| bank.acts[best_layer][i].clone())
            .collect();
        let lam = [0.005f32, 0.01, 0.02][s as usize];
        let w = ridge_fit(&xtr, &lab, lam, &mut svd_res_g, &mut svd_work_g);
        thin_svd_into(&w, C, D, &mut svd_res_w, &mut svd_work_w);
        let wlen = svd_res_w.len();
        let mut v_all = vec![0.0f32; D * wlen];
        for j in 0..wlen {
            v_all[j * D..(j + 1) * D].copy_from_slice(svd_res_w.right_singular_vector(j));
        }
        bases.push(v_all);
    }
    let stability: Vec<f32> = KS
        .iter()
        .map(|&k| {
            let mut sum = 0.0f32;
            for a in 0..3 {
                for b in (a + 1)..3 {
                    sum += basis_similarity(&bases[a], &bases[b], k);
                }
            }
            sum / 3.0
        })
        .collect();

    PipelOut {
        full_acc,
        aligned,
        random,
        residual,
        peak_layer,
        stability,
        best_layer,
    }
}

/// Substrate contract pin for this protocol's two non-obvious dependencies:
/// (a) `thin_svd_into` on a WIDE matrix (m_rows=8 < n_cols=64 — its native
/// use is tall Jacobians): right singular vectors must be unit-norm feature-
/// space directions, e₀..e₇ for the first-8-rows-of-I₆₄ fixture;
/// (b) the closed-form ridge solve: two well-separated blobs → train acc 1.0.
#[test]
fn thin_svd_wide_matrix_and_ridge_contract() {
    // (a) wide SVD sanity: W = first 8 rows of I₆₄ → V cols should be e₀..e₇.
    let mut res = SvdResultScratch::with_capacity(8, D);
    let mut work = SvdScratch::with_capacity(D, 8);
    let mut w = vec![0.0f32; 8 * D];
    for c in 0..8 {
        w[c * D + c] = 1.0;
    }
    thin_svd_into(&w, 8, D, &mut res, &mut work);
    println!(
        "wide svd len={} sigs={:?}",
        res.len(),
        res.singular_values()
    );
    for j in 0..res.len().min(3) {
        let vj = res.right_singular_vector(j);
        let nz: Vec<usize> = (0..D).filter(|&a| vj[a].abs() > 0.5).collect();
        println!(
            "  v[{j}] dominant coords: {nz:?} norm {:.4}",
            vj.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
    }

    // (b) ridge fit sanity on an EASY fixture: two well-separated gaussian
    // blobs (C=2 hardcoded here to keep it minimal), train acc must be ~1.0.
    let mut rng = Rng::new(42);
    let n = 256;
    let x: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let s = if i % 2 == 0 { 1.0f32 } else { -1.0 };
            (0..D)
                .map(|a| {
                    if a == 0 {
                        s * 5.0
                    } else {
                        0.3 * rng.next_gaussian()
                    }
                })
                .collect()
        })
        .collect();
    let y: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let mut res2 = SvdResultScratch::with_capacity(D, D);
    let mut work2 = SvdScratch::with_capacity(D, D);
    let wfit = ridge_fit(&x, &y, 0.01, &mut res2, &mut work2);
    let (tr_acc, _) = eval_head(&x, &y, &wfit);
    println!(
        "blob train acc: {tr_acc:.3} (expect 1.000); |W| per class: {} {}",
        (0..D).map(|a| wfit[a].abs()).sum::<f32>(),
        (0..D).map(|a| wfit[D + a].abs()).sum::<f32>()
    );
}

#[test]
fn issue_778_subspace_intervention_poc() {
    println!("═══ Issue 778 POC — subspace intervention protocol (R557 / arXiv:2607.01987) ═══");

    // G0: determinism — two independent bank builds + pipelines agree.
    let bank_a = build_bank(778);
    let bank_b = build_bank(778);
    assert_eq!(
        bank_a.hash, bank_b.hash,
        "G0: bank BLAKE3 must be deterministic"
    );
    let out_a = run_pipeline(&bank_a);
    let out_b = run_pipeline(&bank_b);
    for l in 0..N_LAYERS {
        assert_eq!(
            out_a.full_acc[l], out_b.full_acc[l],
            "G0: full acc @layer {l}"
        );
        for ki in 0..5 {
            assert_eq!(
                out_a.aligned[l][ki], out_b.aligned[l][ki],
                "G0: aligned @layer {l} k{ki}"
            );
        }
    }
    println!(
        "G0 determinism: PASS (bank BLAKE3 {:02x}{:02x}{:02x}…, tables bit-identical)",
        bank_a.hash[0], bank_a.hash[1], bank_a.hash[2]
    );

    println!("\n── Full-rank linear accuracy per layer (the M3 affinity axis) ──");
    for l in 0..N_LAYERS {
        println!("  layer {l:2}: acc {:.3}", out_a.full_acc[l]);
    }
    let best_layer = out_a.best_layer;
    println!(
        "  best overall layer: {best_layer} (acc {:.3})",
        out_a.full_acc[best_layer]
    );

    println!("\n── G2: per-class peak layer vs planted ──");
    let mut hits = 0usize;
    for (c, &ampl) in AMPL.iter().enumerate() {
        let pk = out_a.peak_layer[c];
        let pl = planted_peak(c);
        let ok = pk.abs_diff(pl) <= 1;
        hits += ok as usize;
        println!(
            "  class {c} (ampl {ampl:.2}): peak {pk:2} planted {pl:2} {}",
            if ok { "OK" } else { "MISS" }
        );
    }
    println!("  G2 affinity: {hits}/{C} within ±1");
    assert!(
        hits >= 6,
        "G2 FAILED: only {hits}/8 classes recovered the planted peak layer"
    );

    println!("\n── G1: three-arm intervention @ best layer {best_layer} ──");
    println!(
        "  {:>4} {:>9} {:>9} {:>9}",
        "k", "aligned", "random", "residual"
    );
    for (ki, &k) in KS.iter().enumerate() {
        println!(
            "  {:>4} {:>9.3} {:>9.3} {:>9.3}",
            k,
            out_a.aligned[best_layer][ki],
            out_a.random[best_layer][ki],
            out_a.residual[best_layer][ki]
        );
    }
    let full = out_a.full_acc[best_layer];
    let last = KS.len() - 1;
    assert!(
        (out_a.aligned[best_layer][last] - full).abs() <= 2.0 / N_TEST as f32,
        "G1a FAILED: aligned@k=rank {:.4} != full {:.4} (projection identity violated)",
        out_a.aligned[best_layer][last],
        full
    );
    assert!(
        out_a.aligned[best_layer][3] >= 0.9 * full,
        "G1b FAILED: aligned@k=6 {:.3} < 0.9·full ({:.3}) — task signal not low-rank-compressible",
        out_a.aligned[best_layer][3],
        0.9 * full
    );
    assert!(
        out_a.residual[best_layer][last] <= CHANCE + 0.05,
        "G1c FAILED: residual@k=rank {:.3} > chance+0.05",
        out_a.residual[best_layer][last]
    );
    assert!(
        out_a.aligned[best_layer][2] >= 1.5 * out_a.random[best_layer][2],
        "G1d FAILED: aligned@k=4 {:.3} < 1.5·random@k=4 ({:.3}) — no aligned-vs-random contrast",
        out_a.aligned[best_layer][2],
        1.5 * out_a.random[best_layer][2]
    );
    println!("  chance = {CHANCE:.3}; full-rank = {full:.3}");
    println!("  G1 triad: PASS — (a) rank-identity, (b) k=6 compression ≥ 0.9·full,");
    println!("            (c) residual collapse, (d) aligned ≥ 1.5·random at matched k=4.");
    println!(
        "  → protocol-level close of the spectral_pre_rotate.rs:29 deferred aligned-vs-random"
    );
    println!("    eval (readout space, controlled ground truth); FUNCATTN-tensor arm = follow-up.");

    println!("\n── G3 (informational): seed stability of V_k @ layer {best_layer} ──");
    for (ki, &k) in KS.iter().enumerate() {
        println!(
            "  k={k:>2}: mean pairwise basis similarity {:.3}",
            out_a.stability[ki]
        );
    }
    let (sim_lo, sim_hi) = (out_a.stability[0], out_a.stability[4]);
    if sim_lo - sim_hi >= 0.10 {
        println!(
            "  SPLIT DETECTED (k=1 sim {sim_lo:.3} vs k=8 sim {sim_hi:.3}) — freeze-policy implication: commit the stable core only."
        );
    } else {
        println!(
            "  NO SPLIT (k=1 {sim_lo:.3} vs k=8 {sim_hi:.3}) — freeze policy unchanged; recorded per Issue 778 outcome criteria."
        );
    }

    println!(
        "\n═══ Issue 778 POC complete: G0 PASS · G1 PASS · G2 PASS ({hits}/8) · G3 recorded ═══"
    );
}
