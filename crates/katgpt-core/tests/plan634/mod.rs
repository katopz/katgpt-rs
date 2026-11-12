//! Plan 634 Phase-2 shared fixtures + instruments (T2.1–T2.5).
#![allow(dead_code)] // per-target inclusion: each battery uses a subset
//! NOT a test target (subdirectory — cargo auto-discovers only top-level
//! `tests/*.rs`); every battery includes this via `#[path = "plan634/mod.rs"]`.
//!
//! Contents:
//! - [`planted_clique_graph`] — the paper Def 5.5 fixture (K cliques of M,
//!   K-cycle meta-graph ⇒ r = 2, c = 1 pinned+asserted; intra weight 1,
//!   complete-bipartite joins of weight 1/M ⇒ uniform degree d = M−1+r).
//! - [`circulant_expander`] — the well-mixed fixture (no clusters).
//! - [`eig_sym`] — symmetric eigendecomposition through the in-crate
//!   `karc::jacobi_eigen` (dynamic f64), eigenvalues sorted DESCENDING.
//! - [`perron_deflated_radius`] — max |λ| over non-Perron eigenvalues of a
//!   small row-stochastic (generally non-symmetric) chain: deflate the
//!   Perron block with power-iterated left/right pairs, then read the
//!   spectral radius by norm growth (`‖Bᵏv‖^(1/k)` — converges for real
//!   AND complex-dominant spectra, which a plain power iteration does not).
//! - [`xorshift64`] — the pinned seeded RNG for the sampling arms.
//! - [`spearman_rho`] — rank correlation (first-index tie order, the
//!   deterministic house tie rule).
//! - [`raw_truncated_phi`] / [`raw_phi_total_mass`] — the runtime-k
//!   truncated-kernel helpers for the REPORTED truncation rows (they read
//!   the embedding's public fields directly — an independent authority,
//!   never the code under test).

use katgpt_core::diversity::diffusion::{DiffusionEmbedding, SimGraph};
use katgpt_core::karc::jacobi_eigen;

/// The planted clique-graph (paper Def 5.5, plan T2.1): `K` cliques of size
/// `M`; the meta-graph is the `K`-cycle, so every clique has meta-degree
/// `r = 2` and any two distinct meta-neighbors of one clique share EXACTLY
/// `c = 1` common meta-neighbor (the clique itself) — pinned by
/// construction and asserted here. Intra-clique edges carry weight 1; a
/// meta-edge joins its two cliques as a complete bipartite join of weight
/// `1/M`, which makes every node's degree exactly `d = (M−1) + r` (the
/// `d` the paper's `B_M = c + (d−c)/r` consumes) and keeps the
/// meta-subspace walk-invariant — the regime where the plan's closed-form
/// SRW gap `γ_srw = (r/(M−1+r))·(1 − λ₂(P_Q))` is EXACT.
///
/// Returns the graph, the cluster index per node (node `i·M + j` ∈ clique
/// `i`), `r`, `d`, and the meta-graph adjacency (`K×K`, 0/1).
pub fn planted_clique_graph(
    k: usize,
    m: usize,
) -> (SimGraph, Vec<usize>, usize, usize, Vec<Vec<bool>>) {
    assert!(m >= 3, "M−1 >= r=2 needs M >= 3");
    assert!(k >= 5, "the K-cycle pins c = 1 only for K >= 5");
    assert!(m > 2, "Def 5.5 condition M−1 >= r");
    let n = k * m;
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    // Intra-clique: K_M, unit weight.
    for c in 0..k {
        for i in 0..m {
            for j in (i + 1)..m {
                edges.push((c * m + i, c * m + j, 1.0));
            }
        }
    }
    // Meta-edges (K-cycle): complete bipartite join, weight 1/M.
    // Indexed loop by need: the body mutates BOTH meta[c] and meta[nxt] —
    // an iter_mut() borrow of meta conflicts with the cross-element write
    // (E0499), so clippy::needless_range_loop is a false positive here.
    let mut meta = vec![vec![false; k]; k];
    #[allow(clippy::needless_range_loop)]
    for c in 0..k {
        let nxt = (c + 1) % k;
        meta[c][nxt] = true;
        meta[nxt][c] = true;
        let w = 1.0 / m as f32;
        for i in 0..m {
            for j in 0..m {
                edges.push((c * m + i, nxt * m + j, w));
            }
        }
    }
    // The c = 1 pin, computed not assumed: any two distinct meta-neighbors
    // of one clique share exactly one common meta-neighbor.
    for c in 0..k {
        let nbrs: Vec<usize> = (0..k).filter(|&j| meta[c][j]).collect();
        assert_eq!(nbrs.len(), 2, "meta-degree r must be 2 (K-cycle)");
        let common: Vec<usize> = (0..k)
            .filter(|&j| meta[nbrs[0]][j] && meta[nbrs[1]][j])
            .collect();
        assert_eq!(
            common,
            vec![c],
            "Def 5.5 parameter c must be exactly 1 (got {common:?} at clique {c})"
        );
    }
    let cluster_of: Vec<usize> = (0..n).map(|x| x / m).collect();
    let d = (m - 1) + 2;
    (SimGraph::build(n, &edges), cluster_of, 2, d, meta)
}

/// The well-mixed fixture (plan T2.4/F4, F5 tie arm): the circulant graph
/// on `n` nodes with symmetric offsets — degree `2·offsets.len()`, no
/// planted clusters, good expansion.
pub fn circulant_expander(n: usize, offsets: &[usize]) -> SimGraph {
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for &o in offsets {
            let j = (i + o) % n;
            if !edges
                .iter()
                .any(|&(a, b, _)| (a, b) == (i.min(j), i.max(j)))
            {
                edges.push((i, j, 1.0));
            }
        }
    }
    SimGraph::build(n, &edges)
}

/// Symmetric eigendecomposition through the in-crate authority
/// (`karc::jacobi_eigen`, dynamic f64) — eigenvalues sorted DESCENDING,
/// eigenvectors re-sorted to match (`out_vecs[k*n + x]` row-major).
pub fn eig_sym(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut vals = vec![0.0_f64; n];
    let mut vecs = vec![0.0_f64; n * n];
    let mut scratch = vec![0.0_f64; n * n];
    jacobi_eigen(&mut vals, &mut vecs, a, &mut scratch, n, 1e-12, 200);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        vals[j]
            .partial_cmp(&vals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sv: Vec<f64> = order.iter().map(|&k| vals[k]).collect();
    let sw: Vec<f64> = (0..n * n)
        .map(|cell| {
            let (x, k) = (cell / n, cell % n);
            let src = order[k];
            vecs[x * n + src]
        })
        .collect();
    (sv, sw)
}

/// Max |λ| over the NON-Perron eigenvalues of a row-stochastic chain
/// (generally non-symmetric): `1 − radius` is the modulus spectral gap.
///
/// Step 1 — power-iterate `Pᵀ` for the left Perron vector `ℓ` (stationary
/// of the time-reversed chain; converges by Perron–Frobenius), and `P` for
/// the right Perron vector `π`. Step 2 — deflate
/// `B = P − π·ℓᵀ/(ℓᵀ·π)` (the Perron block is exactly zeroed; every other
/// eigenvalue survives). Step 3 — read the radius by norm growth:
/// `v ← Bv/‖Bv‖`, `ρ_k = ‖Bv_k‖ → ρ`. The norm-growth read converges to
/// the spectral radius even when the dominant deflated eigenvalue is
/// COMPLEX (a plain vector power iteration would oscillate there) — the
/// modulus is the gap statistic this fixture needs.
pub fn perron_deflated_radius(p: &[f64], n: usize, iters: usize) -> f64 {
    // Right Perron vector π (power iteration on P, renormalized each step).
    let mut pi = vec![1.0 / n as f64; n];
    // Left Perron vector ℓ (power iteration on Pᵀ).
    let mut ell = vec![1.0 / n as f64; n];
    for _ in 0..iters {
        // π ← P·π
        let mut nxt = vec![0.0_f64; n];
        for (i, row) in p.chunks_exact(n).enumerate() {
            let acc: f64 = row.iter().zip(pi.iter()).map(|(a, b)| a * b).sum();
            nxt[i] = acc;
        }
        let s: f64 = nxt.iter().sum();
        for v in &mut nxt {
            *v /= s;
        }
        pi = nxt;
        // ℓ ← Pᵀ·ℓ
        let mut nxt = vec![0.0_f64; n];
        for j in 0..n {
            let acc: f64 = (0..n).map(|i| p[i * n + j] * ell[i]).sum();
            nxt[j] = acc;
        }
        let s: f64 = nxt.iter().sum();
        for v in &mut nxt {
            *v /= s;
        }
        ell = nxt;
    }
    let lp: f64 = ell.iter().zip(pi.iter()).map(|(a, b)| a * b).sum();
    // Deflate: B = P − π·ℓᵀ / (ℓᵀπ).
    let mut b = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = p[i * n + j] - pi[i] * ell[j] / lp;
        }
    }
    // Radius by norm growth.
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut rho = 0.0_f64;
    for _ in 0..iters {
        let mut bv = vec![0.0_f64; n];
        for (i, row) in b.chunks_exact(n).enumerate() {
            bv[i] = row.iter().zip(v.iter()).map(|(a, x)| a * x).sum();
        }
        let norm: f64 = bv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-300 {
            return 0.0; // B is nilpotent — every non-Perron eigenvalue is 0
        }
        rho = norm;
        for (slot, &x) in v.iter_mut().zip(bv.iter()) {
            *slot = x / norm;
        }
    }
    rho
}

/// The pinned seeded RNG (xorshift64*) — every sampling arm carries an
/// explicit constant, same seed ⇒ byte-identical draws.
pub struct Xorshift64(u64);

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform index in [0, n).
    pub fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Standard normal via Box–Muller (deterministic given the stream).
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Spearman rank correlation (first-index tie order — the deterministic
/// house tie rule; the report arm's inputs are near-tie-free by
/// construction, so the tie policy never carries the verdict).
pub fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let rank = |v: &[f64]| -> Vec<usize> {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0_usize; v.len()];
        for (r, &idx) in order.iter().enumerate() {
            ranks[idx] = r;
        }
        ranks
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f64;
    let ma: f64 = (n - 1.0) / 2.0; // ranks are a permutation of 0..n−1
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = ra[i] as f64 - ma;
        let db = rb[i] as f64 - ma;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return f64::NAN;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Raw truncated kernel cell `Σ_{k<d} λ_k^{2τ}·ψ_k(x)·ψ_k(y)/(√π(x)√π(y))`
/// read DIRECTLY from the embedding's public fields (f64) — the runtime-k
/// authority for the REPORTED truncation rows (`d` may exceed `n`; the
/// extra coordinates are zero).
pub fn raw_truncated_phi(
    emb: &DiffusionEmbedding,
    x: usize,
    y: usize,
    tau: usize,
    d: usize,
) -> f64 {
    let stride = emb.eigvecs.len() / emb.eigvals.len();
    let k_max = emb.eigvals.len().min(d);
    let mut acc = 0.0_f64;
    for k in 0..k_max {
        acc += emb.eigvals[k].powi((2 * tau) as i32)
            * emb.eigvecs[k * stride + x]
            * emb.eigvecs[k * stride + y]
            * emb.inv_sqrt_deg[x]
            * emb.inv_sqrt_deg[y];
    }
    acc
}

/// The kernel's TOTAL mass `Σ_{all k} λ_k^{2τ}·ψ_k(x)·ψ_k(y)/(√π√π)` —
/// the reference for the cell-level ABSOLUTE truncation error companion.
pub fn raw_phi_total_mass(emb: &DiffusionEmbedding, x: usize, y: usize, tau: usize) -> f64 {
    raw_truncated_phi(emb, x, y, tau, emb.eigvals.len())
}

/// The plan's literal truncation statistic (F2/F5 reported rows):
/// `(tail, total)` where `tail = Σ_{k≥d, by |λ|} λ_k^{2τ}` and
/// `total = Σ_{all k} λ_k^{2τ}` — pure eigenvalue quantities, always
/// meaningful (a per-pair CELL-relative error divides by near-zero cells
/// and explodes; the cell-level companion below stays ABSOLUTE).
pub fn eigenvalue_tail_mass(emb: &DiffusionEmbedding, tau: usize, d: usize) -> (f64, f64) {
    let mut tail = 0.0_f64;
    let mut total = 0.0_f64;
    for (k, &lam) in emb.eigvals.iter().enumerate() {
        let contrib = lam.powi((2 * tau) as i32);
        total += contrib;
        if k >= d {
            tail += contrib;
        }
    }
    (tail, total)
}

/// The cell-level ABSOLUTE truncation error `|φ_full − φ_trunc|` — the
/// companion the reported rows print beside the eigenvalue tail mass
/// (absolute, never a ratio: the kernel's near-zero cells make relative
/// errors explode meaningless).
pub fn truncation_abs_cell_error(
    emb: &DiffusionEmbedding,
    x: usize,
    y: usize,
    tau: usize,
    d: usize,
) -> f64 {
    let full = raw_phi_total_mass(emb, x, y, tau);
    let truncated = raw_truncated_phi(emb, x, y, tau, d);
    (full - truncated).abs()
}
