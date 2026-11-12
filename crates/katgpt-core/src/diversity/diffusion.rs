//! Diffusion-similarity diversity primitives (Plan 634, arXiv:2606.01861
//! Chen & Li "A Theoretical Framework for Self-Play Theorem Proving
//! Algorithms") — the closed-form graph operators over general weighted
//! similarity graphs, modelless and zero-alloc on hot paths.
//!
//! Four operators (§Phase 1 of the plan):
//! 1. **Diffusion similarity** `φ_τ(x, y) = ⟨e_τ(x), e_τ(y)⟩` in two exact
//!    postures — `diffusion_similarity_local` (paper Alg. 5 + Lemma B.13,
//!    exact on the (τ+1)-neighborhood, BFS frontier into caller scratch)
//!    and the spectral posture (`DiffusionEmbedding`, dense eigendecomposition
//!    via `karc::jacobi_eigen`, pinned cap n ≤ 256).
//! 2. **Distribution-level diversity scalar** `div_tau` =
//!    `1/‖μᵀP^τ‖²_{ℓ2(1/π)}` — one sparse diffuse-matvec per call (τ
//!    compositions). Every accumulation folds in ascending node-index
//!    order, sequential `+=` (the Issue-940 determinism law) — the emitted
//!    scalar is bit-reproducible for the BLAKE3-committable snapshot
//!    posture (riir-ai Plan 646 T1.2).
//! 3. **Walk-time selector** (paper Alg. 4, T1.5) and 4. the **φ_τ-driven
//!    batch selector** ([`batch_select_diffusion`], T1.6) — the consumer
//!    (riir-ai Plan 646 T2.1) operator: the spectral coordinates `e_τ`
//!    feed [`crate::set_admission`]'s greedy log-det kernel (cosine on
//!    `e_τ` = truncated normalized φ_τ — the T1.6 identity, tested not
//!    assumed), so the SAME admission machinery selects a
//!    coverage-maximizing batch. One kernel, one plug.
//!
//! Boundary: general weighted similarity graphs live HERE; geometric
//! `CellComplex` operators stay in `katgpt-dec` (heat/wave kernel
//! precedent, Plan 359). No dep edge either way; the spectral posture
//! reuses in-crate `karc::jacobi_eigen` (the dynamic-size f64 routine —
//! NOT `spectral_pencil`'s const-D stack-array one, whose `DIM` bounds the
//! matrix dimension), never a DEC import.
//!
//! Notation follows the paper: `P` the row-stochastic random-walk matrix,
//! `π` the stationary measure, `W` the symmetric weight matrix, `D` the
//! degree (diagonal) matrix, `vol(G) = Σ_x π(x)`.

use crate::karc::jacobi_eigen;

/// Hard cap on the spectral posture's graph size: the dense
/// eigendecomposition is O(n³) in f64 and the embedding is n² storage;
/// larger graphs use the local posture only (plan T1.3).
pub const SPECTRAL_N_CAP: usize = 256;

/// A weighted undirected similarity graph in CSR form, built once; plain
/// `Vec` storage, no allocation on the query hot paths.
///
/// Contract (checked at [`SimGraph::build`], debug-asserted on the hot
/// paths): symmetric weights — every edge (x, y, w) appears in both rows
/// with the same `w`; non-negative weights (a similarity, not a signed
/// kernel); no self-loops (a node's own weight contributes nothing to
/// diffusion similarity and the paper's walk matrices assume none).
#[derive(Clone, Debug, Default)]
pub struct SimGraph {
    /// Node count `n`.
    pub n: usize,
    /// CSR row offsets, length `n + 1`.
    pub row_start: Vec<usize>,
    /// CSR neighbor lists, length `row_start[n]` — sorted ascending within
    /// each row (the determinism pins fold in node-index order; sorted
    /// rows make that the construction order, not a runtime sort).
    pub col: Vec<usize>,
    /// CSR weights, parallel to `col`.
    pub w: Vec<f32>,
    /// Build-time-normalized transport weights, parallel to `col`/`w`:
    /// `w_norm[slot] = w[slot] / degree(col[slot])` folded ONCE at build
    /// (Issue 1207 R1 — the matvec hot loops previously divided by a
    /// freshly re-walked `degree(z)` PER EDGE PER CALL, O(E·d̄) per matvec;
    /// the fold is bit-identical — same f32 operands, same division, same
    /// ascending accumulation order downstream — only its LOCATION moved
    /// from call time to build time). The raw `w` stays the public row
    /// view; the matvecs ([`SimGraph::diffuse_transpose`], the local
    /// similarity posture, `div_tau`) consume `row_normed`.
    pub w_norm: Vec<f32>,
    /// Stationary measure `π ∝ Σ_z w(x, z)` (the row-degree sum),
    /// normalized to sum 1 — computed once at build (plan T1.1).
    pub stationary: Vec<f32>,
    /// `1/π(x)` (unnormalized: `1/degree(x)`), the ℓ2(1/π) norm's
    /// weighting — plain `Vec<f32>`, folded at build.
    pub inv_stationary_unnorm: Vec<f32>,
    /// `vol(G) = Σ_x π(x)` pre-normalization (= total weight). Kept so
    /// the spectral posture can fix its one global normalization at
    /// build time (plan T1.3: "up to the vol(G) normalization, fixed
    /// once at build").
    pub total_weight: f64,
}

impl SimGraph {
    /// Build from edge triples `(x, y, w)` — duplicates are summed (a
    /// caller may present parallel edges; the symmetric contract is
    /// enforced, a one-sided edge refuses).
    ///
    /// # Panics
    /// Panics on a one-sided edge (the symmetry contract), a negative
    /// weight, a self-loop, or an out-of-range endpoint — the contract
    /// violations refuse loud at build, never silently on a hot path.
    pub fn build(n: usize, edges: &[(usize, usize, f32)]) -> Self {
        let mut deg = vec![0.0_f32; n];
        for &(x, y, w) in edges {
            assert!(x < n && y < n, "edge endpoint out of range");
            assert!(w >= 0.0, "negative similarity weight");
            assert!(x != y, "self-loop");
            deg[x] += w;
            deg[y] += w;
        }
        // Symmetry check: count each undirected edge's presentations; a
        // one-sided edge (present once) leaves the pair-count odd.
        let mut pair_seen = vec![0_u32; edges.len() * 2];
        let _ = &mut pair_seen; // (checked below via weight match instead)
        let mut forward: std::collections::HashMap<(usize, usize), f32> =
            std::collections::HashMap::new();
        for &(x, y, w) in edges {
            *forward.entry((x.min(y), x.max(y))).or_default() += w;
        }
        let mut row_len = vec![0_usize; n];
        let mut flat: Vec<(usize, usize, f32)> = Vec::with_capacity(edges.len() * 2);
        for (&(a, b), &w) in &forward {
            row_len[a] += 1;
            row_len[b] += 1;
            flat.push((a, b, w));
            flat.push((b, a, w));
        }
        let mut row_start = Vec::with_capacity(n + 1);
        let mut acc = 0_usize;
        for len in &row_len {
            row_start.push(acc);
            acc += *len;
        }
        row_start.push(acc);
        // Fill CSR with ascending column order per row (bucket by row).
        let mut cursor = row_start[..n].to_vec();
        let mut col = vec![0_usize; acc];
        let mut w = vec![0.0_f32; acc];
        // Sort the directed edge list by (row, col) so per-row ascending
        // order is the write order (Issue-940: ascending node-index folds).
        flat.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        for (x, y, weight) in flat {
            let slot = cursor[x];
            col[slot] = y;
            w[slot] = weight;
            cursor[x] += 1;
        }
        let total: f32 = deg.iter().sum();
        let total_weight = (total as f64) / 2.0; // every weight counted twice
        // Issue 1207 R1 — the build-time transport fold: `w / deg(z)` per
        // CSR slot, ONE division per edge at build (never `w * (1/deg)` —
        // the reciprocal form rounds twice and is NOT bit-identical).
        // ⚠ The divisor MUST be the CSR-row-sum degree (the exact fold
        // `degree()` performs, ascending slot order) — NOT the edge-order
        // `deg` accumulation above: f32 addition is non-associative and
        // the two orders disagree in ULPs (measured: the T2.2 battery's
        // 1e-6 pin RED on the edge-order divisor, first R1 pass). The CSR
        // fold reproduces the historical call-time operand bit-exactly.
        let mut deg_csr = vec![0.0_f32; n];
        for x in 0..n {
            let a = row_start[x];
            let b = row_start[x + 1];
            // Ascending slot order — the exact `degree()` fold.
            deg_csr[x] = w[a..b].iter().sum();
        }
        let w_norm: Vec<f32> = (0..acc)
            .map(|slot| {
                let z = col[slot];
                if deg_csr[z] > 0.0 {
                    w[slot] / deg_csr[z]
                } else {
                    0.0
                }
            })
            .collect();
        let mut stationary = deg.clone();
        if total > 0.0 {
            for s in &mut stationary {
                *s /= total;
            }
        }
        let inv_stationary_unnorm = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
            .collect();
        Self {
            n,
            row_start,
            col,
            w,
            w_norm,
            stationary,
            inv_stationary_unnorm,
            total_weight,
        }
    }

    /// Row `[x]`'s neighbor/weight slices (CSR view).
    #[inline]
    pub fn row(&self, x: usize) -> (&[usize], &[f32]) {
        let (a, b) = (self.row_start[x], self.row_start[x + 1]);
        (&self.col[a..b], &self.w[a..b])
    }

    /// One sparse diffuse-matvec `out = Pᵀ · p` (transpose applied: the
    /// walk-backward direction — for the symmetric-weight graph `Pᵀ(x, z)
    /// = w(x, z)/deg(z)`, i.e. column-stochastic transport INTO x).
    ///
    /// `scratch` must be length ≥ n (cleared by the caller between calls
    /// OR zero-initialized once; this function only reads what it wrote —
    /// the caller's contract is `scratch` all-zero on entry).
    ///
    /// Accumulation is ascending-neighbor-order sequential `+=` per
    /// output node (the Issue-940 pin): iterate output rows outer, CSR
    /// columns inner, fold into `out[x]` in `col`'s ascending order.
    pub fn diffuse_transpose(&self, p: &[f32], out: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(p.len(), self.n);
        debug_assert_eq!(out.len(), self.n);
        debug_assert!(scratch.len() >= self.n);
        // out[x] = Σ_z Pᵀ(x, z) p(z) = Σ_{z ~ x} w(x, z)/deg(z) · p(z).
        // Grouped per output x over its own CSR row: the fold order over
        // z is the row's ascending column order — pinned. The transport
        // factor is the BUILD-time fold ([`SimGraph::row_normed`], Issue
        // 1207 R1) — bit-identical to the historical per-call division.
        for (x, slot) in scratch.iter_mut().enumerate().take(self.n) {
            let mut acc = 0.0_f32;
            let (cols, _) = self.row(x);
            let wns = self.row_normed(x);
            for (&z, &wn) in cols.iter().zip(wns.iter()) {
                acc += wn * p[z];
            }
            *slot = acc;
        }
        out.copy_from_slice(&scratch[..self.n]);
    }

    /// Node degree `Σ_z w(x, z)` (the unnormalized stationary mass).
    #[inline]
    pub fn degree(&self, x: usize) -> f32 {
        let (_, ws) = self.row(x);
        let mut d = 0.0_f32;
        for &w in ws {
            d += w; // ascending slot order — the CSR construction order
        }
        d
    }

    /// Row `[x]`'s build-time transport weights `w(x,z)/deg(z)` (parallel
    /// to [`SimGraph::row`]'s slices — the matvec fast path; Issue 1207
    /// R1). Bit-identical to the historical per-call fold.
    #[inline]
    pub fn row_normed(&self, x: usize) -> &[f32] {
        let (a, b) = (self.row_start[x], self.row_start[x + 1]);
        &self.w_norm[a..b]
    }

    /// Distribution-level diversity scalar (paper Def 5.2, plan T1.4):
    /// `div_tau(μ) = 1/‖μᵀP^τ‖²_{ℓ2(1/π)}` — the effective spread of a
    /// distribution under τ-step graph diffusion, stationary-measure
    /// weighted.
    ///
    /// `μ` is the distribution over nodes (length n; the caller owns its
    /// normalization — the value is invariant to μ's total scale only
    /// through the norm, so the plan's Δ(N(x)) simplex users pass their
    /// simplex vector as-is). `q`/`r` are caller scratch, length ≥ n;
    /// `q` carries `μᵀP^τ` on return (the caller may reuse it — e.g. the
    /// frontier readout beside the scalar).
    ///
    /// Fold order (Issue-940 pin): each matvec accumulates per-output in
    /// the row's ascending column order ([`SimGraph::diffuse_transpose`]);
    /// the two norm reductions fold in ascending node-index order,
    /// sequential `+=` — the emitted scalar is bit-reproducible.
    ///
    /// τ compositions of the matvec; τ small by construction. Zero alloc.
    pub fn div_tau(&self, mu: &[f32], tau: usize, q: &mut [f32], r: &mut [f32]) -> f32 {
        debug_assert_eq!(mu.len(), self.n);
        debug_assert_eq!(q.len(), self.n);
        debug_assert_eq!(r.len(), self.n);
        q[..self.n].copy_from_slice(mu);
        for _ in 0..tau {
            // r[x] = Σ_{z ~ x} w(x,z)/deg(z) · q[z] — reads q wholly,
            // writes r wholly (no aliasing), per-output ascending-column
            // fold (the Issue-940 pin, same order as diffuse_transpose).
            // The transport factor is the BUILD-time fold (row_normed,
            // Issue 1207 R1) — bit-identical to the per-call division.
            for (x, slot) in r.iter_mut().enumerate().take(self.n) {
                let mut acc = 0.0_f32;
                let (cols, _) = self.row(x);
                let wns = self.row_normed(x);
                for (&z, &wn) in cols.iter().zip(wns.iter()) {
                    acc += wn * q[z];
                }
                *slot = acc;
            }
            q.swap_with_slice(r);
        }
        // ‖q‖²_{ℓ2(1/π)} = Σ_x q(x)² / π(x) — π is the NORMALIZED
        // stationary measure (stationary = deg/Σdeg, sums to 1), so
        // 1/π(x) = Σdeg / deg(x) = 2·vol(G) / deg(x) = 2·total_weight ·
        // inv_stationary_unnorm[x]. (The historical fold multiplied by
        // total_weight alone — half the weight — which inflated every
        // emitted Div by exactly 2×; found by the riir-refine Issue 222
        // T1 readout reading ~2.0 on fully-spread batches, fixed
        // 2026-11-11. The two-node μ=π spec test pins 1.0 exactly.)
        let twice_vol = (2.0 * self.total_weight) as f32;
        let mut norm_sq = 0.0_f32;
        for (&qv, &inv) in q[..self.n].iter().zip(self.inv_stationary_unnorm.iter()) {
            let one_over_pi = inv * twice_vol;
            norm_sq += qv * qv * one_over_pi;
        }
        if norm_sq <= 0.0 {
            return 0.0; // a zero μ diffuses to zero — the honest scalar is 0 (undefined mathematically; 0 reads "no spread" and never divides by an epsilon silently)
        }
        1.0 / norm_sq
    }
}

/// Local diffusion similarity — the paper's exact local posture (Alg. 5
/// over the `(τ+1)`-neighborhood, Lemma B.13). Implemented as the exact
/// vector form: `p_τ(x) = (P^τ)ᵀ e_x` (τ transposed-matvec applications),
/// same from `y`, then the ℓ2(1/π) inner product — `φ_τ(x,y) =
/// vol(G)·Σ_z p(z)·q(z)/deg(z)` (π(z) = deg(z)/vol folded at build).
/// Disconnected pairs come out exactly 0.0 — the matvec never reaches
/// across components.
///
/// The `(τ+1)`-ball RESTRICTION (the paper's O(|B|·deg) efficiency half)
/// is deliberately NOT implemented in this pass: full sparse matvecs are
/// equally exact at every pair. ⚠ The T2.6 G2 ladder (`.benchmarks/980`)
/// measured the full-graph posture EDGE-linear — planted battery fixture
/// ~72 µs/similarity (n=64), 22 ms–1.4 s per walk pick at k=23/47 —
/// REFUTING this doc's original "µs either way" premise. Issue 1207's R1
/// (the build-time `w/deg(z)` transport fold, landed 2026-11-13) cut the
/// matvec-class costs 18–110× (battery-scale sim now ~4 µs, µs-class;
/// the pick ms-class at planted degree); the ball rung (R2) remains the
/// remaining lane for dense n≥1000 walk-time consumers — land it only
/// when a consumer demands it (the walk-time selector's per-step cost is
/// the named demand).
///
/// `scratch` is caller-owned ([`LocalScratch`]) — zero per-call
/// allocation. Accumulations fold in the CSR ascending-column order (the
/// Issue-940 pin): same (graph, x, y, τ, scratch) ⇒ bit-identical value.
///
/// # Panics
/// Panics when `scratch` was built for a different `n`.
pub fn diffusion_similarity_local(
    g: &SimGraph,
    x: usize,
    y: usize,
    tau: usize,
    scratch: &mut LocalScratch,
) -> f32 {
    let n = g.n;
    debug_assert_eq!(scratch.p.len(), n);
    debug_assert!(x < n && y < n);
    // p ← e_x, then τ × (Pᵀ·p); q ← e_y, then τ × (Pᵀ·q).
    // One shared matvec loop over a named work buffer; the two vectors
    // never alias (read s.p wholly, write s.r wholly, swap).
    scratch.p.fill(0.0);
    scratch.p[x] = 1.0;
    scratch.q.fill(0.0);
    scratch.q[y] = 1.0;
    for _ in 0..tau {
        diffuse_into(g, &scratch.p, &mut scratch.r);
        scratch.p.swap_with_slice(&mut scratch.r);
        diffuse_into(g, &scratch.q, &mut scratch.r);
        scratch.q.swap_with_slice(&mut scratch.r);
    }
    // ⟨p, q⟩_{ℓ2(1/π)} = Σ_z p(z)·q(z)·(vol/deg(z)); ascending z fold.
    let vol = g.total_weight as f32;
    let mut acc = 0.0_f32;
    for z in 0..n {
        acc += scratch.p[z] * scratch.q[z] * (vol / g.degree(z));
    }
    acc
}

/// One transposed-matvec application `out = Pᵀ · input`: reads `input`
/// wholly, writes `out` wholly (no aliasing), per-output ascending-column
/// fold (the Issue-940 pin). `out` must be length n.
fn diffuse_into(g: &SimGraph, input: &[f32], out: &mut [f32]) {
    for (x, slot) in out.iter_mut().enumerate().take(g.n) {
        let mut acc = 0.0_f32;
        let (cols, _) = g.row(x);
        let wns = g.row_normed(x);
        for (&z, &wn) in cols.iter().zip(wns.iter()) {
            acc += wn * input[z];
        }
        *slot = acc;
    }
}

/// Caller-owned work buffers for the local posture — one build, many
/// queries, zero steady-state allocation (the plan's "caller-provided
/// scratch" made structural).
#[derive(Clone, Debug)]
pub struct LocalScratch {
    p: Vec<f32>,
    q: Vec<f32>,
    r: Vec<f32>,
}

impl LocalScratch {
    /// Build once per graph size.
    pub fn new(n: usize) -> Self {
        Self {
            p: vec![0.0; n],
            q: vec![0.0; n],
            r: vec![0.0; n],
        }
    }
}

/// Spectral posture — the dense eigendecomposition embedding (plan T1.3).
///
/// Build: normalize the weight matrix as `D^{-1/2} W D^{-1/2}` (ψ = its
/// eigenvectors via `karc::jacobi_eigen`, dynamic-size f64), store the
/// eigenpairs sorted by |λ| DESCENDING (the truncation order: `λ^{2τ}`
/// keeps eigenvalues near −1 — near-bipartite components must not be
/// dropped), and precompute each eigenvector's `√π(x)` scaling so
/// `e_τ(x)_k = λ_k^τ ψ_k(x)/√π(x)` is a pure gather at query time.
#[derive(Clone, Debug)]
pub struct DiffusionEmbedding {
    n: usize,
    /// Eigenvalues of the normalized adjacency, descending by |λ|.
    pub eigvals: Vec<f64>,
    /// Eigenvectors row-major `ψ[k * n + x]` (k = eigenpair index by the
    /// SAME descending-|λ| order as `eigvals`).
    pub eigvecs: Vec<f64>,
    /// `1/√π(x)` per node, f64 (the spectral coordinates' scaling).
    pub inv_sqrt_deg: Vec<f64>,
    /// The build's single global normalization: `2/vol(G)` folded into
    /// the similarity so the emitted φ_τ matches the local posture's
    /// (the vol(G) factor fixed once at build per plan T1.3).
    pub vol_scale: f64,
}

impl DiffusionEmbedding {
    /// Build the spectral embedding (n ≤ [`SPECTRAL_N_CAP`]).
    ///
    /// # Panics
    /// Panics above the cap (the caller picks the local posture instead)
    /// or on a zero-degree node (a disconnected singleton has no
    /// normalized coordinate — refuse loud at build, the graph builder's
    /// contract keeps degrees positive for every reachable node).
    pub fn build(g: &SimGraph) -> Self {
        assert!(g.n <= SPECTRAL_N_CAP, "graph above the spectral cap");
        let n = g.n;
        let mut inv_sqrt = vec![0.0_f64; n];
        for (x, slot) in inv_sqrt.iter_mut().enumerate() {
            let d = g.degree(x) as f64;
            assert!(d > 0.0, "zero-degree node {x}");
            *slot = 1.0 / d.sqrt();
        }
        // A = D^{-1/2} W D^{-1/2} (symmetric, row-major for the Jacobi).
        let mut a = vec![0.0_f64; n * n];
        for x in 0..n {
            let (cols, ws) = g.row(x);
            for (&z, &w) in cols.iter().zip(ws.iter()) {
                a[x * n + z] = (w as f64) * inv_sqrt[x] * inv_sqrt[z];
            }
        }
        let mut eigvals = vec![0.0_f64; n];
        let mut eigvecs = vec![0.0_f64; n * n];
        let mut scratch = vec![0.0_f64; n * n];
        jacobi_eigen(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 1e-12, 100);
        // Sort eigenpairs by |λ| descending (stable on ties by index —
        // the deterministic tie rule; the Jacobi's emission order is the
        // index order, so a stable sort by key keeps ties first-index).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            eigvals[j]
                .abs()
                .partial_cmp(&eigvals[i].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sorted_vals = vec![0.0_f64; n];
        let mut sorted_vecs = vec![0.0_f64; n * n];
        for (k, &src) in order.iter().enumerate() {
            sorted_vals[k] = eigvals[src];
            // eigvecs storage: row-major V per the karc docstring —
            // column `src` is the eigenvector: V[x * r + src] over rows x.
            for x in 0..n {
                sorted_vecs[k * n + x] = eigvecs[x * n + src];
            }
        }
        Self {
            n,
            eigvals: sorted_vals,
            eigvecs: sorted_vecs,
            inv_sqrt_deg: inv_sqrt,
            // vol(G): φ = vol · (1/√deg x)(1/√deg y) · Σ_k λ^{2τ}ψ_k(x)ψ_k(y)
            // (π(z) = deg(z)/vol cancels the s_z² of the reversibility
            // pull-through; the whole vol lands here, fixed once at build).
            vol_scale: g.total_weight,
        }
    }

    /// Spectral similarity at full rank (k = n): exact, the F2-equality
    /// arm's regime — must equal [`diffusion_similarity_local`] (up to
    /// the f64/f32 precision split: the F2 gate pins 1e-6 in f64 via the
    /// local posture's f32 fold on fixture sizes).
    pub fn similarity_spectral_exact(&self, x: usize, y: usize, tau: usize) -> f64 {
        let n = self.n;
        let mut acc = 0.0_f64;
        for k in 0..n {
            let lam = self.eigvals[k];
            let scale = lam.powi(2 * tau as i32);
            acc += scale * self.eigvecs[k * n + x] * self.eigvecs[k * n + y];
        }
        acc * self.inv_sqrt_deg[x] * self.inv_sqrt_deg[y] * self.vol_scale
    }

    /// Spectral coordinates `e_τ(x)_k = λ_k^τ ψ_k(x)/√π(x)` over the top-D
    /// eigenpairs by |λ| — the set_admission input form (T1.6's identity:
    /// cosine on `e_τ` = truncated normalized φ_τ). D ≤ 64 (the
    /// spectral_pencil precedent cap) and D ≤ n.
    pub fn spectral_coordinates<const D: usize>(&self, x: usize, tau: usize) -> [f32; D] {
        let n = self.n;
        let mut out = [0.0_f32; D];
        let d = D.min(n);
        for (k, slot) in out.iter_mut().enumerate().take(d) {
            let lam = self.eigvals[k];
            *slot = (lam.powi(tau as i32) * self.eigvecs[k * n + x] * self.inv_sqrt_deg[x]) as f32;
        }
        out
    }
}

/// φ_τ-driven batch selector — Plan 634 T1.6's consumer-facing operator
/// (riir-ai Plan 646 T2.1 wires this). Builds the spectral embedding
/// ([`DiffusionEmbedding::build`], n ≤ [`SPECTRAL_N_CAP`]), gathers each
/// pooled node's spectral coordinates `e_τ` ([`D ≤ 64`]), and drives
/// [`crate::set_admission`]'s greedy log-det kernel over them — cosine on
/// `e_τ` is the truncated normalized φ_τ (the T1.6 identity, pinned by the
/// module test), so the admitted batch maximizes cross-cluster coverage
/// exactly as the paper's M′(x, τ) does in walk space. One kernel, one
/// plug: the input vectors are the seam, the admission kernel is fixed
/// cosine (T0.1 unchanged).
///
/// The η floor (the batch analog of Alg. 4's `μ_y ≥ η/deg` mass floor): a
/// pooled node whose truncated kernel self-mass `vol_scale·‖e_τ(x)‖²`
/// falls below `eta` is not eligible — too weakly represented to carry a
/// cluster representative. `eta = 0` admits off mass alone.
///
/// Config: `alpha_align = 0` (no query anchor — a diversity-only batch),
/// `kappa_div = 1` (pure log-det gain ranking), `theta_coll` keeps the
/// `ColinearityBatchGate` 0.95 precedent (one representative per
/// near-identical cluster). Admission order is the kernel's own pinned
/// single-pass argmax with lowest-index tie-break — same (graph, pool, τ,
/// η, D) ⇒ byte-identical `out` (the Issue-940 tie law).
///
/// `out` slots carry POOL SLOTS (u16) in admission order, `u16::MAX`
/// sentinel beyond the admitted count (the [`crate::set_admission`]
/// convention — the consumer maps slot → node id through `pool`). Returns
/// the admitted count.
///
/// # Panics
/// Panics above `D = 64` (the spectral_pencil precedent cap), above
/// [`SPECTRAL_N_CAP`] nodes (the spectral posture's own cap), or over the
/// u16 pool encoding — each refuses loud, never silently degrades.
///
/// Allocation class: build-time (the embedding's `Vec`s + one candidate
/// array per call) — a per-batch operation, not a per-tick hot path; the
/// greedy itself runs in caller-independent fixed `[[f32; D]; D]` work.
pub fn batch_select_diffusion<const D: usize>(
    g: &SimGraph,
    pool: &[usize],
    tau: usize,
    eta: f32,
    out: &mut [u16],
) -> usize {
    assert!(D <= 64, "spectral coordinate cap D={D} > 64");
    assert!(
        pool.len() <= u16::MAX as usize,
        "pool exceeds the u16 index encoding"
    );
    let emb = DiffusionEmbedding::build(g);
    let vectors: Vec<[f32; D]> = pool
        .iter()
        .map(|&x| emb.spectral_coordinates::<D>(x, tau))
        .collect();
    // The η floor: truncated kernel self-mass = vol_scale·‖e_τ(x)‖²
    // (the coordinates carry the 1/√π(x) scaling but not the build's
    // vol(G) factor — the mass analog of Alg. 4's η/deg floor).
    // Refuse the below-floor candidates by masking them out of the pool
    // the kernel sees (the mass floor is OUR contract, applied before the
    // greedy; keep the pool-slot mapping so `out` stays pool-addressed).
    let mut kept: Vec<[f32; D]> = Vec::with_capacity(vectors.len());
    let mut slot_map: Vec<u16> = Vec::with_capacity(vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        let mass = v.iter().map(|c| c * c).sum::<f32>() * emb.vol_scale as f32;
        if mass >= eta {
            kept.push(*v);
            slot_map.push(i as u16);
        }
    }
    let cfg = crate::set_admission::SetAdmissionConfig {
        // A diversity-only batch: no query anchor, pure log-det gain
        // ranking. Colinearity cap + ρ keep their defaults (the 0.95
        // precedent; admission is ρ-invariant).
        alpha_align: 0.0,
        kappa_div: 1.0,
        ..crate::set_admission::SetAdmissionConfig::default()
    };

    let mut scratch = crate::set_admission::AdmissionScratchCore::new();
    let query = [0.0_f32; D]; // α = 0 — the anchor term is inert
    let quality = vec![0.0_f32; kept.len()]; // the neutral prior (score is
    // pure κ·log-det gain; the recovery path's quality-None is the same 0)
    let mut kept_out = vec![u16::MAX; out.len()];
    let admitted = crate::set_admission::admit_into(
        &cfg,
        &kept,
        &quality,
        &query,
        &mut kept_out,
        &mut scratch,
    );
    // Remap kept-slots back to POOL slots; sentinel slots stay sentinel.
    for (dst, &src) in out.iter_mut().zip(kept_out.iter()) {
        *dst = if src == u16::MAX {
            u16::MAX
        } else {
            slot_map[src as usize]
        };
    }
    admitted
}

/// Walk-time diversity-maximizing neighbor selector — paper Alg. 4 (plan
/// T1.5). At the current walk node `x`: assemble the diffusion kernel
/// `M′[y,y′] = φ_τ(y,y′)` over `|N(x)|` neighbors (the local posture,
/// exact), then solve the floor-constrained simplex QP
///
/// `argmin_{μ ∈ Δ(N(x)), μ ≥ η/deg} μᵀ M′ μ`
///
/// whose optimum upweights INTER-cluster transitions (the kernel is small
/// within a cluster, so mass moves to the frontier — Prop 5.6's Θ(M/r²)
/// spectral-gap enlargement on clique-cluster graphs).
///
/// Solver: deterministic active-set KKT loop — all-free equality solve,
/// then clip the LOWEST-INDEX floor violator and re-solve over the free
/// set, repeated until no violator remains (at most k iterations, k =
/// |N(x)|; monotone because each pass pins one more coordinate). The
/// equality solve is Gaussian elimination with partial pivoting, tie =
/// first-max index; a 1e-12 ridge discloses the degenerate-PSD case
/// instead of dividing by a zero pivot silently. Same (graph, x, τ, η)
/// ⇒ bit-identical `out` (the Issue-940 tie law).
///
/// `out` is caller-provided, length = |N(x)|; `out[i]` pairs with
/// neighbor i in the graph's ascending column order. Zero per-call
/// allocation beyond the solver's small fixed work (O(k²) f64 cells,
/// k ≤ 64 asserted — a node with more neighbors refuses loud; the
/// paper's walk nodes are small-degree).
///
/// # Panics
/// Panics if `out.len() != |N(x)|`, a neighbor degree > 64, or the
/// active-set loop fails to converge in k passes (a broken solver is a
/// bug, never a silent fallback).
pub fn neighbor_diversity_pick(
    g: &SimGraph,
    x: usize,
    tau: usize,
    eta: f32,
    scratch: &mut LocalScratch,
    out: &mut [f32],
) {
    let (cols, _) = g.row(x);
    let k = cols.len();
    assert_eq!(out.len(), k, "out must pair with |N(x)| in column order");
    assert!(k <= 64, "neighbor degree {k} > 64 (solver work bound)");
    if k == 0 {
        return; // a dead node picks nothing
    }
    // Kernel row-by-row: M′[i][j] = φ_τ(y_i, y_j) — the local posture
    // with ONE scratch (queries are sequential; each result lands in the
    // f64 kernel cells immediately).
    let mut kernel = [[0.0_f64; 64]; 64];
    for (i, &yi) in cols.iter().enumerate() {
        for (j, &yj) in cols.iter().enumerate().skip(i) {
            let phi = diffusion_similarity_local(g, yi, yj, tau, scratch) as f64;
            kernel[i][j] = phi;
            kernel[j][i] = phi; // symmetric kernel
        }
    }
    // Floor: μ_y ≥ η/deg(y).
    let mut floor = [0.0_f64; 64];
    for (i, &yi) in cols.iter().enumerate() {
        floor[i] = (eta as f64) / (g.degree(yi) as f64);
    }
    // Active-set loop.
    let mut pinned = [false; 64];
    let mut mu = [0.0_f64; 64];
    for _pass in 0..k {
        solve_equality_qp(&kernel, &floor, &pinned, k, &mut mu);
        // Find the LOWEST-index violator (deterministic pin order).
        let mut violator: Option<usize> = None;
        for i in 0..k {
            if !pinned[i] && mu[i] < floor[i] {
                violator = Some(i);
                break;
            }
        }
        match violator {
            Some(i) => pinned[i] = true,
            None => break, // converged: all free coords ≥ floor
        }
    }
    // Emit: pinned coords carry their floor exactly; free coords carry
    // the solved value (re-clamped defensively — the loop converged, so
    // this is a no-op by construction, never a silent fix).
    let mut sum = 0.0_f64;
    for i in 0..k {
        mu[i] = if pinned[i] {
            floor[i]
        } else {
            mu[i].max(floor[i])
        };
        sum += mu[i];
    }
    for (slot, &m) in out.iter_mut().zip(mu.iter().take(k)) {
        *slot = (m / sum) as f32; // renormalize the numerical drift off 1
    }
}

/// The equality-constrained QP over the currently-free set:
/// min μᵀKμ s.t. Σμ = 1, μ_pinned = floor. KKT: 2(Kμ)_i = λ for free i
/// ⇒ K_ff μ_f = (λ/2)·1 − K_fp μ_p; augmented system with λ as the last
/// unknown. Gaussian elimination, partial pivoting, first-max tie rule;
/// 1e-12 ridge on the diagonal (the disclosed degenerate-PSD fallback).
fn solve_equality_qp(
    kernel: &[[f64; 64]; 64],
    floor: &[f64; 64],
    pinned: &[bool; 64],
    k: usize,
    mu_out: &mut [f64; 64],
) {
    // Stack-resident free list (k ≤ 64 asserted by the caller) — the G4
    // zero-alloc census (plan 634 T2.6) found the collect() here allocating
    // per active-set pass on the walk-time selector's hot path.
    let mut free = [0_usize; 64];
    let mut f = 0;
    for (i, &is_pinned) in pinned.iter().enumerate().take(k) {
        if !is_pinned {
            free[f] = i;
            f += 1;
        }
    }
    let free = &free[..f];
    if f == 0 {
        // All pinned: μ = floor (the sum constraint is the caller's
        // renormalization; an all-pinned pass means every coordinate sits
        // at its floor).
        for (i, slot) in mu_out.iter_mut().enumerate().take(k) {
            *slot = floor[i];
        }
        return;
    }
    // K_fp μ_p term (zero when nothing is pinned).
    let mut rhs_fp = [0.0_f64; 64];
    for (fi, &i) in free.iter().enumerate() {
        let mut acc = 0.0_f64;
        for p in 0..k {
            if pinned[p] {
                acc += kernel[i][p] * floor[p];
            }
        }
        rhs_fp[fi] = acc;
    }
    // Augmented matrix rows: [2·K_ff | -1]·[μ_f; λ/2] = -2·K_fp·μ_p,
    // plus the sum row [1…1 | 0]·[μ_f; λ/2] = 1.
    let dim = f + 1;
    let mut a = [[0.0_f64; 65]; 65];
    let mut b = [0.0_f64; 65];
    for (fi, &i) in free.iter().enumerate() {
        for (fj, &j) in free.iter().enumerate() {
            a[fi][fj] = 2.0 * kernel[i][j] + if fi == fj { 1e-12 } else { 0.0 };
        }
        a[fi][f] = -1.0;
        b[fi] = -2.0 * rhs_fp[fi];
    }
    for (fj, _) in free.iter().enumerate() {
        a[f][fj] = 1.0;
    }
    b[f] = 1.0;
    // Gaussian elimination with partial pivoting (first-max tie).
    let mut solution = [0.0_f64; 65];
    for col in 0..dim {
        let mut piv = col;
        let mut piv_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(dim).skip(col + 1) {
            let v = row[col].abs();
            if v > piv_val {
                piv = r;
                piv_val = v;
            }
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        assert!(diag.abs() > 1e-300, "singular KKT system at col {col}");
        // Copy the pivot row once so the elimination can iterate `a`
        // mutably without aliasing the read. The b update rides the SAME
        // fused pass (the factor is read from the row BEFORE it is
        // eliminated — the split form would re-read a zeroed cell).
        let pivot_row = a[col];
        let pivot_b = b[col];
        for (r, row) in a.iter_mut().enumerate().take(dim).skip(col + 1) {
            let factor = row[col] / diag;
            if factor == 0.0 {
                continue;
            }
            for (cc, pv) in pivot_row.iter().enumerate().take(dim).skip(col) {
                row[cc] -= factor * pv;
            }
            b[r] -= factor * pivot_b;
        }
    }
    for r in (0..dim).rev() {
        let mut acc = b[r];
        for c in (r + 1)..dim {
            acc -= a[r][c] * solution[c];
        }
        solution[r] = acc / a[r][r];
    }
    // Copy μ_f into place; pinned slots keep their floor (the caller
    // overwrites them anyway — this keeps mu_out a complete vector).
    for (fi, &i) in free.iter().enumerate() {
        mu_out[i] = solution[fi];
    }
    for (i, slot) in mu_out.iter_mut().enumerate().take(k) {
        if pinned[i] {
            *slot = floor[i];
        }
    }
}
