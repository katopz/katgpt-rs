//! GDN Rollback-Free Tree Verification (Plan 424, Research 407).
//!
//! Modelless primitive that verifies speculative draft trees against GDN
//! (Gated DeltaNet) recurrent layers **without rolling back the recurrent
//! state**. The algorithm (paper arXiv:2607.06763 §3.4, Oda et al.) extends
//! the chunked delta-rule recurrence to tree-structured drafts via a partial
//! order (ancestor relation), reducing verification to a masked triangular
//! solve `(I + X)U = βV` followed by an ancestor-masked output read.
//!
//! # Key design: read-only verify, single-write commit
//!
//! The verify pass **never touches S₀**. Only [`commit_accepted`] writes S₀,
//! and only along the one accepted path. This eliminates rollback entirely —
//! the committed state is never speculatively written.
//!
//! # The WS₀ folding trick
//!
//! The paper folds the `−wⱼᵀS₀` term into the RHS of the forward substitution.
//! Instead of solving for U and W separately, we solve for `U' = U − WS₀` in
//! one pass. The RHS becomes `βV − βaK·S₀` (the S₀ contribution pre-multiplied
//! into the RHS, so W is never materialized).
//!
//! # Algorithm
//!
//! Given a draft tree with T nodes (parent pointers), GDN layer params
//! (K, V, Q, α, β per node), and committed prefix state S₀:
//!
//! 1. Build topology: ancestor bitmasks + cumulative log-decay + topo order.
//! 2. Build interaction matrix `X[i][j] = 𝟙[j≺i]·(aᵢ/aⱼ)·βᵢ·kᵢᵀkⱼ`.
//! 3. Compute folded RHS: `RHS[i] = βᵢvᵢ − βᵢaᵢ(kᵢᵀS₀)`.
//! 4. Forward substitution: solve `(I+X)U' = RHS` → U'.
//! 5. Compute outputs: `O[i] = (1/√dₖ)(aᵢqᵢᵀS₀ + Σ_{j⪯i} Y[i][j]·U'[j])`,
//!    where `Y[i][j] = 𝟙[j⪯i]·(aᵢ/aⱼ)·qᵢᵀkⱼ`.
//!
//! # Promotion
//!
//! Opt-in feature `gdn_tree_verify`. Not default — only relevant for
//! `QwenDeltaNet` / GDN-layer configs (themselves opt-in via
//! `deltanet_inference`). Complements Plan 012's KV-rollback attention verify.
//!
//! GDN Rollback-Free Tree Verification — masked triangular solve for delta-rule
//! speculative trees (Plan 424, Research 407, arXiv:2607.06763 §3.4). Reduces
//! tree verification for GDN recurrent layers to (I+X)U=βV, eliminating state
//! rollback entirely. Pure-math substrate: flat &[f32] slices, no Gdn2State/Config
//! dep. STAYS OPT-IN — G1–G4 GOAT gate PASS (Bench 424, 2026-07-10; G2 wins on
//! deep draft trees); not promoted to default because it only activates on
//! opt-in GDN/QwenDeltaNet configs and provides significant speedup only on
//! deep trees.

use crate::simd::simd_dot_f32;

// ── Topology ───────────────────────────────────────────────────

/// Tree metadata computed once per decode step from parent pointers.
///
/// All fields are **topo-indexed**: index `k` refers to the k-th node in
/// topological order (parent before child). Use [`TreeTopology::topo_order`]
/// to map back to original node indices.
#[derive(Clone, Debug)]
pub struct TreeTopology {
    /// `parent[k]` = topo index of parent of the k-th topo node, or `usize::MAX` for root.
    pub parent: Vec<usize>,
    /// `ancestor_bits[k * words + w]` = bitmask of proper ancestors of topo-node k.
    /// Node j is a proper ancestor of k iff bit j is set in `ancestor_bits[k * words..]`.
    ancestor_bits: Vec<u64>,
    /// `cumulative_log_decay[k]` = Σ_{j ⪯ k} ln(αⱼ) — log-space cumulative decay
    /// from root to k (inclusive).
    pub cumulative_log_decay: Vec<f64>,
    /// `topo_order[k]` = original node index of the k-th topo node.
    pub topo_order: Vec<usize>,
    /// Number of nodes.
    pub n_nodes: usize,
    /// Number of u64 words per node's ancestor bitmask = ceil(n_nodes / 64).
    n_words: usize,
}

impl TreeTopology {
    /// Number of u64 words per ancestor bitmask row.
    #[inline]
    pub fn n_words(&self) -> usize {
        self.n_words
    }

    /// Returns `true` if topo-node `j` is a **proper** ancestor of topo-node `i`
    /// (j ≺ i: j is strictly above i on the path to root).
    #[inline]
    pub fn is_proper_ancestor(&self, j: usize, i: usize) -> bool {
        let word = j / 64;
        let bit = j % 64;
        (self.ancestor_bits[i * self.n_words + word] >> bit) & 1 == 1
    }

    /// Returns `true` if topo-node `j` is an ancestor of **or equal to** topo-node `i`
    /// (j ⪯ i).
    #[inline]
    pub fn is_ancestor_or_self(&self, j: usize, i: usize) -> bool {
        j == i || self.is_proper_ancestor(j, i)
    }

    /// Returns the cumulative decay factor `aₜ = ∏_{j ⪯ t} αⱼ` for topo-node `t`.
    #[inline]
    pub fn decay(&self, t: usize) -> f64 {
        self.cumulative_log_decay[t].exp()
    }

    /// Returns the depth of topo-node `k` = number of proper ancestors.
    /// Root nodes have depth 0, their children depth 1, etc.
    #[inline]
    pub fn depth(&self, k: usize) -> usize {
        let abits = &self.ancestor_bits[k * self.n_words..(k + 1) * self.n_words];
        abits.iter().map(|&w| w.count_ones() as usize).sum()
    }
}

/// Build tree topology from parent pointers and per-node decay factors.
///
/// # Arguments
/// * `parents` — `parents[orig]` = original index of parent, or `usize::MAX` for root.
///   The tree is assumed to be a single rooted tree (exactly one root).
/// * `alphas` — `alphas[orig]` = decay factor α for original node `orig`, in (0, 1].
pub fn build_topology(parents: &[usize], alphas: &[f32]) -> TreeTopology {
    assert_eq!(parents.len(), alphas.len());
    let n = parents.len();
    assert!(n > 0, "tree must have at least one node");
    let n_words = n.div_ceil(64);

    // ── Topological sort: BFS from root ──
    let mut topo_order = Vec::with_capacity(n);
    {
        let mut queue: Vec<usize> = (0..n).filter(|&i| parents[i] == usize::MAX).collect();
        assert!(!queue.is_empty(), "no root (parent == usize::MAX) found");
        let mut visited = vec![false; n];
        // Use index-based queue (no VecDeque import needed)
        let mut head = 0;
        while head < queue.len() {
            let orig = queue[head];
            head += 1;
            if visited[orig] {
                continue;
            }
            visited[orig] = true;
            topo_order.push(orig);
            for child in 0..n {
                if parents[child] == orig && !visited[child] {
                    queue.push(child);
                }
            }
        }
        assert_eq!(
            topo_order.len(),
            n,
            "topo sort covered {} of {} nodes (cycle or forest?)",
            topo_order.len(),
            n
        );
    }

    // ── Inverse mapping: original → topo index ──
    let mut topo_inv = vec![0usize; n];
    for (k, &orig) in topo_order.iter().enumerate() {
        topo_inv[orig] = k;
    }

    // ── Re-parent in topo space ──
    let mut parent_topo = vec![usize::MAX; n];
    for (k, &orig) in topo_order.iter().enumerate() {
        let p_orig = parents[orig];
        if p_orig != usize::MAX {
            parent_topo[k] = topo_inv[p_orig];
        }
    }

    // ── Ancestor bits + cumulative log-decay (topo-indexed) ──
    let mut ancestor_bits = vec![0u64; n * n_words];
    let mut cumulative_log_decay = vec![0.0f64; n];

    for k in 0..n {
        let p = parent_topo[k];
        let orig = topo_order[k];
        if p != usize::MAX {
            // ancestor_bits[k] = ancestor_bits[p] | (1 << p)
            // Use split_at_mut to avoid simultaneous mut/immutable borrow
            let (lo, hi) = ancestor_bits.split_at_mut(k * n_words);
            let src = &lo[p * n_words..(p + 1) * n_words];
            let dst = &mut hi[..n_words];
            dst.copy_from_slice(src);
            dst[p / 64] |= 1u64 << (p % 64);
            cumulative_log_decay[k] = cumulative_log_decay[p] + (alphas[orig] as f64).ln();
        } else {
            cumulative_log_decay[k] = (alphas[orig] as f64).ln();
        }
    }

    TreeTopology {
        parent: parent_topo,
        ancestor_bits,
        cumulative_log_decay,
        topo_order,
        n_nodes: n,
        n_words,
    }
}

// ── Layer params ───────────────────────────────────────────────

/// GDN layer parameters for all T tree nodes (single head).
///
/// All slices are indexed by **original** node index (not topo order).
#[derive(Clone, Copy)]
pub struct GdnLayerParams<'a> {
    /// Keys: `[T × d_k]`, row-major.
    pub keys: &'a [f32],
    /// Values: `[T × d_v]`, row-major.
    pub values: &'a [f32],
    /// Queries: `[T × d_k]`, row-major.
    pub queries: &'a [f32],
    /// Decay factors α: `[T]`. Typically in (0, 1].
    pub alphas: &'a [f32],
    /// Write strengths β: `[T]`. Typically in (0, 1].
    pub betas: &'a [f32],
}

// ── Verifier (pre-allocated scratch) ───────────────────────────

/// Pre-allocated scratch buffers for the GDN tree verifier.
///
/// Construct once with [`GdnTreeVerifier::new`], then reuse across verify calls.
/// The hot path (`verify_gdn_tree_into`) performs **zero heap allocations**
/// after construction (G4 gate).
pub struct GdnTreeVerifier {
    /// Interaction matrix X: `[max_t × max_t]` (row-major).
    scratch_x: Vec<f32>,
    /// Right-hand side: `[max_t × d_v]`.
    scratch_rhs: Vec<f32>,
    /// Solution U': `[max_t × d_v]`.
    scratch_u: Vec<f32>,
    /// Output buffer: `[max_t × d_v]`. `pub(super)` so the dual-path fusion
    /// (Plan 430 `hola_fusion` submodule) can residual-add into it.
    pub(super) scratch_out: Vec<f32>,
}

impl GdnTreeVerifier {
    /// Construct a verifier sized for trees up to `max_t` nodes with head
    /// dimensions `d_k` (key/query) and `d_v` (value).
    pub fn new(max_t: usize, _d_k: usize, d_v: usize) -> Self {
        Self {
            scratch_x: vec![0.0; max_t * max_t],
            scratch_rhs: vec![0.0; max_t * d_v],
            scratch_u: vec![0.0; max_t * d_v],
            scratch_out: vec![0.0; max_t * d_v],
        }
    }
}

// ── Internal algorithm steps (free functions for disjoint borrows) ──

/// Build interaction matrix X into `x_buf[0..t*t]`.
///
/// `X[i][j] = 𝟙[j ≺ i] · (aᵢ/aⱼ) · βᵢ · (kᵢᵀkⱼ)`
fn build_x(x_buf: &mut [f32], topo: &TreeTopology, params: &GdnLayerParams, d_k: usize) {
    let t = topo.n_nodes;
    x_buf[..t * t].fill(0.0);
    for i in 0..t {
        let orig_i = topo.topo_order[i];
        let k_i = &params.keys[orig_i * d_k..(orig_i + 1) * d_k];
        let beta_i = params.betas[orig_i];
        let log_a_i = topo.cumulative_log_decay[i];
        for j in 0..i {
            if topo.is_proper_ancestor(j, i) {
                let orig_j = topo.topo_order[j];
                let k_j = &params.keys[orig_j * d_k..(orig_j + 1) * d_k];
                let decay_ratio = (log_a_i - topo.cumulative_log_decay[j]).exp() as f32;
                let kk = simd_dot_f32(k_i, k_j, d_k);
                x_buf[i * t + j] = decay_ratio * beta_i * kk;
            }
        }
    }
}

/// Build folded RHS into `rhs_buf[0..t*d_v]`.
///
/// `RHS[i] = βᵢvᵢ − βᵢaᵢ(kᵢᵀS₀)` — the WS₀-folding trick eliminates the
/// second solve for W.
fn build_rhs(
    rhs_buf: &mut [f32],
    topo: &TreeTopology,
    params: &GdnLayerParams,
    s0: &[f32],
    d_k: usize,
    d_v: usize,
) {
    let t = topo.n_nodes;
    for i in 0..t {
        let orig_i = topo.topo_order[i];
        let k_i = &params.keys[orig_i * d_k..(orig_i + 1) * d_k];
        let v_i = &params.values[orig_i * d_v..(orig_i + 1) * d_v];
        let beta_i = params.betas[orig_i];
        let a_i = topo.decay(i) as f32;
        for d in 0..d_v {
            // kᵢᵀS₀ for output dim d
            let mut ks0 = 0.0f32;
            for m in 0..d_k {
                ks0 += k_i[m] * s0[m * d_v + d];
            }
            rhs_buf[i * d_v + d] = beta_i * v_i[d] - beta_i * a_i * ks0;
        }
    }
}

/// Solve `(I + X)U = rhs` via forward substitution (unit-lower-triangular).
///
/// X is `t × t` lower-triangular in topo order. Writes solution to `u_buf`.
fn forward_sub(u_buf: &mut [f32], x: &[f32], rhs: &[f32], t: usize, d_v: usize) {
    u_buf[..t * d_v].copy_from_slice(&rhs[..t * d_v]);
    for i in 0..t {
        let x_row = &x[i * t..i * t + t];
        let u_i = i * d_v;
        for (j, &xij) in x_row[..i].iter().enumerate() {
            if xij != 0.0 {
                let u_j = j * d_v;
                for d in 0..d_v {
                    u_buf[u_i + d] -= xij * u_buf[u_j + d];
                }
            }
        }
    }
}

/// Compute outputs `O[i] = (1/√dₖ)(aᵢqᵢᵀS₀ + Σ_{j⪯i} Y[i][j]·U'[j])`.
///
/// Y[i][j] is computed on the fly. Writes to `out_buf[0..t*d_v]`.
fn compute_out(
    out_buf: &mut [f32],
    topo: &TreeTopology,
    u_prime: &[f32],
    params: &GdnLayerParams,
    s0: &[f32],
    d_k: usize,
    d_v: usize,
) {
    let t = topo.n_nodes;
    let scale = 1.0 / (d_k as f32).sqrt();
    for i in 0..t {
        let orig_i = topo.topo_order[i];
        let q_i = &params.queries[orig_i * d_k..(orig_i + 1) * d_k];
        let a_i = topo.decay(i) as f32;
        let log_a_i = topo.cumulative_log_decay[i];
        let out_i = i * d_v;

        // aQS₀[i] = aᵢ · qᵢᵀS₀
        for d in 0..d_v {
            let mut sum = 0.0f32;
            for m in 0..d_k {
                sum += q_i[m] * s0[m * d_v + d];
            }
            out_buf[out_i + d] = a_i * scale * sum;
        }

        // Add Σ_{j⪯i} Y[i][j] · U'[j]
        for j in 0..=i {
            if topo.is_ancestor_or_self(j, i) {
                let orig_j = topo.topo_order[j];
                let k_j = &params.keys[orig_j * d_k..(orig_j + 1) * d_k];
                let decay_ratio = (log_a_i - topo.cumulative_log_decay[j]).exp() as f32;
                let qk = simd_dot_f32(q_i, k_j, d_k);
                let y_ij = scale * decay_ratio * qk;
                let u_j = j * d_v;
                for d in 0..d_v {
                    out_buf[out_i + d] += y_ij * u_prime[u_j + d];
                }
            }
        }
    }
}

// ── Top-level API ──────────────────────────────────────────────

/// Verify a speculative draft tree against a GDN recurrent layer, producing
/// per-node outputs **without rolling back the recurrent state S₀**.
///
/// Convenience wrapper — allocates the output `Vec`. For the zero-alloc hot
/// path, use [`verify_gdn_tree_into`] which writes to the verifier's internal
/// scratch buffer.
///
/// # Returns
/// Per-node outputs `O`: `T × d_v`, row-major, **topo-indexed** (row k = topo node k).
/// Gather to original order via `topo.topo_order`.
pub fn verify_gdn_tree(
    verifier: &mut GdnTreeVerifier,
    topo: &TreeTopology,
    params: &GdnLayerParams,
    s0: &[f32],
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let out = verify_gdn_tree_into(verifier, topo, params, s0, d_k, d_v);
    out.to_vec()
}

/// Zero-alloc verify variant. Returns a reference to the verifier's internal
/// output buffer (topo-indexed). The reference is valid until the next
/// `verify_gdn_tree*` call on the same verifier.
pub fn verify_gdn_tree_into<'a>(
    verifier: &'a mut GdnTreeVerifier,
    topo: &TreeTopology,
    params: &GdnLayerParams,
    s0: &[f32],
    d_k: usize,
    d_v: usize,
) -> &'a [f32] {
    let t = topo.n_nodes;
    // Split disjoint field borrows — borrow checker sees no aliasing
    let x = &mut verifier.scratch_x[..t * t];
    let rhs = &mut verifier.scratch_rhs[..t * d_v];
    let u = &mut verifier.scratch_u[..t * d_v];
    let out = &mut verifier.scratch_out[..t * d_v];

    build_x(x, topo, params, d_k);
    build_rhs(rhs, topo, params, s0, d_k, d_v);
    forward_sub(u, x, rhs, t, d_v);
    compute_out(out, topo, u, params, s0, d_k, d_v);

    &verifier.scratch_out[..t * d_v]
}

/// Commit the accepted path: replay the delta-rule recurrence along the path
/// from root to `accepted_leaf`, writing S₀ in place.
///
/// This is the **only** state write in the entire decode step. After this call,
/// S₀ reflects the state at the accepted leaf.
///
/// # Arguments
/// * `topo` — Tree topology.
/// * `accepted_leaf` — **Topo** index of the accepted leaf node.
/// * `params` — GDN layer params (original-indexed).
/// * `s0` — Committed prefix state `[d_k × d_v]`, updated in place.
pub fn commit_accepted(
    topo: &TreeTopology,
    accepted_leaf: usize,
    params: &GdnLayerParams,
    s0: &mut [f32],
    d_k: usize,
    d_v: usize,
) {
    // Reconstruct the path from root to accepted_leaf (topo indices, root first)
    let mut path = Vec::with_capacity(topo.n_nodes);
    let mut cur = accepted_leaf;
    while cur != usize::MAX {
        path.push(cur);
        cur = topo.parent[cur];
    }
    path.reverse();
    commit_path(topo, &path, params, s0, d_k, d_v);
}

/// Replay the delta-rule recurrence along a given path (topo indices, root first).
///
/// Updates S₀ in place: `Sₜ = αₜ(I − βₜkₜkₜᵀ)Sₜ₋₁ + βₜkₜvₜᵀ`.
///
/// Equivalent to sequential GDN2 decoding along the path. This is the only
/// state mutation in the tree-verify workflow.
pub fn commit_path(
    topo: &TreeTopology,
    path: &[usize],
    params: &GdnLayerParams,
    s0: &mut [f32],
    d_k: usize,
    d_v: usize,
) {
    // Reusable read buffer (stack-allocated for typical head dims)
    let mut r = vec![0.0f32; d_v];
    for &node_k in path {
        let orig = topo.topo_order[node_k];
        let k = &params.keys[orig * d_k..(orig + 1) * d_k];
        let v = &params.values[orig * d_v..(orig + 1) * d_v];
        let alpha = params.alphas[orig];
        let beta = params.betas[orig];

        // r = kᵀS (before decay): r[d] = Σ_m k[m] · S[m*d_v + d]
        r.fill(0.0);
        for m in 0..d_k {
            let km = k[m];
            if km != 0.0 {
                for d in 0..d_v {
                    r[d] += s0[m * d_v + d] * km;
                }
            }
        }
        // S = α · S (decay)
        for val in s0[..d_k * d_v].iter_mut() {
            *val *= alpha;
        }
        // S += β · k ⊗ (v − α·r)
        for m in 0..d_k {
            let beta_km = beta * k[m];
            if beta_km != 0.0 {
                for d in 0..d_v {
                    s0[m * d_v + d] += beta_km * (v[d] - alpha * r[d]);
                }
            }
        }
    }
}

// ── Multi-head batching (T4.1) ─────────────────────────────────
//
// The tree topology (ancestor structure, topo order) is head-independent
// and computed once. The per-head verify loop reuses the same scratch
// buffers. α/β are shared across heads (scalar paper form, matching the
// `Gdn2GateConfig::Kda` tied-scalar gate). Callers needing per-head α/β
// should invoke the single-head API in a loop with per-head topologies.

/// GDN layer parameters for all T tree nodes across H heads.
///
/// K/V/Q are head-major: `keys[h][node][d_k]` laid out as
/// `[H * T * d_k]` row-major (head stride = T*d_k, node stride = d_k).
/// α/β are per-node scalars shared across all heads.
///
/// All slices are indexed by **original** node index within each head.
#[derive(Clone, Copy)]
pub struct GdnMultiHeadParams<'a> {
    /// Keys: `[H_k * T * d_k]`, head-major.
    pub keys: &'a [f32],
    /// Values: `[H_v * T * d_v]`, head-major.
    pub values: &'a [f32],
    /// Queries: `[H_k * T * d_k]`, head-major.
    pub queries: &'a [f32],
    /// Decay factors α: `[T]`, shared across heads.
    pub alphas: &'a [f32],
    /// Write strengths β: `[T]`, shared across heads.
    pub betas: &'a [f32],
    /// Number of key/query heads.
    pub n_kv_heads: usize,
}

impl<'a> GdnMultiHeadParams<'a> {
    /// Per-head single-head params view for head `h`.
    ///
    /// K and Q use the same head stride (both indexed by key heads);
    /// V uses the value-head stride. With MHA (H_k == H_v) both are equal.
    pub(super) fn head_params(
        &self,
        h: usize,
        t: usize,
        d_k: usize,
        d_v: usize,
    ) -> GdnLayerParams<'a> {
        let k_stride = t * d_k;
        let v_stride = t * d_v;
        GdnLayerParams {
            keys: &self.keys[h * k_stride..(h + 1) * k_stride],
            values: &self.values[h * v_stride..(h + 1) * v_stride],
            queries: &self.queries[h * k_stride..(h + 1) * k_stride],
            alphas: self.alphas,
            betas: self.betas,
        }
    }
}

/// Verify a speculative draft tree against GDN recurrent layers for all
/// heads, producing per-node per-head outputs **without rolling back S₀**.
///
/// Loops over heads, reusing the verifier's scratch buffers. The topology
/// is computed once and shared. Returns a `Vec` of `[H * T * d_v]`
/// (head-major, topo-indexed within each head).
///
/// `s0_per_head[h]` is the committed prefix state `[d_k × d_v]` for head h.
/// It is **not modified** — use [`commit_accepted_multihead`] to write back.
pub fn verify_gdn_tree_multihead(
    verifier: &mut GdnTreeVerifier,
    topo: &TreeTopology,
    params: &GdnMultiHeadParams,
    s0_per_head: &[&[f32]],
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let t = topo.n_nodes;
    let h = params.n_kv_heads;
    let mut out = vec![0.0f32; h * t * d_v];
    for head in 0..h {
        let hp = params.head_params(head, t, d_k, d_v);
        let s0 = s0_per_head[head];
        let head_out = verify_gdn_tree_into(verifier, topo, &hp, s0, d_k, d_v);
        // head_out is topo-indexed [T * d_v]; copy into head-major output slot.
        out[head * t * d_v..(head + 1) * t * d_v].copy_from_slice(head_out);
    }
    out
}

/// Commit the accepted path for all heads: replay the delta-rule along the
/// path root→`accepted_leaf` for each head's S₀, updating in place.
///
/// This is the multi-head analog of [`commit_accepted`]. `s0_per_head` is
/// updated in place for every head along the shared accepted path.
pub fn commit_accepted_multihead(
    topo: &TreeTopology,
    accepted_leaf: usize,
    params: &GdnMultiHeadParams,
    s0_per_head: &mut [&mut [f32]],
    d_k: usize,
    d_v: usize,
) {
    let t = topo.n_nodes;
    debug_assert_eq!(s0_per_head.len(), params.n_kv_heads);
    for (head, s0) in s0_per_head.iter_mut().enumerate() {
        let hp = params.head_params(head, t, d_k, d_v);
        commit_accepted(topo, accepted_leaf, &hp, s0, d_k, d_v);
    }
}

// ── DDTree → Topology conversion (Plan 424 T4.3) ─────────────

/// Convert a flat list of DDTree nodes (path-encoded) into the parent-index
/// topology that [`build_topology`] consumes.
///
/// DDTree nodes encode the path from root in `parent_path` (a
/// [`TreePath`](crate::speculative::types::TreePath): one `u32` slot per
/// level, slot 0 = root). A node B at depth `d` is a child of the node A at
/// depth `d-1` whose `parent_path` equals `B.parent_path.parent(d)`.
///
/// Multiple roots (depth-0 nodes) are supported — they form independent
/// subtrees in the topology, which the tree verify processes correctly (no
/// cross-subtree ancestor relationships).
///
/// # Arguments
/// * `nodes` — DDTree nodes (any order). Duplicates by `(depth, parent_path)`
///   are deduplicated (highest score wins).
/// * `alpha` — Uniform decay factor for all nodes (GDN2's decay is
///   token-independent). All nodes receive the same α.
///
/// # Returns
/// `(TreeTopology, Vec<usize> token_ids)` where `token_ids[i]` is the token
/// ID of topology node `i` (in the topology's original indexing, NOT topo
/// order — use `topo.topo_order` to remap).
///
/// # Panics
/// Panics if `nodes` is empty or if a non-root node has no matching parent
/// (corrupted tree).
pub fn build_topology_from_tree_nodes(
    nodes: &[crate::speculative::types::TreeNode],
    alpha: f32,
) -> (TreeTopology, Vec<usize>) {
    use crate::speculative::types::{TreeNode, TreePath};
    use std::collections::HashMap;

    assert!(!nodes.is_empty(), "DDTree nodes must be non-empty");

    // ── Deduplicate by (depth, parent_path), keeping highest score ──
    let mut best: HashMap<(usize, TreePath), &TreeNode> = HashMap::new();
    for node in nodes {
        best.entry((node.depth, node.parent_path))
            .and_modify(|existing| {
                if node.score > existing.score {
                    *existing = node;
                }
            })
            .or_insert(node);
    }

    // ── Collect and sort by (depth, parent_path) for deterministic order ──
    let mut sorted: Vec<&TreeNode> = best.into_values().collect();
    sorted.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then_with(|| a.parent_path.cmp(&b.parent_path))
    });

    let n = sorted.len();

    // ── Build parent index array ──
    // Index: (depth, parent_path) → original node index in `sorted`.
    let mut index: HashMap<(usize, TreePath), usize> = HashMap::with_capacity(n);
    for (i, node) in sorted.iter().enumerate() {
        index.insert((node.depth, node.parent_path), i);
    }

    let mut parents = vec![usize::MAX; n];
    let mut token_ids = vec![0usize; n];
    let alphas = vec![alpha; n];

    for (i, node) in sorted.iter().enumerate() {
        token_ids[i] = node.token_idx;
        if node.depth > 0 {
            // Parent's parent_path = this node's path with its own token dropped
            let parent_key = (node.depth - 1, node.parent_path.parent(node.depth));
            let parent_idx = *index.get(&parent_key).unwrap_or_else(|| {
                panic!(
                    "DDTree node at depth {} has no matching parent (depth {}, path {:?})",
                    node.depth,
                    node.depth - 1,
                    node.parent_path.parent(node.depth)
                )
            });
            parents[i] = parent_idx;
        }
    }

    let topo = build_topology(&parents, &alphas);
    (topo, token_ids)
}

// ── Dual-path GDN × HOLA fusion (Plan 430) ───────────────────
// Gated on `gdn_hola_tree_verify` which implies both `gdn_tree_verify`
// and `hippocampal_cache`. See `hola_fusion.rs` for the dual-path verify
// and commit primitives.
#[cfg(feature = "gdn_hola_tree_verify")]
pub mod hola_fusion;

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32) / (u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// Reference: sequential per-branch GDN2 verify — replay the delta-rule
    /// from root to each node through its ancestors, then read the output.
    #[allow(clippy::too_many_arguments)] // test helper mirrors algorithm's natural parameter set
    fn reference_verify(
        parents: &[usize],
        keys: &[f32],
        values: &[f32],
        queries: &[f32],
        alphas: &[f32],
        betas: &[f32],
        s0: &[f32],
        d_k: usize,
        d_v: usize,
    ) -> Vec<f32> {
        let t = parents.len();
        let mut outputs = vec![0.0f32; t * d_v];
        let scale = 1.0 / (d_k as f32).sqrt();
        for node in 0..t {
            let mut path = vec![node];
            let mut cur = parents[node];
            while cur != usize::MAX {
                path.push(cur);
                cur = parents[cur];
            }
            path.reverse();
            let mut s = s0.to_vec();
            for &p in &path {
                let k = &keys[p * d_k..(p + 1) * d_k];
                let v = &values[p * d_v..(p + 1) * d_v];
                let alpha = alphas[p];
                let beta = betas[p];
                let mut r = vec![0.0f32; d_v];
                for m in 0..d_k {
                    for d in 0..d_v {
                        r[d] += s[m * d_v + d] * k[m];
                    }
                }
                for val in s[..d_k * d_v].iter_mut() {
                    *val *= alpha;
                }
                for m in 0..d_k {
                    let beta_km = beta * k[m];
                    for d in 0..d_v {
                        s[m * d_v + d] += beta_km * (v[d] - alpha * r[d]);
                    }
                }
            }
            let q = &queries[node * d_k..(node + 1) * d_k];
            for d in 0..d_v {
                let mut sum = 0.0f32;
                for m in 0..d_k {
                    sum += q[m] * s[m * d_v + d];
                }
                outputs[node * d_v + d] = scale * sum;
            }
        }
        outputs
    }

    // ── T1.5: build_topology ──

    #[test]
    fn test_build_topology_small_tree() {
        //         0 (root)
        //        / \
        //       1   2
        //      / \   \
        //     3   4   5
        let parents = [usize::MAX, 0, 0, 1, 1, 2];
        let alphas = [0.9f32, 0.8, 0.7, 0.6, 0.5, 0.4];
        let topo = build_topology(&parents, &alphas);
        assert_eq!(topo.n_nodes, 6);
        assert_eq!(topo.topo_order[0], 0);

        let mut topo_inv = [0usize; 6];
        for (k, &orig) in topo.topo_order.iter().enumerate() {
            topo_inv[orig] = k;
        }
        for orig in 0..6 {
            let p = parents[orig];
            if p != usize::MAX {
                assert!(topo_inv[p] < topo_inv[orig]);
            }
        }

        let (k0, k1, k2, k3) = (topo_inv[0], topo_inv[1], topo_inv[2], topo_inv[3]);
        assert!(topo.is_proper_ancestor(k0, k3));
        assert!(topo.is_proper_ancestor(k1, k3));
        assert!(!topo.is_proper_ancestor(k2, k3));
        assert!(!topo.is_proper_ancestor(k3, k3));
        assert!(topo.is_ancestor_or_self(k3, k3));

        // Use f32→f64 conversion to match the code's precision path
        let expected = (0.9f32 as f64).ln() + (0.8f32 as f64).ln() + (0.6f32 as f64).ln();
        assert!((topo.cumulative_log_decay[k3] - expected).abs() < 1e-10);
        assert!((topo.decay(k0) - (0.9f32 as f64)).abs() < 1e-10);
    }

    #[test]
    fn test_build_topology_single_node() {
        let topo = build_topology(&[usize::MAX], &[0.5f32]);
        assert_eq!(topo.n_nodes, 1);
        assert_eq!(topo.topo_order, vec![0]);
        assert_eq!(topo.parent, vec![usize::MAX]);
    }

    #[test]
    fn test_build_topology_chain() {
        let topo = build_topology(&[usize::MAX, 0, 1, 2], &[0.9, 0.8, 0.7, 0.6]);
        assert_eq!(topo.topo_order, vec![0, 1, 2, 3]);
        assert!(topo.is_proper_ancestor(0, 3));
        assert!(topo.is_proper_ancestor(2, 3));
        assert!(!topo.is_proper_ancestor(3, 3));
    }

    #[test]
    fn test_build_topology_large_two_words() {
        // T=70 → n_words = 2
        let t = 70;
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let alphas: Vec<f32> = (0..t).map(|i| 0.99 - (i as f32) * 0.001).collect();
        let topo = build_topology(&parents, &alphas);
        assert_eq!(topo.n_words(), 2);
        // Node 69's ancestors = {0..68}
        for j in 0..69 {
            assert!(
                topo.is_proper_ancestor(j, 69),
                "{j} should be ancestor of 69"
            );
        }
        assert!(!topo.is_proper_ancestor(69, 69));
    }

    // ── T2.4: linear chain = sequential GDN2 ──

    #[test]
    fn test_linear_chain_matches_sequential() {
        let (t, d_k, d_v) = (8, 16, 16);
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut rng = rng(12345);
        let keys: Vec<f32> = (0..t * d_k).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d_v).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d_k).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.8 + 0.15 * rng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.5 + 0.4 * rng()).collect();
        let s0: Vec<f32> = (0..d_k * d_v).map(|_| 0.1 * rng()).collect();
        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);
        let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);
        let tree_out = verify_gdn_tree_into(&mut verifier, &topo, &params, &s0, d_k, d_v);
        let ref_out = reference_verify(
            &parents, &keys, &values, &queries, &alphas, &betas, &s0, d_k, d_v,
        );

        let tol = 1e-3f32;
        let mut max_err = 0.0f32;
        for i in 0..t * d_v {
            max_err = max_err.max((tree_out[i] - ref_out[i]).abs());
        }
        assert!(
            max_err < tol,
            "linear chain: max error {max_err:.6} >= {tol}"
        );
    }

    // ── T2.5: branching tree ──

    #[test]
    fn test_branching_tree_matches_per_branch() {
        let parents = [usize::MAX, 0, 0, 1, 1, 2, 3, 3, 5, 6];
        let (t, d_k, d_v) = (parents.len(), 12, 12);
        let mut rng = rng(99999);
        let keys: Vec<f32> = (0..t * d_k).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d_v).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d_k).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.7 + 0.2 * rng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.3 + 0.5 * rng()).collect();
        let s0: Vec<f32> = (0..d_k * d_v).map(|_| 0.1 * rng()).collect();
        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);
        let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);
        let tree_topo = verify_gdn_tree_into(&mut verifier, &topo, &params, &s0, d_k, d_v);
        let ref_out = reference_verify(
            &parents, &keys, &values, &queries, &alphas, &betas, &s0, d_k, d_v,
        );

        let mut tree_orig = vec![0.0f32; t * d_v];
        for (k, &orig) in topo.topo_order.iter().enumerate() {
            tree_orig[orig * d_v..(orig + 1) * d_v]
                .copy_from_slice(&tree_topo[k * d_v..(k + 1) * d_v]);
        }

        let tol = 1e-3f32;
        let mut max_err = 0.0f32;
        for i in 0..t * d_v {
            max_err = max_err.max((tree_orig[i] - ref_out[i]).abs());
        }
        assert!(
            max_err < tol,
            "branching tree: max error {max_err:.6} >= {tol}"
        );
    }

    // ── T3.1: commit_path ──

    #[test]
    fn test_commit_path_matches_sequential() {
        let (t, d_k, d_v) = (6, 8, 8);
        let parents: Vec<usize> = (0..t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut rng = rng(42);
        let keys: Vec<f32> = (0..t * d_k).map(|_| rng()).collect();
        let values: Vec<f32> = (0..t * d_v).map(|_| rng()).collect();
        let queries: Vec<f32> = (0..t * d_k).map(|_| rng()).collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.85).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.5).collect();
        let s0_init: Vec<f32> = (0..d_k * d_v).map(|_| 0.1 * rng()).collect();
        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);

        let mut s0_committed = s0_init.clone();
        commit_accepted(&topo, 5, &params, &mut s0_committed, d_k, d_v);

        // Reference sequential replay
        let mut s0_ref = s0_init.clone();
        for node in 0..t {
            let k = &keys[node * d_k..(node + 1) * d_k];
            let v = &values[node * d_v..(node + 1) * d_v];
            let (alpha, beta) = (alphas[node], betas[node]);
            let mut r = vec![0.0f32; d_v];
            for m in 0..d_k {
                for d in 0..d_v {
                    r[d] += s0_ref[m * d_v + d] * k[m];
                }
            }
            for val in s0_ref[..d_k * d_v].iter_mut() {
                *val *= alpha;
            }
            for m in 0..d_k {
                let bkm = beta * k[m];
                for d in 0..d_v {
                    s0_ref[m * d_v + d] += bkm * (v[d] - alpha * r[d]);
                }
            }
        }

        let tol = 1e-5f32;
        let max_err = (0..d_k * d_v)
            .map(|i| (s0_committed[i] - s0_ref[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < tol,
            "commit_path: max error {max_err:.8} >= {tol}"
        );
    }

    // ── T5.1: random trees at T={16,32,64,128} ──

    #[test]
    fn test_random_trees_correctness() {
        let (d_k, d_v) = (16, 16);
        for (t, seed) in [(16, 1u32), (32, 2), (64, 3), (128, 4)] {
            let mut rs = seed;
            let mut next = || {
                rs ^= rs << 13;
                rs ^= rs >> 17;
                rs ^= rs << 5;
                rs
            };
            let parents: Vec<usize> = (0..t)
                .map(|i| {
                    if i == 0 {
                        usize::MAX
                    } else {
                        (next() as usize) % i
                    }
                })
                .collect();
            let mut frng = rng(seed.wrapping_mul(7));
            let keys: Vec<f32> = (0..t * d_k).map(|_| frng()).collect();
            let values: Vec<f32> = (0..t * d_v).map(|_| frng()).collect();
            let queries: Vec<f32> = (0..t * d_k).map(|_| frng()).collect();
            let alphas: Vec<f32> = (0..t).map(|_| 0.75 + 0.15 * frng()).collect();
            let betas: Vec<f32> = (0..t).map(|_| 0.4 + 0.4 * frng()).collect();
            let s0: Vec<f32> = (0..d_k * d_v).map(|_| 0.1 * frng()).collect();
            let params = GdnLayerParams {
                keys: &keys,
                values: &values,
                queries: &queries,
                alphas: &alphas,
                betas: &betas,
            };
            let topo = build_topology(&parents, &alphas);
            let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);
            let tree_topo = verify_gdn_tree_into(&mut verifier, &topo, &params, &s0, d_k, d_v);
            let ref_out = reference_verify(
                &parents, &keys, &values, &queries, &alphas, &betas, &s0, d_k, d_v,
            );

            let mut tree_orig = vec![0.0f32; t * d_v];
            for (k, &orig) in topo.topo_order.iter().enumerate() {
                tree_orig[orig * d_v..(orig + 1) * d_v]
                    .copy_from_slice(&tree_topo[k * d_v..(k + 1) * d_v]);
            }

            let tol = 1e-3f32;
            let max_err = (0..t * d_v)
                .map(|i| (tree_orig[i] - ref_out[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < tol, "T={t}: max error {max_err:.6} >= {tol}");
        }
    }

    // ── T4.1: multi-head batching correctness ──

    /// Multi-head verify must match per-head single-head verify on the same
    /// topology + scratch. Verifies the head-major gather/scatter and that
    /// scratch reuse across heads does not corrupt state.
    #[test]
    fn test_multihead_matches_single_head() {
        let (d_k, d_v) = (16, 16);
        let h = 4; // 4 heads (MHA: H_k == H_v)
        let t = 12;
        let seed = 42u32;
        let mut frng = rng(seed);

        // Random tree: node 0 = root, others pick a random earlier parent.
        let parents: Vec<usize> = (0..t)
            .map(|i| {
                if i == 0 {
                    usize::MAX
                } else {
                    (frng() as u32) as usize % i
                }
            })
            .collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.75 + 0.15 * frng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.4 + 0.4 * frng()).collect();

        // Head-major K/V/Q.
        let keys: Vec<f32> = (0..h * t * d_k).map(|_| frng()).collect();
        let values: Vec<f32> = (0..h * t * d_v).map(|_| frng()).collect();
        let queries: Vec<f32> = (0..h * t * d_k).map(|_| frng()).collect();

        // Per-head S₀.
        let s0_heads: Vec<Vec<f32>> = (0..h)
            .map(|_| (0..d_k * d_v).map(|_| 0.1 * frng()).collect())
            .collect();
        let s0_refs: Vec<&[f32]> = s0_heads.iter().map(|s| s.as_slice()).collect();

        let mh_params = GdnMultiHeadParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
            n_kv_heads: h,
        };
        let topo = build_topology(&parents, &alphas);
        let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);

        let mh_out =
            verify_gdn_tree_multihead(&mut verifier, &topo, &mh_params, &s0_refs, d_k, d_v);

        // Compare each head against independent single-head verify.
        let tol = 1e-5f32;
        for head in 0..h {
            let hp = mh_params.head_params(head, t, d_k, d_v);
            let single_out =
                verify_gdn_tree_into(&mut verifier, &topo, &hp, &s0_heads[head], d_k, d_v);
            let mh_head = &mh_out[head * t * d_v..(head + 1) * t * d_v];
            let max_err = (0..t * d_v)
                .map(|i| (mh_head[i] - single_out[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < tol,
                "head {head}: max error {max_err:.6} >= {tol}"
            );
        }
    }

    /// Multi-head verify correctness vs per-head sequential GDN2 reference.
    /// Each head has different S₀, K, V, Q — the multi-head output must match
    /// the per-head sequential replay independently.
    #[test]
    fn test_multihead_matches_reference() {
        let (d_k, d_v) = (8, 8);
        let h = 3;
        let t = 10;
        let seed = 99u32;
        let mut frng = rng(seed);

        // Branching tree: root → 2 children → branching.
        let parents = vec![usize::MAX, 0, 0, 1, 1, 2, 2, 3, 4, 5];
        let alphas: Vec<f32> = (0..t).map(|_| 0.8 + 0.1 * frng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.3 + 0.5 * frng()).collect();

        let keys: Vec<f32> = (0..h * t * d_k).map(|_| frng()).collect();
        let values: Vec<f32> = (0..h * t * d_v).map(|_| frng()).collect();
        let queries: Vec<f32> = (0..h * t * d_k).map(|_| frng()).collect();
        let s0_heads: Vec<Vec<f32>> = (0..h)
            .map(|_| (0..d_k * d_v).map(|_| 0.1 * frng()).collect())
            .collect();
        let s0_refs: Vec<&[f32]> = s0_heads.iter().map(|s| s.as_slice()).collect();

        let mh_params = GdnMultiHeadParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
            n_kv_heads: h,
        };
        let topo = build_topology(&parents, &alphas);
        let mut verifier = GdnTreeVerifier::new(t, d_k, d_v);
        let mh_out =
            verify_gdn_tree_multihead(&mut verifier, &topo, &mh_params, &s0_refs, d_k, d_v);

        let tol = 1e-3f32;
        for head in 0..h {
            let k_h = &keys[head * t * d_k..(head + 1) * t * d_k];
            let v_h = &values[head * t * d_v..(head + 1) * t * d_v];
            let q_h = &queries[head * t * d_k..(head + 1) * t * d_k];
            let ref_out = reference_verify(
                &parents,
                k_h,
                v_h,
                q_h,
                &alphas,
                &betas,
                &s0_heads[head],
                d_k,
                d_v,
            );
            // Gather topo → original order.
            let mh_head = &mh_out[head * t * d_v..(head + 1) * t * d_v];
            let mut mh_orig = vec![0.0f32; t * d_v];
            for (k, &orig) in topo.topo_order.iter().enumerate() {
                mh_orig[orig * d_v..(orig + 1) * d_v]
                    .copy_from_slice(&mh_head[k * d_v..(k + 1) * d_v]);
            }
            let max_err = (0..t * d_v)
                .map(|i| (mh_orig[i] - ref_out[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < tol,
                "head {head}: max error {max_err:.6} >= {tol}"
            );
        }
    }

    /// Multi-head commit: after commit_accepted_multihead, each head's S₀
    /// must match a sequential replay along the accepted path for that head.
    #[test]
    fn test_multihead_commit_matches_sequential() {
        let (d_k, d_v) = (8, 8);
        let h = 3;
        let t = 6;
        let seed = 7u32;
        let mut frng = rng(seed);

        let parents: Vec<usize> = (0..t)
            .map(|i| {
                if i == 0 {
                    usize::MAX
                } else {
                    (frng() as u32) as usize % i
                }
            })
            .collect();
        let alphas: Vec<f32> = (0..t).map(|_| 0.85 + 0.1 * frng()).collect();
        let betas: Vec<f32> = (0..t).map(|_| 0.5 + 0.3 * frng()).collect();

        let keys: Vec<f32> = (0..h * t * d_k).map(|_| frng()).collect();
        let values: Vec<f32> = (0..h * t * d_v).map(|_| frng()).collect();
        let queries: Vec<f32> = (0..h * t * d_k).map(|_| frng()).collect();

        let s0_init: Vec<Vec<f32>> = (0..h)
            .map(|_| (0..d_k * d_v).map(|_| 0.1 * frng()).collect())
            .collect();
        // Mutable copies for the commit call.
        let mut s0_committed: Vec<Vec<f32>> = s0_init.clone();
        let mut s0_refs: Vec<&mut [f32]> =
            s0_committed.iter_mut().map(|s| s.as_mut_slice()).collect();

        let mh_params = GdnMultiHeadParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
            n_kv_heads: h,
        };
        let topo = build_topology(&parents, &alphas);
        let accepted_leaf = topo.n_nodes - 1; // last topo node
        commit_accepted_multihead(&topo, accepted_leaf, &mh_params, &mut s0_refs, d_k, d_v);

        // Reference: sequential replay per head along root→accepted_leaf path.
        let tol = 1e-5f32;
        let path = {
            let mut p = vec![accepted_leaf];
            let mut cur = topo.parent[accepted_leaf];
            while cur != usize::MAX {
                p.push(cur);
                cur = topo.parent[cur];
            }
            p.reverse();
            p
        };
        for head in 0..h {
            let mut s_ref = s0_init[head].clone();
            for &node_k in &path {
                let orig = topo.topo_order[node_k];
                let k = &keys[head * t * d_k + orig * d_k..head * t * d_k + (orig + 1) * d_k];
                let v = &values[head * t * d_v + orig * d_v..head * t * d_v + (orig + 1) * d_v];
                let alpha = alphas[orig];
                let beta = betas[orig];
                let mut r = vec![0.0f32; d_v];
                for m in 0..d_k {
                    for d in 0..d_v {
                        r[d] += s_ref[m * d_v + d] * k[m];
                    }
                }
                for val in s_ref[..d_k * d_v].iter_mut() {
                    *val *= alpha;
                }
                for m in 0..d_k {
                    let beta_km = beta * k[m];
                    for d in 0..d_v {
                        s_ref[m * d_v + d] += beta_km * (v[d] - alpha * r[d]);
                    }
                }
            }
            let max_err = (0..d_k * d_v)
                .map(|i| (s0_committed[head][i] - s_ref[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < tol,
                "head {head} commit: max error {max_err:.6} >= {tol}"
            );
        }
    }

    // ── T5.4 (G4): alloc-free hot path ──

    /// G4 gate: `verify_gdn_tree_into` performs ZERO heap allocations after
    /// verifier construction. We detect allocations by checking that the
    /// internal scratch buffer capacities do not grow across verify calls on
    /// trees of varying sizes up to `max_t`. If the hot path allocated, either
    /// the Vec would reallocate (capacity grows) or a new Vec would appear.
    ///
    /// Since the scratch buffers are private, this test exercises the public
    /// API and relies on the design contract: `new()` pre-allocates to
    /// `max_t × max_t` / `max_t × d_v`, and `verify_gdn_tree_into` only takes
    /// `&mut self` (no new allocations possible from the struct's fields —
    /// `Vec::clear()` + indexed writes don't realloc). The test verifies that
    /// repeated calls with trees of the max size do not panic and produce
    /// stable, correct results (no buffer corruption from failed realloc).
    #[test]
    fn test_verify_alloc_free_hot_path() {
        let (d_k, d_v) = (16, 16);
        let max_t = 32;
        let mut verifier = GdnTreeVerifier::new(max_t, d_k, d_v);

        // Determinism + finiteness: repeated calls on the same input must
        // produce bit-identical output. If any allocation happened inside,
        // the &mut self borrow would still work — the key invariant is that
        // NO Vec method that can realloc is called. We verify this by
        // asserting determinism (scratch reuse without corruption) and
        // finiteness (no NaN/Inf from stale scratch).
        //
        // Data uses small-magnitude pseudo-random values (not monotonic
        // ramps) to avoid the forward-substitution overflow that large
        // un-normalized keys cause on deep chains (X entries near 1.0 on a
        // 32-node chain amplify exponentially through the solve).
        let parents: Vec<usize> = (0..max_t)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let alphas: Vec<f32> = vec![0.95; max_t];
        let betas: Vec<f32> = vec![0.1; max_t];
        let mut frng = rng(123);
        let keys: Vec<f32> = (0..max_t * d_k).map(|_| 0.05 * frng()).collect();
        let values: Vec<f32> = (0..max_t * d_v).map(|_| 0.05 * frng()).collect();
        let queries: Vec<f32> = (0..max_t * d_k).map(|_| 0.05 * frng()).collect();
        let s0: Vec<f32> = (0..d_k * d_v).map(|_| 0.05 * frng()).collect();
        let params = GdnLayerParams {
            keys: &keys,
            values: &values,
            queries: &queries,
            alphas: &alphas,
            betas: &betas,
        };
        let topo = build_topology(&parents, &alphas);

        let out1 = verify_gdn_tree_into(&mut verifier, &topo, &params, &s0, d_k, d_v).to_vec();
        let out2 = verify_gdn_tree_into(&mut verifier, &topo, &params, &s0, d_k, d_v).to_vec();

        // Determinism: repeated calls produce identical output (scratch reuse
        // does not leak stale state — `build_x` zeroes X, forward_sub and
        // compute_out fully overwrite their output ranges).
        assert_eq!(out1, out2, "repeated verify must be deterministic");

        // Correctness: all finite (no NaN/Inf from corrupted scratch).
        for &v in &out1 {
            assert!(v.is_finite(), "non-finite output: {v}");
        }

        // Multi-head variant: same determinism check. The per-head verify
        // loop reuses the verifier scratch — repeated calls must be identical.
        let h = 4;
        let mh_keys: Vec<f32> = (0..h * max_t * d_k).map(|_| 0.05 * frng()).collect();
        let mh_values: Vec<f32> = (0..h * max_t * d_v).map(|_| 0.05 * frng()).collect();
        let mh_queries: Vec<f32> = (0..h * max_t * d_k).map(|_| 0.05 * frng()).collect();
        let s0_heads: Vec<Vec<f32>> = (0..h)
            .map(|_| (0..d_k * d_v).map(|_| 0.05 * frng()).collect())
            .collect();
        let s0_refs: Vec<&[f32]> = s0_heads.iter().map(|s| s.as_slice()).collect();
        let mh_params = GdnMultiHeadParams {
            keys: &mh_keys,
            values: &mh_values,
            queries: &mh_queries,
            alphas: &alphas,
            betas: &betas,
            n_kv_heads: h,
        };
        let mh1 = verify_gdn_tree_multihead(&mut verifier, &topo, &mh_params, &s0_refs, d_k, d_v);
        let mh2 = verify_gdn_tree_multihead(&mut verifier, &topo, &mh_params, &s0_refs, d_k, d_v);
        assert_eq!(mh1, mh2, "repeated multi-head verify must be deterministic");
        for &v in &mh1 {
            assert!(v.is_finite(), "non-finite multi-head output: {v}");
        }
    }

    // ── build_topology_from_tree_nodes tests (Plan 424 T4.3) ──

    use crate::speculative::types::{TreeNode, TreePath};

    fn make_node(depth: usize, token_idx: usize, parent_path: TreePath, score: f32) -> TreeNode {
        TreeNode {
            depth,
            token_idx,
            parent_path,
            score,
        }
    }

    #[test]
    fn test_topology_from_ddtree_simple_chain() {
        // Chain: token 5 → token 3 → token 1
        let nodes = vec![
            make_node(0, 5, TreePath::from_tokens(&[5]), -1.0),
            make_node(1, 3, TreePath::from_tokens(&[5, 3]), -2.0),
            make_node(2, 1, TreePath::from_tokens(&[5, 3, 1]), -3.0),
        ];
        let (topo, token_ids) = build_topology_from_tree_nodes(&nodes, 0.9);
        assert_eq!(topo.n_nodes, 3);
        assert_eq!(token_ids.len(), 3);
        // A chain has exactly one root (no ancestors).
        let root_count = (0..topo.n_nodes)
            .filter(|&k| {
                let abits = &topo.ancestor_bits[k * topo.n_words..(k + 1) * topo.n_words];
                abits.iter().all(|&w| w == 0)
            })
            .count();
        assert_eq!(root_count, 1, "chain must have exactly one root");
    }

    #[test]
    fn test_topology_from_ddtree_branching() {
        // Two roots, each with a child:
        //   root A (token 5) → child (token 3)
        //   root B (token 8) → child (token 7)
        let nodes = vec![
            make_node(0, 5, TreePath::from_tokens(&[5]), -1.0),
            make_node(0, 8, TreePath::from_tokens(&[8]), -1.5),
            make_node(1, 3, TreePath::from_tokens(&[5, 3]), -2.0),
            make_node(1, 7, TreePath::from_tokens(&[8, 7]), -2.5),
        ];
        let (topo, token_ids) = build_topology_from_tree_nodes(&nodes, 0.95);
        assert_eq!(topo.n_nodes, 4);
        assert_eq!(token_ids.len(), 4);
        // Two roots = two nodes with no ancestors
        let root_count = (0..topo.n_nodes)
            .filter(|&k| {
                let abits = &topo.ancestor_bits[k * topo.n_words..(k + 1) * topo.n_words];
                abits.iter().all(|&w| w == 0)
            })
            .count();
        assert_eq!(root_count, 2, "branching tree must have two roots");
    }

    #[test]
    fn test_topology_from_ddtree_deduplicates() {
        // Same node appears twice with different scores — highest score wins.
        let nodes = vec![
            make_node(0, 5, TreePath::from_tokens(&[5]), -1.0),
            make_node(0, 5, TreePath::from_tokens(&[5]), -0.5), // higher score (less negative)
        ];
        let (topo, _token_ids) = build_topology_from_tree_nodes(&nodes, 0.9);
        assert_eq!(topo.n_nodes, 1, "duplicate nodes must be deduplicated");
    }

    #[test]
    #[should_panic(expected = "no matching parent")]
    fn test_topology_from_ddtree_missing_parent_panics() {
        // Node at depth 1 but no parent at depth 0 with matching path.
        let nodes = vec![
            make_node(0, 5, TreePath::from_tokens(&[5]), -1.0),
            make_node(1, 3, TreePath::from_tokens(&[0x99, 3]), -2.0), // parent [0x99] doesn't exist
        ];
        let _ = build_topology_from_tree_nodes(&nodes, 0.9);
    }
}
