//! Exact recursive least-squares associative memory (`rls_depth_mem`).
//!
//! Plan 630 Phase 1 · [arXiv:2610.18809] ("Depth Memory": exact RLS across
//! depth) · distilled via [Research 629](../../../.research/629_Depth_Memory_RLS_Depth_Mixing.md)
//! §E1/E6. The exact-online rung of the associative-memory ladder this crate
//! ships in approximation form (`super::state::DeltaMemoryState`, δ-rule) and
//! batch form (`crate::linalg::ridge_solve` + `hebbian_kernel_memory`).
//!
//! # The axis law (why exact is free here)
//!
//! A sequence must be scanned before its tail is known, so sequence memory is
//! stuck with approximations (the δ-rule's per-step rank-1 forgetting).
//! **Depth, loops, and pipeline stages are naturally sequential** — write j
//! happens strictly after write j−1 and before write j+1 — so the exact
//! estimator can be maintained online at the same O(RANK²) cost class. The
//! headline property, gated by `g1_batch_equivalence_decayed_ridge`: after N
//! observations the online state equals the batch **decayed**-ridge solution
//! over the same window (`crate::linalg::ridge_solve_direct_f64`).
//!
//! # The state (per write, ω = 1 folded out)
//!
//! ```text
//! k̃ = k                                  (caller-L2-normed key)
//! κ  = P·k̃ / (λ + k̃ᵀP·k̃)                 (gain)
//! u  = v − M·k̃                           (pseudo-value / residual)
//! M ← M + u·κᵀ                           (rank-1 write)
//! P ← (P − κκᵀ·(λ + k̃ᵀPk̃)) / λ          (symmetric downdate)
//! ```
//!
//! The P update is **symmetric by construction**: it is algebraically the
//! textbook `(P − κk̃ᵀP)/λ`, but the downdate numerator `Pk̃k̃ᵀP` is formed
//! through `κκᵀ·denom`, whose (i,j) and (j,i) entries are the same f32
//! products in the same order (IEEE multiply commutes bit-exactly), so P
//! stays bit-symmetric — the textbook form loses symmetry in f32 and, with
//! forgetting, that is the classic route to a non-positive-definite P.
//!
//! # Honest limits (both bind)
//!
//! 1. The `(u, κ)` journal ([`RlsCtx`]) supports **unwind** and **windowed
//!    reads** (Phase 2) but NOT re-derivation under a different λ — κ depends
//!    on λ, so re-fitting under a new schedule needs the raw (k, v) rows.
//! 2. P restoration requires the saved triangle ([`RlsCtx::p_prev_tri`]);
//!    [`RlsMemoryState::unwind_step`] is **M-only** and is therefore
//!    **read-rollback-only**: any `write` after an unwind without a
//!    [`RlsMemoryState::p_restore`] uses a stale gain — a caller bug this doc
//!    names (the paper's own backward pass saves P rather than unwinding it;
//!    inverse Sherman–Morrison would divide by λ per step and is numerically
//!    unstable — deliberately not attempted).
//!
//! # Determinism (Issue-940 law)
//!
//! This state is **local-only** (never synced, never replay-compared): all
//! dots are **sequential f32 folds in fixed index order** (never a
//! platform-lane SIMD kernel), the fold order is the formula order above, and
//! the symmetric-P argument makes the result independent of (i,j) visit
//! order. Bit-reproducible per seed, asserted in tests.
//!
//! # Caller contract
//!
//! Keys MUST be L2-normalized before `write`/`read` (the
//! `FeatureHasher` convention this module mirrors); values should be O(1).
//! Unnormalized keys blow up `k̃ᵀPk̃` and destabilize the forward itself —
//! a generator contract, not the state's job to fix.

/// One journal record: everything needed to unwind one exact-RLS write.
///
/// `u`/`kappa` reproduce the write's rank-1 term (`M += u·κᵀ`, so
/// `M -= u·κᵀ` undoes it); `p_prev_tri` is the **pre-write** P's upper
/// triangle (including the diagonal), packed row-major over `i ≤ j` — 36
/// f32 at rank 8, cheaper than the state itself (the paper's own context
/// shape).
///
/// ⛔ `TRI` exists because stable Rust forbids const-generic arithmetic in
/// array lengths (`generic_const_exprs`): `RANK·(RANK+1)/2` cannot be spelled
/// as a type. The defaults `<RANK = 8, TRI = 36>` ARE that product at the
/// default rank; a non-default rank must pass the literal product explicitly
/// (`RlsCtx::<16, 136>`), and a mis-sized ctx fails LOUD at the pack/unpack
/// bounds — never silently.
#[derive(Clone, Copy, Debug)]
pub struct RlsCtx<const RANK: usize = 8, const TRI: usize = 36> {
    /// Pseudo-value residual `u = v − M·k̃` at write time (pre-update M).
    pub u: [f32; RANK],
    /// Gain `κ = P·k̃ / (λ + k̃ᵀP·k̃)` at write time (pre-update P).
    pub kappa: [f32; RANK],
    /// Pre-write P, upper triangle incl. diagonal (packed `i ≤ j`, row-major).
    pub p_prev_tri: [f32; TRI],
}

impl<const RANK: usize, const TRI: usize> Default for RlsCtx<RANK, TRI> {
    fn default() -> Self {
        Self {
            u: [0.0; RANK],
            kappa: [0.0; RANK],
            p_prev_tri: [0.0; TRI],
        }
    }
}

impl<const RANK: usize, const TRI: usize> RlsCtx<RANK, TRI> {
    /// Pack the upper triangle (incl. diagonal) of a symmetric matrix,
    /// row-major over `i ≤ j`. Bounds-panics loud when `TRI` is smaller than
    /// the rank's triangle (`RANK·(RANK+1)/2`) — the mis-sized-ctx failure.
    fn pack_into(p: &[[f32; RANK]; RANK], tri: &mut [f32; TRI]) {
        let mut idx = 0;
        for (i, row) in p.iter().enumerate() {
            for &v in row.iter().skip(i) {
                tri[idx] = v;
                idx += 1;
            }
        }
    }

    /// Unpack the triangle into a full symmetric matrix (lower mirrored).
    /// Reads exactly the rank's triangle; surplus `TRI` words are ignored.
    fn unpack_from(tri: &[f32; TRI]) -> [[f32; RANK]; RANK] {
        let mut p = [[0.0f32; RANK]; RANK];
        let mut idx = 0;
        for i in 0..RANK {
            // Row i's upper part (incl. diagonal) comes straight from the tri.
            for cell in p[i].iter_mut().skip(i) {
                *cell = tri[idx];
                idx += 1;
            }
            // Mirror into column i of the rows above (disjoint via split).
            let (head, tail) = p.split_at_mut(i);
            let row_i = &tail[0];
            for (j, row_j) in head.iter_mut().enumerate() {
                row_j[i] = row_i[j];
            }
        }
        p
    }
}

/// Fixed-rank exact-RLS associative memory: `M` solves the λ-weighted ridge
/// problem over the observation window, online, in O(RANK²) per write.
///
/// `new()` defaults: `alpha = 1.0`, `lambda = 0.99` (fixed schedules per
/// Research 629 §E6 — no learned scalars; the tuned values are not claimed).
#[derive(Clone, Debug)]
pub struct RlsMemoryState<const RANK: usize = 8> {
    /// Associative memory, row-major `m[i][j]` = M[i, j]; `read = M·q`.
    m: [[f32; RANK]; RANK],
    /// Inverse covariance estimate (SPD by induction; bit-symmetric).
    p: [[f32; RANK]; RANK],
    /// Prior scale: `P₀ = α·I`; the ridge term decays as `λ^N/α`.
    alpha: f32,
    /// Forgetting factor in `(0, 1]` (1.0 = plain ridge, no forgetting).
    lambda: f32,
    /// ‖u‖ of the most recent write (T3.5) — the depth-regression residual
    /// as a surprise scalar. Composes with the shipped `delta_mem`
    /// surprise-gate law (`DEFAULT_THETA_SURPRISE`): a consumer can refuse
    /// writes whose prediction residual is already tiny. Not a gate here —
    /// the accessor only reports; gating is a consumer decision.
    last_surprise: f32,
}

impl<const RANK: usize> RlsMemoryState<RANK> {
    /// Default schedule (`alpha = 1.0`, `lambda = 0.99`); `M = 0`, `P = αI`.
    pub fn new() -> Self {
        Self::with_schedule(1.0, 0.99)
    }

    /// Explicit fixed schedule. Panics on `alpha <= 0` or `λ ∉ (0, 1]`.
    pub fn with_schedule(alpha: f32, lambda: f32) -> Self {
        assert!(alpha > 0.0, "alpha must be positive, got {alpha}");
        assert!(
            lambda > 0.0 && lambda <= 1.0,
            "lambda must be in (0, 1], got {lambda}"
        );
        let mut p = [[0.0f32; RANK]; RANK];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = alpha;
        }
        Self {
            m: [[0.0; RANK]; RANK],
            p,
            alpha,
            lambda,
            last_surprise: 0.0,
        }
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// ‖u‖ after the most recent write (T3.5) — 0.0 before any write. The
    /// E5 surprise scalar: how much the write's value disagreed with what
    /// the memory already predicted (`u = v − M·k̃`).
    ///
    /// ⛔ Under forgetting, one write absorbs only the fraction
    /// `α‖k‖²/(λ + α‖k‖²)` of the residual (≈ 0.5 at the defaults) — so a
    /// repeat write's surprise CONTRACTS GEOMETRICALLY by that factor per
    /// repeat; it is not near-zero after one repeat. A θ-threshold consumer
    /// gates on this decay law, not on a single-repeat cliff (pinned by
    /// `e5_surprise_novel_high_repeat_low`).
    pub fn surprise(&self) -> f32 {
        self.last_surprise
    }

    /// Exact-RLS write (T1.2). See the module doc for the formula and the
    /// caller-L2-normed-key contract. Panics on dimension mismatch or on a
    /// non-positive denominator `λ + k̃ᵀP·k̃` (a corrupted-P loud failure —
    /// with the symmetric update the denominator is positive in exact
    /// arithmetic whenever P is SPD).
    pub fn write(&mut self, key: &[f32], value: &[f32]) {
        self.write_inner(key, value, None);
    }

    /// Journal-carrying write: identical arithmetic to [`Self::write`], plus
    /// the record an [`Self::unwind_step`] (and Phase 2's windowed reader)
    /// consumes — `u`, `kappa`, and the **pre-write** P triangle. Overwrites
    /// `ctx` fully; journal order is caller-owned (append-only ring).
    pub fn write_ctx(&mut self, key: &[f32], value: &[f32], ctx: &mut RlsCtx<RANK>) {
        self.write_inner(key, value, Some(ctx));
    }

    fn write_inner(&mut self, key: &[f32], value: &[f32], ctx: Option<&mut RlsCtx<RANK>>) {
        assert_eq!(key.len(), RANK, "key dimension must match RANK");
        assert_eq!(value.len(), RANK, "value dimension must match RANK");

        // k̃ = k — the observation weight ω is the fixed schedule ω = 1 and is
        // folded out entirely (no multiply, per Plan 630 T1.2).
        // Sequential folds in fixed index order (Issue-940 local-only law).
        let mut pk = [0.0f32; RANK]; // P·k̃, one row-dot per output dim
        for (pk_i, p_row) in pk.iter_mut().zip(self.p.iter()) {
            let mut s = 0.0f32;
            for (p_ij, &k_j) in p_row.iter().zip(key.iter()) {
                s += p_ij * k_j;
            }
            *pk_i = s;
        }
        let mut kpk = 0.0f32; // k̃ᵀ·P·k̃
        for (&k_i, &pk_i) in key.iter().zip(pk.iter()) {
            kpk += k_i * pk_i;
        }
        let denom = self.lambda + kpk;
        assert!(
            denom > 0.0,
            "RLS denominator λ + k̃ᵀPk̃ = {denom} ≤ 0 — P lost positive definiteness"
        );

        let mut kappa = [0.0f32; RANK];
        for (kappa_i, &pk_i) in kappa.iter_mut().zip(pk.iter()) {
            *kappa_i = pk_i / denom;
        }

        let mut u = [0.0f32; RANK]; // u = v − M·k̃ (pre-update M)
        let mut u_norm_sq = 0.0f32;
        for ((u_i, &v_i), m_row) in u.iter_mut().zip(value.iter()).zip(self.m.iter()) {
            let mut s = 0.0f32;
            for (m_ij, &k_j) in m_row.iter().zip(key.iter()) {
                s += m_ij * k_j;
            }
            *u_i = v_i - s;
            u_norm_sq += *u_i * *u_i;
        }
        self.last_surprise = u_norm_sq.sqrt();

        // Journal capture BEFORE any mutation: pre-write P triangle + the
        // record quantities (T1.5 contract).
        if let Some(ctx) = ctx {
            ctx.u = u;
            ctx.kappa = kappa;
            RlsCtx::pack_into(&self.p, &mut ctx.p_prev_tri);
        }

        // Rank-1 memory write: M += u·κᵀ.
        for (m_row, &u_i) in self.m.iter_mut().zip(u.iter()) {
            for (m_ij, &kappa_j) in m_row.iter_mut().zip(kappa.iter()) {
                *m_ij += u_i * kappa_j;
            }
        }

        // Symmetric-by-construction downdate:
        //   P ← (P − κκᵀ·denom)/λ  ≡  (P − κk̃ᵀP)/λ
        // (i,j) and (j,i) evaluate identical products in identical order, so
        // bit-symmetry is preserved without mirroring (module doc).
        let inv_lambda = 1.0 / self.lambda;
        for (p_row, &kappa_i) in self.p.iter_mut().zip(kappa.iter()) {
            for (p_ij, &kappa_j) in p_row.iter_mut().zip(kappa.iter()) {
                *p_ij = (*p_ij - kappa_i * kappa_j * denom) * inv_lambda;
            }
        }
    }

    /// Read into a pre-allocated buffer: `out = M·q` (mirrors
    /// `DeltaMemoryState::read_into`'s row shape; sequential folds).
    pub fn read_into(&self, query: &[f32], out: &mut [f32]) {
        assert_eq!(query.len(), RANK, "query dimension must match RANK");
        assert_eq!(out.len(), RANK, "output dimension must match RANK");
        for (out_i, m_row) in out.iter_mut().zip(self.m.iter()) {
            let mut s = 0.0f32;
            for (m_ij, &q_j) in m_row.iter().zip(query.iter()) {
                s += m_ij * q_j;
            }
            *out_i = s;
        }
    }

    /// Read: `M·q` (fixed-size array form).
    pub fn read(&self, query: &[f32]) -> [f32; RANK] {
        let mut out = [0.0f32; RANK];
        self.read_into(query, &mut out);
        out
    }

    /// M-only unwind (T1.5): `M ← M − u·κᵀ` in place from the write's record.
    ///
    /// **Read-rollback-only.** This restores the memory surface but NOT the
    /// gain surface: after an unwind, `self.p` is the POST-write P of the
    /// undone write. Resuming writes on that stale gain is a caller bug —
    /// call [`Self::p_restore`] with the SAME record first (the paper's own
    /// backward pass saves P; restoring it by inverse Sherman–Morrison would
    /// divide by λ per step and is numerically unstable — not attempted).
    pub fn unwind_step(&mut self, ctx: &RlsCtx<RANK>) {
        for (m_row, &u_i) in self.m.iter_mut().zip(ctx.u.iter()) {
            for (m_ij, &kappa_j) in m_row.iter_mut().zip(ctx.kappa.iter()) {
                *m_ij -= u_i * kappa_j;
            }
        }
    }

    /// Re-install P exactly from a record's pre-write triangle (bit-copy of
    /// the saved symmetric matrix; the lower triangle is the mirror). Pair
    /// with [`Self::unwind_step`] to resume writing after a rollback.
    pub fn p_restore(&mut self, ctx: &RlsCtx<RANK>) {
        self.p = RlsCtx::unpack_from(&ctx.p_prev_tri);
    }
}

impl<const RANK: usize> Default for RlsMemoryState<RANK> {
    fn default() -> Self {
        Self::new()
    }
}

/// Bounded correction window over a frozen base state (Phase 2, T2.1):
/// reads `M_j·x = M_b·x + Σᵢ uᵢ(κᵢᵀx)` for the ≤ E journaled writes — the
/// windowed-traffic form of the last E states WITHOUT materializing E
/// intermediate `RlsMemoryState`s (the paper's read-side amortization).
///
/// Corrections are summed in **write order** (append order — pinned; the
/// Phase-2 G1 test replays that order against the materialized state).
/// Construction allocates once (`with_capacity(max)`); `read_into` itself is
/// zero-alloc. `RlsMemoryState` stays the only writer — the window is a
/// READ-ONLY view; never `push` a record that the base has already absorbed
/// (the G1 contract is base + window == the state AFTER exactly those
/// writes), and never mix records from two states (stale-gain class).
pub struct WindowedReader<'a, const RANK: usize = 8, const TRI: usize = 36> {
    /// The frozen base the corrections apply on top of.
    base: &'a RlsMemoryState<RANK>,
    /// Journaled corrections, write order (append order) preserved.
    corrections: Vec<RlsCtx<RANK, TRI>>,
    /// The ≤ E bound (capacity was pre-reserved; `push` refuses past it).
    max: usize,
}

impl<'a, const RANK: usize, const TRI: usize> WindowedReader<'a, RANK, TRI> {
    /// Window over `base` holding at most `max` corrections (pre-reserved).
    pub fn new(base: &'a RlsMemoryState<RANK>, max: usize) -> Self {
        Self {
            base,
            corrections: Vec::with_capacity(max),
            max,
        }
    }

    /// Append one journaled correction (write order). Panics loud past the
    /// ≤ E bound — roll the window instead (drop the oldest, re-base, or
    /// narrow the query set; the caller owns the policy).
    pub fn push(&mut self, ctx: RlsCtx<RANK, TRI>) {
        assert!(
            self.corrections.len() < self.max,
            "WindowedReader bound E={} exceeded — roll the window before pushing more",
            self.max
        );
        self.corrections.push(ctx);
    }

    /// Drop all corrections (back to the bare base read).
    pub fn clear(&mut self) {
        self.corrections.clear();
    }

    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// `out = M_j·x` where `M_j = M_b + Σᵢ uᵢκᵢᵀ` over the window, summed
    /// in write order. Zero-alloc; sequential folds (Issue-940 local-only).
    pub fn read_into(&self, x: &[f32], out: &mut [f32]) {
        self.base.read_into(x, out);
        for ctx in &self.corrections {
            let mut s = 0.0f32; // κᵢᵀx
            for (&k_i, &x_i) in ctx.kappa.iter().zip(x.iter()) {
                s += k_i * x_i;
            }
            for (out_j, &u_j) in out.iter_mut().zip(ctx.u.iter()) {
                *out_j += u_j * s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta_mem::{DeltaMemoryConfig, DeltaMemoryState};
    use crate::linalg::{ridge_solve_direct_f64, try_cholesky_f32};

    const R: usize = 8;

    fn l2_normalize(v: &mut [f32]) {
        let mut s = 0.0f32;
        for &x in v.iter() {
            s += x * x;
        }
        let n = s.sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    /// Deterministic seeded Gaussian (Box–Muller over fastrand uniforms).
    struct Gauss {
        rng: fastrand::Rng,
        spare: Option<f32>,
    }

    impl Gauss {
        fn new(seed: u64) -> Self {
            Self {
                rng: fastrand::Rng::with_seed(seed),
                spare: None,
            }
        }
        fn next(&mut self) -> f32 {
            if let Some(s) = self.spare.take() {
                return s;
            }
            let u1 = self.rng.f32().max(1e-7);
            let u2 = self.rng.f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            self.spare = Some(r * theta.sin());
            r * theta.cos()
        }
    }

    /// Seeded unit-norm key stream + O(1) value stream (the caller contract:
    /// the GENERATOR honors the L2 law the envelope lesson pinned —
    /// unnormalized keys with k̃ᵀPk̃ >> 1 destabilize the forward itself).
    fn gen_stream(seed: u64, n: usize) -> (Vec<[f32; R]>, Vec<[f32; R]>) {
        let mut g = Gauss::new(seed);
        let mut keys = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        for _ in 0..n {
            let mut k = [0.0f32; R];
            for x in k.iter_mut() {
                *x = g.next();
            }
            l2_normalize(&mut k);
            let mut v = [0.0f32; R];
            for x in v.iter_mut() {
                *x = 0.5 * g.next();
            }
            keys.push(k);
            vals.push(v);
        }
        (keys, vals)
    }

    fn run_online(
        alpha: f32,
        lambda: f32,
        keys: &[[f32; R]],
        vals: &[[f32; R]],
    ) -> RlsMemoryState<R> {
        let mut st = RlsMemoryState::<R>::with_schedule(alpha, lambda);
        for i in 0..keys.len() {
            st.write(&keys[i], &vals[i]);
        }
        st
    }

    /// f64 batch oracle for the DECAYED-ridge objective RLS actually solves:
    ///   min_M  Σᵢ λ^{N−i}‖vᵢ − M kᵢ‖² + (λ^N/α)·‖M‖²
    /// Builds the weighted Gram `Σ λ^{N−i}·kᵢkᵢᵀ + (λ^N/α)·I` and the
    /// weighted cross-cov `Σ λ^{N−i}·kᵢvᵢᵀ` (the `XᵀY` shape
    /// `ridge_solve_direct_f64` consumes — the test owns the weighting),
    /// returns M (transposed out of the function's Wᵀ layout) as f32.
    fn batch_oracle(
        alpha: f32,
        lambda: f64,
        keys: &[[f32; R]],
        vals: &[[f32; R]],
    ) -> [[f32; R]; R] {
        let n = keys.len() as i32;
        let reg = lambda.powi(n) / (alpha as f64);
        let mut gram_flat = [0.0f64; R * R];
        let mut cov_flat = [0.0f64; R * R]; // XᵀY shape: d_k × d_v
        for (i, (k, v)) in keys.iter().zip(vals.iter()).enumerate() {
            // most recent write (i = N−1) has exponent 0 → weight 1
            let w = lambda.powi(n - 1 - i as i32);
            for a in 0..R {
                for b in 0..R {
                    gram_flat[a * R + b] += w * (k[a] as f64) * (k[b] as f64);
                    cov_flat[a * R + b] += w * (k[a] as f64) * (v[b] as f64);
                }
            }
        }
        for d in 0..R {
            gram_flat[d * R + d] += reg;
        }
        let mut w_t = [0.0f64; R * R];
        let mut l = [0.0f64; R * R];
        let mut z = [0.0f64; R * R];
        ridge_solve_direct_f64(&mut w_t, &mut l, &mut z, &gram_flat, &cov_flat, R, R);
        // w_t[a*R+b] = Mᵀ[a][b] = M[b][a]
        let mut m = [[0.0f32; R]; R];
        for b in 0..R {
            for a in 0..R {
                m[b][a] = w_t[a * R + b] as f32;
            }
        }
        m
    }

    fn max_abs_diff(a: &[[f32; R]; R], b: &[[f32; R]; R]) -> f32 {
        let mut m = 0.0f32;
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                m = m.max((x - y).abs());
            }
        }
        m
    }

    fn assert_spd(p: &[[f32; R]; R]) {
        let mut flat = [0.0f32; R * R];
        for (i, row) in p.iter().enumerate() {
            flat[i * R..i * R + R].copy_from_slice(row);
        }
        let mut l = [0.0f32; R * R];
        try_cholesky_f32(&mut l, &flat, R)
            .unwrap_or_else(|e| panic!("P not SPD after a step: pivot {} tol {}", e.pivot, e.tol));
    }

    #[test]
    fn new_state_is_m_zero_p_alpha_identity() {
        let st = RlsMemoryState::<R>::new();
        assert_eq!(st.alpha(), 1.0);
        assert_eq!(st.lambda(), 0.99);
        for row in st.m.iter() {
            assert!(row.iter().all(|&x| x == 0.0));
        }
        for i in 0..R {
            for j in 0..R {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert_eq!(st.p[i][j], expect);
            }
        }
        let q = [0.5f32; R];
        assert!(st.read(&q).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn single_write_matches_closed_form() {
        let (keys, vals) = gen_stream(7, 1);
        let k = keys[0];
        let mut st = RlsMemoryState::<R>::new();
        st.write(&k, &vals[0]);
        // Expected from the recursion with M₀ = 0, P₀ = αI:
        //   κ = αk/(λ + α), u = v, M = v·κᵀ
        let alpha = 1.0f32;
        let lambda = 0.99f32;
        let mut expected = [[0.0f32; R]; R];
        for i in 0..R {
            for j in 0..R {
                let kappa_j = alpha * k[j] / (lambda + alpha);
                expected[i][j] = vals[0][i] * kappa_j;
            }
        }
        let dev = max_abs_diff(&st.m, &expected);
        assert!(dev < 1e-6, "single-write closed form dev {dev}");
        assert_spd(&st.p);
    }

    /// G1 — the plan's headline gate: after N online writes the state equals
    /// the batch DECAYED-ridge solution (`ridge_solve_direct_f64` over the
    /// same λ-weighted window, ridge term λ^N/α — NOT a fixed 1/α), across
    /// four (α, λ) schedules. P stays SPD (Cholesky) after EVERY step on a
    /// full N ≥ 1e3 sweep; the `λ + k̃ᵀPk̃ > 0` denominator invariant is
    /// asserted inside every write.
    #[test]
    fn g1_batch_equivalence_decayed_ridge() {
        let regimes: [(f32, f64, usize, u64); 4] = [
            (1.0, 0.99, 1000, 101),
            (1.0, 0.90, 400, 202),
            (1.0, 0.999, 1000, 303),
            (0.5, 1.0, 500, 404),
        ];
        let mut worst = 0.0f32;
        for &(alpha, lambda, n, seed) in regimes.iter() {
            let (keys, vals) = gen_stream(seed, n);
            // SPD after every step over the full sweep.
            {
                let mut st = RlsMemoryState::<R>::with_schedule(alpha, lambda as f32);
                for i in 0..keys.len() {
                    st.write(&keys[i], &vals[i]);
                    assert_spd(&st.p);
                }
            }
            let st = run_online(alpha, lambda as f32, &keys, &vals);
            let m_batch = batch_oracle(alpha, lambda, &keys, &vals);
            let dev = max_abs_diff(&st.m, &m_batch);
            eprintln!("g1 regime α={alpha} λ={lambda} N={n}: max|M_online − M_batch| = {dev:.3e}");
            worst = worst.max(dev);
        }
        // Pinned f32 tolerance (Plan 630 T1.4): the measured worst across
        // regimes is printed above; the 1e-3 pin carries ≥100× margin over
        // the expected ~1e-5 f32 accumulation noise (≤1e3 rank-1 updates,
        // contracted by λ < 1).
        assert!(
            worst < 1e-3,
            "G1 batch equivalence FAILED: max dev {worst:.3e} ≥ 1e-3"
        );
    }

    /// G1 round-trip (T1.5): W journal writes → W unwinds (reverse order) →
    /// `p_restore` from the FIRST record → M within a pinned ε of pre-write
    /// and P BIT-EXACT. Also pins the stale-P hazard: M-only unwind leaves P
    /// different from the pre-write P (restore is mandatory before re-write).
    #[test]
    fn unwind_restore_roundtrip() {
        const W: usize = 64;
        let (keys, vals) = gen_stream(999, W);
        let mut st = RlsMemoryState::<R>::new();
        let m0 = st.m;
        let p0 = st.p;
        let mut journal: Vec<RlsCtx<R>> = Vec::with_capacity(W);
        for i in 0..W {
            let mut ctx = RlsCtx::<R>::default();
            st.write_ctx(&keys[i], &vals[i], &mut ctx);
            journal.push(ctx);
        }
        // Unwind in reverse write order (LIFO — the journal's natural order,
        // pinned here and in the doc).
        for ctx in journal.iter().rev() {
            st.unwind_step(ctx);
        }
        let m_only_dev = max_abs_diff(&st.m, &m0);
        eprintln!("unwind roundtrip: max|M_unwound − M₀| = {m_only_dev:.3e}");
        // P after M-only unwind is the LAST write's post-P — NOT P₀.
        assert_ne!(st.p, p0, "M-only unwind must NOT be sold as restoring P");
        // Restore P from the FIRST write's record (P_prev of write 1 = P₀).
        st.p_restore(&journal[0]);
        assert_eq!(st.p, p0, "p_restore must re-install P bit-exactly");
        // M tolerance: identical operands added then subtracted per rank-1
        // term; f32 rounding over W=64 terms stays ~1e-6. Pin 1e-4.
        assert!(
            m_only_dev < 1e-4,
            "unwind roundtrip FAILED: max dev {m_only_dev:.3e} ≥ 1e-4"
        );
        assert_spd(&st.p);
    }

    #[test]
    fn determinism_same_seed_bit_identical() {
        let (keys, vals) = gen_stream(42, 256);
        let a = run_online(1.0, 0.99, &keys, &vals);
        let b = run_online(1.0, 0.99, &keys, &vals);
        assert_eq!(a.m, b.m, "same seed must produce bit-identical M");
        assert_eq!(a.p, b.p, "same seed must produce bit-identical P");
    }

    #[test]
    fn written_key_reads_closer_than_unrelated_key() {
        let mut st = RlsMemoryState::<8>::new();
        let (keys, vals) = gen_stream(5, 32);
        for i in 0..keys.len() {
            st.write(&keys[i], &vals[i]);
        }
        let out = st.read(&keys[0]);
        let other = st.read(&keys[31]);
        let d_self: f32 = out
            .iter()
            .zip(vals[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let d_other: f32 = other
            .iter()
            .zip(vals[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            d_self <= d_other,
            "read(key₀) must beat read(key_last) for v₀"
        );
    }

    #[test]
    #[should_panic(expected = "key dimension must match RANK")]
    fn write_rejects_wrong_dim() {
        let mut st = RlsMemoryState::<8>::new();
        let k = [0.1f32; 4];
        let v = [0.1f32; 8];
        st.write(&k, &v);
    }

    /// Phase-2 G1 (T2.2): windowed read == materialized-intermediate read.
    /// Base = the state after B writes (frozen clone); W more journaled
    /// writes on the live state; the windowed reader over (base, journal)
    /// must equal the LIVE state's read (its M absorbed exactly those W
    /// rank-1 terms) across randomized queries. Summation orders differ
    /// (base-first vs absorbed-in-place) — f32 drift is the tolerance.
    #[test]
    fn windowed_read_equals_materialized() {
        const B: usize = 200;
        const W: usize = 64;
        const Q: usize = 32;
        let (keys, vals) = gen_stream(4242, B + W);
        let mut st = RlsMemoryState::<R>::new();
        for i in 0..B {
            st.write(&keys[i], &vals[i]);
        }
        let base = st.clone();
        let mut journal: Vec<RlsCtx<R>> = Vec::with_capacity(W);
        for i in B..B + W {
            let mut ctx = RlsCtx::<R>::default();
            st.write_ctx(&keys[i], &vals[i], &mut ctx);
            journal.push(ctx);
        }
        let mut reader = WindowedReader::<R>::new(&base, W);
        for ctx in journal.drain(..) {
            reader.push(ctx);
        }
        assert_eq!(reader.len(), W);
        // Randomized unit queries (separate generator — independent of the
        // stream generator's state).
        let mut worst = 0.0f32;
        let mut g = Gauss::new(777);
        for _ in 0..Q {
            let mut q = [0.0f32; R];
            for x in q.iter_mut() {
                *x = g.next();
            }
            l2_normalize(&mut q);
            let mut out_w = [0.0f32; R];
            let mut out_m = [0.0f32; R];
            reader.read_into(&q, &mut out_w);
            st.read_into(&q, &mut out_m);
            for (a, b) in out_w.iter().zip(out_m.iter()) {
                worst = worst.max((a - b).abs());
            }
        }
        eprintln!("phase2 g1: max|windowed − materialized| over {Q} queries, W={W} = {worst:.3e}");
        // Different association order over W=64 O(1) terms: ~1e-6 expected.
        assert!(
            worst < 1e-4,
            "windowed != materialized: max dev {worst:.3e} ≥ 1e-4"
        );
    }

    #[test]
    #[should_panic(expected = "bound E=")]
    fn window_reader_refuses_past_bound() {
        let (keys, vals) = gen_stream(31337, 4);
        let mut st = RlsMemoryState::<R>::new();
        // Journal all four writes FIRST (no reader alive yet), then the
        // reader's third push must fire the bound.
        let mut journal: Vec<RlsCtx<R>> = Vec::with_capacity(4);
        for i in 0..4 {
            let mut ctx = RlsCtx::<R>::default();
            st.write_ctx(&keys[i], &vals[i], &mut ctx);
            journal.push(ctx);
        }
        let mut reader = WindowedReader::<R>::new(&st, 2);
        for ctx in journal {
            reader.push(ctx);
        }
    }

    #[test]
    fn window_reader_empty_equals_base() {
        let (keys, vals) = gen_stream(2468, 8);
        let mut st = RlsMemoryState::<R>::new();
        for i in 0..8 {
            st.write(&keys[i], &vals[i]);
        }
        let reader = WindowedReader::<R>::new(&st, 4);
        assert!(reader.is_empty());
        let mut a = [0.0f32; R];
        let mut b = [0.0f32; R];
        reader.read_into(&keys[0], &mut a);
        st.read_into(&keys[0], &mut b);
        assert_eq!(a, b, "empty window must be bit-identical to the base read");
    }

    /// T3.4 (Research 629 E4): the paper's ordering law re-tested on our
    /// fixture style — at fixed inputs, recall error RLS < delta < outer.
    /// Fixture: a random linear map W_true + NOISY REPEATS of earlier keys
    /// (repeats are where exact least-squares averaging beats first-order
    /// overwrite/accumulation). Reconstruction error = mean ‖M·k − v‖² over
    /// the trained associations. Honest-negative clause: an ordering
    /// violation here is a FINDING (recorded in the plan), not a silent pin.
    #[test]
    fn e4_ordering_rls_beats_delta_beats_outer() {
        const N_ASSOC: usize = 24;
        const N_REPEATS: usize = 16; // revisit earlier keys with fresh noise
        let mut g = Gauss::new(606);
        // W_true: an O(1) random 8×8 map the arms must recover.
        let mut w_true = [[0.0f32; R]; R];
        for row in w_true.iter_mut() {
            for x in row.iter_mut() {
                *x = g.next();
            }
        }
        let apply = |k: &[f32; R]| {
            let mut v = [0.0f32; R];
            for (v_i, w_row) in v.iter_mut().zip(w_true.iter()) {
                let mut s = 0.0f32;
                for (&w_ij, &k_j) in w_row.iter().zip(k.iter()) {
                    s += w_ij * k_j;
                }
                *v_i = s;
            }
            v
        };
        // Associations: keys drawn once, values = W_true·k + N(0, 0.05²)
        // noise. Repeats revisit EARLIER keys with INDEPENDENT noise — the
        // exact solver averages them; delta's β-forgetting decays the first
        // observation; outer accumulates both raw.
        let mut base_keys: Vec<[f32; R]> = Vec::with_capacity(N_ASSOC);
        for _ in 0..N_ASSOC {
            let mut k = [0.0f32; R];
            for x in k.iter_mut() {
                *x = g.next();
            }
            l2_normalize(&mut k);
            base_keys.push(k);
        }
        let noisy = |k: &[f32; R], g: &mut Gauss| {
            let mut v = apply(k);
            for x in v.iter_mut() {
                *x += 0.05 * g.next();
            }
            v
        };
        let mut stream: Vec<([f32; R], [f32; R])> = Vec::new();
        for k in base_keys.iter() {
            stream.push((*k, noisy(k, &mut g)));
        }
        for r in 0..N_REPEATS {
            let k = base_keys[r * 3 % N_ASSOC];
            stream.push((k, noisy(&k, &mut g)));
        }

        // Arm 1 — outer: M = Σ v·kᵀ (plain accumulation, no solver).
        let mut outer = [[0.0f32; R]; R];
        for (k, v) in stream.iter() {
            for (o_row, &v_i) in outer.iter_mut().zip(v.iter()) {
                for (o_ij, &k_j) in o_row.iter_mut().zip(k.iter()) {
                    *o_ij += v_i * k_j;
                }
            }
        }
        // Arm 2 — delta (shipped): default schedule.
        let mut delta = DeltaMemoryState::new(DeltaMemoryConfig {
            rank: R,
            ..DeltaMemoryConfig::default()
        });
        for (k, v) in stream.iter() {
            delta.write(k, v);
        }
        // Arm 3 — rls: default schedule.
        let mut rls = RlsMemoryState::<R>::new();
        for (k, v) in stream.iter() {
            rls.write(k, v);
        }

        // Content queries: reconstruction over the trained keys (mean ‖·‖²).
        // All three arms flattened to row-major [f32] (the delta snapshot's
        // public shape).
        let mut outer_flat = Vec::with_capacity(R * R);
        for row in outer.iter() {
            outer_flat.extend_from_slice(row);
        }
        let delta_flat = delta.snapshot().state;
        let mut rls_flat = Vec::with_capacity(R * R);
        for row in rls.m.iter() {
            rls_flat.extend_from_slice(row);
        }
        let names = ["outer", "delta", "rls"];
        let mats: [&[f32]; 3] = [&outer_flat, &delta_flat, &rls_flat];
        let mut err = [0.0f64; 3];
        for k in base_keys.iter() {
            let v_clean = apply(k);
            for (which, m) in mats.iter().enumerate() {
                let mut pred = [0.0f32; R];
                for (i, p_i) in pred.iter_mut().enumerate() {
                    let mut s = 0.0f32;
                    for (&m_ij, &k_j) in m[i * R..i * R + R].iter().zip(k.iter()) {
                        s += m_ij * k_j;
                    }
                    *p_i = s;
                }
                for (a, b) in pred.iter().zip(v_clean.iter()) {
                    err[which] += ((a - b) * (a - b)) as f64;
                }
            }
        }
        for e in err.iter_mut() {
            *e /= N_ASSOC as f64;
        }
        eprintln!(
            "e4 ordering: {} {:.4e} < {} {:.4e} < {} {:.4e} (mean sq recon error)",
            names[2], err[2], names[1], err[1], names[0], err[0]
        );
        assert!(
            err[2] < err[1] && err[1] < err[0],
            "E4 ordering VIOLATED (honest-negative finding): rls {:.4e} delta {:.4e} outer {:.4e}",
            err[2],
            err[1],
            err[0]
        );
    }

    /// T3.5 (Research 629 E5): the surprise scalar. Two laws pinned:
    /// (a) FRESH state — the per-repeat contraction is the closed form
    ///     λ/(λ+α) ≈ 0.4975 at the defaults (each write absorbs
    ///     α/(λ+α‖k‖²) of the residual; surprise decays geometrically);
    /// (b) TRAINED state — the gain on a NEW direction shrinks with the
    ///     conditioned covariance (κᵀk = kᵀPk/(λ+kᵀPk) ≪ 0.5), so surprise
    ///     persists longer — monotone decay is the pin there. Persistence
    ///     on a confident memory is itself signal (inconsistent data), not
    ///     a defect.
    #[test]
    fn e5_surprise_novel_high_repeat_low() {
        // (a) VIRGIN state, single pair: the pre-write P₀ = I makes
        // kᵀPk = α‖k‖² = α exactly, so the per-repeat contraction is the
        // closed form λ/(λ+α) (each write absorbs α/(λ+kᵀPk) of the
        // residual; surprise decays geometrically). Any prior history
        // shrinks kᵀPk below α and the ratio toward 1 — history-dependence
        // is leg (b), not a defect.
        let mut k0 = [0.0f32; R];
        let mut g0 = Gauss::new(1549);
        for x in k0.iter_mut() {
            *x = g0.next();
        }
        l2_normalize(&mut k0);
        let mut v0 = [0.0f32; R];
        for x in v0.iter_mut() {
            *x = 0.5 * g0.next();
        }
        let mut st = RlsMemoryState::<R>::new();
        st.write(&k0, &v0);
        let s1 = st.surprise();
        st.write(&k0, &v0);
        let s2 = st.surprise();
        let ratio = s2 / s1;
        let expect = 0.99f32 / (0.99f32 + 1.0f32); // λ/(λ+α)
        eprintln!("e5 virgin-state repeat ratio {ratio:.4} (closed form λ/(λ+α) = {expect:.4})");
        assert!(
            (ratio - expect).abs() < 1e-3,
            "virgin-state surprise contraction must match λ/(λ+α): {ratio} vs {expect}"
        );

        // (b) trained state: novel high, monotone decay per repeat, no
        // one-step cliff (the conditioned-covariance gain law).
        let (keys, vals) = gen_stream(1552, 64);
        let mut st = RlsMemoryState::<R>::new();
        for i in 0..64 {
            st.write(&keys[i], &vals[i]);
        }
        let mut novel = [0.0f32; R];
        let mut g = Gauss::new(1551);
        for x in novel.iter_mut() {
            *x = g.next();
        }
        l2_normalize(&mut novel);
        let mut v = [0.0f32; R];
        for x in v.iter_mut() {
            *x = 0.5 * g.next();
        }
        st.write(&novel, &v);
        let s_novel = st.surprise();
        assert!(s_novel > 0.1, "novel write must surprise, got {s_novel}");
        let mut s_prev = s_novel;
        let mut s_final = s_novel;
        for _ in 0..8 {
            st.write(&novel, &v);
            s_final = st.surprise();
            assert!(
                s_final < s_prev,
                "surprise must decay monotonically per repeat: {s_final} vs {s_prev}"
            );
            s_prev = s_final;
        }
        eprintln!(
            "e5 trained-state surprise: novel {s_novel:.3e} → after 8 repeats {s_final:.3e} (conditioned-gain persistence)"
        );
        // New-state invariant.
        let fresh = RlsMemoryState::<R>::new();
        assert_eq!(fresh.surprise(), 0.0);
    }
}
